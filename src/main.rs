use image::{GenericImageView, ImageBuffer, Rgba, RgbaImage};
use rand::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, WeightedIndex};
use std::cmp::Ordering;
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::env;
use std::error::Error;
use std::fmt;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantAlgorithm {
    KMeansPerceptual = 0,
    MedianCut = 1,
    Wu = 2,
}

impl From<u32> for QuantAlgorithm {
    fn from(v: u32) -> Self {
        match v {
            1 => QuantAlgorithm::MedianCut,
            2 => QuantAlgorithm::Wu,
            _ => QuantAlgorithm::KMeansPerceptual,
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct Config {
    pub k_colors: usize,
    k_seed: u64,
    /// Input image path only used for CLI use
    #[allow(dead_code)]
    input_path: String,
    /// Output image path only used for CLI use
    #[allow(dead_code)]
    output_path: String,
    max_kmeans_iterations: usize,
    peak_threshold_multiplier: f64,
    peak_distance_filter: usize,
    walker_search_window_ratio: f64,
    walker_min_search_window: f64,
    walker_strength_threshold: f64,
    min_cuts_per_axis: usize,
    fallback_target_segments: usize,
    max_step_ratio: f64,
    no_resize: bool,
    algorithm: QuantAlgorithm,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            k_colors: 16,
            k_seed: 42,
            input_path: "samples/2/skeleton.png".to_string(),
            output_path: "samples/2/skeleton_fixed_clean2.png".to_string(),
            max_kmeans_iterations: 15,
            peak_threshold_multiplier: 0.2,
            peak_distance_filter: 4,
            walker_search_window_ratio: 0.35,
            walker_min_search_window: 2.0,
            walker_strength_threshold: 0.5,
            min_cuts_per_axis: 4,
            fallback_target_segments: 64,
            max_step_ratio: 1.8, // Lowered from 3.0 to catch more skew cases
            no_resize: false,
            algorithm: QuantAlgorithm::KMeansPerceptual,
        }
    }
}

// --- OKLab perceptual color space conversions ---

fn srgb_to_linear(c: f32) -> f32 {
    let c = c / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    let v = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (v * 255.0).round().clamp(0.0, 255.0)
}

fn srgb_to_oklab(r: u8, g: u8, b: u8) -> [f32; 3] {
    let lr = srgb_to_linear(r as f32);
    let lg = srgb_to_linear(g as f32);
    let lb = srgb_to_linear(b as f32);

    let l_ = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb;
    let m_ = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb;
    let s_ = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb;

    let l_ = l_.cbrt();
    let m_ = m_.cbrt();
    let s_ = s_.cbrt();

    [
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ]
}

fn oklab_to_srgb(lab: &[f32; 3]) -> [u8; 3] {
    let l_ = lab[0] + 0.3963377774 * lab[1] + 0.2158037573 * lab[2];
    let m_ = lab[0] - 0.1055613458 * lab[1] - 0.0638541728 * lab[2];
    let s_ = lab[0] - 0.0894841775 * lab[1] - 1.2914855480 * lab[2];

    let l_ = l_ * l_ * l_;
    let m_ = m_ * m_ * m_;
    let s_ = s_ * s_ * s_;

    let r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_;
    let g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_;
    let b = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_;

    [
        linear_to_srgb(r) as u8,
        linear_to_srgb(g) as u8,
        linear_to_srgb(b) as u8,
    ]
}

#[derive(Debug)]
pub enum PixelSnapperError {
    ImageError(image::ImageError),
    InvalidInput(String),
    ProcessingError(String),
}

impl fmt::Display for PixelSnapperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PixelSnapperError::ImageError(e) => write!(f, "Image error: {}", e),
            PixelSnapperError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            PixelSnapperError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
        }
    }
}

impl Error for PixelSnapperError {}

impl From<image::ImageError> for PixelSnapperError {
    fn from(error: image::ImageError) -> Self {
        PixelSnapperError::ImageError(error)
    }
}

#[cfg(target_arch = "wasm32")]
impl From<PixelSnapperError> for wasm_bindgen::JsValue {
    fn from(err: PixelSnapperError) -> wasm_bindgen::JsValue {
        wasm_bindgen::JsValue::from_str(&err.to_string())
    }
}

type Result<T> = std::result::Result<T, PixelSnapperError>;

/// CLI entry point
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn main() -> Result<()> {
    let config = parse_args().unwrap_or_default();
    process_image(&config)
}

fn process_image_bytes_common(input_bytes: &[u8], config: Option<Config>) -> Result<Vec<u8>> {
    let config = config.unwrap_or_default();

    let img = image::load_from_memory(input_bytes)?;
    let (width, height) = img.dimensions();

    validate_image_dimensions(width, height)?;

    let rgba_img = img.to_rgba8();

    let output_img = if config.no_resize {
        // No-resize mode: only palette reduction, preserve original dimensions
        match config.algorithm {
            QuantAlgorithm::KMeansPerceptual => quantize_image_oklab(&rgba_img, &config)?,
            QuantAlgorithm::MedianCut => median_cut_quantize(&rgba_img, config.k_colors)?,
            QuantAlgorithm::Wu => wu_quantize(&rgba_img, config.k_colors)?,
        }
    } else {
        // Existing pipeline: RGB k-means quantize + grid detection + resampling
        let quantized_img = quantize_image(&rgba_img, &config)?;
        let (profile_x, profile_y) = compute_profiles(&quantized_img)?;

        let step_x_opt = estimate_step_size(&profile_x, &config);
        let step_y_opt = estimate_step_size(&profile_y, &config);

        let (step_x, step_y) =
            resolve_step_sizes(step_x_opt, step_y_opt, width, height, &config);

        let raw_col_cuts = walk(&profile_x, step_x, width as usize, &config)?;
        let raw_row_cuts = walk(&profile_y, step_y, height as usize, &config)?;

        let (col_cuts, row_cuts) = stabilize_both_axes(
            &profile_x,
            &profile_y,
            raw_col_cuts,
            raw_row_cuts,
            width as usize,
            height as usize,
            &config,
        );

        resample(&quantized_img, &col_cuts, &row_cuts)?
    };

    // Returns bytes for both implementations
    let mut output_bytes = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut output_bytes);
    output_img
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|e| PixelSnapperError::ImageError(e))?;

    Ok(output_bytes)
}

/// WASM entry point
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn process_image(
    input_bytes: &[u8],
    k_colors: Option<u32>,
) -> std::result::Result<Vec<u8>, wasm_bindgen::JsValue> {
    let mut config = Config::default();
    if let Some(k) = k_colors {
        if k == 0 {
            return Err(wasm_bindgen::JsValue::from_str(
                "k_colors must be greater than 0",
            ));
        }
        config.k_colors = k as usize;
    }

    process_image_bytes_common(input_bytes, Some(config))
        .map_err(|e| wasm_bindgen::JsValue::from(e))
}

/// WASM entry point with advanced options (no_resize + algorithm selection)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn process_image_advanced(
    input_bytes: &[u8],
    k_colors: Option<u32>,
    no_resize: Option<bool>,
    algorithm: Option<u32>,
) -> std::result::Result<Vec<u8>, wasm_bindgen::JsValue> {
    let mut config = Config::default();
    if let Some(k) = k_colors {
        if k == 0 {
            return Err(wasm_bindgen::JsValue::from_str(
                "k_colors must be greater than 0",
            ));
        }
        config.k_colors = k as usize;
    }
    if let Some(nr) = no_resize {
        config.no_resize = nr;
    }
    if let Some(alg) = algorithm {
        config.algorithm = QuantAlgorithm::from(alg);
    }

    process_image_bytes_common(input_bytes, Some(config))
        .map_err(|e| wasm_bindgen::JsValue::from(e))
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn parse_args() -> Option<Config> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        return None;
    }

    let mut config = Config {
        input_path: args[1].clone(),
        output_path: args[2].clone(),
        ..Default::default()
    };

    if let Some(k_arg) = args.get(3) {
        match k_arg.parse::<usize>() {
            Ok(k) if k > 0 => config.k_colors = k,
            _ => eprintln!(
                "Warning: invalid k_colors '{}', falling back to default ({})",
                k_arg, config.k_colors
            ),
        }
    }

    Some(config)
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn process_image(config: &Config) -> Result<()> {
    println!("Processing: {}", config.input_path);

    let img_bytes = std::fs::read(&config.input_path).map_err(|e| {
        PixelSnapperError::ProcessingError(format!("Failed to read input file: {}", e))
    })?;

    let output_bytes = process_image_bytes_common(&img_bytes, Some(config.clone()))?;

    std::fs::write(&config.output_path, output_bytes).map_err(|e| {
        PixelSnapperError::ProcessingError(format!("Failed to write output file: {}", e))
    })?;

    println!("Saved to: {}", config.output_path);
    Ok(())
}

fn validate_image_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(PixelSnapperError::InvalidInput(
            "Image dimensions cannot be zero".to_string(),
        ));
    }
    if width > 10000 || height > 10000 {
        return Err(PixelSnapperError::InvalidInput(
            "Image dimensions too large (max 10000x10000)".to_string(),
        ));
    }
    Ok(())
}

fn quantize_image(img: &RgbaImage, config: &Config) -> Result<RgbaImage> {
    if config.k_colors == 0 {
        return Err(PixelSnapperError::InvalidInput(
            "Number of colors must be greater than 0".to_string(),
        ));
    }

    let opaque_pixels: Vec<[f32; 3]> = img
        .pixels()
        .filter_map(|p| {
            if p[3] == 0 {
                None
            } else {
                Some([p[0] as f32, p[1] as f32, p[2] as f32])
            }
        })
        .collect();
    let n_pixels = opaque_pixels.len();
    if n_pixels == 0 {
        return Ok(img.clone());
    }

    let mut rng = ChaCha8Rng::seed_from_u64(config.k_seed);
    let k = config.k_colors.min(n_pixels);

    fn sample_index(rng: &mut ChaCha8Rng, upper: usize) -> usize {
        debug_assert!(upper > 0);
        let upper = upper as u64;
        rng.gen_range(0..upper) as usize
    }

    fn dist_sq(p: &[f32; 3], c: &[f32; 3]) -> f32 {
        let dr = p[0] - c[0];
        let dg = p[1] - c[1];
        let db = p[2] - c[2];
        dr * dr + dg * dg + db * db
    }

    let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
    let first_idx = sample_index(&mut rng, n_pixels);
    centroids.push(opaque_pixels[first_idx]);
    let mut distances = vec![f32::MAX; n_pixels];

    // Maybe try a faster algorithm for this? like https://crates.io/crates/kmeans_colors
    for _ in 1..k {
        let last_c = centroids.last().unwrap();
        let mut sum_sq_dist = 0.0;

        for (i, p) in opaque_pixels.iter().enumerate() {
            let d_sq = dist_sq(p, last_c);
            if d_sq < distances[i] {
                distances[i] = d_sq;
            }
            sum_sq_dist += distances[i];
        }

        if sum_sq_dist <= 0.0 {
            let idx = sample_index(&mut rng, n_pixels);
            centroids.push(opaque_pixels[idx]);
        } else {
            let dist = WeightedIndex::new(&distances).map_err(|e| {
                PixelSnapperError::ProcessingError(format!("Failed to sample new centroid: {}", e))
            })?;
            let idx = dist.sample(&mut rng);
            centroids.push(opaque_pixels[idx]);
        }
    }

    let mut prev_centroids = centroids.clone();
    for iteration in 0..config.max_kmeans_iterations {
        let mut sums = vec![[0.0f32; 3]; k];
        let mut counts = vec![0usize; k];

        for p in &opaque_pixels {
            let mut min_dist = f32::MAX;
            let mut best_k = 0;

            for (i, c) in centroids.iter().enumerate() {
                let d = dist_sq(p, c);
                if d < min_dist {
                    min_dist = d;
                    best_k = i;
                }
            }
            sums[best_k][0] += p[0];
            sums[best_k][1] += p[1];
            sums[best_k][2] += p[2];
            counts[best_k] += 1;
        }

        for i in 0..k {
            if counts[i] > 0 {
                let fcount = counts[i] as f32;
                centroids[i] = [
                    sums[i][0] / fcount,
                    sums[i][1] / fcount,
                    sums[i][2] / fcount,
                ];
            }
        }

        if iteration > 0 {
            let mut max_movement = 0.0f32;
            for (new_c, old_c) in centroids.iter().zip(prev_centroids.iter()) {
                let movement = dist_sq(new_c, old_c);
                if movement > max_movement {
                    max_movement = movement;
                }
            }

            if max_movement < 0.01 {
                break;
            }
        }

        prev_centroids.copy_from_slice(&centroids);
    }

    let mut new_img = RgbaImage::new(img.width(), img.height());
    for (x, y, pixel) in img.enumerate_pixels() {
        if pixel[3] == 0 {
            new_img.put_pixel(x, y, *pixel);
            continue;
        }
        let p = [pixel[0] as f32, pixel[1] as f32, pixel[2] as f32];
        let mut min_dist = f32::MAX;
        let mut best_c = [pixel[0], pixel[1], pixel[2]];

        for c in &centroids {
            let d = dist_sq(&p, c);
            if d < min_dist {
                min_dist = d;
                best_c = [c[0].round() as u8, c[1].round() as u8, c[2].round() as u8];
            }
        }
        new_img.put_pixel(x, y, Rgba([best_c[0], best_c[1], best_c[2], pixel[3]]));
    }
    Ok(new_img)
}

/// K-means++ clustering in OKLab perceptual color space (no-resize mode only)
fn quantize_image_oklab(img: &RgbaImage, config: &Config) -> Result<RgbaImage> {
    if config.k_colors == 0 {
        return Err(PixelSnapperError::InvalidInput(
            "Number of colors must be greater than 0".to_string(),
        ));
    }

    // Convert opaque pixels to OKLab
    let opaque_pixels: Vec<[f32; 3]> = img
        .pixels()
        .filter_map(|p| {
            if p[3] == 0 {
                None
            } else {
                Some(srgb_to_oklab(p[0], p[1], p[2]))
            }
        })
        .collect();
    let n_pixels = opaque_pixels.len();
    if n_pixels == 0 {
        return Ok(img.clone());
    }

    let mut rng = ChaCha8Rng::seed_from_u64(config.k_seed);
    let k = config.k_colors.min(n_pixels);

    fn sample_index(rng: &mut ChaCha8Rng, upper: usize) -> usize {
        debug_assert!(upper > 0);
        rng.gen_range(0..upper as u64) as usize
    }

    fn dist_sq(p: &[f32; 3], c: &[f32; 3]) -> f32 {
        let dl = p[0] - c[0];
        let da = p[1] - c[1];
        let db = p[2] - c[2];
        dl * dl + da * da + db * db
    }

    // K-means++ initialization in OKLab space
    let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
    let first_idx = sample_index(&mut rng, n_pixels);
    centroids.push(opaque_pixels[first_idx]);
    let mut distances = vec![f32::MAX; n_pixels];

    for _ in 1..k {
        let last_c = centroids.last().unwrap();
        let mut sum_sq_dist = 0.0;

        for (i, p) in opaque_pixels.iter().enumerate() {
            let d_sq = dist_sq(p, last_c);
            if d_sq < distances[i] {
                distances[i] = d_sq;
            }
            sum_sq_dist += distances[i];
        }

        if sum_sq_dist <= 0.0 {
            let idx = sample_index(&mut rng, n_pixels);
            centroids.push(opaque_pixels[idx]);
        } else {
            let dist = WeightedIndex::new(&distances).map_err(|e| {
                PixelSnapperError::ProcessingError(format!("Failed to sample new centroid: {}", e))
            })?;
            let idx = dist.sample(&mut rng);
            centroids.push(opaque_pixels[idx]);
        }
    }

    // K-means iterations in OKLab space
    let mut prev_centroids = centroids.clone();
    for iteration in 0..config.max_kmeans_iterations {
        let mut sums = vec![[0.0f32; 3]; k];
        let mut counts = vec![0usize; k];

        for p in &opaque_pixels {
            let mut min_dist = f32::MAX;
            let mut best_k = 0;

            for (i, c) in centroids.iter().enumerate() {
                let d = dist_sq(p, c);
                if d < min_dist {
                    min_dist = d;
                    best_k = i;
                }
            }
            sums[best_k][0] += p[0];
            sums[best_k][1] += p[1];
            sums[best_k][2] += p[2];
            counts[best_k] += 1;
        }

        for i in 0..k {
            if counts[i] > 0 {
                let fcount = counts[i] as f32;
                centroids[i] = [
                    sums[i][0] / fcount,
                    sums[i][1] / fcount,
                    sums[i][2] / fcount,
                ];
            }
        }

        if iteration > 0 {
            let mut max_movement = 0.0f32;
            for (new_c, old_c) in centroids.iter().zip(prev_centroids.iter()) {
                let movement = dist_sq(new_c, old_c);
                if movement > max_movement {
                    max_movement = movement;
                }
            }

            if max_movement < 0.0001 {
                break;
            }
        }

        prev_centroids.copy_from_slice(&centroids);
    }

    // Convert centroids back to sRGB and remap pixels
    let srgb_centroids: Vec<[u8; 3]> = centroids.iter().map(|c| oklab_to_srgb(c)).collect();

    let mut new_img = RgbaImage::new(img.width(), img.height());
    for (x, y, pixel) in img.enumerate_pixels() {
        if pixel[3] == 0 {
            new_img.put_pixel(x, y, *pixel);
            continue;
        }
        let p = srgb_to_oklab(pixel[0], pixel[1], pixel[2]);
        let mut min_dist = f32::MAX;
        let mut best_idx = 0;

        for (i, c) in centroids.iter().enumerate() {
            let d = dist_sq(&p, c);
            if d < min_dist {
                min_dist = d;
                best_idx = i;
            }
        }
        let rgb = srgb_centroids[best_idx];
        new_img.put_pixel(x, y, Rgba([rgb[0], rgb[1], rgb[2], pixel[3]]));
    }
    Ok(new_img)
}

/// Median Cut color quantization (no-resize mode only)
fn median_cut_quantize(img: &RgbaImage, k_colors: usize) -> Result<RgbaImage> {
    if k_colors == 0 {
        return Err(PixelSnapperError::InvalidInput(
            "Number of colors must be greater than 0".to_string(),
        ));
    }

    let opaque_pixels: Vec<[u8; 3]> = img
        .pixels()
        .filter_map(|p| {
            if p[3] == 0 {
                None
            } else {
                Some([p[0], p[1], p[2]])
            }
        })
        .collect();

    if opaque_pixels.is_empty() {
        return Ok(img.clone());
    }

    // A "box" in color space containing a set of pixel indices
    struct ColorBox {
        pixels: Vec<[u8; 3]>,
    }

    impl ColorBox {
        fn longest_axis(&self) -> usize {
            let (mut min_r, mut min_g, mut min_b) = (255u8, 255u8, 255u8);
            let (mut max_r, mut max_g, mut max_b) = (0u8, 0u8, 0u8);
            for p in &self.pixels {
                min_r = min_r.min(p[0]);
                min_g = min_g.min(p[1]);
                min_b = min_b.min(p[2]);
                max_r = max_r.max(p[0]);
                max_g = max_g.max(p[1]);
                max_b = max_b.max(p[2]);
            }
            let range_r = max_r - min_r;
            let range_g = max_g - min_g;
            let range_b = max_b - min_b;
            if range_r >= range_g && range_r >= range_b {
                0
            } else if range_g >= range_b {
                1
            } else {
                2
            }
        }

        fn average_color(&self) -> [u8; 3] {
            if self.pixels.is_empty() {
                return [0, 0, 0];
            }
            let (mut sr, mut sg, mut sb) = (0u64, 0u64, 0u64);
            for p in &self.pixels {
                sr += p[0] as u64;
                sg += p[1] as u64;
                sb += p[2] as u64;
            }
            let n = self.pixels.len() as u64;
            [(sr / n) as u8, (sg / n) as u8, (sb / n) as u8]
        }

        fn split(mut self) -> (ColorBox, ColorBox) {
            let axis = self.longest_axis();
            self.pixels.sort_by_key(|p| p[axis]);
            let mid = self.pixels.len() / 2;
            let right = self.pixels.split_off(mid);
            (ColorBox { pixels: self.pixels }, ColorBox { pixels: right })
        }
    }

    let k = k_colors.min(opaque_pixels.len());
    let mut boxes = vec![ColorBox {
        pixels: opaque_pixels,
    }];

    // Split boxes until we have k boxes (or can't split further)
    while boxes.len() < k {
        // Find the box with the largest volume (most pixels) that can be split
        let split_idx = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.pixels.len() > 1)
            .max_by_key(|(_, b)| b.pixels.len())
            .map(|(i, _)| i);

        match split_idx {
            Some(idx) => {
                let b = boxes.remove(idx);
                let (left, right) = b.split();
                boxes.push(left);
                boxes.push(right);
            }
            None => break, // All boxes have 1 pixel, can't split further
        }
    }

    // Build palette from box averages
    let palette: Vec<[u8; 3]> = boxes.iter().map(|b| b.average_color()).collect();

    fn dist_sq_u8(a: &[u8; 3], b: &[u8; 3]) -> u32 {
        let dr = a[0] as i32 - b[0] as i32;
        let dg = a[1] as i32 - b[1] as i32;
        let db = a[2] as i32 - b[2] as i32;
        (dr * dr + dg * dg + db * db) as u32
    }

    // Remap pixels to nearest palette entry
    let mut new_img = RgbaImage::new(img.width(), img.height());
    for (x, y, pixel) in img.enumerate_pixels() {
        if pixel[3] == 0 {
            new_img.put_pixel(x, y, *pixel);
            continue;
        }
        let p = [pixel[0], pixel[1], pixel[2]];
        let mut min_dist = u32::MAX;
        let mut best_c = p;
        for c in &palette {
            let d = dist_sq_u8(&p, c);
            if d < min_dist {
                min_dist = d;
                best_c = *c;
            }
        }
        new_img.put_pixel(x, y, Rgba([best_c[0], best_c[1], best_c[2], pixel[3]]));
    }
    Ok(new_img)
}

/// Wu's Optimal Color Quantization (Xiaolin Wu, 1992)
/// Uses 3D histogram with moment tables and greedy variance-minimizing splits.
fn wu_quantize(img: &RgbaImage, k_colors: usize) -> Result<RgbaImage> {
    if k_colors == 0 {
        return Err(PixelSnapperError::InvalidInput(
            "Number of colors must be greater than 0".to_string(),
        ));
    }

    const HIST_SIZE: usize = 33; // indices 0..32, where bin = channel/8 + 1
    const TOTAL: usize = HIST_SIZE * HIST_SIZE * HIST_SIZE;

    fn idx(r: usize, g: usize, b: usize) -> usize {
        r * HIST_SIZE * HIST_SIZE + g * HIST_SIZE + b
    }

    // Build 3D histogram (weight, moment_r, moment_g, moment_b, moment2)
    let mut wt = vec![0i64; TOTAL];
    let mut mr = vec![0i64; TOTAL];
    let mut mg = vec![0i64; TOTAL];
    let mut mb = vec![0i64; TOTAL];
    let mut m2 = vec![0.0f64; TOTAL];

    for pixel in img.pixels() {
        if pixel[3] == 0 {
            continue;
        }
        let ri = (pixel[0] as usize >> 3) + 1;
        let gi = (pixel[1] as usize >> 3) + 1;
        let bi = (pixel[2] as usize >> 3) + 1;
        let i = idx(ri, gi, bi);
        wt[i] += 1;
        mr[i] += pixel[0] as i64;
        mg[i] += pixel[1] as i64;
        mb[i] += pixel[2] as i64;
        m2[i] += (pixel[0] as f64).powi(2)
            + (pixel[1] as f64).powi(2)
            + (pixel[2] as f64).powi(2);
    }

    // Compute cumulative moments using 3D prefix sums
    // Process along each axis sequentially
    for r in 1..HIST_SIZE {
        for g in 1..HIST_SIZE {
            for b in 1..HIST_SIZE {
                let i = idx(r, g, b);
                let i_r = idx(r - 1, g, b);
                let i_g = idx(r, g - 1, b);
                let i_b = idx(r, g, b - 1);
                let i_rg = idx(r - 1, g - 1, b);
                let i_rb = idx(r - 1, g, b - 1);
                let i_gb = idx(r, g - 1, b - 1);
                let i_rgb = idx(r - 1, g - 1, b - 1);

                wt[i] = wt[i] + wt[i_r] + wt[i_g] + wt[i_b]
                    - wt[i_rg] - wt[i_rb] - wt[i_gb] + wt[i_rgb];
                mr[i] = mr[i] + mr[i_r] + mr[i_g] + mr[i_b]
                    - mr[i_rg] - mr[i_rb] - mr[i_gb] + mr[i_rgb];
                mg[i] = mg[i] + mg[i_r] + mg[i_g] + mg[i_b]
                    - mg[i_rg] - mg[i_rb] - mg[i_gb] + mg[i_rgb];
                mb[i] = mb[i] + mb[i_r] + mb[i_g] + mb[i_b]
                    - mb[i_rg] - mb[i_rb] - mb[i_gb] + mb[i_rgb];
                m2[i] = m2[i] + m2[i_r] + m2[i_g] + m2[i_b]
                    - m2[i_rg] - m2[i_rb] - m2[i_gb] + m2[i_rgb];
            }
        }
    }

    // A box in histogram space
    #[derive(Clone)]
    struct WuBox {
        r0: usize, r1: usize,
        g0: usize, g1: usize,
        b0: usize, b1: usize,
    }

    // Volume (sum of a moment within a box) using inclusion-exclusion
    fn vol(b: &WuBox, moment: &[i64]) -> i64 {
        let idx = |r, g, b| r * HIST_SIZE * HIST_SIZE + g * HIST_SIZE + b;
        moment[idx(b.r1, b.g1, b.b1)]
            - moment[idx(b.r1, b.g1, b.b0)]
            - moment[idx(b.r1, b.g0, b.b1)]
            + moment[idx(b.r1, b.g0, b.b0)]
            - moment[idx(b.r0, b.g1, b.b1)]
            + moment[idx(b.r0, b.g1, b.b0)]
            + moment[idx(b.r0, b.g0, b.b1)]
            - moment[idx(b.r0, b.g0, b.b0)]
    }

    fn vol_f64(b: &WuBox, moment: &[f64]) -> f64 {
        let idx = |r, g, b| r * HIST_SIZE * HIST_SIZE + g * HIST_SIZE + b;
        moment[idx(b.r1, b.g1, b.b1)]
            - moment[idx(b.r1, b.g1, b.b0)]
            - moment[idx(b.r1, b.g0, b.b1)]
            + moment[idx(b.r1, b.g0, b.b0)]
            - moment[idx(b.r0, b.g1, b.b1)]
            + moment[idx(b.r0, b.g1, b.b0)]
            + moment[idx(b.r0, b.g0, b.b1)]
            - moment[idx(b.r0, b.g0, b.b0)]
    }

    // Variance of a box = m2 - (mr^2 + mg^2 + mb^2) / wt
    fn variance(b: &WuBox, wt: &[i64], mr: &[i64], mg: &[i64], mb: &[i64], m2: &[f64]) -> f64 {
        let w = vol(b, wt) as f64;
        if w <= 0.0 {
            return 0.0;
        }
        let vr = vol(b, mr) as f64;
        let vg = vol(b, mg) as f64;
        let vb = vol(b, mb) as f64;
        let vm2 = vol_f64(b, m2);
        vm2 - (vr * vr + vg * vg + vb * vb) / w
    }

    // "Bottom" slice helper for computing partial sums along a cut axis
    fn bottom(b: &WuBox, axis: usize, pos: usize, moment: &[i64]) -> i64 {
        let idx = |r, g, b| r * HIST_SIZE * HIST_SIZE + g * HIST_SIZE + b;
        match axis {
            0 => { // cut along R
                -moment[idx(pos, b.g1, b.b1)]
                    + moment[idx(pos, b.g1, b.b0)]
                    + moment[idx(pos, b.g0, b.b1)]
                    - moment[idx(pos, b.g0, b.b0)]
            }
            1 => { // cut along G
                -moment[idx(b.r1, pos, b.b1)]
                    + moment[idx(b.r1, pos, b.b0)]
                    + moment[idx(b.r0, pos, b.b1)]
                    - moment[idx(b.r0, pos, b.b0)]
            }
            _ => { // cut along B
                -moment[idx(b.r1, b.g1, pos)]
                    + moment[idx(b.r1, b.g0, pos)]
                    + moment[idx(b.r0, b.g1, pos)]
                    - moment[idx(b.r0, b.g0, pos)]
            }
        }
    }

    fn top(b: &WuBox, axis: usize, pos: usize, moment: &[i64]) -> i64 {
        let idx = |r, g, b| r * HIST_SIZE * HIST_SIZE + g * HIST_SIZE + b;
        match axis {
            0 => {
                moment[idx(pos, b.g1, b.b1)]
                    - moment[idx(pos, b.g1, b.b0)]
                    - moment[idx(pos, b.g0, b.b1)]
                    + moment[idx(pos, b.g0, b.b0)]
            }
            1 => {
                moment[idx(b.r1, pos, b.b1)]
                    - moment[idx(b.r1, pos, b.b0)]
                    - moment[idx(b.r0, pos, b.b1)]
                    + moment[idx(b.r0, pos, b.b0)]
            }
            _ => {
                moment[idx(b.r1, b.g1, pos)]
                    - moment[idx(b.r1, b.g0, pos)]
                    - moment[idx(b.r0, b.g1, pos)]
                    + moment[idx(b.r0, b.g0, pos)]
            }
        }
    }

    // Try to find the best cut position along a given axis for a box
    fn maximize(
        b: &WuBox, axis: usize,
        wt: &[i64], mr: &[i64], mg: &[i64], mb: &[i64],
    ) -> Option<(usize, f64)> {
        let whole_wt = vol(b, wt);
        let whole_mr = vol(b, mr);
        let whole_mg = vol(b, mg);
        let whole_mb = vol(b, mb);

        let (lo, hi) = match axis {
            0 => (b.r0, b.r1),
            1 => (b.g0, b.g1),
            _ => (b.b0, b.b1),
        };

        if hi <= lo + 1 {
            return None;
        }

        let base_wt = bottom(b, axis, lo, wt);
        let base_mr = bottom(b, axis, lo, mr);
        let base_mg = bottom(b, axis, lo, mg);
        let base_mb = bottom(b, axis, lo, mb);

        let mut best_pos = None;
        let mut best_max = 0.0f64;

        for pos in (lo + 1)..hi {
            let half_wt = base_wt + top(b, axis, pos, wt);
            if half_wt <= 0 {
                continue;
            }
            let half_mr = base_mr + top(b, axis, pos, mr);
            let half_mg = base_mg + top(b, axis, pos, mg);
            let half_mb = base_mb + top(b, axis, pos, mb);

            let temp = (half_mr as f64 * half_mr as f64
                + half_mg as f64 * half_mg as f64
                + half_mb as f64 * half_mb as f64)
                / half_wt as f64;

            let other_wt = whole_wt - half_wt;
            if other_wt <= 0 {
                continue;
            }
            let other_mr = whole_mr - half_mr;
            let other_mg = whole_mg - half_mg;
            let other_mb = whole_mb - half_mb;

            let temp2 = (other_mr as f64 * other_mr as f64
                + other_mg as f64 * other_mg as f64
                + other_mb as f64 * other_mb as f64)
                / other_wt as f64;

            let total = temp + temp2;
            if total > best_max {
                best_max = total;
                best_pos = Some(pos);
            }
        }

        best_pos.map(|pos| (pos, best_max))
    }

    // Initialize with one box spanning the entire histogram
    let k = k_colors.min(HIST_SIZE - 1);
    let mut boxes = vec![WuBox {
        r0: 0, r1: HIST_SIZE - 1,
        g0: 0, g1: HIST_SIZE - 1,
        b0: 0, b1: HIST_SIZE - 1,
    }];
    let mut var_cache: Vec<f64> = vec![variance(&boxes[0], &wt, &mr, &mg, &mb, &m2)];

    // Greedy splitting loop
    for _ in 1..k {
        // Find box with maximum variance
        let split_idx = var_cache
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, v)| (i, *v));

        let (best_idx, best_var) = match split_idx {
            Some((i, v)) if v > 0.0 => (i, v),
            _ => break,
        };
        let _ = best_var;

        let b = &boxes[best_idx];

        // Try cutting along each axis, pick the best
        let cut_r = maximize(b, 0, &wt, &mr, &mg, &mb);
        let cut_g = maximize(b, 1, &wt, &mr, &mg, &mb);
        let cut_b = maximize(b, 2, &wt, &mr, &mg, &mb);

        let best_cut = [cut_r, cut_g, cut_b]
            .iter()
            .enumerate()
            .filter_map(|(axis, opt)| opt.map(|(pos, score)| (axis, pos, score)))
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

        match best_cut {
            Some((axis, pos, _)) => {
                let parent = boxes[best_idx].clone();
                let mut b1 = parent.clone();
                let mut b2 = parent;

                match axis {
                    0 => { b1.r1 = pos; b2.r0 = pos; }
                    1 => { b1.g1 = pos; b2.g0 = pos; }
                    _ => { b1.b1 = pos; b2.b0 = pos; }
                }

                let v1 = variance(&b1, &wt, &mr, &mg, &mb, &m2);
                let v2 = variance(&b2, &wt, &mr, &mg, &mb, &m2);

                boxes[best_idx] = b1;
                var_cache[best_idx] = v1;
                boxes.push(b2);
                var_cache.push(v2);
            }
            None => break,
        }
    }

    // Compute palette: average color of each box
    let palette: Vec<[u8; 3]> = boxes
        .iter()
        .map(|b| {
            let w = vol(b, &wt);
            if w == 0 {
                return [0, 0, 0];
            }
            let r = (vol(b, &mr) as f64 / w as f64).round().clamp(0.0, 255.0) as u8;
            let g = (vol(b, &mg) as f64 / w as f64).round().clamp(0.0, 255.0) as u8;
            let bl = (vol(b, &mb) as f64 / w as f64).round().clamp(0.0, 255.0) as u8;
            [r, g, bl]
        })
        .collect();

    // Build lookup table: for each histogram bin, find the nearest palette entry
    let mut lut = vec![0u8; TOTAL];
    for ri in 1..HIST_SIZE {
        for gi in 1..HIST_SIZE {
            for bi in 1..HIST_SIZE {
                let bin_r = ((ri as u32 - 1) * 8 + 4) as u8; // bin center
                let bin_g = ((gi as u32 - 1) * 8 + 4) as u8;
                let bin_b = ((bi as u32 - 1) * 8 + 4) as u8;
                let mut min_dist = u32::MAX;
                let mut best = 0u8;
                for (pi, pc) in palette.iter().enumerate() {
                    let dr = bin_r as i32 - pc[0] as i32;
                    let dg = bin_g as i32 - pc[1] as i32;
                    let db = bin_b as i32 - pc[2] as i32;
                    let d = (dr * dr + dg * dg + db * db) as u32;
                    if d < min_dist {
                        min_dist = d;
                        best = pi as u8;
                    }
                }
                lut[idx(ri, gi, bi)] = best;
            }
        }
    }

    // Remap pixels using the lookup table
    let mut new_img = RgbaImage::new(img.width(), img.height());
    for (x, y, pixel) in img.enumerate_pixels() {
        if pixel[3] == 0 {
            new_img.put_pixel(x, y, *pixel);
            continue;
        }
        let ri = (pixel[0] as usize >> 3) + 1;
        let gi = (pixel[1] as usize >> 3) + 1;
        let bi = (pixel[2] as usize >> 3) + 1;
        let c = palette[lut[idx(ri, gi, bi)] as usize];
        new_img.put_pixel(x, y, Rgba([c[0], c[1], c[2], pixel[3]]));
    }
    Ok(new_img)
}

fn compute_profiles(img: &RgbaImage) -> Result<(Vec<f64>, Vec<f64>)> {
    let (w, h) = img.dimensions();

    if w < 3 || h < 3 {
        return Err(PixelSnapperError::InvalidInput(
            "Image too small (minimum 3x3)".to_string(),
        ));
    }

    let mut col_proj = vec![0.0; w as usize];
    let mut row_proj = vec![0.0; h as usize];

    let gray = |x, y| {
        let p = img.get_pixel(x, y);
        if p[3] == 0 {
            0.0
        } else {
            0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64
        }
    };

    // kernels: [-1, 0, 1]
    for y in 0..h {
        for x in 1..w - 1 {
            let left = gray(x - 1, y);
            let right = gray(x + 1, y);
            let grad = (right - left).abs();
            col_proj[x as usize] += grad;
        }
    }
    for x in 0..w {
        for y in 1..h - 1 {
            let top = gray(x, y - 1);
            let bottom = gray(x, y + 1);
            let grad = (bottom - top).abs();
            row_proj[y as usize] += grad;
        }
    }

    Ok((col_proj, row_proj))
}

fn estimate_step_size(profile: &[f64], config: &Config) -> Option<f64> {
    if profile.is_empty() {
        return None;
    }

    let max_val = profile.iter().cloned().fold(0.0 / 0.0, f64::max);
    if max_val == 0.0 {
        return None; // Decide later
    }
    let threshold = max_val * config.peak_threshold_multiplier;

    let mut peaks = Vec::new();
    for i in 1..profile.len() - 1 {
        if profile[i] > threshold && profile[i] > profile[i - 1] && profile[i] > profile[i + 1] {
            peaks.push(i);
        }
    }

    if peaks.len() < 2 {
        return None;
    }

    let mut clean_peaks = vec![peaks[0]];
    for &p in peaks.iter().skip(1) {
        if p - clean_peaks.last().unwrap() > (config.peak_distance_filter - 1) {
            clean_peaks.push(p);
        }
    }

    if clean_peaks.len() < 2 {
        return None;
    }

    // Compute diffs
    let mut diffs: Vec<f64> = clean_peaks
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64)
        .collect();

    // Median
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Some(diffs[diffs.len() / 2])
}

fn resolve_step_sizes(
    step_x_opt: Option<f64>,
    step_y_opt: Option<f64>,
    width: u32,
    height: u32,
    config: &Config,
) -> (f64, f64) {
    match (step_x_opt, step_y_opt) {
        (Some(sx), Some(sy)) => {
            let ratio = if sx > sy { sx / sy } else { sy / sx };
            if ratio > config.max_step_ratio {
                let smaller = sx.min(sy);
                (smaller, smaller)
            } else {
                let avg = (sx + sy) / 2.0;
                (avg, avg)
            }
        }

        (Some(sx), None) => (sx, sx),

        (None, Some(sy)) => (sy, sy),

        (None, None) => {
            let fallback_step =
                ((width.min(height) as f64) / config.fallback_target_segments as f64).max(1.0);
            (fallback_step, fallback_step)
        }
    }
}

fn stabilize_both_axes(
    profile_x: &[f64],
    profile_y: &[f64],
    raw_col_cuts: Vec<usize>,
    raw_row_cuts: Vec<usize>,
    width: usize,
    height: usize,
    config: &Config,
) -> (Vec<usize>, Vec<usize>) {
    let col_cuts_pass1 = stabilize_cuts(
        profile_x,
        raw_col_cuts.clone(),
        width,
        &raw_row_cuts,
        height,
        config,
    );
    let row_cuts_pass1 = stabilize_cuts(
        profile_y,
        raw_row_cuts.clone(),
        height,
        &raw_col_cuts,
        width,
        config,
    );

    // Check if the results are coherent
    let col_cells = col_cuts_pass1.len().saturating_sub(1).max(1);
    let row_cells = row_cuts_pass1.len().saturating_sub(1).max(1);
    let col_step = width as f64 / col_cells as f64;
    let row_step = height as f64 / row_cells as f64;

    let step_ratio = if col_step > row_step {
        col_step / row_step
    } else {
        row_step / col_step
    };

    if step_ratio > config.max_step_ratio {
        let target_step = col_step.min(row_step);

        let final_col_cuts = if col_step > target_step * 1.2 {
            snap_uniform_cuts(
                profile_x,
                width,
                target_step,
                config,
                config.min_cuts_per_axis,
            )
        } else {
            col_cuts_pass1
        };

        let final_row_cuts = if row_step > target_step * 1.2 {
            snap_uniform_cuts(
                profile_y,
                height,
                target_step,
                config,
                config.min_cuts_per_axis,
            )
        } else {
            row_cuts_pass1
        };

        (final_col_cuts, final_row_cuts)
    } else {
        (col_cuts_pass1, row_cuts_pass1)
    }
}

// Tried uniform grid instead of an elastic-ish walker, but the result was a bit worse.
// Keeping the walker for now. But some distortions might happen...
fn walk(profile: &[f64], step_size: f64, limit: usize, config: &Config) -> Result<Vec<usize>> {
    if profile.is_empty() {
        return Err(PixelSnapperError::ProcessingError(
            "Cannot walk on empty profile".to_string(),
        ));
    }

    let mut cuts = vec![0];
    let mut current_pos = 0.0;
    let search_window =
        (step_size * config.walker_search_window_ratio).max(config.walker_min_search_window);
    let mean_val: f64 = profile.iter().sum::<f64>() / profile.len() as f64;

    while current_pos < limit as f64 {
        let target = current_pos + step_size;
        if target >= limit as f64 {
            cuts.push(limit);
            break;
        }

        let start_search = ((target - search_window) as usize).max((current_pos + 1.0) as usize);
        let end_search = ((target + search_window) as usize).min(limit);

        if end_search <= start_search {
            current_pos = target;
            continue;
        }

        let mut max_val = -1.0;
        let mut max_idx = start_search;
        for i in start_search..end_search {
            if profile[i] > max_val {
                max_val = profile[i];
                max_idx = i;
            }
        }

        if max_val > mean_val * config.walker_strength_threshold {
            cuts.push(max_idx);
            current_pos = max_idx as f64;
        } else {
            cuts.push(target as usize);
            current_pos = target;
        }
    }
    Ok(cuts)
}

fn stabilize_cuts(
    profile: &[f64],
    cuts: Vec<usize>,
    limit: usize,
    sibling_cuts: &[usize],
    sibling_limit: usize,
    config: &Config,
) -> Vec<usize> {
    if limit == 0 {
        return vec![0];
    }

    let cuts = sanitize_cuts(cuts, limit);
    let min_required = config.min_cuts_per_axis.max(2).min(limit.saturating_add(1));
    let axis_cells = cuts.len().saturating_sub(1);
    let sibling_cells = sibling_cuts.len().saturating_sub(1);
    let sibling_has_grid =
        sibling_limit > 0 && sibling_cells >= min_required.saturating_sub(1) && sibling_cells > 0;
    let steps_skewed = sibling_has_grid && axis_cells > 0 && {
        let axis_step = limit as f64 / axis_cells as f64;
        let sibling_step = sibling_limit as f64 / sibling_cells as f64;
        let step_ratio = axis_step / sibling_step;
        step_ratio > config.max_step_ratio || step_ratio < 1.0 / config.max_step_ratio
    };
    let has_enough = cuts.len() >= min_required;

    if has_enough && !steps_skewed {
        return cuts;
    }

    let mut target_step = if sibling_has_grid {
        sibling_limit as f64 / sibling_cells as f64
    } else if config.fallback_target_segments > 1 {
        limit as f64 / config.fallback_target_segments as f64
    } else if axis_cells > 0 {
        limit as f64 / axis_cells as f64
    } else {
        limit as f64
    };
    if !target_step.is_finite() || target_step <= 0.0 {
        target_step = 1.0;
    }

    snap_uniform_cuts(profile, limit, target_step, config, min_required)
}

fn sanitize_cuts(mut cuts: Vec<usize>, limit: usize) -> Vec<usize> {
    if limit == 0 {
        return vec![0];
    }

    let mut has_zero = false;
    let mut has_limit = false;

    for value in cuts.iter_mut() {
        if *value == 0 {
            has_zero = true;
        }
        if *value >= limit {
            *value = limit;
        }
        if *value == limit {
            has_limit = true;
        }
    }

    if !has_zero {
        cuts.push(0);
    }
    if !has_limit {
        cuts.push(limit);
    }

    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

fn snap_uniform_cuts(
    profile: &[f64],
    limit: usize,
    target_step: f64,
    config: &Config,
    min_required: usize,
) -> Vec<usize> {
    if limit == 0 {
        return vec![0];
    }
    if limit == 1 {
        return vec![0, 1];
    }

    // Get desired cells
    let mut desired_cells = if target_step.is_finite() && target_step > 0.0 {
        (limit as f64 / target_step).round() as usize
    } else {
        0
    };
    desired_cells = desired_cells
        .max(min_required.saturating_sub(1))
        .max(1)
        .min(limit);

    let cell_width = limit as f64 / desired_cells as f64;
    let search_window =
        (cell_width * config.walker_search_window_ratio).max(config.walker_min_search_window);
    let mean_val = if profile.is_empty() {
        0.0
    } else {
        profile.iter().sum::<f64>() / profile.len() as f64
    };

    let mut cuts = Vec::with_capacity(desired_cells + 1);
    cuts.push(0);
    for idx in 1..desired_cells {
        let target = cell_width * idx as f64;
        let prev = *cuts.last().unwrap();
        if prev + 1 >= limit {
            break;
        }
        let mut start = ((target - search_window).floor() as isize)
            .max(prev as isize + 1)
            .max(0);
        let mut end = ((target + search_window).ceil() as isize).min(limit as isize - 1);
        if end < start {
            start = prev as isize + 1;
            end = start;
        }
        let start = start as usize;
        let end = end as usize;
        let mut best_idx = start.min(profile.len().saturating_sub(1));
        let mut best_val = -1.0;
        for i in start..=end.min(profile.len().saturating_sub(1)) {
            let v = profile.get(i).copied().unwrap_or(0.0);
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        let strength_threshold = mean_val * config.walker_strength_threshold;
        if best_val < strength_threshold {
            let mut fallback_idx = target.round() as isize;
            if fallback_idx <= prev as isize {
                fallback_idx = prev as isize + 1;
            }
            if fallback_idx >= limit as isize {
                fallback_idx = (limit as isize - 1).max(prev as isize + 1);
            }
            best_idx = fallback_idx as usize;
        }
        cuts.push(best_idx);
    }
    if *cuts.last().unwrap() != limit {
        cuts.push(limit);
    }
    cuts = sanitize_cuts(cuts, limit);
    cuts
}

fn resample(img: &RgbaImage, cols: &[usize], rows: &[usize]) -> Result<RgbaImage> {
    if cols.len() < 2 || rows.len() < 2 {
        return Err(PixelSnapperError::ProcessingError(
            "Insufficient grid cuts for resampling".to_string(),
        ));
    }

    let out_w = (cols.len().max(1) - 1) as u32;
    let out_h = (rows.len().max(1) - 1) as u32;
    let mut final_img: RgbaImage = ImageBuffer::new(out_w, out_h);

    for (y_i, w_y) in rows.windows(2).enumerate() {
        for (x_i, w_x) in cols.windows(2).enumerate() {
            let ys = w_y[0];
            let ye = w_y[1];
            let xs = w_x[0];
            let xe = w_x[1];

            if xe <= xs || ye <= ys {
                continue;
            }

            let mut counts: HashMap<[u8; 4], usize> = HashMap::new();

            for y in ys..ye {
                for x in xs..xe {
                    if x < img.width() as usize && y < img.height() as usize {
                        let p = img.get_pixel(x as u32, y as u32).0;
                        *counts.entry(p).or_insert(0) += 1;
                    }
                }
            }

            let mut best_pixel = [0, 0, 0, 0];

            let mut candidates: Vec<([u8; 4], usize)> = counts.into_iter().collect();
            candidates.sort_by(|a, b| {
                let count_cmp = b.1.cmp(&a.1);
                if count_cmp == Ordering::Equal {
                    a.0.cmp(&b.0)
                } else {
                    count_cmp
                }
            });

            if let Some(winner) = candidates.first() {
                best_pixel = winner.0;
            }

            final_img.put_pixel(x_i as u32, y_i as u32, Rgba(best_pixel));
        }
    }
    Ok(final_img)
}
