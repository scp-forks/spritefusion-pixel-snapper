//! Evaluation harness (test-only).
//!
//! True pixel art, damaged the way image generators and upscalers damage it, and a
//! score for how much of it each mode gets back. The art is generated here, so the
//! truth is known exactly: the grid, every cell's colour, and where the dithering is.
//!
//! * `cargo test --release` runs a small fixed subset as a regression guard.
//! * `cargo test --release eval_report -- --ignored --nocapture` prints the full
//!   tables that `docs/research/05-evaluation.md` records.
//! * `cargo test --release potts_report -- --ignored --nocapture` prints the
//!   edge-weighted Potts against sieve comparison on `other/wolf.png`.

use crate::cleanup::{self, Strength, Vote};
use crate::grid::GridFitter;
use crate::{run_pipeline, Cleanup, Config};
use image::{imageops, Rgba, RgbaImage};
use rand::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// A piece of true pixel art: one image pixel per logical pixel.
pub struct Art {
    pub image: RgbaImage,
    /// True where the pixel belongs to a dithered (checkerboard) area.
    pub dither: Vec<bool>,
    pub colours: usize,
}

const PALETTE: [[u8; 3]; 8] = [
    [28, 24, 36],    // outline
    [92, 148, 220],  // sky
    [236, 240, 244], // cloud / highlight
    [74, 150, 70],   // grass
    [150, 100, 60],  // wood
    [214, 180, 110], // sand
    [200, 60, 50],   // accent
    [120, 124, 136], // stone
];

/// A small scene: sky over ground, outlined boxes with highlights and single-pixel
/// details, and optionally a band where sky and ground are dithered into each other.
pub fn scene(seed: u64, w: u32, h: u32, dither: bool) -> Art {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let px = |i: usize| Rgba([PALETTE[i][0], PALETTE[i][1], PALETTE[i][2], 255]);
    let mut image = RgbaImage::new(w, h);
    let mut mask = vec![false; (w * h) as usize];
    let horizon = h * 3 / 5;
    let band = if dither { 6 } else { 0 };
    for y in 0..h {
        for x in 0..w {
            let c = if y + band / 2 < horizon {
                1
            } else if y >= horizon + band / 2 {
                3
            } else {
                mask[(y * w + x) as usize] = true;
                if (x + y) % 2 == 0 { 1 } else { 3 }
            };
            image.put_pixel(x, y, px(c));
        }
    }
    // Clouds: flat blobs.
    for _ in 0..3 {
        let (cx, cy) = (rng.gen_range(4..w - 12), rng.gen_range(2..horizon.saturating_sub(band + 6).max(3)));
        let (bw, bh) = (rng.gen_range(6..12), rng.gen_range(2..4));
        for y in cy..(cy + bh).min(h) {
            for x in cx..(cx + bw).min(w) {
                image.put_pixel(x, y, px(2));
                mask[(y * w + x) as usize] = false;
            }
        }
    }
    // Outlined boxes with a highlight and a one-pixel detail.
    for _ in 0..5 {
        let (bw, bh) = (rng.gen_range(6..14), rng.gen_range(6..12));
        let (x0, y0) = (rng.gen_range(1..w - bw - 1), rng.gen_range(horizon.saturating_sub(bh / 2)..h - bh - 1));
        let fill = [4, 5, 6, 7][rng.gen_range(0..4)];
        for y in y0..y0 + bh {
            for x in x0..x0 + bw {
                let edge = x == x0 || y == y0 || x == x0 + bw - 1 || y == y0 + bh - 1;
                image.put_pixel(x, y, px(if edge { 0 } else { fill }));
                mask[(y * w + x) as usize] = false;
            }
        }
        image.put_pixel(x0 + 2, y0 + 2, px(2));
        image.put_pixel(x0 + bw - 3, y0 + bh - 3, px(0));
    }
    Art { image, dither: mask, colours: PALETTE.len() }
}

/// What was done to a piece of art to make it look generated.
#[derive(Clone, Copy, Debug)]
pub struct Damage {
    /// Source pixels per logical pixel; fractional on purpose.
    pub scale: f32,
    /// Gaussian blur, in source pixels.
    pub blur: f32,
    /// Uniform noise amplitude per channel.
    pub noise: i32,
    /// JPEG quality, or `None` for lossless.
    pub jpeg: Option<u8>,
}

pub fn damage(art: &Art, d: Damage, seed: u64) -> RgbaImage {
    let (w, h) = art.image.dimensions();
    let (tw, th) = ((w as f32 * d.scale).round() as u32, (h as f32 * d.scale).round() as u32);
    // Crisp upscale by 8, then a smooth resize to the target: edges about a pixel soft.
    let big = imageops::resize(&art.image, w * 8, h * 8, imageops::FilterType::Nearest);
    let mut out = imageops::resize(&big, tw, th, imageops::FilterType::Triangle);
    if d.blur > 0.0 {
        out = imageops::blur(&out, d.blur);
    }
    if d.noise > 0 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for p in out.pixels_mut() {
            for ch in 0..3 {
                p[ch] = (p[ch] as i32 + rng.gen_range(-d.noise..=d.noise)).clamp(0, 255) as u8;
            }
        }
    }
    if let Some(q) = d.jpeg {
        let mut bytes = Vec::new();
        let rgb = image::DynamicImage::ImageRgba8(out).to_rgb8();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, q)
            .encode_image(&rgb)
            .unwrap();
        out = image::load_from_memory(&bytes).unwrap().to_rgba8();
    }
    out
}

fn nearest_palette(p: &Rgba<u8>) -> usize {
    (0..PALETTE.len())
        .min_by_key(|&i| (0..3).map(|c| (p[c] as i32 - PALETTE[i][c] as i32).pow(2)).sum::<i32>())
        .unwrap()
}

/// How a result compares with the truth.
#[derive(Default, Clone, Copy, Debug)]
pub struct Score {
    /// Share of logical pixels whose colour (matched to the true palette) is right,
    /// sampled at the centre of each. Speckle lives on edges, so this does not see it.
    pub accuracy: f64,
    /// The same inside dithered areas only; NaN when the art has none.
    pub dither_accuracy: f64,
    /// For a result the size of the damaged image: the share of **all** its pixels
    /// that have the colour of the logical pixel they belong to, i.e. how close it is
    /// to a crisp upscale of the truth. NaN for a snapped result.
    pub pixel_accuracy: f64,
}

/// Score `out` against `art`. `out` may be the small snap image or a full-size one:
/// either way it is sampled at the centre of every true logical pixel. `full_size`
/// says whether `out` is a full-size result, for [`Score::pixel_accuracy`].
pub fn score(art: &Art, out: &RgbaImage, full_size: bool) -> Score {
    let (w, h) = art.image.dimensions();
    let (sx, sy) = (out.width() as f64 / w as f64, out.height() as f64 / h as f64);
    let (mut right, mut d_right, mut d_total) = (0u32, 0u32, 0u32);
    for y in 0..h {
        for x in 0..w {
            let ox = (((x as f64 + 0.5) * sx) as u32).min(out.width() - 1);
            let oy = (((y as f64 + 0.5) * sy) as u32).min(out.height() - 1);
            let ok = nearest_palette(out.get_pixel(ox, oy)) == nearest_palette(art.image.get_pixel(x, y));
            right += ok as u32;
            if art.dither[(y * w + x) as usize] {
                d_total += 1;
                d_right += ok as u32;
            }
        }
    }
    let pixel_accuracy = if full_size {
        let mut same = 0u64;
        for (x, y, p) in out.enumerate_pixels() {
            // Pixel centres, as resizing uses: a perfect crisp upscale must score 1.
            let tx = (((x as f64 + 0.5) / sx) as u32).min(w - 1);
            let ty = (((y as f64 + 0.5) / sy) as u32).min(h - 1);
            same += (nearest_palette(p) == nearest_palette(art.image.get_pixel(tx, ty))) as u64;
        }
        same as f64 / (out.width() as u64 * out.height() as u64) as f64
    } else {
        f64::NAN
    };
    Score {
        accuracy: right as f64 / (w * h) as f64,
        dither_accuracy: if d_total == 0 { f64::NAN } else { d_right as f64 / d_total as f64 },
        pixel_accuracy,
    }
}

/// One mode of the tool, as the evaluation runs it.
#[derive(Clone, Copy, Debug)]
pub struct Mode {
    pub name: &'static str,
    pub no_resize: bool,
    pub cleanup: Cleanup,
    pub strength: Strength,
    pub fitter: GridFitter,
    pub vote: Vote,
}

pub const MODES: [Mode; 9] = [
    Mode { name: "snap, standard", no_resize: false, cleanup: Cleanup::Off, strength: Strength::Light, fitter: GridFitter::Walk, vote: Vote::Majority },
    Mode { name: "snap, elastic", no_resize: false, cleanup: Cleanup::Off, strength: Strength::Light, fitter: GridFitter::Dp, vote: Vote::Majority },
    Mode { name: "snap, elastic, detail", no_resize: false, cleanup: Cleanup::Off, strength: Strength::Light, fitter: GridFitter::Dp, vote: Vote::Detail },
    Mode { name: "no-resize, plain", no_resize: true, cleanup: Cleanup::Off, strength: Strength::Light, fitter: GridFitter::Walk, vote: Vote::Majority },
    Mode { name: "blocks, standard", no_resize: true, cleanup: Cleanup::Blocks, strength: Strength::Light, fitter: GridFitter::Walk, vote: Vote::Majority },
    Mode { name: "blocks, elastic", no_resize: true, cleanup: Cleanup::Blocks, strength: Strength::Light, fitter: GridFitter::Dp, vote: Vote::Majority },
    Mode { name: "smooth, light", no_resize: true, cleanup: Cleanup::Smooth, strength: Strength::Light, fitter: GridFitter::Dp, vote: Vote::Majority },
    Mode { name: "smooth, medium", no_resize: true, cleanup: Cleanup::Smooth, strength: Strength::Medium, fitter: GridFitter::Dp, vote: Vote::Majority },
    Mode { name: "smooth, strong", no_resize: true, cleanup: Cleanup::Smooth, strength: Strength::Strong, fitter: GridFitter::Dp, vote: Vote::Majority },
];

/// Result of one mode on one damaged image.
pub struct Outcome {
    pub score: Score,
    /// Grid found, when the mode uses one.
    pub grid: Option<(usize, usize)>,
}

pub fn run(art: &Art, damaged: &RgbaImage, mode: Mode) -> Outcome {
    let config = Config {
        // Headroom over the true palette: a blurred image also holds every blend.
        k_colors: art.colours + 4,
        no_resize: mode.no_resize,
        cleanup: mode.cleanup,
        cleanup_strength: mode.strength,
        grid_fitter: mode.fitter,
        vote: mode.vote,
        ..Default::default()
    };
    let out = run_pipeline(damaged, &config).unwrap();
    Outcome {
        score: score(art, &out.image, mode.no_resize),
        grid: out.grid.map(|g| (g.cols.len() - 1, g.rows.len() - 1)),
    }
}

pub const DAMAGES: [(&str, Damage); 6] = [
    ("crisp x5 (nearest-like)", Damage { scale: 5.0, blur: 0.0, noise: 0, jpeg: None }),
    ("x5.3, soft", Damage { scale: 5.3, blur: 0.6, noise: 3, jpeg: None }),
    ("x7.5, soft", Damage { scale: 7.5, blur: 0.8, noise: 4, jpeg: None }),
    ("x9.3, softer", Damage { scale: 9.3, blur: 1.2, noise: 5, jpeg: None }),
    ("x7.5, soft, JPEG 80", Damage { scale: 7.5, blur: 0.8, noise: 4, jpeg: Some(80) }),
    ("x4.2, soft", Damage { scale: 4.2, blur: 0.5, noise: 3, jpeg: None }),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regions;

    const SCENES: usize = 6;
    const SIZE: (u32, u32) = (72, 54);

    fn arts(dither: bool) -> Vec<Art> {
        (0..SCENES as u64).map(|s| scene(100 + s, SIZE.0, SIZE.1, dither)).collect()
    }

    /// The generated art really is what it claims to be.
    #[test]
    fn scenes_are_valid_pixel_art() {
        for dither in [false, true] {
            for art in arts(dither) {
                assert_eq!(art.image.dimensions(), SIZE);
                assert!(regions::count_unique_colors(&art.image) <= PALETTE.len());
                assert_eq!(art.dither.iter().any(|&d| d), dither);
                // Scoring the truth against itself, at any size, is perfect.
                assert_eq!(score(&art, &art.image, false).accuracy, 1.0);
                // ...including fractional sizes, where the borders of the logical pixels
                // do not fall on whole source pixels.
                for scale in [5.0f32, 4.2, 5.3, 7.5, 9.3, 2.67] {
                    let (bw, bh) = ((SIZE.0 as f32 * scale).round() as u32, (SIZE.1 as f32 * scale).round() as u32);
                    let big = imageops::resize(&art.image, bw, bh, imageops::FilterType::Nearest);
                    let s = score(&art, &big, true);
                    assert_eq!((s.accuracy, s.pixel_accuracy), (1.0, 1.0), "scale {scale}");
                }
            }
        }
        // Damage changes the size as asked and is deterministic.
        let art = scene(1, 40, 30, true);
        let d = DAMAGES[1].1;
        assert_eq!(damage(&art, d, 7).dimensions(), (212, 159));
        assert_eq!(damage(&art, d, 7), damage(&art, d, 7));
    }

    /// Regression guard on a fixed subset: one scene, the elastic fitter, four kinds
    /// of damage. Floors are a little under what was measured when this was written
    /// (docs/research/05-evaluation.md). Scene 100 is the one whose sparse edges double
    /// the legacy step estimate.
    #[test]
    fn recovery_does_not_regress() {
        let truth = (SIZE.0 as usize, SIZE.1 as usize);
        let (elastic, plain, blocks) = (MODES[1], MODES[3], MODES[5]);
        for dither in [false, true] {
            let art = scene(100, SIZE.0, SIZE.1, dither);
            // Crisp art comes back exactly.
            let crisp = damage(&art, DAMAGES[0].1, 0);
            let out = run(&art, &crisp, elastic);
            assert_eq!(out.grid, Some(truth), "crisp, dither {dither}");
            assert_eq!(out.score.accuracy, 1.0, "crisp, dither {dither}");
            let full = run(&art, &crisp, blocks);
            assert_eq!(full.score.pixel_accuracy, 1.0, "crisp blocks, dither {dither}");

            // Soft art at fractional scales: the right size to within a cell, nearly
            // every cell right, and full-size Blocks at least as close to a crisp
            // upscale as plain palette reduction.
            for d in [DAMAGES[1].1, DAMAGES[2].1, DAMAGES[3].1] {
                let soft = damage(&art, d, 0);
                let snap = run(&art, &soft, elastic);
                let (cols, rows) = snap.grid.unwrap();
                assert!(
                    cols.abs_diff(truth.0) <= 1 && rows.abs_diff(truth.1) <= 1,
                    "{d:?} dither {dither}: {cols}x{rows}"
                );
                assert!(snap.score.accuracy > 0.98, "{d:?} dither {dither}: {:.4}", snap.score.accuracy);
                let (b, p) = (run(&art, &soft, blocks).score, run(&art, &soft, plain).score);
                assert!(b.pixel_accuracy + 0.002 >= p.pixel_accuracy, "{d:?} dither {dither}: blocks {:.4} < plain {:.4}", b.pixel_accuracy, p.pixel_accuracy);
                if dither {
                    assert!(b.dither_accuracy > 0.95, "{d:?}: blocks kept {:.3} of the dithering", b.dither_accuracy);
                }
            }
        }
    }

    fn mean(v: &[f64]) -> f64 {
        let v: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
        if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
    }

    /// The full tables. Slow (about a minute); run on demand.
    #[test]
    #[ignore]
    fn eval_report() {
        for dither in [false, true] {
            let arts = arts(dither);
            println!("\n### {} ({} scenes of {}x{} logical pixels)\n", if dither { "With a dithered band" } else { "No dithering" }, SCENES, SIZE.0, SIZE.1);
            println!("| damage | mode | grid exact | grid within 2 % | cells right | dithered cells right | pixels right |");
            println!("|---|---|---|---|---|---|---|");
            for (dname, d) in DAMAGES {
                let damaged: Vec<RgbaImage> = arts.iter().enumerate().map(|(i, a)| damage(a, d, i as u64)).collect();
                for mode in MODES {
                    let outcomes: Vec<Outcome> = arts.iter().zip(&damaged).map(|(a, img)| run(a, img, mode)).collect();
                    let grids: Vec<(usize, usize)> = outcomes.iter().filter_map(|o| o.grid).collect();
                    let exact = grids.iter().filter(|&&g| g == (SIZE.0 as usize, SIZE.1 as usize)).count();
                    let close = grids
                        .iter()
                        .filter(|&&(c, r)| (c as f64 / SIZE.0 as f64 - 1.0).abs() <= 0.02 && (r as f64 / SIZE.1 as f64 - 1.0).abs() <= 0.02)
                        .count();
                    let grid_cols = if grids.is_empty() { ("-".to_string(), "-".to_string()) } else { (format!("{exact}/{}", grids.len()), format!("{close}/{}", grids.len())) };
                    let acc = mean(&outcomes.iter().map(|o| o.score.accuracy).collect::<Vec<_>>());
                    let dacc = mean(&outcomes.iter().map(|o| o.score.dither_accuracy).collect::<Vec<_>>());
                    let pacc = mean(&outcomes.iter().map(|o| o.score.pixel_accuracy).collect::<Vec<_>>());
                    let pct = |v: f64| if v.is_nan() { "-".to_string() } else { format!("{:.1} %", v * 100.0) };
                    println!(
                        "| {dname} | {} | {} | {} | {} | {} | {} |",
                        mode.name,
                        grid_cols.0,
                        grid_cols.1,
                        pct(acc),
                        pct(dacc),
                        pct(pacc)
                    );
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Open experiment: edge-weighted Potts smoothing against the sieve
    // ------------------------------------------------------------------

    /// Iterated conditional modes on a Potts energy whose smoothness term is weak
    /// across strong source edges:
    ///   E = sum_p |c_p - palette[l_p]|^2  +  lambda * sum_{p~q} w_pq [l_p != l_q],
    ///   w_pq = exp(-|c_p - c_q|^2 / (2 sigma^2)),  4-neighbours.
    ///
    /// `nearest[p]` is the palette label closest to pixel p's colour. A pixel tries its
    /// current label, its neighbours' labels and its nearest label. That is enough: any
    /// other label disagrees with every neighbour, as the nearest one may, and costs
    /// more than the nearest one does in colour.
    fn potts_icm(labels: &mut [u16], nearest: &[u16], lab: &[[f32; 3]], pal: &[[f32; 3]], w: usize, h: usize, lambda: f32, sigma: f32, sweeps: usize) -> bool {
        for _ in 0..sweeps {
            let mut changed = 0usize;
            for i in 0..w * h {
                let nb = potts_neighbours(i, lab, w, h, sigma);
                let mut best = (potts_energy(i, labels[i], labels, &nb, lab, pal, lambda), labels[i]);
                for l in nb.iter().flatten().map(|&(j, _)| labels[j]).chain([nearest[i]]) {
                    let e = potts_energy(i, l, labels, &nb, lab, pal, lambda);
                    if e < best.0 {
                        best = (e, l);
                    }
                }
                if best.1 != labels[i] {
                    labels[i] = best.1;
                    changed += 1;
                }
            }
            if changed == 0 {
                return true;
            }
        }
        false
    }

    fn potts_neighbours(i: usize, lab: &[[f32; 3]], w: usize, h: usize, sigma: f32) -> [Option<(usize, f32)>; 4] {
        let d2 = |a: usize, b: usize| (lab[a][0] - lab[b][0]).powi(2) + (lab[a][1] - lab[b][1]).powi(2) + (lab[a][2] - lab[b][2]).powi(2);
        let (x, y) = (i % w, i / w);
        let at = |j: usize| Some((j, (-d2(i, j) / (2.0 * sigma * sigma)).exp()));
        [
            if x > 0 { at(i - 1) } else { None },
            if x + 1 < w { at(i + 1) } else { None },
            if y > 0 { at(i - w) } else { None },
            if y + 1 < h { at(i + w) } else { None },
        ]
    }

    fn potts_energy(i: usize, l: u16, labels: &[u16], nb: &[Option<(usize, f32)>; 4], lab: &[[f32; 3]], pal: &[[f32; 3]], lambda: f32) -> f32 {
        let p = pal[l as usize];
        let mut e = (lab[i][0] - p[0]).powi(2) + (lab[i][1] - p[1]).powi(2) + (lab[i][2] - p[2]).powi(2);
        for &(j, wt) in nb.iter().flatten() {
            if labels[j] != l {
                e += lambda * wt;
            }
        }
        e
    }

    /// When the optimiser reports convergence, no pixel can lower its energy with any
    /// label of the palette, not only with the ones it tried (Codex, Phase 7 review:
    /// trying only current and neighbouring labels stops short of that).
    #[test]
    fn potts_converges_to_a_true_local_minimum() {
        let mut rng = ChaCha8Rng::seed_from_u64(31);
        let (w, h) = (24usize, 18usize);
        let pal: Vec<[f32; 3]> = (0..6).map(|i| [0.15 + 0.15 * i as f32, 0.02 * i as f32, -0.03 * i as f32]).collect();
        let lab: Vec<[f32; 3]> = (0..w * h).map(|_| [rng.gen_range(0.1..0.95), rng.gen_range(-0.05..0.12), rng.gen_range(-0.18..0.05)]).collect();
        let nearest: Vec<u16> = lab
            .iter()
            .map(|c| (0..pal.len() as u16).min_by(|&a, &b| {
                let d = |l: u16| (0..3).map(|k| (c[k] - pal[l as usize][k]).powi(2)).sum::<f32>();
                d(a).partial_cmp(&d(b)).unwrap()
            }).unwrap())
            .collect();
        for (lambda, sigma) in [(0.01f32, 0.05f32), (0.05, 0.3), (0.3, 1.0)] {
            // Start from a scrambled labelling, so that a pixel's nearest label is often
            // absent from its neighbourhood.
            let mut labels: Vec<u16> = (0..w * h).map(|_| rng.gen_range(0..pal.len() as u16)).collect();
            assert!(potts_icm(&mut labels, &nearest, &lab, &pal, w, h, lambda, sigma, 200), "did not converge");
            for i in 0..w * h {
                let nb = potts_neighbours(i, &lab, w, h, sigma);
                let own = potts_energy(i, labels[i], &labels, &nb, &lab, &pal, lambda);
                for l in 0..pal.len() as u16 {
                    assert!(potts_energy(i, l, &labels, &nb, &lab, &pal, lambda) >= own - 1e-7, "pixel {i} prefers label {l}");
                }
            }
        }
    }

    /// Edge agreement of a label map with its source, with one tolerance for both
    /// directions of the comparison: one pixel, along the axis the edge crosses.
    ///
    /// * kept: share of the source's strong edges (over 0.10 OKLab between two
    ///   neighbouring pixels) that have a label boundary of the same orientation in the
    ///   same row or column, at most one pixel away. A de-blended edge legitimately
    ///   moves by a pixel; two pixels is a different edge.
    /// * spurious: share of label boundaries with no source edge of the same
    ///   orientation (over 0.02) at most one pixel away: boundaries in flat areas.
    fn edge_agreement(labels: &[u16], lab: &[[f32; 3]], w: usize, h: usize) -> (f64, f64) {
        let d = |a: usize, b: usize| ((lab[a][0] - lab[b][0]).powi(2) + (lab[a][1] - lab[b][1]).powi(2) + (lab[a][2] - lab[b][2]).powi(2)).sqrt();
        let (mut strong, mut kept, mut boundaries, mut spurious) = (0u64, 0u64, 0u64, 0u64);
        // Horizontal pass: pairs (x, x + 1) in a row. Then the same down columns.
        for (len, lines, at) in [
            (w, h, &(|line: usize, k: usize| line * w + k) as &dyn Fn(usize, usize) -> usize),
            (h, w, &(|line: usize, k: usize| k * w + line) as &dyn Fn(usize, usize) -> usize),
        ] {
            for line in 0..lines {
                let cut: Vec<bool> = (0..len - 1).map(|k| labels[at(line, k)] != labels[at(line, k + 1)]).collect();
                let edge: Vec<f32> = (0..len - 1).map(|k| d(at(line, k), at(line, k + 1))).collect();
                let near = |v: &dyn Fn(usize) -> bool, k: usize| (k.saturating_sub(1)..=(k + 1).min(len - 2)).any(|j| v(j));
                for k in 0..len - 1 {
                    if edge[k] > 0.10 {
                        strong += 1;
                        kept += near(&|j| cut[j], k) as u64;
                    }
                    if cut[k] {
                        boundaries += 1;
                        spurious += !near(&|j| edge[j] >= 0.02, k) as u64;
                    }
                }
            }
        }
        (kept as f64 / strong.max(1) as f64, spurious as f64 / boundaries.max(1) as f64)
    }

    #[test]
    fn edge_agreement_tolerates_one_pixel_and_no_more() {
        // An 12 x 4 image: dark | light with the source edge between columns 5 and 6.
        let (w, h) = (12usize, 4usize);
        let lab: Vec<[f32; 3]> = (0..w * h).map(|i| [if i % w < 6 { 0.2 } else { 0.8 }, 0.0, 0.0]).collect();
        let cut_at = |c: usize| -> Vec<u16> { (0..w * h).map(|i| (i % w >= c) as u16).collect() };
        assert_eq!(edge_agreement(&cut_at(6), &lab, w, h), (1.0, 0.0), "exact");
        assert_eq!(edge_agreement(&cut_at(5), &lab, w, h), (1.0, 0.0), "one pixel left");
        assert_eq!(edge_agreement(&cut_at(7), &lab, w, h), (1.0, 0.0), "one pixel right");
        assert_eq!(edge_agreement(&cut_at(8), &lab, w, h), (0.0, 1.0), "two pixels is another edge");
        assert_eq!(edge_agreement(&cut_at(4), &lab, w, h), (0.0, 1.0));
        // A boundary across the edge (between rows 1 and 2) is no evidence of keeping it.
        let crossing: Vec<u16> = (0..w * h).map(|i| (i / w >= 2) as u16).collect();
        assert_eq!(edge_agreement(&crossing, &lab, w, h), (0.0, 1.0));
        // No boundaries at all: nothing kept, nothing spurious.
        assert_eq!(edge_agreement(&vec![0; w * h], &lab, w, h), (0.0, 0.0));
    }

    #[test]
    #[ignore]
    fn potts_report() {
        use crate::indexed::{index_image, render};
        for name in ["wolf", "colliseum"] {
            let path = format!("{}/other/{name}.png", env!("CARGO_MANIFEST_DIR"));
            let source = image::open(&path).unwrap().to_rgba8();
            let (w, h) = (source.width() as usize, source.height() as usize);
            let config = Config { k_colors: 16, no_resize: true, ..Default::default() };
            let quantized = run_pipeline(&source, &config).unwrap().image;
            let lab: Vec<[f32; 3]> = source.pixels().map(|p| crate::srgb_to_oklab(p[0], p[1], p[2])).collect();
            let base = index_image(&source, &quantized).unwrap();
            let pal: Vec<[f32; 3]> = base.palette.iter().map(|c| crate::srgb_to_oklab(c[0], c[1], c[2])).collect();
            let fidelity = |labels: &[u16]| {
                labels.iter().zip(&lab).map(|(&l, c)| {
                    let p = pal[l as usize];
                    ((c[0] - p[0]).powi(2) + (c[1] - p[1]).powi(2) + (c[2] - p[2]).powi(2)).sqrt() as f64
                }).sum::<f64>() / labels.len() as f64
            };
            println!("\n### {name}.png, 16 colours\n");
            println!("| method | regions < 4 px | regions < 16 px | mean dE to source | strong edges kept (within 1 px) | boundaries with no edge within 1 px | seconds |");
            println!("|---|---|---|---|---|---|---|");
            let report = |label: String, labels: &[u16], secs: f64| {
                let mut ix = index_image(&source, &quantized).unwrap();
                ix.labels.copy_from_slice(labels);
                let rendered = render(&ix);
                // Set PSNAP_EVAL_OUT to a directory to keep the images for a look.
                if let Ok(dir) = std::env::var("PSNAP_EVAL_OUT") {
                    let file: String = label.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect();
                    rendered.save(format!("{dir}/{name}_{file}.png")).unwrap();
                }
                let stats = regions::image_stats(&rendered);
                let (kept, spurious) = edge_agreement(labels, &lab, w, h);
                println!("| {label} | {} | {} | {:.4} | {:.1} % | {:.1} % | {secs:.2} |", stats.regions_lt4, stats.regions_lt16, fidelity(labels), kept * 100.0, spurious * 100.0);
            };
            report("plain palette reduction".into(), &base.labels, 0.0);

            // The parameters shipping Smooth uses on this image: reach and thresholds
            // scaled by the detected pitch (standard fitter, as by default).
            let profiles = crate::grid::profiles(&source, &quantized, &config).unwrap();
            let grid = crate::grid::detect(&profiles, source.width(), source.height(), &config).unwrap();
            let pitch = cleanup::estimated_pitch(Some(&grid));
            let radius = cleanup::unmix_radius(pitch);
            println!("| (pitch {pitch:?}, un-mix reach {radius}) | | | | | | |");
            let nearest = base.labels.clone();
            let (lambda, sigma) = (0.01f32, 0.05f32);
            for strength in [Strength::Light, Strength::Medium, Strength::Strong] {
                let min_area = cleanup::sieve_min_area(strength, pitch);
                // Shipping Smooth.
                let t = std::time::Instant::now();
                let mut ix = index_image(&source, &quantized).unwrap();
                cleanup::unmix(&mut ix, &lab, &pal, radius);
                cleanup::sieve(&mut ix, &lab, &pal, min_area);
                report(format!("un-mix + sieve, {} (< {min_area:.0} px)", strength.name()), &ix.labels, t.elapsed().as_secs_f64());
                // The same sieve after Potts instead of un-mix.
                let t = std::time::Instant::now();
                let mut ix = index_image(&source, &quantized).unwrap();
                potts_icm(&mut ix.labels, &nearest, &lab, &pal, w, h, lambda, sigma, 20);
                cleanup::sieve(&mut ix, &lab, &pal, min_area);
                report(format!("Potts + sieve, {} (< {min_area:.0} px)", strength.name()), &ix.labels, t.elapsed().as_secs_f64());
                // And after both.
                let t = std::time::Instant::now();
                let mut ix = index_image(&source, &quantized).unwrap();
                cleanup::unmix(&mut ix, &lab, &pal, radius);
                potts_icm(&mut ix.labels, &nearest, &lab, &pal, w, h, lambda, sigma, 20);
                cleanup::sieve(&mut ix, &lab, &pal, min_area);
                report(format!("un-mix + Potts + sieve, {} (< {min_area:.0} px)", strength.name()), &ix.labels, t.elapsed().as_secs_f64());
            }
            // Potts on its own, over a range of strengths.
            for (lambda, sigma) in [(0.002f32, 0.05f32), (0.01, 0.05), (0.02, 0.05), (0.02, 0.1), (0.05, 0.1)] {
                let t = std::time::Instant::now();
                let mut labels = base.labels.clone();
                potts_icm(&mut labels, &nearest, &lab, &pal, w, h, lambda, sigma, 20);
                report(format!("Potts alone, lambda {lambda}, sigma {sigma}"), &labels, t.elapsed().as_secs_f64());
            }
        }
    }
}
