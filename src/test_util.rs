//! Helpers shared by the test modules.

use image::{Rgba, RgbaImage};
use rand::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Deterministic noise in [-amp, amp] for test images.
pub fn jitter(rng: &mut ChaCha8Rng, amp: i32) -> i32 {
    rng.gen_range(-amp..=amp)
}


/// Cells of random palette colours with the given column widths and row heights,
/// optionally box-blurred (3x3) and noised: a stand-in for AI "pixel art".
/// Returns the image and the palette index of every cell, row-major.
pub fn synthetic_lattice(
    col_widths: &[u32],
    row_heights: &[u32],
    palette: &[[u8; 3]],
    blur: bool,
    noise: i32,
    seed: u64,
) -> (RgbaImage, Vec<usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let (w, h) = (col_widths.iter().sum::<u32>(), row_heights.iter().sum::<u32>());
    let truth: Vec<usize> = (0..col_widths.len() * row_heights.len())
        .map(|_| rng.gen_range(0..palette.len()))
        .collect();

    let cell_of = |sizes: &[u32], pos: u32| -> usize {
        let mut end = 0;
        sizes.iter().position(|&s| { end += s; pos < end }).unwrap()
    };
    let mut sharp = vec![[0f32; 3]; (w * h) as usize];
    for y in 0..h {
        let cy = cell_of(row_heights, y);
        for x in 0..w {
            let c = palette[truth[cy * col_widths.len() + cell_of(col_widths, x)]];
            sharp[(y * w + x) as usize] = [c[0] as f32, c[1] as f32, c[2] as f32];
        }
    }

    let mut img = RgbaImage::new(w, h);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut acc = [0f32; 3];
            let mut n = 0f32;
            let r = if blur { 1 } else { 0 };
            for dy in -r..=r {
                for dx in -r..=r {
                    let (sx, sy) = (x + dx, y + dy);
                    if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                        let s = sharp[(sy as u32 * w + sx as u32) as usize];
                        for ch in 0..3 { acc[ch] += s[ch]; }
                        n += 1.0;
                    }
                }
            }
            let mut px = [0u8, 0, 0, 255];
            for ch in 0..3 {
                let v = (acc[ch] / n).round() as i32 + if noise > 0 { jitter(&mut rng, noise) } else { 0 };
                px[ch] = v.clamp(0, 255) as u8;
            }
            img.put_pixel(x as u32, y as u32, Rgba(px));
        }
    }
    (img, truth)
}

/// How well `out` reproduces the cell map of a lattice from [`synthetic_lattice`]:
/// the share of true cells whose centre pixel has the colour that their palette index
/// most often received. 1.0 means every cell came out as the right colour.
pub fn recovered_fraction(
    out: &RgbaImage,
    col_widths: &[u32],
    row_heights: &[u32],
    truth: &[usize],
) -> f64 {
    use std::collections::HashMap;
    let centres = |sizes: &[u32]| -> Vec<u32> {
        let mut start = 0;
        sizes.iter().map(|&s| { let c = start + s / 2; start += s; c }).collect()
    };
    let (xs, ys) = (centres(col_widths), centres(row_heights));
    let colour_at = |cx: usize, cy: usize| out.get_pixel(xs[cx], ys[cy]).0;

    let mut votes: HashMap<usize, HashMap<[u8; 4], usize>> = HashMap::new();
    for cy in 0..ys.len() {
        for cx in 0..xs.len() {
            *votes.entry(truth[cy * xs.len() + cx]).or_default().entry(colour_at(cx, cy)).or_default() += 1;
        }
    }
    let expected: HashMap<usize, [u8; 4]> = votes
        .into_iter()
        .map(|(idx, v)| (idx, v.into_iter().max_by_key(|&(c, n)| (n, c)).unwrap().0))
        .collect();
    // Two different true colours collapsing onto one output colour is not recovery.
    let mut distinct: Vec<[u8; 4]> = expected.values().copied().collect();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != expected.len() {
        return 0.0;
    }

    let mut right = 0;
    for cy in 0..ys.len() {
        for cx in 0..xs.len() {
            if colour_at(cx, cy) == expected[&truth[cy * xs.len() + cx]] {
                right += 1;
            }
        }
    }
    right as f64 / truth.len() as f64
}

/// Eight colours far apart in OKLab, for lattices whose cells must stay distinguishable.
pub const DISTINCT_8: [[u8; 3]; 8] = [
    [20, 20, 24], [235, 232, 225], [200, 50, 45], [50, 160, 70],
    [45, 80, 190], [240, 200, 40], [150, 60, 170], [120, 120, 125],
];

/// Stricter than [`recovered_fraction`]: the share of **pixels** that carry the colour
/// expected for their true cell (the colour found at that cell's centre must be the
/// consistent one for its palette index, and every pixel of the cell must match it).
/// A grid that is off by one pixel loses a row or column of every cell here.
pub fn recovered_pixel_fraction(
    out: &RgbaImage,
    col_widths: &[u32],
    row_heights: &[u32],
    truth: &[usize],
) -> f64 {
    use std::collections::HashMap;
    let starts = |sizes: &[u32]| -> Vec<u32> {
        let mut start = 0;
        sizes.iter().map(|&s| { let c = start; start += s; c }).collect()
    };
    let (xs, ys) = (starts(col_widths), starts(row_heights));
    let mut votes: HashMap<usize, HashMap<[u8; 4], usize>> = HashMap::new();
    for cy in 0..ys.len() {
        for cx in 0..xs.len() {
            let c = out.get_pixel(xs[cx] + col_widths[cx] / 2, ys[cy] + row_heights[cy] / 2).0;
            *votes.entry(truth[cy * xs.len() + cx]).or_default().entry(c).or_default() += 1;
        }
    }
    let expected: HashMap<usize, [u8; 4]> = votes
        .into_iter()
        .map(|(idx, v)| (idx, v.into_iter().max_by_key(|&(c, n)| (n, c)).unwrap().0))
        .collect();
    let mut right = 0u64;
    for cy in 0..ys.len() {
        for cx in 0..xs.len() {
            let want = expected[&truth[cy * xs.len() + cx]];
            for y in ys[cy]..ys[cy] + row_heights[cy] {
                for x in xs[cx]..xs[cx] + col_widths[cx] {
                    if out.get_pixel(x, y).0 == want {
                        right += 1;
                    }
                }
            }
        }
    }
    right as f64 / (out.width() as u64 * out.height() as u64) as f64
}
