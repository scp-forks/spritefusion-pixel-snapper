//! The pixel grid of an image: where the cuts between logical pixels fall.

// Phase 1 adds these as plumbing; the first callers arrive in Phase 2.
#![allow(dead_code)]

use crate::{
    compute_profiles, estimate_step_size, resolve_step_sizes, sanitize_cuts, stabilize_both_axes,
    walk, Config, Result,
};
use image::RgbaImage;

/// No cell may be narrower than this, on either axis.
pub const MIN_CELL_PX: usize = 2;

/// A refined grid must have at least this many times the cells of the grid it refines,
/// otherwise it is not a distinct, finer rung.
const MIN_REFINEMENT_GAIN: f64 = 1.2;

#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    /// Column cuts, strictly increasing, from 0 to the image width.
    pub cols: Vec<usize>,
    /// Row cuts, strictly increasing, from 0 to the image height.
    pub rows: Vec<usize>,
    /// Step the cuts were walked with, per axis, in source pixels.
    pub step: (f64, f64),
    /// False when neither axis produced an estimate and `step` is only the
    /// `min(width, height) / fallback_target_segments` fallback. That number is not
    /// a pitch and must not be used as one.
    pub step_estimated: bool,
}

impl Grid {
    pub fn cell_count(&self) -> usize {
        (self.cols.len() - 1) * (self.rows.len() - 1)
    }
}

/// The grid detection of the legacy snap path, unchanged: gradient profiles of the
/// quantized image, median peak spacing, greedy walk, cross-axis stabilisation.
pub fn detect_grid(quantized: &RgbaImage, config: &Config) -> Result<Grid> {
    let (width, height) = quantized.dimensions();
    let (profile_x, profile_y) = compute_profiles(quantized)?;

    let step_x_opt = estimate_step_size(&profile_x, config);
    let step_y_opt = estimate_step_size(&profile_y, config);
    let step_estimated = step_x_opt.is_some() || step_y_opt.is_some();

    let (step_x, step_y) = resolve_step_sizes(step_x_opt, step_y_opt, width, height, config);

    let raw_col_cuts = walk(&profile_x, step_x, width as usize, config)?;
    let raw_row_cuts = walk(&profile_y, step_y, height as usize, config)?;

    let (cols, rows) = stabilize_both_axes(
        &profile_x,
        &profile_y,
        raw_col_cuts,
        raw_row_cuts,
        width as usize,
        height as usize,
        config,
    );

    Ok(Grid { cols, rows, step: (step_x, step_y), step_estimated })
}

/// A grid `scale` times finer than `grid`, or `None` when that is not achievable
/// (the result would not have at least 1.2x the cells).
///
/// No cell of the result is under [`MIN_CELL_PX`] on either axis, whatever the input
/// grid looked like: the legacy walker can emit 1 px cells, and those are merged away
/// here too. (The plan asked for a median-cell check; enforcing the minimum on every
/// cell is stronger and makes that check redundant.)
///
/// Integer scales subdivide each existing cell, so coarse cuts are kept wherever the
/// coarse cells were themselves wide enough. Other scales re-walk the profiles at
/// `step / scale`.
pub fn refine_grid(
    grid: &Grid,
    quantized: &RgbaImage,
    config: &Config,
    scale: f64,
) -> Option<Grid> {
    if !scale.is_finite() || scale <= 1.0 {
        return None;
    }

    let rounded = scale.round();
    let (cols, rows) = if (scale - rounded).abs() < 1e-9 {
        let factor = rounded as usize; // saturates for absurd scales; capped per cell below
        (
            merge_narrow_intervals(subdivide_cuts(&grid.cols, factor)),
            merge_narrow_intervals(subdivide_cuts(&grid.rows, factor)),
        )
    } else {
        let (profile_x, profile_y) = compute_profiles(quantized).ok()?;
        let rewalk = |profile: &[f64], step: f64, cuts: &[usize]| -> Option<Vec<usize>> {
            let limit = *cuts.last()?;
            let walked = walk(profile, (step / scale).max(1.0), limit, config).ok()?;
            Some(merge_narrow_intervals(sanitize_cuts(walked, limit)))
        };
        (
            rewalk(&profile_x, grid.step.0, &grid.cols)?,
            rewalk(&profile_y, grid.step.1, &grid.rows)?,
        )
    };

    let refined = Grid {
        cols,
        rows,
        step: (grid.step.0 / scale, grid.step.1 / scale),
        step_estimated: grid.step_estimated,
    };

    let finer = refined.cell_count() as f64 >= MIN_REFINEMENT_GAIN * grid.cell_count() as f64;
    finer.then_some(refined)
}

/// Split every interval into `factor` equal parts, or into as many as fit without
/// any part going under [`MIN_CELL_PX`].
fn subdivide_cuts(cuts: &[usize], factor: usize) -> Vec<usize> {
    // No capacity hint: `factor` is caller-controlled and may be enormous, while the
    // real number of parts is capped by each cell's width.
    let mut out = Vec::new();
    for pair in cuts.windows(2) {
        let (start, width) = (pair[0], pair[1] - pair[0]);
        let parts = factor.min(width / MIN_CELL_PX).max(1);
        for i in 0..parts {
            out.push(start + i * width / parts);
        }
    }
    out.push(*cuts.last().unwrap());
    out
}

/// Remove cuts until no interval is under [`MIN_CELL_PX`], merging each narrow
/// interval into its narrower neighbour. The first and last cuts are never removed.
fn merge_narrow_intervals(mut cuts: Vec<usize>) -> Vec<usize> {
    loop {
        if cuts.len() <= 2 {
            return cuts;
        }
        let Some(i) = cuts.windows(2).position(|p| p[1] - p[0] < MIN_CELL_PX) else {
            return cuts;
        };
        // Interval i spans cuts[i]..cuts[i + 1]. Its neighbours are i - 1 and i + 1.
        let left = (i > 0).then(|| cuts[i] - cuts[i - 1]);
        let right = (i + 2 < cuts.len()).then(|| cuts[i + 2] - cuts[i + 1]);
        let remove = match (left, right) {
            (Some(l), Some(r)) => if l <= r { i } else { i + 1 },
            (Some(_), None) => i,
            (None, Some(_)) => i + 1,
            (None, None) => return cuts,
        };
        cuts.remove(remove);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn cuts_from_widths(widths: &[usize]) -> Vec<usize> {
        let mut cuts = vec![0];
        for w in widths {
            cuts.push(cuts.last().unwrap() + w);
        }
        cuts
    }

    fn min_interval(cuts: &[usize]) -> usize {
        cuts.windows(2).map(|p| p[1] - p[0]).min().unwrap()
    }

    fn is_valid(cuts: &[usize], limit: usize) -> bool {
        cuts.first() == Some(&0)
            && cuts.last() == Some(&limit)
            && cuts.windows(2).all(|p| p[0] < p[1])
    }

    /// A checkerboard with cells of the given widths. With `soft_edges` the first
    /// pixel of every cell is a blend with the previous cell, as in upscaled art.
    fn checkerboard(widths: &[usize], soft_edges: bool) -> RgbaImage {
        let cuts = cuts_from_widths(widths);
        let size = *cuts.last().unwrap() as u32;
        let cell = |p: u32| cuts.iter().rposition(|&c| c as u32 <= p).unwrap();
        let on_cut = |p: u32| soft_edges && p > 0 && cuts.contains(&(p as usize));
        let mut img = RgbaImage::new(size, size);
        for (x, y, px) in img.enumerate_pixels_mut() {
            let v = if on_cut(x) || on_cut(y) {
                125
            } else if (cell(x) + cell(y)) % 2 == 0 {
                30
            } else {
                220
            };
            *px = Rgba([v, v, v, 255]);
        }
        img
    }

    fn striped(widths: &[usize]) -> RgbaImage {
        checkerboard(widths, true)
    }

    const IRREGULAR: [usize; 5] = [3, 9, 2, 10, 5];

    #[test]
    fn subdivision_never_goes_under_the_minimum_cell() {
        let cuts = cuts_from_widths(&IRREGULAR);
        for factor in [2, 3, 4, 7] {
            let fine = subdivide_cuts(&cuts, factor);
            assert!(is_valid(&fine, 29), "factor {factor}: {fine:?}");
            assert!(min_interval(&fine) >= MIN_CELL_PX, "factor {factor}: {fine:?}");
            // Every coarse cut survives.
            assert!(cuts.iter().all(|c| fine.contains(c)), "factor {factor}: {fine:?}");
        }
        // Widths 3 and 2 cannot be split at all; 9, 10 and 5 can.
        assert_eq!(subdivide_cuts(&cuts, 2), vec![0, 3, 7, 12, 14, 19, 24, 26, 29]);
    }

    #[test]
    fn merging_removes_slivers_including_at_the_borders() {
        assert_eq!(merge_narrow_intervals(vec![0, 1, 6, 12]), vec![0, 6, 12]);
        assert_eq!(merge_narrow_intervals(vec![0, 6, 11, 12]), vec![0, 6, 12]);
        // A sliver between a narrow and a wide cell joins the narrow one.
        assert_eq!(merge_narrow_intervals(vec![0, 3, 4, 12]), vec![0, 4, 12]);
        assert_eq!(merge_narrow_intervals(vec![0, 8, 9, 12]), vec![0, 8, 12]);
        // A run of slivers pairs up into cells of the minimum width.
        assert_eq!(merge_narrow_intervals(vec![0, 1, 2, 3, 4, 10]), vec![0, 2, 4, 10]);
        // An image narrower than the minimum keeps its two outer cuts.
        assert_eq!(merge_narrow_intervals(vec![0, 1]), vec![0, 1]);
    }

    #[test]
    fn refinement_at_integer_and_fractional_scales_respects_the_minimum() {
        let img = striped(&[3, 9, 2, 10, 5, 8, 8, 7, 9, 8]);
        let limit = img.width() as usize;
        let grid = Grid {
            cols: cuts_from_widths(&[3, 9, 2, 10, 5, 8, 8, 7, 9, 8]),
            rows: cuts_from_widths(&[3, 9, 2, 10, 5, 8, 8, 7, 9, 8]),
            step: (7.0, 7.0),
            step_estimated: true,
        };
        let config = Config::default();
        for scale in [1.5, 2.0, 3.0] {
            let fine = refine_grid(&grid, &img, &config, scale)
                .unwrap_or_else(|| panic!("scale {scale} should be achievable"));
            assert!(is_valid(&fine.cols, limit) && is_valid(&fine.rows, limit), "scale {scale}");
            assert!(min_interval(&fine.cols) >= MIN_CELL_PX, "scale {scale}: {:?}", fine.cols);
            assert!(min_interval(&fine.rows) >= MIN_CELL_PX, "scale {scale}: {:?}", fine.rows);
            assert!(fine.cell_count() as f64 >= 1.2 * grid.cell_count() as f64, "scale {scale}");
            assert!(fine.step_estimated);
        }
    }

    /// The legacy walker can hand over grids that already contain 1 px cells, at the
    /// borders or inside. Refinement must not pass them on.
    #[test]
    fn refinement_removes_existing_one_pixel_cells() {
        let img = RgbaImage::from_pixel(24, 24, Rgba([0, 0, 0, 255]));
        let config = Config::default();
        for widths in [
            vec![1usize, 7, 8, 8],  // first
            vec![8, 7, 1, 8],       // interior
            vec![8, 8, 7, 1],       // last
            vec![1, 6, 1, 7, 8, 1], // several
        ] {
            let grid = Grid {
                cols: cuts_from_widths(&widths),
                rows: cuts_from_widths(&[8, 8, 8]),
                step: (8.0, 8.0),
                step_estimated: true,
            };
            for scale in [1.5, 2.0, 3.0] {
                let fine = refine_grid(&grid, &img, &config, scale)
                    .unwrap_or_else(|| panic!("{widths:?} at scale {scale}"));
                assert!(is_valid(&fine.cols, 24) && is_valid(&fine.rows, 24));
                assert!(min_interval(&fine.cols) >= MIN_CELL_PX, "{widths:?} x{scale}: {:?}", fine.cols);
                assert!(min_interval(&fine.rows) >= MIN_CELL_PX, "{widths:?} x{scale}: {:?}", fine.rows);
            }
        }
        // Codex's example from the Phase 1 review.
        let grid = Grid { cols: vec![0, 1, 5], rows: vec![0, 4], step: (2.5, 4.0), step_estimated: true };
        let img = RgbaImage::from_pixel(5, 4, Rgba([0, 0, 0, 255]));
        let fine = refine_grid(&grid, &img, &config, 2.0).unwrap();
        assert!(min_interval(&fine.cols) >= MIN_CELL_PX, "{:?}", fine.cols);
    }

    /// The acceptance rule is "at least 1.2x the cells", exactly.
    #[test]
    fn refinement_needs_twenty_percent_more_cells() {
        let config = Config::default();
        let img = RgbaImage::from_pixel(32, 3, Rgba([0, 0, 0, 255]));
        let rows = vec![0, 3]; // one row that cannot be split
        let grid_with = |widths: &[usize]| Grid {
            cols: cuts_from_widths(widths),
            rows: rows.clone(),
            step: (3.0, 3.0),
            step_estimated: true,
        };
        // Ten cells, of which only the 4 px ones can be halved.
        let eleven = grid_with(&[3, 3, 3, 3, 3, 3, 3, 3, 3, 4]); // 10 -> 11 cells
        let twelve = grid_with(&[3, 3, 3, 3, 3, 3, 3, 3, 4, 4]); // 10 -> 12 cells
        assert_eq!(subdivide_cuts(&eleven.cols, 2).len() - 1, 11);
        assert_eq!(refine_grid(&eleven, &img, &config, 2.0), None);
        assert_eq!(refine_grid(&twelve, &img, &config, 2.0).unwrap().cell_count(), 12);
    }

    /// A scale far beyond anything meaningful must neither panic nor allocate wildly.
    #[test]
    fn absurd_scales_are_capped_by_cell_width() {
        let config = Config::default();
        let img = RgbaImage::from_pixel(8, 8, Rgba([0, 0, 0, 255]));
        let grid = Grid { cols: vec![0, 4, 8], rows: vec![0, 4, 8], step: (4.0, 4.0), step_estimated: true };
        for scale in [1e6, 1e20, f64::MAX] {
            let fine = refine_grid(&grid, &img, &config, scale).unwrap();
            assert_eq!(fine.cols, vec![0, 2, 4, 6, 8], "scale {scale}");
            assert_eq!(fine.rows, vec![0, 2, 4, 6, 8], "scale {scale}");
        }
    }

    #[test]
    fn refinement_gives_up_when_it_cannot_get_finer() {
        // 2 px cells cannot be split, by any route.
        let widths = [2usize; 12];
        let img = striped(&widths);
        let grid = Grid {
            cols: cuts_from_widths(&widths),
            rows: cuts_from_widths(&widths),
            step: (2.0, 2.0),
            step_estimated: true,
        };
        let config = Config::default();
        for scale in [1.5, 2.0, 3.0] {
            assert_eq!(refine_grid(&grid, &img, &config, scale), None, "scale {scale}");
        }
        // Scales that are not refinements at all.
        for scale in [1.0, 0.5, f64::NAN, f64::INFINITY] {
            assert_eq!(refine_grid(&grid, &img, &config, scale), None, "scale {scale}");
        }
    }

    #[test]
    fn detection_reports_whether_the_step_was_estimated() {
        let config = Config::default();
        let lattice = striped(&[8; 12]);
        let grid = detect_grid(&lattice, &config).unwrap();
        assert!(grid.step_estimated);
        assert!(is_valid(&grid.cols, 96) && is_valid(&grid.rows, 96));
        assert_eq!(grid.cols.len() - 1, 12);

        let flat = RgbaImage::from_pixel(96, 96, Rgba([40, 40, 40, 255]));
        let grid = detect_grid(&flat, &config).unwrap();
        assert!(!grid.step_estimated, "a flat image has no pitch to estimate");
        assert!(is_valid(&grid.cols, 96) && is_valid(&grid.rows, 96));
    }

    /// Documents a limitation of the legacy estimator, found while writing these
    /// tests. `estimate_step_size` wants a strict local maximum, and a perfectly
    /// sharp edge puts two equal samples in the gradient profile, so crisp pixel art
    /// (nearest-neighbour upscaled) yields no estimate at all and falls back to
    /// `min(w, h) / 64`. The legacy path is frozen, so this is recorded, not fixed here.
    #[test]
    fn crisp_edges_defeat_the_legacy_step_estimate() {
        let crisp = checkerboard(&[8; 12], false);
        let grid = detect_grid(&crisp, &Config::default()).unwrap();
        assert!(!grid.step_estimated);
    }
}
