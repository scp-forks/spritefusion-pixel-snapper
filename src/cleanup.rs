//! Spatial cleanup of a palette-reduced image at its original size.
//!
//! Every quantizer ends by giving each pixel a palette colour on its own. On blurry
//! "pixel art" that leaves speckle: blends between two logical pixels land on some
//! third palette colour. The stages here put the missing spatial step back.

use crate::grid::{detect_grid, refine_grid, Grid};
use crate::indexed::{index_image, render, IndexedImage, TRANSPARENT};
use crate::{Config, Result};
use image::RgbaImage;

/// Grid scales tried, finest first, when the requested one cannot be honoured.
const FALLBACK_SCALES: [f64; 3] = [3.0, 2.0, 1.5];

/// What a cleanup stage produced, and on which grid.
pub struct Cleaned {
    pub image: RgbaImage,
    /// The grid the cells were voted on. `None` when the image was passed through.
    pub grid: Option<Grid>,
}

/// The detected grid refined by `scale`, or by the nearest smaller standard scale
/// that is achievable, or `base` itself. Refined grids never have a cell under 2 px;
/// `base` is returned as given, so pass `Grid::with_min_cells` if that matters.
pub fn grid_at_scale(base: Grid, quantized: &RgbaImage, config: &Config, scale: f64) -> Grid {
    if scale <= 1.0 {
        return base;
    }
    std::iter::once(scale)
        .chain(FALLBACK_SCALES.into_iter().filter(|&s| s < scale))
        .find_map(|s| refine_grid(&base, quantized, config, s))
        .unwrap_or(base)
}

/// **Blocks** ("snap in place"): one colour per grid cell, painted back at the
/// original size.
///
/// * Colour: the majority label among the cell's pixels with `alpha > 0`; ties go to
///   the label that appears first in the image.
/// * Alpha: every pixel keeps its own source alpha. Fully transparent pixels are copied
///   from the source, so an opaque feature inside a mostly transparent cell keeps its
///   shape and takes the cell's colour.
pub fn blocks(source: &RgbaImage, quantized: &RgbaImage, config: &Config) -> Result<Cleaned> {
    let mut ix = index_image(source, quantized)?;
    // A grid needs at least 3 px on each side; smaller images have nothing to snap.
    if ix.palette.is_empty() || source.width() < 3 || source.height() < 3 {
        return Ok(Cleaned { image: render(&ix), grid: None });
    }

    // The detected grid can contain 1 px cells. A cell that small cannot hold a vote
    // (it would keep the very speckle this is meant to remove), so they are merged
    // away before anything else. `grid_at_scale` keeps the guarantee: every refinement
    // enforces it, and its last resort is this base grid.
    let base = detect_grid(quantized, config)?.with_min_cells();
    let grid = grid_at_scale(base, quantized, config, config.grid_scale);

    let winners = vote_cells(&ix, &grid);
    paint_cells(&mut ix, &grid, &winners);
    Ok(Cleaned { image: render(&ix), grid: Some(grid) })
}

/// Majority label per cell (row-major over cells), [`TRANSPARENT`] for a cell with no
/// opaque pixel.
pub fn vote_cells(ix: &IndexedImage, grid: &Grid) -> Vec<u16> {
    let w = ix.width();
    let mut counts = vec![0u32; ix.palette.len()];
    let mut seen: Vec<u16> = Vec::new();
    let mut winners = Vec::with_capacity(grid.cell_count());

    for row in grid.rows.windows(2) {
        for col in grid.cols.windows(2) {
            for y in row[0]..row[1] {
                for &label in &ix.labels[y * w + col[0]..y * w + col[1]] {
                    if label == TRANSPARENT {
                        continue;
                    }
                    if counts[label as usize] == 0 {
                        seen.push(label);
                    }
                    counts[label as usize] += 1;
                }
            }
            // Highest count wins; among equals, the lowest label (first to appear in
            // the image), so the result does not depend on scan order within the cell.
            let winner = seen
                .iter()
                .copied()
                .max_by(|&a, &b| counts[a as usize].cmp(&counts[b as usize]).then(b.cmp(&a)))
                .unwrap_or(TRANSPARENT);
            winners.push(winner);
            for label in seen.drain(..) {
                counts[label as usize] = 0;
            }
        }
    }
    winners
}

/// Give every opaque pixel the label of its cell.
pub fn paint_cells(ix: &mut IndexedImage, grid: &Grid, winners: &[u16]) {
    let w = ix.width();
    let cells_per_row = grid.cols.len() - 1;
    for (cy, row) in grid.rows.windows(2).enumerate() {
        for (cx, col) in grid.cols.windows(2).enumerate() {
            let winner = winners[cy * cells_per_row + cx];
            if winner == TRANSPARENT {
                continue;
            }
            for y in row[0]..row[1] {
                for label in &mut ix.labels[y * w + col[0]..y * w + col[1]] {
                    if *label != TRANSPARENT {
                        *label = winner;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{recovered_fraction, recovered_pixel_fraction, synthetic_lattice, DISTINCT_8};
    use crate::{regions, run_pipeline, Cleanup};
    use image::Rgba;
    use rand::prelude::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn blocks_config(k: usize, grid_scale: f64) -> Config {
        Config { k_colors: k, no_resize: true, cleanup: Cleanup::Blocks, grid_scale, ..Default::default() }
    }

    fn ix_from<'a>(source: &'a RgbaImage) -> IndexedImage<'a> {
        index_image(source, source).unwrap()
    }

    fn whole_image_grid(w: usize, h: usize, cols: Vec<usize>, rows: Vec<usize>) -> Grid {
        assert_eq!((*cols.last().unwrap(), *rows.last().unwrap()), (w, h));
        Grid { cols, rows, step: (1.0, 1.0), step_estimated: true }
    }

    #[test]
    fn vote_is_the_majority_and_ties_go_to_the_first_colour_seen() {
        let (a, b, c) = (Rgba([10, 0, 0, 255]), Rgba([0, 20, 0, 255]), Rgba([0, 0, 30, 255]));
        // Two cells of 3x2. Left: a a b / b b a -> tie 3:3, a appeared first.
        // Right: c b b / b c c -> tie 3:3 between b (label 1) and c (label 2): b.
        let mut img = RgbaImage::new(6, 2);
        for (i, px) in [a, a, b, c, b, b, b, b, a, b, c, c].into_iter().enumerate() {
            img.put_pixel(i as u32 % 6, i as u32 / 6, px);
        }
        let ix = ix_from(&img);
        let grid = whole_image_grid(6, 2, vec![0, 3, 6], vec![0, 2]);
        assert_eq!(vote_cells(&ix, &grid), vec![0, 1]);

        // A clear majority beats palette order.
        img.put_pixel(0, 0, b);
        let ix = ix_from(&img);
        assert_eq!(ix.palette[0], [0, 20, 0]);
        assert_eq!(vote_cells(&ix, &grid)[0], 0);
        img.put_pixel(3, 0, Rgba([0, 0, 30, 255]));
        img.put_pixel(4, 0, Rgba([0, 0, 30, 255]));
        let ix = ix_from(&img);
        assert_eq!(ix.palette[vote_cells(&ix, &grid)[1] as usize], [0, 0, 30]);
    }

    #[test]
    fn transparent_pixels_do_not_vote_and_are_not_painted() {
        let hidden = Rgba([9, 9, 9, 0]);
        let (a, b) = (Rgba([200, 0, 0, 255]), Rgba([0, 0, 200, 255]));
        // One 4x1 cell: three transparent pixels and one opaque `b`; then a cell of `a`.
        let mut img = RgbaImage::new(8, 1);
        for (x, px) in [hidden, hidden, b, hidden, a, a, a, a].into_iter().enumerate() {
            img.put_pixel(x as u32, 0, px);
        }
        let mut ix = ix_from(&img);
        let grid = whole_image_grid(8, 1, vec![0, 4, 8], vec![0, 1]);
        let winners = vote_cells(&ix, &grid);
        assert_eq!(ix.palette[winners[0] as usize], [0, 0, 200]);
        paint_cells(&mut ix, &grid, &winners);
        assert_eq!(render(&ix), img, "a lone opaque pixel keeps its place and colour");

        // A cell with nothing opaque has no winner and is left alone.
        let empty = RgbaImage::from_pixel(4, 1, hidden);
        let mut ix = ix_from(&empty);
        let grid = whole_image_grid(4, 1, vec![0, 4], vec![0, 1]);
        let winners = vote_cells(&ix, &grid);
        assert_eq!(winners, vec![TRANSPARENT]);
        paint_cells(&mut ix, &grid, &winners);
        assert_eq!(render(&ix), empty);
    }

    // The lattice tests ask for 12 colours although the lattice has 8. A blurred image
    // also contains every blend between neighbouring cells, and with exactly 8 slots
    // k-means spends some of them on blends and merges two of the real colours, which
    // no spatial step can undo. (Blocks then uses 8 or 9 of the 12.)
    const LATTICE_K: usize = 12;

    #[test]
    fn regular_blurred_lattice_is_recovered() {
        let (cols, rows) = ([8u32; 24], [8u32; 24]);
        let (img, truth) = synthetic_lattice(&cols, &rows, &DISTINCT_8, true, 4, 21);
        let out = run_pipeline(&img, &blocks_config(LATTICE_K, 1.0)).unwrap();
        assert_eq!(out.image.dimensions(), img.dimensions());

        let recovered = recovered_fraction(&out.image, &cols, &rows, &truth);
        assert!(recovered >= 0.99, "recovered {:.1}% of cells", recovered * 100.0);

        // And the speckle is gone: nothing under 4 px, where the plain reduction has plenty.
        let plain = run_pipeline(&img, &Config { cleanup: Cleanup::Off, ..blocks_config(LATTICE_K, 1.0) }).unwrap();
        let (before, after) = (regions::image_stats(&plain.image), regions::image_stats(&out.image));
        assert!(before.regions_lt16 > 50, "the test image should speckle, got {}", before.regions_lt16);
        assert_eq!(after.regions_lt4, 0);
        assert!(after.unique_colors <= LATTICE_K);

        // Pixel for pixel. Measured when written: 75 % of pixels right for the plain
        // reduction, 90 % for Blocks. The missing 10 % is the legacy grid placing about
        // four cuts in ten one pixel early (its gradient is centred on a pixel, an edge
        // lies between two); a better fitter should raise this, never lower it.
        let exact = recovered_pixel_fraction(&out.image, &cols, &rows, &truth);
        let exact_plain = recovered_pixel_fraction(&plain.image, &cols, &rows, &truth);
        assert!(exact >= 0.88, "{:.1}% of pixels right", exact * 100.0);
        assert!(exact >= exact_plain + 0.10, "Blocks {exact:.3} against plain {exact_plain:.3}");
    }

    #[test]
    fn irregular_blurred_lattice_is_recovered() {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let cols: Vec<u32> = (0..24).map(|_| rng.gen_range(6..=11)).collect();
        let rows: Vec<u32> = (0..24).map(|_| rng.gen_range(6..=11)).collect();
        let (img, truth) = synthetic_lattice(&cols, &rows, &DISTINCT_8, true, 4, 22);
        let out = run_pipeline(&img, &blocks_config(LATTICE_K, 1.0)).unwrap();
        let recovered = recovered_fraction(&out.image, &cols, &rows, &truth);
        assert!(recovered >= 0.95, "recovered {:.1}% of cells", recovered * 100.0);
    }

    /// One-pixel checkerboard: the walker's grid is full of 1 px cells, and a 1 px cell
    /// votes for itself. Blocks must not hand the speckle back (Codex, Phase 2 review).
    #[test]
    fn one_pixel_cells_are_merged_before_voting() {
        let mut img = RgbaImage::new(32, 24);
        for (x, y, p) in img.enumerate_pixels_mut() {
            let v = if (x + y) % 2 == 0 { 20 } else { 230 };
            *p = Rgba([v, v, v, 255]);
        }
        assert_eq!(regions::image_stats(&img).regions_lt4, 32 * 24);
        for scale in [1.0, 2.0, 3.0] {
            let out = run_pipeline(&img, &blocks_config(2, scale)).unwrap();
            let grid = out.grid.unwrap();
            let narrowest = |cuts: &[usize]| cuts.windows(2).map(|p| p[1] - p[0]).min().unwrap();
            assert!(narrowest(&grid.cols) >= 2 && narrowest(&grid.rows) >= 2, "scale {scale}: {grid:?}");
            assert_eq!(regions::image_stats(&out.image).regions_lt4, 0, "scale {scale}");
        }
    }

    /// Votes are counted per pixel, whatever the pixel's alpha, as long as it is not 0.
    /// Two faint pixels of A outvote one solid pixel of B. Checked on an irregular,
    /// multi-row grid with transparent holes, against the complete expected output.
    #[test]
    fn partial_alpha_pixels_vote_with_full_weight() {
        let a = |alpha| Rgba([200, 30, 30, alpha]);
        let b = |alpha| Rgba([30, 30, 200, alpha]);
        let c = |alpha| Rgba([30, 180, 60, alpha]);
        let hole = Rgba([7, 7, 7, 0]);
        // Columns cut at 3 and 5 (widths 3, 2, 2); rows cut at 1 and 3 (heights 1, 2).
        #[rustfmt::skip]
        let src = [
            a(1),   a(1),   b(255),   hole,   c(255),   b(9),  b(9),
            hole,   c(255), c(40),    a(255), hole,     hole,  hole,
            b(255), b(255), c(255),   hole,   a(3),     c(80), hole,
        ];
        #[rustfmt::skip]
        let want = [
            a(1),   a(1),   a(255),   hole,   c(255),   b(9),  b(9),   // A 2:1; C alone; B 2:0
            hole,   c(255), c(40),    a(255), hole,     hole,  hole,   // C 3:2 over B; A 2:0; C alone
            c(255), c(255), c(255),   hole,   a(3),     c(80), hole,
        ];
        let mut img = RgbaImage::new(7, 3);
        let mut expected = RgbaImage::new(7, 3);
        for i in 0..21u32 {
            img.put_pixel(i % 7, i / 7, src[i as usize]);
            expected.put_pixel(i % 7, i / 7, want[i as usize]);
        }
        let mut ix = ix_from(&img);
        let grid = whole_image_grid(7, 3, vec![0, 3, 5, 7], vec![0, 1, 3]);
        let winners = vote_cells(&ix, &grid);
        paint_cells(&mut ix, &grid, &winners);
        assert_eq!(render(&ix), expected);
    }

    /// A lattice that the coarse grid explains exactly comes out identical, pixel for
    /// pixel, on the coarse grid and on every subdivision of it.
    #[test]
    fn an_exact_lattice_is_unchanged_on_its_grid_and_on_finer_ones() {
        let (cols, rows) = (vec![6u32, 9, 7, 11, 8], vec![8u32, 6, 10, 7]);
        let (img, _) = synthetic_lattice(&cols, &rows, &DISTINCT_8, false, 0, 13);
        let cuts = |sizes: &[u32]| {
            let mut out = vec![0usize];
            for s in sizes {
                out.push(out.last().unwrap() + *s as usize);
            }
            out
        };
        let base = Grid { cols: cuts(&cols), rows: cuts(&rows), step: (8.0, 8.0), step_estimated: true };
        let config = Config::default();
        for scale in [1.0, 2.0, 3.0] {
            let grid = grid_at_scale(base.clone(), &img, &config, scale);
            if scale > 1.0 {
                assert!(grid.cell_count() > 3 * base.cell_count(), "scale {scale}");
            }
            let mut ix = ix_from(&img);
            let winners = vote_cells(&ix, &grid);
            paint_cells(&mut ix, &grid, &winners);
            assert_eq!(render(&ix), img, "scale {scale}");
        }
    }

    /// On a blurred, noisy lattice a finer grid still gives every cell its colour.
    /// (Centre pixels only: the estimator's one-pixel cut error is measured elsewhere.)
    #[test]
    fn doubling_the_grid_recovers_the_same_cell_colours() {
        let (cols, rows) = ([8u32; 24], [8u32; 24]);
        let (img, truth) = synthetic_lattice(&cols, &rows, &DISTINCT_8, true, 4, 21);
        let coarse = run_pipeline(&img, &blocks_config(LATTICE_K, 1.0)).unwrap();
        let fine = run_pipeline(&img, &blocks_config(LATTICE_K, 2.0)).unwrap();
        let (cg, fg) = (coarse.grid.unwrap(), fine.grid.unwrap());
        assert!(fg.cell_count() as f64 >= 3.5 * cg.cell_count() as f64,
            "{} cells against {}", fg.cell_count(), cg.cell_count());
        let recovered = recovered_fraction(&fine.image, &cols, &rows, &truth);
        assert!(recovered >= 0.99, "recovered {:.1}% of cells", recovered * 100.0);
    }

    #[test]
    fn output_is_deterministic_and_within_the_palette() {
        let (img, _) = synthetic_lattice(&[7; 20], &[7; 20], &DISTINCT_8, true, 5, 4);
        for scale in [1.0, 1.5, 2.0] {
            let a = run_pipeline(&img, &blocks_config(6, scale)).unwrap().image;
            let b = run_pipeline(&img, &blocks_config(6, scale)).unwrap().image;
            assert_eq!(a, b, "scale {scale}");
            assert!(regions::image_stats(&a).unique_colors <= 6, "scale {scale}");
        }
    }

    /// A lattice with a transparent surround, a soft (partial-alpha) rim and a 1 px
    /// opaque line running out through the transparent area.
    fn sprite_with_alpha() -> RgbaImage {
        let (lattice, _) = synthetic_lattice(&[8; 10], &[8; 10], &DISTINCT_8, true, 3, 5);
        let mut img = RgbaImage::from_pixel(128, 128, Rgba([255, 0, 255, 0]));
        for (x, y, p) in lattice.enumerate_pixels() {
            let edge = x.min(y).min(79 - x).min(79 - y);
            let alpha = match edge { 0 => 70, 1 => 160, _ => 255 };
            img.put_pixel(x + 24, y + 24, Rgba([p[0], p[1], p[2], alpha]));
        }
        for x in 4..24 {
            img.put_pixel(x, 61, Rgba([20, 20, 24, 255])); // the 1 px line
        }
        img
    }

    #[test]
    fn alpha_is_preserved_pixel_for_pixel() {
        let img = sprite_with_alpha();
        let out = run_pipeline(&img, &blocks_config(8, 1.0)).unwrap().image;
        assert!(out.pixels().any(|p| p[3] > 0 && p[3] < 255), "test image must have partial alpha");
        for (src, dst) in img.pixels().zip(out.pixels()) {
            assert_eq!(src[3], dst[3], "alpha changed");
            if src[3] == 0 {
                assert_eq!(src, dst, "a transparent pixel's hidden colour changed");
            }
        }
    }

    #[test]
    fn a_thin_opaque_line_in_transparent_cells_survives() {
        let img = sprite_with_alpha();
        let out = run_pipeline(&img, &blocks_config(8, 1.0)).unwrap().image;
        let line_colour = out.get_pixel(4, 61).0;
        assert!(line_colour[0] < 60 && line_colour[3] == 255, "line colour {line_colour:?}");
        for x in 4..24 {
            // Still there, still one pixel thick, along its whole length.
            assert_eq!(out.get_pixel(x, 61)[3], 255, "line broken at x={x}");
            assert_eq!(out.get_pixel(x, 60)[3], 0, "line thickened at x={x}");
            assert_eq!(out.get_pixel(x, 62)[3], 0, "line thickened at x={x}");
            // Its colour is the cell's vote. Cells that hold nothing but the line vote
            // for the line; the last cell or two also hold the sprite's edge and may not.
            if x < 14 {
                assert_eq!(out.get_pixel(x, 61).0, line_colour, "line recoloured at x={x}");
            }
        }
    }

    #[test]
    fn nothing_to_snap_passes_through() {
        let config = blocks_config(4, 1.0);
        let clear = RgbaImage::from_pixel(40, 40, Rgba([3, 2, 1, 0]));
        let out = run_pipeline(&clear, &config).unwrap();
        assert_eq!(out.image, clear);
        assert!(out.grid.is_none());

        // Too small for grid detection: palette reduction only, no error.
        for (w, h) in [(1, 1), (2, 9), (9, 2)] {
            let tiny = RgbaImage::from_pixel(w, h, Rgba([90, 10, 10, 255]));
            let out = run_pipeline(&tiny, &config).unwrap();
            assert_eq!(out.image.dimensions(), (w, h));
            assert!(out.grid.is_none());
        }
    }

    #[test]
    fn unachievable_scale_falls_back_to_a_coarser_one() {
        let config = Config::default();
        let img = RgbaImage::from_pixel(30, 30, Rgba([0, 0, 0, 255]));
        // 5 px cells: x3 would need 1 px parts (only x2 fits), so x3 becomes x2.
        let cuts: Vec<usize> = (0..=6).map(|i| i * 5).collect();
        let base = Grid { cols: cuts.clone(), rows: cuts.clone(), step: (5.0, 5.0), step_estimated: true };
        let at3 = grid_at_scale(base.clone(), &img, &config, 3.0);
        let at2 = grid_at_scale(base.clone(), &img, &config, 2.0);
        assert_eq!(at3.cols, at2.cols);
        assert_eq!(at2.cols.len() - 1, 12);
        // 2 px cells cannot be refined at all (every `refine_grid` call returns None):
        // the base grid comes back.
        let cuts: Vec<usize> = (0..=15).map(|i| i * 2).collect();
        let base = Grid { cols: cuts.clone(), rows: cuts, step: (2.0, 2.0), step_estimated: true };
        for scale in [1.5, 2.0, 3.0] {
            assert_eq!(refine_grid(&base, &img, &config, scale), None, "scale {scale}");
        }
        assert_eq!(grid_at_scale(base.clone(), &img, &config, 3.0), base);
        assert_eq!(grid_at_scale(base.clone(), &img, &config, 1.0), base);
    }
}
