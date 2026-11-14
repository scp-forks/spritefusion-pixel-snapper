//! Several results at different grid sizes from one image, for the user to audition.
//!
//! No measurable property of an image picks the right size: on the test images the
//! estimate was right once, a step too coarse once, 2.6 times too coarse once, and
//! meaningless once. So the engine offers a ladder and a person chooses.

use crate::cleanup::{self, grid_at_scale};
use crate::grid::{self, Grid, GridFitter, LabImage, Profiles, MIN_CELL_PX};
use crate::{quantize_image, quantize_no_resize, resample, Cleanup, Config, PixelSnapperError, Result};
use image::RgbaImage;

/// Grid scales offered, relative to the detected grid.
pub const LADDER_SCALES: [f64; 4] = [1.0, 1.5, 2.0, 3.0];

/// A rung whose cell count is within this fraction of one already offered is dropped.
const DUPLICATE_CELL_RATIO: f64 = 0.05;

/// Lattice score at or above which the image clearly has a pixel grid.
pub const CLEAR_LATTICE_SCORE: f32 = 0.20;
/// Lattice score below which no pixel grid was found. Provisional: six images gave
/// 0.52, 0.38 and 0.24 for real lattices, 0.10 for mixed blocks, 0.09 and 0.02 for none.
pub const NO_LATTICE_SCORE: f32 = 0.07;

/// What a lattice score means, in a word: "clear", "weak" or "none".
pub fn lattice_label(score: f32) -> &'static str {
    if score >= CLEAR_LATTICE_SCORE {
        "clear"
    } else if score >= NO_LATTICE_SCORE {
        "weak"
    } else {
        "none"
    }
}

/// The grid `config` asks for: the typed width if there is one, else the grid scale.
pub fn requested_grid(
    base: &Grid,
    profiles: &Profiles,
    source: &RgbaImage,
    config: &Config,
) -> Result<Grid> {
    if config.target_width > 0 {
        let lab = LabImage::new(source);
        let score = lab.lattice_score(base);
        grid_for_width(config.target_width, score, base, profiles, source.width(), source.height())
    } else {
        Ok(grid_at_scale(base.clone(), profiles, config, config.grid_scale))
    }
}

/// A grid about `target_width` cells wide.
///
/// When the image has a lattice, the grid follows it (see [`grid::grid_for_cols`]) and
/// the width may come out a few cells off. When it has none (`lattice_score` under
/// [`NO_LATTICE_SCORE`]), following it buys almost nothing and the user asked for a
/// number: the width is honoured exactly with an even grid.
pub fn grid_for_width(
    target_width: u32,
    lattice_score: f32,
    base: &Grid,
    profiles: &Profiles,
    width: u32,
    height: u32,
) -> Result<Grid> {
    let most = width as usize / MIN_CELL_PX;
    if target_width as usize > most.max(1) {
        return Err(PixelSnapperError::InvalidInput(format!(
            "A target width of {} needs cells under {} px; the most this {} px wide image allows is {}",
            target_width, MIN_CELL_PX, width, most.max(1)
        )));
    }
    let pitch = width as f64 / target_width as f64;
    if lattice_score < NO_LATTICE_SCORE {
        let rows = ((height as f64 / pitch).round() as usize).clamp(1, (height as usize / MIN_CELL_PX).max(1));
        Ok(grid::even_grid(width as usize, height as usize, target_width as usize, rows))
    } else {
        Ok(grid::grid_for_cols(base, profiles, target_width as usize, width, height))
    }
}

/// One rung of the ladder.
pub struct Rung {
    pub grid: Grid,
    /// How many times finer than the detected grid this was asked to be.
    pub grid_scale: f64,
}

/// The detected grid, its finer rungs, and the typed width if there is one: distinct
/// grids only, coarsest first.
pub fn ladder(
    base: &Grid,
    lattice_score: f32,
    profiles: &Profiles,
    width: u32,
    height: u32,
    config: &Config,
) -> Result<Vec<Rung>> {
    let mut rungs = vec![Rung { grid: base.clone(), grid_scale: 1.0 }];
    let offer = |rungs: &mut Vec<Rung>, grid: Grid, grid_scale: f64| {
        let cells = grid.cell_count() as f64;
        let duplicate = rungs.iter().any(|r| {
            let other = r.grid.cell_count() as f64;
            (cells - other).abs() <= DUPLICATE_CELL_RATIO * other
        });
        if !duplicate {
            rungs.push(Rung { grid, grid_scale });
        }
    };

    for &scale in &LADDER_SCALES[1..] {
        if let Some(fine) = grid::refine(base, profiles, config, scale) {
            offer(&mut rungs, fine, scale);
        }
    }
    if config.target_width > 0 {
        let typed = grid_for_width(config.target_width, lattice_score, base, profiles, width, height)?;
        let scale = (typed.cols.len() - 1) as f64 / (base.cols.len() - 1) as f64;
        if lattice_score < NO_LATTICE_SCORE {
            // The width was promised exactly, so it is always offered, however close
            // to another rung it is, unless it is that rung, cut for cut.
            if !rungs.iter().any(|r| r.grid.cols == typed.cols && r.grid.rows == typed.rows) {
                rungs.push(Rung { grid: typed, grid_scale: scale });
            }
        } else {
            // On a lattice the width is a request, and the answer may be a few cells
            // off. If a rung within a twentieth of that width is already there, it is
            // the answer, whatever its height came out as: a second grid two columns
            // away would be a near-duplicate thumbnail.
            let target = config.target_width as f64;
            let already = rungs.iter().any(|r| {
                ((r.grid.cols.len() - 1) as f64 - target).abs() <= DUPLICATE_CELL_RATIO * target
            });
            if !already {
                offer(&mut rungs, typed, scale);
            }
        }
    }

    // Coarsest first; rungs of equal cell count keep the order they were offered in.
    rungs.sort_by_key(|r| r.grid.cell_count());
    Ok(rungs)
}

/// One finished result of the audition.
pub struct CandidateImage {
    pub image: RgbaImage,
    /// The grid it was built on; `None` for modes that use none.
    pub grid: Option<Grid>,
    pub grid_scale: f64,
    /// Lattice score of the image on its detected grid (the same for every candidate
    /// of one image), or 0 when no grid is involved.
    pub lattice_score: f32,
    /// True for the result the original `process_image` gives: snap mode, the legacy
    /// fitter, the detected grid.
    pub is_legacy: bool,
}

/// Every candidate for `rgba` under `config`.
///
/// * Snap mode: one small image per rung. With the legacy fitter the rung at the
///   detected grid is the legacy output, byte for byte.
/// * No-resize with Blocks: one full-size image per rung.
/// * Anything else uses no grid and yields its single result.
///
/// The image is quantized once; each further rung costs a vote and a paint.
pub fn build(rgba: &RgbaImage, config: &Config) -> Result<Vec<CandidateImage>> {
    let (width, height) = rgba.dimensions();

    if !config.no_resize {
        if config.cleanup != Cleanup::Off {
            return Err(PixelSnapperError::InvalidInput(format!(
                "Cleanup '{}' needs no-resize mode: the snap output is already one colour per cell",
                config.cleanup.name()
            )));
        }
        let quantized = quantize_image(rgba, config)?;
        let profiles = grid::profiles(rgba, &quantized, config)?;
        // The legacy grid is used exactly as detected, 1 px cells included.
        let base = grid::detect(&profiles, width, height, config)?;
        let score = LabImage::new(rgba).lattice_score(&base);
        let legacy_fitter = config.grid_fitter == GridFitter::Walk;
        return ladder(&base, score, &profiles, width, height, config)?
            .into_iter()
            .map(|rung| {
                Ok(CandidateImage {
                    image: resample(&quantized, &rung.grid.cols, &rung.grid.rows)?,
                    is_legacy: legacy_fitter && rung.grid == base,
                    grid: Some(rung.grid),
                    grid_scale: rung.grid_scale,
                    lattice_score: score,
                })
            })
            .collect();
    }

    let quantized = quantize_no_resize(rgba, config)?;
    if config.cleanup == Cleanup::Blocks {
        if let Some(prep) = cleanup::prepare_blocks(rgba, &quantized, config)? {
            let score = LabImage::new(rgba).lattice_score(&prep.base);
            return Ok(ladder(&prep.base, score, &prep.profiles, width, height, config)?
                .into_iter()
                .map(|rung| CandidateImage {
                    image: prep.render(&rung.grid),
                    grid: Some(rung.grid),
                    grid_scale: rung.grid_scale,
                    lattice_score: score,
                    is_legacy: false,
                })
                .collect());
        }
    }

    let image = match config.cleanup {
        Cleanup::Smooth => cleanup::smooth(rgba, &quantized, config)?.image,
        Cleanup::Off | Cleanup::Blocks => quantized,
    };
    Ok(vec![CandidateImage { image, grid: None, grid_scale: 1.0, lattice_score: 0.0, is_legacy: false }])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{synthetic_lattice, DISTINCT_8};
    use crate::{process_candidates_bytes, process_image_bytes_common, run_pipeline};
    use image::Rgba;
    use rand::prelude::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn snap(fitter: GridFitter) -> Config {
        Config { k_colors: 12, grid_fitter: fitter, ..Default::default() }
    }

    fn blocks(fitter: GridFitter) -> Config {
        Config { no_resize: true, cleanup: Cleanup::Blocks, ..snap(fitter) }
    }

    fn blurred_lattice(cell: u32, cells: usize, seed: u64) -> RgbaImage {
        synthetic_lattice(&vec![cell; cells], &vec![cell; cells], &DISTINCT_8, true, 4, seed).0
    }

    fn noise(w: u32, h: u32, seed: u64) -> RgbaImage {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut img = RgbaImage::new(w, h);
        for p in img.pixels_mut() {
            *p = Rgba([rng.gen(), rng.gen(), rng.gen(), 255]);
        }
        img
    }

    fn png(img: &RgbaImage) -> Vec<u8> {
        crate::encode_png(img).unwrap()
    }

    fn widths(candidates: &[CandidateImage]) -> Vec<usize> {
        candidates.iter().map(|c| c.grid.as_ref().unwrap().cols.len() - 1).collect()
    }

    #[test]
    fn ladder_is_distinct_and_runs_coarse_to_fine() {
        let img = blurred_lattice(9, 24, 1);
        for config in [snap(GridFitter::Walk), snap(GridFitter::Dp), blocks(GridFitter::Walk), blocks(GridFitter::Dp)] {
            let candidates = build(&img, &config).unwrap();
            assert!(candidates.len() >= 3, "{} candidates", candidates.len());
            let cells: Vec<usize> = candidates.iter().map(|c| c.grid.as_ref().unwrap().cell_count()).collect();
            for pair in cells.windows(2) {
                assert!(pair[1] as f64 > pair[0] as f64 * (1.0 + DUPLICATE_CELL_RATIO), "{cells:?}");
            }
            // The detected grid is the coarsest rung and is marked as scale 1.
            assert_eq!(candidates[0].grid_scale, 1.0);
            for c in &candidates {
                let g = c.grid.as_ref().unwrap();
                let narrowest = |cuts: &[usize]| cuts.windows(2).map(|p| p[1] - p[0]).min().unwrap();
                if c.grid_scale > 1.0 || config.no_resize {
                    assert!(narrowest(&g.cols) >= MIN_CELL_PX && narrowest(&g.rows) >= MIN_CELL_PX);
                }
                assert_eq!(c.lattice_score, candidates[0].lattice_score);
            }
        }
    }

    /// The candidate marked legacy is what `process_image` has always returned, byte
    /// for byte, on every fixture including the one with transparency.
    #[test]
    fn the_legacy_candidate_is_the_legacy_output() {
        let fixtures: [&[u8]; 4] = [
            include_bytes!("../tests/fixtures/wolf_ear_128.png"),
            include_bytes!("../tests/fixtures/colliseum_128.png"),
            include_bytes!("../tests/fixtures/ogtrail_128.png"),
            include_bytes!("../tests/fixtures/wolf_ear_128_alpha.png"),
        ];
        for bytes in fixtures {
            let legacy = process_image_bytes_common(bytes, None).unwrap();
            let records = process_candidates_bytes(bytes, &Config::default()).unwrap();
            let marked: Vec<_> = records.iter().filter(|r| r.is_legacy).collect();
            assert_eq!(marked.len(), 1, "exactly one legacy candidate");
            assert_eq!(marked[0].png, legacy);
            assert_eq!(marked[0].grid_scale, 1.0);
            assert_eq!((marked[0].width, marked[0].height), (marked[0].cols, marked[0].rows));
            let decoded = image::load_from_memory(&marked[0].png).unwrap().to_rgba8();
            assert_eq!(marked[0].unique_colors as usize, crate::regions::image_stats(&decoded).unique_colors);
            assert_eq!(*marked[0].col_cuts.last().unwrap(), 128);

            // The elastic fitter never claims to be the legacy result.
            let dp = Config { grid_fitter: GridFitter::Dp, ..Default::default() };
            assert!(process_candidates_bytes(bytes, &dp).unwrap().iter().all(|r| !r.is_legacy));
        }
    }

    #[test]
    fn candidates_are_what_the_single_result_calls_give() {
        let img = blurred_lattice(8, 20, 2);
        for fitter in [GridFitter::Walk, GridFitter::Dp] {
            // Blocks: full-size images, and the rungs at scales 1 and 2 are exactly what
            // asking for those scales directly returns.
            let candidates = build(&img, &blocks(fitter)).unwrap();
            for c in &candidates {
                assert_eq!(c.image.dimensions(), img.dimensions());
                assert!(!c.is_legacy);
            }
            for scale in [1.0, 2.0] {
                let direct = run_pipeline(&img, &Config { grid_scale: scale, ..blocks(fitter) }).unwrap();
                let rung = candidates.iter().find(|c| c.grid_scale == scale).expect("rung is offered");
                assert_eq!(rung.image, direct.image, "{fitter:?} scale {scale}");
                assert_eq!(rung.grid, direct.grid);
            }
            // Snap: each image is one pixel per cell of its grid.
            for c in build(&img, &snap(fitter)).unwrap() {
                let g = c.grid.as_ref().unwrap();
                assert_eq!(c.image.dimensions(), ((g.cols.len() - 1) as u32, (g.rows.len() - 1) as u32));
            }
        }
        // Modes without a grid give their one result and no cuts.
        for cleanup in [Cleanup::Off, Cleanup::Smooth] {
            let config = Config { no_resize: true, cleanup, k_colors: 12, ..Default::default() };
            let candidates = build(&img, &config).unwrap();
            assert_eq!(candidates.len(), 1);
            assert!(candidates[0].grid.is_none());
            assert_eq!(candidates[0].image, run_pipeline(&img, &config).unwrap().image);
            let records = process_candidates_bytes(&png(&img), &config).unwrap();
            assert_eq!((records[0].cols, records[0].rows), (0, 0));
            assert!(records[0].col_cuts.is_empty() && records[0].row_cuts.is_empty());
        }
        // Cleanup without no-resize is refused here as everywhere else.
        assert!(build(&img, &Config { cleanup: Cleanup::Blocks, ..Default::default() }).is_err());
    }

    #[test]
    fn lattice_score_tells_a_grid_from_none() {
        let dp = snap(GridFitter::Dp);
        let score = |img: &RgbaImage| build(img, &dp).unwrap()[0].lattice_score;

        // A regular lattice, aligned with the image and offset by half a cell.
        let aligned = blurred_lattice(8, 24, 3);
        assert!(score(&aligned) > 0.3, "aligned: {}", score(&aligned));
        let offset = image::imageops::crop_imm(&aligned, 4, 4, 180, 180).to_image();
        assert!(score(&offset) > 0.3, "offset: {}", score(&offset));

        // The case a same-cell-count comparison gets wrong: a perfectly regular, crisp
        // lattice. An even grid matches it exactly; the score must still be high.
        let crisp = synthetic_lattice(&[8; 24], &[8; 24], &DISTINCT_8, false, 0, 3).0;
        assert!(score(&crisp) > 0.9, "crisp: {}", score(&crisp));

        // No lattice.
        let flat = RgbaImage::from_pixel(96, 96, Rgba([90, 120, 60, 255]));
        assert_eq!(score(&flat), 0.0);
        assert!(score(&noise(160, 160, 4)).abs() < 0.05, "noise: {}", score(&noise(160, 160, 4)));

        assert_eq!(lattice_label(0.52), "clear");
        assert_eq!(lattice_label(CLEAR_LATTICE_SCORE), "clear");
        assert_eq!(lattice_label(0.10), "weak");
        assert_eq!(lattice_label(NO_LATTICE_SCORE), "weak");
        assert_eq!(lattice_label(0.02), "none");
        assert_eq!(lattice_label(-0.3), "none");
    }

    /// Blocks of 12 px on the left half, 4 px on the right, with a 4 px-wide line
    /// through the fine half. Whatever the estimate lands on, the ladder must reach a
    /// rung fine enough for the line, and the line must be there.
    #[test]
    fn mixed_block_sizes_get_a_rung_that_keeps_the_fine_detail() {
        let (coarse, _) = synthetic_lattice(&[12; 10], &[12; 20], &DISTINCT_8[1..4], true, 3, 5);
        let (fine, _) = synthetic_lattice(&[4; 30], &[4; 60], &DISTINCT_8[1..4], true, 3, 6);
        let mut img = RgbaImage::new(240, 240);
        image::imageops::replace(&mut img, &coarse, 0, 0);
        image::imageops::replace(&mut img, &fine, 120, 0);
        let line = Rgba([20, 20, 24, 255]); // DISTINCT_8[0], used nowhere else
        for y in 20..220 {
            for x in 180..184 {
                img.put_pixel(x, y, line);
            }
        }

        for fitter in [GridFitter::Walk, GridFitter::Dp] {
            let candidates = build(&img, &blocks(fitter)).unwrap();
            let median_cell = |c: &CandidateImage| {
                let g = c.grid.as_ref().unwrap();
                let mut w: Vec<usize> = g.cols.windows(2).map(|p| p[1] - p[0]).collect();
                w.sort_unstable();
                w[w.len() / 2]
            };
            let fine_rungs: Vec<&CandidateImage> = candidates.iter().filter(|c| median_cell(c) <= 4).collect();
            assert!(!fine_rungs.is_empty(), "{fitter:?}: cells {:?}", candidates.iter().map(median_cell).collect::<Vec<_>>());
            // The line survives where, on most of its rows, there is a pixel of the
            // line's own colour (dark on all three channels: the background's green is
            // dark only in red) with the background's colours either side of it.
            let dark = |p: &Rgba<u8>| p[0] < 80 && p[1] < 80 && p[2] < 80;
            let rows_with_line = |c: &CandidateImage| {
                (40..200)
                    .filter(|&y| {
                        (180..184).any(|x| dark(c.image.get_pixel(x, y)))
                            && !dark(c.image.get_pixel(174, y))
                            && !dark(c.image.get_pixel(189, y))
                    })
                    .count()
            };
            let survives = fine_rungs.iter().any(|c| rows_with_line(c) >= 150);
            assert!(survives, "{fitter:?}: the line is gone from every fine rung");

            // The check itself must be able to fail: paint the line out, or recolour
            // it with the background's green, and it does.
            let mut erased = fine_rungs[0].image.clone();
            let mut recoloured = fine_rungs[0].image.clone();
            for y in 0..240 {
                for x in 176..188 {
                    erased.put_pixel(x, y, Rgba([235, 232, 225, 255]));
                    recoloured.put_pixel(x, y, Rgba([50, 160, 70, 255]));
                }
            }
            for (name, image) in [("erased", erased), ("recoloured", recoloured)] {
                let fake = CandidateImage { image, grid: None, grid_scale: 1.0, lattice_score: 0.0, is_legacy: false };
                assert!(rows_with_line(&fake) < 150, "{fitter:?}: the check passes on a {name} line");
            }
        }
    }

    #[test]
    fn a_typed_width_is_fitted_on_a_lattice_and_exact_without_one() {
        // Lattice of 24 cells of 9 px: asking for 48 gives a fitted grid near 48.
        let img = blurred_lattice(9, 24, 7);
        for fitter in [GridFitter::Walk, GridFitter::Dp] {
            let config = Config { target_width: 48, ..snap(fitter) };
            let direct = run_pipeline(&img, &config).unwrap();
            let got = direct.image.width() as i32;
            assert!((got - 48).abs() <= 4, "{fitter:?}: asked for 48, got {got}");
            // It is one of the candidates too, unless it duplicates the x2 rung.
            let candidates = build(&img, &config).unwrap();
            assert!(widths(&candidates).iter().any(|&w| (w as i32 - 48).abs() <= 4), "{:?}", widths(&candidates));
            // A coarser size than the detected grid is available on request.
            let coarse = run_pipeline(&img, &Config { target_width: 12, ..snap(fitter) }).unwrap();
            assert!((coarse.image.width() as i32 - 12).abs() <= 2, "{fitter:?}: {}", coarse.image.width());
        }

        // No lattice: the number asked for, exactly, on both axes' aspect.
        let img = noise(200, 120, 8);
        for target in [16u32, 50, 77, 100] {
            let out = run_pipeline(&img, &Config { target_width: target, ..snap(GridFitter::Dp) }).unwrap();
            assert_eq!(out.image.width(), target);
            assert_eq!(out.image.height(), (120.0 * target as f64 / 200.0).round() as u32);
            let blocks_out = run_pipeline(&img, &Config { target_width: target, ..blocks(GridFitter::Dp) }).unwrap();
            assert_eq!(blocks_out.grid.unwrap().cols.len() - 1, target as usize);
            assert_eq!(blocks_out.image.dimensions(), (200, 120));
        }

        // The same promise in the audition (Codex, Phase 5 review: the exact rung used
        // to be dropped when another rung was within a twentieth of it). A rectangular
        // image, widths next to and equal to existing rungs.
        let config = snap(GridFitter::Dp);
        let plain = build(&img, &config).unwrap();
        assert!(plain[0].lattice_score < NO_LATTICE_SCORE);
        let existing = widths(&plain);
        for &rung in &existing {
            for target in [rung.saturating_sub(2).max(1), rung, rung + 1, rung + 2] {
                if target > 100 {
                    continue;
                }
                let got = build(&img, &Config { target_width: target as u32, ..config.clone() }).unwrap();
                let exact_rows = (120.0 * target as f64 / 200.0).round() as u32;
                assert!(
                    got.iter().any(|c| c.image.dimensions() == (target as u32, exact_rows)),
                    "asked for {target} beside the {rung} rung, got {:?}",
                    got.iter().map(|c| c.image.dimensions()).collect::<Vec<_>>()
                );
                // Never the same grid twice.
                for (i, a) in got.iter().enumerate() {
                    assert!(got[i + 1..].iter().all(|b| a.grid != b.grid), "duplicate grid for target {target}");
                }
                // Still coarsest first.
                let cells: Vec<usize> = got.iter().map(|c| c.grid.as_ref().unwrap().cell_count()).collect();
                assert!(cells.windows(2).all(|p| p[0] <= p[1]), "{cells:?}");
            }
        }

        // Cells under 2 px are refused, with the limit in the message.
        let err = run_pipeline(&img, &Config { target_width: 101, ..snap(GridFitter::Dp) });
        assert!(matches!(err, Err(PixelSnapperError::InvalidInput(ref m)) if m.contains("100")), "{:?}", err.err());
        assert!(build(&img, &Config { target_width: 101, ..snap(GridFitter::Dp) }).is_err());
    }

    #[test]
    fn a_typed_width_that_matches_a_rung_is_not_offered_twice() {
        let img = blurred_lattice(9, 24, 7);
        let config = snap(GridFitter::Dp);
        let plain = widths(&build(&img, &config).unwrap());
        for &existing in &plain {
            let with = widths(&build(&img, &Config { target_width: existing as u32, ..config.clone() }).unwrap());
            assert_eq!(with, plain, "asking for {existing} changed the ladder");
        }
    }

    #[test]
    fn even_grids_and_typed_pitches_are_well_formed() {
        let g = grid::even_grid(100, 30, 7, 4);
        assert_eq!(g.cols, vec![0, 14, 29, 43, 57, 71, 86, 100]);
        assert_eq!(g.rows, vec![0, 8, 15, 23, 30]);
        assert!(!g.step_estimated);
        // More cells than pixels collapses to one cell per pixel.
        assert_eq!(grid::even_grid(3, 2, 9, 9).cols, vec![0, 1, 2, 3]);

        let img = blurred_lattice(9, 24, 7);
        for fitter in [GridFitter::Walk, GridFitter::Dp] {
            let config = snap(fitter);
            let p = grid::profiles(&img, &img, &config).unwrap();
            let base = grid::detect(&p, 216, 216, &config).unwrap();
            for target in [1usize, 5, 12, 24, 36, 48, 72, 90, 108] {
                let g = grid::grid_for_cols(&base, &p, target, 216, 216);
                for cuts in [&g.cols, &g.rows] {
                    assert_eq!((cuts[0], *cuts.last().unwrap()), (0, 216));
                    assert!(cuts.windows(2).all(|c| c[1] - c[0] >= MIN_CELL_PX), "{fitter:?} target {target}: {cuts:?}");
                }
            }
            // Under the legacy fitter, doubling the detected width is a subdivision.
            if fitter == GridFitter::Walk {
                let doubled = grid::grid_for_cols(&base, &p, 2 * (base.cols.len() - 1), 216, 216);
                assert!(base.with_min_cells().cols.iter().all(|c| doubled.cols.contains(c)));
            }
        }
    }
}
