//! The pixel grid of an image: where the cuts between logical pixels fall.


use crate::{
    compute_profiles, estimate_step_size, resolve_step_sizes, sanitize_cuts, srgb_to_oklab,
    stabilize_both_axes, walk, Config, PixelSnapperError, Result,
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

    /// The same grid with every cell under [`MIN_CELL_PX`] merged into a neighbour.
    /// The legacy walker emits such cells; the snap output keeps them (it is frozen),
    /// everything new must call this first.
    pub fn with_min_cells(mut self) -> Grid {
        self.cols = merge_narrow_intervals(self.cols);
        self.rows = merge_narrow_intervals(self.rows);
        self
    }
}

/// The evidence a grid is fitted to: edge strength along each axis, as the chosen
/// fitter sees it. Computed once per image and reused for every refinement.
pub struct Profiles {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub fitter: GridFitter,
}

/// Profiles for `config.grid_fitter`. `Walk` reads the quantized image (as the legacy
/// path always has), `Dp` reads the source.
pub fn profiles(source: &RgbaImage, quantized: &RgbaImage, config: &Config) -> Result<Profiles> {
    match config.grid_fitter {
        GridFitter::Walk => {
            let (x, y) = compute_profiles(quantized)?;
            Ok(Profiles { x, y, fitter: GridFitter::Walk })
        }
        GridFitter::Dp => {
            if source.width() < 3 || source.height() < 3 {
                return Err(PixelSnapperError::InvalidInput(
                    "Image too small (minimum 3x3)".to_string(),
                ));
            }
            let EdgeProfiles { x, y } = edge_profiles(source);
            Ok(Profiles { x, y, fitter: GridFitter::Dp })
        }
    }
}

/// Fit a grid of `width` x `height` to `profiles`.
///
/// Both fitters estimate the step the same way (median spacing of profile peaks) and
/// share the fallback. `Walk` then runs the legacy sequence unchanged: greedy walk and
/// cross-axis stabilisation. `Dp` runs the elastic fit on each axis.
pub fn detect(profiles: &Profiles, width: u32, height: u32, config: &Config) -> Result<Grid> {
    let step_x_opt = estimate_step_size(&profiles.x, config);
    let step_y_opt = estimate_step_size(&profiles.y, config);
    let step_estimated = step_x_opt.is_some() || step_y_opt.is_some();

    let (step_x, step_y) = resolve_step_sizes(step_x_opt, step_y_opt, width, height, config);

    let (cols, rows) = match profiles.fitter {
        GridFitter::Walk => {
            let raw_col_cuts = walk(&profiles.x, step_x, width as usize, config)?;
            let raw_row_cuts = walk(&profiles.y, step_y, height as usize, config)?;
            stabilize_both_axes(
                &profiles.x,
                &profiles.y,
                raw_col_cuts,
                raw_row_cuts,
                width as usize,
                height as usize,
                config,
            )
        }
        GridFitter::Dp => {
            let (pitch, cols, rows) = dp_fit_both_axes(profiles, step_x, step_y, width, height);
            return Ok(Grid { cols, rows, step: (pitch, pitch), step_estimated });
        }
    };

    Ok(Grid { cols, rows, step: (step_x, step_y), step_estimated })
}

/// The grid detection of the legacy snap path: gradient profiles of the quantized
/// image, median peak spacing, greedy walk, cross-axis stabilisation.
#[cfg(test)]
pub fn detect_grid(quantized: &RgbaImage, config: &Config) -> Result<Grid> {
    let walk_config = Config { grid_fitter: GridFitter::Walk, ..config.clone() };
    let p = profiles(quantized, quantized, &walk_config)?;
    detect(&p, quantized.width(), quantized.height(), &walk_config)
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
/// coarse cells were themselves wide enough. Other scales re-fit the profiles at
/// `step / scale` with the fitter that produced them.
pub fn refine(grid: &Grid, profiles: &Profiles, config: &Config, scale: f64) -> Option<Grid> {
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
        let refit = |profile: &[f64], step: f64, cuts: &[usize]| -> Option<Vec<usize>> {
            let limit = *cuts.last()?;
            let fine_step = (step / scale).max(1.0);
            match profiles.fitter {
                GridFitter::Walk => {
                    let walked = walk(profile, fine_step, limit, config).ok()?;
                    Some(merge_narrow_intervals(sanitize_cuts(walked, limit)))
                }
                GridFitter::Dp => Some(dp_cuts(profile, fine_step, limit)),
            }
        };
        (
            refit(&profiles.x, grid.step.0, &grid.cols)?,
            refit(&profiles.y, grid.step.1, &grid.rows)?,
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

/// [`refine`] with the legacy fitter, for tests written against it.
#[cfg(test)]
pub fn refine_grid(
    grid: &Grid,
    quantized: &RgbaImage,
    config: &Config,
    scale: f64,
) -> Option<Grid> {
    let walk_config = Config { grid_fitter: GridFitter::Walk, ..config.clone() };
    let p = profiles(quantized, quantized, &walk_config).ok()?;
    refine(grid, &p, &walk_config, scale)
}

/// The elastic fit on both axes, at the pitch near the estimate that fits best.
///
/// The step estimate is a median of whole-pixel peak spacings, so at a 5 px pitch it is
/// off by up to 10 % before anything else goes wrong, and missing peaks pull it up
/// further (the wolf: 6.0 for a lattice of about 5). The fit can absorb that, but a
/// grid fitted at the wrong pitch still has the wrong number of cells. So the fit is
/// run at a range of pitches around the estimate and the one with the best total score
/// on both axes is kept. Because a cut only earns what its edge stands above the local
/// level (see [`edge_gains`]), a pitch does not win by having more cuts, only by
/// putting them on edges.
fn dp_fit_both_axes(
    profiles: &Profiles,
    step_x: f64,
    step_y: f64,
    width: u32,
    height: u32,
) -> (f64, Vec<usize>, Vec<usize>) {
    // The fallback step of a small image can be under the smallest cell there is.
    let estimate = ((step_x + step_y) / 2.0).max(MIN_CELL_PX as f64);
    // One set of gains for every candidate, so that their scores are comparable.
    let (gains_x, gains_y) = (edge_gains(&profiles.x, estimate), edge_gains(&profiles.y, estimate));
    let fit = |pitch: f64| {
        let (cols, sx) = dp_fit(&gains_x, pitch, width as usize);
        let (rows, sy) = dp_fit(&gains_y, pitch, height as usize);
        (sx + sy, pitch, cols, rows)
    };
    let mut best = fit(estimate);
    for i in 0..DP_PITCH_CANDIDATES {
        let t = i as f64 / (DP_PITCH_CANDIDATES - 1) as f64;
        let pitch = estimate * (DP_PITCH_SEARCH.0 + t * (DP_PITCH_SEARCH.1 - DP_PITCH_SEARCH.0));
        if pitch < MIN_CELL_PX as f64 {
            continue;
        }
        let candidate = fit(pitch);
        if candidate.0 > best.0 {
            best = candidate;
        }
    }
    (best.1, best.2, best.3)
}

/// How well `grid` explains `source`: the mean OKLab distance of every opaque pixel
/// from the mean colour of its cell. 0 means every cell is one flat colour.
pub fn residual(source: &RgbaImage, grid: &Grid) -> f64 {
    let w = source.width() as usize;
    let lab: Vec<Option<[f32; 3]>> = source
        .pixels()
        .map(|p| (p[3] != 0).then(|| srgb_to_oklab(p[0], p[1], p[2])))
        .collect();
    let (mut total, mut count) = (0.0f64, 0u64);
    for row in grid.rows.windows(2) {
        for col in grid.cols.windows(2) {
            let (mut sum, mut n) = ([0.0f64; 3], 0u64);
            for y in row[0]..row[1] {
                for p in lab[y * w + col[0]..y * w + col[1]].iter().flatten() {
                    for ch in 0..3 {
                        sum[ch] += p[ch] as f64;
                    }
                    n += 1;
                }
            }
            if n == 0 {
                continue;
            }
            let mean = [sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64];
            for y in row[0]..row[1] {
                for p in lab[y * w + col[0]..y * w + col[1]].iter().flatten() {
                    let d: f64 = (0..3).map(|ch| (p[ch] as f64 - mean[ch]).powi(2)).sum();
                    total += d.sqrt();
                }
            }
            count += n;
        }
    }
    if count == 0 { 0.0 } else { total / count as f64 }
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

// ---------------------------------------------------------------------------
// The elastic ("DP") fitter
// ---------------------------------------------------------------------------

/// Which grid fitter to use. Values are part of the WASM and CLI interface.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridFitter {
    /// The original: luma gradients of the quantized image, greedy walk.
    Walk = 0,
    /// Edge profiles of the source in OKLab, globally optimal elastic fit.
    Dp = 1,
}

impl GridFitter {
    pub fn try_from_u32(v: u32) -> std::result::Result<Self, String> {
        match v {
            0 => Ok(GridFitter::Walk),
            1 => Ok(GridFitter::Dp),
            _ => Err(format!("Invalid grid fitter: {}. Valid values: 0 (Walk), 1 (DP)", v)),
        }
    }

    pub fn from_name(name: &str) -> std::result::Result<Self, String> {
        match name {
            "walk" => Ok(GridFitter::Walk),
            "dp" => Ok(GridFitter::Dp),
            _ => Err(format!("Invalid grid fitter '{}'. Valid values: walk, dp", name)),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            GridFitter::Walk => "walk",
            GridFitter::Dp => "dp",
        }
    }
}

/// Interior steps of the DP fit stay within this fraction of the pitch.
const DP_STEP_TOLERANCE: f64 = 0.25;
/// Weight of the quadratic penalty on a step's deviation from the pitch.
const DP_STEP_PENALTY: f64 = 4.0;
/// Weight of the edge term against the step penalty. See [`edge_gains`].
const DP_EDGE_WEIGHT: f64 = 1.0;
/// The fit is tried at this many pitches...
const DP_PITCH_CANDIDATES: usize = 37;
/// ...spread evenly over this range, as multiples of the step estimate.
const DP_PITCH_SEARCH: (f64, f64) = (0.8, 1.25);
/// The first and last cell may be this fraction of the pitch at the narrowest.
const DP_BORDER_FRACTION: f64 = 0.35;
/// OKLab distance credited to an edge between an opaque and a transparent pixel.
const ALPHA_EDGE: f64 = 0.3;

/// Edge strength at every possible cut, per axis. `x[c]` is the summed OKLab distance
/// between columns `c - 1` and `c` over all rows: the evidence for a cut *at* `c`,
/// which is where an edge between two pixels actually lies. (The legacy profile is a
/// central difference, centred on a pixel; a crisp edge puts two equal samples in it,
/// which its estimator cannot see and which lands cuts one pixel early about half
/// the time.)
pub struct EdgeProfiles {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

pub fn edge_profiles(source: &RgbaImage) -> EdgeProfiles {
    let (w, h) = (source.width() as usize, source.height() as usize);
    let lab: Vec<Option<[f32; 3]>> = source
        .pixels()
        .map(|p| (p[3] != 0).then(|| srgb_to_oklab(p[0], p[1], p[2])))
        .collect();
    let edge = |a: &Option<[f32; 3]>, b: &Option<[f32; 3]>| -> f64 {
        match (a, b) {
            (Some(a), Some(b)) => {
                let (dl, da, db) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
                ((dl * dl + da * da + db * db) as f64).sqrt()
            }
            (None, None) => 0.0,
            _ => ALPHA_EDGE,
        }
    };
    let mut x = vec![0.0; w];
    let mut y = vec![0.0; h];
    for row in 0..h {
        for col in 0..w {
            let i = row * w + col;
            if col > 0 {
                x[col] += edge(&lab[i - 1], &lab[i]);
            }
            if row > 0 {
                y[row] += edge(&lab[i - w], &lab[i]);
            }
        }
    }
    EdgeProfiles { x, y }
}

/// Cuts from 0 to `limit` that maximise the edge strength collected at the cuts
/// (measured against its local level) minus a penalty for steps that deviate from
/// `pitch`, with every interior step within [`DP_STEP_TOLERANCE`] of the pitch.
/// Globally optimal (Viterbi over cut positions), unlike the greedy walker, and never
/// emits a cell under [`MIN_CELL_PX`].
pub fn dp_cuts(profile: &[f64], pitch: f64, limit: usize) -> Vec<usize> {
    dp_fit(&edge_gains(profile, pitch), pitch, limit).0
}

/// What a cut at each position earns: how far the edge there stands above its
/// surroundings (the median of the profile within `window_pitch` either side), in units
/// of the profile's mean.
///
/// Measuring against the local level is what keeps the fit honest about cell count. In
/// a flat area and in an evenly textured one alike, every cut earns nothing, so only
/// the step penalty acts and the cells come out at the pitch. Measured against zero,
/// every extra cut would pay, and soft lattices are over-segmented (235 columns on the
/// wolf where about 200 is right). Measured against the global mean, cuts in flat areas
/// would cost, and those areas are stretched by an eighth. The median, unlike the
/// mean, is not raised by the very edges being measured.
fn edge_gains(profile: &[f64], window_pitch: f64) -> Vec<f64> {
    let mean = profile.iter().sum::<f64>() / profile.len().max(1) as f64;
    if !(mean > 0.0) || !window_pitch.is_finite() {
        return vec![0.0; profile.len()];
    }
    let half = (window_pitch.round() as usize).max(1);
    let mut window: Vec<f64> = Vec::with_capacity(2 * half + 1);
    (0..profile.len())
        .map(|c| {
            window.clear();
            window.extend_from_slice(&profile[c.saturating_sub(half)..(c + half + 1).min(profile.len())]);
            let mid = window.len() / 2;
            let (_, median, _) = window.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
            DP_EDGE_WEIGHT * (profile[c] - *median) / mean
        })
        .collect()
}

/// The step sizes a fit at `pitch` may use.
struct StepRange {
    /// Interior cells: within [`DP_STEP_TOLERANCE`] of the pitch, rounded inwards, but
    /// always at least two consecutive sizes so that every length can be tiled.
    lo: usize,
    hi: usize,
    /// The first and last cell may be partial (the image can start and end mid-cell),
    /// down to this width. They pay the same penalty as any other step for being
    /// short: a partial cell has to be earned by the edges it lets the rest land on.
    /// (Free partial cells make the fit shave a sliver off every border.)
    border_lo: usize,
}

impl StepRange {
    fn new(pitch: f64) -> StepRange {
        let lo = ((pitch * (1.0 - DP_STEP_TOLERANCE)).ceil() as usize).max(MIN_CELL_PX);
        let hi = ((pitch * (1.0 + DP_STEP_TOLERANCE)).floor() as usize).max(lo + 1);
        let border_lo = ((pitch * DP_BORDER_FRACTION) as usize).clamp(MIN_CELL_PX, lo);
        StepRange { lo, hi, border_lo }
    }
}

fn step_penalty(step: usize, pitch: f64) -> f64 {
    DP_STEP_PENALTY * ((step as f64 - pitch) / pitch).powi(2)
}

/// Score of a set of cuts under the rules of [`dp_fit`], or `None` if they break them.
/// Independent of the recurrence; the tests use it to check the fit against brute force.
#[cfg(test)]
fn score_cuts(gains: &[f64], pitch: f64, cuts: &[usize]) -> Option<f64> {
    let range = StepRange::new(pitch);
    let cells = cuts.len().checked_sub(1)?;
    if cells == 0 || cuts[0] != 0 {
        return None;
    }
    let mut score = 0.0;
    for (i, pair) in cuts.windows(2).enumerate() {
        let step = pair[1].checked_sub(pair[0])?;
        let border = i == 0 || i + 1 == cells;
        let min = if border { range.border_lo } else { range.lo };
        if step < min || step > range.hi {
            return None;
        }
        score -= step_penalty(step, pitch);
        if i + 1 < cells {
            score += gains.get(pair[1]).copied().unwrap_or(0.0);
        }
    }
    Some(score)
}

/// The fit itself, on gains from [`edge_gains`]. Returns the cuts and their score:
/// gains collected at the interior cuts minus step penalties. The cuts always run from
/// 0 to `limit`, and are the best-scoring set that keeps every interior cell within
/// [`StepRange`] and the two border cells no narrower than a partial cell may be.
/// Scores of fits to the same gains are comparable between pitches.
fn dp_fit(gains: &[f64], pitch: f64, limit: usize) -> (Vec<usize>, f64) {
    if !pitch.is_finite() || pitch < MIN_CELL_PX as f64 {
        return (vec![0, limit], f64::NEG_INFINITY);
    }
    if limit < 2 * MIN_CELL_PX {
        return (vec![0, limit], 0.0); // one forced cell: says nothing about the pitch
    }
    let range = StepRange::new(pitch);
    let gain = |c: usize| gains.get(c).copied().unwrap_or(0.0);

    const UNREACHED: f64 = f64::NEG_INFINITY;
    let mut best = vec![UNREACHED; limit + 1];
    let mut prev = vec![0usize; limit + 1];
    best[0] = 0.0;

    for c in 1..=limit {
        let last = c == limit;
        // The far border is not an edge of the picture and earns nothing.
        let here = if last { 0.0 } else { gain(c) };
        let mut top = UNREACHED;
        let mut from = 0;
        for step in range.border_lo..=range.hi.min(c) {
            let f = c - step;
            // A cell is a border cell if it touches either end of the axis.
            let border = f == 0 || last;
            if best[f] == UNREACHED || (!border && step < range.lo) {
                continue;
            }
            let v = best[f] - step_penalty(step, pitch);
            if v > top {
                top = v;
                from = f;
            }
        }
        if top > UNREACHED {
            best[c] = top + here;
            prev[c] = from;
        }
    }

    if best[limit] == UNREACHED {
        // No tiling of this length at this pitch (only possible for very short axes).
        return (vec![0, limit], UNREACHED);
    }
    let mut cuts = vec![limit];
    while *cuts.last().unwrap() != 0 {
        cuts.push(prev[*cuts.last().unwrap()]);
    }
    cuts.reverse();
    (cuts, best[limit])
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

    // ------------------------------------------------------------ DP fitter

    use crate::test_util::{recovered_pixel_fraction, synthetic_lattice, DISTINCT_8};
    use crate::{run_pipeline, Cleanup};
    use rand::prelude::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn dp_config() -> Config {
        Config { grid_fitter: GridFitter::Dp, ..Default::default() }
    }

    fn dp_grid(img: &RgbaImage) -> Grid {
        let config = dp_config();
        let p = profiles(img, img, &config).unwrap();
        detect(&p, img.width(), img.height(), &config).unwrap()
    }

    fn random_widths(n: usize, lo: u32, hi: u32, seed: u64) -> Vec<u32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n).map(|_| rng.gen_range(lo..=hi)).collect()
    }

    fn cuts_u32(widths: &[u32]) -> Vec<usize> {
        cuts_from_widths(&widths.iter().map(|&w| w as usize).collect::<Vec<_>>())
    }

    #[test]
    fn the_default_fitter_is_the_legacy_one() {
        assert_eq!(Config::default().grid_fitter, GridFitter::Walk);
        assert_eq!(GridFitter::try_from_u32(0), Ok(GridFitter::Walk));
        assert_eq!(GridFitter::try_from_u32(1), Ok(GridFitter::Dp));
        assert!(GridFitter::try_from_u32(2).is_err());
        assert_eq!(GridFitter::from_name("dp"), Ok(GridFitter::Dp));
        assert!(GridFitter::from_name("DP").is_err());
        assert_eq!((GridFitter::Walk.name(), GridFitter::Dp.name()), ("walk", "dp"));
    }

    /// Crisp pixel art is where the legacy estimator sees nothing. The edge profile
    /// has one clean peak per cut, so the DP fitter finds the lattice exactly.
    #[test]
    fn dp_finds_a_crisp_lattice_exactly() {
        let crisp = checkerboard(&[8; 12], false);
        let grid = dp_grid(&crisp);
        assert!(grid.step_estimated);
        let truth: Vec<usize> = (0..=12).map(|i| i * 8).collect();
        assert_eq!(grid.cols, truth);
        assert_eq!(grid.rows, truth);
        assert_eq!(residual(&crisp, &grid), 0.0);
    }

    /// A sprite on a flat background: edges only in the middle of the frame. The cell
    /// count across the empty margins must follow the pitch (a crisp 5x upscale of the
    /// wolf, 205x208 logical pixels, came back as 197x204 before gains were measured
    /// against the local level, and as 216x214 when that level was a local mean).
    #[test]
    fn dp_keeps_the_pitch_across_flat_margins() {
        let (lattice, _) = synthetic_lattice(&[5; 30], &[5; 30], &DISTINCT_8, false, 0, 77);
        let mut img = RgbaImage::from_pixel(400, 350, Rgba([250, 250, 250, 255]));
        for (x, y, p) in lattice.enumerate_pixels() {
            img.put_pixel(x + 125, y + 100, *p);
        }
        let grid = dp_grid(&img);
        assert_eq!((grid.cols.len() - 1, grid.rows.len() - 1), (80, 70));
        assert_eq!(residual(&img, &grid), 0.0);
        assert!((grid.step.0 - 5.0).abs() < 0.15, "pitch {}", grid.step.0);
    }

    #[test]
    fn dp_finds_an_irregular_crisp_lattice_exactly() {
        // 7..10 around a mean of 8.5: inside the +-25 % a fit is allowed to stretch.
        let (cols, rows) = (random_widths(30, 7, 10, 41), random_widths(30, 7, 10, 42));
        let (img, _) = synthetic_lattice(&cols, &rows, &DISTINCT_8, false, 0, 43);
        let grid = dp_grid(&img);
        assert_eq!(grid.cols, cuts_u32(&cols));
        assert_eq!(grid.rows, cuts_u32(&rows));
    }

    /// Blurred, noisy, with large flat areas (most neighbouring cells share a colour, so
    /// many cuts have no edge at all): the cell count must stay within 8 % of the truth.
    /// Without a cost per cut the fit over-segments exactly this kind of image; without
    /// the pitch search it inherits the whole-pixel error of the step estimate.
    #[test]
    fn dp_cell_count_is_right_on_soft_lattices() {
        let mostly_flat = [DISTINCT_8[0], DISTINCT_8[1], DISTINCT_8[1], DISTINCT_8[1], DISTINCT_8[2]];
        for (lo, hi, seed) in [(4u32, 6u32, 1u64), (5, 6, 2), (5, 5, 6), (7, 10, 3), (9, 9, 4), (11, 14, 7)] {
            let (cols, rows) = (random_widths(40, lo, hi, seed), random_widths(40, lo, hi, seed + 100));
            let (img, _) = synthetic_lattice(&cols, &rows, &mostly_flat, true, 4, seed);
            let grid = dp_grid(&img);
            let (nx, ny) = (grid.cols.len() - 1, grid.rows.len() - 1);
            assert!((37..=43).contains(&nx) && (37..=43).contains(&ny), "cells {lo}..{hi}: {nx}x{ny}, truth 40x40");
            // The pitch it settled on is the mean cell size, not the rounded estimate.
            let mean_cell = (img.width() + img.height()) as f64 / 80.0;
            assert!((grid.step.0 / mean_cell - 1.0).abs() < 0.05, "cells {lo}..{hi}: pitch {:.2}, mean cell {mean_cell:.2}", grid.step.0);
            assert!(min_interval(&grid.cols) >= MIN_CELL_PX && min_interval(&grid.rows) >= MIN_CELL_PX);
        }
    }

    /// The fit against brute force: on short axes, enumerate every legal set of cuts
    /// and check that the fit returns the best-scoring one, and that the score it
    /// reports is the score of what it returns (Codex, Phase 4 review: an earlier
    /// version removed border slivers after optimising and kept the old score).
    #[test]
    fn dp_fit_is_optimal_and_reports_its_own_score() {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        for _ in 0..300 {
            let limit = rng.gen_range(4usize..=16);
            let pitch = [2.0, 2.5, 3.0, 3.7, 4.0, 5.0, 6.4, 8.0][rng.gen_range(0..8)];
            let gains: Vec<f64> = (0..limit)
                .map(|_| if rng.gen_bool(0.35) { rng.gen_range(0.5..5.0) } else { rng.gen_range(-0.3..0.3) })
                .collect();
            let (cuts, score) = dp_fit(&gains, pitch, limit);

            let mut best: Option<f64> = None;
            for mask in 0u32..(1 << (limit - 1)) {
                let mut candidate = vec![0];
                candidate.extend((1..limit).filter(|c| mask >> (c - 1) & 1 == 1));
                candidate.push(limit);
                if let Some(sc) = score_cuts(&gains, pitch, &candidate) {
                    best = Some(best.map_or(sc, |b: f64| b.max(sc)));
                }
            }
            match best {
                Some(b) => {
                    assert!(is_valid(&cuts, limit), "{cuts:?}");
                    let own = score_cuts(&gains, pitch, &cuts).expect("the fit returned illegal cuts");
                    assert!((own - score).abs() < 1e-9, "reported {score}, actual {own} for {cuts:?}");
                    assert!((score - b).abs() < 1e-9, "fit {score} against brute force {b}: {cuts:?} limit {limit} pitch {pitch}");
                }
                None => assert_eq!((cuts, score), (vec![0, limit], f64::NEG_INFINITY), "limit {limit} pitch {pitch}"),
            }
        }

        // Codex's example: the old code returned [0,2,11,17,24] (score 12.4) and
        // reported 17.8; the best legal fit is [0,2,11,19,24].
        let mut gains = vec![0.0; 24];
        for (c, v) in [(2, 3.0), (11, 3.0), (17, 3.0), (19, 4.0), (23, 4.0)] {
            gains[c] = v;
        }
        let (cuts, score) = dp_fit(&gains, 8.0, 24);
        assert_eq!(cuts, vec![0, 2, 11, 19, 24]);
        assert!((score - score_cuts(&gains, 8.0, &cuts).unwrap()).abs() < 1e-9);
    }

    /// Interior steps stay within a quarter of the pitch, rounded inwards.
    #[test]
    fn dp_interior_steps_stay_within_tolerance() {
        let r = StepRange::new(5.0);
        assert_eq!((r.lo, r.hi, r.border_lo), (4, 6, 2));
        let r = StepRange::new(8.0);
        assert_eq!((r.lo, r.hi, r.border_lo), (6, 10, 2));
        let r = StepRange::new(12.0);
        assert_eq!((r.lo, r.hi, r.border_lo), (9, 15, 4));
        // The smallest pitches still get two step sizes, or odd lengths could not be tiled.
        let r = StepRange::new(2.0);
        assert_eq!((r.lo, r.hi, r.border_lo), (2, 3, 2));

        // A profile that peaks every 3 px cannot pull a 5 px fit down to 3 px cells.
        let profile: Vec<f64> = (0..120).map(|c| if c % 3 == 0 { 5.0 } else { 0.0 }).collect();
        let cuts = dp_cuts(&profile, 5.0, 120);
        let interior: Vec<usize> = cuts.windows(2).map(|p| p[1] - p[0]).collect();
        assert!(interior[1..interior.len() - 1].iter().all(|s| (4..=6).contains(s)), "{interior:?}");
    }

    /// Gains are measured against the local **median**. A local mean would be raised
    /// by the edge being measured and push its neighbours below zero.
    #[test]
    fn edge_gains_use_a_baseline_that_edges_do_not_move() {
        // Level 1 everywhere, one edge of 10 at index 20, and a stretch of level 3.
        let mut profile = vec![1.0; 60];
        profile[20] = 10.0;
        for v in &mut profile[40..60] {
            *v = 3.0;
        }
        let gains = edge_gains(&profile, 5.0);
        let mean = profile.iter().sum::<f64>() / 60.0;
        assert!((gains[20] - 9.0 / mean).abs() < 1e-12, "the edge keeps its full height: {}", gains[20]);
        for c in (0..35).filter(|&c| c != 20) {
            assert_eq!(gains[c], 0.0, "position {c} next to an edge must stay at zero");
        }
        // An evenly textured stretch is as neutral as a flat one.
        for c in 46..54 {
            assert_eq!(gains[c], 0.0, "position {c}");
        }
        assert!(edge_gains(&[0.0; 30], 5.0).iter().all(|&g| g == 0.0));
    }

    /// A lattice where three boundaries in four have no edge at all: the cell count
    /// still follows the pitch.
    #[test]
    fn dp_bridges_missing_edges() {
        let profile: Vec<f64> = (0..400).map(|c| if c > 0 && c % 20 == 0 { 8.0 } else { 0.0 }).collect();
        let cuts = dp_cuts(&profile, 5.0, 400);
        assert_eq!(cuts, (0..=80).map(|i| i * 5).collect::<Vec<_>>());
    }

    #[test]
    fn dp_handles_short_and_edgeless_axes() {
        // No spacing estimate and a fallback step under 2 px: one clear boundary must
        // still be found (Codex, Phase 4 review: this used to give a 1x1 grid).
        let mut halves = RgbaImage::from_pixel(64, 64, Rgba([20, 20, 20, 255]));
        for y in 0..64 {
            for x in 32..64 {
                halves.put_pixel(x, y, Rgba([230, 230, 230, 255]));
            }
        }
        let grid = dp_grid(&halves);
        assert!(!grid.step_estimated);
        assert!(grid.cols.contains(&32), "{:?}", grid.cols);
        assert!(grid.cols.len() > 2 && grid.rows.len() > 2);
        assert_eq!(residual(&halves, &grid), 0.0);

        // A 3 px axis is one forced cell and must not veto the pitch search on the
        // other axis: the long axis of a crisp 5 px lattice is still found exactly.
        let (strip, _) = synthetic_lattice(&[5; 40], &[3], &DISTINCT_8, false, 0, 31);
        for img in [strip.clone(), image::imageops::rotate90(&strip)] {
            let grid = dp_grid(&img);
            let (long, short) = if img.width() > img.height() { (&grid.cols, &grid.rows) } else { (&grid.rows, &grid.cols) };
            assert_eq!(short, &vec![0, 3]);
            assert_eq!(long.len() - 1, 40, "{long:?}");
            assert_eq!(residual(&img, &grid), 0.0);
        }
    }

    #[test]
    fn dp_cuts_are_valid_for_awkward_inputs() {
        // Degenerate: too short, no pitch, pitch under the minimum cell.
        assert_eq!(dp_cuts(&[1.0, 1.0, 1.0], 2.0, 3), vec![0, 3]);
        assert_eq!(dp_cuts(&[1.0; 40], f64::NAN, 40), vec![0, 40]);
        assert_eq!(dp_cuts(&[1.0; 40], 1.0, 40), vec![0, 40]);
        // A flat profile gives an even grid at the pitch.
        let even = dp_cuts(&[0.0; 80], 8.0, 80);
        assert!(is_valid(&even, 80));
        assert_eq!(even.len() - 1, 10);
        // Small pitches and noise: always valid, never a cell under 2 px.
        let mut rng = ChaCha8Rng::seed_from_u64(8);
        for pitch in [2.0, 2.2, 2.7, 3.0, 3.5, 5.3, 17.9] {
            for limit in [4usize, 5, 7, 31, 200] {
                let profile: Vec<f64> = (0..limit).map(|_| rng.gen::<f64>()).collect();
                let cuts = dp_cuts(&profile, pitch, limit);
                assert!(is_valid(&cuts, limit), "pitch {pitch} limit {limit}: {cuts:?}");
                assert!(min_interval(&cuts) >= MIN_CELL_PX.min(limit), "pitch {pitch} limit {limit}: {cuts:?}");
            }
        }
    }

    #[test]
    fn dp_refinement_respects_the_same_rules() {
        let (cols, rows) = (random_widths(30, 7, 10, 51), random_widths(30, 7, 10, 52));
        let (img, _) = synthetic_lattice(&cols, &rows, &DISTINCT_8, true, 4, 53);
        let config = dp_config();
        let p = profiles(&img, &img, &config).unwrap();
        let grid = detect(&p, img.width(), img.height(), &config).unwrap();
        for scale in [1.5, 2.0, 3.0] {
            // Integer scales subdivide and always succeed here. At 1.5 the fit is re-run
            // at two thirds of the pitch and goes wherever the edges and the step
            // bounds let it; see `fractional_refinement_goes_where_the_edges_are`.
            let Some(fine) = refine(&grid, &p, &config, scale) else {
                assert_eq!(scale, 1.5, "scale {scale} must be achievable");
                continue;
            };
            assert!(is_valid(&fine.cols, img.width() as usize) && is_valid(&fine.rows, img.height() as usize));
            assert!(min_interval(&fine.cols) >= MIN_CELL_PX && min_interval(&fine.rows) >= MIN_CELL_PX);
            assert!(fine.cell_count() as f64 >= 1.2 * grid.cell_count() as f64, "scale {scale}");
            // A subdivision keeps every coarse cut, so it never explains the image worse.
            // (A re-fit at 1.5 is not nested in the coarse grid and can.)
            if scale != 1.5 {
                assert!(residual(&img, &fine) <= residual(&img, &grid) + 1e-9, "scale {scale}");
            }
        }
    }

    /// A clean lattice has no structure at two thirds of its pitch, so a x1.5 re-fit
    /// does not produce a "1.5 times finer" grid. What it does depends on whether the
    /// allowed steps can reach a whole subdivision.
    #[test]
    fn fractional_refinement_goes_where_the_edges_are() {
        let config = dp_config();
        // 8 px cells: at 5.33 px the steps may be 4..6, and 4 + 4 lands on every edge.
        // The result is the x2 grid.
        let img = checkerboard(&[8; 12], false);
        let p = profiles(&img, &img, &config).unwrap();
        let grid = detect(&p, 96, 96, &config).unwrap();
        let fine = refine(&grid, &p, &config, 1.5).expect("8 px cells can be halved");
        assert_eq!(fine.cols, (0..=24).map(|i| i * 4).collect::<Vec<_>>());

        // 3 px cells: at 2 px the steps may be 2 or 3, and only 3 lands on the edges.
        // The re-fit returns the original cuts, which is not finer: refused.
        let img = checkerboard(&[3; 30], false);
        let p = profiles(&img, &img, &config).unwrap();
        let grid = Grid {
            cols: (0..=30).map(|i| i * 3).collect(),
            rows: (0..=30).map(|i| i * 3).collect(),
            step: (3.0, 3.0),
            step_estimated: true,
        };
        assert_eq!(dp_cuts(&p.x, 2.0, 90), grid.cols);
        assert_eq!(refine(&grid, &p, &config, 1.5), None);
    }

    #[test]
    fn edge_profiles_count_transparency_edges_once() {
        // Row 0: opaque, opaque (same colour), transparent, transparent.
        let mut img = RgbaImage::from_pixel(4, 3, Rgba([50, 60, 70, 255]));
        for y in 0..3 {
            img.put_pixel(2, y, Rgba([255, 0, 0, 0]));
            img.put_pixel(3, y, Rgba([0, 255, 0, 0]));
        }
        let p = edge_profiles(&img);
        assert_eq!(p.x, vec![0.0, 0.0, 3.0 * ALPHA_EDGE, 0.0], "hidden colours of transparent pixels do not count");
        assert_eq!(p.y, vec![0.0, 0.0, 0.0]);

        let clear = RgbaImage::from_pixel(5, 5, Rgba([9, 9, 9, 0]));
        let p = edge_profiles(&clear);
        assert!(p.x.iter().chain(&p.y).all(|&v| v == 0.0));
    }

    #[test]
    fn residual_is_zero_only_on_the_right_grid() {
        let (cols, rows) = (vec![6u32, 9, 7, 11, 8], vec![8u32, 6, 10, 7]);
        let (img, _) = synthetic_lattice(&cols, &rows, &DISTINCT_8, false, 0, 13);
        let exact = Grid { cols: cuts_u32(&cols), rows: cuts_u32(&rows), step: (8.0, 8.0), step_estimated: true };
        assert_eq!(residual(&img, &exact), 0.0);
        let shifted = Grid { cols: vec![0, 3, 20, 41], rows: vec![0, 5, 31], ..exact.clone() };
        assert!(residual(&img, &shifted) > 0.05);
        // Transparent pixels are not measured.
        let clear = RgbaImage::from_pixel(41, 31, Rgba([1, 2, 3, 0]));
        assert_eq!(residual(&clear, &shifted), 0.0);
    }

    /// The point of the exercise, end to end: Blocks on the blurred lattice from the
    /// Phase 2 tests. The legacy grid gets about 90 % of pixels right because four cuts
    /// in ten sit one pixel early. Measured when written: 90.7 % and 100 %.
    #[test]
    fn dp_grid_puts_cuts_on_the_edges() {
        let (cols, rows) = ([8u32; 24], [8u32; 24]);
        let (img, truth) = synthetic_lattice(&cols, &rows, &DISTINCT_8, true, 4, 21);
        let blocks = |fitter| {
            let config = Config { k_colors: 12, no_resize: true, cleanup: Cleanup::Blocks, grid_fitter: fitter, ..Default::default() };
            recovered_pixel_fraction(&run_pipeline(&img, &config).unwrap().image, &cols, &rows, &truth)
        };
        let (walk, dp) = (blocks(GridFitter::Walk), blocks(GridFitter::Dp));
        assert!(dp >= 0.97, "DP: {:.1}% of pixels right", dp * 100.0);
        assert!(dp > walk + 0.05, "DP {dp:.3} against walk {walk:.3}");
    }
}
