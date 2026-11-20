//! Spatial cleanup of a palette-reduced image at its original size.
//!
//! Every quantizer ends by giving each pixel a palette colour on its own. On blurry
//! "pixel art" that leaves speckle: blends between two logical pixels land on some
//! third palette colour. The stages here put the missing spatial step back.

use crate::grid::{self, Grid, Profiles};
use crate::indexed::{index_image, render, IndexedImage, TRANSPARENT};
use crate::{regions, srgb_to_oklab, Config, Result};
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
pub fn grid_at_scale(base: Grid, profiles: &Profiles, config: &Config, scale: f64) -> Grid {
    if scale <= 1.0 {
        return base;
    }
    std::iter::once(scale)
        .chain(FALLBACK_SCALES.into_iter().filter(|&s| s < scale))
        .find_map(|s| grid::refine(&base, profiles, config, s))
        .unwrap_or(base)
}

/// Everything Blocks needs that does not depend on which grid is used: computed once,
/// then rendered on as many grids as the caller wants to offer.
pub struct BlocksPrep<'a> {
    /// The quantized image as labels, after un-mixing. This is what cells vote on.
    unmixed: IndexedImage<'a>,
    /// The labels as the quantizer left them, before un-mixing. The detail vote reads
    /// these: un-mixing would already have resolved the pixels it wants to hear from.
    raw_labels: Vec<u16>,
    /// The source in OKLab, row-major.
    lab: Vec<[f32; 3]>,
    pub profiles: Profiles,
    /// The detected grid, with no cell under 2 px.
    pub base: Grid,
    /// Decide alpha per cell instead of per pixel; see [`BlocksPrep::render`].
    cell_alpha: bool,
}

/// Prepare Blocks for `source`, or `None` when there is nothing to snap (no opaque
/// pixel, or an image under 3 px on a side, which is too small to hold a grid).
pub fn prepare_blocks<'a>(
    source: &'a RgbaImage,
    quantized: &RgbaImage,
    config: &Config,
) -> Result<Option<BlocksPrep<'a>>> {
    let mut ix = index_image(source, quantized)?;
    if ix.palette.is_empty() || source.width() < 3 || source.height() < 3 {
        return Ok(None);
    }

    // The detected grid can contain 1 px cells. A cell that small cannot hold a vote
    // (it would keep the very speckle this is meant to remove), so they are merged
    // away before anything else. Every refinement keeps the guarantee.
    let profiles = grid::profiles(source, quantized, config)?;
    let base = grid::detect(&profiles, source.width(), source.height(), config)?.with_min_cells();

    // Un-mix before voting: blend pixels then vote for one of the two colours they lie
    // between instead of for an unrelated third. (Measured in research on the coliseum
    // at twice the detected grid: 208 stray regions against 637 when voting on the raw
    // labels.)
    let raw_labels = ix.labels.clone();
    let lab = source_oklab(source);
    let pal_lab = palette_oklab(&ix.palette);
    unmix(&mut ix, &lab, &pal_lab, unmix_radius(estimated_pitch(Some(&base))));

    let cell_alpha = config.alpha_threshold > 0;
    Ok(Some(BlocksPrep { unmixed: ix, raw_labels, lab, profiles, base, cell_alpha }))
}

impl BlocksPrep<'_> {
    /// One colour per cell of `grid`, painted at the original size.
    ///
    /// * Colour, `Vote::Majority`: the majority label among the cell's pixels with
    ///   `alpha > 0`, counted after un-mixing; ties go to the label that appears first
    ///   in the image. `Vote::Detail`: see [`vote_cells_detail`], on the labels as
    ///   quantized.
    /// * Alpha, by default (`Config::alpha_threshold > 0`): per cell, like the colour.
    ///   A cell is opaque when at least half of its pixels are (the rule the snap
    ///   output's detail vote uses), and fully transparent, `[0, 0, 0, 0]`, otherwise.
    ///   The sprite's outline then follows the grid like everything else in it,
    ///   instead of cutting cells in two along wherever the source's soft edge crossed
    ///   the threshold. A feature thinner than half a cell inside transparent cells
    ///   goes with them, as it would in the snapped output; a finer grid keeps it.
    /// * Alpha with `alpha_threshold == 0` (partial alpha kept): every pixel keeps its
    ///   own source alpha. Fully transparent pixels are copied from the source, so an
    ///   opaque feature inside a mostly transparent cell keeps its shape and takes the
    ///   cell's colour.
    pub fn render(&self, grid: &Grid, vote: Vote) -> RgbaImage {
        let labels = match vote {
            Vote::Majority => self.unmixed.labels.clone(),
            Vote::Detail => self.raw_labels.clone(),
        };
        let mut ix = IndexedImage {
            source: self.unmixed.source,
            palette: self.unmixed.palette.clone(),
            labels,
        };
        let winners = match vote {
            Vote::Majority => vote_cells(&ix, grid),
            Vote::Detail => vote_cells_detail(&ix, &self.lab, grid),
        };
        if self.cell_alpha {
            return paint_cells_opaque(&ix, grid, &winners);
        }
        paint_cells(&mut ix, grid, &winners);
        render(&ix)
    }
}

/// Paint each cell whole: its winning colour, fully opaque, when at least half of its
/// pixels are opaque; `[0, 0, 0, 0]` otherwise.
fn paint_cells_opaque(ix: &IndexedImage, grid: &Grid, winners: &[u16]) -> RgbaImage {
    let w = ix.width();
    let mut out = RgbaImage::new(ix.source.width(), ix.source.height());
    let cells_per_row = grid.cols.len() - 1;
    for (cy, row) in grid.rows.windows(2).enumerate() {
        for (cx, col) in grid.cols.windows(2).enumerate() {
            let winner = winners[cy * cells_per_row + cx];
            if winner == TRANSPARENT {
                continue; // no opaque pixel at all
            }
            let area = (row[1] - row[0]) * (col[1] - col[0]);
            let opaque: usize = (row[0]..row[1])
                .map(|y| ix.labels[y * w + col[0]..y * w + col[1]].iter().filter(|&&l| l != TRANSPARENT).count())
                .sum();
            if opaque * 2 < area {
                continue;
            }
            let [r, g, b] = ix.palette[winner as usize];
            for y in row[0]..row[1] {
                for x in col[0]..col[1] {
                    out.put_pixel(x as u32, y as u32, image::Rgba([r, g, b, 255]));
                }
            }
        }
    }
    out
}

/// **Blocks** ("snap in place"): one colour per grid cell, painted back at the
/// original size, on the grid that `config` asks for. See [`BlocksPrep::render`].
pub fn blocks(source: &RgbaImage, quantized: &RgbaImage, config: &Config) -> Result<Cleaned> {
    let Some(prep) = prepare_blocks(source, quantized, config)? else {
        return Ok(Cleaned { image: quantized.clone(), grid: None });
    };
    let grid = crate::candidates::requested_grid(&prep.base, &prep.profiles, source, config)?;
    Ok(Cleaned { image: prep.render(&grid, config.vote), grid: Some(grid) })
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

// ---------------------------------------------------------------------------
// Smooth: un-mix + colour-aware sieve
// ---------------------------------------------------------------------------

/// A pixel is "core" (trusted to show a logical pixel's real colour) when the OKLab
/// range of the source pixels in its 3x3 neighbourhood is under this.
pub const CORE_FLATNESS: f32 = 0.03;

/// Un-mixing only re-labels a pixel whose source colour is explained as a blend: within
/// this OKLab distance of the straight line between two colours found in nearby flat
/// areas (or of one such colour). Anything further off is not a blend but a feature
/// too small to have a flat interior of its own, such as a thin outline or a 2x2
/// highlight, and is left alone.
pub const BLEND_TOLERANCE: f32 = 0.05;

/// Passes of the sieve before giving up on stragglers.
const SIEVE_MAX_PASSES: usize = 32;

/// Smallest pitch at which sieve thresholds scale with the cell instead of being
/// absolute: below it a sixteenth of a cell is under the 4 px floor anyway.
const CELL_RELATIVE_MIN_PITCH: f64 = 8.0;

/// How much the sieve may remove. Values are part of the WASM and CLI interface.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Strength {
    Light = 1,
    Medium = 2,
    Strong = 3,
}

impl Strength {
    pub fn try_from_u32(v: u32) -> std::result::Result<Self, String> {
        match v {
            1 => Ok(Strength::Light),
            2 => Ok(Strength::Medium),
            3 => Ok(Strength::Strong),
            _ => Err(format!(
                "Invalid cleanup strength: {}. Valid values: 1 (Light), 2 (Medium), 3 (Strong)",
                v
            )),
        }
    }

    pub fn from_name(name: &str) -> std::result::Result<Self, String> {
        match name {
            "light" => Ok(Strength::Light),
            "medium" => Ok(Strength::Medium),
            "strong" => Ok(Strength::Strong),
            _ => Err(format!("Invalid strength '{}'. Valid values: light, medium, strong", name)),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Strength::Light => "light",
            Strength::Medium => "medium",
            Strength::Strong => "strong",
        }
    }
}

/// Regions smaller than this many pixels are dissolved. `pitch` is the estimated size
/// of a logical pixel, or `None` when the image gave no estimate; a fallback step must
/// never be passed here.
pub fn sieve_min_area(strength: Strength, pitch: Option<f64>) -> f64 {
    let (floor, cell_fraction) = match strength {
        Strength::Light => (4.0, 1.0 / 16.0),
        Strength::Medium => (8.0, 1.0 / 8.0),
        Strength::Strong => (16.0, 1.0 / 4.0),
    };
    match pitch {
        Some(p) if p >= CELL_RELATIVE_MIN_PITCH => f64::max(floor, p * p * cell_fraction),
        _ => floor,
    }
}

/// How far an un-mixed pixel may look for core pixels. It has to see the flat areas
/// on **both** sides of the ramp it sits in, and a core pixel is by definition at
/// least one pixel away from any ramp, so even a one-pixel ramp needs a reach of 2.
/// Wider ramps come with coarser grids: a third of the pitch, plus that one pixel.
pub fn unmix_radius(pitch: Option<f64>) -> usize {
    match pitch {
        Some(p) => ((p / 3.0).round() as usize).clamp(1, 3) + 1,
        None => 2,
    }
}

/// The estimated pitch of `quantized`, or `None` when there is no estimate (image too
/// small, flat, or perfectly crisp) and only the fallback step exists.
pub fn estimated_pitch(grid: Option<&Grid>) -> Option<f64> {
    grid.filter(|g| g.step_estimated).map(|g| (g.step.0 + g.step.1) / 2.0)
}

/// OKLab of every source pixel (row-major). Values at transparent pixels are unused.
fn source_oklab(source: &RgbaImage) -> Vec<[f32; 3]> {
    source.pixels().map(|p| srgb_to_oklab(p[0], p[1], p[2])).collect()
}

fn palette_oklab(palette: &[[u8; 3]]) -> Vec<[f32; 3]> {
    palette.iter().map(|c| srgb_to_oklab(c[0], c[1], c[2])).collect()
}

fn dist_sq(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let (dl, da, db) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    dl * dl + da * da + db * db
}

/// **Smooth**: un-mix, then sieve. Needs no grid; uses the pitch only to scale itself.
pub fn smooth(source: &RgbaImage, quantized: &RgbaImage, config: &Config) -> Result<Cleaned> {
    let mut ix = index_image(source, quantized)?;
    if ix.palette.is_empty() {
        return Ok(Cleaned { image: render(&ix), grid: None });
    }
    // Grid detection is only a pitch estimate here, and is allowed to fail.
    let grid = grid::profiles(source, quantized, config)
        .and_then(|p| grid::detect(&p, source.width(), source.height(), config))
        .ok();
    let pitch = estimated_pitch(grid.as_ref());

    let lab = source_oklab(source);
    let pal_lab = palette_oklab(&ix.palette);
    unmix(&mut ix, &lab, &pal_lab, unmix_radius(pitch));
    sieve(&mut ix, &lab, &pal_lab, sieve_min_area(config.cleanup_strength, pitch));
    Ok(Cleaned { image: render(&ix), grid: None })
}

/// **Un-mix.** Blurry pixel art has a ramp of blended colours wherever two logical
/// pixels meet, and palette reduction paints that ramp with whatever palette colours
/// lie between the two, usually ones that belong to neither. Here a pixel that is not
/// in a flat ("core") neighbourhood, and whose source colour is a blend of colours that
/// core pixels within `radius` carry (see [`BLEND_TOLERANCE`]), takes the label at the
/// nearer end of that blend. A ramp between A and B therefore resolves to A or B.
///
/// Everything else is left alone: pixels with no core pixel in reach, and pixels whose
/// colour is not a blend of what surrounds them. The second kind are features too
/// small to contain a flat area (an outline, a 2x2 highlight). Without that test they
/// would be repainted as their surroundings and no later stage could bring them back.
pub fn unmix(ix: &mut IndexedImage, lab: &[[f32; 3]], pal_lab: &[[f32; 3]], radius: usize) {
    let (w, h) = (ix.width(), ix.height());
    let core = core_mask(&ix.labels, lab, w, h);
    let before = ix.labels.clone();
    let r = radius as isize;
    let tolerance_sq = BLEND_TOLERANCE * BLEND_TOLERANCE;
    let mut candidates: Vec<u16> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if before[i] == TRANSPARENT || core[i] {
                continue;
            }
            candidates.clear();
            for ny in (y as isize - r).max(0)..=(y as isize + r).min(h as isize - 1) {
                for nx in (x as isize - r).max(0)..=(x as isize + r).min(w as isize - 1) {
                    let n = ny as usize * w + nx as usize;
                    if core[n] && !candidates.contains(&before[n]) {
                        candidates.push(before[n]);
                    }
                }
            }
            // Sorted, so the result does not depend on the order the window was scanned in.
            candidates.sort_unstable();
            if let Some(label) = blend_endpoint(&lab[i], &candidates, pal_lab, tolerance_sq) {
                ix.labels[i] = label;
            }
        }
    }
}

/// If `colour` lies within the blend tolerance of the segment between two of the
/// candidate colours (a single candidate is a segment of length zero), the candidate
/// at the nearer end of the best such segment. Otherwise `None`: not a blend.
fn blend_endpoint(
    colour: &[f32; 3],
    candidates: &[u16],
    pal_lab: &[[f32; 3]],
    tolerance_sq: f32,
) -> Option<u16> {
    let mut best: Option<(f32, u16)> = None;
    for (n, &a) in candidates.iter().enumerate() {
        let pa = &pal_lab[a as usize];
        for &b in &candidates[n..] {
            let pb = &pal_lab[b as usize];
            let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let len_sq = dist_sq(&ab, &[0.0; 3]);
            let t = if len_sq > 0.0 {
                (((colour[0] - pa[0]) * ab[0] + (colour[1] - pa[1]) * ab[1] + (colour[2] - pa[2]) * ab[2])
                    / len_sq)
                    .clamp(0.0, 1.0)
            } else {
                0.0
            };
            let foot = [pa[0] + t * ab[0], pa[1] + t * ab[1], pa[2] + t * ab[2]];
            let d = dist_sq(colour, &foot);
            if d <= tolerance_sq && best.map_or(true, |(bd, _)| d < bd) {
                best = Some((d, if t <= 0.5 { a } else { b }));
            }
        }
    }
    best.map(|(_, label)| label)
}

/// Opaque pixels whose 3x3 neighbourhood (opaque pixels only, clipped at the border)
/// spans less than [`CORE_FLATNESS`] in OKLab.
fn core_mask(labels: &[u16], lab: &[[f32; 3]], w: usize, h: usize) -> Vec<bool> {
    let mut core = vec![false; labels.len()];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if labels[i] == TRANSPARENT {
                continue;
            }
            let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let n = ny * w + nx;
                    if labels[n] == TRANSPARENT {
                        continue;
                    }
                    for ch in 0..3 {
                        lo[ch] = lo[ch].min(lab[n][ch]);
                        hi[ch] = hi[ch].max(lab[n][ch]);
                    }
                }
            }
            let range = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
            core[i] = dist_sq(&range, &[0.0; 3]) < CORE_FLATNESS * CORE_FLATNESS;
        }
    }
    core
}

/// **Colour-aware sieve.** Every 4-connected region smaller than `min_area` pixels is
/// merged into the **adjacent** label whose palette colour is closest to the region's
/// own mean source colour. Smallest regions first, each seeing the merges before it;
/// a region that an earlier merge has just joined is left for the next pass. Repeated
/// until nothing small is left or [`SIEVE_MAX_PASSES`] is reached.
///
/// Merging into the nearest *surviving* pixel instead (the obvious implementation, and
/// the research's first one) erases any area that is a fine mix of two similar palette
/// colours, such as a dark outline quantized to two near-blacks: every piece is small,
/// all are dissolved, and the nearest survivors are the background.
pub fn sieve(ix: &mut IndexedImage, lab: &[[f32; 3]], pal_lab: &[[f32; 3]], min_area: f64) {
    let (w, h) = (ix.width(), ix.height());
    for _ in 0..SIEVE_MAX_PASSES {
        let as_u32: Vec<u32> = ix
            .labels
            .iter()
            .map(|&l| if l == TRANSPARENT { regions::NO_LABEL } else { l as u32 })
            .collect();
        let comps = regions::connected_components(&as_u32, w, h);

        // Pixels of every small region, grouped by region.
        let mut small: Vec<u32> = (0..comps.area.len() as u32)
            .filter(|&c| (comps.area[c as usize] as f64) < min_area)
            .collect();
        if small.is_empty() {
            return;
        }
        small.sort_by_key(|&c| (comps.area[c as usize], c)); // smallest first, then raster order
        let mut slot = vec![u32::MAX; comps.area.len()];
        for (s, &c) in small.iter().enumerate() {
            slot[c as usize] = s as u32;
        }
        let mut members: Vec<Vec<u32>> = small
            .iter()
            .map(|&c| Vec::with_capacity(comps.area[c as usize] as usize))
            .collect();
        for (i, &c) in comps.id.iter().enumerate() {
            if c != regions::NO_LABEL && slot[c as usize] != u32::MAX {
                members[slot[c as usize] as usize].push(i as u32);
            }
        }

        let mut changed = false;
        let mut neighbours: Vec<u16> = Vec::new();
        for (s, &c) in small.iter().enumerate() {
            let pixels = &members[s];
            let own = ix.labels[pixels[0] as usize];

            let mut mean = [0f32; 3];
            let mut absorbed = false;
            neighbours.clear();
            for &p in pixels {
                let i = p as usize;
                for ch in 0..3 {
                    mean[ch] += lab[i][ch];
                }
                let (x, y) = (i % w, i / w);
                let mut look = |n: usize| {
                    let l = ix.labels[n];
                    if comps.id[n] == c || l == TRANSPARENT {
                        return;
                    }
                    if l == own {
                        absorbed = true;
                    } else if !neighbours.contains(&l) {
                        neighbours.push(l);
                    }
                };
                if x > 0 { look(i - 1); }
                if x + 1 < w { look(i + 1); }
                if y > 0 { look(i - w); }
                if y + 1 < h { look(i + w); }
            }
            // A region merged earlier in this pass now carries this region's label and
            // touches it: the two have become one, possibly large, region. Leave it for
            // the next pass to measure. Merging it onward instead lets two similar
            // colours chase each other (a checkerboard just inverts every pass) while a
            // third colour eats in from the edge.
            if absorbed {
                changed = true;
                continue;
            }
            for ch in 0..3 {
                mean[ch] /= pixels.len() as f32;
            }

            let target = neighbours.iter().copied().min_by(|&a, &b| {
                dist_sq(&mean, &pal_lab[a as usize])
                    .partial_cmp(&dist_sq(&mean, &pal_lab[b as usize]))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            // No differently-coloured neighbour: an island in transparency. Leave it.
            if let Some(label) = target {
                for &p in pixels {
                    ix.labels[p as usize] = label;
                }
                changed = true;
            }
        }
        if !changed {
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// The detail vote and the lost-detail map
// ---------------------------------------------------------------------------

/// How a cell's colour is chosen. Values are part of the WASM and CLI interface.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Vote {
    /// The colour most of the cell's pixels have. Clean; erases anything thinner
    /// than half a cell.
    Majority = 0,
    /// Pixels vote in proportion to how much they differ from their surroundings, so
    /// a thin line or a small highlight can win its cell. Keeps detail that Majority
    /// erases, and turns texture into noise: an alternative, not a default.
    Detail = 1,
}

impl Vote {
    pub fn try_from_u32(v: u32) -> std::result::Result<Self, String> {
        match v {
            0 => Ok(Vote::Majority),
            1 => Ok(Vote::Detail),
            _ => Err(format!("Invalid vote: {}. Valid values: 0 (Majority), 1 (Detail)", v)),
        }
    }

    pub fn from_name(name: &str) -> std::result::Result<Self, String> {
        match name {
            "majority" => Ok(Vote::Majority),
            "detail" => Ok(Vote::Detail),
            _ => Err(format!("Invalid vote '{}'. Valid values: majority, detail", name)),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Vote::Majority => "majority",
            Vote::Detail => "detail",
        }
    }
}

/// Mean source colour of every cell (row-major), `None` for a cell with no opaque pixel.
fn cell_means(labels: &[u16], lab: &[[f32; 3]], w: usize, grid: &Grid) -> Vec<Option<[f32; 3]>> {
    let mut means = Vec::with_capacity(grid.cell_count());
    for row in grid.rows.windows(2) {
        for col in grid.cols.windows(2) {
            let (mut sum, mut n) = ([0f64; 3], 0u32);
            for y in row[0]..row[1] {
                for i in y * w + col[0]..y * w + col[1] {
                    if labels[i] != TRANSPARENT {
                        for ch in 0..3 {
                            sum[ch] += lab[i][ch] as f64;
                        }
                        n += 1;
                    }
                }
            }
            means.push((n > 0).then(|| {
                [(sum[0] / n as f64) as f32, (sum[1] / n as f64) as f32, (sum[2] / n as f64) as f32]
            }));
        }
    }
    means
}

/// **Detail vote** (after Weber et al., "Rapid, Detail-Preserving Image Downscaling",
/// 2016): the label with the greatest total weight wins the cell, where a pixel's
/// weight is the OKLab distance of its source colour from the local average, and the
/// local average is the cell means smoothed over 3x3 cells. A pixel that looks like
/// its surroundings barely votes; one that stands out votes loudly.
///
/// Two consequences worth knowing, both measured:
///
/// * In a flat neighbourhood the average is the neighbourhood's colour, so almost any
///   pixel of another colour outweighs the rest of its cell. That is what keeps a
///   lettering stroke a third of a cell wide, and it is also why texture and leftover
///   speckle come out as noise. (A weight of distance to the power 0.5 or 0.75 was
///   tried: it lost the lettering this vote exists to keep.)
/// * Where neighbouring cells have different colours the average sits between them,
///   every pixel is far from it, and a thin feature no longer stands out: there this
///   vote behaves much like the majority.
///
/// Ties go to the lowest label. Cells with no opaque pixel get [`TRANSPARENT`].
pub fn vote_cells_detail(ix: &IndexedImage, lab: &[[f32; 3]], grid: &Grid) -> Vec<u16> {
    let w = ix.width();
    let (nx, ny) = (grid.cols.len() - 1, grid.rows.len() - 1);
    let means = cell_means(&ix.labels, lab, w, grid);

    // 3x3 binomial smoothing of the cell means, over the cells that exist.
    const KERNEL: [f32; 3] = [1.0, 2.0, 1.0];
    let smoothed = |cx: usize, cy: usize| -> Option<[f32; 3]> {
        let (mut sum, mut weight) = ([0f32; 3], 0f32);
        for (dy, ky) in KERNEL.iter().enumerate() {
            for (dx, kx) in KERNEL.iter().enumerate() {
                let (x, y) = (cx as isize + dx as isize - 1, cy as isize + dy as isize - 1);
                if x < 0 || y < 0 || x >= nx as isize || y >= ny as isize {
                    continue;
                }
                if let Some(m) = means[y as usize * nx + x as usize] {
                    let k = kx * ky;
                    for ch in 0..3 {
                        sum[ch] += k * m[ch];
                    }
                    weight += k;
                }
            }
        }
        (weight > 0.0).then(|| [sum[0] / weight, sum[1] / weight, sum[2] / weight])
    };

    let mut totals = vec![0f64; ix.palette.len()];
    let mut seen: Vec<u16> = Vec::new();
    let mut winners = Vec::with_capacity(grid.cell_count());
    for (cy, row) in grid.rows.windows(2).enumerate() {
        for (cx, col) in grid.cols.windows(2).enumerate() {
            let Some(reference) = smoothed(cx, cy) else {
                winners.push(TRANSPARENT);
                continue;
            };
            for y in row[0]..row[1] {
                for i in y * w + col[0]..y * w + col[1] {
                    let label = ix.labels[i];
                    if label == TRANSPARENT {
                        continue;
                    }
                    if !seen.contains(&label) {
                        seen.push(label);
                    }
                    // The small constant lets a perfectly flat cell still elect its colour.
                    totals[label as usize] += dist_sq(&lab[i], &reference).sqrt() as f64 + 1e-6;
                }
            }
            let winner = seen
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    totals[a as usize]
                        .partial_cmp(&totals[b as usize])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(b.cmp(&a))
                })
                .unwrap_or(TRANSPARENT);
            winners.push(winner);
            for label in seen.drain(..) {
                totals[label as usize] = 0.0;
            }
        }
    }
    winners
}

/// The snap output for the detail vote: one pixel per cell of `grid`, coloured by
/// [`vote_cells_detail`]. A cell is transparent when fewer than half of its pixels are
/// opaque, and fully opaque otherwise. (The majority snap output is the legacy
/// `resample`, which votes on whole RGBA values and is not touched.)
pub fn resample_detail(source: &RgbaImage, quantized: &RgbaImage, grid: &Grid) -> Result<RgbaImage> {
    let ix = index_image(source, quantized)?;
    let lab = source_oklab(source);
    let winners = vote_cells_detail(&ix, &lab, grid);
    let w = ix.width();
    let (nx, ny) = (grid.cols.len() - 1, grid.rows.len() - 1);
    let mut out = RgbaImage::new(nx as u32, ny as u32);
    for (cy, row) in grid.rows.windows(2).enumerate() {
        for (cx, col) in grid.cols.windows(2).enumerate() {
            let winner = winners[cy * nx + cx];
            let area = (row[1] - row[0]) * (col[1] - col[0]);
            let opaque = (row[0]..row[1])
                .map(|y| ix.labels[y * w + col[0]..y * w + col[1]].iter().filter(|&&l| l != TRANSPARENT).count())
                .sum::<usize>();
            if winner != TRANSPARENT && opaque * 2 >= area {
                let c = ix.palette[winner as usize];
                out.put_pixel(cx as u32, cy as u32, image::Rgba([c[0], c[1], c[2], 255]));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{detect_grid, refine_grid};
    use crate::test_util::{recovered_fraction, recovered_pixel_fraction, synthetic_lattice, DISTINCT_8};

    /// Legacy-fitter profiles of an image that is its own quantization.
    fn walk_profiles(img: &RgbaImage) -> Profiles {
        grid::profiles(img, img, &Config::default()).unwrap()
    }
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
            let grid = grid_at_scale(base.clone(), &walk_profiles(&img), &config, scale);
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

    /// Partial alpha kept (`alpha_threshold == 0`): Blocks leaves alpha alone.
    fn blocks_config_keep_alpha(k: usize) -> Config {
        Config { alpha_threshold: 0, ..blocks_config(k, 1.0) }
    }

    #[test]
    fn alpha_is_preserved_pixel_for_pixel() {
        let img = sprite_with_alpha();
        let out = run_pipeline(&img, &blocks_config_keep_alpha(8)).unwrap().image;
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
        let out = run_pipeline(&img, &blocks_config_keep_alpha(8)).unwrap().image;
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

    /// By default every cell is opaque or transparent as a whole, by the half rule.
    #[test]
    fn alpha_is_decided_per_cell() {
        let mut img = sprite_with_alpha();
        crate::binarize_alpha(&mut img, crate::DEFAULT_ALPHA_THRESHOLD);
        let config = blocks_config(8, 1.0);
        let grid = run_pipeline(&img, &config).unwrap().grid.expect("a grid");
        let out = run_pipeline(&img, &config).unwrap().image;
        let mut cells = 0;
        for row in grid.rows.windows(2) {
            for col in grid.cols.windows(2) {
                let area = (row[1] - row[0]) * (col[1] - col[0]);
                let src_opaque = (row[0]..row[1])
                    .flat_map(|y| (col[0]..col[1]).map(move |x| (x, y)))
                    .filter(|&(x, y)| img.get_pixel(x as u32, y as u32)[3] > 0)
                    .count();
                let first = *out.get_pixel(col[0] as u32, row[0] as u32);
                for y in row[0]..row[1] {
                    for x in col[0]..col[1] {
                        assert_eq!(*out.get_pixel(x as u32, y as u32), first, "cell not uniform");
                    }
                }
                let expect_opaque = src_opaque > 0 && src_opaque * 2 >= area;
                assert_eq!(first[3] == 255, expect_opaque, "cell at {col:?}x{row:?}: {src_opaque}/{area} opaque");
                if !expect_opaque {
                    assert_eq!(first.0, [0, 0, 0, 0]);
                }
                cells += 1;
            }
        }
        assert!(cells > 100);
        // The 1 px line is thinner than half of any cell it crosses, so it goes, as it
        // would in the snapped output.
        assert!((4..20).all(|x| out.get_pixel(x, 61)[3] == 0));
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
        let at3 = grid_at_scale(base.clone(), &walk_profiles(&img), &config, 3.0);
        let at2 = grid_at_scale(base.clone(), &walk_profiles(&img), &config, 2.0);
        assert_eq!(at3.cols, at2.cols);
        assert_eq!(at2.cols.len() - 1, 12);
        // 2 px cells cannot be refined at all (every `refine_grid` call returns None):
        // the base grid comes back.
        let cuts: Vec<usize> = (0..=15).map(|i| i * 2).collect();
        let base = Grid { cols: cuts.clone(), rows: cuts, step: (2.0, 2.0), step_estimated: true };
        for scale in [1.5, 2.0, 3.0] {
            assert_eq!(refine_grid(&base, &img, &config, scale), None, "scale {scale}");
        }
        assert_eq!(grid_at_scale(base.clone(), &walk_profiles(&img), &config, 3.0), base);
        assert_eq!(grid_at_scale(base.clone(), &walk_profiles(&img), &config, 1.0), base);
    }

    // ---------------------------------------------------------------- Smooth

    fn smooth_config(k: usize, strength: Strength) -> Config {
        Config {
            k_colors: k,
            no_resize: true,
            cleanup: Cleanup::Smooth,
            cleanup_strength: strength,
            ..Default::default()
        }
    }

    /// Run a stage on an image that is already its own quantization.
    fn staged(img: &RgbaImage, stage: impl FnOnce(&mut IndexedImage, &[[f32; 3]], &[[f32; 3]])) -> RgbaImage {
        let mut ix = ix_from(img);
        let (lab, pal_lab) = (source_oklab(img), palette_oklab(&ix.palette));
        stage(&mut ix, &lab, &pal_lab);
        render(&ix)
    }

    fn lightness(p: &Rgba<u8>) -> f32 {
        srgb_to_oklab(p[0], p[1], p[2])[0]
    }

    #[test]
    fn thresholds_and_radius_follow_the_table() {
        use Strength::*;
        // No pitch, or a pitch under 8 px: absolute thresholds.
        for pitch in [None, Some(2.0), Some(7.9)] {
            assert_eq!(sieve_min_area(Light, pitch), 4.0, "{pitch:?}");
            assert_eq!(sieve_min_area(Medium, pitch), 8.0, "{pitch:?}");
            assert_eq!(sieve_min_area(Strong, pitch), 16.0, "{pitch:?}");
        }
        // From 8 px: a sixteenth, an eighth, a quarter of a cell, never under the floor.
        assert_eq!(sieve_min_area(Light, Some(8.0)), 4.0);
        assert_eq!(sieve_min_area(Medium, Some(8.0)), 8.0);
        assert_eq!(sieve_min_area(Strong, Some(8.0)), 16.0);
        assert_eq!(sieve_min_area(Light, Some(16.0)), 16.0);
        assert_eq!(sieve_min_area(Medium, Some(16.0)), 32.0);
        assert_eq!(sieve_min_area(Strong, Some(16.0)), 64.0);

        assert_eq!(unmix_radius(None), 2);
        assert_eq!(unmix_radius(Some(3.0)), 2);
        assert_eq!(unmix_radius(Some(5.5)), 3);
        assert_eq!(unmix_radius(Some(9.3)), 4);
        assert_eq!(unmix_radius(Some(40.0)), 4);

        // A fallback step is not a pitch.
        let cuts = vec![0, 16, 32];
        let fallback = Grid { cols: cuts.clone(), rows: cuts.clone(), step: (16.0, 16.0), step_estimated: false };
        assert_eq!(estimated_pitch(Some(&fallback)), None);
        let estimated = Grid { step: (9.0, 10.0), step_estimated: true, ..fallback };
        assert_eq!(estimated_pitch(Some(&estimated)), Some(9.5));
        assert_eq!(estimated_pitch(None), None);
    }

    /// The failure that sank the research's first sieve: a dark outline quantized to a
    /// fine mix of two near-blacks, every piece of it "small". It must stay dark and
    /// connected, even at the strongest setting.
    #[test]
    fn sieve_keeps_an_outline_made_of_two_similar_colours() {
        let (white, grey) = (Rgba([245, 245, 245, 255]), Rgba([120, 118, 124, 255]));
        let blacks = [Rgba([12, 10, 22, 255]), Rgba([26, 20, 34, 255])];
        let (w, h, line) = (40u32, 30u32, 18u32..23);
        let mut img = RgbaImage::new(w, h);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = if line.contains(&x) {
                blacks[((x + y) % 2) as usize]
            } else if x < line.start {
                white
            } else {
                grey
            };
        }
        let line_lightness = |im: &RgbaImage| -> f32 {
            let mut sum = 0.0;
            for y in 0..h {
                for x in line.clone() {
                    sum += lightness(im.get_pixel(x, y));
                }
            }
            sum / (h * 5) as f32
        };

        let out = staged(&img, |ix, lab, pal| sieve(ix, lab, pal, sieve_min_area(Strength::Strong, None)));

        for y in 0..h {
            let mut run = 0;
            let mut longest = 0;
            for x in 0..w {
                run = if lightness(out.get_pixel(x, y)) < 0.3 { run + 1 } else { 0 };
                longest = longest.max(run);
            }
            assert!(longest >= 3, "row {y}: the outline is only {longest} px wide");
        }
        let (before, after) = (line_lightness(&img), line_lightness(&out));
        assert!((before - after).abs() < 0.05, "outline lightness {before:.3} -> {after:.3}");
        // The checker pattern itself is speckle and is gone.
        assert_eq!(regions::image_stats(&out).regions_lt4, 0);
        // The flat areas either side are untouched.
        assert_eq!(*out.get_pixel(3, 3), white);
        assert_eq!(*out.get_pixel(35, 20), grey);
    }

    /// The same outline through the whole Smooth sequence. Its near-blacks are no
    /// blend of the white and grey beside it, so un-mixing must leave it alone at any
    /// radius, and the sieve then finishes the job (Codex, Phase 3 review).
    #[test]
    fn unmix_then_sieve_keeps_the_outline_at_every_radius() {
        let (white, grey) = (Rgba([245, 245, 245, 255]), Rgba([120, 118, 124, 255]));
        let blacks = [Rgba([12, 10, 22, 255]), Rgba([26, 20, 34, 255])];
        let mut img = RgbaImage::new(40, 30);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = if (18..23).contains(&x) {
                blacks[((x + y) % 2) as usize]
            } else if x < 18 {
                white
            } else {
                grey
            };
        }
        for radius in [2, 3, 4] {
            let out = staged(&img, |ix, lab, pal| {
                unmix(ix, lab, pal, radius);
                sieve(ix, lab, pal, sieve_min_area(Strength::Strong, None));
            });
            for y in 0..30 {
                for x in 18..23 {
                    assert!(lightness(out.get_pixel(x, y)) < 0.3, "radius {radius}: ({x},{y}) is no longer dark");
                }
                assert_eq!(*out.get_pixel(17, y), white, "radius {radius}");
                assert_eq!(*out.get_pixel(23, y), grey, "radius {radius}");
            }
        }
    }

    /// A 2x2 feature has no flat interior, so it is never "core". Un-mixing must not
    /// repaint it as the background, or Light's promise to keep it is empty.
    #[test]
    fn unmix_then_light_sieve_keeps_a_two_by_two_feature() {
        let field = Rgba([60, 110, 170, 255]);
        let feature = Rgba([200, 40, 40, 255]);
        let mut img = RgbaImage::from_pixel(24, 24, field);
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            img.put_pixel(14 + dx, 14 + dy, feature);
        }
        for radius in [2, 3, 4] {
            let out = staged(&img, |ix, lab, pal| {
                unmix(ix, lab, pal, radius);
                sieve(ix, lab, pal, sieve_min_area(Strength::Light, None));
            });
            assert_eq!(out, img, "radius {radius}");
        }
    }

    /// The sieve's destination is the adjacent label nearest the region's **source**
    /// colour, which is not always the one nearest its current palette colour, and
    /// never a label that is not adjacent, however close.
    #[test]
    fn sieve_merges_by_source_colour_into_adjacent_labels_only() {
        let (dark, mid, light, far) = (
            Rgba([100, 100, 100, 255]),
            Rgba([128, 128, 128, 255]),
            Rgba([210, 210, 210, 255]),
            Rgba([200, 200, 200, 255]),
        );
        // Quantized: dark on the left, light on the right, a 1x3 `mid` region between
        // them, and a block of `far` in a corner touching nothing but `light`.
        // Source: the same, except that the region's pixels are really 200 grey: nearer
        // `light` (and nearest `far`), although their label `mid` is nearer `dark`.
        let (w, h) = (16u32, 12u32);
        let mut quantized = RgbaImage::new(w, h);
        let mut source = RgbaImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let q = if x >= 12 && y < 4 {
                    far
                } else if x == 7 && (4..7).contains(&y) {
                    mid
                } else if x < 8 {
                    dark
                } else {
                    light
                };
                quantized.put_pixel(x, y, q);
                source.put_pixel(x, y, if q == mid { Rgba([200, 200, 200, 255]) } else { q });
            }
        }
        // Sanity: by palette colour the region would go to `dark`.
        let l = |p: Rgba<u8>| lightness(&p);
        assert!((l(mid) - l(dark)).abs() < (l(mid) - l(light)).abs());

        let mut ix = index_image(&source, &quantized).unwrap();
        let (lab, pal_lab) = (source_oklab(&source), palette_oklab(&ix.palette));
        sieve(&mut ix, &lab, &pal_lab, 4.0);
        let out = render(&ix);
        for y in 4..7 {
            assert_eq!(*out.get_pixel(7, y), light, "the region should join `light`");
        }
        assert_eq!(*out.get_pixel(14, 1), far, "the non-adjacent block is untouched");
        assert_eq!(regions::image_stats(&out).unique_colors, 3, "`mid` is gone, `far` is not");
    }

    #[test]
    fn blend_test_accepts_ramps_and_rejects_features() {
        let (black, white, red) = ([0.2f32, 0.0, 0.0], [0.95f32, 0.0, 0.0], [0.6f32, 0.2, 0.1]);
        let pal = [black, white, red];
        let tol = BLEND_TOLERANCE * BLEND_TOLERANCE;
        // On the black-white line: resolves to the nearer end.
        assert_eq!(blend_endpoint(&[0.4, 0.0, 0.0], &[0, 1], &pal, tol), Some(0));
        assert_eq!(blend_endpoint(&[0.7, 0.0, 0.0], &[0, 1], &pal, tol), Some(1));
        // Slightly off the line is still a blend; far off is not.
        assert_eq!(blend_endpoint(&[0.5, 0.03, 0.0], &[0, 1], &pal, tol), Some(0));
        assert_eq!(blend_endpoint(&[0.5, 0.2, 0.1], &[0, 1], &pal, tol), None);
        // A single candidate only claims colours close to itself.
        assert_eq!(blend_endpoint(&[0.22, 0.0, 0.0], &[0], &pal, tol), Some(0));
        assert_eq!(blend_endpoint(&[0.6, 0.0, 0.0], &[0], &pal, tol), None);
        // With three candidates the best-fitting pair decides.
        assert_eq!(blend_endpoint(&[0.72, 0.13, 0.065], &[0, 1, 2], &pal, tol), Some(2));
        assert_eq!(blend_endpoint(&[0.5, 0.0, 0.0], &[], &pal, tol), None);
    }

    #[test]
    fn sieve_strength_decides_what_counts_as_a_speck() {
        let field = Rgba([60, 110, 170, 255]);
        let dot = Rgba([240, 230, 60, 255]);
        let feature = Rgba([200, 40, 40, 255]);
        let mut img = RgbaImage::from_pixel(24, 24, field);
        img.put_pixel(5, 5, dot);
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            img.put_pixel(14 + dx, 14 + dy, feature);
        }
        let run = |strength| staged(&img, |ix, lab, pal| sieve(ix, lab, pal, sieve_min_area(strength, None)));

        let light = run(Strength::Light);
        assert_eq!(*light.get_pixel(5, 5), field, "a 1 px dot goes at Light");
        assert_eq!(*light.get_pixel(14, 14), feature, "a 2x2 feature stays at Light");
        assert_eq!(*light.get_pixel(15, 15), feature);

        let medium = run(Strength::Medium);
        assert_eq!(*medium.get_pixel(14, 14), field, "a 2x2 feature goes at Medium (under 8 px)");

        let strong = run(Strength::Strong);
        assert_eq!(strong, RgbaImage::from_pixel(24, 24, field));
    }

    #[test]
    fn sieve_leaves_islands_in_transparency_alone() {
        let mut img = RgbaImage::from_pixel(9, 9, Rgba([0, 0, 0, 0]));
        img.put_pixel(4, 4, Rgba([200, 10, 10, 255])); // nothing opaque next to it
        let out = staged(&img, |ix, lab, pal| sieve(ix, lab, pal, 16.0));
        assert_eq!(out, img);
    }

    /// Black | two-pixel ramp | white, with a mid-grey in the palette that the ramp
    /// was quantized to. Un-mixing must send the ramp to black and white.
    #[test]
    fn unmix_resolves_a_ramp_to_its_two_sides() {
        let (black, grey, white) = (Rgba([20, 20, 20, 255]), Rgba([128, 128, 128, 255]), Rgba([235, 235, 235, 255]));
        let (w, h) = (24u32, 30u32);
        // Source: rows 0..6 are a flat grey band (so grey is a real colour of the image);
        // below it black on the left, white on the right, blended over columns 11 and 12.
        let mut source = RgbaImage::new(w, h);
        let mut quantized = RgbaImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (s, q) = if y < 6 {
                    (grey, grey)
                } else {
                    match x {
                        0..=10 => (black, black),
                        11 => (Rgba([91, 91, 91, 255]), grey),
                        12 => (Rgba([163, 163, 163, 255]), grey),
                        _ => (white, white),
                    }
                };
                source.put_pixel(x, y, s);
                quantized.put_pixel(x, y, q);
            }
        }
        let mut ix = index_image(&source, &quantized).unwrap();
        let (lab, pal_lab) = (source_oklab(&source), palette_oklab(&ix.palette));
        // Too short a reach sees one side of the ramp only, and one colour cannot
        // explain a blend: nothing changes.
        let before = ix.labels.clone();
        unmix(&mut ix, &lab, &pal_lab, 2);
        assert_eq!(ix.labels, before);

        unmix(&mut ix, &lab, &pal_lab, unmix_radius(Some(6.0)));
        let out = render(&ix);

        for y in 10..h {
            assert_eq!(*out.get_pixel(11, y), black, "row {y}");
            assert_eq!(*out.get_pixel(12, y), white, "row {y}");
            assert_eq!(*out.get_pixel(5, y), black);
            assert_eq!(*out.get_pixel(18, y), white);
        }
        // The real grey area is flat, therefore core, therefore untouched.
        for y in 0..5 {
            for x in 0..w {
                assert_eq!(*out.get_pixel(x, y), grey, "({x},{y})");
            }
        }
    }

    #[test]
    fn unmix_leaves_pixels_with_no_flat_area_in_reach() {
        // Every pixel differs strongly from its neighbours: there is no core anywhere.
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut img = RgbaImage::new(16, 16);
        for p in img.pixels_mut() {
            *p = Rgba([rng.gen(), rng.gen(), rng.gen(), 255]);
        }
        let out = staged(&img, |ix, lab, pal| unmix(ix, lab, pal, 3));
        assert_eq!(out, img);
    }

    #[test]
    fn smooth_removes_speckle_from_a_blurred_lattice() {
        let (img, _) = synthetic_lattice(&[8; 24], &[8; 24], &DISTINCT_8, true, 4, 21);
        let plain = run_pipeline(&img, &Config { cleanup: Cleanup::Off, ..smooth_config(LATTICE_K, Strength::Light) }).unwrap();
        let before = regions::image_stats(&plain.image);
        assert!(before.regions_lt16 > 50);
        let mut previous = before.regions_lt16;
        for strength in [Strength::Light, Strength::Medium, Strength::Strong] {
            let out = run_pipeline(&img, &smooth_config(LATTICE_K, strength)).unwrap();
            assert!(out.grid.is_none(), "Smooth reports no grid");
            let after = regions::image_stats(&out.image);
            assert_eq!(after.regions_lt4, 0, "{strength:?}");
            assert!(after.regions_lt16 <= previous, "{strength:?}: {} > {previous}", after.regions_lt16);
            assert!(after.unique_colors <= LATTICE_K);
            previous = after.regions_lt16;
        }
        assert_eq!(previous, 0, "Strong leaves nothing under 16 px");
    }

    #[test]
    fn smooth_preserves_alpha_stays_in_the_palette_and_is_deterministic() {
        let img = sprite_with_alpha();
        let plain = run_pipeline(&img, &Config { cleanup: Cleanup::Off, ..smooth_config(8, Strength::Light) }).unwrap().image;
        let palette: std::collections::HashSet<[u8; 3]> =
            plain.pixels().filter(|p| p[3] > 0).map(|p| [p[0], p[1], p[2]]).collect();
        for strength in [Strength::Light, Strength::Medium, Strength::Strong] {
            let out = run_pipeline(&img, &smooth_config(8, strength)).unwrap().image;
            assert_eq!(out, run_pipeline(&img, &smooth_config(8, strength)).unwrap().image);
            for (src, dst) in img.pixels().zip(out.pixels()) {
                assert_eq!(src[3], dst[3], "alpha changed");
                if src[3] == 0 {
                    assert_eq!(src, dst);
                } else {
                    assert!(palette.contains(&[dst[0], dst[1], dst[2]]), "{dst:?} is not a palette colour");
                }
            }
        }
    }

    /// With no pitch estimate the thresholds must be the absolute ones. The fallback
    /// step of a 640 px image is 10 px; treated as a pitch it would make Light remove
    /// everything under 6.25 px, 2x2 features included.
    #[test]
    fn smooth_does_not_mistake_the_fallback_step_for_a_pitch() {
        let field = Rgba([70, 120, 90, 255]);
        let feature = Rgba([230, 220, 200, 255]);
        let mut img = RgbaImage::from_pixel(640, 640, field);
        let spots: Vec<(u32, u32)> = (0..9).map(|i| (60 + i * 61, 80 + (i * 137) % 480)).collect();
        for &(x, y) in &spots {
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                img.put_pixel(x + dx, y + dy, feature);
            }
        }
        let grid = detect_grid(&img, &Config::default()).unwrap();
        assert!(!grid.step_estimated, "this image must have no pitch estimate");
        assert_eq!(grid.step, (10.0, 10.0));

        let out = run_pipeline(&img, &smooth_config(2, Strength::Light)).unwrap().image;
        for &(x, y) in &spots {
            assert_ne!(out.get_pixel(x, y), out.get_pixel(x + 5, y + 5), "feature at ({x},{y}) was removed");
        }
        assert_eq!(regions::image_stats(&out).regions, 10);
    }

    #[test]
    fn smooth_handles_images_with_nothing_to_do() {
        let clear = RgbaImage::from_pixel(12, 12, Rgba([3, 2, 1, 0]));
        assert_eq!(run_pipeline(&clear, &smooth_config(4, Strength::Strong)).unwrap().image, clear);
        for (w, h) in [(1, 1), (2, 7), (7, 2)] {
            let tiny = RgbaImage::from_pixel(w, h, Rgba([90, 10, 10, 255]));
            let out = run_pipeline(&tiny, &smooth_config(4, Strength::Strong)).unwrap();
            assert_eq!(out.image.dimensions(), (w, h));
        }
    }

    /// Un-mixing before the vote is what makes Blocks pick one of the two real
    /// colours on a cell boundary: same ramp as above, cut down the middle of it.
    #[test]
    fn blocks_votes_on_unmixed_labels() {
        let (img, _) = synthetic_lattice(&[8; 24], &[8; 24], &DISTINCT_8, true, 4, 21);
        let quantized = run_pipeline(&img, &Config { cleanup: Cleanup::Off, ..blocks_config(LATTICE_K, 2.0) }).unwrap().image;
        let config = blocks_config(LATTICE_K, 2.0);
        let grid = grid_at_scale(
            detect_grid(&quantized, &config).unwrap().with_min_cells(),
            &grid::profiles(&img, &quantized, &config).unwrap(),
            &config,
            2.0,
        );

        let raw = {
            let mut ix = index_image(&img, &quantized).unwrap();
            let winners = vote_cells(&ix, &grid);
            paint_cells(&mut ix, &grid, &winners);
            regions::image_stats(&render(&ix)).regions
        };
        let unmixed = regions::image_stats(&run_pipeline(&img, &config).unwrap().image).regions;
        assert!(unmixed < raw, "voting after un-mix left {unmixed} regions, on raw labels {raw}");
    }

    // ------------------------------------------------------- Detail vote

    /// A light field with a dark vertical line `line_w` px wide starting at `line_x`.
    fn line_on_field(w: u32, h: u32, line_x: u32, line_w: u32) -> (RgbaImage, Rgba<u8>, Rgba<u8>) {
        let (field, line) = (Rgba([225, 220, 200, 255]), Rgba([30, 25, 40, 255]));
        let mut img = RgbaImage::from_pixel(w, h, field);
        for y in 0..h {
            for x in line_x..line_x + line_w {
                img.put_pixel(x, y, line);
            }
        }
        (img, field, line)
    }

    fn even_cuts(limit: usize, step: usize) -> Vec<usize> {
        (0..=limit / step).map(|i| i * step).collect()
    }

    /// A line two fifths of a cell wide: the majority paints it out, the detail vote
    /// keeps it, because its pixels are the ones that differ from their surroundings.
    #[test]
    fn a_thin_line_survives_the_detail_vote_and_not_the_majority() {
        let (img, field, line) = line_on_field(60, 40, 21, 2);
        let grid = whole_image_grid(60, 40, even_cuts(60, 5), even_cuts(40, 5));
        let ix = ix_from(&img);
        let lab = source_oklab(&img);
        let line_cell = 4; // columns 20..25
        let colour = |winners: &[u16], cell: usize| {
            let c = ix.palette[winners[cell] as usize];
            Rgba([c[0], c[1], c[2], 255])
        };

        let majority = vote_cells(&ix, &grid);
        let detail = vote_cells_detail(&ix, &lab, &grid);
        for row in 0..8 {
            let cell = row * 12 + line_cell;
            assert_eq!(colour(&majority, cell), field, "row {row}: majority should lose the line");
            assert_eq!(colour(&detail, cell), line, "row {row}: detail should keep the line");
            // Cells that are all field stay field under both.
            for other in [0, 3, 5, 11] {
                assert_eq!(colour(&detail, row * 12 + other), field, "row {row} cell {other}");
            }
        }
    }

    #[test]
    fn detail_vote_handles_flat_and_empty_cells_and_ties() {
        // A perfectly flat image: every pixel is exactly the local average, and the
        // cells must still elect their colour.
        let flat = RgbaImage::from_pixel(12, 12, Rgba([80, 90, 100, 255]));
        let grid = whole_image_grid(12, 12, even_cuts(12, 4), even_cuts(12, 4));
        let ix = ix_from(&flat);
        assert_eq!(vote_cells_detail(&ix, &source_oklab(&flat), &grid), vec![0; 9]);

        // Cells with nothing opaque have no winner; their neighbours are not disturbed.
        let mut img = RgbaImage::from_pixel(12, 4, Rgba([0, 0, 0, 0]));
        for y in 0..4 {
            for x in 8..12 {
                img.put_pixel(x, y, Rgba([200, 30, 30, 255]));
            }
        }
        let grid = whole_image_grid(12, 4, even_cuts(12, 4), vec![0, 4]);
        let ix = ix_from(&img);
        assert_eq!(vote_cells_detail(&ix, &source_oklab(&img), &grid), vec![TRANSPARENT, TRANSPARENT, 0]);

        // An exact tie goes to the lowest label. The Lab values are given directly
        // (0.25 and 0.75 are exact in binary, and so is their mean), so that the two
        // totals are bit-for-bit equal and only the tie-break can decide.
        let (a, b) = (Rgba([100, 100, 100, 255]), Rgba([140, 140, 140, 255]));
        for first in [a, b] {
            let second = if first == a { b } else { a };
            let mut tie = RgbaImage::new(4, 2);
            for (x, y, p) in tie.enumerate_pixels_mut() {
                *p = if (x + y) % 2 == 0 { first } else { second };
            }
            let ix = ix_from(&tie);
            let lab: Vec<[f32; 3]> =
                ix.labels.iter().map(|&l| [if l == 0 { 0.25 } else { 0.75 }, 0.0, 0.0]).collect();
            let grid = whole_image_grid(4, 2, vec![0, 4], vec![0, 2]);
            assert_eq!(vote_cells_detail(&ix, &lab, &grid), vec![0]);
        }
        // (A lone cell of two colours is always a tie under this vote, whatever their
        // shares: the reference is the cell's own mean, and deviations from a mean
        // balance. It is the neighbouring cells that pull the reference to one side.
        // Here that is checked with a neighbour: the same cell next to a flat cell of
        // label 0's colour is won by label 1, the colour that now stands out.)
        let mut pair = RgbaImage::new(8, 2);
        for (x, y, p) in pair.enumerate_pixels_mut() {
            *p = if x >= 4 || (x + y) % 2 == 0 { a } else { b };
        }
        let ix = ix_from(&pair);
        let lab: Vec<[f32; 3]> = ix.labels.iter().map(|&l| [if l == 0 { 0.25 } else { 0.75 }, 0.0, 0.0]).collect();
        let grid = whole_image_grid(8, 2, vec![0, 4, 8], vec![0, 2]);
        assert_eq!(vote_cells_detail(&ix, &lab, &grid), vec![1, 0]);
    }

    /// Through the whole pipeline, in both output kinds, and with the same alpha rules
    /// as the majority vote. The line runs through cells of two colours, so this also
    /// covers neighbourhoods that are not flat (where the vote is weaker, see
    /// `vote_cells_detail`): a line three tenths of a cell wide still wins.
    #[test]
    fn detail_vote_end_to_end() {
        let (cols, rows) = ([10u32; 20], [10u32; 20]);
        let (mut img, _) = synthetic_lattice(&cols, &rows, &DISTINCT_8[1..3], true, 3, 61);
        // A 3 px dark line through the middle of a column of 10 px cells. It is short
        // (two cells of twenty), so that it does not pull the grid onto its own edges:
        // a line the full height of the image gets cuts of its own, and then even the
        // majority keeps it.
        let line_rows = 90u32..110;
        for y in line_rows.clone() {
            for x in 103..106 {
                img.put_pixel(x, y, Rgba([20, 20, 24, 255]));
            }
        }
        let dark = |p: &Rgba<u8>| p[0] < 80 && p[1] < 80 && p[2] < 80;
        // Share of the line's rows (scaled to the image's height) that show it.
        let rows_with_line = |im: &RgbaImage, x0: u32, x1: u32| {
            let scale = im.height() as f64 / 200.0;
            let (y0, y1) = ((92.0 * scale).ceil() as u32, (108.0 * scale).floor() as u32);
            (y0..y1).filter(|&y| (x0..x1).any(|x| dark(im.get_pixel(x, y)))).count() as f64
                / (y1 - y0) as f64
        };

        for fitter in [crate::grid::GridFitter::Walk, crate::grid::GridFitter::Dp] {
            let blocks = |vote| {
                let config = Config { vote, grid_fitter: fitter, ..blocks_config(8, 1.0) };
                run_pipeline(&img, &config).unwrap().image
            };
            let (majority, detail) = (blocks(Vote::Majority), blocks(Vote::Detail));
            assert!(rows_with_line(&majority, 98, 112) < 0.2, "{fitter:?}: majority kept the line");
            assert!(rows_with_line(&detail, 98, 112) > 0.9, "{fitter:?}: detail lost the line");
            assert_eq!(detail.dimensions(), img.dimensions());

        }

        // The small snap output, on a known grid. (Not through grid detection: the
        // legacy walker moves its cuts onto the line for some palettes, and then the
        // line has cells of its own and every vote keeps it.)
        let (field_img, field, line) = line_on_field(60, 40, 21, 2);
        let grid = whole_image_grid(60, 40, even_cuts(60, 5), even_cuts(40, 5));
        let majority = crate::candidates::snap_image(&field_img, &field_img, &grid, Vote::Majority).unwrap();
        let detail = crate::candidates::snap_image(&field_img, &field_img, &grid, Vote::Detail).unwrap();
        assert_eq!((majority.dimensions(), detail.dimensions()), ((12, 8), (12, 8)));
        for (x, y, p) in detail.enumerate_pixels() {
            assert_eq!(*majority.get_pixel(x, y), field, "snap majority kept the line at ({x},{y})");
            assert_eq!(*p, if x == 4 { line } else { field }, "snap detail at ({x},{y})");
        }

        // Alpha with partial alpha kept: Blocks keeps every pixel's own alpha under
        // either vote.
        let sprite = sprite_with_alpha();
        let keep = Config { vote: Vote::Detail, ..blocks_config_keep_alpha(8) };
        let out = run_pipeline(&sprite, &keep).unwrap().image;
        for (src, dst) in sprite.pixels().zip(out.pixels()) {
            assert_eq!(src[3], dst[3]);
            if src[3] == 0 {
                assert_eq!(src, dst);
            }
        }
        assert_eq!(out, run_pipeline(&sprite, &keep).unwrap().image);
        // By default the detail vote, too, decides alpha per cell.
        let out = run_pipeline(&sprite, &Config { vote: Vote::Detail, ..blocks_config(8, 1.0) }).unwrap().image;
        assert!(out.pixels().all(|p| p[3] == 255 || p.0 == [0, 0, 0, 0]));
    }

    /// The snap output of the detail vote is opaque exactly when at least half of the
    /// cell's pixels have any alpha at all, and then fully opaque.
    #[test]
    fn snap_detail_alpha_is_decided_at_half_the_cell() {
        let (red, clear) = (Rgba([200, 30, 30, 255]), Rgba([9, 9, 9, 0]));
        let solid = Rgba([200, 30, 30, 255]);
        let none = Rgba([0, 0, 0, 0]);

        // 4x4 cells with 16, 6, 7, 8, 9 and 10 of 16 pixels opaque.
        let counts = [16u32, 6, 7, 8, 9, 10];
        let mut img = RgbaImage::from_pixel(4 * counts.len() as u32, 4, clear);
        for (cell, &n) in counts.iter().enumerate() {
            for i in 0..n {
                img.put_pixel(cell as u32 * 4 + i % 4, i / 4, red);
            }
        }
        let cols: Vec<usize> = (0..=counts.len()).map(|i| i * 4).collect();
        let grid = whole_image_grid(img.width() as usize, 4, cols, vec![0, 4]);
        let out = resample_detail(&img, &img, &grid).unwrap();
        assert_eq!(out.dimensions(), (counts.len() as u32, 1));
        for (cell, &n) in counts.iter().enumerate() {
            let want = if n >= 8 { solid } else { none };
            assert_eq!(*out.get_pixel(cell as u32, 0), want, "{n} of 16 opaque");
        }

        // An odd area: 3x3 cells with 4 and 5 of 9 opaque (half is 4.5).
        let mut odd = RgbaImage::from_pixel(6, 3, clear);
        for (cell, n) in [(0u32, 4u32), (1, 5)] {
            for i in 0..n {
                odd.put_pixel(cell * 3 + i % 3, i / 3, red);
            }
        }
        let grid = whole_image_grid(6, 3, vec![0, 3, 6], vec![0, 3]);
        let out = resample_detail(&odd, &odd, &grid).unwrap();
        assert_eq!((*out.get_pixel(0, 0), *out.get_pixel(1, 0)), (none, solid));

        // Faint pixels count as present, and the output alpha is still 255, not theirs.
        let mut faint = RgbaImage::from_pixel(4, 4, clear);
        for i in 0..8u32 {
            faint.put_pixel(i % 4, i / 4, Rgba([200, 30, 30, if i % 2 == 0 { 1 } else { 128 }]));
        }
        let grid = whole_image_grid(4, 4, vec![0, 4], vec![0, 4]);
        assert_eq!(*resample_detail(&faint, &faint, &grid).unwrap().get_pixel(0, 0), solid);
    }

    #[test]
    fn vote_names_round_trip() {
        for vote in [Vote::Majority, Vote::Detail] {
            assert_eq!(Vote::from_name(vote.name()), Ok(vote));
            assert_eq!(Vote::try_from_u32(vote as u32), Ok(vote));
        }
        assert!(Vote::try_from_u32(2).is_err());
        assert!(Vote::from_name("Detail").is_err());
    }
}
