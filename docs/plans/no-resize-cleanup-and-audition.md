# Plan: speckle cleanup and size audition

Status: in progress on branch `feature/cleanup-and-audition`. See "Progress" at the end. Revised after a review by Codex (gpt-6.1-sol, xhigh):
`docs/reviews/no-resize-cleanup-and-audition-review.md`. All nine findings were accepted.

Evidence for the choices here is in `docs/research/` (`03-synthesis-and-recommendations.md`,
`04-second-batch.md`). This document says what to build, in what order, and how to know
each step worked. It does not repeat the research.

One caution about that evidence. The research prototypes used a different grid pipeline
(OKLab profiles, a dynamic-programming fit) from the one Phases 2–4 build on (the
shipping `walk` on the quantized image). Region counts and candidate sizes quoted from
the research are therefore expectations, not predictions. Each phase remeasures.

## What the user gets

1. **No-resize output without speckle**, in two styles:
   - **Blocks**: one colour per fitted grid cell, painted back at the original size.
   - **Smooth**: blend pixels resolved to neighbouring flat colours, tiny regions merged
     into their most similar neighbour. Keeps soft shading; works without a grid.
2. **Several sizes to audition** instead of one automatic size, in both snap mode (small
   outputs) and no-resize Blocks mode (full-size outputs): the estimated grid plus finer
   rungs, plus a size typed by hand.
3. **A "detail" alternative for each size** that keeps thin features a majority vote
   erases, and an overlay showing where a size is losing detail.
4. Two bug fixes (Spatial quantizer, Grey Wolf).

## Non-goals

| Not doing | Why |
|---|---|
| Per-tile palettes or per-tile processing | Measured worse than one global palette, with seams; adds no pixel budget |
| Mixed cell sizes in one output | Looks inconsistent; survives only as the lost-detail overlay |
| Dither protection | No dithered test image exists. "Cleanup: Off" stays available as the escape hatch |
| Fixing palette starvation (ogtrail's vest turning green at 16 colours) | Real, but a palette problem; separate plan |
| Automatic choice of the "right" size | Six images needed four different answers; the user picks |
| Rewriting the six quantizers to return labels | Not needed: labels are recovered from their output (Phase 1) |
| Changing the default snap output | The legacy path is kept intact and is always the default candidate |
| Potts-style smoothing | Only a constant-weight version was tested and it lost to the sieve. The edge-weighted version Codex proposed is untested; it is an open experiment, not a build item |

## Constraints taken from the code

- `src/main.rs` is 2 912 lines and is both the library root (`[lib] path = "src/main.rs"`,
  `cdylib` + `rlib`) and the CLI. New code goes in new modules (`mod indexed; mod grid;
  mod cleanup; mod candidates;`) declared from `main.rs`. Existing code is not moved.
- Every quantizer returns an `RgbaImage`. The snap branch of `process_image_bytes_common`
  (`main.rs:189`) inlines grid detection, and its voter (`resample`, `main.rs:2404`)
  takes the majority of whole RGBA tuples.
- `walk` (`main.rs:2197`) can place a cut 1 px after the previous one, and its search
  window is never under 2 px whatever the step. It does not guarantee a minimum cell.
- `resolve_step_sizes` falls back to `min(w, h) / 64` when neither axis yields an
  estimate (`main.rs:2116`). That number is not a pitch and must not be used as one.
- WASM exports `process_image(bytes, k)` and `process_image_advanced(bytes, k, no_resize,
  algorithm)`. Both stay, unchanged in behaviour. wasm-bindgen is pinned at 0.2.105.
- `k_colors` has no upper bound in the CLI or the WASM entry points; only the UI slider
  stops at 64.
- Transparency: every quantizer passes `alpha == 0` pixels through untouched.
  `compute_profiles` treats them as black (`main.rs:2019`), so an edge between opaque
  and transparent pixels does contribute to the grid profile.
- The frontend calls WASM on the main thread. The result cache keys on
  `noresize:k:alg` / `snap:k` and can pin exactly one entry (`utils/result-cache.ts`).
- `CLAUDE.md` says the dev server is on 5173; `package.json` says 5170.

## Phase 0 — Bugs and a safety net

No behaviour change except the two fixes.

| Item | Change | Test |
|---|---|---|
| 0a | `spatial_quantize`: the output loop (`main.rs:1368`) recomputes the nearest palette colour per pixel and ignores the optimised `assignments`. Inside the loop, after the existing `alpha == 0` early-continue, use `assignments[coord_to_idx[y * ww + x].unwrap()]` | An image with transparent pixels scattered between opaque ones of distinctive colours: every opaque output pixel equals `palette[assignments[...]]`, transparent pixels are untouched. Plus: on a blurred synthetic lattice, algorithm 3 leaves fewer small regions than algorithm 0 **and** a marked 2×2 feature survives |
| 0b | `gwo_quantize` returns 5 colours for k = 16 on `colliseum.png`. Find out why (suspects: `repair_wolf_palette`, `dedup_palette_oklab`, sample truncation in `build_weighted_samples`). Time-box to half a day; if not found, document and move on | An image with 16 well-separated colour clusters yields ≥ 14 output colours |
| 0c | Golden test for the legacy snap path: output checksums for small committed fixtures under `tests/fixtures/` (crops of `wolf.png`, `colliseum.png`, `ogtrail.png`, ~128 px, one with transparency added) | Guards every later phase against changing shipping snap output |
| 0d | CLI `--stats`: one JSON line with output size, unique colours, 4-connected regions under 4 px and under 16 px. (The lattice score joins it in Phase 5, where the score is implemented.) | Lets a shell loop over `other/*.png` track regressions without Python |
| 0e | `scripts/corpus-check.sh`: run `--stats` over the six images in `other/` for the modes that exist so far and diff against a committed baseline file | The small corpus check, brought forward from Phase 7 |
| 0f | `CLAUDE.md` port note 5173 → 5170 | — |

Done when: `cargo test` passes and `corpus-check.sh` records a baseline.

## Phase 1 — Plumbing (pure refactor)

New types, no new behaviour.

```rust
// src/indexed.rs
pub struct IndexedImage<'a> {
    pub source: &'a RgbaImage,      // original pixels: alpha, hidden RGB, colours for cleanup and scoring
    pub palette: Vec<[u8; 3]>,      // order = first appearance in raster order; empty if nothing is opaque
    pub labels: Vec<u16>,           // palette index per pixel; u16::MAX where alpha == 0
}
pub fn index_image<'a>(source: &'a RgbaImage, quantized: &RgbaImage) -> Result<IndexedImage<'a>>;
pub fn render(ix: &IndexedImage) -> RgbaImage;   // alpha == 0 pixels are copied from source, RGB included

// src/grid.rs
pub struct Grid { pub cols: Vec<usize>, pub rows: Vec<usize>, pub step: (f64, f64),
    pub step_estimated: bool }      // false when resolve_step_sizes used its fallback
pub fn detect_grid(quantized: &RgbaImage, config: &Config) -> Result<Grid>;   // lifted verbatim from the snap branch
pub fn refine_grid(grid: &Grid, quantized: &RgbaImage, config: &Config, scale: f64) -> Option<Grid>;
```

- `index_image` returns an error above 65 535 distinct colours instead of assuming k ≤ 64.
- `detect_grid` is the existing profiles → `estimate_step_size` → `resolve_step_sizes` →
  `walk` → `stabilize_both_axes` sequence, moved, not edited. Images under 3 px on a side
  return an error from it today; callers that do not need a grid must not call it.
- `refine_grid`:
  - Integer scale `f`: each cell of width `c` is split into `min(f, c / 2)` equal parts
    (integer division), so no part is under 2 px.
  - Non-integer scale: re-run `walk` at `step / scale`, then merge any interval under
    2 px into its narrower neighbour, including the first and last cells.
  - Returns `None` when the result has fewer than 1.2× the cells of the grid it refines
    (the rung failed to get finer) or the median cell is under 2 px.
- The snap branch becomes `detect_grid` + `resample`.

Tests: round trip `render(index_image(src, q)) == q` including transparent pixels with
non-zero hidden RGB; all-transparent image; `refine_grid` on a grid with cell widths
`[3, 9, 2, 10, 5]` at scales 2, 3 and 1.5 never yields an interval under 2 px and
returns `None` where it cannot get finer.

Done when: the Phase 0 golden tests still pass byte-for-byte.

## Phase 2 — Blocks (snap in place)

```
quantize (any of the six, full size) → index_image → [un-mix, Phase 3] → detect_grid → refine_grid
  → vote one label per cell → paint each cell rectangle at the original size
```

- **Colour vote**: majority label among the cell's pixels with `alpha > 0`. Partial-alpha
  pixels vote with weight 1.
- **Alpha**: every output pixel keeps its own source alpha. Pixels with `alpha == 0` are
  copied from the source. A cell with no opaque pixel is left as the source. A thin
  opaque feature inside a mostly transparent cell therefore keeps its shape and takes
  the cell's voted colour. (The small snap output keeps the legacy whole-RGBA vote; that
  is a different code path and is not changed.)
- The grid is detected on the selected algorithm's quantized output at `k` colours.
- Config: `cleanup: Cleanup { Off, Blocks, Smooth }` (default `Off`), `grid_scale: f64`
  (default 1.0).
- CLI: `--cleanup=off|blocks|smooth`, `--grid-scale=1|1.5|2|3`.
- WASM: one new export taking an options object, so later phases do not keep widening a
  positional signature.

```rust
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
#[derive(Clone)]
pub struct ProcessOptions { pub k_colors: u32, pub no_resize: bool, pub algorithm: u32, pub cleanup: u32,
    pub cleanup_strength: u32, pub grid_scale: f64, pub vote: u32, pub target_width: u32, pub grid_fitter: u32 }
#[wasm_bindgen] impl ProcessOptions { #[wasm_bindgen(constructor)] pub fn new() -> ProcessOptions { /* defaults */ } }
#[wasm_bindgen]
pub fn process_image_with(input: &[u8], opts: &ProcessOptions) -> std::result::Result<Vec<u8>, JsValue>;
```

  JS builds it with `new ProcessOptions()` and sets fields; a plain object literal will
  not work. Out-of-range values are rejected inside `process_image_with`, not silently
  clamped. The crate's local `Result` alias takes one type parameter, so exports spell
  out `std::result::Result`.
- Frontend: "No Resize" reveals a Cleanup select (Off / Blocks / Smooth) next to the
  algorithm select, and a Grid select (Auto, ×1.5, ×2, ×3) when Blocks is chosen. The
  cache key gains `cleanup`, `strength`, `gridScale`, `vote`, `targetWidth`, `fitter`.

Tests (synthetic, in `cargo test`):

- Lattice recovery: random 8-colour cells at 8 px, box-blurred by 1 px, plus noise.
  Blocks at scale 1 reproduces the true cell map for ≥ 99 % of cells.
- Irregular lattice: cell widths drawn from 6–11 px. Same check at ≥ 95 %.
- Alpha: (a) a 1 px opaque line through otherwise transparent cells is still present and
  still 1 px wide; (b) partial-alpha pixels keep their alpha values exactly; (c) an
  all-transparent image comes back unchanged.
- Output size equals input size; unique colours ≤ k; deterministic.
- JS smoke test: a Node script loads the generated `pkg`, constructs `ProcessOptions`,
  calls `process_image_with` on a fixture and checks the PNG decodes. Run by hand and in
  `corpus-check.sh`.

First task of the phase: run Blocks at scales 1 and 2 on the six images and record
region counts and grid sizes with the real `walk` pipeline. Replace the expectations
below with what is measured if they differ.

Done when, with 16 colours and k-means: `wolf.png` Blocks at scale 1 leaves under 1 % of
today's regions smaller than 16 px (today 26 399) with the ear outline unbroken, and
`colliseum.png` at scale 2 keeps its window rows (compare
`docs/research/figures/full/colliseum_A2_snap_in_place_x2.png`).

## Phase 3 — Smooth (un-mix + colour-aware sieve)

```
quantize → index_image → un-mix → sieve → render
```

- **Un-mix**: a pixel is "core" when the OKLab range of the *source* pixels in its 3×3
  neighbourhood is under `CORE_FLATNESS = 0.03`. Every non-core pixel takes, among the
  labels held by core pixels within radius `r`, the one closest to its own source
  colour; if none is in reach it keeps its label. Cost: one window scan per non-core pixel.
- **Sieve**: 4-connected components by union-find. Each component under `min_area` is
  merged into the **adjacent** label whose palette colour is closest to the component's
  mean source colour; smallest first; repeat until none remain or 6 passes. The
  research's first sieve (nearest surviving pixel) erased the wolf's outline and must
  not be built.
- **Scale**: if `grid.step_estimated`, `pitch` is the estimated step, otherwise there is
  no pitch.

  | | with a pitch ≥ 8 px | otherwise |
  |---|---|---|
  | un-mix radius `r` | `clamp(round(pitch / 3), 1, 3)` | 1 |
  | Light | sieve under max(4 px, cell / 16) | 4 px |
  | Medium | max(8 px, cell / 8) | 8 px |
  | Strong | max(16 px, cell / 4) | 16 px |

  `r` and the cell fractions were tuned on two images; they are named constants and
  Phase 7 retunes them. Default strength is Light: on a non-lattice image Medium already
  changes about 7 % of pixels visibly.
- `cleanup_strength` has three values (Light, Medium, Strong). There is no strength
  "Off": `Cleanup::Off` bypasses both stages.
- Smooth does not need a grid. If `detect_grid` errors (image under 3 px, flat image) it
  runs with the "otherwise" column.
- Blocks runs un-mix first and votes on the un-mixed labels (measured: 208 stray regions
  against 637 on the coliseum at ×2). Both the raw and the un-mixed label maps are kept,
  because the Detail vote of Phase 5 reads the raw one.

Tests:

- Outline regression: a 5 px dark line made of two alternating near-black labels on
  white, with a grey fill on one side. After Strong: every row crossing the line still
  has a dark run at least 3 px wide (connectivity), and the mean OKLab lightness along
  the line is within 0.05 of the input's (contrast).
- A single isolated 1 px stray dot inside a flat region is removed at Light.
- A 2×2 feature survives Light and is removed at Strong.
- With a fallback step on a 1024×1024 noise-free flat-ish image, Light never removes a
  region of 4 px or more.
- Alpha preserved; never introduces a colour outside the palette; deterministic.

Done when: `wolf.png` Smooth/Medium leaves under 2 % of today's regions smaller than
16 px with the ear outline intact (compare `figures/fig14`), in under 1 s natively at 1 MP.

## Phase 4 — DP grid fit (opt-in)

Moved ahead of the audition so the ladder can be compared under both fitters before the
UI is designed around either.

- `grid_fitter: Walk | Dp`, default `Walk`. `Dp` is the bounded dynamic-programming fit
  from the research: steps within ±25 % of the pitch, quadratic penalty on deviation,
  and a cost of 1 per cut **in units of the profile divided by its mean** (so a cut on
  an average column earns nothing). Profiles are OKLab gradient sums over the source.
- Pitch *estimation* stays as it is. Both alternatives tried in research failed on
  something, and the mixed-block case (ogtrail) is unsolved. The ladder is the mitigation.
- Becomes the default only after it wins on a corpus larger than six images.

Tests:

- Soft lattice (blurred, cell widths 4–6 px, with large flat areas): the number of cells
  is within 8 % of the truth. This is the over-segmentation the wolf exposed (235×237
  against about 200).
- Clean regular lattice: grid identical to the truth.
- Minimum interval of 2 px holds after the same merge rule as `refine_grid`.

Done when: with `Dp`, `colliseum.png` residual is at least 25 % under `Walk`'s, the
sprite sheet's grid is unchanged, and `wolf.png` lands within 8 % of `Walk`'s 205×208.

## Phase 5 — Candidates (the audition)

One call returns several rungs, quantizing once.

```rust
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(getter_with_clone)]
pub struct Candidate { pub png: Vec<u8>, pub cols: u32, pub rows: u32, pub grid_scale: f64,
    pub lattice_score: f32, pub is_legacy: bool, pub col_cuts: Vec<u32>, pub row_cuts: Vec<u32> }
#[wasm_bindgen]
pub fn process_candidates(input: &[u8], opts: &ProcessOptions) -> std::result::Result<Vec<Candidate>, JsValue>;
```

  `getter_with_clone` copies `png` on every read: the frontend reads each field once
  into its own object and calls `.free()` on the `Candidate`.

- **The legacy candidate.** In snap mode the first candidate is produced by the existing
  path, untouched: RGB k-means, `walk`, `resample` with its RGBA vote. It is
  byte-identical to `process_image` and is selected by default. No un-mix, no label vote.
- **Other rungs**: `refine_grid` at 1.5, 2, 3, and at the scale that gives `target_width`
  when the user typed one. `None` results are dropped, as are rungs within 5 % of
  another's cell count. Ordered coarse to fine.
- **Lattice score**: `1 − residual(fitted grid) / geometric mean of residual(even grids
  at 0.8× and 1.25× the pitch)`, residuals being mean OKLab distance from per-cell means;
  defined as 0 when the denominator is 0. On six images: 0.52, 0.38, 0.24 for the real
  lattices, 0.10 mixed blocks, 0.09 and 0.02 for the two with no grid. Shown as a label:
  ≥ 0.20 "clear pixel grid", 0.07–0.20 "weak or mixed grid", below "no pixel grid
  found". Provisional. (The plan's first draft used fitted-against-even-grid here; that
  scores a perfectly regular lattice as zero.)
- **When the lattice score is under 0.07**, a typed size is honoured exactly with an
  even grid, since fitting buys 2–7 % there and the user asked for a specific number.
  Otherwise a typed size is a request to the fitter and the UI shows where it landed.
- Each candidate's `png` is the small image in snap mode and the full-size painted image
  in no-resize Blocks mode. The cuts are returned for the overlay and for debugging.
- Frontend: a strip of thumbnails under the controls, each labelled with its size and
  rendered with `image-rendering: pixelated`. Clicking one makes it the result in the
  existing comparison slider and the download. Changing colours or algorithm regenerates
  the strip. The cache gains a pinned *group* (all candidates currently in the strip) in
  place of its single pinned key, so thumbnails cannot be evicted while visible.
- Timing, recorded on a 1 MP image before any optimisation: first strip; change only the
  typed width; change only the vote. If any exceeds about 1.5 s on the main thread, move
  the call to a Web Worker. If the last two are dominated by re-quantizing, introduce a
  session object that holds the quantized labels. Neither is built speculatively.

Tests: ladder deduplicated and ordered; legacy candidate byte-identical to
`process_image` on every fixture including the one with transparency; lattice score on
an aligned regular grid > 0.3, on the same grid offset by half a cell > 0.3, on a
constant image exactly 0, on noise < 0.05; a mixed-block fixture (12 px blocks on one
half, 4 px on the other, with a marked 4 px-wide line) yields at least one candidate
with median cell ≤ 4 px in which the line survives.

First task of the phase: print the ladder for the six images under `Walk` and under
`Dp` and put both tables in this document.

Done when: `ogtrail.png` offers at least three sizes, the finest at least 2.3× the
estimate's width, and the lettering is readable in the finest two.

## Phase 6 — Detail vote and lost-detail overlay

- `vote: Vote { Majority, Detail }`. Detail weights each pixel's vote by its OKLab
  distance from a 3×3-smoothed image of cell means (DPID, λ = 1), reading the **raw**
  label map. It keeps window rows, car outlines and lettering at coarse sizes, and adds
  noise on textured areas, so it is an alternative per candidate, never the default.
- Lost-detail map: per cell, how much more of the source a 2×2 split would explain.
  Rust rasterises it into the coordinates of the candidate's `png` (source coordinates
  for full-size Blocks, `cols × rows` for small output) as an 8-bit image, so the
  frontend only has to draw it translucently over the result behind a toggle.

Tests: a 1-cell-wide dark line on a light field, sampled at half its pitch, survives
Detail and not Majority; the overlay is zero on the synthetic lattice at scale 1 and
non-zero at scale 0.5; on an irregular lattice the overlay's cell boundaries coincide
with the painted cell boundaries.

Done when: on `ogtrail.png` at the estimate's size, Detail keeps "Oregon Trail" readable
(compare `figures/fig19`), and the overlay lights up the lettering and the wagon, not the sky.

## Phase 7 — Evaluation corpus (grows from Phase 0e)

- Synthetic damage of true pixel art: upscale by a non-integer factor, blur, add noise
  and JPEG, then score recovery of the known cell map. Covers lattice recovery, speckle,
  and false removal of dither once dithered originals are added.
- A folder of real AI images with the size a person picked, to learn which rung people
  choose and to tune the lattice-score labels and sieve strengths.
- Open experiment: edge-weighted Potts cleanup against the sieve at matched cleanup
  strength, scored on outline preservation, with Rust timings.

## Order and size

| Phase | Depends on | Rough size | Ships alone |
|---|---|---|---|
| 0 Bugs + safety net | — | 1 day | yes |
| 1 Plumbing | 0 | 0.5–1 day | yes (invisible) |
| 2 Blocks | 1 | 2 days incl. UI | yes |
| 3 Smooth | 1 | 1–2 days | yes |
| 4 DP grid fit | 1 | 1–2 days | yes, opt-in |
| 5 Candidates | 2 (and 4 for the comparison) | 2–3 days incl. UI | yes |
| 6 Detail + overlay | 5 | 1–2 days | yes |
| 7 Corpus | 0 | ongoing | — |

Phases 2, 3 and 4 are independent of each other. 2 → 5 is the path to the audition.

## Decisions for the owner

| # | Decision | Recommendation |
|---|---|---|
| 1 | Default cleanup when "No Resize" is ticked | Keep **Off** for the first release so nothing changes silently; switch to Blocks once tried on real work |
| 2 | Does the candidate strip replace the single automatic result in snap mode, or sit beside it? | Beside it: the legacy result stays the selected candidate, so today's behaviour is the click-free path |
| 3 | Web Worker for WASM calls | Decide from the Phase 5 timings, not up front |
| 4 | Whether to keep the old WASM exports' behaviour forever | Yes; the fork tracks `Hugo-Dz/spritefusion-pixel-snapper` and the exports are its public API |

## Risks

| Risk | Mitigation |
|---|---|
| Thresholds (0.03 flatness, sieve sizes, lattice-score labels, ±25 %) are tuned on six images | Named constants in one place; Phase 7 retunes them; strengths are user-visible controls |
| Research figures came from a different grid pipeline | Each phase remeasures first; "done when" criteria are stated relative to today's output or to the picture, not to prototype numbers |
| Cleanup flattens intentional dithering | Off remains available; Light only removes regions under 4 px; noted in the UI help text |
| The estimate is wrong (ogtrail: 2.6× too coarse) | The ladder always includes finer rungs and a typed width |
| Blocks on an image with no grid produces an arbitrary mosaic | The lattice-score label says so; Smooth is the suggested mode there |
| Main-thread stalls with several candidates | Measure in Phase 5; Worker or session only if the numbers call for it |

## Progress

Each phase is reviewed by Codex (gpt-6.1-sol, xhigh) before it is committed. Reviews are in
`docs/reviews/`.

### Phase 0 — done

- **0a Spatial**: fixed as planned. Regions under 16 px at 16 colours, plain k-means
  against fixed Spatial: coliseum 17 045 / 329, wolf 26 399 / 1 955, ogtrail 9 963 / 815,
  city 34 565 / 2 386.
- **0b Grey Wolf**: root cause found. The fitness added `0.001 / min_pair_distance²` to
  an MSE of about 10⁻⁴–10⁻³, so the penalty was hundreds of times the error and the
  search optimised for spreading centroids apart; most ended outside the image's colours
  and owned no pixels. Three changes: the penalty is now relative to the MSE and only
  active below a just-visible distance; the best palette ever seen is kept (every wolf,
  leaders included, moves each iteration); centroids that own no pixel are re-seeded
  before the final refinement. Colours used at k = 16: coliseum 5 → 16, ogtrail 7 → 16,
  wolf 10 → 16, city → 16.
- **0c Golden tests**: four fixtures in `tests/fixtures/`, hashes recorded from a build
  of commit 07639e1 and cross-checked against the working tree.
- **0d `--stats`**, **0e `scripts/corpus-check.sh`** with a 42-line baseline (six images
  × snap and six no-resize algorithms), **0f** port note.
- New tests were mutation-checked: with either bug put back, the corresponding test fails.
- Codex review (`docs/reviews/phase-0-review.md`): five findings, all fixed. The two that
  mattered: re-seeding could empty a cluster it had already passed, and
  `build_weighted_samples` returned its bins in `HashMap` order, which made Grey Wolf and
  the differentiable quantizer vary in the last float bit between runs (older than this
  phase; now sorted). Golden tests also assert the PNG bytes, not only the pixels.
- Noticed, not acted on: shipping snap gives 169×169 on the sprite sheet, whose true
  grid is 256×256 (pitch 4). Another wrong estimate for the ladder to mitigate.

