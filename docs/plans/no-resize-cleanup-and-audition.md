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

### Phase 1 — done

- `src/indexed.rs` (`IndexedImage`, `index_image`, `render`) and `src/grid.rs` (`Grid`,
  `detect_grid`, `refine_grid`). The snap branch is now `detect_grid` + `resample`; the
  golden tests and the corpus check pass unchanged.
- `detect_grid` calls the existing functions in `main.rs` in the existing order; they
  were not moved into the module, only the call sequence was.
- Both modules carry `#![allow(dead_code)]` until Phase 2 gives them callers.
- Deviation: `refine_grid` enforces the 2 px minimum on **every** cell (the merge rule
  runs on integer subdivisions too, because the legacy walker can hand over 1 px cells),
  so the plan's "median cell under 2 px" rejection became redundant and was dropped.
- Codex review (`docs/reviews/phase-1-review.md`): three findings, all fixed (surviving
  1 px cells, an allocation sized by an uncapped scale, an untested 1.2× threshold).
- **Found: the legacy estimator cannot see crisp pixel art.** `estimate_step_size`
  requires a strict local maximum in the gradient profile. A perfectly sharp edge gives
  two equal adjacent samples, so nearest-neighbour-upscaled art produces no estimate on
  either axis and falls back to `min(w, h) / 64`. Checked on the real binary: the wolf's
  205×208 snap output, upscaled ×5 with nearest neighbour, snaps back to 68×70. This is
  shipping behaviour, pinned by the test `crisp_edges_defeat_the_legacy_step_estimate`,
  and is **not fixed here** because the legacy path is frozen. It is the strongest
  argument so far for the opt-in fitter of Phase 4, which should carry a plateau-tolerant
  estimate; whether to also fix the legacy default is a decision for the owner.

### Phase 2 — done

- `src/cleanup.rs`: `blocks`, `vote_cells`, `paint_cells`, `grid_at_scale`. Config gained
  `cleanup` and `grid_scale`; CLI `--cleanup=off|blocks`, `--grid-scale=`; WASM
  `ProcessOptions` + `process_image_with`; frontend Cleanup and Grid selects, shown only
  where they have an effect. `scripts/wasm-smoke.mjs` runs the generated package under
  Node and is part of `corpus-check.sh`. Checked in the running app with Playwright.
- **Remeasured with the real pipeline** (16 colours, k-means), as the plan required:

  | image | grid at scale 1 | regions < 16 px, plain → Blocks | grid at scale 2 |
  |---|---|---|---|
  | wolf | 187×187 | 26 399 → 68 (0.3 %) | 363×352 |
  | coliseum | 187×99 | 17 045 → 0 | 374×198 |
  | ogtrail | 200×116 | 9 963 → 158 | 362×211 |
  | city | 109×119 | 34 565 → 508 | 196×202 |
  | sprite sheet | 251×253 | 24 227 → 918 | 452×464 |
  | cat | 228×223 | 185 376 → 1 958 | 382×391 |

  No output has a region under 4 px. At scale 2 a single cell is often under 16 px, so
  the "under 16 px" count stops meaning speckle there. Blocks runs in 0.1–0.5 s natively.
- Both "done when" checks hold: wolf at scale 1 leaves 0.3 % of the small regions with the
  ear outline unbroken, and the coliseum at scale 2 keeps its window rows.
- **The grid depends on which quantized image it is detected on.** Blocks detects on the
  selected algorithm's output, as planned; snap mode detects on its own RGB k-means. On
  the wolf that gives 187×187 against 205×208, and on the sprite sheet 251×253 (right)
  against 169×169 (wrong). Neither input is consistently better. The fitter of Phase 4
  works on the source image and removes the dependence.
- **The legacy grid is about half a pixel off on average.** On a synthetic blurred
  lattice, Blocks puts 90 % of pixels right (75 % for the plain reduction): every cell
  has the right colour, but about four cuts in ten sit one pixel early, because the
  profile's gradient is centred on a pixel while an edge lies between two. The test
  `regular_blurred_lattice_is_recovered` pins the 90 % as a floor; Phase 4 should raise it.
- Deviations from the plan:
  - `ProcessOptions` has only the fields that do something so far (`k_colors`,
    `no_resize`, `algorithm`, `cleanup`, `grid_scale`). The others arrive with their
    phases, each defaulting to "no change".
  - `grid_scale` also applies in snap mode (a finer small output). It was nearly free and
    Phase 5 needs it.
  - A scale that cannot be honoured falls back to the nearest smaller standard scale
    (3 → 2 → 1.5 → 1) instead of failing; `--stats` reports the grid actually used.
  - Lattice tests ask for 12 colours on an 8-colour lattice: with exactly 8, k-means
    spends slots on blend colours and merges two real ones, which no spatial step can
    undo. This is the "fringe colours take palette slots" effect from the research.
  - The irregular lattice test passes at the planned 95 %; a `grid_scale = 2` test from
    the first draft of the plan was kept.
  - Numeric `ProcessOptions` fields are `f64`, not `u32`: wasm-bindgen's `u32` conversion
    happens before any validation and turns NaN into 0 and 1.5 into 1.
- Codex review (`docs/reviews/phase-2-review.md`): six findings, all fixed. The two that
  mattered: Blocks voted on the detected grid's 1 px cells (a 1 px checkerboard came back
  untouched), and invalid JS numbers were coerced into valid options before validation.

### Phase 3 — done

- `cleanup.rs`: `smooth`, `unmix`, `sieve`, `Strength`, `sieve_min_area`, `unmix_radius`.
  CLI `--cleanup=smooth`, `--strength=light|medium|strong`; `ProcessOptions.cleanup_strength`
  (1–3); a Strength select in the app, shown only for Smooth. Blocks now un-mixes before
  it votes.
- Regions under 16 px at 16 colours, k-means, measured from a fresh build of the final code:

  | image | plain | Light (< 4 px removed) | Medium (< 8) | Strong (< 16) |
  |---|---|---|---|---|
  | wolf | 26 399 | 2 932 | 853 | 0 |
  | coliseum | 17 045 | 1 355 | 286 | 0 |
  | ogtrail | 9 963 | 2 529 | 869 | 0 |
  | city | 34 565 | 6 698 | 2 500 | 0 |
  | sprite sheet | 24 227 | 7 261 | 2 675 | 0 |
  | cat | 185 376 | 22 955 | 6 105 | 0 |

  Nothing under 4 px survives any strength. 0.13–0.78 s per image natively. The wolf's
  ear outline is intact at all three.
- **The "done when" is not met as written, and was the wrong test.** It asked for under
  2 % of the wolf's small regions at Medium (528); the result is 853 (3.2 %). Medium is
  defined as removing regions under 8 px, and those 853 are all 8–15 px. The 2 % came
  from the research prototype, which reached it by removing things it should not have
  (next two points). What is now promised and tested: at each strength, nothing under
  that strength's threshold survives; Strong leaves nothing under 16 px.
- **Flaw 1 in the research's sieve, found by the outline test.** On an outline made of a
  perfect checkerboard of two near-blacks the sequential sieve never converged: each
  pixel merged into its still-unprocessed neighbour, so the checkerboard inverted every
  pass while the grey beside it ate in one column at a time until the outline was gone.
  Fix: a small region that an earlier merge in the same pass has just joined is left for
  the next pass, when the two are measured as one region.
- **Flaw 2, in the research's un-mix, found by Codex.** "Take the nearest label that any
  nearby flat area has" also repaints small features that have no flat interior of their
  own: the outline's outer columns, a 2×2 highlight. Fix: a pixel is re-labelled only if
  its source colour is a blend, within 0.05 OKLab of the line between two colours carried
  by nearby flat pixels. This is why the research found un-mix "fusing a small dark
  feature with a dark neighbour" (E14).
- Un-mixing before the vote still helps Blocks: wolf 68 → 51 regions under 16 px at
  scale 1 and 2 707 → 2 018 at scale 2; coliseum at scale 2, 85 → 53.
- Deviations from the plan:
  - The un-mix reach is `clamp(round(pitch / 3), 1, 3) + 1`, or 2 with no pitch. The
    extra pixel is needed because a blend must see flat pixels on both sides, and a flat
    pixel is at least one pixel away from any ramp. The 8 px gate applies only to the
    cell-relative sieve thresholds.
  - The sieve runs up to 32 passes, not 6.
  - Smooth reports no grid in `--stats` even when one was detected for its pitch.
  - Keeping the raw label map next to the un-mixed one is deferred to Phase 6, which is
    the first reader of it.
- A mistake of mine, caught by Codex: the first version of the table above was measured
  with a CLI binary that predated the sieve fix (`cargo test` does not rebuild it), and
  disagreed with the corpus baseline. All figures here are from a fresh build.
- Codex review (`docs/reviews/phase-3-review.md`): four findings, three fixed, one deferred.

### Phase 4 — done

- `grid.rs`: `GridFitter { Walk, Dp }`, `edge_profiles`, `dp_cuts`, a shared
  `profiles` / `detect` / `refine` API used by snap mode, Blocks and Smooth, and
  `residual`. CLI `--grid-fitter=walk|dp`; `ProcessOptions.grid_fitter`; a "Grid fit"
  select in the app (Standard / Elastic (beta)), hidden only for plain palette reduction.
  `--stats` gained `fitter` and `residual`. The default is `Walk`; all 78 legacy lines
  of the corpus baseline are unchanged and the golden tests pass.
- Results at 16 colours, snap mode (grid, residual of the source on that grid):

  | image | Walk | DP |
  |---|---|---|
  | wolf | 205×208, 0.0134 | 206×203, 0.0128 |
  | coliseum | 188×100, 0.0214 | 183×99, 0.0195 |
  | ogtrail | 181×103, 0.0360 | 196×112, 0.0323 |
  | city | 111×120, 0.0695 | 100×95, 0.0731 |
  | sprite sheet (true grid 256×256) | 169×169, 0.0682 | **256×256**, 0.0268 |
  | cat | 228×219, 0.0281 | 256×256, 0.0265 |
  | wolf's snap output upscaled ×5 and ×4, nearest neighbour (true grid 205×208) | 68×70, 0.036 | **205×208, 0.0000** |

  Same speed as Walk (0.4–0.5 s per megapixel image, natively).
- "Done when", item by item:
  - *Sprite sheet's grid unchanged*: met, and better than that. DP gets the true 256×256
    where Walk gets 169×169.
  - *Wolf within 8 % of Walk's 205×208*: met (206×203).
  - *Coliseum residual at least 25 % under Walk's*: **not met: 9 %** (0.0195 against
    0.0214). The 25 % was never a fair target. It came from the research's "29–35 %
    better", which compared the elastic fit with *uniform* grids and was then misquoted,
    by me, as an advantage over the walker. Corrected in `02-experiments.md` and `03`.
- Where the DP fitter earns its place instead:
  - **Crisp pixel art.** Its profile is the difference between adjacent columns, which
    peaks on the cut itself, so the step estimate works where the legacy one sees a
    two-sample plateau and gives up. This is the fix for the bug found in Phase 1.
  - **Cut placement.** On the synthetic blurred lattice, Blocks puts 100.0 % of pixels
    right with DP against 90.7 % with Walk (test `dp_grid_puts_cuts_on_the_edges`).
  - **Cell count.** 40×40 on every synthetic soft lattice tried, where Walk is off by one
    on half of them.
- Three things the research prototype did not have, each added because a measurement
  failed without it:
  1. *Edge gains against the local median.* The prototype's "cost 1 per cut" (gain
     measured against the global mean) makes cuts in flat areas cost, and stretches flat
     areas by an eighth: the crisp wolf came back 197×204. Measured against a local mean
     it over-corrects (216×214), because the mean is raised by the very edges being
     measured. Against the local median the count is exact (205×208).
  2. *Pitch search.* The step estimate is a median of whole-pixel spacings, so at 5 px it
     is off by up to 10 %, and missing peaks pull it higher (6.0 on the wolf). The fit is
     run at 37 pitches from 0.8× to 1.25× the estimate and the best total score kept.
     On synthetic lattices the pitch it settles on is within 5 % of the true mean cell.
     (An earlier attempt, re-estimating the pitch from the fitted cuts and iterating,
     drifted steadily towards smaller cells on every image without a crisp lattice.)
  3. *×1.5 is not a size, it is a search.* A clean lattice has nothing at two thirds of
     its pitch. The re-fit goes wherever the edges and the step bounds let it: on 8 px
     cells that is the ×2 grid, on 3 px cells it is the original cuts, which is not finer,
     so `refine` returns `None` and the rung is dropped. On images with sub-cell detail it
     finds intermediate structure. Integer scales subdivide and always work.
  4. *Borders inside the optimisation.* The prototype let the first and last cut float
     and then deleted border slivers, which left a grid the optimiser had not chosen and
     a score that did not belong to it. Cuts now run from 0 to the limit by construction,
     and a brute-force test checks the result is optimal.
- Known limits, not addressed:
  - The step estimate (shared with Walk) ignores peaks closer than 4 px, so lattices
    under about 4 px are estimated too coarse; the search's lower bound of 0.8× recovers
    only part of that (a 3.5 px synthetic lattice comes out at 4.0).
  - On the wolf the right answer is not knowable: its stair-steps measure anywhere from
    4 to 6 px. Both fitters are defensible there.
  - The city (no lattice) gets a slightly coarser grid from DP (100×95) than from Walk
    (111×120). Neither means anything; the lattice score of Phase 5 is what should say so.
  - Interior steps are now really within ±25 % of the pitch (they were ±40 % at small
    pitches before the review). The research saw steps of 6 to 12 on the coliseum at a
    pitch of 9.3; the fit now allows 7 to 11 there.
- Codex review (`docs/reviews/phase-4-review.md`): six findings, all fixed. The table
  above was remeasured from a fresh build afterwards.
- Decision for the owner, unchanged from Phase 1 but now with evidence: the legacy
  default cannot measure crisp pixel art (68×70 instead of 205×208). DP fixes that and
  is opt-in. Making it the default would change every existing output.

### Phase 5 — done

- `src/candidates.rs`: `build` (every candidate for an image), `ladder`,
  `grid_for_width`, `requested_grid`, `lattice_label`. `grid.rs`: `LabImage` with
  `residual` and `lattice_score`, `even_grid`, `grid_for_cols`. `cleanup.rs`:
  `prepare_blocks` / `BlocksPrep::render`, so Blocks quantizes and un-mixes once and
  renders on any number of grids. WASM `process_candidates` returning `Candidate`s;
  `ProcessOptions.target_width`. CLI `--candidates`, `--target-width=`; `--stats` gained
  `lattice_score`.
- Frontend: a "Sizes" strip of thumbnails under the controls in snap mode and Blocks,
  with the lattice verdict and an "Add width" box; clicking a size puts it in the
  comparison view and the download. The Grid select from Phase 2 is gone (the strip
  replaces it). The cache now holds sets of results and pins the whole set on screen.
  Changing colours, algorithm or grid fit keeps the size the user was looking at (the
  nearest within a tenth).
- **The ladders** (first task of the phase), 16 colours. `*` marks the legacy result.

  Snap mode:

  | image | Walk | DP | lattice score, Walk / DP |
  |---|---|---|---|
  | wolf | *205×208, 283×290, 392×380, 461×461 | 206×203, 309×304, 412×406, 449×451 | 0.12 / 0.18 weak |
  | coliseum | *188×100, 301×178, 374×198, 561×297 | 183×99, 286×166, 366×198, 549×297 | 0.30 / 0.37 clear |
  | ogtrail | *181×103, 252×143, 337×191, 427×246 | 196×112, 288×166, 392×224, 425×233 | 0.03 none / 0.08 weak |
  | city | *111×120, 171×158, 196×203, 229×228 | 100×95, 147×143, 200×190, 228×230 | 0.05 / 0.04 none |
  | sprite sheet | *169×169, 260×263, 338×337, 428×426 | 256×256, 447×441, 509×511 | 0.04 none / 0.52 clear |
  | cat | *228×219, 330×319, 390×388, 445×452 | 256×256, 345×343, 512×511 | 0.01 / 0.01 none |

  Blocks (full-size output; grid shown):

  | image | Walk | DP |
  |---|---|---|
  | wolf | 187×187, 255×271, 363×352, 472×462 | 206×203, 309×304, 412×406, 449×451 |
  | coliseum | 187×99, 306×180, 374×198, 561×297 | 183×99, 286×166, 366×198, 549×297 |
  | ogtrail | 200×116, 278×156, 362×211, 438×238 | 196×112, 288×166, 392×224, 425×233 |
  | city | 109×119, 168×154, 196×202, 231×229 | 100×95, 147×143, 200×190, 228×230 |
  | sprite sheet | 251×253, 452×464 | 256×256, 447×441, 509×511 |
  | cat | 228×223, 335×321, 382×391, 441×454 | 256×256, 345×343, 512×511 |

  The "×3" rung is rarely three times as wide: a 5 px cell can only be halved.
- "Done when": met. ogtrail offers four sizes, the finest 2.36× the estimate's width,
  and the lettering reads "Oregon Trail Mix" in the three finest (252, 337, 427 wide);
  at the estimate (181) it reads "Dragon Trail Mr".
- **The lattice score depends on the grid as well as the image.** It is high only when
  the fitted grid has found the lattice. With the legacy fitter the sprite sheet scores
  0.04, "none", because the grid it found (169×169) is wrong; with DP it scores 0.52.
  The wolf scores 0.12 to 0.18, "weak", lower than the research's 0.24, which was
  measured on a different grid. The labels (clear from 0.20, none under 0.07) are
  provisional and the wolf shows the upper one may be too strict.
- **Timing** (plan: measure before deciding on a Worker or a session object). Longest
  main-thread task in Chromium on this machine, in milliseconds:

  | action | wolf, 1.05 MP | coliseum, 1.57 MP |
  |---|---|---|
  | snap mode, first strip | 664 | 1 010 |
  | snap mode, type a width | 689 | 1 035 |
  | snap mode, switch to the elastic fitter | 842 | 1 273 |
  | Blocks, first strip (4–5 full-size images) | 1 273 | 1 915 |
  | Blocks, type a width | 1 184 | 1 906 |
  | Smooth | 978 | 1 545 |

  Natively a whole ladder takes 0.5 to 0.9 s. At 1 MP nothing exceeds the plan's 1.5 s,
  so by the plan's own rule no Worker was built. At 1.6 MP Blocks does (1.9 s). Typing a
  width costs a full run, because the image is re-quantized each time; a session object
  holding the quantized image would make it nearly free. Neither was built: the plan
  said not to build them speculatively, and this is the owner's decision 3.
- Deviations from the plan:
  - *One lattice score per image*, measured on the detected grid and repeated on every
    candidate, not one per candidate. It is a statement about the image; per-rung
    values would cost three more passes each and mean little.
  - *Typed widths under the legacy fitter re-divide the detected grid* instead of
    re-fitting. Both of the legacy code's own ways of fitting at a given pitch drift
    badly at a pitch the image has no edges for (61 and 35 columns when asked for 48 on
    a clean 24-cell lattice). DP re-fits, as planned.
  - *Deduplication of typed widths.* Where no pixel grid was found the typed width is
    exact and is always offered (unless it is an existing rung cut for cut). Where one
    was found, a rung within a twentieth of the typed width is treated as the answer and
    no second, near-identical rung is added.
  - *A typed width is refused, not clamped,* when it would need cells under 2 px.
  - `target_width` also works in the single-result calls, where it takes the place of
    `grid_scale`.
  - `Candidate` carries `width`, `height` and `unique_colors` in addition to the planned
    fields, so the frontend no longer decodes every result to count its colours.
  - The candidates follow the selected fitter. With DP selected there is no legacy
    candidate in the strip; switching the Grid fit select back gives it.
  - `process_candidates` is used for every mode; modes without a grid return one result
    and the strip hides itself.
- The frontend gained unit tests (`spritefusion-frontend/tests`, `npm test`, 14 tests on
  Node's built-in runner): the result cache, the selection rule, and the settings keys.
  `scripts/corpus-check.sh` runs them.
- Codex review (`docs/reviews/phase-5-review.md`): four findings, all fixed. The two
  that mattered: a typed width could vanish from the strip when it was close to another
  size, and a large set arriving could be evicted from the cache before it was shown.

