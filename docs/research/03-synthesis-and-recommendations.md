# 03 — Synthesis and recommendations

Three independent lines of work fed this document:

- **Literature and tooling survey** (`01-literature-survey.md`, Claude subagent, web research).
- **Experiments** (`02-experiments.md`, Claude, measured on the images in `other/`).
- **Independent report** (`codex-independent-report.md`, Codex gpt-6.1-sol at xhigh reasoning,
  own code review and web research, written without seeing the other two).

Nothing in `src/` or the frontend was changed. Everything below is a recommendation.

## The short version

1. **Speckle is a missing stage, not a bad quantizer.** All six no-resize algorithms end
   by giving each pixel its nearest palette colour, independently. All three lines of work
   reached this conclusion separately.
2. **The fix for pixel art is "snap in place".** Fit the grid, give every cell one colour,
   paint the cells back at the original size. It is the only approach that yields square
   logical pixels, it reuses machinery the snap mode already has, and it is what every
   comparable tool does.
3. **Resolution is a ladder to audition, not one answer.** The native grid is often one
   step too coarse (it is on the coliseum). Offer the fitted grid and its finer
   neighbours, each in a "clean" and a "detail" flavour, with a score and a lost-detail map.
4. **Do not tile with independent palettes or grids.** It is measurably worse than a
   global palette and it seams. What does help is fitting the *grid* locally while keeping
   one palette and one coordinate system.
5. **Two bugs are worth fixing regardless:** the Spatial quantizer discards its own
   de-speckled result, and Grey Wolf collapses to 5 colours on the coliseum.

## Where the three lines agree

| Finding | Experiments | Codex | Literature |
|---|---|---|---|
| Per-pixel assignment is the root cause; no palette change fixes it | E1: 15–18k fragments for five quantizers | §1.1: all six end in independent assignment | No tool despeckles per pixel; all vote per cell |
| Snap in place is the primary fix | E3, E5, `fig9` | A1, "highest priority" | Idea #2; what every tool does |
| Build colours from cell interiors, not raw pixels | cell-first palette, dE 0.0304 → 0.0291 | A1 "structure vs final palette" | pixel-art-fixer "two-stage packing" |
| Score a grid by within-cell variance | "residual", lattice score (E9) | B5 | pixel-art-fixer "distillability" |
| Too coarse is much worse than slightly too fine | native loses windows, ×2 keeps them | B1: include finer harmonics | pixel-art-fixer states the same asymmetry |
| JPEG's 8 px blocking fakes a grid | E9: cat scores 0.91 periodicity, 0.04 lattice | §1.4: needs a confidence test | pixel-art-fixer removes the JPEG lattice first |
| Independent tiles are the wrong decomposition | E8 | C, §8.1 | §8: seams in every precedent |
| Variable-size pixels are not a good output style | `fig6` looks inconsistent | C4: "explicit option only" | "mixels", a named flaw |

Three separately derived versions of the same scoring idea is the strongest agreement
here: measure how much of the image a piecewise-constant picture on that grid fails to
explain.

## Where they disagree, and how the evidence settles it

**Arbitrary sizes (164, 205, 256).** Codex recommends a geometric ladder of deliberate
sizes, each reconstructed fresh from the source. The literature says only integer ratios
keep block structure and found no crisp method for anything else. The experiments split
the difference: an arbitrary *uniform* grid is poor (256×144 reconstructs worse than the
fitted 193×99), but an arbitrary *target* handed to the elastic fitter works (246×138
and 324×191 both land on real edges, E10). **Resolution:** accept any size as a request,
fit a grid near it, and report where it landed and how well it fits. On a strong lattice
the fit will snap to the real rungs on its own (the sprite sheet lands on 132, 256 or 512
whatever is asked). Codex is right on one point that should be a rule: every candidate is
reconstructed from the source, never by enlarging a smaller candidate.

**Even lattice or snapped cuts.** pixel-art-fixer dropped per-cut snapping because it
cascades into slivers and double-wide cells. unfake and xprite use a bounded DP instead.
E13 tested this directly: on the coliseum a bounded elastic fit beats the *best possible*
even lattice by 29 %, and on the clean sprite sheet the two are identical. **Resolution:**
replace the greedy `walk()` with a bounded DP. The failure pixel-art-fixer describes is a
property of unbounded greedy snapping, which is what `walk()` currently does.

**"Spatial quantization creates speckle by design."** The survey says this about Puzicha's
spatial colour quantization, which is joint quantization *and dithering*. That is true of
that algorithm and not of algorithm 3 in this repo, which is k-means with a
neighbour-agreement bonus. Codex found that its labels are discarded at output; using them
removes 97 % of fragments (E12). Same name, different algorithms.

**Potts/ICM cleanup.** Codex ranks an edge-weighted Potts cleanup highly. Measured, ICM is
the gentlest cleanup but a weak optimiser: at a safe λ it leaves a sixth to a third of the
fragments and is the slowest option tried (E3). A component sieve removes the rest in a
fraction of the time. **Resolution:** sieve first; Potts only if the sieve proves too
blunt, and then as component moves, not pixel moves.

**How to judge detail.** E11 showed that mean colour error against the source *prefers*
the muddier result, because averaging is what minimises it. Any automated loop needs the
separate cleanup and detail measures Codex describes in §10, and ground truth (below).

## Answers to the three questions

### A. Speckle

Cause, measured (E6): 88 % of speckle pixels sit on the blurred borders between logical
pixels. A blend from colour A to colour B passes near an unrelated palette colour C, and
C gets painted as a line of dots. The rest is soft shading inside a cell that straddles a
palette boundary. It grows in proportion to the number of colours.

Two families fix it, and they look different:

| | Blocky: snap in place | Smooth: un-mix + sieve |
|---|---|---|
| What it does | one colour per fitted cell, painted back at full size | blend pixels may only take a colour found in nearby flat areas; then dissolve regions smaller than a fraction of a cell |
| Look | square logical pixels, real pixel art | flat and clean, organic outlines, reads as posterised |
| Needs a grid | yes | only for the size threshold |
| Fragments left (coliseum) | 0 at native, 208 at ×2 | 0 |
| Fails when | the grid is wrong or too coarse; anything thinner than a cell is voted away | small features next to a similar region get fused; never becomes blocky |
| Use for | images with a lattice score well above 0 | images with no lattice (the cat), or as a pre-pass before voting |

They combine: un-mixing first and then voting gave the cleanest ×2 result (208 fragments
against 637 for voting on raw labels).

Dithering is the unsolved part. Everything above flattens a checkerboard. The principle
all three sources agree on is scale separation: variation *inside* a cell is noise,
variation *between* cells is art, so cleanup thresholds belong in cell units and
one-cell features on the cell map are kept by default. No test image with real dithering
was available, so this is untested here. pixel-art-fixer lists dither-locked images as a
known, possibly unfixable, failure of grid detection itself.

### B. Multiple candidates

Recommended behaviour:

1. Fit the native grid (bounded DP). Compute the lattice score. If it is near zero, say
   so and fall back to plain target sizes: this is not recoverable pixel art.
2. Build the ladder from the fit: native, then finer rungs (×2 always; ×1.5 and ×3 when
   the fitter lands on something distinct), plus any size the user types. Deduplicate
   rungs that land on the same grid. Lean fine: offer coarser-than-native only on request.
3. Reconstruct each rung from the source in two flavours:
   - **clean**: majority vote per cell;
   - **detail**: vote weighted by how much each pixel differs from its surroundings (E11),
     which keeps windows and thin lines that a majority erases.
4. Show for each: size, pixel count, fit (residual), and "efficiency" (how many more pixels
   an unaligned grid would need to do as well; it peaks at the native grid).
5. Show a lost-detail map: cells where a 2×2 split would explain much more of the source
   (the quadtree split test from E5). This is the "look here" overlay.

Each rung exists both as a small image for editing and as a full-size snap-in-place
image, so this one mechanism serves the resize mode and the no-resize mode.

On the coliseum the ladder would read: 193×99 (native, loses half-pitch detail),
324×191, 386×198 (the one that looks right), 579×297.

### C. Tiling

The tiny-cars intuition is correct about the problem and wrong about the cure. Codex's
arithmetic makes it plain: four quadrants each reduced to 128×128 stitch into 256×256.
That is the same pixel budget as reducing the whole image to 256×256, which needs no
tiling and has no seams. Tiling adds no capacity; a finer rung does.

What was measured (E8):

- Independent palettes: 2×2 tiles at 16 colours each use 64 colours for dE 0.0149; one
  global 64-colour palette gets 0.0087. Every tile border shows a colour step in flat areas.
- Independent grids: 9 % better fit on the coliseum, 0 % on the clean sprite sheet,
  with sliver cells at tile borders.

The sound versions, in order of value:

1. **Finer or detail-flavoured rungs** (above). This is the actual answer to the cars.
2. **Shared rows, per-band columns**: the 9 % gain with no slivers and no seams, because
   bands are cut along grid lines that already exist. A stepping stone to (3).
3. **Tiles for estimation only**: estimate pitch and phase per overlapping window, smooth
   into one continuous mesh, render once with one palette. High effort; only worth it once
   real failure cases show non-separable drift.
4. **Asset-aware regions**: for sprite sheets, where regions really are separate
   pictures, split on the background colour and treat each sprite as its own image with
   its own grid. This is the one place per-region grids are legitimate.
5. **Global palette from regional representatives**: sample colours per region so a huge
   sky does not starve small objects, then fit one palette.

Mixed pixel sizes in one output ("mixels") are best kept as the lost-detail overlay, not
as a style.

## Proposed build order

Smallest and most certain first. Each step is independently shippable.

| # | Step | Why here | Size |
|---|---|---|---|
| 0 | Use the Spatial quantizer's own labels at output; investigate Grey Wolf's 5-colour collapse | Bugs; step 0a is two lines and removes 97 % of fragments for that algorithm | tiny |
| 1 | Separate palette, label map, grid and rendering in `main.rs` (Codex §9.1) | Every later step needs to mix and match these | medium |
| 2 | **Snap in place** for no-resize: existing grid → cell vote → paint original rectangles, with a grid-scale control (native / ×2 / ×3) | The main fix for A; mostly existing code | small |
| 3 | Replace `walk()` with a bounded DP; profiles in OKLab instead of luma; lattice score with JPEG guard | 29–35 % better fit on the hard image, no change on the clean one | medium |
| 4 | Cell-first colour: vote on structure labels, colour from interior pixels carrying the winning label, palette fitted on cells | Stops accent colours being lost and fringe colours taking palette slots | medium |
| 5 | Audition ladder in WASM + UI: candidates, clean/detail flavours, scores, lost-detail overlay | Problem B; the result cache already keys on settings | medium–large |
| 6 | Smooth mode: un-mix + scale-aware sieve, used automatically when the lattice score is low | Covers non-lattice inputs | small–medium |
| 7 | Evaluation harness: synthetic damage of true pixel art (pixel-bench style) plus the metrics in `experiments/lab.py` | Makes thresholds tunable on more than three images | medium |
| 8 | Per-band columns, then estimated mesh; asset-aware splitting for sprite sheets | Problem C, only after 2–5 expose real failures | large |

Costs at 1 megapixel are all linear in pixels apart from palette fitting: profiles and
cell statistics are single passes, the DP is O(length × step range) per axis, and palette
fitting on ~20–80k cells is roughly two orders of magnitude cheaper than on a million
pixels. The numpy prototypes run each stage in 0.1–1 s unoptimised.

## Open questions

| Question | Experiment |
|---|---|
| Do the thresholds (sieve fraction, flatness 0.03, DP ±25 %, split τ) hold beyond one image? | Step 7 harness on a set of real AI images and synthetic damage |
| Can intentional dithering be protected? | Images with ordered and irregular dither; measure false deletion and false protection separately |
| Is the detail-weighted vote promoting noise elsewhere in the frame? | Full-frame review at λ 0.5 and 1 on several images; only one crop was inspected |
| How often is the native grid one octave too coarse? | Run the ladder on a corpus and record which rung a human picks |
| Does the lattice score threshold separate real cases? | Three images give 0.04, 0.40, 0.52; the gap needs filling |
| Is drift non-separable often enough to justify a mesh? | Compare global, banded and tile-estimated fits on more images |
| How does any of this behave with transparency? | None of the three samples has alpha; the survey notes a vendor claim of alpha fringes in upstream output, unchecked here |
| How does this tool compare with the 2026 crop (pixel-art-fixer, xprite, unfake)? | Run them on the same images; pixel-bench already includes a Pixel Snapper port. Vendor numbers are self-reported and contradict each other |

## Caveats on this research

- Three test images, one of them a negative control. Read every number as "on the
  coliseum" unless stated otherwise.
- The survey's claims carry verification tags; the vendor benchmark figures in it are
  self-reported. One cited paper could not be opened.
- The Codex report's parameter defaults are its own proposals, not measurements; where one
  was tested here (sieve at 0.10, the Spatial fix, the weighted vote) it is noted in
  `02-experiments.md`.
