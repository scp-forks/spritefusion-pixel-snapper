# 02 — Experiments (Claude)

Empirical side of the no-resize deep dive. Everything here was measured on the sample
images in `other/` with the prototypes in `experiments/` (numpy/scipy, not production
code). `experiments/run_all.sh` reproduces every number.

Test images:

| image | size | what it is |
|---|---|---|
| `colliseum.png` | 1672×941 | AI pixel art, the primary case. Fuzzy logical pixels ≈ 9.3 px |
| `pixel-art-sample-1024x1024.png` | 1024×1024 | AI sprite sheet, clean lattice at exactly 4 px, JPEG-ish noise |
| `cat-1024x1024.png` | 1024×1024 | **Not** pixel art (crayon illustration). Negative control |

Colour distances are Euclidean in OKLab (0.02 ≈ just visible, 0.10 = obviously a different colour).

## Metrics

Computed on the output label map (one label per palette colour):

- **comps** — number of 4-connected same-colour regions.
- **tiny** — regions smaller than ¼ of a logical pixel (pitch²/4). These cannot be an
  intended pixel; this is the speckle count.
- **sub** — regions smaller than ¾ of a logical pixel.
- **tiny_px_pct** — share of the image area inside tiny regions.
- **tv** — share of neighbouring pixel pairs with different labels (label total variation).
- **pure95** — share of grid cells where ≥95 % of pixels have the cell's majority colour.
- **dE_mean / dE_p99 / bad_pct** — fidelity to the source: mean, 99th percentile, and share
  of pixels more than 0.10 away from the source colour.

Speckle metrics alone are gameable (a single flat colour scores perfectly), so every
table also carries fidelity, and the visual crops in `figures/` are the final judge.

## E1 — Baseline: speckle is real and is not about the quantizer

`colliseum.png`, k=16, shipping `--no-resize`:

| algorithm | colours | comps | tiny | tiny px % | pure95 % | dE_mean |
|---|---|---|---|---|---|---|
| 0 k-means OKLab | 16 | 20 379 | 17 676 | 4.38 | 54.7 | 0.0204 |
| 1 median cut | 16 | 17 190 | 14 864 | 3.62 | 57.6 | 0.0224 |
| 2 Wu | 16 | 18 545 | 16 057 | 3.74 | 58.0 | 0.0219 |
| 3 spatial | 16 | 20 450 | 17 776 | 4.38 | 54.6 | 0.0203 |
| 4 grey wolf | **5** | 4 822 | 3 736 | 1.03 | 70.2 | 0.0476 |
| 5 differentiable | 16 | 20 303 | 17 461 | 4.38 | 53.7 | 0.0203 |

For scale: the whole image only has about 19 000 logical pixels. Five of the six
quantizers produce roughly as many *stray fragments* as there are logical pixels, and
only ~55 % of cells come out clean. The spread between algorithms is small (15–18k
fragments). That confirms the starting hypothesis: this is a spatial problem, not a
palette-selection problem. See `figures/fig1_orig_vs_baseline.png`.

Side finding: grey wolf returns 5 colours when asked for 16 on this image. That looks
like a palette-collapse bug (its low speckle is only because it has 5 colours). Not
investigated further.

## E6 — Why it speckles

**Speckle scales with palette size.** Same image, k-means OKLab:

| k | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| tiny fragments | 3 361 | 8 005 | 17 676 | 29 230 | 55 920 |
| tiny px % | 0.94 | 2.11 | 4.38 | 6.67 | 11.82 |

**It is an edge phenomenon, not noise inside flat areas.** Classifying each pixel by
the colour range in its 5×5 neighbourhood:

| | flat (<0.03) | soft (0.03–0.10) | hard edge (≥0.10) |
|---|---|---|---|
| all pixels | 58.4 % | 18.4 % | 23.3 % |
| pixels in tiny fragments | 2.6 % | 9.8 % | **87.6 %** |

In-region noise is tiny (median 0.0015) compared with the spacing between palette
colours (0.03–0.10), so flat interiors are quantized stably. The fragments live on the
blurred transitions between logical pixels.

**Mechanism: blend ramps pass through unrelated palette colours.** The AI output
anti-aliases every logical-pixel border over 2–4 px. A ramp from colour A to colour B
passes through intermediate colours, and with 16 palette entries one of them is often
close to some third entry C. Per-pixel nearest-colour assignment then paints a thin
line or dot of C between A and B. Evidence:

- 46 % of fragment pixels have an assignment margin (distance to 2nd-best minus best
  palette colour) under 0.01, versus 10 % of all pixels. They sit on decision boundaries.
- The mid-tone browns act as "fringe colours": palette entries (120,89,82), (156,92,70)
  and (107,68,64) have 54–64 % of their pixels on edges and 13 % in tiny fragments,
  against 1–2 % for the sky colours.

The second, smaller source is soft shading inside a logical pixel (9.8 % of fragment
pixels): a low-contrast gradient that straddles a palette decision boundary breaks into
dots. This is what shows up as dark flecks inside the arch windows.

More colours make both worse because decision boundaries get denser.

## E2 — The grid in `colliseum.png` is real but elastic

- Global pitch ≈ 9.1–9.4 px (autocorrelation), but the spectrum has several peaks
  between 8.9 and 9.8: the pitch is not constant.
- Per-tile analysis (160 px tiles): row cuts keep a constant phase along each row
  (horizontal grid lines are straight across the image), while the row pitch changes
  from ≈10.0 at the top to ≈9.1–9.4 in the middle and ≈10.0 at the bottom. Column
  cuts are noisier, with low coherence in the sky and the town.
- Detail lives at **half pitch**: thin vertical lines, arch tops and window frames are
  ≈4–5 px wide. A 9.3 px cell cannot hold them. The spectrum shows this as a secondary
  peak at 4.6–4.9 px.

The shipping snap pipeline returns 188×100 for this image, which is the native grid,
and is exactly the "too aggressive" result: `figures/fig4_*` shows windows and lines
being wiped out at that resolution.

### An elastic comb fit beats the greedy walker and beats any uniform grid

`cuts_dp` in `lab.py` is a Viterbi (dynamic-programming) version of the walker: choose
all cuts at once to maximise edge strength under the cuts, with each step constrained
to pitch × (1 ± 0.25) and a quadratic penalty for deviating from the pitch. It is
globally optimal for its objective, where `walk()` is greedy and can lock onto a wrong
edge and never recover.

"Residual" below is the mean dE between the source and a reconstruction where every
cell is filled with its mean colour: how well a piecewise-constant image on that grid
can explain the source.

| grid on `colliseum.png` | cells | residual |
|---|---|---|
| uniform, same cell count | 193×99 | 0.0291 |
| elastic fit, native pitch | 193×99 | **0.0188** (−35 %) |
| elastic fit ×2 subdivision | 386×198 | **0.0124** |
| uniform, same cell count as ×2 | 386×198 | 0.0175 |

An aligned grid at native resolution explains the image about as well as an unaligned
grid with 4× the pixels. Alignment matters more than resolution.

### Lattice confidence: periodicity strength is fooled by JPEG

The cat illustration has no pixel grid, but its gradient profile has a period of
*exactly* 8.0 with autocorrelation 0.91 — stronger than either real pixel-art image.
That is JPEG 8×8 blocking. Any detector that trusts periodicity strength will "find" a
grid there. A score based on the residual is not fooled (E9):

`lattice score = 1 − residual(fitted grid at p) / geomean(residual(uniform grids at 0.8p, 1.25p))`

| image | detected pitch | ACF strength | lattice score |
|---|---|---|---|
| colliseum | 9.25 | 0.42 / 0.60 | **0.40** |
| pixel-art-sample | 4.00 | 0.48 / 0.45 | **0.52** |
| cat (not pixel art) | 8.00 | 0.91 / 0.88 | **0.04** |

## E3 / E7 — Problem A: ways to remove speckle at full resolution

All rows use the same 16-colour palette from the shipping k-means-OKLab output, so only
the spatial strategy differs. `colliseum.png`:

| variant | comps | tiny | tiny px % | pure95 % | dE_mean | bad % | time* |
|---|---|---|---|---|---|---|---|
| baseline | 20 379 | 17 676 | 4.38 | 54.7 | 0.0204 | 0.00 | — |
| median 5×5 prefilter, then quantize | 16 748 | 14 318 | 2.69 | 57.2 | 0.0223 | 0.92 | 2.1 s |
| mode filter 3×3 ×2 | 5 006 | 2 531 | 1.07 | 61.0 | 0.0240 | 1.99 | 0.4 s |
| mode filter 5×5 ×2 | 2 811 | 711 | 0.45 | 62.8 | 0.0252 | 2.59 | 0.4 s |
| Potts ICM λ=0.001 | 8 579 | 6 050 | 2.11 | 59.9 | 0.0207 | 0.01 | 6.6 s |
| Potts ICM λ=0.003 | 5 394 | 2 988 | 1.26 | 62.7 | 0.0211 | 0.04 | 6.6 s |
| dissolve components < ¼ cell | 2 536 | 0 | 0 | 60.5 | 0.0229 | 1.47 | 0.2 s |
| dissolve components < ½ cell | 1 631 | 3 | 0 | 63.3 | 0.0244 | 2.22 | 0.2 s |
| ICM λ=0.001 + dissolve < ¼ cell | 2 470 | 1 | 0 | 62.6 | 0.0225 | 1.08 | 6.9 s |
| un-mix (thr 0.03, r 3) + dissolve < ¼ cell | 2 396 | 0 | 0 | 65.1 | 0.0234 | 1.56 | 1.0 s |
| **snap in place, native grid, majority** | 1 903 | 0 | 0 | 100 | 0.0304 | 5.09 | 0.1 s |
| snap in place, native, interior mean | 2 517 | 0 | 0 | 100 | 0.0300 | 5.01 | 0.1 s |
| snap in place, native, cell-first palette | 2 430 | 0 | 0 | 100 | 0.0291 | 4.84 | 0.4 s |
| **snap in place, ×2 grid, majority of un-mixed labels** | 2 684 | 208 | 0.25 | — | 0.0266 | 3.16 | ~1.2 s |

\* unoptimised numpy on 1.57 MP; indicative of relative cost only.

What the variants are:

- **Prefilter then quantize** (median/bilateral): weak. It blurs the ramps a little but
  they still cross the same decision boundaries.
- **Mode filter**: cheap and effective on dots, but it rounds corners and erodes
  1-px-wide features, and it has no idea which colour is "right".
- **Potts ICM**: minimises colour error + λ × (number of disagreeing neighbours). The
  best fidelity of any cleanup (dE barely moves) but ICM is a weak optimiser: it gets
  stuck and leaves half the fragments at a λ that is still safe. Slowest.
- **Dissolve small components**: every connected region smaller than a threshold is
  absorbed by its surroundings. Removes 100 % of fragments by definition. The threshold
  is naturally expressed in logical-pixel units.
- **Un-mix** (`core_constrained` in `lab.py`): pixels in flat neighbourhoods are trusted
  "core" pixels. Every other pixel (the blend ramps) may only take a colour that some
  core pixel within r px already has, picking the closest. A ramp between A and B
  resolves to A or B and can no longer turn into C. This attacks the mechanism found in
  E6 directly. Edges land at the 50 % point of each ramp.
- **Snap in place**: fit the grid, give each cell one colour, paint it back at full
  resolution. This is the only family that produces *square* logical pixels.

Visual verdict (`figures/fig3`, `fig7`, `fig8`):

- Grid-free cleanup (un-mix or ICM, then dissolve) gives flat, speckle-free regions,
  but shapes keep the organic outlines of the blurry source. It reads as a posterised
  or vectorised image, not as pixel art. It also damages small features when the
  dissolve threshold is too high (`fig8`: a window turns into a blob).
- Snap in place gives the true pixel-art look. On the native grid it loses half-pitch
  detail (windows, thin lines). On the ×2 grid it keeps them (`fig5`,
  `figures/full/colliseum_A2_snap_in_place_x2.png`).
- On the clean sprite sheet, snap in place on the fitted 256×256 grid turns a fringed,
  noisy baseline (16 618 fragments) into textbook pixel art (0 fragments), `fig9`. It
  also deletes the 1-px guide lines between sprites, a reminder that anything thinner
  than a cell is voted away.

### Cell-first palette

Building the palette from per-cell interior means (one sample per logical pixel, border
pixels excluded) instead of from raw pixels improves fidelity at the same k (0.0304 →
0.0291) because no palette slots are spent on blend colours. It changes nothing when
there is no grid.

### How to vote a cell

Majority-of-labels, interior-mean → nearest palette colour, and cell-first palette are
within 4 % of each other on fidelity. Interior mean keeps slightly more regions (it does
not let a large fringe outvote the interior) and needs no prior quantization.
Majority-of-un-mixed-labels gave the cleanest ×2 result (208 tiny fragments vs 637 for
majority of raw labels).

## E4 / E5 / E10 — Problem B: which resolutions to offer

### Arbitrary sizes are bad; fitted sizes are good

| candidate for `colliseum.png` | pixels | residual |
|---|---|---|
| arbitrary uniform 128×72 | 9 216 | 0.0395 |
| arbitrary uniform 164×92 | 15 088 | 0.0350 |
| **fitted native 193×99** | 19 107 | **0.0188** |
| arbitrary uniform 256×144 | 36 864 | 0.0238 |
| arbitrary uniform 320×180 | 57 600 | 0.0198 |
| fitted, target pitch p/1.5 → 324×191 | 61 884 | 0.0127 |
| **fitted ×2 → 386×198** | 76 428 | **0.0124** |
| arbitrary uniform 512×288 | 147 456 | 0.0136 |
| fitted ×3 → 579×297 | 171 963 | 0.0099 |

A plain 256-wide resize is worse than the fitted 193-wide one, and 512-wide is worse
than fitted 386-wide. "Give me 128, 164, 256" as literal sizes is the wrong interface;
"give me the grids that fit, near those sizes" is the right one.

### The ladder falls out of the data

Run the elastic fit at a range of target pitches and see where it lands (E10):

- Clean sprite sheet (true pitch 4): target pitches 6, 5, 4 and 3.2 all land on
  ≈256×256; targets 2.67 to 1.33 all land on ≈512×512; target 8 gives 132×132. The fit
  locks onto the lattice, so the distinct outcomes {132, 256, 512} *are* the candidate
  list.
- Coliseum (soft lattice): 177–193×99 for targets around native, then 246×138, 324×191,
  396×208. Plateaus are weaker because the lattice is weaker.

"Efficiency" = pixels an unaligned grid would need for the same residual ÷ pixels the
fitted grid uses. It peaks at the native grid (3.3× on coliseum, 2.6× on the sample) and
decays on both sides, which makes it a usable ranking signal: the peak is the "optimal"
candidate, and the curve tells how much each extra step buys.

### Integer refinement vs refit

Subdividing the native fit by 2 (386×198, residual 0.0124) and refitting from scratch at
half pitch (396×208, 0.0115) are close. Subdivision keeps exact 2×2 nesting with the
native candidate (useful for an adaptive mode and for comparing candidates); refitting
tracks real half-pitch edges slightly better. The 1.5× rung only works as a refit
(0.0127); building it by subdividing 3 and merging 2 is poor (0.0210), as expected,
since a third of its cells straddle a logical-pixel border.

### Adaptive (quadtree) snap

Keep native cells, split a cell 2×2 only where the split explains at least τ² more
variance per pixel (`adaptive_snap`). On coliseum, τ=0.03 splits 12 % of cells: 26 184
leaves instead of 76 428 at full ×2, with fidelity close to ×2 (dE 0.0267 vs 0.0246).
Visually (`fig6`) the mix of big and small squares looks inconsistent, so this is more
convincing as a **detail-loss map** (the split mask says where the native candidate is
losing something) than as an output style.

## E8 — Problem C: tiling

### Independent palettes per tile: dominated by a global palette, and it seams

| `colliseum.png` | total colours | dE_mean | false seams* |
|---|---|---|---|
| global, k=16 | 16 | 0.0228 | — |
| 2×2 tiles, k=16 each | 64 | 0.0149 | 100 % |
| global, k=64 | 64 | **0.0087** | — |
| 4×4 tiles, k=16 each | 256 | 0.0111 | 100 % |
| global, k=256 | 256 | **0.0050** | — |

\* share of pixel pairs straddling a tile border whose source colours are identical
(<0.01) but whose output colours differ. On non-seam control lines the same statistic is
1–8 %.

For the same total colour budget a single global palette is far more faithful, and
independent tiles put a colour step across every seam in flat areas such as the sky
(each tile picks its own slightly different sky colour). Tiling also does nothing for
speckle: fragments per tile match the per-k numbers.

### Independent grids per tile: a real but modest gain

Residual at native pitch on `colliseum.png`:

| grid | cells | residual |
|---|---|---|
| global uniform | 19 107 | 0.0291 |
| global elastic (separable) | 19 107 | 0.0188 |
| 2×2 tiles, own elastic fit | 20 196 | 0.0179 |
| 4×4 tiles, own elastic fit | 20 723 | 0.0171 |
| 8×8 tiles, own elastic fit | 21 246 | 0.0165 |
| 8 horizontal bands: shared rows, own column cuts | 19 096 | 0.0172 |

Local fits recover another ~9 % (compare: uniform → elastic was 35 %, native → ×2 was
34 %). Independent tiles also add sliver cells along tile borders (the cell count
grows). The banded variant gets the same gain with no slivers because bands are cut on
shared row lines. On the clean sprite sheet local fits gain nothing (<1 %): its lattice
is already global.

So the instinct behind the tiling idea is right in one specific sense — the grid is not
perfectly global and local fitting helps — but cutting the image into independent
quadrants is the weakest way to get it. See the synthesis for the stronger forms.

## E12 — Check of a Codex finding: the spatial quantizer throws its own result away

Codex's independent code review (see `codex-independent-report.md` §1.2) reported that
`spatial_quantize` optimises a spatially regularised label map and then ignores it: the
output loop at `src/main.rs:1368` recomputes the nearest palette colour per pixel. I
confirmed this by reading the code, and it explains why algorithm 3 scored the same as
plain k-means in E1 (20 450 vs 20 379 regions).

To measure what is being lost I patched a scratch copy of the crate (outside the repo) so
the output uses `assignments[i]`. Same image, k=16:

| algorithm 3 | comps | tiny | tiny px % | dE_mean | bad % |
|---|---|---|---|---|---|
| shipping (labels discarded) | 20 450 | 17 776 | 4.38 | 0.0203 | 0.00 |
| labels used (2-line change) | 2 781 | 486 | 0.37 | 0.0222 | 0.89 |

A 97 % drop in fragments from a two-line change. The look (`figures/fig10_*`) is the
grid-free look again: flat and clean, but with rounded, melted shapes, because its 3×3
neighbourhood prior knows nothing about square cells. It rounds small windows into dots.

## E11 — A detail-preserving vote rescues the native-resolution candidate

Also prompted by Codex (its B3: adapt DPID, Weber et al. 2016, to label voting). In each
cell, weight every source pixel by how far its colour is from the smoothed low-res image
(distance^λ), then take the weighted majority of palette labels. Pixels that differ from
their surroundings count for more, so a window that fills 40 % of a cell can win it.

Native grid (193×99), same palette:

| reducer | comps | dE_mean | cells changed vs plain mean |
|---|---|---|---|
| plain mean → nearest colour | 2 524 | 0.0302 | — |
| DPID-weighted mean, λ=1 | 2 877 | 0.0313 | 8.8 % |
| DPID-weighted label vote, λ=1 | 2 065 | 0.0322 | 15.6 % |

By dE it is slightly *worse*; by eye it is clearly better (`figures/fig11_*`, `fig12_*`):
the windows survive as single dark pixels, the pillars are clean, and the muddy
in-between browns that plain averaging produces are gone. Two lessons: (1) every rung of
the ladder should come in a "clean" (majority) and a "detail" (weighted vote) flavour,
and (2) mean dE against the blurry source is not a trustworthy judge of detail
preservation; it rewards averaging.

## E13 — Fairness check: is "elastic beats uniform" just a phase effect?

E2 compared the elastic fit with a uniform grid at phase 0. That is a weak opponent, and
the literature survey found that pixel-art-fixer deliberately uses an *even* lattice
because snapping individual cuts to gradient maxima "cascades into sliver and double-wide
cells". So I searched for the best possible even lattice: pitch (±8 %, non-square allowed)
and phase tuned per axis by brute force.

| `colliseum.png`, native pitch | cells | residual |
|---|---|---|
| even lattice, phase 0 | 181×102 | 0.0323 |
| **best** even lattice (pitch 9.16 × 9.34, tuned phase) | 184×101 | 0.0265 |
| even columns + elastic rows | — | 0.0227 |
| elastic columns + even rows | — | 0.0230 |
| elastic fit (bounded DP, steps within ±25 %) | 193×99 | **0.0188** |

The elastic fit is still 29 % better than the best even lattice, and both axes
contribute. On the clean sprite sheet the two are identical (0.0268), so the bounded DP
costs nothing when the lattice is regular. The cascade failure pixel-art-fixer describes
belongs to unbounded greedy snapping, which is what the shipping `walk()` does (±35 %
window, greedy, with step lengths from 6 to 12 px on this image); a DP with a bounded
step and a deviation penalty does not show it here.

## E14 — Sieve threshold

Codex proposed dissolving components below 0.10 of a cell; E3 used 0.25.

| `colliseum.png` | tiny left (raw labels) | tiny left (after un-mix) | bad % (after un-mix) |
|---|---|---|---|
| sieve < 0.05 cell (4 px) | 4 596 | 1 386 | 1.34 |
| sieve < 0.10 cell (9 px) | 1 992 | 669 | 1.41 |
| sieve < 0.25 cell (21 px) | 0 | 0 | 1.56 |

0.10 is gentler but leaves a visible remainder. The damaged window in `fig8`/`fig13`
survives at either threshold, so that damage comes from the un-mix step fusing a small
dark feature with a dark neighbour, not from the sieve. Grid-free cleanup has no way to
know the feature was supposed to be a rectangle.

## What the experiments say, in one place

1. Speckle is caused by per-pixel assignment of blurred borders and soft shading; it is
   independent of the quantizer and grows with k.
2. Two families fix it. Grid-free cleanup (un-mix + dissolve small components) is robust
   and needs no grid but leaves organic shapes. Snap in place gives square pixels but
   needs a good grid and the right resolution.
3. The grid fit is the lever: an elastic global fit is worth more than doubling
   resolution, and the fit itself tells you which resolutions exist in the image.
4. For this kind of image the native grid is too coarse; the ×2 rung is the sweet spot.
   No single rung is right for every image, which is the case for an audition ladder.
5. Tiling with independent palettes is strictly worse than a global palette. Local
   *grid* fitting has a modest payoff and should be done with shared structure, not
   independent tiles.
6. The spatial quantizer (algorithm 3) already computes a de-speckled label map and
   discards it. Using it removes 97 % of fragments, in the grid-free style.
7. At a fixed resolution, a detail-weighted vote keeps small features that a plain
   majority or mean erases.

## Limits of this evidence

- Three images, one of which is a negative control. Every threshold above (¼ cell,
  0.03 flatness, τ, ±25 % step tolerance) is tuned on essentially one image.
- No image with intentional dithering was available. Dissolve, mode filter and cell
  voting will all flatten a checkerboard dither; none of the "protect dither" ideas in
  the synthesis has been tested.
- Fidelity is measured against the blurry source, which is itself not the target. A
  perfect cleanup has non-zero dE. Use it for relative comparisons only.
- Timings are numpy, single-threaded, unoptimised.
