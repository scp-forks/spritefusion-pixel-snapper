# 05 — Evaluation on generated pixel art (Claude)

Phase 7 of `docs/plans/no-resize-cleanup-and-audition.md`. Every earlier number in this
folder comes from six real images whose right answer nobody knows. This file measures
the tool on images whose right answer is known exactly, because they are generated:
true pixel art, then damaged the way image generators and upscalers damage it.

The harness is `src/eval.rs` (test-only code in the crate, no new dependency):

```bash
cargo test --release                                             # a small fixed subset, as a regression guard
cargo test --release eval_report  -- --ignored --nocapture       # the tables below (about 40 s)
cargo test --release potts_report -- --ignored --nocapture       # the Potts comparison below
```

## Method

- **Art.** Six scenes of 72×54 logical pixels in 8 colours: sky over ground, flat
  clouds, five boxes with 1 px outlines, a highlight and a single-pixel detail each.
  A second set adds a 6-row band where sky and ground are dithered into each other
  (checkerboard), which is the case every cleanup was expected to destroy.
- **Damage.** Upscale to a fractional size with soft edges, blur, add noise, and in
  one case JPEG-compress:

  | name | pixels per logical pixel | blur σ | noise | JPEG |
  |---|---|---|---|---|
  | crisp ×5 | 5.0 | 0 | 0 | — |
  | ×5.3, soft | 5.3 | 0.6 | ±3 | — |
  | ×7.5, soft | 7.5 | 0.8 | ±4 | — |
  | ×9.3, softer | 9.3 | 1.2 | ±5 | — |
  | ×7.5, soft, JPEG 80 | 7.5 | 0.8 | ±4 | quality 80 |
  | ×4.2, soft | 4.2 | 0.5 | ±3 | — |

- **Modes.** Snap with the standard and the elastic fitter (and the detail vote);
  no-resize plain, Blocks with either fitter, Smooth at three strengths. 12 colours for
  8-colour art, for the reason found in Phase 2 (a blurred image also holds every blend).
- **Scores.**
  - *grid exact / within 2 %*: scenes out of 6 where the grid found is 72×54, or within
    2 % of it on both axes.
  - *cells right*: share of logical pixels whose colour is right, sampling the result
    at the centre of each. Speckle lives on edges, so this cannot see it.
  - *dithered cells right*: the same inside the dithered band.
  - *pixels right*: for full-size results, the share of all pixels that match a crisp
    upscale of the truth (each pixel compared with the logical pixel its centre falls
    in). This is the one that sees speckle and misplaced cell borders.

## What it shows

1. **The standard fitter rarely finds the true grid.** Within 2 % of the true size in
   0 to 2 scenes of 6 in eight of the twelve rows, and exactly right in 7 of 72 cases
   overall (the elastic fitter: 50 of 72). It still gets 89 to 97 % of cells right, because a slightly wrong grid is
   mostly right.
2. **The elastic fitter finds it.** Within 2 % in 6 of 6 scenes in ten of twelve rows,
   with 98.8 to 100 % of cells right in those rows. The exceptions are the two JPEG
   rows: 4 of 6 and 3 of 6 scenes, and 97.9 % and 94.0 % of cells.
3. **This run found, and led to fixing, the elastic fitter's weakest part.** Its first
   results here were poor on two kinds of damage: within 2 % in 1 of 6 scenes at ×9.3
   (2 of 6 with dithering), and one crisp scene returned 44×35 instead of 72×54. The cause in both was the legacy step estimate it
   inherited (median spacing of profile peaks): on sparse art most neighbouring edges
   are two cells apart, so the median doubles; on soft art noise peaks drag it down.
   The elastic fitter now estimates its own pitch (`grid::comb_pitch`): it lays combs
   of every spacing on the edge profiles and takes the coarsest one that collects
   nearly as much edge energy as the best. After that: 6 of 6 at ×9.3 and on crisp art.
4. **Dithering survives.** This was the main open question of the research.
   - Blocks with the elastic fitter keeps 91 to 100 % of dithered cells. With the right
     grid a dither cell *is* a grid cell, so there is nothing to flatten.
   - Smooth keeps 100 % at Light and Medium in every row, and at Strong everywhere
     except ×4.2, where it keeps 60 %: a 4.2 px cell is under 18 px, at the edge of
     Strong's 16 px threshold once blur has shrunk it.
   - Snap with the standard fitter keeps only 43 to 76 %. The damage to dithering comes
     from a wrong grid, not from cleanup.
   - In snap mode with the elastic fitter the majority keeps 90 to 100 % of dithered
     cells and the detail vote 100 %, outside the JPEG row. In the JPEG row both keep
     54 %, because there the grid itself is off.
5. **Blocks is only as good as its grid.** On the pixels-right score, Blocks on the
   standard fitter's grid is *worse* than plain palette reduction in eight of twelve
   rows (for example 98.5 % against 99.2 % at ×5.3), because its cell borders are
   misplaced. On the elastic fitter's grid it is never below plain, and is the best or
   equal best full-size mode in eleven of twelve rows (99.9 % against 97.0 % for plain
   at ×5.3 with dithering). Smooth is at or above plain in every row except Strong at
   ×4.2 with dithering (95.0 % against 96.6 %), the row where it flattens the dither.
6. **JPEG is the remaining weakness.** JPEG adds its own 8×8 lattice. At a 7.5 px
   pitch the two lattices are close enough to confuse the fit in a third to a half of
   the scenes.
7. **This art barely speckles.** Plain palette reduction already has 96.6 to 100 % of
   pixels right, because eight well-separated colours leave few blends to land on a
   third colour. The harness therefore says a lot about grids and dithering and little
   about speckle removal; the real images in `04-second-batch.md` remain the evidence
   for that.

## Edge-weighted Potts against un-mixing

The research left this open: Codex proposed an edge-weighted Potts cleanup, and only a
constant-weight version had been tested. Here the edge-weighted one (ICM, 4-neighbours,
smoothness weight `exp(-|Δcolour|² / 2σ²)`, λ 0.01, σ 0.05) takes the place of un-mixing
in shipping Smooth, or is added after it, with the same sieve threshold in every arm
and the parameters Smooth actually uses on each image (reach and thresholds scaled by
the detected pitch).

- *strong edges kept*: share of the source's strong edges (over 0.10 OKLab between two
  neighbouring pixels) that still have a colour boundary of the same orientation at
  most one pixel away.
- *boundaries with no edge*: share of the result's colour boundaries with no source
  edge (over 0.02) at most one pixel away: boundaries in flat areas.

### wolf.png, 16 colours

| method | regions < 4 px | regions < 16 px | mean dE to source | strong edges kept (within 1 px) | boundaries with no edge within 1 px | seconds |
|---|---|---|---|---|---|---|
| plain palette reduction | 23133 | 26399 | 0.0053 | 100.0 % | 8.2 % | 0.00 |
| (pitch Some(6.0), un-mix reach 3) | | | | | | |
| un-mix + sieve, light (< 4 px) | 0 | 2932 | 0.0076 | 97.7 % | 9.0 % | 0.10 |
| Potts + sieve, light (< 4 px) | 0 | 2552 | 0.0070 | 97.7 % | 6.7 % | 1.09 |
| un-mix + Potts + sieve, light (< 4 px) | 0 | 2282 | 0.0072 | 97.7 % | 6.7 % | 1.10 |
| un-mix + sieve, medium (< 8 px) | 0 | 853 | 0.0081 | 95.8 % | 8.7 % | 0.12 |
| Potts + sieve, medium (< 8 px) | 0 | 826 | 0.0076 | 95.9 % | 6.6 % | 1.07 |
| un-mix + Potts + sieve, medium (< 8 px) | 0 | 714 | 0.0078 | 95.8 % | 6.7 % | 1.14 |
| un-mix + sieve, strong (< 16 px) | 0 | 0 | 0.0086 | 94.2 % | 8.1 % | 0.13 |
| Potts + sieve, strong (< 16 px) | 0 | 0 | 0.0081 | 94.5 % | 6.4 % | 1.08 |
| un-mix + Potts + sieve, strong (< 16 px) | 0 | 0 | 0.0082 | 94.4 % | 6.5 % | 1.15 |
| Potts alone, lambda 0.002, sigma 0.05 | 13927 | 16210 | 0.0055 | 100.0 % | 5.1 % | 0.70 |
| Potts alone, lambda 0.01, sigma 0.05 | 9175 | 11086 | 0.0059 | 99.9 % | 5.7 % | 1.02 |
| Potts alone, lambda 0.02, sigma 0.05 | 7697 | 9553 | 0.0061 | 99.9 % | 5.9 % | 1.00 |
| Potts alone, lambda 0.02, sigma 0.1 | 4942 | 6648 | 0.0064 | 99.5 % | 6.6 % | 1.10 |
| Potts alone, lambda 0.05, sigma 0.1 | 3379 | 5014 | 0.0068 | 99.1 % | 6.9 % | 1.00 |

### colliseum.png, 16 colours

| method | regions < 4 px | regions < 16 px | mean dE to source | strong edges kept (within 1 px) | boundaries with no edge within 1 px | seconds |
|---|---|---|---|---|---|---|
| plain palette reduction | 11557 | 17045 | 0.0204 | 100.0 % | 15.8 % | 0.00 |
| (pitch Some(9.5), un-mix reach 4) | | | | | | |
| un-mix + sieve, light (< 6 px) | 0 | 1355 | 0.0221 | 99.5 % | 17.7 % | 0.25 |
| Potts + sieve, light (< 6 px) | 0 | 2026 | 0.0209 | 99.9 % | 11.9 % | 1.58 |
| un-mix + Potts + sieve, light (< 6 px) | 0 | 1208 | 0.0214 | 99.9 % | 13.0 % | 1.74 |
| un-mix + sieve, medium (< 11 px) | 0 | 286 | 0.0222 | 99.4 % | 17.1 % | 0.23 |
| Potts + sieve, medium (< 11 px) | 0 | 392 | 0.0212 | 99.7 % | 11.8 % | 1.57 |
| un-mix + Potts + sieve, medium (< 11 px) | 0 | 280 | 0.0216 | 99.7 % | 12.7 % | 1.73 |
| un-mix + sieve, strong (< 23 px) | 0 | 0 | 0.0223 | 99.2 % | 16.1 % | 0.24 |
| Potts + sieve, strong (< 23 px) | 0 | 0 | 0.0215 | 99.5 % | 11.2 % | 1.62 |
| un-mix + Potts + sieve, strong (< 23 px) | 0 | 0 | 0.0218 | 99.4 % | 11.8 % | 1.74 |
| Potts alone, lambda 0.002, sigma 0.05 | 3161 | 6938 | 0.0205 | 100.0 % | 11.7 % | 1.52 |
| Potts alone, lambda 0.01, sigma 0.05 | 1442 | 4350 | 0.0208 | 100.0 % | 12.1 % | 1.50 |
| Potts alone, lambda 0.02, sigma 0.05 | 1084 | 3783 | 0.0209 | 100.0 % | 12.1 % | 1.52 |
| Potts alone, lambda 0.02, sigma 0.1 | 584 | 2572 | 0.0212 | 99.9 % | 13.3 % | 1.50 |
| Potts alone, lambda 0.05, sigma 0.1 | 396 | 2091 | 0.0214 | 99.8 % | 13.4 % | 1.53 |

- **Potts alone does not remove speckle.** Every setting tried keeps nearly all strong
  edges and leaves hundreds to thousands of regions under 4 px (3 379 to 13 927 on the
  wolf, 396 to 3 161 on the coliseum).
- **In place of un-mixing, it is not consistently better.** Fewer small regions on the
  wolf (2 552 against 2 932 at Light), more on the coliseum (2 026 against 1 355).
- **Added after un-mixing, it helps a little everywhere**: 22 % and 16 % fewer regions
  under 16 px on the wolf at Light and Medium, 11 % and 2 % on the coliseum; slightly
  closer to the source; and fewer boundaries in flat areas (9.0 → 6.7 % and
  17.7 → 13.0 % at Light). Strong edges are kept equally by all three arms.
- It costs about a second per megapixel in this unoptimised form, against a tenth to a
  quarter of a second for un-mix + sieve.

So the proposal neither replaces what was built nor is clearly worth its cost as an
addition, on two images. It was not built into the tool.

(The first version of this comparison, in the draft Codex reviewed, concluded that Potts
was the better pre-pass. It had run un-mix + sieve without the pitch scaling shipping
Smooth uses, which weakened that arm; Codex caught it. The table above uses the shipping
parameters.)

## Real images and the size a person picked

The second half of Phase 7 needs something no script can produce: for each real image,
the size a person actually chose. `other/picks.tsv` is the place to record it, one line
per image and mode, as you use the tool:

```
image	mode	picked_cols	picked_rows	note
```

With a few dozen lines it becomes possible to check which rung people choose (detected,
1.5×, 2×, 3×, or typed), how often the detected one is right, and whether the lattice
score's labels (clear from 0.20, none under 0.07) fall where they should. The file is
there with a header and no entries.

## Limits

- Generated art is cleaner and more regular than what image generators produce: one
  pitch per image, no drift, no mixed block sizes, 8 colours.
- Six scenes per row; a difference of one scene in six is not significant.
- The elastic fitter's pitch estimate has a known blind spot, pinned by a test: a
  profile with a few dozen strong edges at unrelated positions and nothing else reads
  as a fine lattice with most of its edges missing.
- The pitch estimate's constants (lift 0.15, near-best 0.85, edge share 0.2) were set
  on this harness and the six real images.

## Full tables

### No dithering (6 scenes of 72x54 logical pixels)

| damage | mode | grid exact | grid within 2 % | cells right | dithered cells right | pixels right |
|---|---|---|---|---|---|---|
| crisp x5 (nearest-like) | snap, standard | 0/6 | 0/6 | 91.2 % | - | - |
| crisp x5 (nearest-like) | snap, elastic | 6/6 | 6/6 | 100.0 % | - | - |
| crisp x5 (nearest-like) | snap, elastic, detail | 6/6 | 6/6 | 100.0 % | - | - |
| crisp x5 (nearest-like) | no-resize, plain | - | - | 100.0 % | - | 100.0 % |
| crisp x5 (nearest-like) | blocks, standard | 1/6 | 2/6 | 99.9 % | - | 98.8 % |
| crisp x5 (nearest-like) | blocks, elastic | 6/6 | 6/6 | 100.0 % | - | 100.0 % |
| crisp x5 (nearest-like) | smooth, light | - | - | 100.0 % | - | 100.0 % |
| crisp x5 (nearest-like) | smooth, medium | - | - | 100.0 % | - | 100.0 % |
| crisp x5 (nearest-like) | smooth, strong | - | - | 100.0 % | - | 100.0 % |
| x5.3, soft | snap, standard | 1/6 | 1/6 | 93.2 % | - | - |
| x5.3, soft | snap, elastic | 3/6 | 6/6 | 99.4 % | - | - |
| x5.3, soft | snap, elastic, detail | 3/6 | 6/6 | 99.4 % | - | - |
| x5.3, soft | no-resize, plain | - | - | 100.0 % | - | 99.2 % |
| x5.3, soft | blocks, standard | 0/6 | 0/6 | 99.3 % | - | 98.5 % |
| x5.3, soft | blocks, elastic | 3/6 | 6/6 | 100.0 % | - | 100.0 % |
| x5.3, soft | smooth, light | - | - | 100.0 % | - | 99.8 % |
| x5.3, soft | smooth, medium | - | - | 100.0 % | - | 99.8 % |
| x5.3, soft | smooth, strong | - | - | 100.0 % | - | 99.9 % |
| x7.5, soft | snap, standard | 0/6 | 2/6 | 96.0 % | - | - |
| x7.5, soft | snap, elastic | 4/6 | 6/6 | 99.6 % | - | - |
| x7.5, soft | snap, elastic, detail | 4/6 | 6/6 | 99.6 % | - | - |
| x7.5, soft | no-resize, plain | - | - | 100.0 % | - | 99.5 % |
| x7.5, soft | blocks, standard | 1/6 | 2/6 | 99.3 % | - | 99.0 % |
| x7.5, soft | blocks, elastic | 4/6 | 6/6 | 100.0 % | - | 99.7 % |
| x7.5, soft | smooth, light | - | - | 100.0 % | - | 99.6 % |
| x7.5, soft | smooth, medium | - | - | 100.0 % | - | 99.6 % |
| x7.5, soft | smooth, strong | - | - | 100.0 % | - | 99.7 % |
| x9.3, softer | snap, standard | 1/6 | 2/6 | 94.9 % | - | - |
| x9.3, softer | snap, elastic | 6/6 | 6/6 | 99.8 % | - | - |
| x9.3, softer | snap, elastic, detail | 6/6 | 6/6 | 99.8 % | - | - |
| x9.3, softer | no-resize, plain | - | - | 100.0 % | - | 99.1 % |
| x9.3, softer | blocks, standard | 0/6 | 2/6 | 99.4 % | - | 98.8 % |
| x9.3, softer | blocks, elastic | 6/6 | 6/6 | 100.0 % | - | 99.9 % |
| x9.3, softer | smooth, light | - | - | 100.0 % | - | 99.4 % |
| x9.3, softer | smooth, medium | - | - | 100.0 % | - | 99.4 % |
| x9.3, softer | smooth, strong | - | - | 100.0 % | - | 99.5 % |
| x7.5, soft, JPEG 80 | snap, standard | 0/6 | 1/6 | 93.8 % | - | - |
| x7.5, soft, JPEG 80 | snap, elastic | 1/6 | 4/6 | 97.9 % | - | - |
| x7.5, soft, JPEG 80 | snap, elastic, detail | 1/6 | 4/6 | 97.9 % | - | - |
| x7.5, soft, JPEG 80 | no-resize, plain | - | - | 100.0 % | - | 99.4 % |
| x7.5, soft, JPEG 80 | blocks, standard | 0/6 | 2/6 | 98.7 % | - | 98.4 % |
| x7.5, soft, JPEG 80 | blocks, elastic | 1/6 | 4/6 | 100.0 % | - | 99.7 % |
| x7.5, soft, JPEG 80 | smooth, light | - | - | 100.0 % | - | 99.7 % |
| x7.5, soft, JPEG 80 | smooth, medium | - | - | 100.0 % | - | 99.7 % |
| x7.5, soft, JPEG 80 | smooth, strong | - | - | 100.0 % | - | 99.7 % |
| x4.2, soft | snap, standard | 0/6 | 2/6 | 93.5 % | - | - |
| x4.2, soft | snap, elastic | 2/6 | 6/6 | 99.3 % | - | - |
| x4.2, soft | snap, elastic, detail | 2/6 | 6/6 | 99.4 % | - | - |
| x4.2, soft | no-resize, plain | - | - | 100.0 % | - | 99.1 % |
| x4.2, soft | blocks, standard | 1/6 | 1/6 | 99.2 % | - | 97.9 % |
| x4.2, soft | blocks, elastic | 2/6 | 6/6 | 100.0 % | - | 99.8 % |
| x4.2, soft | smooth, light | - | - | 100.0 % | - | 99.2 % |
| x4.2, soft | smooth, medium | - | - | 100.0 % | - | 99.3 % |
| x4.2, soft | smooth, strong | - | - | 100.0 % | - | 99.3 % |

### With a dithered band (6 scenes of 72x54 logical pixels)

| damage | mode | grid exact | grid within 2 % | cells right | dithered cells right | pixels right |
|---|---|---|---|---|---|---|
| crisp x5 (nearest-like) | snap, standard | 1/6 | 3/6 | 93.9 % | 69.9 % | - |
| crisp x5 (nearest-like) | snap, elastic | 6/6 | 6/6 | 100.0 % | 100.0 % | - |
| crisp x5 (nearest-like) | snap, elastic, detail | 6/6 | 6/6 | 100.0 % | 100.0 % | - |
| crisp x5 (nearest-like) | no-resize, plain | - | - | 100.0 % | 100.0 % | 100.0 % |
| crisp x5 (nearest-like) | blocks, standard | 3/6 | 3/6 | 100.0 % | 100.0 % | 97.1 % |
| crisp x5 (nearest-like) | blocks, elastic | 6/6 | 6/6 | 100.0 % | 100.0 % | 100.0 % |
| crisp x5 (nearest-like) | smooth, light | - | - | 100.0 % | 100.0 % | 100.0 % |
| crisp x5 (nearest-like) | smooth, medium | - | - | 100.0 % | 100.0 % | 100.0 % |
| crisp x5 (nearest-like) | smooth, strong | - | - | 100.0 % | 100.0 % | 100.0 % |
| x5.3, soft | snap, standard | 0/6 | 2/6 | 91.3 % | 44.0 % | - |
| x5.3, soft | snap, elastic | 4/6 | 6/6 | 99.3 % | 92.8 % | - |
| x5.3, soft | snap, elastic, detail | 4/6 | 6/6 | 100.0 % | 100.0 % | - |
| x5.3, soft | no-resize, plain | - | - | 100.0 % | 100.0 % | 97.0 % |
| x5.3, soft | blocks, standard | 1/6 | 2/6 | 99.8 % | 99.1 % | 97.5 % |
| x5.3, soft | blocks, elastic | 4/6 | 6/6 | 100.0 % | 99.7 % | 99.9 % |
| x5.3, soft | smooth, light | - | - | 100.0 % | 100.0 % | 99.4 % |
| x5.3, soft | smooth, medium | - | - | 100.0 % | 100.0 % | 99.5 % |
| x5.3, soft | smooth, strong | - | - | 100.0 % | 99.9 % | 99.6 % |
| x7.5, soft | snap, standard | 2/6 | 4/6 | 93.0 % | 55.7 % | - |
| x7.5, soft | snap, elastic | 6/6 | 6/6 | 99.7 % | 98.9 % | - |
| x7.5, soft | snap, elastic, detail | 6/6 | 6/6 | 99.8 % | 100.0 % | - |
| x7.5, soft | no-resize, plain | - | - | 99.8 % | 100.0 % | 97.1 % |
| x7.5, soft | blocks, standard | 1/6 | 4/6 | 99.8 % | 100.0 % | 98.8 % |
| x7.5, soft | blocks, elastic | 6/6 | 6/6 | 99.8 % | 100.0 % | 99.1 % |
| x7.5, soft | smooth, light | - | - | 99.8 % | 100.0 % | 98.8 % |
| x7.5, soft | smooth, medium | - | - | 99.8 % | 100.0 % | 98.9 % |
| x7.5, soft | smooth, strong | - | - | 99.8 % | 100.0 % | 98.9 % |
| x9.3, softer | snap, standard | 2/6 | 6/6 | 96.7 % | 76.0 % | - |
| x9.3, softer | snap, elastic | 6/6 | 6/6 | 99.8 % | 97.7 % | - |
| x9.3, softer | snap, elastic, detail | 6/6 | 6/6 | 100.0 % | 100.0 % | - |
| x9.3, softer | no-resize, plain | - | - | 100.0 % | 100.0 % | 97.2 % |
| x9.3, softer | blocks, standard | 1/6 | 2/6 | 99.7 % | 96.7 % | 98.3 % |
| x9.3, softer | blocks, elastic | 6/6 | 6/6 | 99.9 % | 99.3 % | 99.7 % |
| x9.3, softer | smooth, light | - | - | 100.0 % | 100.0 % | 98.4 % |
| x9.3, softer | smooth, medium | - | - | 100.0 % | 100.0 % | 98.4 % |
| x9.3, softer | smooth, strong | - | - | 100.0 % | 100.0 % | 98.4 % |
| x7.5, soft, JPEG 80 | snap, standard | 0/6 | 5/6 | 94.9 % | 61.3 % | - |
| x7.5, soft, JPEG 80 | snap, elastic | 0/6 | 3/6 | 94.0 % | 53.5 % | - |
| x7.5, soft, JPEG 80 | snap, elastic, detail | 0/6 | 3/6 | 94.0 % | 53.5 % | - |
| x7.5, soft, JPEG 80 | no-resize, plain | - | - | 100.0 % | 100.0 % | 97.0 % |
| x7.5, soft, JPEG 80 | blocks, standard | 1/6 | 4/6 | 99.9 % | 99.7 % | 99.1 % |
| x7.5, soft, JPEG 80 | blocks, elastic | 0/6 | 3/6 | 99.9 % | 99.2 % | 98.9 % |
| x7.5, soft, JPEG 80 | smooth, light | - | - | 100.0 % | 100.0 % | 98.6 % |
| x7.5, soft, JPEG 80 | smooth, medium | - | - | 100.0 % | 100.0 % | 98.7 % |
| x7.5, soft, JPEG 80 | smooth, strong | - | - | 100.0 % | 100.0 % | 98.7 % |
| x4.2, soft | snap, standard | 0/6 | 1/6 | 89.0 % | 42.8 % | - |
| x4.2, soft | snap, elastic | 6/6 | 6/6 | 98.8 % | 89.7 % | - |
| x4.2, soft | snap, elastic, detail | 6/6 | 6/6 | 99.8 % | 100.0 % | - |
| x4.2, soft | no-resize, plain | - | - | 99.7 % | 100.0 % | 96.6 % |
| x4.2, soft | blocks, standard | 0/6 | 4/6 | 99.2 % | 95.2 % | 96.3 % |
| x4.2, soft | blocks, elastic | 6/6 | 6/6 | 98.8 % | 91.0 % | 98.7 % |
| x4.2, soft | smooth, light | - | - | 99.7 % | 100.0 % | 97.1 % |
| x4.2, soft | smooth, medium | - | - | 99.7 % | 100.0 % | 97.3 % |
| x4.2, soft | smooth, strong | - | - | 95.9 % | 60.5 % | 95.0 % |
test eval::tests::eval_report ... ok
