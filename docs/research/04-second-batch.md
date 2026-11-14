# 04 — Second batch of samples (Claude)

Three more images were added after the first round was written up. They were chosen to
stress the first round's conclusions, and they did: one conclusion held as stated, two
needed qualifying, and three of the prototypes turned out to have defects that the
coliseum had not exposed.

Scripts: `experiments/e15_pitch_spectrum.py`, `e16_second_batch.py`, `e17_local_pitch.py`.
All runs use k-means perceptual, 16 colours, as in round one.

| image | size | what it is | shipping snap mode gives |
|---|---|---|---|
| `wolf.png` | 1024×1024 | AI pixel art, one sprite on white. Blocks ≈ 5 px, painterly shading inside the ear and fur | 205×208, looks right |
| `ogtrail.png` | 960×540 | AI "pixel art" scene with **several block sizes at once**: sky clouds in ≈ 12 px blocks, wagon and people at ≈ 4 px with 2 px details, lettering with 2 px strokes | 181×103, illegible |
| `isometric-city-with-tiny-cars.png` | 512×512 | Cel-shaded illustration with 1 px outlines. **No pixel lattice.** Cars are about 10×14 px | 111×120, cars are smudges |

## What changed, in short

| Round-one statement | After the second batch |
|---|---|
| Speckle is a missing spatial stage, not a quantizer problem | **Holds.** Same mechanism on the wolf (tan dots along the black outline). On the city most small fragments are real detail, so "speckle" and "detail" cannot be told apart by size alone |
| Snap in place is the fix for pixel art | **Holds where there is a lattice** (wolf). On ogtrail and the city the fitted grid is barely better than an even one, so snap in place there is a downscale-and-repaint, useful but not grid recovery |
| The native grid is one step too coarse; ×2 is the sweet spot | **Coliseum-specific.** Wolf: native is right. ogtrail: the usable size is about 2.6× finer than the one shipping picks. City: no native grid at all |
| Un-mix + dissolve small regions is the safe grid-free cleanup | **The dissolve step was defective**: it erased the wolf's ear outline. Fixed with a colour-aware version (below). Un-mix itself does little when blocks are small |
| Autocorrelation finds the pitch | **Wrong on two of three new images.** Replaced with a score-per-pitch sweep (E15) |
| The elastic grid fit is strictly better | It over-segments soft lattices (235 columns on the wolf where ≈ 200 is right). Fixed with a per-cut cost |
| A lattice score near zero flags "not pixel art" | The gap is narrower than three images suggested: 0.10 (cat), 0.19 (city), 0.23 (ogtrail), 0.30 (wolf) |
| Tiling adds no detail capacity | **Holds**, and the city shows it directly. One new use for regions: block size really does differ by region on ogtrail |

## E15 — Finding the pitch: a score per candidate pitch

`lab.pitch_acf` (first strong autocorrelation peak) reported 3.6/3.1 px on the wolf
(real ≈ 5) and 1.5/5.0 px on ogtrail. Round one's coliseum and sprite-sheet values were
right, which is why this was not noticed.

Replacement: for each candidate pitch, fit a nearly rigid grid (steps within ±8 %) and
compare its residual with what unaligned even grids achieve at the same number of cells.
`score = 1 − fitted / unaligned`. A lattice scores high at its pitch and at every finer
pitch that contains it, then falls. The pitch is read as the largest effective pitch
still within 85 % of the best score.

| image | best score | pitch read off | shape of the curve | other estimates |
|---|---|---|---|---|
| sprite sheet | 0.56 | 4.0 | flat ≈ 0.55 up to 4, then drops to 0.17 | acf 4.0 |
| coliseum | 0.43 | 9.7 | flat ≈ 0.40 up to 9.5, then 0.25, then 0.15 | acf 9.3; shipping 8.9 |
| wolf | 0.30 | 5.9 | flat ≈ 0.29 up to 5.5, then slides (0.25, 0.21, 0.19) | acf 3.1–3.6 (wrong); shipping 5.0 |
| ogtrail | 0.23 | 3.4 | no step at all: slides from 0.23 at 2 px to 0.09 at 9 px | acf 1.5 / 5.0; shipping 5.3 |
| city | 0.19 | 2.8 | 0.19 at 2–2.5 px, ≈ 0.10 beyond | acf 4.1 / 28.6; shipping 4.5 |
| cat | 0.10 | — | flat, never above 0.10 | — |

The shape carries more than the number. A step means one lattice (sprite sheet,
coliseum). A slide means a soft lattice or several block sizes (wolf, ogtrail). Flat and
low means none (cat). The city sits awkwardly in between: something repeats at 2–3 px
(probably outline width), but it is not a pixel grid.

Limits: the 85 % rule and the score levels are tuned on these six images. On the wolf
the rule reads 5.9 where run lengths and the shipping estimator both say 5.0. The sweep
takes 1–2 minutes per image in numpy; it is a diagnostic here, not a proposed
implementation. The score formula differs from E9's (same idea, normalised by cell count
instead of by nominal pitch), so the cat's 0.10 here and 0.04 there are not comparable.

### Fix to the grid fitter

The elastic fit maximises edge strength summed over its cuts, so every extra cut is
rewarded. On a crisp lattice that does not matter. On the wolf it produced 235×237 cells
at a nominal pitch of 5. Charging one unit per cut (`cuts_dp(..., cut_cost=1)`, so a cut
on an average column earns nothing) gives:

| image | cells before | cells after | shipping | residual after | even grid, same cell count |
|---|---|---|---|---|---|
| wolf, pitch 5 | 235×237 | 199×200 | 205×208 | 0.0123 | 0.0160 |
| coliseum, pitch 9.3 | 191×99 | 185×99 | 188×100 | 0.0192 | 0.0298 |
| sprite sheet, pitch 4 | 256×256 | 256×256 | — | 0.0268 | 0.0268 |
| ogtrail, pitch 5.3 | 207×120 | 187×105 | 181×103 | 0.0329 | 0.0369 |
| ogtrail, pitch 2 | 463×262 | 473×266 | — | 0.0165 | 0.0169 |

Round-one coliseum numbers were produced without the cut cost and are left as they
were; the difference is 0.0189 against 0.0192.

The last column is the useful one. How much a fitted grid beats an even grid with the
same number of cells says how much lattice there is to recover: 36 % on the coliseum,
23 % on the wolf, 11 % on ogtrail at 5.3 px, 2 % on ogtrail at 2 px and on the city at
2 px.

## E16 — The pipeline on the three images

Fragment counts are absolute here (regions under 4 px and under 16 px), because a
quarter of a logical pixel is a single pixel when the pitch is 2.

### Baseline: all three speckle

| image | regions < 4 px | regions < 16 px | share of image in regions < 16 px |
|---|---|---|---|
| wolf | 23 133 | 26 399 | 5.0 % |
| ogtrail | 7 223 | 9 963 | 5.8 % |
| city | 27 761 | 34 565 | 33.2 % |

The wolf's 5 % is nearly all on the outline and in the soft shading of the ear
(`fig14`, second panel): the fringe case described in the original request. A third of
the city is in small regions, and most of that is content (window rows, 1 px outlines),
not noise.

### The dissolve step from round one was wrong

Round one dissolved every small region into the label of the nearest surviving pixel.
The wolf's outline quantises to a fine mix of two near-black palette colours. Each piece
is small, all of them are dissolved at once, and the nearest survivors are the white
background and the grey fur. The outline is eaten and the fur spikes disappear (`fig14`,
third panel).

`lab.sieve_colour` instead merges each small region into the **adjacent** label whose
colour is closest to the region's own mean source colour, smallest regions first.

| after un-mix, then dissolve < 8 px | regions < 16 px left | dE p99 | pixels > 0.10 off |
|---|---|---|---|
| wolf, nearest survivor (old) | 231 | 0.248 | 2.1 % |
| wolf, colour-aware | 253 | 0.148 | 1.6 % |
| ogtrail, old | 616 | 0.228 | 4.2 % |
| ogtrail, colour-aware | 626 | 0.148 | 3.7 % |
| city, old | 1 045 | 0.510 | 13.4 % |
| city, colour-aware | 1 321 | 0.261 | 6.9 % |
| coliseum (< ¼ cell), old | 0 tiny | 0.126 | 1.56 % |
| coliseum (< ¼ cell), colour-aware | 0 tiny | 0.116 | 1.39 % |

Same cleanliness, roughly half the damage, on every image. Cost in numpy is 0.5–2 s
against 0.2 s. The round-one coliseum result stands, slightly improved.

### Un-mix does little when blocks are small

Un-mixing needs flat "core" pixels within reach of every blend pixel.

| image | flat core share | fragments removed by un-mix alone |
|---|---|---|
| wolf (radius 2) | 86 % | 34 % |
| ogtrail (radius 1) | 59 % | 4 % |
| city (radius 1) | 29 % | 1 % |

On the coliseum it was the step that removed the stray browns. At a 2–3 px pitch the
sieve does almost all the work, and the sieve cannot tell a stray dot from a window.

### Snap in place and the candidates

All cells are voted from the cleaned label map ("majority") or with the detail-weighted
vote of E11 on the raw labels ("detail").

**Wolf** (`fig14`; `figures/full/wolf_*`)

| output | regions < 16 px | dE mean |
|---|---|---|
| baseline | 26 399 | 0.0053 |
| un-mix + colour sieve (8 px) | 253 | 0.0094 |
| snap in place 199×200 | 34 | 0.0146 |
| snap in place 392×385 | 372 | 0.0117 |
| snap in place 102×103 | 0 | 0.0245 |

Two readings of "keep it at high resolution but reduce the colours" give two answers.
If the blocks should stay blocks, snap in place at 199×200 on the full canvas is that
result, and it is the shipping snap output painted back at 1024×1024. If the soft
shading inside the blocks should survive, it is un-mix + colour sieve. Both are clean
around the ear tips.

**ogtrail** (`fig15`, `fig16`, `fig19`; `figures/full/ogtrail_*`)

| output | grid | regions < 16 px | dE mean | lettering |
|---|---|---|---|---|
| baseline, full size | — | 9 963 | 0.0403 | legible, speckled |
| un-mix + colour sieve (8 px) | — | 626 | 0.0426 | legible |
| snap in place | 473×266 | 740 | 0.0495 | legible |
| snap in place | 323×186 | 680 | 0.0515 | legible |
| snap in place | 238×139 | 251 | 0.0549 | degraded, still readable |
| snap in place | 187×105 | 22 | 0.0586 | broken with majority ("Gregon T-ail Mia"); nearly readable with the detail vote |
| shipping snap | 181×103 | — | — | gone ("Dragon Trail Mr") |

The shipping estimate of 5.3 px sits between the sky's blocks and the wagon's, and is
too coarse for everything that matters in the picture. At almost the same size, the
fitted grid with the detail vote (187×105) keeps the lettering nearly readable where
shipping does not, so part of the mess is the reducer and part is the size. Nothing in
the image says 323 is right and 187 is wrong except a person looking at it.

At 16 colours the blue vest turns dark green and the red hair brown in every variant,
baseline included. That is the palette running out, not a spatial problem, but it will
dominate a first impression of any of these outputs.

**City** (`fig17`, `fig18`; `figures/full/city_*`)

| output | grid | pixels > 0.10 off | cars |
|---|---|---|---|
| baseline, full size | — | 0.2 % | intact |
| un-mix + colour sieve (4 px) | — | 3.9 % | intact |
| un-mix + colour sieve (8 px) | — | 6.9 % | intact; thin facade lines start to break |
| un-mix + colour sieve (16 px) | — | 11.3 % | losing their outlines |
| snap in place | 249×256 | 17 % | recognisable |
| snap in place | 176×171 | 19–23 % | light blobs (majority); outlined blobs (detail vote) |
| snap in place | 127×130 | 22–29 % | blobs |
| shipping snap | 111×120 | — | smudges |

The detail-weighted vote keeps the cars' dark outlines at 176 and 127, and in the same
frame turns the textured facades into noise (`fig18`). That answers a round-one open
question: yes, it promotes noise elsewhere. It stays an alternative flavour, not a
default.

## E17 — Block size by region

The E15 sweep on a 4×4 tiling of each image. Each entry is pitch / best score.

```
coliseum                           ogtrail
 9.7/0.70  9.4/0.56 10.1/0.34  9.4/0.49     11.3/0.38  7.1/0.31  2.2/0.19 12.1/0.29
 9.1/0.40  9.0/0.46  9.1/0.50  4.7/0.39      5.9/0.31  2.9/0.32  7.1/0.30 12.5/0.26
 9.6/0.51  9.4/0.50  9.1/0.47  9.2/0.46      3.8/0.29  3.8/0.26  4.5/0.25  4.5/0.22
 9.6/0.56  9.4/0.55  9.7/0.53  9.5/0.54      6.9/0.29  3.8/0.26  3.8/0.25  5.7/0.29

wolf                               city
   --        --      5.8/0.31  7.2/0.39      2.8/0.20  2.7/0.37  2.8/0.33  2.8/0.34
 8.1/0.37  6.0/0.33  5.8/0.24  6.9/0.35      2.8/0.22  2.8/0.28  2.8/0.20  2.9/0.28
 5.9/0.33  7.2/0.34  5.8/0.36    --          2.5/0.16  2.6/0.21  2.8/0.25  2.8/0.20
 6.1/0.56  6.9/0.47 10.0/0.40  4.8/0.53      2.2/0.26  2.8/0.26 10.2/0.38  2.9/0.28
```

- The coliseum is one lattice: 15 of 16 tiles read 9.0–10.1.
- ogtrail is not: about 12 px in the sky, about 4 px across the wagon and grass, mixed
  values where a tile straddles both.
- The estimator is noisy at this tile size. The wolf is a single sprite and its tiles
  still range from 4.8 to 10.0. Per-tile pitch is not yet reliable enough to drive
  anything automatically.

What this means for the tiling idea: on a mixed image the block size does depend on
where you look, and a single global estimate (the median of edge spacings, as shipping
does) lands between the sizes. But the output does not need tiles. A 12 px sky block is
a 3×3 group of 4 px cells, so one grid at the finest well-supported pitch represents
both. Regions are useful for *choosing* the pitch (take the finest pitch that a
substantial part of the image supports, not the global median), not for processing.

## What the second batch says

1. **Three images, three different right sizes, for three different reasons.** Wolf:
   the estimate is right. ogtrail: the estimate is 2.6× too coarse because the image
   mixes block sizes. City: there is no grid to estimate. This is the strongest evidence
   so far for an audition ladder, and it says the ladder must reach well below the
   estimate, not stop at ×2.
2. **How far the fitted grid beats an even one is the measure of "is this pixel art".**
   36 %, 23 %, 11 %, 2 %. When it is small, snap in place still works as a clean
   downscale-and-repaint, but calling it grid recovery is wrong and the result should
   not be expected to look hand-pixelled.
3. **The grid-free cleanup needs the colour-aware sieve.** The round-one sieve should
   not be built.
4. **On non-lattice images, small regions are mostly detail.** No size threshold
   separates them from noise. Cleanup strength has to be a control, and its default
   should be gentle (4 px on the city already moves 4 % of pixels visibly).
5. **Tiles still add no capacity.** The cars survive at about 250 cells across the whole
   image and not at 176 or below. Four quadrants at 128 is that same 256.
6. **Pitch estimation is the weakest part of the prototype chain.** Two estimators have
   now failed on something (autocorrelation on the wolf and ogtrail; shipping's median
   spacing on ogtrail). The sweep in E15 is more robust but slow and still tuned by hand.

## Limits

- Six images in total, of which two are not lattice pixel art and one is mixed.
  Thresholds remain hand-tuned.
- Only 16 colours were run. Speckle roughly doubles with each doubling of colours (E1),
  and ogtrail visibly needs more than 16.
- Still no image with intentional dithering.
- The wolf has no transparency (alpha is 255 everywhere), so alpha handling is still
  untested.
- Lettering and car legibility were judged by eye on one crop each (`fig16`–`fig19`).
