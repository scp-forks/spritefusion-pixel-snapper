# Research: no-resize speckle, multi-scale candidates, tiling

Deep dive from November 2026 into three questions about cleaning AI-generated pixel art
without (or with less) downscaling. No production code was changed.

Read in this order:

| File | What it is |
|---|---|
| [`03-synthesis-and-recommendations.md`](03-synthesis-and-recommendations.md) | **Start here.** Answers, agreements and disagreements between the three lines of work, proposed build order, open questions |
| [`02-experiments.md`](02-experiments.md) | Measurements on the images in `other/`: baseline speckle, cause analysis, cleanup variants, resolution sweeps, tiling |
| [`04-second-batch.md`](04-second-batch.md) | Three more images (wolf, ogtrail, isometric city) run through the same pipeline; what held, what did not, and three prototype fixes |
| [`05-evaluation.md`](05-evaluation.md) | Generated pixel art with known answers, damaged and recovered: grid recovery, dithering survival, and edge-weighted Potts against the sieve |
| [`01-literature-survey.md`](01-literature-survey.md) | Papers and open-source tools, with verification tags per claim |
| [`codex-independent-report.md`](codex-independent-report.md) | Independent report by Codex (gpt-6.1-sol, xhigh), unedited |
| [`figures/`](figures/) | Comparison crops (`fig*.png`, panels left to right as named) and full-size results (`full/`) |
| [`experiments/`](experiments/) | Prototype code; `run_all.sh` reproduces every number |

The implementation plan that follows from this research is
[`../plans/no-resize-cleanup-and-audition.md`](../plans/no-resize-cleanup-and-audition.md),
with Codex's review of it in [`../reviews/`](../reviews/no-resize-cleanup-and-audition-review.md).

Figures worth opening first:

- `figures/fig1_orig_vs_baseline.png` — the problem.
- `figures/full/colliseum_A0_baseline_kmeans16.png` vs `figures/full/colliseum_A2_snap_in_place_x2.png` — before and after, full frame.
- `figures/fig12_native-majority_native-dpid-vote_x2-majority.png` — why one resolution is not enough.
- `figures/fig9_sample_orig_baseline_snap-in-place.png` — the same fix on a clean lattice.
- `figures/fig10_spatial_shipping_vs_assignments-used.png` — the Spatial quantizer bug.
- `figures/fig14_wolf-ear_*.png` — wolf ear: baseline speckle, the broken and the fixed dissolve step, snap in place.
- `figures/fig16_ogtrail-text_*.png` — ogtrail lettering at four sizes; the shipping size is the last panel.
- `figures/fig17_city-cars_*.png` — the tiny cars from full size down to the shipping size.

One correction to keep in mind when reading the survey: its statement that spatial colour
quantization "creates speckle by design" refers to Puzicha's algorithm (which dithers),
not to algorithm 3 in this repo.
