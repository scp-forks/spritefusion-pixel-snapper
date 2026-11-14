# 01 - Literature and tooling survey

Retrieved 2026-11-13. Scope: prior art relevant to three questions about Pixel Snapper.

- **A. Speckle in no-resize mode**: flat regions and crisp blocky edges after palette reduction at full resolution, without destroying 1-logical-pixel details or intentional dithering.
- **B. Multi-scale candidates**: choosing, producing and ranking outputs at several resolutions.
- **C. Tiling / region-wise processing**: soundness, seams, better variants.

## How to read this document

Every external claim carries a source tag:

| Tag | Meaning |
|---|---|
| **[opened]** | I fetched the page, or read the source file / PDF text directly. Web pages were read through a summarising fetch tool, so wording is paraphrased unless quoted. |
| **[source]** | I read the actual source code or the extracted PDF text myself (highest confidence). |
| **[snippet]** | The claim comes from a search-result summary only. The URL appeared in results but I did not open it. |
| **[unverified]** | From my own memory, or my own inference. Not confirmed by anything I opened. |
| **(my synthesis)** | A design suggestion of mine, built on the cited material. Not something a source states. |

Benchmark numbers quoted from tool authors are **self-reported by vendors** and two of them contradict each other (section 2.12). "Pixel Snapper" in those tables is the upstream `Hugo-Dz/spritefusion-pixel-snapper`, not this fork.

`s` denotes the logical-pixel pitch in real pixels (roughly 4-12 here). `K` is the palette size, `N` the number of real pixels (about 1e6).

## Headline findings

1. **The problem space got crowded in 2026.** At least six open-source tools now target exactly "fix AI pixel art" (pixel-art-fixer, xprite-pixelate, proper-pixel-art, pixel-respecter, perfectPixel, unfake's Rust core), and there is an open benchmark, `pixel-bench`, that already bundles a port of Pixel Snapper as a baseline. No academic paper on grid recovery from AI-generated pixel art turned up; the state of the art lives in READMEs.
2. **Nobody ships a full-resolution despeckler as a named technique.** Every tool gets flat regions and blocky edges *by construction*: reconstruct one colour per detected cell, then optionally nearest-upscale back to source size. Question A is therefore either (a) solved by reusing the snap machinery and rendering back at full size, or (b) genuinely under-explored if the grid must not be used.
3. **Speckle and intentional detail separate by scale.** A speckle is 1 to a few real pixels; an intentional detail or a dither cell is about `s x s` real pixels. Label-map cleanup with thresholds expressed in units of `s^2` (connected-component sieve, mode filter of radius about `s/4`) is the cheap, well-precedented route (ImageMagick, GDAL, ArcGIS all ship it). The hard residue is the anti-aliasing fringe, which is thin but long and needs a width or "in-between colour" rule rather than an area rule.
4. **A reusable no-reference score exists for B.** pixel-art-fixer's "distillability" score is within-cell variance computed from summed-area tables, with an anti-phase term and a trend baseline. It is designed to arbitrate between a pitch and its multiples/divisors, which is the same decision as ranking candidate resolutions.
5. **Independent per-tile processing is the wrong decomposition for C.** The precedents (console attribute tables, S3TC, tile palette quantizers) all show visible block seams. The tools that handle drift use tiles only to *estimate* (local pitch, local phase), then build one continuous mesh and one global palette.
6. **Spatial colour quantization is the wrong tool for A.** It is defined as joint quantization and dithering, so it produces speckle on purpose.

---

## 1. Pixel-art-specific academic work

### 1.1 Gerstner et al., "Pixelated Image Abstraction" (NPAR 2012)

- **What**: Converts a natural image to a low-resolution, small-palette pixel image by jointly optimising a superpixel-to-output-pixel mapping and the palette. A journal extension adds user constraints.
- **Algorithm [source]** (PDF text):
  - Work in CIELAB. One superpixel per output pixel, initialised on the regular output lattice.
  - Loop: (1) one modified SLIC step, (2) associate superpixels to palette colours, (3) refine palette, (4) if palette converged, lower temperature and possibly split colours.
  - SLIC modification: the colour used in the SLIC distance for a superpixel is its **assigned palette colour**, not its mean colour, so the segmentation is optimised against the palette. Compactness `m = 45` (versus `m = 10` suggested by the SLIC authors), which weights position heavily and keeps cells near their lattice site.
  - After each SLIC step: **Laplacian smoothing** of superpixel centres (move each centre 40% of the way to the mean of its 4-neighbours), then a **bilateral filter** over the `w_out x h_out` image of superpixel mean colours.
  - Palette by mass-constrained deterministic annealing: start with one colour at `T0 = 1.1 * Tc` (`Tc` = twice the variance along the major principal axis), cool by `alpha = 0.7`, split a colour along its principal axis when its two sub-clusters separate. Soft assignment probabilities depend on LAB distance and temperature; at the final temperature it is equivalent to k-means.
  - Post-step: palette saturation by a factor `beta = 1.1`.
  - Stated limitation: no dithering. The authors say selective dithering would need the user to mark regions or texture analysis.
- **Bearing**:
  - A: the lattice-initialised, high-compactness SLIC is a principled "elastic logical pixel": each superpixel hugs one fuzzy blob. Rendered back at full size it gives flat regions, but **not** blocky ones, since superpixel outlines are free-form. It would have to be re-rasterised as lattice cells to look like pixel art, at which point it is just another grid resampler.
  - B: it is a downscaler to an *arbitrary* target size, so it is a candidate generator for non-native grids. Designed for natural images, and Kopf et al. report it loses thin connected lines on cartoon input (1.2).
  - C: the Laplacian-smoothing step is a useful precedent for keeping locally adapted cells from drifting apart.
- **Cost**: iterative; SLIC is linear in `N` per step but many annealing stages. No runtime found in the text I read. Heavy for interactive WASM use.
- **URLs**: https://gfx.cs.princeton.edu/pubs/Gerstner_2012_PIA/Gerstner_2012_PIA_full.pdf ; journal version https://cragl.cs.gmu.edu/pixelate/pixelated_image_abstraction-journal.pdf [snippet] ; NumPy reimplementation https://github.com/kbullaughey/Pixelated-Image-Abstraction [snippet].

### 1.2 Kopf, Shamir, Peers, "Content-Adaptive Image Downscaling" (SIGGRAPH Asia 2013)

- **What**: Downscaling where each output pixel has its own bilateral kernel (spatial Gaussian times colour Gaussian) whose position and shape adapt to image features. Has a dedicated pixel-art section (5.1).
- **Algorithm [source]** (PDF text):
  - Model the input as a mixture: output pixel `k` has spatial mean `mu_k`, spatial covariance `Sigma_k`, colour `nu_k`, colour bandwidth `sigma_k`.
  - **E-step**: `gamma_k(i)` = normalised weight of kernel `k` at input pixel `i` (weights normalised across kernels per input pixel).
  - **M-step**: update `mu_k`, `Sigma_k`, `nu_k` as `gamma`-weighted statistics.
  - **Constraint step** (not derivable from EM, done as a third step): (i) move `mu_k` halfway to the mean of its 4-neighbours and clamp to a box around the output pixel centre; (ii) clamp eigenvalues of `Sigma_k` to `[0.05, 0.1]`; (iii) `sigma_k` is not estimated but starts "crisp" and is raised by 10% for kernels that grow too dominant or that would cause staircasing.
  - For pixel art they **disable the edge-orientation (anti-staircase) constraint** to get the blocky look.
  - Palette reduction as a post-process: mean-shift segmentation (Epanechnikov kernel, spatial bandwidth `h_s = 4`, colour bandwidth tuned to hit a colour count) for cartoon/vector input; plain k-means for natural images, because mean shift "does not work well" there.
  - Stated limitations: flicker under animation, broken symmetry, does not prevent aliasing, "does not always produce better results on images with blurred features", cannot match expert pixel artists, multiple local minima.
- **Bearing**:
  - B: the best-documented way to downscale to a **coarser-than-native** grid while keeping outlines connected. Its claimed advantage over Gerstner is exactly "keeps line features connected".
  - A: skeptical. The paper itself warns about blurred features, and AI pseudo-pixels are blurred features.
  - Note: unfake's Rust core ships a `content_adaptive.rs` whose header calls it "Öztireli-Gross EM" [source]; an EM with clamped covariance eigenvalues is this paper's formulation, so that label looks like a misattribution [unverified, my inference].
- **Cost**: high. A later paper quotes over an hour for a 24 MP image with this method [snippet, https://dl.acm.org/doi/10.1145/2980179.2980239]. unfake bounds cost with a "RATIO_CAP=3 pre-shrink" [source].
- **URLs**: https://johanneskopf.de/publications/downscaling/paper/downscaling.pdf ; project page https://johanneskopf.de/publications/downscaling/ [snippet].

### 1.3 Öztireli and Gross, "Perceptually Based Downscaling of Images" (SIGGRAPH 2015)

- **What**: Downscaling that maximises SSIM against the input in closed form; only box filters and arithmetic.
- **Algorithm** (as implemented in a C gist I opened; the paper PDF itself was not opened):
  - Per plane, with integer factor `s`: `l` = `s x s` block mean, `l2` = block mean of squares.
  - For each 2x2 patch `P` of low-res cells: `m = mean(l)`, `S_l = mean(l^2) - m^2`, `S_h = mean(l2) - m^2`, `r = sqrt(S_h / S_l)` (gist uses a fixed `r` when `S_l < 1e-6`).
  - Output `d(x,y)` = average over the four patches containing the cell of `m + r * (l(x,y) - m)`.
  - Intuition: start from the box-filtered image and locally stretch contrast so each patch keeps the variance the full-resolution patch had.
- **Bearing**: B. Cheap way to keep a small detail's *contrast* alive when going below native resolution. But it creates new colours (overshoot) and must be followed by palette snapping; on already-flat cells it is a no-op. Low value at native resolution.
- **Cost**: O(N), trivial.
- **URLs**: gist https://gist.github.com/HybridDog/dd95a99d411972f030fb18543280ad60 [opened] ; paper https://www.cl.cam.ac.uk/~aco41/Files/Sig15PerceptualDownscaling.pdf [snippet].

### 1.4 Weber et al., "Rapid, Detail-Preserving Image Downscaling" (DPID, SIGGRAPH Asia 2016)

- **What**: A single-pass downscaler that up-weights input pixels that differ from their neighbourhood.
- **Algorithm [snippet]** (formula as restated in arXiv 1804.04076): `O[p] = (1/k_p) * sum_{q in patch(p)} I[q] * ||I[q] - G[p]||^lambda`, where `G` is a guidance image (a smoothed box-downscale) and `k_p` normalises. `lambda = 0` is a box filter; `lambda` in 0.5-1 was preferred in the user study. The exact guidance filter (I recall a box downscale then a 3x3 binomial blur) is [unverified].
- **Bearing**: B. This is the "do not crush the 1-pixel highlight" kernel: the minority pixels in a cell win. That is the *opposite* of majority vote, so it is useful as an alternative candidate, or selectively where majority vote would delete a compact minority blob. It also amplifies speckle and JPEG noise for the same reason, so it must run after cleanup, never before.
- **Cost**: O(N). Reported 24 MP in about 40 ms on GPU [snippet].
- **URLs**: https://dl.acm.org/doi/10.1145/2980179.2980239 [snippet] ; https://github.com/mergian/dpid [opened; README has no algorithm text, BSD-3] ; formula https://arxiv.org/pdf/1804.04076 [snippet].

### 1.5 Kopf and Lischinski, "Depixelizing Pixel Art" (SIGGRAPH 2011)

- **What**: Vectorises true pixel art. Relevant here for its *connectivity heuristics*, not for the vectorisation.
- **Algorithm [snippet]**: build a similarity graph (8-neighbours, edges between similar colours); resolve each crossing diagonal pair in a 2x2 block by voting with three heuristics: **curves** (prefer the diagonal on the longer valence-2 chain), **sparse pixels** (prefer the sparser colour within a local window, treated as foreground), **islands** (never orphan a pixel whose only link is that diagonal). Exact weights were not in what I read.
- **Bearing**: A, as a *protect list*. These heuristics formalise "this single pixel is part of a 1-pixel line, not noise". A despeckler working on the snapped (1x) image should refuse to delete a pixel that is part of a valence-2 chain. Not applicable at full resolution, where a logical pixel is a blob.
- **Cost**: trivial for the heuristics alone.
- **URLs**: https://johanneskopf.de/publications/pixelart/paper/pixel.pdf [snippet] ; https://www.johanneskopf.de/publications/pixelart/index.html [snippet].

### 1.6 Learned pixelization and generation (2018-2026)

All of these go photo/illustration to pixel art or text to pixel art. None recovers a grid from existing fake pixel art.

| Work | What it does | Bearing | URL |
|---|---|---|---|
| Han et al., "Deep Unsupervised Pixelization" (2018) | GAN-based image to pixel art | None directly | https://dl.acm.org/doi/10.1145/3272127.3275082 [snippet] |
| Wu et al., "Make Your Own Sprites: Aliasing-Aware and Cell-Controllable Pixelization" (TOG 41(6), 2022) | Two stages, one cell-aware and one aliasing-aware; a reference pixel-art image regularises cell structure; ships an `AliasNet` trained on an aliasing dataset; `--cell_size` any integer from 2 | A, conceptually: they treat "remove anti-aliasing" as its own learned stage, separate from cell size and colour. Needs a GPU and PyTorch, so not usable in WASM | https://github.com/WuZongWei6/Pixelization [snippet] ; https://dl.acm.org/doi/10.1145/3550454.3555482 [snippet] |
| Binninger and Sorkine-Hornung, "SD-piXL" (arXiv 2410.06236, SIGGRAPH Asia 2024) | Optimises an `H x W x n` logit tensor (softmax over a fixed n-colour palette, Gumbel-softmax for crispness) with score distillation; optional image conditioning | B, as an expensive oracle for "what should this look like at 64x64". One snippet quotes about 4 hours per image [snippet, via arXiv 2512.07834] | https://arxiv.org/abs/2410.06236 [opened, abstract] |
| Lei, Xu, Zhang, "An art-oriented pixelation method for cartoon images" (Visual Computer, online 2023) | Gridding, content extraction, separate handling of contour and non-contour regions | The contour/non-contour split mirrors "structure first, colour second" (2.1). Details not read | https://link.springer.com/article/10.1007/s00371-022-02763-0 [snippet] |
| Shang and Wong, "Automatic Portrait Image Pixelization" (Computers and Graphics 95, 2021) | SLIC plus fuzzy ISODATA for portraits | Superpixels for pixelization precedent | https://www.researchgate.net/publication/348706011_Automatic_Portrait_Image_Pixelization [snippet] |
| Zhang et al., "PixelBench" (arXiv 2611.02540, Nov 2026) | Benchmarks LLMs emitting palette plus index grid directly; uses Gerstner as a baseline | Shows the abstract defines no pixel-art quality metric. Not useful for metrics | https://arxiv.org/abs/2611.02540 [opened, abstract] |
| Seo et al., "Structure-Aware Pixel Art Scaling via Block Size Detection" (Applied Sciences 16(5):2314, Feb 2026) | Detects the integer block size `n` of pre-scaled pixel art, collapses to `n = 1`, then rescales; proposes Color Loss, Block Size Consistency and Reversibility metrics | B and metrics. The only paper found that names "block size consistency" as a metric. Appears aimed at clean nearest-neighbour upscales. Full text returned HTTP 403, so the detection algorithm is **unverified** | https://www.mdpi.com/2076-3417/16/5/2314 [snippet] |

---

## 2. Practical / open-source tools

This is where the transferable material is. Ordered roughly by how much I learned from each.

### 2.1 Retro Diffusion `pixel-art-fixer` (Astropulse, 2026, MIT, Python reference plus dependency-free Rust core)

The most thoroughly documented pipeline. `docs/HOW_IT_WORKS.md` is a 41 KB design rationale, read directly.

- **Failure taxonomy [source]**: (1) octave/harmonic ambiguity, "the dominant failure class"; (2) content scale versus pixel scale (the strongest periodicity is often sprite layout or texture, not the grid); (3) cell count off by a few under drift, because one global step cannot summarise a drifting grid; (4) heuristic churn.
- **Asymmetry worth remembering [source]**: "A grid that is one octave too coarse produces a visibly worse downscale; a grid that is slightly too fine reconstructs almost perfectly."
- **Detection [source]**:
  - Stage 1, three cheap phase-invariant detectors; exit early if they agree on output size within `max(1 cell, 1%)`.
    - `autocorr`: feature map = `|2nd difference|` of luminance (survives bilinear/bicubic blur, where the first difference is flat) plus `0.7 * |1st difference|` of median-quantised luminance. **Sum the feature over row bands (12 / 24 / all lines) before autocorrelating**, because "the lattice lives in boundary positions shared across rows, not in any single row"; per-line ACF of a mushy image shows no lattice peaks. Score steps with a comb-minus-anti-comb.
    - `runlengths`: soft-GCD comb over distances between colour-change boundaries.
    - `selfsim`: shift-dissimilarity curve that dips at multiples of the step; per-tile votes; under drift the answer is the area-weighted **harmonic mean** of local steps.
  - Stage 2 (only on disagreement): fused spectral/tile evidence, within-cell variance contrast from summed-area tables, and **distillability**.
  - Decision rules: per-axis candidate pool; "the smallest qualifying step wins"; near-equal axes snap square but square is a soft prior (they cite real `6.38 x 8.00` and `4.0 x 6.0` cells).
  - JPEG's 8 px lattice is detected and removed first.
- **Distillability score [source]** (central for B):
  - Quantise first (k-means, K=14) so AA ramps snap flat; without this "residual ramp variance swamps the curve".
  - Round-trip error of box-downscale then nearest-upscale at step `s`, phase `p` equals total within-cell variance: `E(s,p) = sum(I^2) - sum_cells (sum_cell I)^2 / n_cell`, computed from cumulative sums of `I` and `I^2` with fractional boundaries (interpolated cumsums).
  - `score(s) = max(er - eb, 0) * max(1 - eb / trend(s), 0)` where `eb` = error at the best phase, `er` = error at best phase + `s/2`, `trend(s)` = running median of `E` over `s`.
  - First factor is about 0 below the true step (any phase subdivides cells) and large at it. Second factor goes to about 0 at multiples of the true step.
  - Use the absolute `(er - eb)` normalised by total variance, never the ratio `(er - eb)/(er + eb)`, which saturates when both are tiny.
- **Reconstruction, "two-stage packing" [source]** (central for A and B):
  1. Even lattice at `x * cols / w`. **Per-cut snapping to gradient maxima was dropped**: "on detailed art the strong gradients are content edges ... not pixel boundaries, and snapping to them cascades into sliver and double-wide cells."
  2. Quantise for structure only, `K` adaptive in [16, 48].
  3. Stage 1, placement: each cell takes a **centre-weighted vote over labels**.
  4. Stage 2, colour: each cell gets the centre-weighted **mean of the original, un-quantised pixels that carry the winning label**. Blend pixels carry the other label, so they are excluded.
  - Their critique of Pixel Snapper specifically: quantising everything to about 16 flat colours before pooling gives crisp votes but "throws real colors away (a rare highlight, a 1px accent) and is the weakest part of that tool."
  - "A final palette-clamp to a clean flat palette is deliberately left as future work."
- **Known limitations [source]**: extreme jitter with no lattice; slow drift counting (ACF locks to the *modal* spacing, not the count-defining *mean*; "needs a phase-tracking boundary counter (walker/DP)"); **dither-locked images** (within-cell dither is real signal below the true step, so the true step is structurally suppressed; "suspected unfixable when both axes carry the sub-lattice"); cells under 3 px.
- **Bearing**:
  - A: two-stage packing is the cleanest answer to "fuzzy fringes of intermediate colours": the fringe label never wins a centre-weighted cell vote, and the colour comes from interior pixels.
  - B: distillability is a ready-made candidate score; the coarse/fine asymmetry says candidate sets should lean fine.
  - C: tiles are used for voting only. They explicitly want a walker/DP for drift counting, which Pixel Snapper already has. That is a point in the walker's favour that the same document's "drop per-cut snapping" finding argues against. See 3.3.
- **Cost**: Rust core "typically well under a second"; the arbitration stage is quoted at 15-37 s on big images in the design doc (which implementation is not stated).
- **URLs**: https://github.com/Retro-Diffusion/pixel-art-fixer [opened] ; `docs/HOW_IT_WORKS.md` in that repo [source] ; https://retrodiffusion.ai/tools/pixel-art-fixer [snippet].

### 2.2 `rhinoc/xprite-pixelate` (2026)

- **What**: Grid recovery plus image-to-pixel-art conversion; claims the best numbers on its own benchmark.
- **Algorithm [opened, README]**:
  1. Estimate tilt; detect on a straightened copy.
  2. Seven detectors propose grids, plus a WASM port of pixel-art-fixer's autocorrelation and run-length vote.
  3. **Dynamic program per axis** places every cut on a colour edge: "Each cell stays within 0.75 to 1.25 of the pitch, and a change of width costs extra."
  4. Drop a candidate whose cuts miss colour edges that a coarser grid follows.
  5. **Choose by minimum description length**: the grid that describes the image in the fewest bits wins; a grid not phase-locked to content pays a penalty. Drift and sheared rows are modelled as strip shifts or a phase field.
  6. Sample one colour per cell (`TwoStage`, `MostCommon`, `Median`, `Average`, `KMeans`, `Contrast`...); adaptive palette snapping merges near-duplicate shades.
  7. Binarise alpha.
  - Extras: `removeNoise` ("light smoothing of similar neighbour colours", filter unspecified), `preserveDimensions` (nearest-upscale result back to source size), `prepass` using PixelOE outline expansion, `targetSize` / `blockSize` for arbitrary-size conversion.
- **Bearing**:
  - A: `preserveDimensions` is the "no-resize = snap then scale back" pattern.
  - B: MDL as a model-selection criterion across grids is the most principled ranking idea found. The README does not give the code-length formula; that needs a source read.
  - C: "strip shifts or a phase field" is the right abstraction for non-separable drift: a field of offsets on one mesh, not independent tiles.
- **Cost**: median 2.8 s per image on its AI suite, 0.27 s on pixel-bench (self-reported).
- **URL**: https://github.com/rhinoc/xprite-pixelate [opened].

### 2.3 `jenissimo/unfake.js` (Rust/WASM core since 2026, MIT)

- **What**: Browser tool; closest architectural sibling to this repo (Rust core compiled to WASM, no OpenCV).
- **Algorithms**:
  - Detection: `runs` (GCD of colour run lengths) and `edge` (tiled Sobel plus autocorrelation period); releases now default to the edge method because "runs-first auto routing proved less robust on real AI pixel art" [snippet, releases page]. Older article describes: pick the highest-variance tiles of a 3x3 split, Sobel profiles, histogram of peak spacings, then phase by scoring every offset `0..scale-1` as the sum of profile values on the comb [opened, dev.to].
  - **Elastic grid [source, `detect/elastic.rs` header]**: "Models that *draw* pixel art, rather than resample a canvas, jitter the cell size - 40..49 px columns around a 43 px period - so any fixed grid drifts half a cell somewhere and doubles or drops a pixel there. Per axis, a DP on a quarter-px lattice picks boundaries that sit on edges while keeping every cell within 0.75..1.3 of the period. The fit replaces the regular grid only on an axis where it captures clearly more edge energy; clean grids stay regular." Edge strength gets half credit one pixel off, normalised to its 90th percentile; quadratic cost per cell for leaving the period; skipped for small cells ("edges on nearly every pixel, nothing to fit").
  - Downscale [source, `downscale.rs`]: methods Nearest, Median, Mode, Mean, Dominant, plus `qvote` ("quantize to 24 colors (median-cut, no dither) then mode vote per block") and content-adaptive. Each cell is reduced over its **inner region** only: "pixels whose centers are at least a quarter cell from the borders", so border blur does not vote. `Dominant` uses a dominance threshold (0.15 in production) with mean fallback.
  - **Morphological cleanup [source, `morphology.rs`]**: open then close with a 2x2 kernel, per channel including alpha (erode = local min, dilate = local max). Has a unit test named `open_removes_single_speckle`.
  - **Jaggy cleaner [source, `utils.js`]**: on the 1x image, make a pixel transparent if it has zero opaque orthogonal neighbours and exactly one opaque diagonal neighbour. It is alpha-only; it does not look at colour.
  - Quantisation: libimagequant.
- **Bearing**:
  - A: the only tool with explicit despeckle code, and it is minimal: a 2x2 open/close and one alpha rule. Per-channel min/max morphology on RGB invents colours, so it has to run before quantisation or be redone on labels. The "inner region" vote is the simplest fringe suppressor.
  - B/C: the elastic-grid DP is a direct, bounded alternative to a greedy walker, with a guard that keeps clean grids regular.
- **Cost**: all O(N) or O(N * small).
- **URLs**: https://github.com/jenissimo/unfake.js/ [opened] ; source files under `crates/unfake-core/src/` and `browser-tool/lib/utils.js` [source] ; https://dev.to/jenissimo/how-to-tame-your-ai-pixel-art-3pk5 [opened] ; https://github.com/jenissimo/unfake.js/releases [snippet].

### 2.4 `KennethJAllen/proper-pixel-art`

- **Algorithm [opened, README]**: trim and flatten alpha; **nearest-upscale 2x** first; Canny; morphological closing; probabilistic Hough, keep near-axis-aligned lines, cluster; snap each line to the nearest peak of a gradient projection profile; pitch = median gap (fallbacks: profile-peak spacing, autocorrelation period); reconcile axes assuming square pixels; **validate by reconstruction error** (within-cell colour variance), replacing the estimate only when another "scores decisively better", as a guard against locking onto 2x the true width; fill the mesh by evenly subdividing gaps between detected lines, each interpolated line snapped to a nearby profile peak.
- Colouring: `palette` mode = most common quantised colour per cell; `dominant` mode (default) = dominant colour by offset binning, then near-duplicate colours merged by agglomerative clustering with a distance cutoff "so flat areas collapse to one color".
- Video/GIF: grid and palette are estimated once over sampled frames and shared.
- **Bearing**: B (validation by within-cell variance, independently arrived at); C (subdivide between trusted lines = a mesh that is locally adaptive but globally continuous; shared grid and palette across frames is the same argument as shared across tiles).
- **URL**: https://github.com/KennethJAllen/proper-pixel-art [opened].

### 2.5 `alexkorol/pixel-respecter` ("Pixel Perfecter")

- **Algorithm [opened, README]**: four detectors propose `(cell size, offset)`: exact nearest-neighbour check, Hough mesh (credited to proper-pixel-art), autocorrelation of boundary-mass projections, phase-fold sweep. All candidates are ranked by **one metric: inter-cell contrast divided by sqrt(intra-cell variance)**, which the README says peaks at the true grid and resolves 5-versus-10 px harmonics toward native. Fractional pitch triggers "mesh mode": sub-pixel pitch from autocorrelation peaks, lines snapped to boundary-mass peaks, variable mesh used directly.
- Colour: map to palette-label space ("never dithered"), modal colour per cell by bincount, **cell interiors margin-trimmed so "anti-aliased borders don't vote"**.
- `--mode adaptive`: "fits per-region grids when a single lattice fails". No detail on how regions are stitched.
- Diagnostics: reconstruction compared against source with halo/core separation; overlay modes `halo_suppressed`, `core_only`.
- **Bearing**: B (a second, simpler no-reference grid score); C (the only tool advertising per-region grids, but undocumented; needs a source read before relying on it); A (halo/core separation is a useful evaluation idea: measure error on cell cores separately from cell borders).
- **URL**: https://github.com/alexkorol/pixel-respecter [opened].

### 2.6 `theamusing/perfectPixel`

- **Algorithm [opened, README]**: (1) grid size from the FFT magnitude of the image, (2) Sobel edges, refine grid lines by aligning to edges within a search window (`refine_intensity`, recommended 0-0.5 of a cell), (3) sample per cell with `center`, `median` or `majority`. `fix_square` forces square output when nearly square.
- Third-party notes say the FFT path is limited to about 4-20 px cells by a hard-coded `max_pixel_size` and that auto-detection is unreliable on AI output [snippet, https://github.com/PeaarLi/ComfyUI-PixelPipeline].
- **Bearing**: FFT magnitude alone is the weakest detector in both vendor benchmarks. Not worth adopting.
- **URL**: https://github.com/theamusing/perfectPixel [opened].

### 2.7 Astropulse `pixeldetector` (2023) and K-Centroid

- **pixeldetector [opened, raw source via fetch]**: per-axis edge energy = Euclidean RGB norm of adjacent-pixel differences summed along the other axis; `scipy.signal.find_peaks` with no real constraints; pitch = **median of peak spacings**; target size = `round(W / hf) x round(H / vf)`; downscale by `kCentroid` with 2 centroids. Palette size by an elbow rule on distortion over `k = 1..max`.
- **kCentroid [opened]**: for each output cell, crop the source tile, quantise the tile to `k` colours, output the **most populous cluster's colour**. With `k = 2` this is "majority of a 2-colour reduction", which suppresses AA at tile edges.
- Shipped in the Retro Diffusion Aseprite extension as Edit > FX > "Repair pixel art", and separately as "K-Centroid Resize" [snippet].
- **Bearing**: B. K-centroid is a cheap per-cell robust colour estimate that works at any target size, so it is a natural non-native downscaler. Its failure mode is the same as majority vote: a minority detail in a cell disappears.
- **URLs**: https://github.com/Astropulse/pixeldetector [snippet] ; https://raw.githubusercontent.com/Astropulse/pixeldetector/main/pixeldetector.py [opened] ; https://github.com/Astropulse/K-Centroid-Aseprite [opened; README has no algorithm text] ; https://astropulse.gumroad.com/l/K-Centroid [snippet].

### 2.8 `KohakuBlueleaf/PixelOE`

- **What**: "Detail-oriented pixelization based on contrast-aware outline expansion". Image to pixel art, not grid recovery.
- **Algorithm [opened, README]**:
  - Outline expansion: grayscale; local median, local max and min per patch; build a weight map from two terms (brighter details in darker surroundings; distance between bright and dark extremes), normalised to 0-1; blend an **eroded** and a **dilated** copy of the image by that weight; then closing and opening to clean up. Effect: thin features are thickened to at least a cell before downscaling.
  - `contrast` downscale: in LAB, per patch on L pick min, max or centre depending on whether the patch distribution is skewed dark or bright (keeps the minority extreme); median filter on a and b.
  - Other modes: `center`, `k-centroid`, `bicubic`, `nearest`.
- **Bearing**: B. "Thicken thin features first, then downscale" is the practical trick for coarser-than-native candidates. Risk: it changes shapes, so it is a stylisation, not a restoration.
- **URL**: https://github.com/KohakuBlueleaf/PixelOE [opened].

### 2.9 Other tools

| Tool | What it does (as documented) | Bearing | URL |
|---|---|---|---|
| Pixel Rehab (pirillo) | Three detectors combined; low-confidence flag when they disagree; cuts snapped to boundaries; centre-weighted vote per cell; default 24 colours; manual cols/rows override. FAQ: "AI pixel grids drift. The cell size is rarely a clean whole number and often changes slightly across the image." No multi-resolution candidates, no despeckle | Confirms the mainstream recipe | https://arcade.pirillo.com/pixel-rehab.html [opened] |
| ComfyUI-PixelArt-Detector | Wraps pixeldetector; PIL quantise (MEDIANCUT / MAXCOVERAGE / FASTOCTREE), OpenCV and pycluster k-means/k-medians; `cleanup_colors` = "eliminate less used colors" below a usage threshold (suggested 0.01-0.05) | "Drop rarely used colours" is a palette-level despeckle. It removes rare *colours*, not rare *positions*, so it cannot fix speckle made of common colours | https://github.com/dimtoneff/ComfyUI-PixelArt-Detector [opened] |
| Aseprite Pixel Stylizer (itch.io script) | "Remove Isolated Spots": detects isolated pixels and patches of area <= N, blends into surrounding colours, "edge pixels are automatically protected"; also "Outline Dedup" for outlines thickened by sampling offset | Direct precedent for an area-threshold sieve with an edge-protection rule on the 1x image. Internals not documented | https://youhua-uncle.itch.io/aseprite-pixel-stylizer [opened] |
| Aseprite | `SpriteSize` methods: `nearest`, `bilinear`, `rotSprite`. Pixel-perfect is a **stroke** intertwiner that drops the middle point of L-shaped corners while drawing; it does not clean existing art. A "Despeckle (median filter)" FX entry is listed by one third-party page | No built-in pixel-art-aware downscale or stray-pixel removal found | https://www.aseprite.org/api/command/SpriteSize [opened] ; https://deepwiki.com/aseprite/aseprite/5.3-importexport [snippet] ; https://2dgames.jp/entry/aseprite-tips/ [snippet] |
| Pixelorama | Lists cleanEdge, OmniScale and rotxel transforms | Upscale/rotate algorithms; not relevant to downscale or cleanup | https://orama-interactive.itch.io/pixelorama [snippet] |
| Astropulse `sd-palettize` | k-means palette reduction (per a Hacker News commenter) | None beyond what this repo has | https://news.ycombinator.com/item?id=46058566 [opened] |
| `alvincrespo/pxl-cleanup` | Background removal, grid recovery via unfake, k-means palette, canvas placement | Pipeline glue only | https://github.com/alvincrespo/pxl-cleanup [snippet] |
| Upstream Pixel Snapper | README options today: colour count, `--pixel-size` override, `--palette` hex list, batch mode. No no-resize mode, no multi-resolution, no despeckle, no tiling | This fork is already ahead on palette modes | https://github.com/Hugo-Dz/spritefusion-pixel-snapper [opened] |

### 2.10 General-purpose image tools with relevant operators

| Tool / operator | What it does | Bearing on A | URL |
|---|---|---|---|
| ImageMagick `-mode geometry` | "Make each pixel the 'predominant color' of the neighborhood" | Mode (majority) filter. Correct operator for label maps | https://imagemagick.org/command-line-options/ [opened] |
| ImageMagick `-statistic Mode` | "mode (most frequent) value **per channel** in neighborhood" | Per-channel mode on RGB can output a colour that is in no input pixel. Must be applied to a palette-index image, not RGB | same [opened] |
| ImageMagick `-paint radius` | "Each pixel is replaced by the most frequent color in a circular neighborhood" | Same idea, circular window | same [opened] |
| ImageMagick `-kuwahara` | "Edge preserving noise reduction filter" | See 5.5 | same [opened] |
| ImageMagick `-mean-shift WxH+distance` | "Image noise removal and color reduction/segmentation"; iterated until convergence | See 5.3 | same [opened] |
| ImageMagick `-connected-components` with `connected-components:area-threshold=N` | Labels 4- or 8-connected regions; regions under the area threshold are merged into their surrounding region; `mean-color=true` to output colours | The canonical "remove small islands" operator. Caveat in the docs about stacked objects; a docs example runs it twice | https://imagemagick.org/connected-components/ [snippet] ; https://github.com/ImageMagick/ImageMagick/discussions/5163 [snippet] |
| GDAL `gdal_sieve` | "removes raster polygons smaller than a provided threshold size (in pixels) and replaces them with the pixel value of the largest neighbour polygon"; `-4` / `-8` connectedness | Same operator from remote sensing, with a defined fill rule (largest neighbour) | https://gdal.org/en/stable/programs/gdal_sieve.html [opened] |
| ArcGIS "Generalization of classified raster imagery" | Recommended chain: Majority Filter (isolated cells), Boundary Clean (smooth zone edges), Region Group plus area threshold, **Nibble** (fill removed regions from the nearest valid cell) | A complete, field-tested despeckle recipe for categorical rasters. Their fill rule (nearest) differs from GDAL's (largest neighbour) | https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/generalization-of-classified-raster-imagery.htm [snippet] |
| pngquant / libimagequant | `--floyd=N` (0-1), `--nofs`/`--ordered`, `--posterize bits`. Source comments say dithering is restricted using `edges`, `noise` and `dither_map` buffers because dithering on edges looks jagged and textured regions hide error anyway; the level is reshaped non-linearly | For A the relevant setting is simply dithering 0. `--posterize` reduces colour precision, not spatial speckle. Their edge/noise maps are a precedent for *selective* treatment by local texture | https://www.mankier.com/1/pngquant [snippet] ; https://github.com/lovell/libimagequant/blob/main/libimagequant.c [snippet] ; https://pngquant.org/lib/ [snippet] |

### 2.11 `Retro-Diffusion/pixel-bench` (MIT)

- **What**: An open benchmark for pixel-art reconstruction. You supply true 1x pixel art; a distortion engine damages it deterministically; methods are scored against exact ground truth. Bundled methods: `fixer`, `snapper` (a port of PixelSnapper), `pixeldetector`, `unfake` (optional), `naive`.
- **Distortions [source, `docs/DISTORTIONS.md`]**: 43 types in suite v1 including `clean_nn`, `fractional`, `nonsquare`, `soft_bilinear`, `soft_bicubic`, `jpeg`, `noise`, `drift` ("cell size smoothly grows and shrinks across the image"), `cell_texture` and `painterly` (in-cell texture), `dead_cells` ("scattered single wrong-coloured native pixels"), `break_outlines`, `mush` ("sub-cell displacement and soft focus over a bilinear grid"), `mush_warp`, `kitchen_sink`. The engine keeps per-pixel source-coordinate maps `U`/`V`.
- **Metrics [source, `docs/METRICS.md`]**: see section 9.1.
- **Bearing**: this is the ready-made harness for an automated experiment loop on B, and the `cell_texture` / `painterly` / `mush` categories are the closest synthetic stand-ins for the speckle problem in A. It measures 1x reconstructions; a no-resize output would be scored by nearest-downsampling it to the true size, or by upscaling ground truth and comparing at full size (my synthesis).
- **URL**: https://github.com/Retro-Diffusion/pixel-bench [opened] plus docs [source].

### 2.12 What the vendor benchmarks say (treat with care)

| Source | Suite | Pixel Snapper (upstream) exact size | Best tool exact size |
|---|---|---|---|
| pixel-art-fixer README | pixel-bench, 4,300 inputs | 4% (within 1 px: 19%; grid alignment 63%) | pixel-art-fixer 77% |
| xprite-pixelate README | pixel-bench suite v1, 7,651 cases | 6.6% | Xprite 87.1%, pixel-art-fixer 77.8% |
| xprite-pixelate README | its own "AI suite", 1,746 cases | 56.8% | Xprite 83.6%, proper-pixel-art 72.2%, pixel-art-fixer 49.7% |

- The two suites disagree sharply about the ranking of everything except the author's own tool, and the case counts for "pixel-bench suite v1" differ between the two READMEs (4,300 versus 7,651). xprite's own README warns "Synthetic damage does not establish real-world success rates."
- Robust takeaway: on pixel-bench's synthetic upscale-and-damage suite, upstream Pixel Snapper rarely lands on the exact native size; on xprite's AI-style suite it is mid-pack. Why the two differ so much was not investigated. In xprite's per-category table Pixel Snapper is weakest on "Small cells (4-8 px)" (42.6%) and "Drift and WebP" (48.6%), which overlaps this project's target range.
- xprite also reports a "Fringe (alpha 1-254)" rate of 91% for Pixel Snapper, meaning partially transparent output pixels. Whether this fork still does that was not checked.
- URLs: https://github.com/Retro-Diffusion/pixel-art-fixer [opened] ; https://github.com/rhinoc/xprite-pixelate [opened].

---

## 3. Grid detection on fuzzy pixel art

No academic treatment specific to pixel art was found beyond the MDPI block-size paper (1.6). The following is distilled from the tools above.

### 3.1 Signals

| Signal | How | Strength | Weakness | Source |
|---|---|---|---|---|
| First-difference profile, peak spacing median | Sum of gradient norm per column/row; `find_peaks`; median gap | Trivial | Any texture inside cells shifts the median; no sub-pixel; this is the pixeldetector approach and scores near the naive baseline | 2.7 |
| Second-difference feature | `abs(d2 luminance)` | Survives bilinear/bicubic blur, where first differences are flat | Noise-sensitive | 2.1 |
| Banded autocorrelation | Sum feature over row bands, then ACF, then comb-minus-anti-comb | Phase-invariant; sub-percent step error reported | Locks to the modal spacing under drift | 2.1 |
| Run-length GCD / soft-GCD comb | Distances between colour changes | Exact on clean upscales | Brittle on AI output (unfake demoted it) | 2.1, 2.3 |
| Shift self-similarity | Dissimilarity of image with itself shifted by `t`; dips at multiples of `s` | Works without edges | Texture periodicity | 2.1 |
| FFT magnitude peak | Peak of 2D or profile spectrum | Simple | Harmonics; limited cell-size range in the one implementation found | 2.6 |
| Canny + Hough mesh | Lines, cluster, median gap | Finds actual line positions, not just a period | Needs clear edges; proper-pixel-art upscales 2x first to help | 2.4 |
| Within-cell variance / reconstruction error | Summed-area tables over candidate `(s, p)` | Directly measures "does this grid explain the image" | Monotone in `s` unless normalised; see 2.1 for the fix | 2.1, 2.4, 2.5 |

### 3.2 Choosing among divisors and multiples

- pixel-art-fixer: anti-phase and trend factors (2.1); final rule "the smallest qualifying step wins"; a penalty for steps above the "detail scale".
- proper-pixel-art: replace the estimate only when an alternative "scores decisively better" on within-cell variance, explicitly to avoid 2x lock-in.
- pixel-respecter: inter-cell contrast / sqrt(intra-cell variance).
- xprite: MDL, plus dropping a candidate whose cuts miss edges that a coarser grid follows.
- **Dither is the adversary**: a 2-cell checkerboard has real energy at `2s`, and in-cell dither has real energy at `s/2`. pixel-art-fixer lists dither-locked images as probably unfixable when both axes carry the sub-lattice. For B this is a reason to *show* both octaves to the user instead of trying to decide (my synthesis).

### 3.3 Drift, non-square cells, phase

- **Non-square**: all serious tools estimate per axis and treat square as a soft prior. pixel-art-fixer cites real `6.38 x 8.00` cells.
- **Phase**: unfake's older pipeline scores each integer offset by comb sum. pixel-art-fixer avoids global phase in detection altogether because sprite sheets have per-region phase.
- **Drift, three schools**:
  1. *Even lattice, get the count right* (pixel-art-fixer default). Argument: gradient maxima on detailed art are content edges, and snapping cuts to them cascades into sliver and double-wide cells.
  2. *Bounded DP per axis* (unfake elastic grid, xprite). Cells constrained to 0.75-1.25 or 0.75-1.3 of the pitch, quadratic or extra cost for width changes, edge reward with half credit one pixel off, and a gate: use the elastic fit only where it captures clearly more edge energy than the regular grid.
  3. *Anchor lines plus subdivision* (proper-pixel-art, pixel-respecter mesh mode). Trust detected lines; subdivide evenly between them; snap interpolated lines to nearby profile peaks.
- These conflict on whether to snap cuts at all. The reconciling observation (my synthesis): school 1 is right when the source is a *resampled canvas* (grid truly regular, edges noisy), school 2 when the model *drew* irregular cells. unfake's gate (compare edge capture of elastic versus regular) is the explicit test for which regime an image is in. Pixel Snapper's walker is a greedy version of school 2 without the gate or the global optimality.
- **Non-separable drift** (x-pitch varying with y): only xprite mentions handling it ("strip shifts or a phase field"). A per-axis mesh cannot represent it. This is the one place where region-wise estimation is necessary (section 8).

---

## 4. Label-map regularisation (question A)

Why speckle appears with every quantiser (my synthesis, consistent with the owner's diagnosis): nearest-palette assignment is a per-pixel decision with no spatial term. A fuzzy blob's pixels form a cloud in colour space; wherever that cloud straddles a Voronoi boundary between two palette entries, the label map flickers. Changing the palette moves the boundaries, not the fact that clouds straddle them. Fringes are the same effect on the AA ramp between two regions: the ramp passes through whatever palette entries lie between the two endpoint colours.

### 4.1 Potts / MRF energy on the label map

- **What**: Minimise `E(L) = sum_p D_p(L_p) + beta * sum_{(p,q) neighbours} [L_p != L_q]`, with `D_p(l)` = colour distance from pixel `p` to palette entry `l`.
- **Direct precedent**: Lacroix and Idrissa, "Optimal palette extraction as part of scanned graphics vectorization" (CGIV 2010) do exactly this after palette extraction [source, PDF text]. Start from nearest-palette labels; data term = colour distance (they use max of Euclidean and CMC in L\*a\*b\*); constraint term = count of 4- or 8-neighbours with a different label; `beta` trades them off. Optimisation by exhaustive per-pixel label search (that is ICM) or simulated annealing. They note the Potts term "favours more regions than lines, and will avoid to keep isolated points unless E1 is very low."
- That note is the crux for pixel art: Potts penalises boundary length, so it erodes thin lines and rounds corners. At full resolution a 1-logical-pixel line is `s` real pixels wide, so there is a usable window: `beta` large enough to kill 1-3 px speckle but too small to remove an `s`-wide feature. The window narrows as `s` approaches 4.
- **Solvers**:
  - **ICM** (Besag 1986): sweep pixels, set each to the label minimising its local energy; greedy local minimum [snippet, https://vision.middlebury.edu/MRF/eccv06/SZSVKATR-ECCV06.pdf]. Cost O(N * K) per sweep; a handful of sweeps. Trivial to hand-roll in Rust with no dependencies.
  - **Alpha-expansion** (Boykov, Veksler, Zabih, PAMI 2001): for each label, solve a binary min-cut deciding which pixels switch to it. Much stronger minima than ICM on Potts energies; factor-2 bound for Potts is [unverified, from memory]. Needs a max-flow implementation; K min-cuts on a 1e6-node graph per sweep. My estimate is seconds to tens of seconds in WASM [unverified].
  - **Pottslab** (Storath and Weinmann): fast solver for the L2-Potts / piecewise-constant Mumford-Shah problem on vector-valued images; MIT; "linear complexity in the number of channels" [opened]. It works on *colours* not labels, so it is a prefilter (section 5), and it is Matlab/Java.
- **Bearing**: A. With ICM this is a "data-aware mode filter": unlike a plain mode filter it will keep a small region whose colour is strongly distinct (high cost to relabel) and remove one that is nearly ambiguous (low cost). That is the right behaviour for "protect real details, remove quantisation flicker".
- **Does not help with**: making edges blocky or grid-aligned. Potts minimises boundary length, which straightens edges but also rounds corners. It has no notion of the grid.
- **URLs**: https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cgiv/5/1/art00051 [source] ; https://www.csd.uwo.ca/~yboykov/Papers/pami01.pdf [snippet] ; https://github.com/mstorath/Pottslab [opened].

### 4.2 Mode / majority filter

- **What**: Replace each label by the most frequent label in a window. ImageMagick `-mode`, ArcGIS Majority Filter (2.10).
- **Algorithm**: sliding window histogram of K bins; O(N * K) with incremental updates, or O(N * w^2) naive.
- **Behaviour**: removes features narrower than about half the window; rounds corners at the window radius. Equivalent to one ICM sweep with a zero data term.
- **Bearing**: A. Works when window radius `r` satisfies speckle size < `r` < `s/2`. Blind to how confident the original assignment was, so strictly dominated by ICM with a data term, but cheaper and easier to reason about.
- **Scale it by `s`** (my synthesis): `r = max(1, round(s/4))`. For `s = 4` that is a 3x3 window, which only removes isolated single pixels.

### 4.3 Small connected-component removal (sieve)

- **What**: Label connected components of equal palette index; reassign components below an area threshold.
- **Precedents**: ImageMagick `area-threshold`, GDAL sieve (fill with largest neighbour), ArcGIS Region Group + Nibble (fill from nearest), Aseprite Pixel Stylizer "Remove Isolated Spots (area <= N)" with edge protection (2.9, 2.10).
- **Algorithm**: union-find or two-pass labelling, O(N). For each small component choose the fill label as the neighbour label with the longest shared border (or the largest neighbour, or the lowest total data cost). Process smallest-first so merged islands can grow past the threshold; repeat until stable.
- **Bearing**: A, and the best fit for "do not destroy intentional detail", because the criterion is area in real pixels and the intentional features have area about `s^2`. Threshold as a fraction of `s^2` (my synthesis): start around `0.25 * s^2`. A full-resolution dither checkerboard is made of `s x s` blocks, each far above the threshold, so it survives; so does a 1-logical-pixel highlight.
- **Does not help with**: fringes. An AA fringe between two regions is 1-2 px wide but can run the whole length of an edge, so its area is large. It needs a width test: remove a component (or strip) whose maximum inscribed width is below about `s/3`, splitting it between its two neighbours along its medial line (my synthesis). An extra safety test for fringe-ness: its palette colour lies near the segment between the two neighbouring regions' colours in OKLab, which a real outline of a distinct colour will not.
- **Caveat**: fuzzy blobs are not full `s x s` squares after quantisation; a real single-pixel detail may quantise to an area well under `s^2`. The threshold needs tuning against data, and an ICM-style data-cost check before deleting is a sensible guard.

### 4.4 Morphology on label maps

- unfake does a 2x2 open then close per channel (2.3). On RGB this invents colours; on a K-label map, greyscale morphology is not defined without an ordering.
- The label-safe equivalent is per-label binary opening (remove parts of each label mask thinner than the structuring element) followed by filling the vacated pixels from neighbours, which is the width test of 4.3 in other words.
- Cost O(N * K) naive. Not worth it over the sieve plus width rule, in my assessment.

### 4.5 Total-variation-regularised or assignment-flow labelling

- Convex relaxations of Potts (TV on label indicator functions) and "Image Labeling by Assignment" exist as alternatives to graph cuts [snippet, https://arxiv.org/pdf/1504.05776 ; https://arxiv.org/pdf/1603.05285]. They need iterative proximal solvers over an `N x K` tensor (16 million floats here).
- No evidence they beat ICM plus sieve for this use, and they share Potts' corner-rounding. Not recommended.

### 4.6 PosterChild (Chao, Singh, Gingold, EGSR 2021)

- Posterisation by palette extraction, multi-label optimisation into rough solid regions, then blend refinement; boundary smoothing by frequency-guided median filtering away from detail; about a minute per image [snippet].
- Confirms that the graphics literature solves "flat regions from a palette" with multi-label optimisation, but it aims at smooth poster shapes, the opposite of blocky. Reference only.
- URL: https://cragl.cs.gmu.edu/posterchild/ [snippet].

### 4.7 Spatial colour quantisation (Puzicha et al.) and why it makes things worse

- **What**: Puzicha, Held, Ketterer, Buhmann, Fellner, "On spatial quantization of color images" (IEEE TIP 9(4), 2000; ECCV 1998 version). Implemented as `scolorq`, ported to Rust as `rscolorq`.
- The rscolorq README defines it as "simultaneously performing halftoning (dithering) and color quantization" [opened]. Palette and assignment are optimised under one cost based on a simplified model of human perception, by deterministic annealing with a multiscale scheme [snippet].
- Mechanism [unverified, from memory of the paper]: the cost compares the *low-pass filtered* original with the *low-pass filtered* quantised image. Any high-frequency arrangement of labels whose local average matches the original has near-zero cost, so the optimiser spreads error into fine dither. Consistent with rscolorq exposing a filter size (1, 3 or 5) and warning that 1x1 "doesn't produce an image resembling the input" [opened].
- **Bearing**: A, negatively. This quantiser is rewarded for speckle. The repo's `spatial_quantize` mode should be expected to be the *worst* offender, and no tuning of it addresses A. The idea can be inverted, though: a Potts term is the "anti-dither" counterpart, penalising label changes instead of forgiving them.
- **URLs**: https://github.com/okaneco/rscolorq [opened] ; https://pubmed.ncbi.nlm.nih.gov/18255438/ [snippet] ; https://link.springer.com/content/pdf/10.1007/BFb0055690.pdf [snippet].

### 4.8 Grid-aware regularisation (what the tools actually do)

- Every tool in section 2 regularises with the strongest possible prior: **the label is constant per grid cell**. Majority (or centre-weighted, or inner-region) vote per cell is the exact minimiser of a Potts energy with infinite within-cell coupling and zero between-cell coupling.
- Rendering that back at source size on the detected mesh (xprite `preserveDimensions`) gives flat regions and blocky edges by construction, keeps every 1-cell detail that wins its cell, and keeps cell-level dithering.
- What it loses: sub-cell detail, and anything the grid got wrong (a detail straddling two cells is split or dropped).
- This is the only approach found that produces *blocky* edges. Per-pixel methods (4.1-4.3) produce flat regions with smooth or ragged edges that follow the blob outlines.
- Middle ground (my synthesis): use the detected mesh as a **soft** prior. For example ICM where the pairwise weight is high for neighbours inside the same cell and low across a detected cut line. Cells then become flat unless the data strongly disagrees, and edges prefer to sit on cut lines. No source does this; it is a hypothesis to test.

---

## 5. Edge-preserving prefilters and superpixels (before quantisation)

General point (my synthesis): prefiltering in colour space helps the *palette* as well as the label map, because AA ramp colours and noise stop competing for palette entries. Post-filtering in label space cannot do that. The two are complementary.

| Filter | Core algorithm | Effect on fuzzy pseudo-pixels | Risk | Cost | URL |
|---|---|---|---|---|---|
| **Median** | Per-channel or vector median in a window | Removes speckle smaller than half the window; keeps step edges | Rounds corners; deletes lines thinner than half the window; per-channel median invents colours | O(N * w) with histograms | https://imagemagick.org/command-line-options/ [opened] |
| **Bilateral** (Tomasi and Manduchi 1998) | Weighted mean with spatial and range Gaussians | Flattens in-region noise | Does **not** remove an AA ramp: ramp pixels are similar to their ramp neighbours, so the ramp survives and still quantises to a fringe | O(N * w^2) naive | No URL opened; standard reference (ICCV 1998) seen only as a citation in search results. Behavioural claims here are [unverified], from general knowledge |
| **Guided filter** (He, Sun, Tang) | Local linear model of output in terms of a guide; box filters only; O(N) regardless of radius | Like bilateral without gradient reversal | Same ramp problem; preserves soft edges as soft | O(N) | https://en.wikipedia.org/wiki/Guided_filter [snippet] |
| **Mean shift** (Comaniciu and Meer 2002) | Each pixel iterates to the local mode in joint space-colour | Collapses each blob to its mode colour, a true piecewise-constant result; Kopf et al. used it (`h_s = 4`) for pixel-art palette reduction on cartoon/vector input | Kopf et al.: does not work well on natural images. Slow. Colour bandwidth is image-dependent | O(N * w^2 * iters) | https://johanneskopf.de/publications/downscaling/paper/downscaling.pdf [source] ; https://imagemagick.org/command-line-options/ [opened] |
| **L0 gradient smoothing** (Xu, Lu, Xu, Jia, SIGGRAPH Asia 2011) | Minimise `sum (S - I)^2 + lambda * #{p : grad S_p != 0}` by half-quadratic splitting. Alternate: (h,v) = (0,0) where `(dxS)^2 + (dyS)^2 <= lambda/beta`, else the gradient; then solve for S with FFTs; `beta` from `2*lambda`, multiplied by `kappa = 2` until `1e5`; 20-30 iterations | Globally removes low-amplitude gradients and **sharpens ramps into steps**. This is the same model as Potts on colours | Needs an FFT (the repo has none); can posterise gentle shading into bands; 3 s for a 600x400 single channel in 2011 Matlab | O(N log N) per iteration | https://legacy-sites.tlt.harvard.edu/~cs278/papers/l0.pdf [source] |
| **Kuwahara** | Output = mean of whichever of four overlapping quadrants around the pixel has the lowest variance | Turns ramps into steps and flattens interiors; cheap | Quadrant size must stay below about `s/2` or 1-cell details vanish; classic blocky artefacts (possibly welcome here); noise-sensitive selection | O(N) with summed-area tables | https://en.wikipedia.org/wiki/Kuwahara_filter [snippet] ; https://isis.astrogeology.usgs.gov/8.3.0/Application/presentation/Tabbed/kuwahara/kuwahara.html [snippet] |

- **Assessment**: the two filters that attack the fringe (ramp to step) are L0 smoothing and Kuwahara; mean shift does too but at high cost. Bilateral and guided filters are the commonly suggested ones and are the least useful for A, because the fringe is a smooth ramp, which is exactly what they preserve.
- Kuwahara with radius scaled to `s` is the cheapest experiment. L0 is the most principled but needs an FFT and a `lambda` sweep.

### 5.1 Superpixels (SLIC)

- **SLIC** (Achanta et al., TPAMI 2012): k-means in 5-D (L, a, b, x, y) with distance combining colour and position terms, compactness `m`, search limited to a window proportional to superpixel size, so linear in N [snippet, https://infoscience.epfl.ch/record/177415/files/Superpixel_PAMI2011-2.pdf].
- **For pixel art**: Gerstner (1.1) is the lattice-initialised, high-compactness variant with one superpixel per output pixel; Shang and Wong use SLIC for portraits (1.6).
- **Bearing**:
  - A: seeding SLIC on the detected mesh (one seed per cell, `S = s`) gives regions that follow blob outlines. Assigning each superpixel one palette label removes in-blob speckle. Edges follow blobs, so they are not blocky.
  - C: this is the "adaptive logical pixel" variant. It handles local drift without any tiling, since each superpixel moves independently, with Laplacian smoothing to keep the topology.
- **Skeptical note**: SLIC has no rectangularity or row/column alignment term, and at `s = 4-6` a superpixel is 16-36 pixels, where SLIC is at the edge of being meaningful. No tool in section 2 uses superpixels for restoration; they all use rectangular cells.
- **Cost**: O(N) per iteration, about 10 iterations.

---

## 6. Downscaling to a non-native grid (question B)

### 6.1 Candidate selection

No source generates a *set* of candidate resolutions for audition; all pick one and offer a manual override (Pixel Rehab, upstream Pixel Snapper `--pixel-size`, perfectPixel `grid_size`). What the sources do support:

- **Octave neighbours are the meaningful alternatives.** The dominant ambiguity is x2 or /2 of the detected pitch (2.1). Offering `s*`, `s*/2` and `2*s*` covers the detector's most likely mistakes.
- **Lean fine.** Too coarse is visibly worse; slightly too fine is nearly lossless (2.1).
- **Per-axis.** Cells are not always square, so candidates should be `(cols, rows)` pairs, not a single size.
- **Detector disagreement is a candidate source.** Tools that run several detectors (2.1, 2.2, 2.5) already have a pool; the pool, deduplicated, is the candidate list.
- **Fixed targets (128/164/256)** have no support in the literature as such. Their distance from the native grid in units of cells determines how they should be produced (6.2).

### 6.2 Producing a candidate at target size `T` when native is `n`

| Case | Method | Why | Source |
|---|---|---|---|
| `T = n` | Per-cell label vote, colour from winning-label originals | Standard | 2.1 |
| `T = k * n`, integer `k` | Reconstruct at `n`, nearest-upscale by `k` | Exact, no decisions | 1.6 (MDPI: only integer multiples preserve structure) |
| `T` slightly different from `n` (within a few %) | Resample the **source** on an even lattice of `T` cells with the same cell vote | This is just a different count hypothesis; pixel-art-fixer's even lattice does exactly this | 2.1 |
| `T > n`, non-integer ratio | Even lattice of `T` cells on the source, cell vote. Logical pixels come out with alternating widths (1 and 2 output pixels) | Unavoidable. This is "pixel wobble" / "mixels". Nearly lossless but visibly uneven | 1.6 ; https://www.theseus.fi/server/api/core/bitstreams/1b8afc42-3592-4d68-8ffd-1b746e0db39a/content [snippet] |
| `T < n` (coarser than native) | Information must be discarded. Options: majority / k-centroid (drops minority detail), DPID or PixelOE `contrast` (keeps minority extremes), PixelOE outline expansion first (thickens thin features), Kopf content-adaptive (keeps lines connected, slow), Gerstner (joint with palette) | Here the choice of kernel is the whole question | 1.1-1.4, 2.7, 2.8 |

- Two-stage ordering matters for `T != n` (my synthesis): going source to `T` directly lets fuzzy source blobs vote; going source to `n` to `T` votes on clean cells. For `T < n` the second is better behaved: reduce the clean `n` image with a detail-aware rule (for example: a `2x2` block of native cells takes the majority label unless a minority label forms a feature that would otherwise disappear entirely from the output).
- Non-integer ratios below native have no crisp solution in any source. The MDPI paper's premise is that only integer multiples or exact reciprocals keep block structure [snippet].

### 6.3 Ranking candidates and flagging lost detail

- **Rank by grid fit**: distillability (2.1), contrast / sqrt(variance) (2.5), MDL (2.2). All are computed on the *source* and a candidate grid, need no ground truth, and cost O(N) per candidate with summed-area tables.
- **Caution**: these scores answer "is this the native grid", not "does this look best". A coarser candidate that drops a tiny highlight may score well. Ranking should be a default ordering for audition, not a filter.
- **Flag lost detail** (my synthesis, no source does this):
  - *Component survival*: compute connected components of the full-resolution label map (after despeckle). For each component of area at least about `0.25 * s^2`, check whether its label survives in any output cell it overlaps. Components with zero surviving cells are lost details; draw their bounding boxes on the candidate.
  - *Minority mass*: per output cell, the fraction of source pixels not carrying the winning label, restricted to the cell's inner region. High values in compact clusters mean a cell is covering two things.
  - *Round-trip residual*: upscale the candidate to source size, compare with the quantised source, and report error on cell cores separately from cell borders (pixel-respecter's halo/core split). Core error means lost content; border error is just AA.

---

## 7. Dithering: detecting and protecting it

- No pixel-art-specific dither detector was found.
- Classical work: recognising ordered-dither versus error-diffusion patterns (https://ieeexplore.ieee.org/abstract/document/7858502/ [snippet]); inverse halftoning given a known mask (https://www.researchgate.net/publication/221337370_Fast_Inverse_Halftoning_Algorithm_for_Ordered_Dithered_Images [snippet]); a patent on edge detection inside dithered images (https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6473525 [snippet]). All assume true 1-pixel halftone dots, not `s`-pixel blobs.
- What the pixel-art sources say:
  - Gerstner et al. avoid dithering entirely and call selective dithering future work [source].
  - pixel-art-fixer: dither is real signal at `s/2` or `2s` and breaks pitch detection [source].
  - Artist conventions: an "orphan pixel" is an isolated pixel with no cluster contact, and is discouraged except as a small point of interest; pixels belonging to AA or shading curves are exempt (https://saint11.art/blog/glossary/ [snippet]). Derek Yu's cleanup step deletes stray pixels and thins lines to 1 px (https://www.derekyu.com/makegames/pixelart.html [snippet]).
- **Practical rule** (my synthesis): at full resolution, intentional dither is a periodic alternation at period `2s` of blocks each about `s x s`; quantisation speckle is aperiodic and sub-`s`. An area threshold in units of `s^2` separates them without any explicit dither detector. On the 1x image the distinction is harder: a despeckler there should exempt pixels in a local checkerboard (each of the 4-neighbours has the *same* other label) and pixels on valence-2 chains (1.5).

---

## 8. Tiled / blockwise processing (question C)

### 8.1 Precedents for per-block palettes, and their seams

| System | Local-palette rule | What happens at block borders | URL |
|---|---|---|---|
| NES background | Each 16x16 px area picks one of four 3-colour subpalettes plus a shared backdrop colour | Artists design around it; converting arbitrary images shows blocks | https://www.nesdev.org/wiki/PPU_attribute_tables [opened] |
| ZX Spectrum | Two colours per 8x8 cell | "Attribute clash", a named visual artefact | https://www.wikidata.org/wiki/Q4231817 [snippet] |
| BTC / S3TC (BC1) | Per 4x4 block: two endpoint colours plus two interpolated, 2-bit index per texel; block palette is collinear in RGB | Banding and block artefacts, worst on "hand-drawn cartoon-style art" | https://en.wikipedia.org/wiki/S3_Texture_Compression [snippet] |
| Tile palette quantisers (rilden `tiledpalettequant`, Aikku93 `tilequant`) | Global set of `P` palettes; each tile assigned to one. rilden's update: sample a random tile, find its best palette, sample a colour from the tile, pull the nearest entry of that palette toward it; distance `2dR^2 + 4dG^2 + dB^2` | Users reported visible square artefacts at tile boundaries, especially in dark areas; the author's mitigation was dithering; a proposed mitigation was to up-weight error near tile borders | https://forums.nesdev.org/viewtopic.php?t=24117&start=15 [opened] ; https://github.com/Aikku93/tilequant [snippet] |

- **Lesson**: even when tile palettes are optimised *jointly* (the tile quantisers share a small set of palettes and co-optimise assignment), seams are visible. Fully independent per-tile palettes are strictly worse: the same flat region crossing a tile border gets two slightly different colours, which is the most visible possible artefact in flat-shaded art.
- The useful structure in the tile quantisers is "a small global family of palettes, local choice among them", not "a palette per tile".

### 8.2 Seam handling in blockwise processing generally

- **CLAHE** is the standard example of doing it right: statistics are computed per tile, but each pixel's mapping is a **bilinear blend of the neighbouring tiles' lookup tables**, so the *parameters* vary smoothly while the output is computed once [snippet, https://doi.org/10.3390/math13203282]. Even so, an implementation issue shows blockiness appears as soon as border tiles are handled inconsistently [snippet, https://github.com/boostorg/gil/issues/513].
- **Principle** (my synthesis): interpolate parameters across tiles, never blend or butt-join outputs. For a discrete output (palette labels, cell boundaries) blending outputs is not even defined, so overlap-and-feather cannot work here. An overlap/halo helps only as context for *estimation*.

### 8.3 What the pixel-art tools do

- **Tiles for estimation only**: pixel-art-fixer votes per tile and takes the area-weighted harmonic mean of local steps under drift, then builds one grid [source]. unfake selects the highest-variance tiles of a 3x3 split for its Sobel profiles [opened].
- **One continuous mesh**: unfake and xprite fit variable-width cells by per-axis DP; proper-pixel-art subdivides between anchor lines. A per-axis mesh is seamless by construction.
- **Phase field / strip shifts**: xprite's stated model for drift and sheared rows [opened]. Not documented in detail.
- **Per-region grids**: pixel-respecter `--mode adaptive` "fits per-region grids when a single lattice fails" [opened]. No description of stitching. This is the only instance found of independent region grids, and it is positioned as a fallback.
- **Shared palette and grid across frames**: proper-pixel-art and pixel-respecter both estimate once and share across a sprite sheet or video [opened]. Same logic applies across tiles.

### 8.4 Assessment of the variants in the question

| Variant | Verdict | Reasoning |
|---|---|---|
| Independent tiles, own palette each | Unsound | Seams in flat regions; all precedents in 8.1 |
| Independent tiles, own grid each, butt-joined | Unsound in general | Cell rows/columns fail to meet at the seam; a logical pixel straddling the seam is split into two partial cells. Tolerable only where the seam falls on background |
| Overlap / halo | Helps estimation, not stitching | Discrete outputs cannot be feathered. A halo gives each tile enough context for a stable pitch estimate |
| Global palette + local assignment | Sound, and it is the default already | "Local assignment" with a spatial prior is section 4 |
| Local palettes, then merged | Sound and potentially useful | Extract candidate colours per region, merge near-duplicates globally (agglomerative with a distance cutoff as in proper-pixel-art, or xprite's adaptive snapping), assign with the merged palette. Benefit: a small, locally important colour (a 1-cell red gem) gets a palette entry it would lose in a global k-means. Addresses pixel-art-fixer's critique that global 16-colour quantisation discards rare accents |
| Per-tile *estimation* of pitch/phase, one interpolated mesh | Sound; best match to how drift behaves | CLAHE-style: estimate `(s_x, s_y, phase)` per tile with a halo, smooth the field, integrate to cut positions. Handles non-separable drift that a per-axis walker cannot |
| Quadtree / variable-size logical pixels | Not recommended | No pixel-art precedent found. Mixed pixel sizes are a named flaw ("mixels") in the community, "commonly considered a visual flaw" [snippet, https://www.theseus.fi/server/api/core/bitstreams/1b8afc42-3592-4d68-8ffd-1b746e0db39a/content]. Output is also no longer a plain low-resolution image |
| Superpixels as adaptive cells | Sound for flattening, not for blocky output | Section 5.1 |

- **Sizing note** (my synthesis): a tile needs enough cells for a stable period estimate. With `s` about 8, a quadrant of a 1024 image has 64 cells per side, which is plenty; a 128 px tile has 16, which is marginal for autocorrelation and too few when the tile is mostly flat. pixel-art-fixer's "dispersion-adaptive tile voting" and unfake's "pick high-variance tiles" both exist because many tiles carry no grid signal at all.

---

## 9. Metrics for an automated experiment loop

### 9.1 With ground truth (synthetic damage of true pixel art)

From pixel-bench [source, `docs/METRICS.md`]:

| Metric | Definition | Notes |
|---|---|---|
| `exact`, `within1`, `within5pct`, `rel_err` | Recovered `cols x rows` versus truth | For B's native-grid candidate |
| `delta_e`, `psnr` | Reconstruction nearest-resampled to true size, mean CIELAB dE76 / PSNR against the original | "Colour error over all samples flatters methods that over-segment", so also report it on the exact-size subset |
| `pixel_match` | Share of native pixels whose colour matches exactly (tiny tolerance) | Sensitive to palette choice |
| `grid_align` | F1 of implied grid lines against true cut positions, one-to-one matching | Independent of colour and of getting the count exactly right |
| `ncolor_err`, `palette_de` | Palette size error; mean distance from each recovered colour to nearest true colour | |

- For A (no-resize), a ground-truth loop is: take true pixel art, damage it with `mush`, `cell_texture`, `painterly`, `soft_*`, quantise at full size, despeckle, then compare labels against the truth nearest-upscaled onto the damaged geometry (the engine's `U`/`V` maps give the correspondence). Report label disagreement separately on cell cores and cell borders (my synthesis).
- xprite's metrics: exact size, within 1 cell, pixel match and dE on exact-size cases only, alpha fringe rate [opened].
- MDPI paper: Color Loss, Block Size Consistency, Reversibility [snippet; definitions unverified].

### 9.2 Without ground truth

| Metric | Measures | Computation | Source |
|---|---|---|---|
| Distillability at grid `(s, p)` | How well a grid explains the image, octave-aware | Summed-area tables; 2.1 | pixel-art-fixer [source] |
| Inter-cell contrast / sqrt(intra-cell variance) | Same, simpler | Per-cell mean and variance | pixel-respecter [opened] |
| Description length | Grid plus residual coding cost | Formula not documented | xprite [opened] |
| Patch density / mean patch size | Number of connected label components per unit area; mean component area | CC labelling; FRAGSTATS treats NP, PD and MPS as carrying the same information; the 4- versus 8-neighbour rule changes MPS by about 2x on one test map | https://www.fs.usda.gov/pnw/pubs/pnw_gtr351.pdf [snippet] ; https://www.mdpi.com/2220-9964/8/12/586 [snippet] |
| Edge density | Label-boundary length per unit area (count of 4-neighbour pairs with different labels). This is exactly the Potts smoothness term | One pass | same [snippet] ; 4.1 |
| Small-component area fraction | Share of pixels in components of area below `alpha * s^2` | CC labelling | my synthesis, from 4.3 |
| Component area histogram in units of `s^2` | Speckle shows as mass near 0; clean pixel art has mass at integer multiples of 1 | CC labelling | my synthesis |
| Boundary on-grid fraction | Share of label-boundary length lying within 1 px of a detected cut line | Needs the mesh | my synthesis; analogous to unfake's "share of the edge energy that lies on the boundaries" [source] |
| Orphan-pixel count (1x output) | Isolated pixels with no same-label 4-neighbour | One pass | definition from https://saint11.art/blog/glossary/ [snippet] |

- Applying landscape-ecology metrics to speckle is my inference; the FRAGSTATS sources define the metrics but do not discuss classification speckle.
- **Guard against the trivial optimum**: every flatness metric is maximised by a single-colour image. Each must be paired with a fidelity term (dE to the quantised source on cell cores, or component-survival from 6.3). The Potts energy of 4.1 is one principled way to combine the two into a single number.
- SSIM / MS-SSIM against the fuzzy source reward keeping the fuzz and are the wrong target for A. Not recommended as the primary metric (my assessment).

---

## 10. Most transferable ideas, ranked by expected value for this project

| # | Idea | Question | Why it ranks here | Effort | Main risk | Source |
|---|---|---|---|---|---|---|
| 1 | **Scale-aware connected-component sieve on the label map** (area threshold in units of `s^2`, fill by longest shared border), plus a **width rule for fringes** with an in-between-colour check | A | Directly targets both symptoms; O(N); no dependencies; three independent toolchains ship the area half of it | Low | Threshold tuning at small `s`; fringe rule is untested | 2.10, 4.3 |
| 2 | **No-resize as "snap, then render back on the detected mesh"**, offered alongside per-pixel cleanup | A | The only approach that yields blocky edges; reuses existing code; matches what every other tool does | Low | Inherits every grid error; loses sub-cell detail | 2.2, 4.8 |
| 3 | **Two-stage packing**: structure by centre-weighted / inner-region label vote, colour from original pixels carrying the winning label | A, B | Removes fringe influence on votes and recovers accent colours that global quantisation discards | Low-medium | Output is no longer strictly on the K palette unless re-clamped | 2.1, 2.3, 2.5 |
| 4 | **Distillability (within-cell variance with anti-phase and trend) from summed-area tables** as the candidate score and octave arbiter | B | Purpose-built for the x2 / /2 decision; O(N) per candidate; documented pitfalls | Medium | Dither-locked images defeat it; measures grid fit, not visual preference | 2.1 |
| 5 | **pixel-bench-style synthetic-damage harness** with ground truth, plus the no-reference metrics of 9.2 | A, B, C | Makes every other experiment measurable; a snapper port already exists in the benchmark | Medium | Synthetic damage may not match real model output (xprite's own warning) | 2.11, 9 |
| 6 | **Candidate set = detected grid, its octave neighbours, per-axis variants, detector disagreements**; lean fine; never auto-hide | B | Matches the documented failure taxonomy | Low | More candidates to look at | 2.1, 3.2, 6.1 |
| 7 | **Lost-detail flags by component survival and minority mass** | B | Turns "audition by eye" into "look here" | Medium | Synthesis; no precedent; depends on a clean full-resolution label map (so depends on #1) | 6.3 |
| 8 | **ICM on a Potts energy with a colour data term** (optionally with weaker coupling across detected cut lines) | A | Data-aware alternative to the mode filter; hand-rollable; published precedent for palette label maps | Medium | Corner rounding and thin-line erosion as `beta` grows; one more parameter | 4.1, 4.8 |
| 9 | **Bounded DP for cell boundaries with a regular-versus-elastic gate**, replacing or checking the greedy walker | B, C | Two tools converged on it; the gate addresses pixel-art-fixer's "snapping cascades" objection | Medium-high | Regression risk on images the walker handles | 2.2, 2.3, 3.3 |
| 10 | **Tiles for estimation only**: per-tile pitch/phase with halo, smoothed field, one mesh; global palette | C | The sound version of tiling; handles non-separable drift | High | Many tiles carry no signal; field smoothing is new code | 2.1, 8.2, 8.3 |
| 11 | **Local palette extraction, global merge** | C, A | Protects rare local accent colours | Medium | Palette size control; merge threshold | 2.4, 8.4 |
| 12 | **Kuwahara (radius about `s/3`) or L0 smoothing as a prefilter** | A | Converts ramps to steps before quantisation, helping the palette too | Low (Kuwahara) / high (L0, needs FFT) | Detail loss at small `s`; banding of soft shading | 5 |
| 13 | **Detail-aware kernels for coarser-than-native candidates** (k-centroid, DPID-style minority emphasis, outline expansion) | B | Only matters for `T < n`; several cheap options | Low-medium | Stylisation, not restoration; amplifies noise if run before cleanup | 1.4, 2.7, 2.8 |
| 14 | Alpha-expansion graph cuts | A | Better minima than ICM | High (needs max-flow in WASM) | Cost for marginal gain over #1 + #8 | 4.1 |
| 15 | Lattice-seeded SLIC superpixels | A, C | Elegant adaptive cells | Medium-high | Not blocky; unproven at `s` of 4-6 | 1.1, 5.1 |
| - | Independent per-tile palettes or grids | C | **Negative value** | - | Seams | 8.1 |
| - | Quadtree / variable-size logical pixels | C | **Negative value** | - | "Mixels" | 8.4 |
| - | Spatial colour quantisation for A | A | **Negative value** | - | Creates speckle by design | 4.7 |
| - | Bilateral / guided prefilter for fringes | A | **Near zero** | - | Preserves the ramp that causes the fringe | 5 |

---

## 11. Things looked for and not found

- **Any academic paper (arXiv 2022-2026 or otherwise) on recovering the native grid of AI-generated or "fake" pixel art.** Searches returned pixel-space diffusion papers (unrelated use of "pixel") and general restoration. The only adjacent paper is the 2026 MDPI block-size paper, whose method I could not read (HTTP 403).
- **A published full-resolution "pixel-art despeckle" algorithm** that keeps resolution and uses knowledge of the logical-pixel scale. Tools either snap to cells or apply generic 2x2 morphology.
- **Any tool that outputs multiple candidate resolutions for audition**, or any method for ranking candidates by perceived quality instead of grid fit.
- **A lost-detail metric** for pixel-art downscaling. The component-survival idea in 6.3 is mine.
- **A crisp method for non-integer downscaling of pixel art below native resolution.** Sources agree only integer ratios preserve block structure.
- **Quadtree, mixed-resolution, or variable-logical-pixel pixel art methods.** None found; the community term for the result is pejorative.
- **A dither detector for pixel art**, especially for `s`-pixel blobs. Only classical halftone work.
- **Documentation of how per-region grids are stitched** (pixel-respecter adaptive mode) or how xprite's "phase field" is represented. Both need a source read.
- **xprite's MDL code-length formula.** README only says "fewest bits".
- **Seam-reconciliation literature for per-tile palette quantisation** beyond forum discussion.
- **Total-variation-regularised colour quantisation as a named method.** Found TV/Potts labelling papers in general and one MRF-on-palette-labels paper (CGIV 2010), nothing under that name.
- **Details of Aseprite / LibreSprite built-in cleanup.** No native stray-pixel removal found; LibreSprite scaling options not found; GIMP and Photoshop pixel-artist despeckle recipes not found in searchable form.
- **K-Centroid's documented internals** (the README has none; the description comes from pixeldetector's `kCentroid` function and a product page).
- **DPID's guidance-image definition and the Depixelizing heuristics' exact weights** (from memory only; marked unverified where used).
- **Exact approximation bound and runtime of alpha-expansion at 1 MP in WASM** (my estimates; unverified).
- **Independent, third-party benchmark numbers.** All accuracy figures are from tool authors and partly contradict each other.
- **Whether this fork still emits partially transparent output pixels** (xprite reports a 91% alpha-fringe rate for upstream). Not checked; outside the scope of a literature survey.
