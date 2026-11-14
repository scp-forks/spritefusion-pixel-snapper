# Pixel Snapper: spatial cleanup, resolution auditions, and regional processing

*Independent investigation, November 13, 2026. I inspected the Rust pipeline, frontend controls, prior notes, and all three sample images; measured the source images using an in-memory PNG decoder; and researched the papers and tools cited below. No project files were modified. Proposed algorithms and parameter defaults are recommendations for experiments, not measured improvements.*

## Main findings

**Problem A needs a spatial reconstruction stage.** Palette selection alone cannot enforce “one logical cell, one color.” The most direct solution is to reuse grid detection and cell voting, then paint the winning color across each cell at the original resolution. A separate, conservative label-cleanup stage should handle images whose grid is uncertain.

There is also a concrete implementation issue: **algorithm 3 computes spatial assignments and then discards them during output**, reverting to independent nearest-palette assignment. Fixing that is the first useful experiment.

**Problem B should expose a family of fresh reconstructions from the source image.** Include the detected grid, several progressively finer target grids, and user-entered sizes. Integer refinements of a credible native grid deserve inclusion, but arbitrary sizes such as 164 are valid alternatives. Enlarging an already reduced image cannot recover the details discarded by that reduction.

**Problem C is useful for local evidence and computation, but independent quadrants are a weak default.** At an unchanged total output size, tiling creates no additional pixel capacity. The stronger approach shares a global palette and output coordinate system, while estimating local grid evidence and processing neighborhoods with overlap.

The central design change is to treat these as separate decisions:

1. What are the logical cells or output samples?
2. What colors should the palette contain?
3. Which label belongs to each cell or source pixel?
4. At what dimensions should the result be rendered?

The present no-resize pipeline principally addresses decision 2 and makes decision 3 independently for each source pixel.

---

## 1. Diagnosis grounded in this repository

### 1.1 What the six quantizers actually do

The branch at [src/main.rs:199](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:199) sends no-resize images directly to one quantizer. It does not run profiles, grid estimation, walking, cell voting, or spatial cleanup.

| Algorithm | Palette construction in this implementation | Final output assignment | Consequence for speckles |
|---|---|---|---|
| **0: OKLab k-means** | K-means++ followed by Lloyd iterations in OKLab. | Every original pixel independently chooses its nearest centroid. | Small variations crossing a palette boundary become different labels. |
| **1: Median Cut** | RGB boxes selected by population-weighted variance, followed by three RGB k-means refinement iterations. | Independent nearest-palette assignment in RGB. | It has no region or cell constraint despite the different initializer. |
| **2: Wu** | RGB moment histogram and variance-reducing box splits. | A nearest-palette lookup for each 4-level-per-channel RGB bin. | The lookup is spatially independent; nearby source colors can fall into different labels. |
| **3: Spatial** | OKLab initialization, then 20 iterations using eight-neighbor label agreement and palette updates. | **Recomputes nearest-palette labels from the original image.** | The computed spatial label map is lost. Only its influence on the palette remains. |
| **4: Grey Wolf** | Population search over palette coordinates, with weighted color error, diversity pressure, repair, and full-image k-means refinement. | Independent nearest-palette assignment in OKLab. | Its objective and output contain no logical-cell constraint. |
| **5: Differentiable Palette** | Soft color assignments optimize palette coordinates on weighted color samples; final full-image k-means refinement. | Independent nearest-palette assignment in OKLab. | Differentiability does not provide spatial understanding here. |

Relevant implementations: [k-means](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:518), [Median Cut](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:664), [Wu](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:883), [Spatial](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:1230), [Grey Wolf](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:1396), and [Differentiable Palette](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:1579).

The owner’s diagnosis is substantially correct: five implementations explicitly end in independent color assignment, and the sixth unintentionally does so too. Changing palette optimizers can change the amount and location of speckling without fixing that structural limitation.

### 1.2 The spatial algorithm has a specific output bug

Inside `spatial_quantize`, the optimization updates `assignments[i]` using:

```text
color_distance(pixel, palette[j])
    - spatial_weight * neighboring_agreement(j)
```

However, the output loop at [src/main.rs:1368](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:1368) calculates color distances again and writes `srgb_palette[best]`. It never uses the optimized assignment for that pixel.

The immediate correction is conceptually:

```rust
let i = coord_to_idx[y * width + x].unwrap();
let rgb = srgb_palette[assignments[i]];
```

Preserve the existing handling of transparent pixels.

This correction should be evaluated independently before changing the optimization. It will reveal what the current spatial model actually produces.

It will not solve everything:

- Its neighborhood is only **3×3 source pixels**, smaller than many implied 4–10-pixel cells.
- Its neighbor weights do not decrease across strong image boundaries.
- In-place raster-order updates introduce directional dependence.
- Its increasing spatial weight changes the objective during optimization.
- Empty-cluster repair can seed isolated assignments.
- The fixed maximum spatial bonus, `0.15`, is large relative to many squared OKLab color differences. Increasing it further would be a poor first response.

There is also avoidable work: the implementation recounts eight neighbors separately for every candidate palette label. A neighbor-label histogram calculated once per pixel would reduce that part of the cost from \(O(Kd)\) to \(O(K+d)\).

### 1.3 Why source variation becomes visible speckling

For a palette \(P\), independent assignment is:

\[
L(p)=\arg\min_k \|I(p)-P_k\|^2.
\]

Suppose a human sees one flat brown block, but its source pixels contain subtly different browns. If those colors straddle the boundary between two palette entries, the result contains both labels. Neither entry needs to be a bad palette choice.

The important causes are separable:

| Cause | Assessment for this project |
|---|---|
| **Texture or shading inside implied cells** | Strongly supported by visual inspection and source color counts. These variations survive independent assignment whenever they cross palette boundaries. |
| **Anti-aliased borders** | Visually present. Border mixtures may become an extra band of palette labels, rather than a crisp transition between the adjacent regions. |
| **JPEG-like chroma or resampling damage** | Plausible, but unproven for these files. PNG storage does not establish how the image was generated or previously processed. |
| **Near-duplicate palette colors** | A possible amplifier: several entries can divide variations within one intended material. I did not measure the generated palettes, so this remains a hypothesis. |
| **Excessively large \(K\)** | Can allocate colors to unwanted texture or fringes. Reducing \(K\) may help, but cannot enforce spatial consistency. |
| **Explicit dithering** | **Not the explanation in the current six implementations.** I found no error-diffusion or ordered-dithering output stage. |
| **Spatial regularization** | Absent from the final output in all six paths, including the spatial-output bug. |
| **Rare-detail sampling** | A secondary risk for algorithms 4 and 5. Their representative sampler truncates to the most populated bins if its cap is exceeded. |

The last point is worth correcting in the documentation: comments describing preservation of all histogram bins do not match the truncation at [src/main.rs:1886](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:1886). Whether that cap is reached on these samples still needs measurement.

### 1.4 Measurements of the supplied images

These are direct measurements of decoded source RGB values. “Equal neighbors” counts matching horizontal and vertical adjacent pairs.

| Image | Actual dimensions | Distinct RGB colors | Equal neighbors |
|---|---:|---:|---:|
| [colliseum.png](/Users/shaun/dev/spritefusion-pixel-snapper/other/colliseum.png) | **1672×941** | **196,522** | **19.67%** |
| [cat-1024x1024.png](/Users/shaun/dev/spritefusion-pixel-snapper/other/cat-1024x1024.png) | 1024×1024 | 84,150 | 6.69% |
| [pixel-art-sample-1024x1024.png](/Users/shaun/dev/spritefusion-pixel-snapper/other/pixel-art-sample-1024x1024.png) | 1024×1024 | 178,545 | 18.52% |

All three are 8-bit RGB PNGs without an alpha channel, so transparency handling cannot explain speckles in these particular inputs.

I also sampled 8×8 windows every 16 pixels, grouping colors into 5-bit RGB bins. The median proportion occupied by the most frequent bin was:

- Coliseum: **56.25%**
- Cat: **20.31%**
- Sprite sheet: **29.69%**

These are unaligned diagnostic windows, **not detected logical cells**. They show substantial local color variation; they do not establish a correct grid.

Visual implications:

- **Coliseum:** broad block-like regions coexist with roof texture, windows, lettering, crosses, statues, and thin architectural features. A single coarse grid will trade some of these away.
- **Cat:** this is a textured illustration, with no convincing pre-existing pixel lattice. Processing it is primarily pixel-art abstraction, rather than grid recovery.
- **Sprite sheet:** different assets occupy separate regions, including very small items and fine outlines. Region boundaries and reserved background handling deserve explicit treatment.

A grid-cleaning default should therefore have a confidence test. It should not interpret every input as a damaged enlargement of native pixel art.

### 1.5 What the snap pipeline already contributes—and what it assumes

The snap branch performs RGB quantization, computes grayscale derivative profiles, estimates peak spacing, walks cuts, stabilizes them, and votes within each rectangular cell.

Useful existing pieces:

- [Profiles](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:2007)
- [Step estimation](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:2049)
- [Walker](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:2197)
- [Cell voting](/Users/shaun/dev/spritefusion-pixel-snapper/src/main.rs:2404)

Important limitations:

1. Median spacing between selected peaks is a heuristic, not proof of a native pitch. It can select a harmonic or a repeated feature scale.
2. Grayscale profiles can miss strong chromatic boundaries with similar luminance.
3. The axis-resolution logic initially reconciles pitch estimates toward square cells. This can obscure genuinely non-square scaling.
4. One column-cut array and one row-cut array cannot describe general two-dimensional grid drift.
5. `resample` chooses the most frequent exact **RGBA tuple**, not a perceptually grouped color or an alpha-weighted color.
6. A plurality can win without exceeding 50%.
7. Voting creates crisp cells but makes no explicit attempt to preserve a small minority feature.

The detected grid should be presented as a **grid hypothesis**, not an objectively optimal resolution.

---

## 2. Prior art for Problem A: clean regions and sharp boundaries

The cost assessments below are my engineering estimates for application here.

| Prior art | What it does | Application and limitations | Cost / implementation fit |
|---|---|---|---|
| **Categorical majority filters** | Replace a label when a sufficiently large, sometimes contiguous neighborhood agrees. ArcGIS documents both majority thresholds and connectivity conditions. | A useful baseline for isolated wrong labels. Eight-neighbor smoothing can alter rectangular corners; repeated filtering can erase thin features. Operate on palette labels, with confidence gates. [Documentation](https://pro.arcgis.com/en/pro-app/2.8/tool-reference/spatial-analyst/majority-filter.htm) | Small fixed neighborhoods are \(O(N)\); easy Rust implementation. |
| **Connected-component area filtering** | Finds connected objects and removes structures below an area threshold. Area opening is a related morphological operation. | Better matched to “small islands inside a region” than unrestricted blur. For palette images, find spatial components first: all occurrences of a palette index are not one connected object. Replacement should choose a neighboring color, rather than turn the island into background. [scikit-image documentation](https://scikit-image.org/docs/stable/api/skimage.morphology.html?highlight=morphology) | \(O(N)\) component labeling, plus merge decisions. Excellent fit. |
| **Bilateral filtering — Tomasi & Manduchi, 1998** | Combines spatial proximity and color similarity to smooth while retaining edges. | Useful before palette fitting or as a guide for label regularization. It does not establish a grid, and residual continuous transitions still need discrete reconstruction. [Paper](https://users.cs.duke.edu/~tomasi/papers/tomasi/tomasiIccv98.pdf) | Direct radius-\(r\) implementation: \(O(Nr^2)\). Small kernels are practical. |
| **Guided filtering — He, Sun & Tang, 2010/2013** | Uses local linear models and a guidance image; admits an exact linear-time implementation. | Useful for estimating broad color regions or boundary confidence. Its output is continuous-valued; it is not a substitute for label selection or cell painting. [Author project](https://people.csail.mit.edu/kaiming/eccv10/index.html) | \(O(N)\) with box-filter statistics; moderate implementation effort. |
| **MRF/Potts label regularization; graph-cut optimization** | Balances per-pixel data costs against penalties for neighboring label disagreement. Boykov, Veksler & Zabih describe expansion and swap moves for discontinuity-preserving energies. | Directly expresses the desired piecewise-constant label map. Edge-weighted penalties can protect boundaries. The model still penalizes tiny legitimate details and dithering unless protected. [Paper](https://www.cs.cornell.edu/~rdz/Papers/BVZ-pami01-final.pdf) | ICM-style local updates are small and dependency-free. Graph cuts require substantially more machinery. |
| **Spatial color quantization — Puzicha et al., 1998/2000** | Jointly optimizes quantization and halftoning using a spatial perceptual model. | Relevant as a joint spatial/color formulation, but its halftoning objective can intentionally produce mixed-color patterns. It does not inherently seek flat flood-filled regions. [Paper](https://doi.org/10.1109/83.841942), [Rust implementation](https://github.com/okaneco/rscolorq) | Iterative and comparatively expensive. Useful comparator, weak default for removing dots. |
| **\(L_0\) gradient minimization — Xu et al., 2011** | Promotes sparse image gradients, suppressing low-amplitude structure and sharpening major transitions. | Conceptually close to removing texture while preserving region boundaries. Small meaningful marks can also disappear, and the result has no square-cell guarantee. [Author project](https://jiaya.me/file/archive/projects/L0smoothing/) | Global iterative solver; less attractive than label cleanup for an initial WASM implementation. |

**Interpretation:** “spatial” can mean two different goals. A method may preserve perceived average color through halftoning, or it may encourage constant-color regions. For this owner’s complaint, the second objective should drive the default.

The prior [palette-reduction notes](/Users/shaun/dev/spritefusion-pixel-snapper/docs/palette-reduction-info.md) recommend adaptive dithering for general quantization quality. That recommendation should not carry over automatically to this cleanup task.

---

## 3. Prior art for Problem B: choosing and reconstructing new resolutions

### 3.1 Gerstner et al.: *Pixelated Image Abstraction*, 2012; expanded 2013

This is particularly relevant because it jointly solves the association between source regions and output pixels, and the reduced palette. It uses modified superpixels and palette optimization. The expanded version adds user constraints. It explicitly addresses abstraction at a requested small output size, rather than simply recovering an existing grid. [2012 project](https://gfx.cs.princeton.edu/pubs/Gerstner_2012_PIA/), [2013 project, paper, and source](https://cragl.cs.gmu.edu/pixelate/journal.html).

**Application:** a strong reference for arbitrary-resolution audition candidates and user protection of important features.

**Engineering assessment:** local superpixel updates plus palette updates suggest roughly \(O(T(N+MK))\) with bounded spatial searches. A faithful implementation is considerably more involved than rectangular cell voting. Geometry can shift to preserve recognizable features, so it should be offered as an abstraction alternative.

### 3.2 Kopf, Shamir & Peers: *Content-Adaptive Image Downscaling*, 2013

The method optimizes the shape and position of bilateral downsampling kernels using constrained expectation-maximization. Its constraints keep supports local and associated with the output lattice; preserving connected linear features is an explicit concern. [Author project](https://johanneskopf.de/publications/downscaling/), [paper](https://www.cs.wm.edu/~ppeers/publications/Kopf2013CID/Kopf_SIGGRAPHAsia2013.pdf), [implementation pseudocode](https://johanneskopf.de/publications/downscaling/paper/pseudocode.pdf).

**Application:** useful when an arbitrary target grid would otherwise split a thin roofline or outline between samples.

**Engineering assessment:** iterative local support optimization is feasible without neural dependencies, but requires more state and numerical care than voting. Kernel colors remain continuous-valued unless followed by palette assignment.

### 3.3 Öztireli & Gross: *Perceptually Based Downscaling of Images*, 2015

This work formulates downscaling using a perceptual image-quality objective and derives a closed-form solution implementable through sums and convolutions. It should not be classified as an expensive iterative optimizer merely because it is perceptually motivated. [Author project](https://cgl.ethz.ch/publications/papers/paperOzt15b.php).

**Application:** an efficient alternative candidate for preserving apparent structure.

**Engineering assessment:** a plausible CPU/WASM implementation. Its perceptual objective does not impose a finite palette or square flat regions, so additional discrete rendering is necessary.

### 3.4 Weber et al.: *Rapid, Detail-Preserving Image Downscaling*, 2016 — DPID

DPID increases the contribution of source pixels whose colors differ from a locally smoothed low-resolution guide. The authors provide CUDA and MATLAB implementations; their reference code handles fractional footprint overlap. [Paper](https://doi.org/10.1145/2980179.2980239), [author repository](https://github.com/mergian/dpid), [reference implementation](https://raw.githubusercontent.com/mergian/dpid/master/matab/dpid.m).

**Application:** my first detail-preserving downscaling experiment after area-weighted mode voting.

**Engineering assessment:** approximately linear source traversal per candidate, plus a small output-grid convolution. Its outlier emphasis is also its risk here: AI texture and speckles may be promoted as “detail.” Use a cleaned guide and compare against the original formulation.

### 3.5 Kopf & Lischinski: *Depixelizing Pixel Art*, 2011

This method reconstructs a smooth vector representation from existing pixel art, resolving ambiguities in diagonal connectivity and fitting contours. [Author project](https://johanneskopf.de/publications/pixelart/).

**Application:** useful ideas about connectivity, especially diagonal features.

**Assessment:** it solves a different rendering problem. A vectorize-and-rasterize round trip can smooth away deliberately blocky structure. I would not make it the main cleanup path.

### 3.6 Recent generation and pixelization work

| Work | Contribution relevant here | Applicability |
|---|---|---|
| **Wu et al., *Make Your Own Sprites*, SIGGRAPH Asia 2022** | Separates cell-aware and aliasing-aware pixelization, using reference pixel-art structure to control cell effects. [Paper record](https://ink.library.smu.edu.sg/sis_research/7872/), [official implementation](https://github.com/WuZongWei6/Pixelization) | Useful conceptual separation of geometry and aliasing. Its learned runtime does not fit the requested implementation constraints. |
| **Binninger & Sorkine-Hornung, SD-\(\pi\)XL, 2024** | Uses score distillation and a differentiable palette-indexed image representation at user-selected dimensions. [arXiv](https://arxiv.org/abs/2410.06236) | Demonstrates a representation that explicitly constrains output size and colors. Its neural optimization is outside the proposed runtime. |
| **Zhang et al., *PixelBench*, November 2026** | Studies LLM generation of palettes followed by grids of color indices, including image-to-pixel tasks at several resolutions. [arXiv](https://arxiv.org/abs/2611.02540) | Relevant to semantic abstraction and evaluation, not a lightweight repair algorithm. This is separate from Retro Diffusion’s similarly named reconstruction benchmark. |
| **Wang et al., *Content-Aware Pixel Art Rendering with Non-Square Pixels*, GI 2026** | Optimizes source filter supports for arbitrary tilings, while rendering on the selected output tiling. [Author-hosted paper](https://cs.uwaterloo.ca/~dvogel/gi2026/papers/1100b.pdf) | Supports separating sampling support from rendered cell shape. It does not establish that independent square tiles can be processed without coordination. |

**My conclusion from this literature:** recovering an implied grid and creating a useful new low-resolution abstraction should be separate modes. A detector is not an arbitrary-resolution downscaler.

---

## 4. Prior art for Problem C: regions and local structure

| Method | Relevant contribution | Assessment for Pixel Snapper |
|---|---|---|
| **SLIC / SLICO — Achanta et al.** | Compact superpixels produced by localized clustering in color and position; author code is available. [EPFL project](https://www.epfl.ch/labs/ivrl/research/slic-superpixels/) | Good intermediate regions for denoising and statistics. Their irregular boundaries should not automatically become the final pixel-art boundaries. Bounded local searches make a Rust port practical. |
| **Mean shift — Comaniciu & Meer, 2002** | Mode seeking in feature space, applied to discontinuity-preserving smoothing and segmentation. [Paper](https://courses.csail.mit.edu/6.869/handouts/PAMIMeanshift.pdf) | Naturally groups nearby similar colors without choosing a palette size first. Bandwidth determines whether texture or fine details survive; direct iterative neighborhood searches can be expensive. |
| **Felzenszwalb & Huttenlocher, 2004** | Graph-based segmentation using evidence for boundaries between regions, with nearly linear running time in graph edges. [Paper](https://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf), [author code](https://cs.brown.edu/people/pfelzens/segment/) | Useful for region confidence and statistics. It produces segmentation, not a uniform pixel lattice; texture and minimum-region settings require care. |
| **Shared histogram/palette construction** | Libimagequant supports palette construction from histograms and multiple images. [Official documentation](https://pngquant.org/lib/) | A practical precedent for accumulating regional evidence while retaining one palette budget. Local processing need not imply independent palettes. |

I would use segmentation primarily to inform assignment and confidence. Directly painting SLIC or mean-shift regions produces a posterized or mosaic appearance unless their labels are subsequently rendered on a deliberate pixel grid.

---

## 5. Practical tools worth comparing

These descriptions establish useful baselines and design precedents. I did not reproduce their performance claims.

| Tool | What it provides | What to learn or test |
|---|---|---|
| **unfake.js / unfake-core** | Current documentation describes a Rust/WASM core with integer, fractional, and elastic grids; several cell reducers; palette reduction; and optional cleanup. [Repository](https://github.com/jenissimo/unfake.js) | A particularly close architectural comparator. Compare cell reducers and confidence handling, rather than just screenshots. |
| **“How to Tame Your AI Pixel Art”** | The author’s 2025 explanation covers informative-tile voting, edge profiles, phase search, and dominant-color block voting. [Article](https://dev.to/jenissimo/how-to-tame-your-ai-pixel-art-3pk5) | Useful implementation narrative. Its historical mean-color fallback illustrates why ambiguous cells can reintroduce mixed colors. Current repository behavior has evolved. |
| **Retro Diffusion Pixel Art Fixer** | A non-neural reconstruction implementation with multiple grid-evidence channels, arbitration, and a structure/color separation during reconstruction. [Repository](https://github.com/Retro-Diffusion/pixel-art-fixer) | Study multi-hypothesis detection and obtaining cell color from original pixels supporting the winning structural label. Treat its comparative benchmark claims as author-reported. |
| **Astropulse PixelDetector** | Its source uses edge-profile peak spacing, then local two-centroid reduction and dominant-color selection per output footprint. [Repository](https://github.com/Astropulse/pixeldetector), [source](https://raw.githubusercontent.com/Astropulse/pixeldetector/main/pixeldetector.py) | A compact baseline for local dominant-cluster sampling. Median peak spacing remains a fragile grid estimator. |
| **A1111 PixelDetector wrapper** | Exposes output scaling, including `Rescale = -1` to return to original dimensions. [Documentation](https://github.com/netux/sd-webui-pixeldetector/blob/main/README.md) | Direct precedent for reconstructing a logical image and displaying it at the original canvas size. |
| **ComfyUI-PixelArt-Detector** | Palette extraction/conversion, reduction, resizing, cleanup options, and dithering controls; recent documentation explicitly separates quantization and reduction methods. [Repository](https://github.com/dimtoneff/ComfyUI-PixelArt-Detector) | A useful workflow comparator. Node packaging alone does not resolve reconstruction ambiguity. |
| **PixelOE** | Classical contrast-aware outline expansion followed by selectable downsampling methods, including center, contrast, and local centroid approaches. It also supplies ComfyUI nodes. [Repository and algorithm description](https://github.com/KohakuBlueleaf/PixelOE) | Strong candidate for a detail-preserving audition variant. Port the small image operations rather than adopting its tensor runtime. Outline expansion may thicken details or amplify noise. |
| **Aseprite / LibreSprite** | Aseprite documents median-based Despeckle, indexed conversion, dithering controls, resizing, and selection-based effects. LibreSprite exposes Despeckle and color-quantization commands. [Aseprite Despeckle](https://www.aseprite.org/api/command/Despeckle), [CLI](https://aseprite.org/docs/cli), [LibreSprite scripting](https://github.com/LibreSprite/LibreSprite/blob/master/SCRIPTING.md) | Good manual and scripted baselines. Median filtering should not be mistaken for cell reconstruction or a categorical mode filter. |
| **pngquant / libimagequant** | Perceptually weighted histogramming, modified Median Cut, refinement, alpha handling, and controllable dithering. Dithering level zero enables non-dithered remapping. [Algorithm](https://pngquant.org/), [API](https://pngquant.org/lib/) | Strong palette baseline with dithering disabled. It still needs a spatial reconstruction stage for this task. |
| **rscolorq** | Rust spatial quantization that jointly performs palette reduction and halftoning. [Repository](https://github.com/okaneco/rscolorq) | Useful for comparison at small palettes, but its mixed-color rendering goal conflicts with a flat-region default. |

---

## 6. Ranked proposals for Problem A

### A0. Fix spatial output and establish a controlled baseline

**Priority: immediate.**

Return the optimized label map from algorithm 3. For a clean experiment:

1. Save the final palette in memory.
2. Render both:
   - independent nearest-palette labels;
   - optimized spatial labels.
3. Compare them using the same palette and input.
4. Inspect thin outlines, small highlights, and low-contrast flat regions separately.

Then improve its implementation:

- Count neighbor labels once per pixel.
- Freeze the palette during final label cleanup.
- Use edge-sensitive pairwise weights.
- Treat \(K\) as a maximum; unused colors are acceptable.
- Use a defined update schedule and tie rule.

This is an important correctness fix, but the main proposed feature is A1.

### A1. Reconstruct cells and paint them back at original resolution

**Priority: highest for pixel-like inputs.**

#### Minimal implementation

Reuse the existing quantized image and cuts:

```text
Q = quantize(source, K)
cuts = detect_grid(Q)

for each rectangle C formed by cuts:
    winner = most_frequent_palette_label(Q within C)
    fill output rectangle C with palette[winner]
```

This guarantees one color per selected cell and preserves canvas dimensions.

Use the original cut rectangles when painting. Resizing the small voted image back uniformly changes geometry when the walker’s cells have unequal widths.

#### Stronger implementation: separate structure colors from final palette colors

A small final palette can merge distinctions before the grid or cell structure has been recovered. Conversely, fitting a palette to raw noisy pixels can spend colors on texture.

I recommend this two-stage design:

```text
P_structure = fit_fast_palette(source, K_structure)
L_structure = assign(source, P_structure)

grid = detect_grid(source, L_structure)

for each cell C:
    votes = histogram of structural labels in C
    winner = stable_dominant_label(votes)

    representative[C] =
        robust color of original pixels in C supporting winner

P_final = fit_palette(representatives, K_final, cell_weights)

for each cell C:
    cell_label[C] = nearest(P_final, representative[C])

render cell_label into:
    native logical image, or
    original rectangles, or
    selected presentation size
```

Proposed starting parameters:

| Parameter | Starting value |
|---|---:|
| Final palette | Existing user choice; initially test 8, 16, 32, 64 |
| Structural palette | `clamp(2 × K_final, 32, 64)` |
| Interior inset | About 10–15% of cell width/height, capped so tiny cells remain usable |
| Dominance confidence | Winner fraction ≥0.75 |
| Winner margin | Winner fraction minus runner-up ≥0.25 |
| Representative | Trimmed mean or approximate medoid of winning-label source pixels |
| Final palette fitting | Weighted OKLab k-means, about 10 iterations |
| Default logical-cell cleanup | None |

These thresholds are experimental defaults.

Use both full-cell and interior votes. A border fringe may dominate the outer ring while the cell interior identifies the intended fill. Disagreement between those votes is valuable uncertainty evidence.

**Do not take the exact RGB mode of raw AI pixels.** Similar intended colors may be split across many almost-identical RGB tuples.

**Do not average all colors across an ambiguous edge.** That can create another fringe color. Estimate the representative from the winning color group.

#### Ambiguous cells need explicit treatment

Offer two behaviors:

- **Strict cells:** every cell gets one color; uncertain cells are marked in an overlay.
- **Preserve more detail:** offer a finer regular-grid candidate before flattening uncertain structure.

Leaving arbitrary source speckles inside uncertain cells would undermine the promise of strict cleanup. Locally subdividing only those cells is possible, but creates the mixed-resolution representation discussed under C.

#### Limits

Cell voting can remove a real feature occupying a minority of a cell. Confidence measures identify ambiguity; they cannot establish artistic intent.

A second issue is color-group fragmentation: if the intended fill is split among several similar structural labels, another label can win the plurality. Test grouping nearby structural colors during voting, or a small local dominant-cluster calculation for uncertain cells.

#### Cost

Let \(N=1{,}048{,}576\), with nominal 8×8 cells giving \(M=16{,}384\).

- Cell histogram accumulation: \(O(N)\).
- Cell representative-to-palette work: \(O(MK)\).
- Palette fitting on representatives: \(O(TMK)\).
- Painting: \(O(N)\).

For \(K=16,T=10\), palette refinement over cells needs about **2.6 million color-distance evaluations**, compared with about **168 million** for ten full-image iterations.

Structural assignment still costs \(O(NK_{\text{structure}})\) unless accelerated or restricted. The main gain is avoiding repeated full-image palette optimization.

### A2. Add conservative cleanup of the discrete label map

**Priority: highest when grid evidence is uncertain, and useful as a finishing stage.**

Use one fixed palette and operate on its labels. Keep the source colors as evidence.

#### A2a. Merge small components using evidence and boundary costs

For each connected same-label component, collect:

- Area and bounding box.
- Adjacent labels and shared boundary lengths.
- Sum of source OKLab values.
- Color fit under its present label.
- Relationship to estimated cell dimensions.
- Protection flags for detail or dithering.

For component \(A\), currently label \(a\), evaluate neighboring label \(b\):

\[
\Delta E =
\sum_{p\in A}
\left(\|I_p-P_b\|^2-\|I_p-P_a\|^2\right)
+
\lambda\sum_{(p,q)\in\partial A}
w_{pq}
\left([b\ne L_q]-[a\ne L_q]\right).
\]

Merge only when:

- The component is small at the relevant scale.
- It is unprotected.
- The move improves the chosen energy.
- Its source-color evidence does not strongly support keeping it.

The color-cost difference can be evaluated from the component’s area and color sum; it does not require rescanning every pixel for every candidate.

Starting threshold for source-pixel cleanup:

```text
A_small = floor(0.1 × estimated_cell_area)
```

At an 8×8 pitch, this is about six source pixels. It deliberately targets sub-cell islands.

**Do not use the same rule on the logical-cell map.** A one-cell component can be an intentional eye, star, window, or dither dot. Default deletion of those is too aggressive.

#### A2b. Optional gated mode filter

A simple baseline:

```text
if at least 6 of 8 neighbors share label b
   and source evidence only weakly favors the current label
   and pixel is not protected:
       replace with b
```

Use one pass initially and evaluate its effect on corners.

Never take a numeric median of palette indices. Their numeric order has no color or geometric meaning.

#### A2c. Edge-weighted Potts cleanup

A suitable objective is:

\[
E(L)=
\sum_p \|I_p-P_{L_p}\|^2
+\lambda\sum_{(p,q)\in\mathcal N}
w_{pq}[L_p\ne L_q].
\]

For four-neighbor edges:

\[
w_{pq}=
\exp\left(
-\frac{\|G_p-G_q\|^2}{2\sigma_e^2}
\right),
\]

where \(G\) is a mildly denoised guide, rather than the noisy input alone.

Proposed implementation:

```text
initialize labels from nearest palette or cell reconstruction
freeze palette

repeat up to 4 sweeps:
    visit pixels using a defined four-neighbor update schedule
    candidates = current label, nearest labels, neighbor labels

    choose label minimizing local data + boundary cost
    accept strict decreases; keep current label on ties

    stop if no changes
```

Starting values in the code’s approximately unit-scaled OKLab space:

- Estimate interior noise \(\sigma_n\), initially clamp to 0.005–0.02.
- Start with \(\lambda=2\sigma_n^2\) per neighboring edge.
- Start with \(\sigma_e=\max(0.03,2\sigma_n)\).
- Use 3–4 sweeps.
- Restrict updates to uncertain labels and suspected artifacts when possible.

These are calibration targets, not universal perceptual thresholds.

ICM-style updates can become trapped in local minima. Component moves complement them by changing an entire island at once. Graph-cut expansion is a later option if the simpler optimizer consistently stalls; a better optimizer will not fix a poorly chosen objective.

### A3. Mild edge-preserving preprocessing

**Priority: supporting option.**

Test:

- Bilateral filter: radius 2, spatial sigma approximately 1.2 source pixels.
- OKLab range sigma around 0.02 as an initial value.
- Alternatively, a small vector-medoid filter.

Then fit the palette or estimate structure from the filtered image.

Retain original pixels for final representative estimation where appropriate.

This can reduce the amount of texture seen by the quantizer, but it does not guarantee flat cells. Strong filtering also removes fine lines and intentional texture. The cat is a good stress test for this distinction.

### A4. Preserve dithering and details at the correct scale

The most useful protection is **scale separation**:

- Variation inside a credible logical cell is a cleanup candidate.
- Variation between credible cells may be intentional art.
- Tiny components on the logical image should be retained by default.

Additional conservative protection can use:

- Repeated two-color patterns across several logical cells.
- Autocorrelation at logical-cell offsets.
- Strong source contrast supporting a small feature.
- Connected thin-line evidence.
- User-painted protection masks.

Periodic checks can recognize some ordered dithering, but irregular intentional dithering can resemble noise. Repeated roof tiles can also resemble dithering. There is no reliable automatic guarantee when the same observed pattern admits both interpretations.

---

## 7. Ranked proposals for Problem B

### B1. Generate a compact resolution audition set

**Priority: highest.**

Use two candidate families.

#### Family 1: grid-recovery hypotheses

Include:

- Current detected grid.
- Plausible finer harmonic, approximately half the detected pitch.
- A coarser harmonic only when its evidence is credible.
- Integer refinements of the detected cell partition.

The harmonic alternatives address detector ambiguity. They should carry confidence and grid-fit information.

#### Family 2: deliberate new target resolutions

For a credible 128-pixel long-side result, a useful initial set is approximately:

```text
128, 164, 205, 256
```

Add requested anchors such as 192, 320, or 512. Deduplicate nearby sizes.

A roughly geometric ladder is preferable to equal numerical increments because it samples multiplicative changes in pixel capacity. The exact ratio is a product choice; I would initially test steps around 1.25–1.3.

**Integer multiples are useful, but should not be mandatory.**

An integer refinement can preserve detected cell boundaries. An arbitrary target can provide a better compromise between detail and editing cost. It should be reconstructed as a new grid.

#### Recompute every candidate from source evidence

```text
source = decode once
guide, structural labels, palette = analyze once

for each target:
    construct target footprints
    reconstruct directly from source/guide
    assign final palette labels
    measure quality and detail risks
    return candidate
```

A 256×256 enlargement of the 128×128 result has no more recovered detail. A fresh 256×256 reconstruction may have more.

#### Preserve aspect ratio deliberately

Default arbitrary-size candidates should preserve the source aspect ratio.

If the detected grid implies a materially different aspect ratio, flag that. A 100×115 detected result from a square source may indicate uncertain detection, non-square source cells, or geometric distortion. It should not silently define the aspect ratio of every later candidate.

### B2. Use area-weighted label voting as the reconstruction baseline

**Priority: highest.**

For target cell \(C_j\), compute exact overlap area \(a_{pj}\) with each source pixel:

\[
V_j(k)=\sum_p a_{pj}[L_p=k].
\]

Then choose:

\[
L_j=\arg\max_k V_j(k).
\]

This supports fractional scale factors, uses all contributing evidence, and outputs a palette label.

For ties, use an explicit rule:

1. Lowest representative color error.
2. Center-supported label.
3. Stable palette index.

Keep the full frame by default. Cropping or padding to a preferred grid should be a visible option with recorded offsets.

#### Reducer comparison

| Reducer | Strength | Main failure |
|---|---|---|
| **Center / nearest sampling** | Fast; preserves a sampled label exactly. | Phase-sensitive; can miss thin lines and whole small features. |
| **Area mean → palette** | Stable appearance and low-frequency color. | Mixes foreground/background; may select a fringe color or lose an outline. |
| **Per-channel median → palette** | Resists isolated outliers. | Can erase minority lines; the channelwise result may not be a source color. |
| **Area-weighted label mode** | Clean labels and dominant-region preservation. | Deletes minority details. |
| **Local dominant cluster** | Groups subtly different samples of the same fill. | More work; a small meaningful color can lose to the dominant cluster. |
| **Detail-weighted reconstruction** | Can retain small contrasting structures. | Can promote noise or exaggerate outlines. |

These are different tradeoffs. A useful audition interface should expose at least a **clean** and a **detail-preserving** candidate at selected sizes.

### B3. Add DPID as the first detail-preserving alternative

**Priority: second.**

A source-based DPID candidate:

```text
A = area_average(source, target)
B = small Gaussian-like smoothing of A

for each target cell j:
    numerator = 0
    denominator = 0

    for each source pixel p overlapping j:
        detail = color_distance(source[p], B[j])
        weight = overlap_area(p, j) × detail^lambda
        numerator += weight × source[p]
        denominator += weight

    use weighted color, or guide fallback if denominator is zero
    map result to final palette
```

Start with \(\lambda=0.5\), and compare 0, 0.5, and 1.

For these inputs, I would also test two explicitly modified variants:

1. Calculate detail weights from a cleaned guide to reduce noise promotion.
2. Accumulate weights by palette label and select a weighted label mode.

The second variant guarantees a discrete output color but is **my proposed adaptation**, not the original DPID algorithm.

Record both visual gains and false detail amplification. A darker or thicker line is not automatically a better-preserved line.

### B4. Add a constrained adaptive-support candidate

**Priority: later.**

If fixed-footprint reducers repeatedly lose important lines, test a lightweight adaptation inspired by pixelated abstraction and content-adaptive supports:

```text
initialize one source support per target pixel
initialize supports on a regular target lattice

repeat 5–10 times:
    assign source evidence to nearby supports using color + position
    update support statistics
    constrain support movement and neighborhood topology
    update discrete output labels

render labels on the regular target grid
```

A preliminary movement limit of roughly one quarter of a target-cell width is a reasonable experiment.

This can move sampling support toward a feature while retaining a regular rendered grid. It is more complex, and it can deform structures. Keep it as a separately labeled abstraction candidate.

### B5. Rank by a quality/editing tradeoff

**Priority: alongside candidate generation.**

Do not rank only by reconstruction error; finer candidates will often win.

Report at least:

- Editable pixel count.
- Color/structure fidelity.
- Detail-risk score.
- Grid confidence.
- Speckle score.
- Protected-feature survival.

Keep the **Pareto frontier**: candidates for which no other result is both smaller and better on the selected quality measures.

A practical recommendation rule is:

> Prefer the smallest candidate that passes calibrated detail-retention thresholds; retain the larger alternatives for audition.

Before sufficient human calibration exists, show a recommendation with uncertainty rather than claiming an objectively best size.

Provide a heatmap of:

- High-contrast source evidence rejected by cell voting.
- Conflicting labels sharing one target cell.
- Thin components losing representation.
- Strong disagreement between clean and detail-preserving reducers.

This directly helps the human inspect places where detail may have been lost.

---

## 8. Ranked proposals for Problem C

### 8.1 Why naive quadrants do not solve the pixel-budget problem

Consider a 1024×1024 input:

- Whole image →128×128: **16,384 output pixels**.
- Four 512×512 quadrants →64×64 each, stitched: **128×128**, also 16,384 pixels.
- Four quadrants →128×128 each, stitched: **256×256**, or 65,536 pixels.

The third result gets four times as many output pixels. A fair comparison is a global 256×256 reconstruction.

Tiling can improve grid estimation or palette allocation, but it cannot preserve arbitrary extra detail at the same sampling density.

Independent tiles additionally risk:

- Different palettes for the same material.
- Discontinuous grid phases.
- Different logical-cell sizes.
- Broken cross-tile lines.
- Duplicate or missing boundary features.
- Up to \(T K\) colors across \(T\) disjoint tile palettes.
- Geometric seams when independently resized outputs are stitched.

### C1. Share the palette and coordinate system; process locally

**Priority: highest.**

```text
estimate global structural evidence
fit one global palette
construct one global output grid

initialize global labels

process tile cores using neighboring halo pixels
reconcile boundary decisions
render once in the global coordinate system
```

Proposed starting point:

- Tile core: 256×256 source pixels.
- Halo: about two estimated logical cells.
- Shared global palette.
- Shared phase/origin.
- Every tile can use every global palette label.

For an 8-pixel pitch, a 16-pixel halo gives an area overhead of approximately:

\[
(288/256)^2 \approx 1.27.
\]

That is about 27% additional neighborhood work.

**Halo requirements depend on the operation.** A radius-\(r\) filter needs radius-\(r\) context for one pass. Iterative local updates can propagate farther. Connected components can extend across the entire image, so a fixed halo is not sufficient to make component decisions globally correct.

For iterative label cleanup, exchange labels between tiles each sweep, or process from a shared global label map. Global four-neighbor label storage is small enough at 1024² that tiling may be unnecessary for memory.

### C2. Use local measurements to detect grid drift

**Priority: second.**

Use overlapping analysis windows, initially perhaps 256×256 with a 128-pixel stride:

```text
for each window:
    estimate several pitch/phase hypotheses
    record evidence and confidence

reject unreliable estimates
fit a spatially regularized global model
reconstruct using the coordinated model
```

Important details:

- Convert local phases to common global coordinates.
- Treat phase as periodic; ordinary averaging fails around phase wrap.
- Constrain abrupt pitch changes unless there is evidence of separate assets.
- Keep cell boundaries ordered and non-crossing.
- Fall back to global evidence in low-information regions.

The current two cut arrays can describe irregular rectangular spacing, but not a grid whose column boundaries vary with vertical position.

For a future two-dimensional grid model, I would use the local deformation primarily to recover source evidence, then render onto one regular target grid. Rendering arbitrary warped cells directly can undermine the desired blocky appearance.

### C3. Build global palettes from regional representatives

**Priority: useful optional enhancement.**

This is stronger than giving every tile its own final palette:

```text
for each region:
    extract robust representative colors and weights

combine all representatives
fit a global K-color palette
reassign all regions to that shared palette
```

A regional weighting scheme can prevent a huge sky region from overwhelming tiny objects. For example, test a mixture of area-weighted and region-balanced samples.

The global palette must still make tradeoffs. If the scene contains more meaningful colors than the budget allows, local evidence cannot eliminate that limitation.

A less desirable sequence is:

```text
quantize tiles independently
stitch
quantize the stitched image again
```

The first stage may already have discarded information needed by the second.

### C4. Use hierarchical, aligned cell sizes only as an explicit option

**Priority: experimental.**

A quadtree-like representation can allocate smaller cells to detailed regions and larger cells to flat regions:

```text
start from a fine aligned grid

for each candidate parent block:
    estimate error from replacing children with one label

    merge if:
        error is sufficiently small
        no protected feature is removed
        no protected dither pattern is removed
```

A corresponding objective is:

\[
E =
\text{representation error}
+\mu\cdot\text{number of leaves}.
\]

Use cell widths such as 2, 4, 8, and 16 source pixels, all aligned to one fine lattice. Restricting adjacent levels can reduce abrupt changes.

Benefits:

- Flat sky can use large blocks.
- Cars, windows, and signs can use finer blocks.
- Axis-aligned geometry remains crisp.

Risks:

- Mixed logical pixel sizes become part of the style.
- High-frequency noise can attract excessive refinement.
- A small high-contrast detail can disappear under a low average-error threshold.
- The resulting high-resolution raster does not become a small uniform native sprite.

This is appropriate for an explicitly adaptive mosaic output. A single-grid pixel-art export still needs a uniform final lattice.

### C5. Use asset-aware regions where appropriate

**Priority: particularly relevant to the supplied sprite sheet.**

A sprite sheet is a better candidate for independent regions than one continuous city scene.

Process known asset rectangles independently when:

- Their boundaries are meaningful.
- Their placement metadata is retained.
- Related assets share a palette where desired.
- Background/chroma colors are handled deliberately.
- Their final scale policy is explicit.

A random quadrant boundary has no comparable semantic justification.

---

## 9. Rust/WASM architecture and expected costs

### 9.1 Refactor results into palettes, labels, and geometry

Have palette methods return reusable data rather than only rendered RGBA:

```text
PaletteResult:
    palette_rgb
    palette_oklab
    labels
    optional assignment confidence

GridResult:
    column cuts / row cuts
    pitch and phase hypotheses
    confidence
    source-to-cell mapping

Candidate:
    target dimensions
    geometry policy
    reducer
    palette
    labels
    quality metrics
```

Then rendering becomes a separate operation.

This enables:

- Any palette builder with any spatial assignment method.
- Fixed-palette ablation experiments.
- Original-size cell painting.
- Multiple resolutions without repeated decoding.
- Shared palettes across auditions.
- Confidence and detail-loss overlays.

Extend the frontend cache key to include grid, target size, reducer, cleanup parameters, and palette identity. The existing cache distinguishes only the current modes and palette/algorithm settings.

### 9.2 Approximate work at 1024×1024

These are operation estimates, not timing benchmarks.

| Operation | Complexity | Approximate scale |
|---|---:|---|
| One full nearest-palette assignment, \(K=16\) | \(O(NK)\) | 16.8M distance evaluations |
| Ten full-image palette iterations, \(K=16\) | \(O(TNK)\) | 168M distance evaluations |
| Ten palette iterations over 8×8 cell representatives | \(O(TMK)\) | 2.6M distance evaluations |
| Cell voting and painting | \(O(N)\) | A few source-image passes |
| Connected components | \(O(N)\) | Linear scan plus traversal |
| Fixed 3×3 neighborhood cleanup | \(O(N)\) | About nine neighborhood samples per pixel |
| Four Potts sweeps with ≤7 candidate labels | \(O(TN(c+d))\) | Roughly 29M candidate evaluations, plus neighbor work |
| Radius-2 bilateral filter | \(O(25N)\) | About 26M neighbor contributions |
| One area/DPID candidate | Approximately \(O(N+M)\) | One or a few source traversals |
| Local adaptive supports | Approximately \(O(TNb+TMK)\) | Several passes with bounded support overlap |

An RGBA8 image is 4 MiB; a `u8` label map is 1 MiB; an OKLab `f32×3` image is 12 MiB. Component IDs and queues add several MiB.

Avoid allocating a full \(N\times K\) float assignment tensor. At \(K=16\), that alone is 64 MiB.

Use deterministic ordering for histogram samples and tie decisions. A fixed RNG seed does not control `HashMap` iteration order or floating-point accumulation order.

The present frontend calls WASM synchronously. Resolution auditions increase the amount of work, so a worker and progressive candidate delivery are sensible architectural improvements.

---

## 10. Evaluation: measure cleanup and preservation separately

### 10.1 Use both synthetic ground truth and real AI images

#### Synthetic reconstruction suite

Start from native, clean pixel art, then generate controlled damage:

- Integer and fractional enlargement.
- Non-square enlargement.
- Grid shifts, jitter, and drift.
- Blur and anti-aliasing.
- In-cell gradients and color noise.
- JPEG-like chroma damage.
- Thin outlines and one-cell highlights.
- Ordered and irregular dithering.

Keep exact ground-truth cells, colors, and coordinate mappings.

Retro Diffusion’s **pixel-bench** supplies a useful framework with versioned distortions and separate size, color, placement, palette, and grid metrics. It requires the user to supply native artwork. [Repository](https://github.com/Retro-Diffusion/pixel-bench), [distortions](https://github.com/Retro-Diffusion/pixel-bench/blob/main/docs/DISTORTIONS.md), [metrics](https://github.com/Retro-Diffusion/pixel-bench/blob/main/docs/METRICS.md).

Its task definition needs inspection before adoption: some “native-art damage” is baked into the ground truth. A benchmark that rewards reproducing a damaged native pixel can disagree with an owner who wants that pixel repaired. [Ground-truth policy](https://github.com/Retro-Diffusion/pixel-bench/blob/main/docs/DISTORTIONS.md).

#### Real-image suite

Use the supplied samples plus a broader set of actual AI outputs. Annotate representative regions:

- Intended flat areas.
- Architectural lines.
- Small highlights.
- Text.
- Foliage.
- Water.
- Dithering.
- Tiny objects.

These inputs often have no uniquely correct original grid. Human judgments are necessary for the intended cleanup.

### 10.2 Speckle metrics

Let \(L\) be the output label map.

| Metric | Definition / use | Safeguard |
|---|---|---|
| **Isolated-label count** | Pixels disagreeing with every valid four-neighbor, also reported for eight-neighbor connectivity. | Legitimate one-cell details and dithering must be measured separately. |
| **Small-component count** | Number of components below several area thresholds. | Express thresholds relative to logical-cell area; distinguish source and logical pixels. |
| **Small-component area fraction** | Total area occupied by small components divided by image area. | Report in annotated flat regions and outside protected texture. |
| **Label total variation** | \(\sum_{(p,q)}[L_p\ne L_q]\), normalized by the number of tested edges. | Low variation alone rewards excessive simplification. |
| **Cell impurity** | \(1-\max_k n_{Ck}/|C|\), averaged over cells. | Measure against independently defined or ground-truth cells where possible. |
| **Pure-cell fraction** | Fraction of cells containing one output label. | Trivially 100% for one-color-per-cell algorithms; useful only with preservation metrics. |
| **Interior boundary density** | Label transitions away from intended region boundaries. | Requires masks or reliable boundary evidence. |

A solid-color output scores perfectly on several cleanup measures. That is why none should be used alone.

### 10.3 Detail-preservation metrics

1. **Native label accuracy**, where native ground truth exists.
2. **Grid-boundary precision and recall**, with one-to-one matching and explicit tolerance.
3. **Thin-line connectivity:** fragmentation and missing lengths in marked outlines.
4. **Small-feature survival:** whether an annotated eye, cross, window, letter stroke, or car remains represented.
5. **Contrast retention:** local foreground/background separation before and after reconstruction.
6. **Topology:** component and hole changes within selected features, considering both four- and eight-connectivity.
7. **Dither preservation:** label proportions and spatial correlation at the logical-cell scale.
8. **Palette error:** meaningful colors retained, duplicated, or merged.
9. **Local structural similarity:** useful supplementary evidence, not a sole objective.

SSIM measures local structural similarity against a reference; its original evaluation does not establish that it captures pixel-art editing quality. [Original paper and implementation](https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html).

For this task, compare against clean ground truth where available. Scoring only against noisy AI input can reward retaining the noise.

### 10.4 Useful no-reference detail-loss warnings

For each target footprint, retain:

```text
dominant label fraction
runner-up fraction
contrast between competing groups
strong minority evidence rejected
number of source components competing for one output cell
disagreement between reducers
```

A possible heatmap value is:

\[
R_j=\sum_{k\ne k^*}V_j(k)\,
\|P_k-P_{k^*}\|,
\]

where \(k^*\) is the selected label.

This measures discarded contrasting evidence, not semantic importance. A lost car and an unwanted bright artifact can both score highly. The heatmap is an inspection aid.

### 10.5 Human evaluation should include editing cost

Use blinded pairwise comparisons at:

- Native output size.
- Integer nearest-neighbor enlargement.
- Equal presentation size.

Ask separately:

- Which result has cleaner intended flat regions?
- Which preserves meaningful detail?
- Which would you choose to edit?
- How many edits, or how much time, are needed for a fixed correction task?

The owner’s preference for fewer editable pixels should be validated with actual touch-up effort, rather than inferred solely from image similarity.

---

## 11. A concrete experiment sequence

### Experiment 1: isolate the spatial-output issue

For algorithm 3, compare its current output with retained optimized labels using the **same final palette**.

Inspect speckle reduction and losses in thin features. This is the cleanest test of the implementation bug.

### Experiment 2: isolate assignment from palette selection

For each fixed palette, compare:

1. Independent nearest labels.
2. One gated mode pass.
3. Component merging.
4. Edge-weighted Potts cleanup.
5. Cell voting painted at original size.

This determines how much improvement comes from spatial assignment alone.

### Experiment 3: compare palette training data

With identical geometry and renderer, fit palettes from:

- Raw pixels.
- Mildly denoised pixels.
- Cell representatives.
- Cell representatives plus protected rare-detail samples.

Measure color fidelity and feature survival alongside speckles.

### Experiment 4: compare resolution candidates

At each selected target size, compare:

- Center sampling.
- Area mean then palette.
- Area-weighted label mode.
- Local dominant cluster.
- DPID then palette.
- DPID-inspired weighted label mode.
- Optional outline-expansion preprocessing.

Always reconstruct from source evidence.

### Experiment 5: test tiling fairly

Compare global and tiled processing at identical:

- Output dimensions.
- Global color budget.
- Reducer.
- Cleanup settings.

Then separately test a larger output budget.

Include artificial features crossing tile boundaries, and vary tile origin. A method whose result changes strongly when tile boundaries move needs better coordination.

### Experiment 6: calibrate recommendation thresholds

Use human selections to calibrate:

- Smallest acceptable resolution.
- Component-removal thresholds.
- Cell-confidence thresholds.
- Detail-risk warnings.
- Preference for clean versus detail-preserving reducers.

Keep held-out images so thresholds do not merely fit the coliseum.

---

## 12. Open questions, risks, and resolving experiments

| Open question / risk | Experiment that would resolve it |
|---|---|
| **How much speckling comes from algorithm 3 discarding labels?** | Render retained spatial labels and nearest labels with the same palette. |
| **Does the coliseum have one usable pitch, or multiple incompatible local pitches?** | Annotate representative boundaries across several regions; compare global and overlapping-window estimates. |
| **Is the detector selecting a harmonic or architectural repetition?** | Evaluate pitch \(s\), \(s/2\), and \(2s\) with annotated cuts and reconstruction quality. |
| **How often are near-duplicate palette colors responsible?** | Measure palette separation and within-cell label fragmentation while holding geometry fixed. |
| **Does building palettes from cell representatives help rare features or suppress them?** | Compare raw, cell-based, and rare-feature-augmented training sets at fixed \(K\). |
| **Which small components are legitimate details?** | Annotate highlights, eyes, stars, and windows; measure survival versus sub-cell artifact removal. |
| **Can dithering be protected reliably?** | Test ordered, irregular, and stochastic dithering alongside similarly scaled corruption; report false protection and false deletion separately. |
| **How much does non-native sampling phase matter?** | Compare modest support-offset alternatives at identical target dimensions. |
| **Does DPID preserve detail or exaggerate noise?** | Measure true-feature survival and false-detail amplification, with original and cleaned guides. |
| **Do local grids improve reconstruction enough to justify complexity?** | Compare global cuts, coordinated local estimates, and independent tiles at equal output size. |
| **Are seams photometric, geometric, or both?** | Test shared versus separate palettes independently from shared versus separate grids. |
| **Does adaptive resolution produce an acceptable pixel-art style?** | Blindly compare uniform-grid and aligned multiscale outputs; record export and editing preferences. |
| **Will synthetic restoration results transfer to actual AI art?** | Keep a separate real-AI evaluation set with human annotations and selections. |
| **What happens with transparency?** | Add hard-alpha sprites, semi-transparent fringes, and hidden RGB tests. The supplied samples cannot answer this. |
| **What runtime and memory are acceptable in browsers?** | Benchmark representative desktop and mobile browsers with peak-memory and responsiveness measurements. |
| **Can external code be reused within the project’s dependency policy?** | Inspect the exact chosen version and license before adoption. Libimagequant documents GPL/commercial licensing. [Documentation](https://pngquant.org/lib/) |

## 13. Recommended implementation order

1. **Fix algorithm 3’s output assignment** and compare it under a fixed palette.
2. **Separate palette, label map, grid, and rendering data.**
3. **Add original-size cell reconstruction** using the existing cuts.
4. **Add explicit target resolution and area-weighted label voting.**
5. **Generate a small audition set directly from source evidence.**
6. **Add conservative component merging and edge-weighted label cleanup.**
7. **Add DPID as a detail-preserving alternative with loss overlays.**
8. **Build the synthetic and real-image evaluation loop.**
9. **Investigate coordinated local grids only after global failure cases are measured.**
10. **Offer adaptive cell sizes as an explicit experimental output style.**

For the primary coliseum case, I would start with **cell reconstruction painted back at full size, a palette fitted from robust cell representatives, and several finer-grid alternatives**. That directly addresses the missing spatial constraint while giving the owner control over the unavoidable detail-versus-resolution tradeoff.