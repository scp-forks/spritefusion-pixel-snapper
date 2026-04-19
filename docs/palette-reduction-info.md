After checking recent arXiv and 2024–2026 papers, my recommendation is **not** to start with a neural palette-learning model for your use case. The newest arXiv work I found that is closest in spirit—**AutoPalette** (2024) and **DCQ** (2026)—is aimed at **dataset/training-set compression**, where the goal is preserving downstream model performance, not producing the best-looking **single quantized image** for games. Related arXiv palette work in diffusion is mostly about **palette transfer / palette-conditioned generation**, not straightforward fixed-resolution reduction of an existing image. ([arXiv](https://arxiv.org/html/2411.11329v1))

For **single-image color reduction without resizing**, the strongest evidence still points to **classical quantization with perceptual color handling**. A 2024 comparative study tested **10 quantizers across 8 image-quality metrics** and found that **ADU** was best overall on the more perception-oriented group of metrics, while **BSITATCQ** was strongest on MSE/PSNR/SAM; the same paper explicitly recommends **not using a single pixelwise metric alone** to judge quality. A new 2026 arXiv paper then showed that plain RGB is not always the best space for k-means: on 148 images, **RGB won about half the time**, but **CIE-XYZ often did better at higher palette sizes**, and **CIE-LUV sometimes did better at lower k**. ([Springer](https://link.springer.com/article/10.1007/s00530-023-01206-7))

So the best practical recommendation is:

1. **Keep the image at its original resolution** and only remap pixels to a smaller palette. Recent quantization papers explicitly describe color quantization as reducing colors while **maintaining image resolution / the same number of pixels**. ([arXiv](https://arxiv.org/html/2409.09488v1))
2. **Quantize in a perceptual space, not raw RGB.** For a robust default, use **OKLab** or **CIELUV**; for **32 colors**, also test **CIE-XYZ** because recent evidence suggests it often helps at higher k. ([arXiv](https://arxiv.org/abs/2601.19117))
3. **Use a hybrid palette builder:** good initialization, then local refinement. The most proven practical baseline here is still **modified median cut + k-means/Voronoi refinement + adaptive dithering**. That is effectively what **pngquant/libimagequant** does: modified median cut, variance-based splitting, perception-weighted histogramming, repeated refinement, final Voronoi iteration, and adaptive edge-aware dithering. ([pngquant.org](https://pngquant.org/))
4. **Evaluate with perceptual metrics** such as **SSIM / MS-SSIM / VIF**, not MSE alone. ([Springer](https://link.springer.com/article/10.1007/s00530-023-01206-7))

For your specific palette sizes, I would use these defaults:

- **32 colors:** perceptual **k-means** in **CIE-XYZ or OKLab**, with a good initializer and mild adaptive dithering. This is the safest “general-purpose game art import” setting from the recent literature. ([arXiv](https://arxiv.org/abs/2601.19117))
- **16 colors:** perceptual **k-means / hybrid median-cut→k-means**, but more conservative dithering. Low-k is where perceptual-space choice matters more, and **CIELUV/OKLab** is a good place to start. ([arXiv](https://arxiv.org/abs/2601.19117))
- **8 colors:** use a **spatial quantization** method if quality matters more than speed. The classic spatial approach simultaneously does palette selection and dithering; the current Rust implementation **rscolorq** explicitly says it is especially good around **4, 8, or 16 colors**, though it is slow on large images. ([GitHub](https://github.com/okaneco/rscolorq))

If you want the best balance of research-backed quality and engineering simplicity, I would build this:

**Recommended algorithm**

- Convert sRGB → linear RGB → **OKLab**.
- Build a weighted histogram / reduced sample set first; recent work like **SoftBinReduce** is aimed exactly at improving the **quality-vs-speed tradeoff** for quantization, especially when data is heavily reduced. ([Springer](https://link.springer.com/article/10.1007/s00530-025-01755-z))
- Initialize palette with **k-means++** or median-cut-style splitting.
- Refine palette with **weighted k-means**.
- Remap every original pixel to its nearest palette entry, preserving width/height.
- Apply **adaptive dithering only where needed**; avoid uniform Floyd–Steinberg over the whole image.
- For **8/16-color** output, optionally run a **spatial quantization** pass instead of ordinary post-dithering.

What I would **not** choose as the default:

- **Latest neural dataset methods** like AutoPalette/DCQ: interesting ideas, but wrong target problem for your case. ([arXiv](https://arxiv.org/html/2411.11329v1))
- **Metaheuristics** like grey wolf optimization as the first implementation: the 2026 paper is promising and beats many older methods, but it is heavier and still not clearly the most practical first choice for a game pipeline. ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1077314226000263?utm_source=chatgpt.com))
- **Plain RGB k-means** with naive Floyd–Steinberg everywhere: recent evidence suggests you can do better with perceptual spaces and smarter dithering. ([arXiv](https://arxiv.org/abs/2601.19117))

If you want a single sentence answer: **implement an OKLab/CIELUV quantizer with k-means refinement and adaptive dithering, and switch to spatial quantization for 8–16 color targets; use libimagequant as your baseline to beat.** ([pngquant.org](https://pngquant.org/))
