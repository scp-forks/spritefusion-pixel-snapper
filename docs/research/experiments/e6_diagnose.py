"""E6: WHY does speckle happen? (a) vs k, (b) where the fragment pixels live, (c) assignment margin,
(d) which palette entries are 'fringe colours'."""
import numpy as np
from lab import *
orig = load(R + 'colliseum.png'); olab = srgb_to_oklab(orig); h, w = orig.shape[:2]
px, py = profiles_lab(olab); pit = (pitch_acf(px)[0] + pitch_acf(py)[0]) / 2
print('(a) speckle vs palette size (shipping k-means OKLab, no-resize)')
for k in (4, 8, 16, 32, 64):
    lab, pal = to_labels(load(O + f'col_k{k}_a0.png'))
    print(f' k={k:2d}', fmt(speckle_metrics(lab, pit)))
lab, pal = to_labels(load(O + 'col_k16_a0.png')); pal_lab = srgb_to_oklab(pal); k = len(pal)
comp = measure.label(lab + 1, connectivity=1, background=0); area = np.bincount(comp.ravel())[comp]
tiny = area < 0.25 * pit * pit
# (b) distance of each pixel to the nearest strong edge of the *original* image
gx = np.zeros((h, w), np.float32); gy = np.zeros((h, w), np.float32)
gx[:, 1:] = np.sqrt(((olab[:, 1:] - olab[:, :-1]) ** 2).sum(-1)); gy[1:] = np.sqrt(((olab[1:] - olab[:-1]) ** 2).sum(-1))
g = np.maximum(gx, gy)
# local contrast = max dE to the 5x5 neighbourhood mean (blur) -> 'is this pixel on a soft edge?'
blur = np.stack([ndi.uniform_filter(olab[..., c], 5) for c in range(3)], -1)
edge = np.sqrt(((ndi.maximum_filter(olab, size=(5, 5, 1)) - ndi.minimum_filter(olab, size=(5, 5, 1))) ** 2).sum(-1))
print('\n(b) local 5x5 colour range (OKLab) at fragment pixels vs all pixels')
for nm, m in (('all pixels', np.ones_like(tiny)), ('tiny-fragment pixels', tiny)):
    v = edge[m]; print(f' {nm:22s} n={m.sum():8d} median range={np.median(v):.4f}  share in flat areas(<0.03)={100*(v<0.03).mean():.1f}%  soft(0.03-0.10)={100*((v>=0.03)&(v<0.10)).mean():.1f}%  hard edge(>=0.10)={100*(v>=0.10).mean():.1f}%')
# (c) assignment margin: how much closer is the chosen palette colour than the runner-up?
flat = olab.reshape(-1, 3); d = np.sqrt(((flat[:, None, :] - pal_lab[None]) ** 2).sum(-1)); ds = np.sort(d, 1)
margin = (ds[:, 1] - ds[:, 0]).reshape(h, w)
print('\n(c) assignment margin d2-d1 (OKLab): small = the pixel sits on a palette decision boundary')
for nm, m in (('all pixels', np.ones_like(tiny)), ('tiny-fragment pixels', tiny)):
    v = margin[m]; print(f' {nm:22s} median={np.median(v):.4f}  share with margin<0.01: {100*(v<0.01).mean():.1f}%  <0.02: {100*(v<0.02).mean():.1f}%')
print(' palette nearest-neighbour distances (sorted):', np.round(np.sort(np.sort(np.sqrt(((pal_lab[:, None] - pal_lab[None]) ** 2).sum(-1)), 1)[:, 1]), 3))
noise = np.sqrt(((olab - blur) ** 2).sum(-1)); print(' in-region noise: median |pixel - 5x5 mean| in flat areas = %.4f' % np.median(noise[edge < 0.03]))
# (d) fringe palette entries
print('\n(d) per palette entry: share of image, share of its pixels on soft/hard edges, share in tiny fragments')
onedge = edge >= 0.06
for j in np.argsort(-np.bincount(lab.ravel(), minlength=k)):
    m = lab == j
    print(f'  #{j:2d} rgb={tuple(int(c) for c in pal[j])!s:16s} area={100*m.mean():5.1f}%  on-edge={100*onedge[m].mean():5.1f}%  in-tiny-fragments={100*tiny[m].mean():5.1f}%')
print('  (image-wide on-edge share = %.1f%%)' % (100 * onedge.mean()))
