"""E9: (1) a 'does this image really have a lattice?' score that is not fooled by JPEG 8x8 blocking,
(2) seam magnitude for per-tile palettes, (3) identity check of snap-in-place on clean pixel art."""
import numpy as np
from lab import *
def residual(olab, cx, cy):
    h, w = olab.shape[:2]; cid, nx, ny = cell_index(cx, cy, w, h); m, _ = cell_mean(olab, cid, nx * ny)
    return float(np.sqrt(((olab - m[cid]) ** 2).sum(-1)).mean())
print('(1) lattice score = 1 - residual(fitted grid @ p) / geometric-mean residual(uniform grids @ 0.8p and 1.25p)')
for name in ('colliseum', 'pixel-art-sample-1024x1024', 'cat-1024x1024'):
    olab = srgb_to_oklab(load(R + name + '.png')); h, w = olab.shape[:2]
    px, py = profiles_lab(olab); (p1, a1), (p2, a2) = pitch_acf(px), pitch_acf(py); p = (p1 + p2) / 2
    fit = residual(olab, cuts_dp(px, p, w), cuts_dp(py, p, h))
    off = [residual(olab, cuts_uniform(w, int(round(w / s))), cuts_uniform(h, int(round(h / s)))) for s in (0.8 * p, 1.25 * p)]
    print(f' {name:28s} pitch={p:.2f} acf-strength x/y={a1:.2f}/{a2:.2f}  fitted={fit:.4f}  off-pitch={off[0]:.4f},{off[1]:.4f}  lattice score={1 - fit / np.sqrt(off[0] * off[1]):.3f}')
print('\n(3) clean pixel art (pitch 4): snap-in-place should be close to a no-op')
name = 'pixel-art-sample-1024x1024'; orig = load(R + name + '.png'); olab = srgb_to_oklab(orig); h, w = orig.shape[:2]
px, py = profiles_lab(olab); p = 4.0; dx, dy = cuts_dp(px, p, w), cuts_dp(py, p, h)
base, pal = to_labels(load(O + 'sample_k16_a0.png')); pal_lab = srgb_to_oklab(pal)
print(' baseline alg0 k=16      ', fmt(speckle_metrics(base, p)), fmt(fidelity(olab, pal_lab[base])))
mean, cid, nx, ny = snap_cells(olab, dx, dy, 1); cl = nearest_labels(mean[None], pal_lab)[0][cid]
print(f' snap-in-place {nx}x{ny}', fmt(speckle_metrics(cl, p)), fmt(fidelity(olab, pal_lab[cl])), ' pixels changed vs baseline: %.1f%%' % (100 * (cl != base).mean()))
save(O + 'sample_snapinplace.png', oklab_to_srgb(pal_lab[cl]))
u, core = core_constrained(olab, pal_lab, base, 0.03, 2); u = merge_small_components(u, 0.25 * p * p)
print(' unmix + cc(1/4 cell)    ', fmt(speckle_metrics(u, p)), fmt(fidelity(olab, pal_lab[u])), ' pixels changed vs baseline: %.1f%%' % (100 * (u != base).mean()))
save(O + 'sample_unmix.png', oklab_to_srgb(pal_lab[u]))
