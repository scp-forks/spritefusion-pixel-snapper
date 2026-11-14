"""E7: grid-free 'un-mixing' (core-constrained reassignment) vs the generic label regularisers."""
import sys, time, numpy as np
from lab import *
name = sys.argv[1] if len(sys.argv) > 1 else 'colliseum'; qpath = sys.argv[2] if len(sys.argv) > 2 else O + 'col_k16_a0.png'
orig = load(R + name + '.png'); olab = srgb_to_oklab(orig); h, w = orig.shape[:2]
base, pal = to_labels(load(qpath)); k = len(pal); pal_lab = srgb_to_oklab(pal)
px, py = profiles_lab(olab); pit = (pitch_acf(px)[0] + pitch_acf(py)[0]) / 2
dx = cuts_dp(px, pit, w); dy = cuts_dp(py, pit, h)
def report(tag, lab, t0=None):
    m = speckle_metrics(lab, pit); m.update(grid_purity(lab, dx, dy, k)); m.update(fidelity(olab, pal_lab[lab]))
    if t0: m['sec'] = time.time() - t0
    save(O + f'{name}_{tag}.png', oklab_to_srgb(pal_lab[lab])); print(f'{tag:26s}', fmt(m), flush=True)
report('v0_baseline', base)
for thr in (0.02, 0.03, 0.05):
    for rad in (2, 4):
        t = time.time(); u, core = core_constrained(olab, pal_lab, base, thr, rad)
        report(f'v10_unmix_t{thr}_r{rad}', u, t)
t = time.time(); u, core = core_constrained(olab, pal_lab, base, 0.03, 3); print('core share %.1f%%' % (100 * core.mean()))
u2 = merge_small_components(u, 0.25 * pit * pit); report('v11_unmix+cc_q', u2, t)
u3 = merge_small_components(mode_filter(u, k, 1, 1), 0.25 * pit * pit); report('v12_unmix+mode3+cc_q', u3, t)
# then snap in place on the un-mixed labels at x2 grid, majority vote
for f in (1, 2):
    cx, cy = subdivide(dx, f), subdivide(dy, f); cid, nx, ny = cell_index(cx, cy, w, h)
    for tag, src in (('raw', base), ('unmix', u2)):
        mode, pur, _ = cell_mode(src, cid, nx * ny, k); report(f'v13_snap_x{f}_mode_of_{tag}', mode[cid])
