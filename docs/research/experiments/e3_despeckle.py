"""E3: Problem A. Same palette (shipping k-means-OKLab output, k=16), different spatial strategies."""
import sys, time, numpy as np
from lab import *
name = sys.argv[1] if len(sys.argv) > 1 else 'colliseum'
qpath = sys.argv[2] if len(sys.argv) > 2 else O + 'col_k16_a0.png'
orig = load(R + name + '.png'); olab = srgb_to_oklab(orig); h, w = orig.shape[:2]
q = load(qpath); base, pal = to_labels(q); k = len(pal); pal_lab = srgb_to_oklab(pal)
px, py = profiles_lab(olab)
pit = (pitch_acf(px)[0] + pitch_acf(py)[0]) / 2
dx = cuts_dp(px, pit, w); dy = cuts_dp(py, pit, h)
rx, ry, _ = grid_rust(q)
print(f'{name}: pitch {pit:.2f}; dp grid {len(dx)-1}x{len(dy)-1}; rust-port grid {len(rx)-1}x{len(ry)-1}')
res = {}
def report(tag, lab, pal_lab=pal_lab, t0=None):
    out_lab = pal_lab[lab]
    m = speckle_metrics(lab, pit); m.update(grid_purity(lab, dx, dy, len(pal_lab))); m.update(fidelity(olab, out_lab))
    m['colors'] = int(len(np.unique(lab)))
    if t0: m['sec'] = time.time() - t0
    res[tag] = m
    save(O + f'{name}_{tag}.png', oklab_to_srgb(out_lab))
    print(f'{tag:22s}', fmt(m), flush=True)

report('v0_baseline', base)
t = time.time(); report('v1_mode3x2', mode_filter(base, k, 1, 2), t0=t)
t = time.time(); report('v1_mode5x2', mode_filter(base, k, 2, 2), t0=t)
t = time.time(); report('v2_cc_q', merge_small_components(base, 0.25 * pit * pit), t0=t)
t = time.time(); report('v2_cc_half', merge_small_components(base, 0.5 * pit * pit), t0=t)
for lam in (0.0003, 0.001, 0.003):
    t = time.time(); icm = potts_icm(olab, pal_lab, base, lam=lam, iters=10); report(f'v3_icm_{lam}', icm, t0=t)
    if lam == 0.001:
        t = time.time(); report('v9_icm+cc_q', merge_small_components(icm, 0.25 * pit * pit), t0=t)
# edge-preserving prefilter: per-channel median in OKLab, then nearest palette
t = time.time(); med = np.stack([ndi.median_filter(olab[..., c], size=5) for c in range(3)], -1)
report('v4_median5', nearest_labels(med, pal_lab), t0=t)
# snap in place
for gname, (cx, cy) in (('rust', (rx, ry)), ('dp', (dx, dy))):
    cid, nx, ny = cell_index(cx, cy, w, h)
    t = time.time(); mode, pur, _ = cell_mode(base, cid, nx * ny, k); report(f'v5_snap_{gname}_mode', paint(mode, cid), t0=t)
    inner = interior_mask(cx, cy, w, h, 2)
    t = time.time(); mean, n = cell_mean(olab, cid, nx * ny, inner)
    lab_cells = nearest_labels(mean[None], pal_lab)[0]
    report(f'v7_snap_{gname}_inmean', paint(lab_cells, cid), t0=t)
    # cell-first palette: cluster the per-cell interior means, not the raw pixels
    t = time.time(); cpal = kmeans_oklab(mean[n > 0], k)
    lab_cells = nearest_labels(mean[None], cpal)[0]
    report(f'v8_snap_{gname}_cellpal', paint(lab_cells, cid), pal_lab=cpal, t0=t)
import json; json.dump(res, open(O + f'{name}_e3.json', 'w'), indent=1)
