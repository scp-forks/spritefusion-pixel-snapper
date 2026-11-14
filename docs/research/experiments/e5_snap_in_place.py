"""E5: snap-in-place at several grid resolutions + adaptive (quadtree) variant, cell-first palette."""
import sys, time, numpy as np
from lab import *
name = sys.argv[1] if len(sys.argv) > 1 else 'colliseum'; k = int(sys.argv[2]) if len(sys.argv) > 2 else 16
orig = load(R + name + '.png'); olab = srgb_to_oklab(orig); h, w = orig.shape[:2]
px, py = profiles_lab(olab); pit = (pitch_acf(px)[0] + pitch_acf(py)[0]) / 2
dx = cuts_dp(px, pit, w); dy = cuts_dp(py, pit, h)
def report(tag, lab, pal_lab, extra=''):
    out_lab = pal_lab[lab]
    m = speckle_metrics(lab, pit); m.update(fidelity(olab, out_lab))
    save(O + f'{name}_{tag}.png', oklab_to_srgb(out_lab)); print(f'{tag:20s}', fmt(m), extra, flush=True)
grids = {'s1_native': (dx, dy), 's2_x2': (subdivide(dx, 2), subdivide(dy, 2)), 's3_x3': (subdivide(dx, 3), subdivide(dy, 3)),
         's2b_halfpitch': (cuts_dp(px, pit / 2, w), cuts_dp(py, pit / 2, h))}
for tag, (cx, cy) in grids.items():
    mean, cid, nx, ny = snap_cells(olab, cx, cy, 1)
    pal = kmeans_oklab(mean, k); cl = nearest_labels(mean[None], pal)[0]
    report(tag, cl[cid], pal, f'grid={nx}x{ny}')
    save(O + f'{name}_{tag}_small.png', oklab_to_srgb(pal[cl].reshape(ny, nx, 3)))
for tau in (0.02, 0.03, 0.045):
    img, split, _ = adaptive_snap(olab, dx, dy, tau=tau)
    flat = img.reshape(-1, 3); uniq, inv = np.unique(np.round(flat, 5), axis=0, return_inverse=True)
    cnt = np.bincount(inv.ravel())
    pal = kmeans_oklab(uniq.astype(np.float32), k, weights=np.sqrt(cnt))
    lab = nearest_labels(uniq.astype(np.float32)[None], pal)[0][inv.ravel()].reshape(h, w)
    report(f's4_adaptive_{tau}', lab, pal, f'split={100*split.mean():.1f}% of native cells -> leaves={int(split.size + 3*split.sum())}')
    vis = (np.kron(split, np.ones((1, 1))) * 255).astype(np.uint8); Image.fromarray(vis).save(O + f'{name}_s4_splitmap_{tau}.png')
