"""E8: Problem C. Independent tiles: (a) local palettes vs one global palette, (b) local grid fits vs one
global separable grid."""
import sys, numpy as np
from lab import *
name = sys.argv[1] if len(sys.argv) > 1 else 'colliseum'
orig = load(R + name + '.png'); olab = srgb_to_oklab(orig); h, w = orig.shape[:2]
px, py = profiles_lab(olab); pit = (pitch_acf(px)[0] + pitch_acf(py)[0]) / 2
rng = np.random.default_rng(0)
def sample(a, n=40000):
    f = a.reshape(-1, 3); return f[rng.choice(len(f), min(n, len(f)), replace=False)]
def seam_stat(out_lab, xs, ys):
    """Share of pixel pairs straddling a line that get different output colours although the source pair is ~identical."""
    def one(a, b, oa, ob):
        same_src = np.sqrt(((oa - ob) ** 2).sum(-1)) < 0.01
        diff_out = np.abs(a - b).sum(-1) > 1e-6
        return (same_src & diff_out).sum(), same_src.sum()
    n = d = 0
    for x in xs: a, b = one(out_lab[:, x - 1], out_lab[:, x], olab[:, x - 1], olab[:, x]); n += a; d += b
    for y in ys: a, b = one(out_lab[y - 1], out_lab[y], olab[y - 1], olab[y]); n += a; d += b
    return 100.0 * n / max(d, 1)
print(f'== (a) palettes: {name} ==')
for T in (1, 2, 4):
    for k in ((16, 64, 256) if T == 1 else (16,)):
        out = np.zeros_like(olab); tot = 0
        xs = [int(round(i * w / T)) for i in range(T + 1)]; ys = [int(round(i * h / T)) for i in range(T + 1)]
        for i in range(T):
            for j in range(T):
                t = olab[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                pal = kmeans_oklab(sample(t), k, iters=20); tot += len(pal)
                out[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = pal[nearest_labels(t, pal)]
        ctrl_x = [x + (w // (2 * T)) for x in xs[:-1]]; ctrl_y = [y + (h // (2 * T)) for y in ys[:-1]]
        seam = seam_stat(out, xs[1:-1], ys[1:-1]) if T > 1 else float('nan')
        lab_id = np.unique(np.round(out.reshape(-1, 3), 4), axis=0, return_inverse=True)[1].reshape(h, w)
        print(f' tiles {T}x{T} k={k:3d}/tile  total colours={tot:4d}  dE_mean={fidelity(olab, out)["dE_mean"]:.5f}  '
              f'false-seam%={seam:.2f}  (same stat on non-seam control lines: {seam_stat(out, ctrl_x, ctrl_y):.2f})  tiny fragments={speckle_metrics(lab_id, pit)["tiny"]}')
        if T == 4: save(O + f'{name}_tiles4_k16.png', oklab_to_srgb(out))
print(f'\n== (b) grids: residual of a per-cell-mean reconstruction, pitch {pit:.2f} ==')
dE = lambda a, b: float(np.sqrt(((a - b) ** 2).sum(-1)).mean())
def recon(a, cx, cy):
    hh, ww = a.shape[:2]; cid, nx, ny = cell_index(cx, cy, ww, hh); m, _ = cell_mean(a, cid, nx * ny); return m[cid], nx * ny
gx, gy = cuts_dp(px, pit, w), cuts_dp(py, pit, h)
r, n = recon(olab, gx, gy); print(f' global separable elastic grid: cells={n}  residual={dE(olab, r):.5f}')
r, n = recon(olab, cuts_uniform(w, len(gx) - 1), cuts_uniform(h, len(gy) - 1)); print(f' global uniform grid          : cells={n}  residual={dE(olab, r):.5f}')
for T in (2, 4, 8):
    out = np.zeros_like(olab); cells = 0
    xs = [int(round(i * w / T)) for i in range(T + 1)]; ys = [int(round(i * h / T)) for i in range(T + 1)]
    for i in range(T):
        for j in range(T):
            t = olab[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]; tx, ty = profiles_lab(t)
            r, n = recon(t, cuts_dp(tx, pit, t.shape[1]), cuts_dp(ty, pit, t.shape[0])); out[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = r; cells += n
    print(f' {T}x{T} independent tiles, own elastic grid (shared pitch): cells={cells}  residual={dE(olab, out):.5f}')
# banded: column cuts refit per horizontal band, row cuts global (a cheap non-separable grid)
for nb in (2, 4, 8):
    out = np.zeros_like(olab); cells = 0
    ys = [gy[int(round(i * (len(gy) - 1) / nb))] for i in range(nb + 1)]   # band edges ON row cuts => no slivers
    for i in range(nb):
        t = olab[ys[i]:ys[i + 1]]; tx, _ = profiles_lab(t)
        rows = [c - ys[i] for c in gy if ys[i] <= c <= ys[i + 1]]
        r, n = recon(t, cuts_dp(tx, pit, w), rows); out[ys[i]:ys[i + 1]] = r; cells += n
    print(f' {nb} horizontal bands: global rows + per-band column cuts: cells={cells}  residual={dE(olab, out):.5f}')
