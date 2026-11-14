"""E4: Problem B. How does 'how well a grid explains the image' vary with grid resolution?
residual(grid) = mean OKLab dE between the original and the per-cell mean painted back.
A real lattice shows up as dips at the native pitch and its integer subdivisions."""
import sys, numpy as np
from lab import *
name = sys.argv[1] if len(sys.argv) > 1 else 'colliseum'
orig = load(R + name + '.png'); olab = srgb_to_oklab(orig); h, w = orig.shape[:2]
px, py = profiles_lab(olab)
pit = (pitch_acf(px)[0] + pitch_acf(py)[0]) / 2
def residual(cx, cy):
    cid, nx, ny = cell_index(cx, cy, w, h)
    mean, _ = cell_mean(olab, cid, nx * ny)
    d = np.sqrt(((olab - mean[cid]) ** 2).sum(-1))
    return float(d.mean()), nx, ny
print(f'{name}: {w}x{h}, acf pitch {pit:.3f}')
print('-- uniform grids (phase 0), cell size s --')
rows = []
for s in np.arange(3.0, 14.01, 0.25):
    r, nx, ny = residual(cuts_uniform(w, int(round(w / s))), cuts_uniform(h, int(round(h / s))))
    rows.append((s, nx, ny, r))
# detrend: residual grows ~ smoothly with s; report dips relative to neighbours
rs = np.array([r[3] for r in rows])
for i, (s, nx, ny, r) in enumerate(rows):
    lo = rs[max(0, i - 2):i + 3]
    flag = ' <-- local min' if r == lo.min() and 0 < i < len(rows) - 1 else ''
    print(f's={s:5.2f}  {nx:4d}x{ny:<4d} residual={r:.5f}{flag}')
print('-- elastic (dp) grid at pitch p and its refinements --')
dx = cuts_dp(px, pit, w); dy = cuts_dp(py, pit, h)
for f in (1, 2, 3, 4):
    r, nx, ny = residual(subdivide(dx, f), subdivide(dy, f))
    ru, _, _ = residual(cuts_uniform(w, nx), cuts_uniform(h, ny))
    print(f'dp x{f}: {nx}x{ny}  residual={r:.5f}   (uniform grid with same cell count: {ru:.5f}, gain {100*(ru-r)/ru:.1f}%)')
# dp grid fitted directly at half pitch
hx = cuts_dp(px, pit / 2, w); hy = cuts_dp(py, pit / 2, h)
r, nx, ny = residual(hx, hy); ru, _, _ = residual(cuts_uniform(w, nx), cuts_uniform(h, ny))
print(f'dp fitted at pitch/2: {nx}x{ny} residual={r:.5f} (uniform same count {ru:.5f}, gain {100*(ru-r)/ru:.1f}%)')
