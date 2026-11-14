"""E10: elastic grid fitted at arbitrary target pitches. residual vs output pixel count, against unaligned uniform grids.
'efficiency' = how many times more pixels an unaligned grid needs to reach the same residual."""
import sys, numpy as np
from lab import *
name = sys.argv[1] if len(sys.argv) > 1 else 'colliseum'
olab = srgb_to_oklab(load(R + name + '.png')); h, w = olab.shape[:2]
px, py = profiles_lab(olab); p0 = (pitch_acf(px)[0] + pitch_acf(py)[0]) / 2
def res(cx, cy):
    cid, nx, ny = cell_index(cx, cy, w, h); m, _ = cell_mean(olab, cid, nx * ny); return float(np.sqrt(((olab - m[cid]) ** 2).sum(-1)).mean()), nx, ny
us = np.arange(2.5, 16.01, 0.25); ur = []; upx = []
for s in us:
    r, nx, ny = res(cuts_uniform(w, int(round(w / s))), cuts_uniform(h, int(round(h / s)))); ur.append(r); upx.append(nx * ny)
ur = np.array(ur); upx = np.array(upx, float)
print(f'{name}: native pitch {p0:.2f}')
print(' pitch  ratio  grid       pixels  residual  unaligned-px-for-same-residual  efficiency')
for s in [p0 * f for f in (2, 1.5, 1.25, 1, 0.8, 2 / 3, 0.5, 0.4, 1 / 3, 0.25)]:
    r, nx, ny = res(cuts_dp(px, s, w), cuts_dp(py, s, h))
    need = float(np.exp(np.interp(r, ur, np.log(upx))))   # ur increases with s
    print(f' {s:5.2f}  {p0/s:4.2f}x  {nx:4d}x{ny:<4d} {nx*ny:7d}  {r:.5f}   {need:9.0f}   {need/(nx*ny):.2f}x')
