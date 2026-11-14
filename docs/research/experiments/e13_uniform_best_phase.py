"""E13: fairness check on 'elastic fit beats uniform by 35%'. E4 compared against a uniform grid at phase 0.
pixel-art-fixer deliberately uses an EVEN lattice (it dropped per-cut snapping). So: how good is the best
possible even lattice (pitch and phase searched per axis, non-square allowed) against the bounded elastic fit?"""
import sys, numpy as np
from lab import *
name = sys.argv[1] if len(sys.argv) > 1 else 'colliseum'
olab = srgb_to_oklab(load(R + name + '.png')); h, w = olab.shape[:2]
px, py = profiles_lab(olab); p0 = (pitch_acf(px)[0] + pitch_acf(py)[0]) / 2
def res(cx, cy):
    cid, nx, ny = cell_index(cx, cy, w, h); m, _ = cell_mean(olab, cid, nx * ny); return float(np.sqrt(((olab - m[cid]) ** 2).sum(-1)).mean())
def even(limit, pitch, phase):
    c = np.round(np.arange(phase, limit, pitch)).astype(int); c = c[(c > 0) & (c < limit)]
    return [0] + sorted(set(c.tolist())) + [limit]
dx, dy = cuts_dp(px, p0, w), cuts_dp(py, p0, h)
print(f'{name}: pitch {p0:.2f}; elastic {len(dx)-1}x{len(dy)-1} residual {res(dx, dy):.5f}; even lattice phase 0: {res(even(w, p0, 0), even(h, p0, 0)):.5f}')
pitches = np.arange(p0 * 0.92, p0 * 1.08 + 1e-9, p0 * 0.005); best = {}
cx, cy = even(w, p0, 0), even(h, p0, 0)
for rnd in range(2):                      # coordinate descent: x then y, twice
    for axis in 'xy':
        cand = []
        for p in pitches:
            for ph in np.arange(0, p, p / 8):
                g = even(w if axis == 'x' else h, p, ph)
                cand.append((res(g, cy) if axis == 'x' else res(cx, g), p, ph, g))
        r, p, ph, g = min(cand, key=lambda t: t[0]); best[axis] = (p, ph)
        if axis == 'x': cx = g
        else: cy = g
print(f' best even lattice: pitch x {best["x"][0]:.3f} phase {best["x"][1]:.2f}; pitch y {best["y"][0]:.3f} phase {best["y"][1]:.2f}; cells {len(cx)-1}x{len(cy)-1}; residual {res(cx, cy):.5f}')
print(f' mixed: even x + elastic y {res(cx, dy):.5f}; elastic x + even y {res(dx, cy):.5f}')
