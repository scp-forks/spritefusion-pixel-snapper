"""E1: how bad is speckle today, per quantizer? E2: what is the logical grid?"""
import sys, numpy as np
from lab import *
orig = load(R + 'colliseum.png'); olab = srgb_to_oklab(orig); h, w = orig.shape[:2]

# --- grid ---
px, py = profiles_lab(olab)
for nm, p in (('x', px), ('y', py)):
    print(nm, 'fft pitch', pitch_fft(p, 5, 20), 'acf pitch', pitch_acf(p))
q0 = load(O + 'col_k16_a0.png')
cx, cy, step = grid_rust(q0)
print('rust-port grid on alg0 output: step', step, 'cells', len(cx) - 1, 'x', len(cy) - 1)
pit = (pitch_acf(px)[0] + pitch_acf(py)[0]) / 2
dx = cuts_dp(px, pit, w); dy = cuts_dp(py, pit, h)
print('dp grid: pitch', pit, 'cells', len(dx) - 1, 'x', len(dy) - 1,
      'step x mean/std', np.diff(dx).mean(), np.diff(dx).std(), 'y', np.diff(dy).mean(), np.diff(dy).std())
print('rust steps x mean/std', np.diff(cx).mean(), np.diff(cx).std(), 'hist', np.bincount(np.diff(cx)))
print('dp   steps x hist', np.bincount(np.diff(dx)))
print('rust steps y hist', np.bincount(np.diff(cy)))
print('dp   steps y hist', np.bincount(np.diff(dy)))
np.save(O + 'dp_cuts_x.npy', np.array(dx)); np.save(O + 'dp_cuts_y.npy', np.array(dy))
np.save(O + 'rust_cuts_x.npy', np.array(cx)); np.save(O + 'rust_cuts_y.npy', np.array(cy))

# --- overlay for eyeballing ---
def overlay(rgb, cx, cy, col):
    o = rgb.copy()
    for c in cx[1:-1]: o[:, c] = (o[:, c] * 0.3 + np.array(col) * 0.7).astype(np.uint8)
    for c in cy[1:-1]: o[c, :] = (o[c, :] * 0.3 + np.array(col) * 0.7).astype(np.uint8)
    return o
save(O + 'grid_rust.png', overlay(orig, cx, cy, (0, 255, 255)))
save(O + 'grid_dp.png', overlay(orig, dx, dy, (0, 255, 0)))

# --- baseline speckle ---
names = ['kmeans-oklab', 'median-cut', 'wu', 'spatial', 'grey-wolf', 'diff-palette']
print('\nbaseline (k=16, no-resize), speckle metrics vs pitch %.2f; purity vs dp grid' % pit)
for a in range(6):
    q = load(O + f'col_k16_a{a}.png'); lab, pal = to_labels(q)
    m = speckle_metrics(lab, pit); m.update(grid_purity(lab, dx, dy, len(pal))); m.update(fidelity(olab, srgb_to_oklab(q)))
    print(f'{names[a]:13s} K={len(pal):2d}', fmt(m))
