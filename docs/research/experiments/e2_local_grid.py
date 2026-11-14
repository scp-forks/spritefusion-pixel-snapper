"""E2: is the logical grid one global separable lattice, or does pitch/phase vary across the image?
Complex demodulation of local boundary profiles: per tile, phase (px) and best local pitch."""
import sys, numpy as np
from lab import *
path = sys.argv[1] if len(sys.argv) > 1 else R + 'colliseum.png'
orig = load(path); olab = srgb_to_oklab(orig); h, w = orig.shape[:2]
dx = np.sqrt(((olab[:, 1:] - olab[:, :-1]) ** 2).sum(-1))  # (h, w-1) boundary between x and x+1
dy = np.sqrt(((olab[1:] - olab[:-1]) ** 2).sum(-1))
px, py = profiles_lab(olab)
P0x = pitch_acf(px)[0]; P0y = pitch_acf(py)[0]
print('global acf pitch x %.3f y %.3f' % (P0x, P0y))
T = int(sys.argv[2]) if len(sys.argv) > 2 else 160
def local_pitch(p, lo, hi):
    p = p - p.mean(); n = len(p); N = 1 << 14
    F = np.abs(np.fft.rfft(p * np.hanning(n), N)); fr = np.fft.rfftfreq(N)
    m = (fr > 1 / hi) & (fr < 1 / lo); i = np.argmax(F[m])
    return 1 / fr[m][i], F[m][i] / (F[m].mean() + 1e-9)
def phase(p, P, x0):
    n = len(p); x = np.arange(n) + x0
    z = (p * np.hanning(n) * np.exp(-2j * np.pi * x / P)).sum()
    return (np.angle(z) / (2 * np.pi) * P) % P, abs(z) / ((p * np.hanning(n)).sum() + 1e-9)
ys = list(range(0, h - T + 1, T)); xs = list(range(0, w - T + 1, T))
for axis, d, P0 in (('x (column cuts)', dx, P0x), ('y (row cuts)', dy, P0y)):
    print(f'\n== {axis}: per-tile local pitch (FFT peak in [0.6P,1.6P]) ==')
    pit = np.zeros((len(ys), len(xs))); ph = np.zeros_like(pit); coh = np.zeros_like(pit)
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            t = d[y0:y0 + T, x0:x0 + T]
            p = t.sum(0) if axis[0] == 'x' else t.sum(1)
            pit[i, j], _ = local_pitch(p, 0.6 * P0, 1.6 * P0)
            ph[i, j], coh[i, j] = phase(p, P0, x0 if axis[0] == 'x' else y0)
    np.set_printoptions(precision=1, suppress=True, linewidth=220)
    print(pit)
    print(f'-- phase in px at global pitch {P0:.2f} (a single global lattice => constant along the cut direction) --')
    print(ph)
    print('-- coherence (0..1) --'); np.set_printoptions(precision=2, suppress=True, linewidth=220); print(coh)
    print('pitch median %.2f  IQR %.2f..%.2f' % (np.median(pit), *np.percentile(pit, [25, 75])))
