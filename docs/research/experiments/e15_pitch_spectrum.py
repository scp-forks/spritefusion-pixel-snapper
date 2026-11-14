"""E15: pitch spectrum. The autocorrelation pitch picker (lab.pitch_acf) failed on the second batch of
images, so score every candidate pitch directly.
For each nominal pitch p fit a nearly rigid elastic grid (steps within +-8%, about +-1 px), take its
effective pitch p_eff = sqrt(pixels / cells) and compare its residual with what unaligned even grids
achieve at the same p_eff (upper envelope of an even-grid sweep). score = 1 - fitted / unaligned.
A real lattice peaks at its pitch and at integer fractions of it; the pitch is the largest p_eff that
still peaks. Usage: e15_pitch_spectrum.py [image names...]"""
import sys, numpy as np
from lab import *
def residual(olab, cx, cy):
    h, w = olab.shape[:2]; cid, nx, ny = cell_index(cx, cy, w, h); m, _ = cell_mean(olab, cid, nx * ny)
    return float(np.sqrt(((olab - m[cid]) ** 2).sum(-1)).mean()), float(np.sqrt(w * h / (nx * ny)))
def unaligned_envelope(olab, lo=1.6, hi=40.0, n=70):
    h, w = olab.shape[:2]; pts = []
    for s in np.geomspace(lo, hi, n):
        r, pe = residual(olab, cuts_uniform(w, max(1, int(round(w / s)))), cuts_uniform(h, max(1, int(round(h / s)))))
        pts.append((pe, r))
    pts = np.array(sorted(pts)); env = ndi.maximum_filter1d(pts[:, 1], 7)   # even grids that happen to align dip below; keep the envelope
    return lambda pe: float(np.exp(np.interp(np.log(pe), np.log(pts[:, 0]), np.log(env))))
def spectrum(olab, ps, tol=0.08):
    h, w = olab.shape[:2]; px, py = profiles_lab(olab); env = unaligned_envelope(olab); out = []
    for p in ps:
        fit, pe = residual(olab, cuts_dp(px, p, w, tol), cuts_dp(py, p, h, tol))
        out.append((pe, fit, 1 - fit / env(pe), p))
    return np.array(sorted(out))
def summarise(sp, bins=np.arange(2.0, 24.01, 0.5)):
    """best score per half-pixel bin of effective pitch"""
    return [(b, sp[(sp[:, 0] >= b) & (sp[:, 0] < b + 0.5), 2].max()) for b in bins if ((sp[:, 0] >= b) & (sp[:, 0] < b + 0.5)).any()]
if __name__ == '__main__':
    names = sys.argv[1:] or ['colliseum', 'pixel-art-sample-1024x1024', 'wolf', 'ogtrail', 'isometric-city-with-tiny-cars', 'cat-1024x1024']
    ps = np.arange(2.0, 24.01, 0.125)
    for name in names:
        olab = srgb_to_oklab(load(R + name + '.png')); sp = spectrum(olab, ps)
        np.save(O + f'{name}_pitch_spectrum.npy', sp)
        i = sp[:, 2].argmax()
        print(f'{name}: best score {sp[i,2]:.3f} at effective pitch {sp[i,0]:.2f} (residual {sp[i,1]:.4f})')
        print('   score x100 by effective pitch: ' + ' '.join(f'{b:g}:{int(round(100*s))}' for b, s in summarise(sp)), flush=True)
