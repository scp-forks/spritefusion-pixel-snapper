"""E16: the second batch of samples (wolf, ogtrail, isometric city) through the same pipeline.
Pitches come from E15 (largest effective pitch whose score is within 85% of the best), because
lab.pitch_acf is wrong on two of the three. Fragment counts use absolute sizes here (< 4 px, < 16 px):
at a 2 px pitch "a quarter of a logical pixel" is a single pixel.
Two prototype fixes the batch forced: the sieve is colour-aware (lab.sieve_colour) and the grid fit
charges one unit per cut (cuts_dp cut_cost=1), which stops it over-segmenting soft lattices.
Usage: e16_second_batch.py [short names...]   (needs <short>_k16_a0.png from the CLI in $PSNAP_OUT)"""
import sys, time, numpy as np
from lab import *
SAMPLES = {  # short -> (file, ladder of nominal pitches; first = finest lattice E15 supports)
    'wolf':    ('wolf', (5.0, 2.5, 10.0)),
    'ogtrail': ('ogtrail', (2.0, 3.0, 4.0, 5.3)),
    'city':    ('isometric-city-with-tiny-cars', (2.0, 3.0, 4.0)),
}
def specks(labels):
    comp = measure.label(labels + 1, connectivity=1, background=0); area = np.bincount(comp.ravel())[1:]
    pa = np.bincount(comp.ravel())[comp]
    return dict(comps=int(len(area)), lt4=int((area < 4).sum()), lt16=int((area < 16).sum()), px_in_lt16_pct=100.0 * float((pa < 16).mean()))
def cliff_pitch(sp, frac=0.85):
    return float(sp[sp[:, 2] >= frac * sp[:, 2].max(), 0].max()), float(sp[:, 2].max())
def dpid_vote(olab, labels, cid, n, k, lam=1.0):
    A, _ = cell_mean(olab, cid, n); ny = cid[-1, 0] // (cid[0, -1] + 1) + 1; nx = cid[0, -1] + 1
    kern = np.array([1, 2, 1], np.float32) / 4
    B = ndi.convolve1d(ndi.convolve1d(A.reshape(ny, nx, 3), kern, axis=0, mode='nearest'), kern, axis=1, mode='nearest').reshape(n, 3)
    wgt = (np.sqrt(((olab - B[cid]) ** 2).sum(-1)) ** lam).ravel() + 1e-9
    return np.bincount((cid * k + labels).ravel(), weights=wgt, minlength=n * k).reshape(n, k).argmax(1).astype(np.int32)
for short in (sys.argv[1:] or list(SAMPLES)):
    name, ladder = SAMPLES[short]
    orig = load(R + name + '.png'); olab = srgb_to_oklab(orig); h, w = orig.shape[:2]
    base, pal = to_labels(load(O + f'{short}_k16_a0.png')); k = len(pal); pal_lab = srgb_to_oklab(pal)
    px, py = profiles_lab(olab)
    try: cp = cliff_pitch(np.load(O + f'{name}_pitch_spectrum.npy'))
    except FileNotFoundError: cp = (float('nan'), float('nan'))
    print(f'\n=== {short} {w}x{h}  palette={k}  E15 pitch={cp[0]:.2f} (best score {cp[1]:.2f})  pitch_acf x/y={pitch_acf(px)[0]}/{pitch_acf(py)[0]}')
    def report(tag, lab, extra=''):
        m = specks(lab); m.update(fidelity(olab, pal_lab[lab])); save(O + f'{short}_{tag}.png', oklab_to_srgb(pal_lab[lab]))
        print(f'  {tag:26s}', fmt(m), extra, flush=True)
    report('b0_baseline', base)
    p0 = ladder[0]; rad = int(np.clip(round(p0 / 2), 1, 3))
    t = time.time(); u, core = core_constrained(olab, pal_lab, base, 0.03, rad)
    report('b1_unmix', u, f'core={100*core.mean():.0f}% radius={rad}')
    for a in (8, 16):
        report(f'b2_unmix+oldsieve{a}', merge_small_components(u, a))                 # round-one sieve: nearest survivor
    for a in (4, 8, 16):
        t = time.time(); us = sieve_colour(u, olab, pal_lab, a); report(f'b3_unmix+sieve{a}', us, f'sec={time.time()-t:.1f}')
        if a == 8: clean = us
    report('b4_sieve8_only', sieve_colour(base, olab, pal_lab, 8))
    for p in ladder:
        cx, cy = cuts_dp(px, p, w, cut_cost=1.0), cuts_dp(py, p, h, cut_cost=1.0); cid, nx, ny = cell_index(cx, cy, w, h); n = nx * ny
        mean, _ = cell_mean(olab, cid, n); res = float(np.sqrt(((olab - mean[cid]) ** 2).sum(-1)).mean())
        ux, uy = cuts_uniform(w, nx), cuts_uniform(h, ny); ucid, _, _ = cell_index(ux, uy, w, h); um, _ = cell_mean(olab, ucid, n)
        ures = float(np.sqrt(((olab - um[ucid]) ** 2).sum(-1)).mean())
        g = f'grid={nx}x{ny} residual fitted={res:.4f} even={ures:.4f}'
        mode, _, _ = cell_mode(clean, cid, n, k); report(f'c_p{p}_majority', mode[cid], g)
        dv = dpid_vote(olab, base, cid, n, k); report(f'c_p{p}_dpid', dv[cid])
        save(O + f'{short}_c_p{p}_majority_small.png', oklab_to_srgb(pal_lab[mode].reshape(ny, nx, 3)))
        save(O + f'{short}_c_p{p}_dpid_small.png', oklab_to_srgb(pal_lab[dv].reshape(ny, nx, 3)))
