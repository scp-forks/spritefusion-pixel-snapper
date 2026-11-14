"""E11: can a detail-preserving reducer (DPID-style weighting, Weber et al. 2016) rescue the native-grid
candidate? Per cell: weight each source pixel by its distance from the smoothed low-res image, ^lam."""
import numpy as np
from lab import *
name = 'colliseum'; k = 16
orig = load(R + name + '.png'); olab = srgb_to_oklab(orig); h, w = orig.shape[:2]
px, py = profiles_lab(olab); pit = (pitch_acf(px)[0] + pitch_acf(py)[0]) / 2
dx = cuts_dp(px, pit, w); dy = cuts_dp(py, pit, h)
base, pal = to_labels(load(O + 'col_k16_a0.png')); pal_lab = srgb_to_oklab(pal)
cid, nx, ny = cell_index(dx, dy, w, h); n = nx * ny
A, _ = cell_mean(olab, cid, n)
kern = np.array([1, 2, 1], np.float32) / 4
B = ndi.convolve1d(ndi.convolve1d(A.reshape(ny, nx, 3), kern, axis=0, mode='nearest'), kern, axis=1, mode='nearest').reshape(n, 3)
def report(tag, cell_lab):
    lab = cell_lab[cid]; m = speckle_metrics(lab, pit); m.update(fidelity(olab, pal_lab[lab]))
    save(O + f'{name}_{tag}.png', oklab_to_srgb(pal_lab[lab])); print(f'{tag:28s}', fmt(m))
    return cell_lab
ref = report('r0_native_mean', nearest_labels(A[None], pal_lab)[0])
dist = np.sqrt(((olab - B[cid]) ** 2).sum(-1))
for lam in (0.5, 1.0, 2.0):
    wgt = (dist ** lam).ravel() + 1e-9
    s = np.stack([np.bincount(cid.ravel(), weights=wgt * olab[..., c].ravel(), minlength=n) for c in range(3)], -1) / np.bincount(cid.ravel(), weights=wgt, minlength=n)[:, None]
    cl = report(f'r1_native_dpid_mean_lam{lam}', nearest_labels(s.astype(np.float32)[None], pal_lab)[0])
    # label-vote variant: DPID weights accumulated per palette label, weighted mode
    cnt = np.bincount((cid * k + base).ravel(), weights=wgt, minlength=n * k).reshape(n, k)
    cl2 = report(f'r2_native_dpid_vote_lam{lam}', cnt.argmax(1).astype(np.int32))
    print(f'     cells changed vs plain mean: dpid-mean {100*(cl != ref).mean():.1f}%  dpid-vote {100*(cl2 != ref).mean():.1f}%')
