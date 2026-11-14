"""Research harness for the no-resize / speckle / multi-scale / tiling deep dive.

Pure numpy/scipy/skimage prototypes. Nothing here is production code; it exists so
variants can be compared with the same metrics. Run from a venv that has
numpy, pillow, scipy, scikit-image.
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure

REPO = Path(__file__).resolve().parents[3]
R = str(REPO / 'other') + '/'                                  # sample images
O = os.environ.get('PSNAP_OUT', '/tmp/psnap/out').rstrip('/') + '/'  # scratch outputs (not committed)
os.makedirs(O, exist_ok=True)

# ---------------------------------------------------------------- colour ----
def srgb_to_oklab(rgb):
    c = rgb.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack([
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s], -1).astype(np.float32)

def oklab_to_srgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    lin = np.stack([
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], -1)
    lin = np.clip(lin, 0, 1)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)

def load(path):
    return np.asarray(Image.open(path).convert('RGB'))

def save(path, rgb):
    Image.fromarray(rgb).save(path)

def to_labels(rgb):
    """Indexed view of an already-quantized image -> (labels HxW int32, palette Kx3 uint8)."""
    flat = rgb.reshape(-1, 3)
    key = (flat[:, 0].astype(np.int64) << 16) | (flat[:, 1].astype(np.int64) << 8) | flat[:, 2]
    uniq, inv = np.unique(key, return_inverse=True)
    pal = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], -1).astype(np.uint8)
    return inv.reshape(rgb.shape[:2]).astype(np.int32), pal

def nearest_labels(lab_img, pal_lab, chunk=1 << 18):
    flat = lab_img.reshape(-1, 3)
    out = np.empty(len(flat), np.int32)
    for i in range(0, len(flat), chunk):
        d = ((flat[i:i + chunk, None, :] - pal_lab[None]) ** 2).sum(-1)
        out[i:i + chunk] = d.argmin(1)
    return out.reshape(lab_img.shape[:2])

def kmeans_oklab(pts, k, weights=None, iters=30, seed=42):
    """k-means++ / Lloyd in OKLab on an (N,3) point set with optional weights."""
    rng = np.random.default_rng(seed)
    n = len(pts)
    w = np.ones(n, np.float64) if weights is None else weights.astype(np.float64)
    cent = [pts[rng.choice(n, p=w / w.sum())]]
    d2 = ((pts - cent[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 * w
        if p.sum() <= 0:
            break
        cent.append(pts[rng.choice(n, p=p / p.sum())])
        d2 = np.minimum(d2, ((pts - cent[-1]) ** 2).sum(1))
    cent = np.array(cent, np.float32)
    for _ in range(iters):
        a = ((pts[:, None, :] - cent[None]) ** 2).sum(-1).argmin(1)
        new = cent.copy()
        for j in range(len(cent)):
            m = a == j
            if m.any():
                new[j] = np.average(pts[m], axis=0, weights=w[m])
        if np.abs(new - cent).max() < 1e-5:
            cent = new
            break
        cent = new
    return cent

# ----------------------------------------------------------------- grid ----
def gray(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

def profiles_rust(rgb):
    """Port of compute_profiles(): central-difference |grad| of luma, summed per col/row."""
    g = gray(rgb.astype(np.float64))
    col = np.zeros(g.shape[1]); row = np.zeros(g.shape[0])
    col[1:-1] = np.abs(g[:, 2:] - g[:, :-2]).sum(0)
    row[1:-1] = np.abs(g[2:, :] - g[:-2, :]).sum(1)
    return col, row

def profiles_lab(lab_img):
    """Boundary profile: p[i] = strength of the boundary between index i-1 and i (OKLab dE)."""
    dx = np.sqrt(((lab_img[:, 1:] - lab_img[:, :-1]) ** 2).sum(-1))
    dy = np.sqrt(((lab_img[1:] - lab_img[:-1]) ** 2).sum(-1))
    col = np.zeros(lab_img.shape[1]); row = np.zeros(lab_img.shape[0])
    col[1:] = dx.sum(0); row[1:] = dy.sum(1)
    return col, row

def estimate_step_rust(profile, thr_mult=0.2, dist_filter=4):
    mx = profile.max()
    if mx == 0: return None
    thr = mx * thr_mult
    p = profile
    pk = [i for i in range(1, len(p) - 1) if p[i] > thr and p[i] > p[i - 1] and p[i] > p[i + 1]]
    if len(pk) < 2: return None
    clean = [pk[0]]
    for q in pk[1:]:
        if q - clean[-1] > dist_filter - 1: clean.append(q)
    if len(clean) < 2: return None
    d = sorted(np.diff(clean))
    return float(d[len(d) // 2])

def walk_rust(profile, step, limit, win_ratio=0.35, min_win=2.0, strength=0.5):
    cuts = [0]; cur = 0.0
    win = max(step * win_ratio, min_win); mean = profile.mean()
    while cur < limit:
        target = cur + step
        if target >= limit:
            cuts.append(limit); break
        s = max(int(target - win), int(cur + 1.0)); e = min(int(target + win), limit)
        if e <= s:
            cur = target; continue
        i = s + int(np.argmax(profile[s:e]))
        if profile[i] > mean * strength:
            cuts.append(i); cur = float(i)
        else:
            cuts.append(int(target)); cur = target
    return sorted(set(min(c, limit) for c in cuts) | {0, limit})

def grid_rust(quant_rgb):
    """The shipping snap-mode grid (minus the skew fallback), given an RGB-kmeans-quantized image."""
    h, w = quant_rgb.shape[:2]
    px, py = profiles_rust(quant_rgb)
    sx, sy = estimate_step_rust(px), estimate_step_rust(py)
    step = (sx + sy) / 2.0
    return walk_rust(px, step, w), walk_rust(py, step, h), step

def pitch_fft(profile, lo=3.0, hi=40.0):
    p = profile - profile.mean(); n = len(p)
    N = 1 << 18
    F = np.abs(np.fft.rfft(p * np.hanning(n), N)); fr = np.fft.rfftfreq(N)
    m = (fr > 1 / hi) & (fr < 1 / lo)
    i = np.argmax(F[m])
    return float(1 / fr[m][i]), float(F[m][i] / (np.median(F[m]) + 1e-9))

def pitch_acf(profile, lo=3, hi=40):
    """First strong autocorrelation peak, refined to sub-pixel using its harmonics."""
    p = profile - profile.mean(); n = len(p)
    f = np.fft.rfft(p, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]; ac /= ac[0]
    pk = [i for i in range(lo, hi) if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1]]
    if not pk: return None, 0.0
    best = max(pk, key=lambda i: ac[i])
    # the fundamental is the smallest lag whose peak is close to the best peak height
    fund = min(i for i in pk if ac[i] > 0.6 * ac[best])
    # refine with multiples
    est = float(fund)
    for mlt in range(2, 9):
        c = int(round(est * mlt))
        if c + 2 >= n: break
        j = c - 2 + int(np.argmax(ac[c - 2:c + 3]))
        est = j / mlt
    return est, float(ac[fund])

def cuts_dp(profile, pitch, limit, tol=0.25, lam=1.0, cut_cost=0.0):
    """Elastic comb: choose cuts 0=c0<c1<...<cn=limit maximising sum(profile[c_i]) minus a
    quadratic penalty on (c_i - c_{i-1} - pitch), with every step within pitch*(1 +- tol).
    Globally optimal (Viterbi), unlike the greedy walker."""
    lo = max(1, int(np.floor(pitch * (1 - tol)))); hi = int(np.ceil(pitch * (1 + tol)))
    pn = profile / (profile.mean() + 1e-9) - cut_cost   # cut_cost=1: a cut on an average column earns nothing (E16)
    NEG = -1e18
    best = np.full(limit + 1, NEG); prev = np.full(limit + 1, -1, np.int64)
    # free start: the first cut may be anywhere in [0, hi] (partial first cell)
    for c in range(0, min(hi, limit) + 1):
        best[c] = pn[c] if c < len(pn) else 0.0
        prev[c] = -1
    steps = np.arange(lo, hi + 1)
    pen = lam * ((steps - pitch) / pitch) ** 2 * 4.0
    for c in range(lo, limit + 1):
        src = c - steps
        ok = src >= 0
        if not ok.any(): continue
        cand = np.where(ok, best[np.clip(src, 0, limit)] - pen, NEG)
        j = int(np.argmax(cand))
        val = cand[j] + (pn[c] if c < len(pn) else 0.0)
        if val > best[c]:
            best[c] = val; prev[c] = src[j]
    # free end: last cut anywhere in [limit-hi, limit]
    end = max(range(max(0, limit - hi), limit + 1), key=lambda c: best[c])
    cuts = []
    c = end
    while c >= 0:
        cuts.append(c); c = prev[c]
    cuts = sorted(set(cuts) | {0, limit})
    # drop slivers created by the free start/end
    out = [cuts[0]]
    for c in cuts[1:]:
        if c - out[-1] < max(2, int(pitch * 0.35)) and c != limit and out[-1] != 0:
            continue
        out.append(c)
    if len(out) > 2 and out[1] - out[0] < max(2, int(pitch * 0.35)): out.pop(1)
    if len(out) > 2 and out[-1] - out[-2] < max(2, int(pitch * 0.35)): out.pop(-2)
    return out

def cuts_uniform(limit, n_cells):
    return sorted(set(int(round(i * limit / n_cells)) for i in range(n_cells + 1)))

def subdivide(cuts, f):
    """Split every cell into f sub-cells (integer refinement of an existing grid)."""
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        for i in range(f):
            out.append(a + int(round(i * (b - a) / f)))
    out.append(cuts[-1])
    return sorted(set(out))

def merge(cuts, f):
    """Keep every f-th cut (integer coarsening)."""
    out = cuts[::f]
    if out[-1] != cuts[-1]: out = out + [cuts[-1]]
    return out

# ---------------------------------------------------------------- cells ----
def cell_index(cx, cy, w, h):
    ix = np.searchsorted(np.asarray(cx), np.arange(w), side='right') - 1
    iy = np.searchsorted(np.asarray(cy), np.arange(h), side='right') - 1
    ix = np.clip(ix, 0, len(cx) - 2); iy = np.clip(iy, 0, len(cy) - 2)
    return iy[:, None] * (len(cx) - 1) + ix[None, :], len(cx) - 1, len(cy) - 1

def interior_mask(cx, cy, w, h, margin):
    """True for pixels at least `margin` px away from every cut (cell interiors)."""
    mx = np.ones(w, bool); my = np.ones(h, bool)
    for c in cx:
        mx[max(0, c - margin):min(w, c + margin)] = False
    for c in cy:
        my[max(0, c - margin):min(h, c + margin)] = False
    return my[:, None] & mx[None, :]

def cell_mode(labels, cid, ncell, k, mask=None):
    """Per-cell majority label + purity (fraction of voting pixels that agree)."""
    idx = cid * k + labels
    if mask is not None:
        idx = idx[mask]
    cnt = np.bincount(idx.ravel(), minlength=ncell * k).reshape(ncell, k)
    tot = cnt.sum(1)
    mode = cnt.argmax(1)
    purity = np.where(tot > 0, cnt.max(1) / np.maximum(tot, 1), np.nan)
    return mode, purity, tot

def cell_mean(lab_img, cid, ncell, mask=None):
    c = cid if mask is None else cid[mask]
    v = lab_img if mask is None else lab_img[mask]
    c = c.ravel(); v = v.reshape(-1, 3)
    n = np.bincount(c, minlength=ncell).astype(np.float64)
    out = np.stack([np.bincount(c, weights=v[:, i], minlength=ncell) for i in range(3)], -1)
    return (out / np.maximum(n, 1)[:, None]).astype(np.float32), n

def paint(cell_vals, cid):
    return cell_vals[cid]

# ------------------------------------------------------- label cleanups ----
def mode_filter(labels, k, radius=1, iters=1):
    lab = labels.copy()
    size = 2 * radius + 1
    for _ in range(iters):
        best = np.zeros(lab.shape, np.float32); out = lab.copy()
        for j in range(k):
            c = ndi.uniform_filter((lab == j).astype(np.float32), size, mode='nearest')
            m = c > best
            best[m] = c[m]; out[m] = j
        if (out == lab).all(): break
        lab = out
    return lab

def merge_small_components(labels, min_area, connectivity=1):
    """Dissolve connected components smaller than min_area into the nearest surviving pixel's label."""
    comp = measure.label(labels + 1, connectivity=connectivity, background=0)
    area = np.bincount(comp.ravel())
    small = area[comp] < min_area
    if not small.any(): return labels
    _, (iy, ix) = ndi.distance_transform_edt(small, return_indices=True)
    return labels[iy, ix]

def potts_icm(lab_img, pal_lab, init, lam=0.02, iters=8, diag=True):
    """Minimise sum dE(pixel, palette[label])^2 + lam * (#disagreeing neighbours), checkerboard ICM."""
    h, w, _ = lab_img.shape; k = len(pal_lab)
    unary = np.empty((h, w, k), np.float32)
    for j in range(k):
        unary[..., j] = ((lab_img - pal_lab[j]) ** 2).sum(-1)
    lab = init.copy()
    yy, xx = np.mgrid[0:h, 0:w]
    kern = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32) if diag else \
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
    if diag:
        kern = kern * np.array([[0.7071, 1, 0.7071], [1, 0, 1], [0.7071, 1, 0.7071]], np.float32)
    nsum = kern.sum()
    for it in range(iters):
        changed = 0
        for par in (0, 1):
            cost = unary.copy()
            for j in range(k):
                same = ndi.convolve((lab == j).astype(np.float32), kern, mode='nearest')
                cost[..., j] += lam * (nsum - same)
            new = cost.argmin(-1).astype(np.int32)
            m = ((yy + xx) & 1) == par
            changed += int((new[m] != lab[m]).sum())
            lab[m] = new[m]
        if changed == 0: break
    return lab

# -------------------------------------------------------------- metrics ----
def speckle_metrics(labels, pitch):
    """Spatial-coherence metrics of a label map. `pitch` = logical pixel size in px."""
    h, w = labels.shape
    comp = measure.label(labels + 1, connectivity=1, background=0)
    area = np.bincount(comp.ravel())[1:]
    cell = pitch * pitch
    px_area = np.bincount(comp.ravel())[comp]
    tv = ((labels[:, 1:] != labels[:, :-1]).sum() + (labels[1:] != labels[:-1]).sum()) / (2.0 * h * w)
    return dict(
        comps=int(len(area)),
        tiny=int((area < 0.25 * cell).sum()),          # fragments smaller than a quarter logical pixel
        sub=int((area < 0.75 * cell).sum()),           # fragments that cannot be one whole logical pixel
        tiny_px_pct=100.0 * float((px_area < 0.25 * cell).mean()),
        tv=float(tv),
    )

def fidelity(orig_lab, out_lab):
    d = np.sqrt(((orig_lab - out_lab) ** 2).sum(-1))
    return dict(dE_mean=float(d.mean()), dE_p95=float(np.percentile(d, 95)), dE_p99=float(np.percentile(d, 99)),
                bad_pct=100.0 * float((d > 0.10).mean()))

def grid_purity(labels, cx, cy, k):
    h, w = labels.shape
    cid, nx, ny = cell_index(cx, cy, w, h)
    _, pur, tot = cell_mode(labels, cid, nx * ny, k)
    return dict(purity_mean=float(np.nanmean(pur)), pure95_pct=100.0 * float(np.nanmean(pur >= 0.95)))

def fmt(d):
    return '  '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in d.items())

# ------------------------------------------------- snap-in-place family ----
def snap_cells(olab, cx, cy, margin=1):
    """Per-cell representative colour = mean of interior pixels (falls back to full cell)."""
    h, w = olab.shape[:2]
    cid, nx, ny = cell_index(cx, cy, w, h)
    inner = interior_mask(cx, cy, w, h, margin) if margin > 0 else None
    mean, n = cell_mean(olab, cid, nx * ny, inner)
    if inner is not None and (n == 0).any():
        full, _ = cell_mean(olab, cid, nx * ny)
        mean[n == 0] = full[n == 0]
    return mean, cid, nx, ny

def cell_sse(olab, cid, ncell, mean):
    d = ((olab - mean[cid]) ** 2).sum(-1)
    return np.bincount(cid.ravel(), weights=d.ravel(), minlength=ncell), np.bincount(cid.ravel(), minlength=ncell)

def adaptive_snap(olab, cx, cy, tau=0.03, f=2, margin=1):
    """Quadtree-style: native cells, but a cell is split f x f when the split explains
    at least tau^2 more per-pixel variance than the parent (i.e. it holds real sub-cell detail).
    Returns a per-pixel OKLab image of leaf colours plus the split mask (per parent cell)."""
    h, w = olab.shape[:2]
    pm, pcid, nx, ny = snap_cells(olab, cx, cy, margin)
    fx, fy = subdivide(cx, f), subdivide(cy, f)
    cm, ccid, cnx, cny = snap_cells(olab, fx, fy, min(margin, 1))
    psse, pn = cell_sse(olab, pcid, nx * ny, pm)
    csse_px = ((olab - cm[ccid]) ** 2).sum(-1)
    csse = np.bincount(pcid.ravel(), weights=csse_px.ravel(), minlength=nx * ny)
    gain = (psse - csse) / np.maximum(pn, 1)
    split = gain > tau * tau
    out = np.where(split[pcid][..., None], cm[ccid], pm[pcid])
    return out, split.reshape(ny, nx), (pm, pcid, cm, ccid)

# ----------------------------------------- core-constrained reassignment ----
def local_range(olab, size=3):
    return np.sqrt(((ndi.maximum_filter(olab, size=(size, size, 1)) - ndi.minimum_filter(olab, size=(size, size, 1))) ** 2).sum(-1))

def core_constrained(olab, pal_lab, labels, flat_thr=0.03, radius=3, core_size=3):
    """'Un-mixing': pixels in flat neighbourhoods are trusted ('core'). Every other pixel (blend ramps,
    anti-aliasing) may only take a label that a core pixel within `radius` already has, choosing the one
    closest in colour. A ramp between A and B therefore resolves to A or B, never to an unrelated
    palette entry C that happens to lie between them."""
    k = len(pal_lab)
    core = local_range(olab, core_size) < flat_thr
    best = np.full(labels.shape, np.inf, np.float32); out = labels.copy()
    size = 2 * radius + 1
    for j in range(k):
        m = core & (labels == j)
        if not m.any(): continue
        near = ndi.maximum_filter(m, size=size)
        d = ((olab - pal_lab[j]) ** 2).sum(-1)
        upd = near & (d < best)
        best[upd] = d[upd]; out[upd] = j
    out[core] = labels[core]
    return out, core

# ------------------------------------------------ colour-aware sieve (E16) ----
def sieve_colour(labels, olab, pal_lab, min_area, max_pass=6):
    """Dissolve components smaller than min_area into the ADJACENT label whose palette colour is
    closest to the component's own mean source colour. Sequential, smallest first, so two small
    neighbours cannot swap. Replaces merge_small_components, which hands every small fragment to
    the nearest surviving pixel and so erases any region that is a fine mix of two similar palette
    colours (the wolf's outline)."""
    lab = labels.copy(); h, w = lab.shape
    for _ in range(max_pass):
        comp = measure.label(lab + 1, connectivity=1, background=0); area = np.bincount(comp.ravel())
        small = np.flatnonzero(area < min_area); small = small[small > 0]
        if len(small) == 0: break
        objs = ndi.find_objects(comp); changed = 0
        for c in small[np.argsort(area[small], kind='stable')]:
            sl = objs[c - 1]; y0, y1 = max(sl[0].start - 1, 0), min(sl[0].stop + 1, h); x0, x1 = max(sl[1].start - 1, 0), min(sl[1].stop + 1, w)
            m = comp[y0:y1, x0:x1] == c; sub = lab[y0:y1, x0:x1]
            own = sub[m][0]
            if not (sub[m] == own).all(): continue          # already repainted this pass
            ring = ndi.binary_dilation(m) & ~m; nb = np.unique(sub[ring]); nb = nb[nb != own]
            if len(nb) == 0: continue
            mean = olab[y0:y1, x0:x1][m].mean(0)
            sub[m] = nb[((pal_lab[nb] - mean) ** 2).sum(-1).argmin()]; changed += 1
        if not changed: break
    return lab
