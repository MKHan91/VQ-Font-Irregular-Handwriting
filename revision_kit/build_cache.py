"""Builds 64x64 uint8 caches for the jamo-recognition probe.
Normalisation (identical for every source): grey -> composite on white if RGBA -> Otsu binarisation ->
crop to inked bounding box -> longest side resized to 56 px (INTER_AREA) -> centred on a 64x64 white canvas.
"""
import sys, os, glob, json, numpy as np, cv2
from multiprocessing import Pool


def norm(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 3 and im.shape[2] == 4:
        a = im[:, :, 3:4].astype(np.float32) / 255.0
        im = (im[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(b < 128)
    if len(ys) < 5:
        return None  # blank
    b = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = b.shape; s = 56.0 / max(h, w)
    b = cv2.resize(b, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.full((64, 64), 255, np.uint8)
    y0, x0 = (64 - b.shape[0]) // 2, (64 - b.shape[1]) // 2
    out[y0:y0 + b.shape[0], x0:x0 + b.shape[1]] = b
    return out


def label(ch):
    c = ord(ch) - 0xAC00
    return (c // 588, (c % 588) // 28, c % 28) if 0 <= c < 11172 else None


def job(p):
    ch = os.path.splitext(os.path.basename(p))[0]
    lab = label(ch) if len(ch) == 1 else None
    if lab is None:
        return None
    x = norm(p)
    return None if x is None else (x, lab, ch)


def build(dirs, out):
    files = []
    for d in dirs:
        files += [(f, os.path.basename(d.rstrip('/'))) for f in sorted(glob.glob(os.path.join(d, '*.png')))]
    with Pool(2) as P:
        res = P.map(job, [f for f, _ in files], chunksize=256)
    keep = [(r, src) for r, (f, src) in zip(res, files) if r is not None]
    X = np.stack([r[0] for r, _ in keep]); Y = np.array([r[1] for r, _ in keep], np.int16)
    meta = {'chars': [r[2] for r, _ in keep], 'source': [s for _, s in keep]}
    np.save(out + '_X.npy', X); np.save(out + '_Y.npy', Y)
    json.dump(meta, open(out + '_meta.json', 'w', encoding='utf-8'), ensure_ascii=False)
    print(out, X.shape, 'skipped', len(files) - len(keep))


if __name__ == '__main__':
    out = sys.argv[1]; dirs = sys.argv[2:]
    build(dirs, out)
