#!/usr/bin/env python3
"""
rk_eval_paired.py - PSNR / SSIM / LPIPS of generated glyphs against the writer's references.

PSNR and SSIM reproduce scripts/coverage_quality_analysis.py exactly (reference read with
PIL .convert("L"); generated image resized to the reference size with LANCZOS; PSNR on 0-255;
skimage SSIM with data_range=255), so values are comparable with the 19.83 dB / 0.852 reported
in the submitted manuscript. LPIPS (AlexNet, lpips 0.1) is computed on both images resized to
128 x 128 (bicubic), replicated to three channels and scaled to [-1, 1].

    python rk_eval_paired.py --gen <dir with generated <char>.png> --ref <reference dir> \
        --chars <txt file with one character per line, optional> --out result.csv [--tag NAME]

Prints mean, SD, SE, 95% CI and median; writes one row per glyph.
"""
import argparse, csv, math, statistics as st
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

_lp = None


def load_gray(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.uint8)


def to_size(b, shape):
    return np.asarray(Image.fromarray(b).resize(shape[::-1], Image.LANCZOS), dtype=np.uint8) if b.shape != shape else b


def psnr(a, b):
    b = to_size(b, a.shape)
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse <= 1e-10 else float(20 * math.log10(255.0) - 10 * math.log10(mse))


def ssim(a, b):
    return float(structural_similarity(a, to_size(b, a.shape), data_range=255))


def lpips_d(a, b):
    global _lp
    try:
        import torch, lpips
    except ImportError:
        return float("nan")
    if _lp is None:
        _lp = lpips.LPIPS(net="alex", verbose=False).eval()
    def t(x):
        x = np.asarray(Image.fromarray(x).resize((128, 128), Image.BICUBIC), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(x)[None, None].repeat(1, 3, 1, 1)
    with torch.no_grad():
        return float(_lp(t(a), t(b)).item())


def summarize(v):
    v = [x for x in v if not math.isnan(x)]
    if not v:
        return {}
    m = st.mean(v); sd = st.stdev(v) if len(v) > 1 else 0.0; se = sd / math.sqrt(len(v))
    return {"n": len(v), "mean": m, "sd": sd, "se": se, "ci95": (m - 1.96 * se, m + 1.96 * se), "median": st.median(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--chars"); ap.add_argument("--out", required=True); ap.add_argument("--tag", default="")
    ap.add_argument("--no-lpips", action="store_true")
    a = ap.parse_args()
    gen, ref = Path(a.gen), Path(a.ref)
    chars = ([c.strip() for c in Path(a.chars).read_text(encoding="utf-8").splitlines() if c.strip()]
             if a.chars else sorted(p.stem for p in ref.glob("*.png")))
    rows, missing = [], []
    for ch in chars:
        g, r = gen / f"{ch}.png", ref / f"{ch}.png"
        if not g.exists():
            missing.append(ch); continue
        A, B = load_gray(r), load_gray(g)
        rows.append([a.tag, ch, psnr(A, B), ssim(A, B), float("nan") if a.no_lpips else lpips_d(A, B)])
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["tag", "char", "psnr", "ssim", "lpips_alex"]); w.writerows(rows)
    for i, name in ((2, "PSNR"), (3, "SSIM"), (4, "LPIPS")):
        s = summarize([r[i] for r in rows])
        if s:
            print(f"{a.tag} {name}: n={s['n']} mean={s['mean']:.4f} sd={s['sd']:.4f} se={s['se']:.4f} "
                  f"ci95=[{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}] median={s['median']:.4f}")
    if missing:
        print(f"[warn] {len(missing)} generated images missing: {''.join(missing[:30])}")


if __name__ == "__main__":
    main()
