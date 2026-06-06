#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coverage_quality_analysis.py
============================

생성된 11,172개 한글 음절에 대해

    "참조 78자 자모(초·중·종) 커버리지"  ×  "이미지 품질 지표"

의 관계를 정량 분석한다.

▶ 자모 커버리지 (gt-free)
    한 음절을 (초성, 중성, 종성) 으로 분해하고, 78자 참조 풀에서
    각각이 한 번이라도 등장한 횟수를 센다. "결손 자모 수(0~3)" 가
    학습 신호 부족도의 단순 지표.

▶ 품질 지표
    GT-free (모든 11172자 적용):
        - ink_ratio       : 검은 픽셀 비율 (잉크 양)
        - bbox_fill       : 글자 bounding box / 이미지 면적
        - n_components    : 8-연결 컴포넌트 수 (획 끊김 proxy, scipy 있을 때)
        - edge_density    : sobel 에지 비율 (획 거칠기 proxy)
    GT-paired (참조 78자 한정):
        - psnr / ssim     : 참조 vs 생성

▶ 산출물
    <out>/per_char_metrics.csv          : 음절별 raw metrics
    <out>/summary_by_coverage.csv       : 커버리지 그룹별 요약 통계
    <out>/plots/coverage_hist.png       : 결손 자모 수 분포
    <out>/plots/<metric>_by_coverage.png: 커버리지별 박스플롯
    <out>/plots/psnr_by_coverage.png    : (GT 있을 때) PSNR 박스플롯
    <out>/plots/scatter_psnr_ink.png    : PSNR vs ink_ratio 산점도
    <out>/INDEX.md                      : 빠른 인덱스
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 선택 의존성
# ---------------------------------------------------------------------------
try:  # 8-연결 컴포넌트
    from scipy import ndimage as _ndi  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _ndi = None
    _HAS_SCIPY = False

try:  # SSIM
    from skimage.metrics import structural_similarity as _ssim  # type: ignore
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _ssim = None
    _HAS_SKIMAGE = False

# ---------------------------------------------------------------------------
# 한글 자모 분해 (외부 라이브러리 없이 unicode 산술)
# ---------------------------------------------------------------------------
_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3
_CHO_COUNT, _JUNG_COUNT, _JONG_COUNT = 19, 21, 28


def decompose_hangul(ch: str) -> Optional[Tuple[int, int, int]]:
    """완성형 음절을 (초성idx, 중성idx, 종성idx) 로 분해. 종성 없으면 0."""
    if not ch:
        return None
    code = ord(ch[0])
    if code < _HANGUL_BASE or code > _HANGUL_LAST:
        return None
    offset = code - _HANGUL_BASE
    cho = offset // (_JUNG_COUNT * _JONG_COUNT)
    jung = (offset % (_JUNG_COUNT * _JONG_COUNT)) // _JONG_COUNT
    jong = offset % _JONG_COUNT
    return cho, jung, jong


def build_jamo_pool(ref_chars: List[str]) -> Tuple[Counter, Counter, Counter]:
    cho_c, jung_c, jong_c = Counter(), Counter(), Counter()
    for ch in ref_chars:
        dec = decompose_hangul(ch)
        if dec is None:
            continue
        cho, jung, jong = dec
        cho_c[cho] += 1
        jung_c[jung] += 1
        if jong != 0:
            jong_c[jong] += 1
    return cho_c, jung_c, jong_c


def missing_components(
    ch: str,
    cho_pool: Counter,
    jung_pool: Counter,
    jong_pool: Counter,
) -> Tuple[int, int, int, int]:
    """
    Returns (missing_cho, missing_jung, missing_jong, total_missing)
    종성이 0(없음)인 음절은 종성을 채워야 할 컴포넌트로 세지 않는다.
    """
    dec = decompose_hangul(ch)
    if dec is None:
        return 0, 0, 0, 0
    cho, jung, jong = dec
    mc = int(cho_pool[cho] == 0)
    mj = int(jung_pool[jung] == 0)
    mn = int(jong != 0 and jong_pool[jong] == 0)
    return mc, mj, mn, mc + mj + mn


# ---------------------------------------------------------------------------
# 이미지 품질 지표
# ---------------------------------------------------------------------------
def load_gray(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    try:
        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception:
        return None


def _binarize(arr: np.ndarray, thr: int = 200) -> np.ndarray:
    return arr < thr  # True = 잉크(검은 픽셀)


def ink_ratio(arr: np.ndarray) -> float:
    return float(_binarize(arr).mean())


def bbox_fill(arr: np.ndarray) -> float:
    mask = _binarize(arr)
    if not mask.any():
        return 0.0
    ys, xs = np.where(mask)
    bb = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return float(bb / mask.size)


def num_components(arr: np.ndarray) -> int:
    mask = _binarize(arr)
    if not mask.any():
        return 0
    if _HAS_SCIPY:
        # 8-연결
        structure = np.ones((3, 3), dtype=bool)
        _, n = _ndi.label(mask, structure=structure)
        return int(n)
    # Fallback: numpy-only flood fill (O(N), 적당히 빠름)
    visited = np.zeros_like(mask)
    H, W = mask.shape
    count = 0
    stack: List[Tuple[int, int]] = []
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not visited[i, j]:
                count += 1
                stack.append((i, j))
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= H or x < 0 or x >= W:
                        continue
                    if not mask[y, x] or visited[y, x]:
                        continue
                    visited[y, x] = True
                    stack.extend([
                        (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1),
                        (y - 1, x - 1), (y - 1, x + 1),
                        (y + 1, x - 1), (y + 1, x + 1),
                    ])
    return count


def edge_density(arr: np.ndarray) -> float:
    """간이 sobel — 외부 의존성 없이 np.gradient 로 근사."""
    gy, gx = np.gradient(arr.astype(np.float32))
    mag = np.hypot(gx, gy)
    return float((mag > 20).mean())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        # 크기 다르면 리사이즈
        b_img = Image.fromarray(b).resize(a.shape[::-1], Image.LANCZOS)
        b = np.asarray(b_img, dtype=np.uint8)
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse <= 1e-10:
        return 99.0
    return float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))


def ssim_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    if not _HAS_SKIMAGE:
        return float("nan")
    if a.shape != b.shape:
        b_img = Image.fromarray(b).resize(a.shape[::-1], Image.LANCZOS)
        b = np.asarray(b_img, dtype=np.uint8)
    try:
        return float(_ssim(a, b, data_range=255))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# matplotlib 한글 폰트
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _setup_korean_font() -> None:
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        str(_REPO_ROOT / "datasets/train_font_ttf/NanumBarunpenR.ttf"),
    ]
    for p in candidates:
        if Path(p).is_file():
            fm.fontManager.addfont(p)
            plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 메인 분석 로직
# ---------------------------------------------------------------------------
def analyze(args: argparse.Namespace) -> None:
    _setup_korean_font()

    ref_dir = Path(args.ref_dir)
    gen_dir = Path(args.gen_dir)
    out_dir = Path(args.out)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    ref_chars = sorted([p.stem for p in ref_dir.glob("*.png") if p.stem])
    ref_set = set(ref_chars)
    cho_pool, jung_pool, jong_pool = build_jamo_pool(ref_chars)
    print(f"[INFO] 참조 글자 수: {len(ref_chars)}")
    print(f"[INFO]   고유 초성 {len(cho_pool)}/{_CHO_COUNT}, "
          f"중성 {len(jung_pool)}/{_JUNG_COUNT}, "
          f"종성 {len(jong_pool)}/{_JONG_COUNT - 1}")

    gen_files = sorted(gen_dir.glob("*.png"))
    if args.limit > 0:
        gen_files = gen_files[: args.limit]
    print(f"[INFO] 분석 대상 생성 이미지: {len(gen_files)}")

    # ---------------- per-char metrics ----------------
    rows: List[dict] = []
    for idx, fp in enumerate(gen_files, start=1):
        ch = fp.stem
        gen = load_gray(fp)
        if gen is None:
            continue
        mc, mj, mn, tot = missing_components(ch, cho_pool, jung_pool, jong_pool)
        row = {
            "char": ch,
            "unicode": f"U+{ord(ch[0]):04X}" if ch else "",
            "missing_cho": mc,
            "missing_jung": mj,
            "missing_jong": mn,
            "missing_total": tot,
            "in_ref78": int(ch in ref_set),
            "ink_ratio": ink_ratio(gen),
            "bbox_fill": bbox_fill(gen),
            "edge_density": edge_density(gen),
            "n_components": num_components(gen) if args.with_components else -1,
            "psnr": float("nan"),
            "ssim": float("nan"),
        }
        if ch in ref_set:
            gt = load_gray(ref_dir / f"{ch}.png")
            if gt is not None:
                row["psnr"] = psnr(gt, gen)
                row["ssim"] = ssim_or_nan(gt, gen)
        rows.append(row)

        if idx % 1000 == 0:
            print(f"  ...{idx}/{len(gen_files)}")

    # ---------------- write CSV ----------------
    cols = ["char", "unicode", "missing_cho", "missing_jung", "missing_jong",
            "missing_total", "in_ref78", "ink_ratio", "bbox_fill",
            "edge_density", "n_components", "psnr", "ssim"]
    per_char_csv = out_dir / "per_char_metrics.csv"
    _write_csv(per_char_csv, rows, cols)
    print(f"[SAVE] {per_char_csv}")

    # ---------------- summary by coverage ----------------
    summary = _summarize_by_coverage(rows)
    summary_csv = out_dir / "summary_by_coverage.csv"
    _write_csv(summary_csv, summary,
               ["missing_total", "count",
                "ink_ratio_mean", "ink_ratio_std",
                "bbox_fill_mean", "bbox_fill_std",
                "edge_density_mean", "edge_density_std",
                "n_components_mean", "n_components_std",
                "psnr_mean", "psnr_std", "psnr_count",
                "ssim_mean", "ssim_std", "ssim_count"])
    print(f"[SAVE] {summary_csv}")

    # ---------------- plots ----------------
    _plot_coverage_hist(rows, out_dir / "plots/coverage_hist.png")
    _plot_metric_by_coverage(rows, "ink_ratio",
                             out_dir / "plots/ink_ratio_by_coverage.png",
                             title="잉크 비율 vs 결손 자모 수")
    _plot_metric_by_coverage(rows, "bbox_fill",
                             out_dir / "plots/bbox_fill_by_coverage.png",
                             title="BBox 채움 비율 vs 결손 자모 수")
    _plot_metric_by_coverage(rows, "edge_density",
                             out_dir / "plots/edge_density_by_coverage.png",
                             title="에지 밀도 vs 결손 자모 수")
    if args.with_components:
        _plot_metric_by_coverage(rows, "n_components",
                                 out_dir / "plots/n_components_by_coverage.png",
                                 title="연결 컴포넌트 수 vs 결손 자모 수")

    psnr_rows = [r for r in rows if not math.isnan(r["psnr"])]
    if psnr_rows:
        _plot_metric_by_coverage(psnr_rows, "psnr",
                                 out_dir / "plots/psnr_by_coverage.png",
                                 title="PSNR vs 결손 자모 수 (참조 78자 한정)")
        _plot_scatter(psnr_rows, "ink_ratio", "psnr",
                      out_dir / "plots/scatter_psnr_ink.png",
                      title="PSNR vs 잉크 비율 (참조 78자)")

    _write_index(out_dir, rows, ref_chars, args)
    print(f"[DONE] {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# helpers: csv / summary / plots / index
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: List[dict], cols: List[str]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _summarize_by_coverage(rows: List[dict]) -> List[dict]:
    by: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by[r["missing_total"]].append(r)
    out: List[dict] = []
    for k in sorted(by.keys()):
        grp = by[k]
        out.append({
            "missing_total": k,
            "count": len(grp),
            "ink_ratio_mean": float(np.mean([r["ink_ratio"] for r in grp])),
            "ink_ratio_std": float(np.std([r["ink_ratio"] for r in grp])),
            "bbox_fill_mean": float(np.mean([r["bbox_fill"] for r in grp])),
            "bbox_fill_std": float(np.std([r["bbox_fill"] for r in grp])),
            "edge_density_mean": float(np.mean([r["edge_density"] for r in grp])),
            "edge_density_std": float(np.std([r["edge_density"] for r in grp])),
            "n_components_mean": float(np.mean([r["n_components"] for r in grp])),
            "n_components_std": float(np.std([r["n_components"] for r in grp])),
            "psnr_mean": _safe_mean([r["psnr"] for r in grp]),
            "psnr_std": _safe_std([r["psnr"] for r in grp]),
            "psnr_count": sum(1 for r in grp if not math.isnan(r["psnr"])),
            "ssim_mean": _safe_mean([r["ssim"] for r in grp]),
            "ssim_std": _safe_std([r["ssim"] for r in grp]),
            "ssim_count": sum(1 for r in grp if not math.isnan(r["ssim"])),
        })
    return out


def _safe_mean(vs: List[float]) -> float:
    arr = [v for v in vs if not math.isnan(v)]
    return float(np.mean(arr)) if arr else float("nan")


def _safe_std(vs: List[float]) -> float:
    arr = [v for v in vs if not math.isnan(v)]
    return float(np.std(arr)) if arr else float("nan")


def _plot_coverage_hist(rows: List[dict], out: Path) -> None:
    counts = Counter(r["missing_total"] for r in rows)
    keys = sorted(counts.keys())
    vals = [counts[k] for k in keys]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar(keys, vals, color="#4C72B0", edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("결손 자모 수 (참조 78자에 없는 초/중/종)")
    ax.set_ylabel("음절 수")
    ax.set_title("자모 커버리지 분포")
    ax.set_xticks(keys)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)


def _plot_metric_by_coverage(rows: List[dict], metric: str, out: Path,
                             title: str) -> None:
    by: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        v = r[metric]
        if isinstance(v, float) and math.isnan(v):
            continue
        if metric == "n_components" and v < 0:
            continue
        by[r["missing_total"]].append(float(v))
    keys = sorted(by.keys())
    data = [by[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    bp = ax.boxplot(
        data, positions=keys, widths=0.6, patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#222"),
        boxprops=dict(facecolor="#C7D7EA", edgecolor="#4C72B0"),
        whiskerprops=dict(color="#4C72B0"),
        capprops=dict(color="#4C72B0"),
    )
    for k in keys:
        ax.text(k, max(by[k]) if by[k] else 0, f"n={len(by[k])}",
                ha="center", va="bottom", fontsize=8, color="#666")
    ax.set_xticks(keys)
    ax.set_xlabel("결손 자모 수")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)


def _plot_scatter(rows: List[dict], x: str, y: str, out: Path, title: str) -> None:
    xs = [r[x] for r in rows]
    ys = [r[y] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter(xs, ys, s=14, alpha=0.55, color="#4C72B0", edgecolor="white")
    if len(xs) >= 2:
        try:
            corr = float(np.corrcoef(xs, ys)[0, 1])
            ax.text(0.02, 0.97, f"Pearson r = {corr:+.3f}",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=9, color="#333",
                    bbox=dict(facecolor="white", edgecolor="#ddd", alpha=0.8))
        except Exception:
            pass
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)


def _write_index(out_dir: Path, rows: List[dict],
                 ref_chars: List[str], args: argparse.Namespace) -> None:
    n_total = len(rows)
    n_gt = sum(1 for r in rows if not math.isnan(r["psnr"]))
    lines = [
        "# Coverage × Quality Analysis",
        "",
        f"- 생성 이미지 수: **{n_total:,}**",
        f"- 참조(GT 보유) 글자 수: **{len(ref_chars)}**, PSNR 산출 가능: **{n_gt}**",
        f"- 참조 폴더: `{args.ref_dir}`",
        f"- 생성 폴더: `{args.gen_dir}`",
        f"- scipy(연결 컴포넌트): {'O' if _HAS_SCIPY else 'X (fallback flood fill)'}",
        f"- skimage(SSIM): {'O' if _HAS_SKIMAGE else 'X (NaN으로 채움)'}",
        "",
        "## 산출물",
        "- `per_char_metrics.csv` — 음절별 raw 지표",
        "- `summary_by_coverage.csv` — 커버리지 그룹별 요약 통계",
        "- `plots/coverage_hist.png` — 결손 자모 수 분포",
        "- `plots/ink_ratio_by_coverage.png`",
        "- `plots/bbox_fill_by_coverage.png`",
        "- `plots/edge_density_by_coverage.png`",
        "- `plots/n_components_by_coverage.png` *(--with-components)*",
        "- `plots/psnr_by_coverage.png` *(참조 78자 한정)*",
        "- `plots/scatter_psnr_ink.png` *(참조 78자 한정)*",
        "",
        "> 결손 자모 수가 0 인 그룹과 1+ 그룹의 분포·평균 차이를 보고서에 인용한다.",
    ]
    (out_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="자모 커버리지 × 품질 분석",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ref-dir", type=str,
                   default="datasets/train_font_image/reference_images_v2",
                   help="GT/참조 글자 폴더 (78자)")
    p.add_argument("--gen-dir", type=str,
                   default="inference_results/target_style_images/reference_images_v2/images",
                   help="생성 이미지 폴더")
    p.add_argument("--out", type=str, default="report/coverage_quality")
    p.add_argument("--limit", type=int, default=0,
                   help=">0 이면 처음 N개만 분석 (스모크 테스트용)")
    p.add_argument("--with-components", action="store_true",
                   help="연결 컴포넌트 수 계산 (scipy 없으면 느려짐)")
    return p.parse_args()


def main() -> int:
    analyze(parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
