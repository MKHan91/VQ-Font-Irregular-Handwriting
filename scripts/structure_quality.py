#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
structure_quality.py
====================

한글 음절을 구조 카테고리로 묶어 품질 지표 분포를 시각화한다.

분류 축 (모두 자동 생성)
------------------------
1. ``has_jongseong``  — ``build_dataset/structure_tags.json`` 원본 라벨
                       (0 = 받침 없음, 1 = 받침 있음)
2. ``jung_shape``     — 자모 기반 파생: 중성 형태
                       horizontal(ㅏㅑㅓㅕㅣ…) / vertical(ㅗㅛㅜㅠㅡ)
                       / mixed(ㅘㅙㅚㅝㅞㅟㅢ)
3. ``structure_class`` — ``jung_shape × has_jongseong`` (총 6 그룹)

데이터 흐름
-----------
* 가능하면 ``--metrics-csv`` 로 전달된 ``per_char_metrics.csv`` (script #3 산출물)
  를 재사용한다. 없으면 ``--ref-dir`` / ``--gen-dir`` 로부터 PSNR/SSIM/ink_ratio
  를 새로 계산한다 (단순 버전, scipy 불필요).

산출물
------
``<out>/per_char_structure.csv``        : 음절별 분류 + 품질 지표
``<out>/summary_<axis>.csv``            : 축별 그룹 요약 통계 (mean/std/count)
``<out>/plots/<axis>_<metric>.png``     : 박스플롯 (그룹 vs 지표)
``<out>/plots/<axis>_counts.png``       : 그룹별 표본 수 막대
``<out>/INDEX.md``
"""

from __future__ import annotations

import argparse
import csv
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

try:
    from skimage.metrics import structural_similarity as _ssim  # type: ignore
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _ssim = None
    _HAS_SKIMAGE = False

# ---------------------------------------------------------------------------
# 한글 자모 분해
# ---------------------------------------------------------------------------
_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3
_CHO_COUNT, _JUNG_COUNT, _JONG_COUNT = 19, 21, 28

# 중성 21개 인덱스 → 형태 분류
# 0:ㅏ 1:ㅐ 2:ㅑ 3:ㅒ 4:ㅓ 5:ㅔ 6:ㅕ 7:ㅖ 8:ㅗ 9:ㅘ 10:ㅙ 11:ㅚ
# 12:ㅛ 13:ㅜ 14:ㅝ 15:ㅞ 16:ㅟ 17:ㅠ 18:ㅡ 19:ㅢ 20:ㅣ
_HORIZONTAL = {0, 1, 2, 3, 4, 5, 6, 7, 20}          # ㅏㅐㅑㅒㅓㅔㅕㅖㅣ
_VERTICAL = {8, 12, 13, 17, 18}                     # ㅗㅛㅜㅠㅡ
_MIXED = {9, 10, 11, 14, 15, 16, 19}                # ㅘㅙㅚㅝㅞㅟㅢ


def decompose(ch: str) -> Optional[Tuple[int, int, int]]:
    if not ch:
        return None
    code = ord(ch[0])
    if code < _HANGUL_BASE or code > _HANGUL_LAST:
        return None
    off = code - _HANGUL_BASE
    cho = off // (_JUNG_COUNT * _JONG_COUNT)
    jung = (off % (_JUNG_COUNT * _JONG_COUNT)) // _JONG_COUNT
    jong = off % _JONG_COUNT
    return cho, jung, jong


def jung_shape_of(jung_idx: int) -> str:
    if jung_idx in _HORIZONTAL:
        return "horizontal"
    if jung_idx in _VERTICAL:
        return "vertical"
    if jung_idx in _MIXED:
        return "mixed"
    return "unknown"


# ---------------------------------------------------------------------------
# 이미지 품질 (CSV 미제공 시 fallback)
# ---------------------------------------------------------------------------
def _load_gray(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    try:
        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception:
        return None


def _ink_ratio(arr: np.ndarray) -> float:
    return float((arr < 200).mean())


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b_img = Image.fromarray(b).resize(a.shape[::-1], Image.LANCZOS)
        b = np.asarray(b_img, dtype=np.uint8)
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse <= 1e-10:
        return 99.0
    return float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))


def _ssim_val(a: np.ndarray, b: np.ndarray) -> float:
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
# matplotlib 한글
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
# 데이터 수집
# ---------------------------------------------------------------------------
def load_structure_tags(path: Path) -> Dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, int] = {}
    for k, v in raw.items():
        try:
            ch = chr(int(k, 16))
        except (ValueError, TypeError):
            ch = k
        out[ch] = int(v)
    return out


def load_metrics_csv(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ch = row["char"]
            out[ch] = {
                "ink_ratio": float(row["ink_ratio"]),
                "bbox_fill": float(row.get("bbox_fill") or "nan"),
                "edge_density": float(row.get("edge_density") or "nan"),
                "psnr": float(row.get("psnr") or "nan"),
                "ssim": float(row.get("ssim") or "nan"),
                "in_ref78": int(row.get("in_ref78") or 0),
            }
    return out


def compute_metrics_from_images(
    gen_dir: Path,
    ref_dir: Path,
    limit: int,
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    ref_set = {p.stem for p in ref_dir.glob("*.png")}
    files = sorted(gen_dir.glob("*.png"))
    if limit > 0:
        files = files[:limit]
    for idx, fp in enumerate(files, start=1):
        ch = fp.stem
        gen = _load_gray(fp)
        if gen is None:
            continue
        rec = {
            "ink_ratio": _ink_ratio(gen),
            "bbox_fill": float("nan"),
            "edge_density": float("nan"),
            "psnr": float("nan"),
            "ssim": float("nan"),
            "in_ref78": int(ch in ref_set),
        }
        if ch in ref_set:
            gt = _load_gray(ref_dir / f"{ch}.png")
            if gt is not None:
                rec["psnr"] = _psnr(gt, gen)
                rec["ssim"] = _ssim_val(gt, gen)
        out[ch] = rec
        if idx % 1000 == 0:
            print(f"  ...{idx}/{len(files)}")
    return out


# ---------------------------------------------------------------------------
# 분석
# ---------------------------------------------------------------------------
METRICS = ["ink_ratio", "bbox_fill", "edge_density", "psnr", "ssim"]


def classify_chars(
    chars: List[str],
    tags: Dict[str, int],
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for ch in chars:
        dec = decompose(ch)
        if dec is None:
            continue
        _, jung, jong = dec
        shape = jung_shape_of(jung)
        has_jong = tags.get(ch, int(jong != 0))
        struct = f"{shape}_{'jong' if has_jong else 'noJong'}"
        out[ch] = {
            "has_jongseong": int(has_jong),
            "jung_shape": shape,
            "structure_class": struct,
        }
    return out


def summarize(rows: List[dict], axis: str) -> List[dict]:
    by: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by[str(r[axis])].append(r)

    res: List[dict] = []
    for key in sorted(by.keys()):
        grp = by[key]
        rec = {"group": key, "count": len(grp)}
        for m in METRICS:
            vals = [r[m] for r in grp if isinstance(r[m], float)
                    and not math.isnan(r[m])]
            rec[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
            rec[f"{m}_std"] = float(np.std(vals)) if vals else float("nan")
            rec[f"{m}_count"] = len(vals)
        res.append(rec)
    return res


def write_csv(path: Path, rows: List[dict], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
_AXIS_LABEL_KO = {
    "has_jongseong": "받침 유무 (0=없음, 1=있음)",
    "jung_shape": "중성 형태",
    "structure_class": "중성형 × 받침",
}


def plot_counts(rows: List[dict], axis: str, out: Path) -> None:
    cnt = Counter(str(r[axis]) for r in rows)
    keys = sorted(cnt.keys())
    vals = [cnt[k] for k in keys]
    fig, ax = plt.subplots(figsize=(max(4.5, 0.8 * len(keys) + 2), 3.6))
    bars = ax.bar(keys, vals, color="#4C72B0", edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel(_AXIS_LABEL_KO.get(axis, axis))
    ax.set_ylabel("음절 수")
    ax.set_title(f"표본 수 — {axis}")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)


def plot_box(rows: List[dict], axis: str, metric: str, out: Path,
             require_paired: bool = False) -> None:
    by: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if not isinstance(v, float) or math.isnan(v):
            continue
        by[str(r[axis])].append(v)

    keys = sorted(by.keys())
    data = [by[k] for k in keys]
    if not data or all(len(d) == 0 for d in data):
        return

    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * len(keys) + 2), 3.8))
    ax.boxplot(
        data, positions=range(len(keys)), widths=0.55, patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#222"),
        boxprops=dict(facecolor="#C7D7EA", edgecolor="#4C72B0"),
        whiskerprops=dict(color="#4C72B0"),
        capprops=dict(color="#4C72B0"),
    )
    means = [float(np.mean(d)) if d else float("nan") for d in data]
    ax.plot(range(len(keys)), means, "D", color="#C44E52",
            markersize=6, label="mean")
    for i, d in enumerate(data):
        ax.text(i, max(d) if d else 0, f"n={len(d)}",
                ha="center", va="bottom", fontsize=8, color="#666")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_xlabel(_AXIS_LABEL_KO.get(axis, axis))
    ax.set_ylabel(metric + (" (GT-paired)" if require_paired else ""))
    ax.set_title(f"{metric} by {axis}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="구조 카테고리별 품질 분석",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--structure-tags", type=str,
                   default="build_dataset/structure_tags.json")
    p.add_argument("--metrics-csv", type=str, default="",
                   help="script #3 의 per_char_metrics.csv (있으면 재사용)")
    p.add_argument("--gen-dir", type=str,
                   default="inference_results/target_style_images/reference_images_v2/images")
    p.add_argument("--ref-dir", type=str,
                   default="datasets/train_font_image/reference_images_v2")
    p.add_argument("--out", type=str, default="report/structure_quality")
    p.add_argument("--limit", type=int, default=0,
                   help="신규 계산 모드에서 처음 N개만 (스모크용)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _setup_korean_font()

    tags = load_structure_tags(Path(args.structure_tags))
    print(f"[INFO] structure_tags: {len(tags)} 글자 로드")

    if args.metrics_csv and Path(args.metrics_csv).is_file():
        print(f"[INFO] metrics CSV 재사용: {args.metrics_csv}")
        metrics = load_metrics_csv(Path(args.metrics_csv))
    else:
        print(f"[INFO] metrics 신규 계산: gen={args.gen_dir}")
        metrics = compute_metrics_from_images(
            Path(args.gen_dir), Path(args.ref_dir), args.limit,
        )

    print(f"[INFO] metrics 보유 글자 수: {len(metrics)}")

    chars = sorted(set(tags.keys()) & set(metrics.keys()))
    classes = classify_chars(chars, tags)
    print(f"[INFO] 분류 가능 글자 수: {len(classes)}")

    # ---------------- per-char output ----------------
    rows: List[dict] = []
    for ch in chars:
        if ch not in classes:
            continue
        rec = {"char": ch, "unicode": f"U+{ord(ch[0]):04X}"}
        rec.update(classes[ch])
        rec.update({m: metrics[ch].get(m, float("nan")) for m in METRICS})
        rec["in_ref78"] = metrics[ch].get("in_ref78", 0)
        rows.append(rec)

    out_dir = Path(args.out)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    per_cols = ["char", "unicode", "has_jongseong", "jung_shape",
                "structure_class", "in_ref78"] + METRICS
    write_csv(out_dir / "per_char_structure.csv", rows, per_cols)
    print(f"[SAVE] {out_dir/'per_char_structure.csv'}")

    summary_cols_base = ["group", "count"]
    for m in METRICS:
        summary_cols_base += [f"{m}_mean", f"{m}_std", f"{m}_count"]

    for axis in ("has_jongseong", "jung_shape", "structure_class"):
        summ = summarize(rows, axis)
        write_csv(out_dir / f"summary_{axis}.csv", summ, summary_cols_base)
        print(f"[SAVE] summary_{axis}.csv")
        plot_counts(rows, axis, out_dir / "plots" / f"{axis}_counts.png")
        for m in METRICS:
            paired = m in ("psnr", "ssim")
            plot_box(rows, axis, m,
                     out_dir / "plots" / f"{axis}_{m}.png",
                     require_paired=paired)

    # ---------------- INDEX ----------------
    n_paired = sum(1 for r in rows
                   if isinstance(r["psnr"], float) and not math.isnan(r["psnr"]))
    lines = [
        "# Structure × Quality Analysis",
        "",
        f"- 총 분석 글자: **{len(rows):,}**",
        f"- PSNR 산출 가능 (참조 78자): **{n_paired}**",
        f"- skimage SSIM: {'O' if _HAS_SKIMAGE else 'X'}",
        "",
        "## 분류 축",
        "- `has_jongseong` (2 그룹) — structure_tags.json 원본 라벨",
        "- `jung_shape` (3 그룹) — horizontal / vertical / mixed",
        "- `structure_class` (6 그룹) — `jung_shape × has_jongseong`",
        "",
        "## 산출 파일",
        "- `per_char_structure.csv`",
        "- `summary_has_jongseong.csv`",
        "- `summary_jung_shape.csv`",
        "- `summary_structure_class.csv`",
        "- `plots/<axis>_counts.png`, `plots/<axis>_<metric>.png`",
        "",
        "> psnr/ssim 박스플롯은 참조 78자에서만 의미 있음 (n 표기 참고).",
    ]
    (out_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
