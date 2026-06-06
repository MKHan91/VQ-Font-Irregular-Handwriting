#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_comparison_grid.py
=======================

[Content | Ref#1 | Ref#2 | Ref#3 | Generated | GT(optional)] 형태의
비교 그리드를 PNG로 일괄 생성한다. 보고서의 정성적 평가 섹션에
직접 첨부할 수 있도록 라벨/제목/페이지 분할까지 지원한다.

Usage
-----
# 1) 명시적 문자 목록
python3 scripts/make_comparison_grid.py \
    --chars "가,각,꽃,울,행,헐,훿" \
    --out report/grids/cherry_picks.png

# 2) 파일에서 읽기 (한 줄에 한 글자 또는 콤마로 나열)
python3 scripts/make_comparison_grid.py \
    --chars-file report/grids/showcase_chars.txt \
    --rows-per-figure 12 \
    --out report/grids/showcase

# 3) 랜덤 자동 샘플링 (참조에 포함된 78자 vs 비포함을 섞어서)
python3 scripts/make_comparison_grid.py \
    --auto 24 --auto-strategy mixed --seed 42 \
    --rows-per-figure 12 \
    --out report/grids/random_mixed

Notes
-----
* `cr_mapping_v2.json` 의 키/값은 hex unicode 문자열("AC00") 형식이므로
  내부에서 `chr(int(h, 16))` 로 한글 문자열로 변환한다.
* GT(정답) 이미지는 78자 참조 폴더(reference_images_v2)에만 존재하므로
  해당 폴더에 없는 문자는 GT 칸이 비어 있거나(`--gt-mode blank`) 생략된다.
* 출력 파일이 여러 장으로 나뉘면 `<out>_p01.png`, `_p02.png` … 형태가 된다.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# 한글 폰트 자동 탐색 (라벨용)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_korean_font() -> Optional[str]:
    """matplotlib 라벨 렌더링용 한글 ttf 경로를 반환한다."""
    candidates = [
        # 시스템
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/NanumGothic.ttf",
        # 프로젝트 내부 (시스템에 한글 폰트가 없을 때 대비)
        str(_REPO_ROOT / "datasets/train_font_ttf/NanumBarunpenR.ttf"),
        str(_REPO_ROOT / "datasets/additional_train_font_ttf/Nanum_Brush_Script.ttf"),
    ]
    for p in candidates:
        if Path(p).is_file():
            return p
    # 시스템 매니저에서 한글 키워드로 탐색
    for f in fm.fontManager.ttflist:
        name = (f.name or "").lower()
        if any(k in name for k in ("nanum", "noto sans cjk", "malgun", "applegothic")):
            return f.fname
    # 마지막으로 ttf 폴더 전체 스캔
    for d in [_REPO_ROOT / "datasets/train_font_ttf",
              _REPO_ROOT / "datasets/additional_train_font_ttf"]:
        if d.is_dir():
            for f in sorted(d.glob("*.ttf")):
                return str(f)
    return None


def _setup_matplotlib_korean() -> None:
    fp = _find_korean_font()
    if fp:
        fm.fontManager.addfont(fp)
        prop = fm.FontProperties(fname=fp)
        plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 데이터 인덱스
# ---------------------------------------------------------------------------
def load_cr_mapping(path: Path) -> Dict[str, List[str]]:
    """{ '가': ['짝','값','각'], ... } 형태로 정규화하여 반환."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {}
    for k, vs in raw.items():
        try:
            ch = chr(int(k, 16))
        except (ValueError, TypeError):
            ch = k  # 이미 문자형이면 그대로
        ref_chars = []
        for v in vs:
            try:
                ref_chars.append(chr(int(v, 16)))
            except (ValueError, TypeError):
                ref_chars.append(v)
        out[ch] = ref_chars
    return out


# ---------------------------------------------------------------------------
# 이미지 로딩
# ---------------------------------------------------------------------------
def load_image(path: Path, size: int) -> Optional[Image.Image]:
    if not path.is_file():
        return None
    try:
        img = Image.open(path).convert("L")
    except Exception:
        return None
    # 정사각으로 패딩 후 리사이즈 (글자 비율 보존)
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def _placeholder(size: int, label: str = "N/A") -> Image.Image:
    """파일이 없을 때 표시할 회색 플레이스홀더."""
    img = Image.new("L", (size, size), 235)
    return img


# ---------------------------------------------------------------------------
# 문자 선정
# ---------------------------------------------------------------------------
def pick_chars(
    cr_map: Dict[str, List[str]],
    ref_set: set,
    args: argparse.Namespace,
) -> List[str]:
    if args.chars:
        chars = [c for c in args.chars.split(",") if c.strip()]
        return [c.strip()[0] for c in chars]

    if args.chars_file:
        text = Path(args.chars_file).read_text(encoding="utf-8")
        toks: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for tok in line.replace(",", " ").split():
                if tok:
                    toks.append(tok[0])
        return toks

    if args.auto and args.auto > 0:
        rng = random.Random(args.seed)
        keys = sorted(cr_map.keys())
        if args.auto_strategy == "with_gt":
            pool = [c for c in keys if c in ref_set]
        elif args.auto_strategy == "without_gt":
            pool = [c for c in keys if c not in ref_set]
        elif args.auto_strategy == "mixed":
            with_gt = [c for c in keys if c in ref_set]
            without_gt = [c for c in keys if c not in ref_set]
            n_w = min(len(with_gt), args.auto // 2)
            n_wo = args.auto - n_w
            return rng.sample(with_gt, n_w) + rng.sample(without_gt, min(len(without_gt), n_wo))
        else:  # random
            pool = keys
        return rng.sample(pool, min(len(pool), args.auto))

    raise SystemExit("[ERR] --chars / --chars-file / --auto 중 하나를 지정하세요.")


# ---------------------------------------------------------------------------
# 그리드 렌더링
# ---------------------------------------------------------------------------
COL_LABELS_BASE = ["Content", "Ref #1", "Ref #2", "Ref #3", "Generated"]


def _resolve_row(
    ch: str,
    cr_map: Dict[str, List[str]],
    content_dir: Path,
    ref_dir: Path,
    gen_dir: Path,
    gt_dir: Path,
    size: int,
    include_gt: bool,
) -> Tuple[List[Image.Image], List[str], bool]:
    refs = cr_map.get(ch, [])[:3]
    while len(refs) < 3:
        refs.append("")

    content = load_image(content_dir / f"{ch}.png", size) or _placeholder(size)
    ref_imgs = [
        (load_image(ref_dir / f"{r}.png", size) if r else None) or _placeholder(size)
        for r in refs
    ]
    gen = load_image(gen_dir / f"{ch}.png", size) or _placeholder(size)
    row = [content, *ref_imgs, gen]
    sub_labels = ["", *refs, ""]

    has_gt = False
    if include_gt:
        gt_path = gt_dir / f"{ch}.png"
        gt_img = load_image(gt_path, size)
        if gt_img is not None:
            row.append(gt_img)
            sub_labels.append("")
            has_gt = True
        else:
            row.append(_placeholder(size))
            sub_labels.append("(no GT)")
    return row, sub_labels, has_gt


def render_grid(
    rows: List[Tuple[str, List[Image.Image], List[str]]],
    col_labels: List[str],
    out_path: Path,
    *,
    title: Optional[str] = None,
    per_image: float = 1.2,
    dpi: int = 160,
) -> None:
    n_rows = len(rows)
    n_cols = len(col_labels)
    fig_w = per_image * n_cols + 0.6
    fig_h = per_image * n_rows + (0.7 if title else 0.4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    # 컬럼 라벨
    for j, lab in enumerate(col_labels):
        axes[0][j].set_title(lab, fontsize=10, pad=4)

    for i, (target, imgs, sub_labels) in enumerate(rows):
        for j, (img, sub) in enumerate(zip(imgs, sub_labels)):
            ax = axes[i][j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#bbbbbb")
                sp.set_linewidth(0.6)
            if sub:
                ax.text(
                    0.5, -0.04, sub,
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=8, color="#555",
                )
        # 좌측 행 라벨 (타겟 글자)
        axes[i][0].set_ylabel(target, fontsize=14, rotation=0,
                              labelpad=18, va="center")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VQ-Font 비교 그리드 생성기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("source")
    src.add_argument("--chars", type=str, default=None,
                     help="콤마로 구분된 타겟 글자 목록 (예: '가,각,꽃')")
    src.add_argument("--chars-file", type=str, default=None,
                     help="한 줄에 한 글자(또는 콤마 나열)로 적힌 파일")
    src.add_argument("--auto", type=int, default=0,
                     help="N개를 자동 샘플링")
    src.add_argument("--auto-strategy", type=str, default="mixed",
                     choices=["random", "with_gt", "without_gt", "mixed"],
                     help="자동 샘플링 정책")
    src.add_argument("--seed", type=int, default=0)

    paths = p.add_argument_group("paths")
    paths.add_argument("--cr-mapping", type=str,
                       default="build_dataset/cr_mapping_v2.json")
    paths.add_argument("--content-dir", type=str,
                       default="datasets/content_font_image/NanumBarunpenR")
    paths.add_argument("--ref-dir", type=str,
                       default="datasets/train_font_image/reference_images_v2")
    paths.add_argument("--gen-dir", type=str,
                       default="inference_results/target_style_images/reference_images_v2/images")
    paths.add_argument("--gt-dir", type=str,
                       default="datasets/train_font_image/reference_images_v2",
                       help="GT 이미지 폴더 (없으면 ref-dir 과 동일 사용)")

    layout = p.add_argument_group("layout")
    layout.add_argument("--out", type=str, required=True,
                        help="출력 파일 경로 또는 (분할 시) 접두사 디렉터리")
    layout.add_argument("--rows-per-figure", type=int, default=12,
                        help="한 PNG에 들어갈 최대 행 수")
    layout.add_argument("--per-image", type=float, default=1.2,
                        help="셀 한 변의 figure 인치 크기")
    layout.add_argument("--cell-size", type=int, default=192,
                        help="셀 한 변의 픽셀 크기 (리사이즈 기준)")
    layout.add_argument("--dpi", type=int, default=160)
    layout.add_argument("--no-gt", action="store_true",
                        help="GT 열을 항상 제외 (Ref vs Gen 만 비교)")
    layout.add_argument("--gt-mode", type=str, default="auto",
                        choices=["auto", "always", "never"],
                        help="auto: GT 있는 행만 GT 열 표시 / always: 항상 / never: 항상 제외")
    layout.add_argument("--title", type=str, default=None,
                        help="figure 상단 제목")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _setup_matplotlib_korean()

    cr_map = load_cr_mapping(Path(args.cr_mapping))
    ref_dir = Path(args.ref_dir)
    ref_set = {p.stem for p in ref_dir.glob("*.png")}

    include_gt_global = not args.no_gt and args.gt_mode != "never"
    chars = pick_chars(cr_map, ref_set, args)
    if not chars:
        print("[ERR] 선정된 글자가 없습니다.", file=sys.stderr)
        return 2

    print(f"[INFO] 대상 글자 수: {len(chars)}")
    print(f"[INFO] 참조 78자 셋 크기: {len(ref_set)}")

    content_dir = Path(args.content_dir)
    gen_dir = Path(args.gen_dir)
    gt_dir = Path(args.gt_dir)

    # 모든 행 데이터 먼저 구성
    rendered_rows: List[Tuple[str, List[Image.Image], List[str], bool]] = []
    any_gt = False
    for ch in chars:
        row, sub, has_gt = _resolve_row(
            ch, cr_map, content_dir, ref_dir, gen_dir, gt_dir,
            args.cell_size, include_gt_global,
        )
        any_gt = any_gt or has_gt
        rendered_rows.append((ch, row, sub, has_gt))

    # GT 열 사용 여부 최종 결정
    if args.gt_mode == "always":
        use_gt_col = True
    elif args.gt_mode == "never" or args.no_gt:
        use_gt_col = False
    else:  # auto
        use_gt_col = any_gt

    if not use_gt_col:
        # GT 칸 제거
        rendered_rows = [
            (ch, row[:5], sub[:5], False) for (ch, row, sub, _) in rendered_rows
        ]
        col_labels = list(COL_LABELS_BASE)
    else:
        col_labels = list(COL_LABELS_BASE) + ["GT"]

    # 페이지 분할 저장
    out_arg = Path(args.out)
    chunks = [
        rendered_rows[i:i + args.rows_per_figure]
        for i in range(0, len(rendered_rows), args.rows_per_figure)
    ]

    if out_arg.suffix.lower() in (".png", ".jpg", ".jpeg"):
        # 단일 파일 모드: 한 페이지로 강제
        if len(chunks) > 1:
            stem = out_arg.with_suffix("")
            suffix = out_arg.suffix
            out_paths = [
                Path(f"{stem}_p{idx + 1:02d}{suffix}") for idx in range(len(chunks))
            ]
        else:
            out_paths = [out_arg]
    else:
        out_arg.mkdir(parents=True, exist_ok=True)
        out_paths = [
            out_arg / f"grid_p{idx + 1:02d}.png" for idx in range(len(chunks))
        ]

    for idx, (chunk, out_path) in enumerate(zip(chunks, out_paths), start=1):
        title = args.title
        if title and len(chunks) > 1:
            title = f"{title}  (p.{idx}/{len(chunks)})"
        render_grid(
            [(ch, row, sub) for (ch, row, sub, _) in chunk],
            col_labels,
            out_path,
            title=title,
            per_image=args.per_image,
            dpi=args.dpi,
        )
        print(f"[SAVE] {out_path}  ({len(chunk)} rows)")

    print(f"[DONE] 총 {len(chunks)} 페이지 / {len(rendered_rows)} 행")
    return 0


if __name__ == "__main__":
    sys.exit(main())
