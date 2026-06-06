#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_sheet.py
===============

생성된 한글 음절 이미지를 하나의 큰 PNG 시트에 타일링한다.
보고서 부록(appendix) 의 "전체 결과 한눈에 보기" 페이지용.

기본 동작
---------
* ``--gen-dir`` 의 모든 ``*.png`` 를 유니코드 정렬 후 격자에 배치
* 각 셀 위에 작은 글자 라벨(원본 문자) 옵션 (``--label``)
* 결과가 너무 클 때를 위해 자동 페이지 분할 (``--rows-per-page``)
* KS X 1001 우선 출력 모드(``--ks-x-1001``): 2,350자 우선

산출물
------
``<out>/sheet.png``                — 단일 페이지일 때
``<out>/sheet_p01.png ...``        — 여러 페이지일 때
``<out>/INDEX.md``                  — 페이지 목록 + 메타정보
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# 폰트 자동 탐색 (라벨용)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_label_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        str(_REPO_ROOT / "datasets/train_font_ttf/NanumBarunpenR.ttf"),
    ]
    for p in candidates:
        if Path(p).is_file():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# KS X 1001 (2,350자) — 유니코드 음절 코드 목록
# ---------------------------------------------------------------------------
def _ks_x_1001_codes() -> List[int]:
    """
    KS X 1001 (한글 부분) 2,350개 음절을 유니코드 코드포인트로 반환.

    Python 의 ``euc-kr`` 코덱은 실제로는 CP949 확장(11,172자 전부 인코딩 가능)
    이므로, EUC-KR 인코딩 결과의 **리드/트레일 바이트 범위를 직접 검사**해서
    표준 KS X 1001 영역만 골라낸다.

        - 리드 바이트:  0xB0 ~ 0xC8
        - 트레일 바이트: 0xA1 ~ 0xFE

    이 영역에는 정확히 2,350개의 한글 음절이 정의돼 있다.
    """
    import codecs
    enc = codecs.getencoder("euc-kr")
    codes: List[int] = []
    for cp in range(0xAC00, 0xD7A4):
        try:
            b, _ = enc(chr(cp))
        except UnicodeEncodeError:
            continue
        if len(b) != 2:
            continue
        lead, trail = b[0], b[1]
        if 0xB0 <= lead <= 0xC8 and 0xA1 <= trail <= 0xFE:
            codes.append(cp)
    return codes


# ---------------------------------------------------------------------------
# 글자 목록 결정
# ---------------------------------------------------------------------------
def pick_chars(args: argparse.Namespace, available: List[str]) -> List[str]:
    avail_set = set(available)

    if args.ks_x_1001:
        ks = [chr(cp) for cp in _ks_x_1001_codes()]
        chars = [c for c in ks if c in avail_set]
        return chars

    if args.chars_file:
        text = Path(args.chars_file).read_text(encoding="utf-8")
        out: List[str] = []
        seen = set()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for tok in line.replace(",", " ").split():
                for ch in tok:
                    if ch in avail_set and ch not in seen:
                        out.append(ch)
                        seen.add(ch)
        return out

    chars = sorted(available, key=lambda c: ord(c[0]))
    if args.limit > 0:
        chars = chars[: args.limit]
    return chars


# ---------------------------------------------------------------------------
# 이미지 로딩/축소
# ---------------------------------------------------------------------------
def load_thumb(path: Path, size: int) -> Optional[Image.Image]:
    if not path.is_file():
        return None
    try:
        img = Image.open(path).convert("L")
    except Exception:
        return None
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# 시트 렌더링
# ---------------------------------------------------------------------------
def render_sheet(
    chars: List[str],
    gen_dir: Path,
    out_path: Path,
    *,
    cell: int,
    cols: int,
    rows: int,
    pad: int,
    label: bool,
    label_h: int,
    title: Optional[str],
) -> int:
    """한 페이지 분량(rows*cols) 의 시트를 만든다. 그린 셀 수 반환."""
    cell_total_w = cell + pad
    cell_total_h = cell + pad + (label_h if label else 0)
    sheet_w = pad + cols * cell_total_w
    title_h = 40 if title else 0
    sheet_h = title_h + pad + rows * cell_total_h

    sheet = Image.new("L", (sheet_w, sheet_h), 245)
    draw = ImageDraw.Draw(sheet)

    if title:
        tfont = _find_label_font(20)
        draw.text((pad, 10), title, fill=0, font=tfont)

    lfont = _find_label_font(max(9, label_h - 2)) if label else None

    drawn = 0
    for idx, ch in enumerate(chars[: rows * cols]):
        r = idx // cols
        c = idx % cols
        x = pad + c * cell_total_w
        y = title_h + pad + r * cell_total_h

        thumb = load_thumb(gen_dir / f"{ch}.png", cell)
        if thumb is None:
            # 빈 셀
            draw.rectangle([x, y, x + cell - 1, y + cell - 1],
                           outline=200, fill=235)
        else:
            sheet.paste(thumb, (x, y))
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline=210)
            drawn += 1

        if label and lfont is not None:
            ly = y + cell + 1
            # 라벨 영역 배경 (흰색) 으로 깔끔하게
            draw.rectangle([x, ly, x + cell - 1, ly + label_h - 1],
                           fill=255)
            try:
                bbox = draw.textbbox((0, 0), ch, font=lfont)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except AttributeError:
                tw, th = draw.textsize(ch, font=lfont)
            draw.text((x + (cell - tw) // 2, ly + (label_h - th) // 2 - 1),
                      ch, fill=80, font=lfont)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # PNG optimize=True 는 너무 느리므로 끔
    sheet.save(out_path, "PNG", compress_level=6)
    return drawn


# ---------------------------------------------------------------------------
# 페이지 분할
# ---------------------------------------------------------------------------
def split_pages(
    chars: List[str], per_page: int,
) -> List[List[str]]:
    return [chars[i: i + per_page] for i in range(0, len(chars), per_page)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="11,172자 썸네일 시트 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("source")
    src.add_argument("--gen-dir", type=str,
                     default="inference_results/target_style_images/reference_images_v2/images")
    src.add_argument("--ks-x-1001", action="store_true",
                     help="KS X 1001 2,350자만 출력")
    src.add_argument("--chars-file", type=str, default="",
                     help="문자 목록 파일")
    src.add_argument("--limit", type=int, default=0,
                     help=">0 이면 처음 N자만 (스모크용)")

    layout = p.add_argument_group("layout")
    layout.add_argument("--out", type=str, default="report/sample_sheet")
    layout.add_argument("--cell", type=int, default=64,
                        help="셀 한 변 픽셀")
    layout.add_argument("--cols", type=int, default=40,
                        help="페이지당 열 수")
    layout.add_argument("--rows-per-page", type=int, default=40,
                        help="페이지당 행 수")
    layout.add_argument("--pad", type=int, default=4,
                        help="셀 간격 픽셀")
    layout.add_argument("--no-label", action="store_true",
                        help="셀 하단 글자 라벨 비활성화")
    layout.add_argument("--label-height", type=int, default=14,
                        help="라벨 영역 높이 (픽셀)")
    layout.add_argument("--title", type=str, default="VQ-Font Generated Samples",
                        help="시트 상단 제목 (빈 문자열이면 생략)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    gen_dir = Path(args.gen_dir)
    if not gen_dir.is_dir():
        print(f"[ERR] gen-dir 없음: {gen_dir}", file=sys.stderr)
        return 2

    available = [p.stem for p in gen_dir.glob("*.png") if p.stem]
    print(f"[INFO] 생성 폴더 글자 수: {len(available)}")

    chars = pick_chars(args, available)
    if not chars:
        print("[ERR] 선정된 글자가 없습니다.", file=sys.stderr)
        return 2

    label = not args.no_label
    per_page = args.cols * args.rows_per_page
    pages = split_pages(chars, per_page)
    print(f"[INFO] 출력 글자 수: {len(chars):,}, "
          f"페이지 수: {len(pages)} ({args.cols}×{args.rows_per_page} = {per_page}/p)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_drawn = 0
    paths: List[Path] = []
    for i, chunk in enumerate(pages, start=1):
        if len(pages) == 1:
            out_path = out_dir / "sheet.png"
        else:
            out_path = out_dir / f"sheet_p{i:02d}.png"
        title = None
        if args.title:
            title = args.title if len(pages) == 1 else \
                f"{args.title}  (p.{i}/{len(pages)})"
        drawn = render_sheet(
            chunk, gen_dir, out_path,
            cell=args.cell, cols=args.cols, rows=args.rows_per_page,
            pad=args.pad, label=label, label_h=args.label_height,
            title=title,
        )
        total_drawn += drawn
        paths.append(out_path)
        print(f"[SAVE] {out_path}  ({drawn}/{len(chunk)} cells)")

    # 인덱스
    lines = [
        "# Sample Sheet",
        "",
        f"- 소스: `{args.gen_dir}`",
        f"- 총 글자 수: **{len(chars):,}** "
        f"({'KS X 1001' if args.ks_x_1001 else 'all'})",
        f"- 그려진 셀: **{total_drawn:,}**",
        f"- 그리드: {args.cols} × {args.rows_per_page}  "
        f"(셀 {args.cell}px, 패딩 {args.pad}px)",
        f"- 페이지 수: **{len(pages)}**",
        "",
        "## 페이지 목록",
    ]
    for p in paths:
        lines.append(f"- [{p.name}]({p.name})")
    (out_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] 총 {total_drawn:,} 셀 / {len(pages)} 페이지 → {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
