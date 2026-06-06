#!/usr/bin/env python3
"""TensorBoard 이벤트 파일에서 스칼라를 추출해 CSV + 보고서용 PNG 곡선을 생성.

사용 예시
---------
# 단일 run
python scripts/export_tb_curves.py \
    --run vqgan=taming/experiments/testtube/version_1 \
    --run vqfont=vq_font_results/runs/vq_font_v4.0 \
    --run finetune=vq_font_results/runs/brush_finetune_v2 \
    --out report/tb_curves

# 한 그래프에 여러 run 오버레이 (Stage1 vs Stage2 비교 등)
python scripts/export_tb_curves.py \
    --run vqfont=vq_font_results/runs/vq_font_v4.0 \
    --run finetune=vq_font_results/runs/brush_finetune_v2 \
    --overlay vqfont,finetune \
    --out report/tb_curves

옵션
----
--smooth   : EMA 계수 (기본 0.9, 0이면 raw)
--max-step : 표시할 최대 step (긴 학습 일부만 잘라보기)
--format   : png|svg|pdf 중 선택 (기본 png+svg 둘 다)
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("[ERROR] tensorboard가 필요합니다. pip install tensorboard", file=sys.stderr)
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# 데이터 로드
# --------------------------------------------------------------------------- #
def find_event_files(run_dir: Path) -> List[Path]:
    """run_dir 하위의 모든 events.out.tfevents.* 파일을 재귀적으로 수집."""
    return sorted(run_dir.rglob("events.out.tfevents.*"))


def load_scalars(run_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """한 run의 모든 scalar 태그를 {tag: [(step, value), ...]}로 반환."""
    event_files = find_event_files(run_dir)
    if not event_files:
        print(f"[WARN] event 파일 없음: {run_dir}", file=sys.stderr)
        return {}

    merged: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for ev_file in event_files:
        # EventAccumulator는 디렉토리 단위로도 동작하지만, version별 분리를 위해 파일별 처리
        ea = EventAccumulator(
            str(ev_file),
            size_guidance={"scalars": 0},  # 0 = 전부 로드
        )
        try:
            ea.Reload()
        except Exception as e:
            print(f"[WARN] {ev_file} 로드 실패: {e}", file=sys.stderr)
            continue

        for tag in ea.Tags().get("scalars", []):
            for ev in ea.Scalars(tag):
                merged[tag].append((ev.step, float(ev.value)))

    # step 기준 정렬·중복 제거
    for tag in merged:
        seen = {}
        for step, val in merged[tag]:
            seen[step] = val  # 같은 step이 여러 파일에 있으면 마지막 값 채택
        merged[tag] = sorted(seen.items())

    return dict(merged)


# --------------------------------------------------------------------------- #
# 가공
# --------------------------------------------------------------------------- #
def ema_smooth(values: List[float], decay: float) -> List[float]:
    if decay <= 0:
        return list(values)
    smoothed = []
    last = values[0]
    for v in values:
        last = last * decay + v * (1.0 - decay)
        smoothed.append(last)
    return smoothed


def write_csv(run_name: str, tag: str, points: List[Tuple[int, float]], out_dir: Path) -> Path:
    safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_")
    csv_path = out_dir / "csv" / run_name / f"{safe_tag}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "value"])
        writer.writerows(points)
    return csv_path


# --------------------------------------------------------------------------- #
# 플롯
# --------------------------------------------------------------------------- #
def plot_single(
    run_name: str,
    tag: str,
    points: List[Tuple[int, float]],
    out_dir: Path,
    smooth: float,
    max_step: int | None,
    formats: List[str],
) -> None:
    steps, values = zip(*points)
    if max_step is not None:
        kept = [(s, v) for s, v in zip(steps, values) if s <= max_step]
        if not kept:
            return
        steps, values = zip(*kept)

    smoothed = ema_smooth(list(values), smooth)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    if smooth > 0:
        ax.plot(steps, values, alpha=0.25, linewidth=0.8, label="raw")
        ax.plot(steps, smoothed, linewidth=1.8, label=f"EMA(decay={smooth})")
    else:
        ax.plot(steps, values, linewidth=1.4, label="raw")

    ax.set_title(f"{run_name} · {tag}")
    ax.set_xlabel("step")
    ax.set_ylabel(tag)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_")
    for ext in formats:
        out_path = out_dir / "plots" / run_name / f"{safe_tag}.{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(
    tag: str,
    runs: Dict[str, List[Tuple[int, float]]],
    out_dir: Path,
    smooth: float,
    max_step: int | None,
    formats: List[str],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    plotted = 0
    for run_name, points in runs.items():
        if not points:
            continue
        steps, values = zip(*points)
        if max_step is not None:
            kept = [(s, v) for s, v in zip(steps, values) if s <= max_step]
            if not kept:
                continue
            steps, values = zip(*kept)
        smoothed = ema_smooth(list(values), smooth)
        ax.plot(steps, smoothed, linewidth=1.8, label=run_name)
        ax.plot(steps, values, linewidth=0.6, alpha=0.2)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_title(f"overlay · {tag}")
    ax.set_xlabel("step")
    ax.set_ylabel(tag)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_")
    for ext in formats:
        out_path = out_dir / "plots" / "_overlay" / f"{safe_tag}.{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_run_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--run 형식: name=path  (받은 값: {spec!r})"
        )
    name, path = spec.split("=", 1)
    return name.strip(), Path(path).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run", action="append", required=True, type=parse_run_spec,
        help="name=path 형식. 여러 번 지정 가능.",
    )
    parser.add_argument(
        "--overlay", action="append", default=[],
        help="콤마 구분 run 이름. 해당 run들을 한 그래프에 모두 그림. 여러 번 지정 가능.",
    )
    parser.add_argument("--out", type=Path, default=Path("report/tb_curves"))
    parser.add_argument("--smooth", type=float, default=0.9)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument(
        "--format", default="png,svg",
        help="콤마 구분. png|svg|pdf 가능.",
    )
    parser.add_argument(
        "--tags", default=None,
        help="콤마 구분 정규식. 지정 시 매칭되는 태그만 처리.",
    )

    args = parser.parse_args()
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    tag_filters = (
        [re.compile(p) for p in args.tags.split(",")] if args.tags else None
    )

    args.out.mkdir(parents=True, exist_ok=True)

    # 1) 로드
    all_runs: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for name, path in args.run:
        print(f"[LOAD] {name}: {path}")
        if not path.exists():
            print(f"  └ 경로 없음, 건너뜀", file=sys.stderr)
            continue
        scalars = load_scalars(path)
        print(f"  └ tags: {len(scalars)}")
        all_runs[name] = scalars

    if not all_runs:
        print("[ERROR] 로드된 run이 없습니다.", file=sys.stderr)
        return 1

    # 2) CSV + per-run plot
    for run_name, scalars in all_runs.items():
        for tag, points in scalars.items():
            if tag_filters and not any(p.search(tag) for p in tag_filters):
                continue
            if not points:
                continue
            write_csv(run_name, tag, points, args.out)
            plot_single(
                run_name, tag, points, args.out,
                smooth=args.smooth, max_step=args.max_step, formats=formats,
            )
        print(f"[SAVE] {run_name} → {args.out / 'plots' / run_name}")

    # 3) 오버레이 plot (공통 태그만)
    for overlay_spec in args.overlay:
        names = [n.strip() for n in overlay_spec.split(",") if n.strip()]
        chosen = {n: all_runs[n] for n in names if n in all_runs}
        if len(chosen) < 2:
            print(f"[WARN] overlay 대상 부족: {overlay_spec}", file=sys.stderr)
            continue
        common_tags = set.intersection(*(set(s.keys()) for s in chosen.values()))
        for tag in sorted(common_tags):
            if tag_filters and not any(p.search(tag) for p in tag_filters):
                continue
            runs_for_tag = {n: chosen[n][tag] for n in chosen}
            plot_overlay(
                tag, runs_for_tag, args.out,
                smooth=args.smooth, max_step=args.max_step, formats=formats,
            )
        print(f"[SAVE] overlay({','.join(chosen.keys())}) → {args.out / 'plots' / '_overlay'}")

    # 4) 인덱스 마크다운
    write_index(args.out, all_runs)
    print(f"[DONE] 결과: {args.out.resolve()}")
    return 0


def write_index(out_dir: Path, all_runs: Dict[str, Dict]) -> None:
    md = ["# TensorBoard 곡선 인덱스", ""]
    for run_name, scalars in all_runs.items():
        md.append(f"## {run_name}")
        md.append("")
        for tag in sorted(scalars.keys()):
            safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_")
            md.append(f"- **{tag}** — `plots/{run_name}/{safe_tag}.png`")
        md.append("")
    (out_dir / "INDEX.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
