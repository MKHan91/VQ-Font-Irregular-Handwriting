#!/usr/bin/env bash
# run_revision.sh - revision experiments for electronics-4654524.
# Run from the VQ-Font project root (the folder with train.py and inference.py):
#     bash revision_kit/run_revision.sh E0      # one block at a time
#     bash revision_kit/run_revision.sh ALL     # everything in priority order
# Blocks: E0 E1 E2 E3 E4 E5 PROBE PACK. See README_KO.md for timings and what each block answers.
set -euo pipefail

# ---------------------------------------------------------------- paths (EDIT THESE) --------
STAGE0_CKPT="taming/experiments/checkpoints/2026-01-11T10-58-28_custom_vqgan/epoch=000781.ckpt"
STAGE1_CKPT="vq_font_results/checkpoints/vq_font_v4.0/last.ckpt"
STAGE2_CKPT="vq_font_results/checkpoints/brush_finetune_v2/last.ckpt"
REFS="datasets/train_font_image/reference_images_v2"          # the 78 handwritten references
CONTENT="datasets/content_font_image/NanumBarunpenR"
CR_MAP="build_dataset/cr_mapping_v2.json"
META1="/home/sona/VQ-Font-Irregular-Handwriting/datasets/handwrite_dataset_v3/meta/train.json"   # meta used in Stage 1
META2="/home/sona/VQ-Font-Irregular-Handwriting/datasets/handwrite_dataset_v3/meta/train.json"   # meta used in Stage 2
FT_CFG="cfgs/custom_finetune.yaml"
ORIG_GEN="inference_results/target_style_images/reference_images_v2/images"   # the 11,172 glyphs of the original submission
ITER=41500            # Stage 2 length actually logged; identical for every trained arm
SEEDS="2"             # add more seeds if time allows, e.g. "2 3 4"
K=5
# --------------------------------------------------------------------------------------------

KIT="$(cd "$(dirname "$0")" && pwd)"
OUT="revision_runs"; PREP="$OUT/prep"; mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/run_log.txt"; }
timeit() { local t0=$(date +%s); "$@"; local t1=$(date +%s); log "  elapsed $((t1-t0)) s : $*"; }

infer() {  # infer <ckpt> <img_path> <mapping.json> <saving_root>
  local ck="$1" img="$2" map="$3" root="$4"; mkdir -p "$root"
  ( export RK_CR_MAPPING="$(realpath "$map")"
    timeit python inference.py --weight "$ck" --img_path "$img" \
      --content_font "$CONTENT" --saving_root "$root" ) 2>&1 | tail -n 5 | tee -a "$OUT/run_log.txt"
  # inference.py writes <saving_root>/<basename img_path>/images/<char>.png
}

train_arm() {  # train_arm <name> <mapping> <meta> <resume_ckpt or ""> [ENV=VAL ...]
  local name="$1" map="$2" meta="$3" resume="$4"; shift 4
  local ov="$OUT/cfg_$name.yaml" seed=2 kv
  for kv in "$@"; do case "$kv" in RK_PY_SEED=*) seed="${kv#*=}";; esac; done
  printf 'content_reference_json: %s\ncr_mapping_path: %s\ndata_meta: %s\niter: %s\nwork_dir: %s\nseed: %s\n' \
      "$(realpath "$map")" "$(realpath "$map")" "$(realpath "$meta")" "$ITER" "$OUT/train" "$seed" > "$ov"
  log "TRAIN $name ($*)"
  ( for kv in "$@"; do export "$kv"; done
    export RK_CR_MAPPING="$(realpath "$map")" RK_TRAINABLE_LOG="$(realpath "$OUT")/trainable_$name.txt"
    timeit python train.py --name "$name" --config_paths "$FT_CFG" "$ov" \
      --vq_gan_resume "$STAGE0_CKPT" --vq_font_resume "$resume" ) 2>&1 | tail -n 20 | tee -a "$OUT/run_log.txt"
}

E0() {  # parameter audit + data checks (CPU, minutes)
  python "$KIT/rk_patch.py" --root .
  python "$KIT/rk_prepare.py" --refs "$REFS" --cr_mapping "$CR_MAP" --meta "$META2" --out "$PREP"
  python "$KIT/rk_audit.py" --stage0 "$STAGE0_CKPT" --stage1 "$STAGE1_CKPT" --stage2 "$STAGE2_CKPT" \
      --meta1 "$META1" --meta2 "$META2" --out "$OUT/E0_audit"
  nvidia-smi --query-gpu=name,memory.total --format=csv > "$OUT/E0_audit/gpu.txt" || true
  pip freeze > "$OUT/E0_audit/pip_freeze.txt" || true
}

E1() {  # inference-time exclusion of the target from S(u), 78 references (minutes)
  infer "$STAGE2_CKPT" "$REFS" "$PREP/mappings/cr_selfexcl78.json" "$OUT/E1/stage2_selfexcl"
  infer "$STAGE1_CKPT" "$REFS" "$PREP/mappings/cr_selfexcl78.json" "$OUT/E1/stage1only_selfexcl"
  for a in stage2_selfexcl stage1only_selfexcl; do
    python "$KIT/rk_eval_paired.py" --gen "$OUT/E1/$a/$(basename "$REFS")/images" --ref "$REFS" \
        --out "$OUT/E1/paired_$a.csv" --tag "$a" | tee -a "$OUT/run_log.txt"
  done
}

E2() {  # Stage-1-only checkpoint over the full inventory (published mapping)
  infer "$STAGE1_CKPT" "$REFS" "$CR_MAP" "$OUT/E2/stage1only_full"
  python "$KIT/rk_eval_paired.py" --gen "$OUT/E2/stage1only_full/$(basename "$REFS")/images" --ref "$REFS" \
      --out "$OUT/E2/paired_stage1only_selfincl.csv" --tag stage1only_selfincl | tee -a "$OUT/run_log.txt"
}

fold_eval() {  # fold_eval <arm> <fold> <ckpt> <seed>
  local arm="$1" k="$2" ck="$3" s="$4" d="$PREP/folds/fold$2"
  infer "$ck" "$d/refs" "$d/cr_heldout.json" "$OUT/E3/${arm}_f${k}_s${s}"
  python "$KIT/rk_eval_paired.py" --gen "$OUT/E3/${arm}_f${k}_s${s}/refs/images" --ref "$REFS" \
      --chars "$d/heldout.txt" --out "$OUT/E3/paired_${arm}_f${k}_s${s}.csv" --tag "${arm}_f${k}_s${s}" | tee -a "$OUT/run_log.txt"
}

E3() {  # 5-fold held-out Stage 2 (arm A = configuration as run) vs Stage-1-only (arm D)
  for s in $SEEDS; do for k in $(seq 0 $((K-1))); do
    d="$PREP/folds/fold$k"
    fold_eval D "$k" "$STAGE1_CKPT" "$s"
    train_arm "rk_A_f${k}_s${s}" "$d/cr_mapping.json" "$d/meta.json" "$STAGE1_CKPT" RK_PY_SEED="$s"
    fold_eval A "$k" "$OUT/train/checkpoints/rk_A_f${k}_s${s}/last.ckpt" "$s"
  done; done
}

E4() {  # ablations on the same folds: B no freezing, C no latent augmentation
  for s in $SEEDS; do for k in $(seq 0 $((K-1))); do
    d="$PREP/folds/fold$k"
    train_arm "rk_B_f${k}_s${s}" "$d/cr_mapping.json" "$d/meta.json" "$STAGE1_CKPT" RK_FREEZE=none RK_PY_SEED="$s"
    fold_eval B "$k" "$OUT/train/checkpoints/rk_B_f${k}_s${s}/last.ckpt" "$s"
    train_arm "rk_C_f${k}_s${s}" "$d/cr_mapping.json" "$d/meta.json" "$STAGE1_CKPT" RK_LATENT_SIGMA=0 RK_PY_SEED="$s"
    fold_eval C "$k" "$OUT/train/checkpoints/rk_C_f${k}_s${s}/last.ckpt" "$s"
  done; done
}

E5() {  # from-scratch control for H2: Stage 0 codebook + randomly initialised generator,
        # all generator modules trainable, Stage 2 schedule, all 78 references.
  train_arm "rk_scratch_s2" "$CR_MAP" "$META2" "" RK_FREEZE=none RK_PY_SEED=2
  infer "$OUT/train/checkpoints/rk_scratch_s2/last.ckpt" "$REFS" "$CR_MAP" "$OUT/E5/scratch_full"
  # held-out version on the folds (comparable with arms A-D)
  for k in $(seq 0 $((K-1))); do
    d="$PREP/folds/fold$k"
    train_arm "rk_F_f${k}_s2" "$d/cr_mapping.json" "$d/meta.json" "" RK_FREEZE=none RK_PY_SEED=2
    fold_eval F "$k" "$OUT/train/checkpoints/rk_F_f${k}_s2/last.ckpt" 2
  done
}

PROBE() {  # jamo-recognition probe on the full inventories (CPU or GPU, minutes each)
  mkdir -p "$OUT/probe"
  python "$KIT/rk_probe_score.py" "$ORIG_GEN" "$OUT/probe/full_pipeline.json" | tee -a "$OUT/run_log.txt"
  [ -d "$OUT/E2/stage1only_full/$(basename "$REFS")/images" ] && \
    python "$KIT/rk_probe_score.py" "$OUT/E2/stage1only_full/$(basename "$REFS")/images" "$OUT/probe/stage1only.json" | tee -a "$OUT/run_log.txt"
  if [ -d "$OUT/E5/scratch_full/$(basename "$REFS")/images" ]; then
    python "$KIT/rk_probe_score.py" "$OUT/E5/scratch_full/$(basename "$REFS")/images" "$OUT/probe/scratch.json" | tee -a "$OUT/run_log.txt"
    python "$KIT/rk_mcnemar.py" "$OUT/probe/full_pipeline_correct.npy" "$OUT/probe/scratch_correct.npy" | tee -a "$OUT/probe/H2_mcnemar.txt" "$OUT/run_log.txt"
  fi
}

PACK() {  # everything to send back (images, CSVs, logs; no checkpoints)
  tar --exclude='*.ckpt' --exclude='*.pth' -czf revision_results.tar.gz "$OUT" \
      vq_font_results/logs 2>/dev/null || tar --exclude='*.ckpt' -czf revision_results.tar.gz "$OUT"
  log "PACKED revision_results.tar.gz ($(du -h revision_results.tar.gz | cut -f1))"
}

case "${1:-}" in
  E0|E1|E2|E3|E4|E5|PROBE|PACK) "$1" ;;
  ALL) E0; E1; E2; E3; E5; PROBE; E4; PACK ;;
  *) echo "usage: $0 {E0|E1|E2|E3|E4|E5|PROBE|PACK|ALL}"; exit 1 ;;
esac
