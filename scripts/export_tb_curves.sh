python3 scripts/export_tb_curves.py \
  --run vqgan=taming/experiments/testtube/version_1 \
  --run vqfont=vq_font_results/runs/vq_font_v4.0 \
  --run finetune=vq_font_results/runs/brush_finetune_v2 \
  --overlay vqfont,finetune \
  --smooth 0.9 \
  --out report/tb_curves