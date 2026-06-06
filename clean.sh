#! /bin/bash

backups_dir="/home/dev/Project/VQ-Font/vq_font_results/checkpoints"
codes_dir="/home/dev/Project/VQ-Font/vq_font_results/codes"
logs_dir="/home/dev/Project/VQ-Font/vq_font_results/logs"
models_dir="/home/dev/Project/VQ-Font/vq_font_results/runs"

model_name="brush_finetune_v2"
rm -rf ${backups_dir}/${model_name}
rm -rf ${codes_dir}/${model_name}
rm -rf ${logs_dir}/${model_name}
rm -rf ${models_dir}/${model_name}
