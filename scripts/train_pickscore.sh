#!/usr/bin/env bash
# Baseline: differentiable reward tuning with the scalar PickScore reward, same generator,
# data and optimisation settings as scripts/train_drf.sh. Only the `prompt` column of the
# annotation file is used.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-models/Wan2.1-T2V-1.3B}
DECODER_PATH=${DECODER_PATH:-models/lightvaew2_1.safetensors}
DATA_PATH=${DATA_PATH:-data/drf_train_annotations.tsv}
OUTPUT_DIR=${OUTPUT_DIR:-output/pickscore_wan1.3b}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}

export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node="${NUM_GPUS}" scripts/train_drf.py \
  --config_path config/wan2.1/wan_civitai.yaml \
  --pretrained_model_name_or_path "${MODEL_DIR}" \
  --reward_decoder lightvae \
  --reward_decoder_path "${DECODER_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --reward_fn PickScoreReward \
  --num_frames 16 \
  --train_sample_height 512 \
  --train_sample_width 288 \
  --video_length 49 \
  --num_inference_steps 25 \
  --backprop_strategy tail \
  --backprop_num_steps 4 \
  --gradient_checkpointing \
  --learning_rate 1e-5 \
  --max_train_steps 3000 \
  --checkpointing_steps 500 \
  --resume_from_checkpoint latest \
  "$@"
