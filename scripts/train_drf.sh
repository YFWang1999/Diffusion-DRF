#!/usr/bin/env bash
# Diffusion-DRF on Wan2.1-T2V-1.3B with a frozen Qwen2.5-VL-7B critic (main setting of the paper).
#
# The paper's runs used 16 x A100-80GB (one sample per GPU, effective batch size 16).
# Adjust the paths below, or override them from the environment.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-models/Wan2.1-T2V-1.3B}                      # Wan-AI/Wan2.1-T2V-1.3B
VLM_PATH=${VLM_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}                   # frozen critic
DECODER_PATH=${DECODER_PATH:-models/lightvaew2_1.safetensors}       # lightx2v/Autoencoders
DATA_PATH=${DATA_PATH:-data/drf_train_annotations.tsv}              # see data/README.md
VIDEO_ROOT=${VIDEO_ROOT:-data/videos}                               # root of the reference videos
OUTPUT_DIR=${OUTPUT_DIR:-output/drf_wan1.3b_qwen7b}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}

export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node="${NUM_GPUS}" scripts/train_drf.py \
  --config_path config/wan2.1/wan_civitai.yaml \
  --pretrained_model_name_or_path "${MODEL_DIR}" \
  --reward_decoder lightvae \
  --reward_decoder_path "${DECODER_PATH}" \
  --data_path "${DATA_PATH}" \
  --video_root "${VIDEO_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --reward_fn QwenReward \
  --vlm_path "${VLM_PATH}" \
  --mix_loss \
  --num_frames 10 \
  --ref_frames_num_phy 10 \
  --max_frame_pixels 65536 \
  --train_sample_height 512 \
  --train_sample_width 288 \
  --video_length 49 \
  --num_inference_steps 25 \
  --guidance_scale 6.0 \
  --backprop_strategy tail \
  --backprop_num_steps 3 \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --adam_weight_decay 3e-2 \
  --adam_epsilon 1e-10 \
  --max_grad_norm 50.0 \
  --max_train_steps 3000 \
  --checkpointing_steps 500 \
  --save_video_steps 10 \
  --resume_from_checkpoint latest \
  --report_to tensorboard \
  "$@"
