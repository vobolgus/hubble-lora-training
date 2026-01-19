#!/bin/bash
# SDXL LoRA Training Script for Hubble Messier Style
# Optimized for Apple Silicon (MPS)

set -e

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false

BASE_DIR="/Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW"
SD_SCRIPTS="${BASE_DIR}/sd-scripts"
DATASET_CONFIG="${BASE_DIR}/classification-base/lora_training/dataset.toml"
OUTPUT_DIR="${BASE_DIR}/classification-base/lora_training/output"
LOG_DIR="${BASE_DIR}/classification-base/lora_training/logs"

# Model path
SDXL_MODEL="${BASE_DIR}/models/sdxl/sd_xl_base_1.0.safetensors"

# Training parameters
NETWORK_DIM=32          # LoRA rank
NETWORK_ALPHA=16        # Usually half of dim for SDXL
LEARNING_RATE=1e-4
MAX_TRAIN_STEPS=1500    # 89 images × ~17 repeats
SAVE_EVERY_N_STEPS=500

cd "${BASE_DIR}"

echo "=========================================="
echo "SDXL LoRA Training: Hubble Messier Style"
echo "=========================================="
echo "Network dim: ${NETWORK_DIM}"
echo "Network alpha: ${NETWORK_ALPHA}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Max steps: ${MAX_TRAIN_STEPS}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Check if model exists
if [ ! -f "${SDXL_MODEL}" ]; then
    echo "ERROR: SDXL model not found at ${SDXL_MODEL}"
    echo "Run: pipenv run python classification-base/download_sdxl.py"
    exit 1
fi

# Switch to main branch for SDXL training
cd "${SD_SCRIPTS}"
git checkout main 2>/dev/null || true
cd "${BASE_DIR}"

ACCELERATE_CONFIG="${BASE_DIR}/classification-base/accelerate_config_mps.yaml"

pipenv run accelerate launch --config_file "${ACCELERATE_CONFIG}" \
    "${SD_SCRIPTS}/sdxl_train_network.py" \
    --pretrained_model_name_or_path "${SDXL_MODEL}" \
    --dataset_config "${DATASET_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_name "hubble_messier_style" \
    --save_model_as safetensors \
    --network_module networks.lora \
    --network_dim ${NETWORK_DIM} \
    --network_alpha ${NETWORK_ALPHA} \
    --optimizer_type adamw \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler cosine \
    --lr_warmup_steps 100 \
    --max_train_steps ${MAX_TRAIN_STEPS} \
    --save_every_n_steps ${SAVE_EVERY_N_STEPS} \
    --mixed_precision fp16 \
    --save_precision fp16 \
    --gradient_checkpointing \
    --cache_latents \
    --cache_latents_to_disk \
    --sdpa \
    --logging_dir "${LOG_DIR}" \
    --seed 42 \
    --caption_extension ".txt" \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --no_half_vae

echo "=========================================="
echo "Training complete!"
echo "LoRA saved to: ${OUTPUT_DIR}/hubble_messier_style.safetensors"
echo "=========================================="