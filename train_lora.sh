#!/bin/bash
# Flux LoRA Training Script for Hubble Messier Style
# Run from: /Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW

set -e

# Environment setup for Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export TOKENIZERS_PARALLELISM=false

# Paths
BASE_DIR="/Users/svyatoslav.suglobov/PycharmProjects/hubble-lora-training"
SD_SCRIPTS="${BASE_DIR}/sd-scripts"
DATASET_CONFIG="${BASE_DIR}/lora_training/dataset.toml"
OUTPUT_DIR="${BASE_DIR}/lora_training/output"
LOG_DIR="${BASE_DIR}/lora_training/logs"

# Training parameters
NETWORK_DIM=16          # LoRA rank (8-32 typical, higher = more expressive)
NETWORK_ALPHA=16        # Usually same as dim
LEARNING_RATE=1e-4      # Learning rate
MAX_TRAIN_STEPS=2000    # Total training steps (89 images × 20 repeats ≈ 1780 per epoch)
SAVE_EVERY_N_STEPS=500  # Checkpoint frequency

cd "${BASE_DIR}"

echo "=========================================="
echo "Flux LoRA Training: Hubble Messier Style"
echo "=========================================="
echo "Network dim: ${NETWORK_DIM}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Max steps: ${MAX_TRAIN_STEPS}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Model paths (downloaded locally)
FLUX_MODEL="${BASE_DIR}/models/flux1-dev"

ACCELERATE_CONFIG="${BASE_DIR}/accelerate_config_mps.yaml"

/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -c "
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
exec(open('${SD_SCRIPTS}/flux_train_network.py').read())
" \
    --pretrained_model_name_or_path "${FLUX_MODEL}/flux1-dev.safetensors" \
    --clip_l "${FLUX_MODEL}/text_encoder/model.safetensors" \
    --t5xxl "${FLUX_MODEL}/text_encoder_2/t5xxl.safetensors" \
    --ae "${FLUX_MODEL}/ae.safetensors" \
    --dataset_config "${DATASET_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_name "hubble_messier_style" \
    --save_model_as safetensors \
    --network_module networks.lora_flux \
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
    --full_fp16 \
    --gradient_checkpointing \
    --cache_latents \
    --sdpa \
    --logging_dir "${LOG_DIR}" \
    --seed 42 \
    --caption_extension ".txt" \
    --max_data_loader_n_workers 0 \
    --metadata_title "Hubble Messier Style" \
    --metadata_author "vobolgus" \
    --metadata_description "LoRA trained on NASA Hubble Messier catalog images for cosmic/space style" \
    --metadata_trigger_phrase "hubble_messier_style"

echo "=========================================="
echo "Training complete!"
echo "LoRA saved to: ${OUTPUT_DIR}/hubble_messier_style.safetensors"
echo "=========================================="