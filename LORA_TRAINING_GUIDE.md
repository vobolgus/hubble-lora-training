# Flux LoRA Training Guide: Hubble Messier Style

This guide walks you through training a LoRA on the Hubble Messier catalog images to create a cosmic/space imagery style.

## Prerequisites

- **Hardware**: M2 Max with 64GB unified memory (sufficient for Flux)
- **Hugging Face**: Account with access to [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- **Python**: 3.10+ recommended
- **Images**: 89 Messier images already downloaded in `messier_images/`

---

## Step 1: Prepare the Dataset

### 1.1 Directory Structure

Create the following structure:
```
lora_training/
├── dataset/
│   ├── img/
│   │   ├── M1.webp
│   │   ├── M1.txt          # caption file
│   │   ├── M2.webp
│   │   ├── M2.txt
│   │   └── ...
│   └── dataset.toml        # dataset config
├── output/                  # trained LoRA will go here
└── config.toml             # training config
```

### 1.2 Resize Images

For Flux, images should be resized to **1024x1024** (or bucketed resolutions). Run:

```bash
cd /Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW/classification-base
python3 prepare_dataset.py
```

This script (provided below) will:
- Copy and resize images to 1024x1024
- Generate caption files for each image

### 1.3 Caption Strategy

For style LoRAs, captions should include:
- A **trigger word** (e.g., `hubble_messier_style`)
- Description of the image content

Example captions:
```
hubble_messier_style, spiral galaxy with blue and pink nebula clouds, deep space photography
hubble_messier_style, globular star cluster with thousands of bright stars, cosmic dust
hubble_messier_style, colorful planetary nebula with concentric gas shells, stellar remnant
```

---

## Step 2: Set Up Training Environment

### 2.1 Install kohya_ss/sd-scripts

```bash
# Clone the repository
cd /Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch for Apple Silicon
pip install torch torchvision torchaudio

# Install requirements
pip install -r requirements.txt

# Install additional dependencies for Flux
pip install accelerate transformers diffusers safetensors
pip install lycoris-lora  # optional, for advanced LoRA types
```

### 2.2 Login to Hugging Face

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Login (you'll need your HF token)
huggingface-cli login
```

### 2.3 Alternative: ai-toolkit (if kohya_ss has issues)

If you encounter MPS issues with kohya_ss, ai-toolkit has better Mac support:

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

---

## Step 3: Configure Training

### 3.1 Dataset Configuration (`dataset.toml`)

Create `lora_training/dataset.toml`:

```toml
[general]
shuffle_caption = true
caption_extension = ".txt"
keep_tokens = 1  # keeps the trigger word in place

[[datasets]]
resolution = 1024
batch_size = 1

  [[datasets.subsets]]
  image_dir = "/Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW/classification-base/lora_training/dataset/img"
  num_repeats = 20  # 89 images × 20 = 1780 steps per epoch
```

### 3.2 Training Configuration

Create `lora_training/config.toml`:

```toml
[model]
pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"
# For local model, use path instead:
# pretrained_model_name_or_path = "/path/to/flux1-dev"

[network]
network_module = "networks.lora_flux"  # Flux-specific LoRA
network_dim = 16        # LoRA rank (16-32 recommended for style)
network_alpha = 16      # Usually same as dim

[optimizer]
optimizer_type = "AdamW8bit"  # Memory efficient
learning_rate = 5e-5
lr_scheduler = "cosine"
lr_warmup_steps = 100

[training]
max_train_steps = 2000
save_every_n_steps = 500
mixed_precision = "bf16"  # Use bf16 on Apple Silicon
gradient_checkpointing = true
gradient_accumulation_steps = 1

[save]
output_dir = "/Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW/classification-base/lora_training/output"
output_name = "hubble_messier_style"
save_model_as = "safetensors"

[logging]
logging_dir = "/Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW/classification-base/lora_training/logs"
```

### 3.3 Key Parameters Explained

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `network_dim` | 16 | LoRA rank. Higher = more expressive but larger file |
| `network_alpha` | 16 | Scaling factor. Usually equals dim |
| `learning_rate` | 5e-5 | How fast the model learns. Lower = more stable |
| `max_train_steps` | 2000 | Total training steps. ~20-25 per image is good |
| `num_repeats` | 20 | How many times to see each image per epoch |

---

## Step 4: Train the LoRA

### 4.1 Run Training with kohya_ss

```bash
cd /Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW/sd-scripts
source venv/bin/activate

# Set environment for Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run training
accelerate launch --mixed_precision bf16 flux_train_network.py \
  --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
  --dataset_config "/Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW/classification-base/lora_training/dataset.toml" \
  --output_dir "/Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW/classification-base/lora_training/output" \
  --output_name "hubble_messier_style" \
  --network_module "networks.lora_flux" \
  --network_dim 16 \
  --network_alpha 16 \
  --optimizer_type "AdamW" \
  --learning_rate 5e-5 \
  --lr_scheduler "cosine" \
  --lr_warmup_steps 100 \
  --max_train_steps 2000 \
  --save_every_n_steps 500 \
  --mixed_precision "bf16" \
  --gradient_checkpointing \
  --cache_latents \
  --save_model_as "safetensors"
```

### 4.2 Alternative: Using ai-toolkit

If kohya_ss doesn't work well on Mac, use ai-toolkit:

Create `ai-toolkit/config/messier_lora.yaml`:

```yaml
job: extension
config:
  name: hubble_messier_style
  process:
    - type: sd_trainer
      training_folder: /Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW/classification-base/lora_training/output
      device: mps
      network:
        type: lora
        linear: 16
        linear_alpha: 16
      save:
        dtype: float16
        save_every: 500
      datasets:
        - folder_path: /Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW/classification-base/lora_training/dataset/img
          caption_ext: txt
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [1024, 1024]
      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: flowmatch
        optimizer: adamw8bit
        lr: 5e-5
        lr_scheduler: cosine
      model:
        name_or_path: black-forest-labs/FLUX.1-dev
        is_flux: true
        quantize: true  # Helps with memory
      sample:
        sampler: flowmatch
        sample_every: 500
        width: 1024
        height: 1024
        prompts:
          - hubble_messier_style, spiral galaxy with colorful nebula clouds
          - hubble_messier_style, dense globular star cluster in deep space
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 3.5
        sample_steps: 20
```

Run training:
```bash
cd /Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW/ai-toolkit
source venv/bin/activate
python run.py config/messier_lora.yaml
```

### 4.3 Monitor Training

- Watch the loss curve - it should decrease over time
- Check sample images generated during training (if enabled)
- Training ~2000 steps on M2 Max should take 2-4 hours

---

## Step 5: Set Up ComfyUI for Inference

### 5.1 Install ComfyUI

```bash
cd /Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

python3 -m venv venv
source venv/bin/activate

pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 5.2 Download/Link Models

```bash
# Create model directories
mkdir -p models/unet models/clip models/vae models/loras

# Option 1: Symlink to HF cache (if already downloaded)
# Option 2: Download manually from HF and place in models/

# For Flux, you need:
# - models/unet/flux1-dev.safetensors (or use HF loader)
# - models/clip/clip_l.safetensors
# - models/clip/t5xxl_fp16.safetensors (or fp8 for less memory)
# - models/vae/ae.safetensors
```

### 5.3 Copy Your LoRA

```bash
cp /Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW/classification-base/lora_training/output/hubble_messier_style.safetensors \
   /Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW/ComfyUI/models/loras/
```

### 5.4 Start ComfyUI

```bash
cd /Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW/ComfyUI
source venv/bin/activate
python main.py --force-fp16
```

Open http://127.0.0.1:8188 in your browser.

### 5.5 ComfyUI Workflow for Flux + LoRA

Import this workflow or build manually:

1. **Load Diffusion Model** → Select FLUX.1-dev
2. **Load LoRA** → Select `hubble_messier_style.safetensors`, strength 0.7-1.0
3. **CLIP Text Encode** → Your prompt with trigger word
4. **KSampler** → Steps: 20-30, CFG: 3.5-4.0, Sampler: euler
5. **VAE Decode** → Connect to output
6. **Save Image** → Save results

---

## Step 6: Generate Required Images

### 6.1 Prompts to Try

Use trigger word `hubble_messier_style` in all prompts:

```
hubble_messier_style, vast spiral galaxy with pink and blue nebula arms, millions of stars, deep space, ultra detailed

hubble_messier_style, colorful planetary nebula with concentric rings of glowing gas, stellar remnant, cosmic art

hubble_messier_style, dense globular cluster with golden and white stars, ancient stellar formation, space photography

hubble_messier_style, collision of two galaxies creating streams of stars and gas, cosmic event, hubble telescope view
```

### 6.2 Generate Banner (1000x240)

In ComfyUI:
1. Set **Empty Latent Image** width=1000, height=240
2. Or generate at 1024x256 and crop/resize
3. Use a prompt like:
   ```
   hubble_messier_style, panoramic view of colorful nebula clouds stretching across deep space, stars scattered throughout, cosmic dust lanes
   ```

### 6.3 Generate Square (512x512)

1. Set **Empty Latent Image** width=512, height=512
2. Or generate at 1024x1024 and downscale
3. Use a prompt like:
   ```
   hubble_messier_style, beautiful spiral galaxy with vibrant purple and blue colors, central bright core, surrounded by stars
   ```

### 6.4 Recommended Settings

| Parameter | Value |
|-----------|-------|
| Steps | 25-30 |
| CFG Scale | 3.5-4.5 |
| Sampler | euler or dpmpp_2m |
| Scheduler | normal or sgm_uniform |
| LoRA Strength | 0.7-1.0 |

---

## Step 7: Upload to CivitAI

### 7.1 Create Account

Go to https://civitai.com/ and create an account if you don't have one.

### 7.2 Upload LoRA

1. Click **Create** → **Upload a Model**
2. Fill in details:
   - **Name**: Hubble Messier Style
   - **Type**: LoRA
   - **Base Model**: Flux.1 Dev
   - **Trigger Words**: `hubble_messier_style`
   - **Description**:
     ```
     LoRA trained on NASA Hubble Space Telescope Messier catalog images.
     Creates cosmic, deep space imagery with colorful nebulae, galaxies,
     and star clusters in the distinctive Hubble photography style.

     Trained on 89 official NASA Hubble images of Messier objects.
     ```
3. Upload:
   - The `.safetensors` file
   - Sample images you generated
4. Set appropriate tags: `space`, `cosmic`, `nebula`, `galaxy`, `hubble`, `astronomy`

### 7.3 Get the URL

After publishing, copy the CivitAI URL for your documentation.

---

## Step 8: Document Artifacts

Create a folder with all artifacts:

```
lora_project_artifacts/
├── code/
│   ├── prepare_dataset.py
│   ├── download_messier.py
│   └── training_config.toml
├── input_images/
│   └── (sample of 5-10 training images)
├── output_images/
│   ├── banner_1000x240.png
│   ├── telegram_512x512.png
│   └── (additional generated samples)
├── lora/
│   └── hubble_messier_style.safetensors
├── comfyui_screenshots/
│   ├── workflow.png
│   └── generation_settings.png
└── README.md  # Include CivitAI URL and summary
```

---

## Troubleshooting

### MPS/Apple Silicon Issues

```bash
# Enable MPS fallback for unsupported operations
export PYTORCH_ENABLE_MPS_FALLBACK=1

# If memory issues occur, try:
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### Out of Memory

- Reduce `batch_size` to 1
- Enable `gradient_checkpointing`
- Use `network_dim` of 8 instead of 16
- For ai-toolkit, enable `quantize: true`

### Slow Training

- Enable `cache_latents` to pre-compute VAE latents
- Use `mixed_precision: bf16`

### LoRA Not Working in ComfyUI

- Ensure the LoRA was trained for Flux (not SD/SDXL)
- Check LoRA strength isn't too low (try 0.8-1.0)
- Verify trigger word is in the prompt

---

## Quick Reference Commands

```bash
# Prepare dataset
cd /Users/svyatoslav.suglobov/PycharmProjects/ImageProcessing-HW/classification-base
python3 prepare_dataset.py

# Train (kohya_ss)
cd ../sd-scripts && source venv/bin/activate
accelerate launch flux_train_network.py [args...]

# Train (ai-toolkit)
cd ../ai-toolkit && source venv/bin/activate
python run.py config/messier_lora.yaml

# Run ComfyUI
cd ../ComfyUI && source venv/bin/activate
python main.py --force-fp16
```

---

## Timeline Estimate

This guide does not include time estimates as requested. Work through each step systematically and proceed to the next when ready.

---

## Resources

- [kohya_ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- [ai-toolkit](https://github.com/ostris/ai-toolkit)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Flux.1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [CivitAI](https://civitai.com/)
- [NASA Hubble Messier Catalog](https://science.nasa.gov/mission/hubble/science/explore-the-night-sky/hubble-messier-catalog/)