#!/usr/bin/env python3
"""
Prepare Messier images dataset for LoRA training.
- Resizes images to 1024x1024
- Generates caption files with trigger word
"""

import os
import shutil
from pathlib import Path
from PIL import Image

# Configuration
SOURCE_DIR = "messier_images"
OUTPUT_DIR = "lora_training/dataset/img"
TARGET_SIZE = 1024
TRIGGER_WORD = "hubble_messier_style"

# Messier object descriptions for better captions
MESSIER_DESCRIPTIONS = {
    1: "crab nebula, supernova remnant with filamentary structure, colorful gas clouds",
    2: "globular cluster with dense core of ancient stars, spherical star formation",
    3: "globular cluster with bright stellar core, thousands of stars",
    4: "globular cluster, nearby stellar sphere with resolved individual stars",
    5: "globular star cluster with dense bright core, ancient stars",
    7: "open star cluster, loose grouping of bright blue and white stars",
    8: "lagoon nebula, emission nebula with bright pink and red gas clouds, star forming region",
    9: "globular cluster with concentrated stellar core",
    10: "globular cluster, spherical collection of ancient stars",
    11: "wild duck cluster, dense open cluster resembling flying birds",
    12: "globular cluster with loose stellar distribution",
    13: "hercules globular cluster, dense sphere of hundreds of thousands of stars",
    14: "globular cluster with moderately concentrated core",
    15: "globular cluster with extremely dense core, ancient stellar population",
    16: "eagle nebula, pillars of creation, towering columns of gas and dust, star nursery",
    17: "omega nebula, swan nebula, bright emission nebula with pink gas clouds",
    19: "globular cluster, oblate spheroid shape, dense stellar core",
    20: "trifid nebula, combination emission reflection and dark nebula, three-lobed structure",
    22: "globular cluster, one of nearest to earth, bright dense core",
    24: "sagittarius star cloud, dense field of milky way stars",
    27: "dumbbell nebula, planetary nebula with bipolar structure, glowing gas shell",
    28: "globular cluster near galactic center",
    30: "globular cluster with collapsed core, dense stellar concentration",
    31: "andromeda galaxy, large spiral galaxy with dust lanes and bright core, nearest major galaxy",
    32: "elliptical galaxy, satellite of andromeda, smooth stellar distribution",
    33: "triangulum galaxy, spiral galaxy with blue star forming regions, loose spiral arms",
    35: "open cluster with scattered bright stars",
    42: "orion nebula, massive star forming region, colorful gas clouds, stellar nursery",
    43: "de mairans nebula, part of orion complex, emission nebula",
    44: "beehive cluster, praesepe, open cluster visible to naked eye",
    45: "pleiades, seven sisters, open cluster with blue reflection nebulae",
    46: "open cluster with planetary nebula, scattered stars",
    48: "open cluster with bright scattered stars",
    49: "elliptical galaxy in virgo cluster, smooth stellar halo",
    51: "whirlpool galaxy, grand design spiral with companion galaxy, prominent spiral arms",
    53: "globular cluster with moderate concentration",
    54: "globular cluster, part of sagittarius dwarf galaxy",
    55: "globular cluster with loose structure, large apparent size",
    56: "globular cluster with moderate stellar density",
    57: "ring nebula, planetary nebula with perfect ring structure, glowing gas torus",
    58: "barred spiral galaxy in virgo cluster",
    59: "elliptical galaxy in virgo cluster",
    60: "elliptical galaxy with companion spiral, interacting pair",
    61: "spiral galaxy with active star formation, bright spiral arms",
    62: "globular cluster with irregular shape",
    63: "sunflower galaxy, flocculent spiral with patchy arms",
    64: "black eye galaxy, spiral with dark dust band across nucleus",
    65: "spiral galaxy in leo triplet, edge-on view with dust lane",
    66: "spiral galaxy in leo triplet with asymmetric arms",
    67: "old open cluster with ancient stars",
    68: "globular cluster with loose structure",
    69: "globular cluster near galactic center",
    70: "globular cluster with dense core",
    71: "loose globular cluster resembling open cluster",
    72: "globular cluster with low concentration",
    74: "grand design spiral galaxy with well-defined arms, face-on view",
    75: "globular cluster with highly concentrated core",
    76: "little dumbbell nebula, planetary nebula with bipolar lobes",
    77: "seyfert galaxy with active nucleus, barred spiral structure",
    78: "reflection nebula with blue scattered starlight, dusty region",
    79: "globular cluster in lepus constellation",
    80: "dense globular cluster with concentrated core",
    81: "bodes galaxy, grand design spiral with prominent dust lanes",
    82: "cigar galaxy, starburst galaxy with galactic superwind, edge-on irregular",
    83: "southern pinwheel, barred spiral with active star formation",
    84: "lenticular galaxy in virgo cluster",
    85: "lenticular galaxy with smooth disk",
    86: "elliptical galaxy in virgo cluster",
    87: "giant elliptical galaxy with relativistic jet, active galactic nucleus",
    88: "spiral galaxy with tightly wound arms",
    89: "elliptical galaxy almost perfectly spherical",
    90: "spiral galaxy with smooth arms, anemic spiral",
    91: "barred spiral galaxy with prominent bar structure",
    92: "globular cluster with dense core, ancient stellar population",
    94: "spiral galaxy with starburst ring, bright inner disk",
    95: "barred spiral galaxy with ring structure",
    96: "intermediate spiral galaxy with asymmetric arms",
    98: "spiral galaxy seen at high inclination",
    99: "spiral galaxy with asymmetric arm structure, disturbed morphology",
    100: "grand design spiral with symmetric arms, face-on view",
    101: "pinwheel galaxy, large face-on spiral with many HII regions",
    102: "spindle galaxy, edge-on lenticular with prominent dust lane",
    104: "sombrero galaxy, edge-on spiral with large bulge and dust lane",
    105: "elliptical galaxy in leo group",
    106: "seyfert galaxy with water maser emission, spiral structure",
    107: "globular cluster with loose structure",
    108: "barred spiral galaxy seen edge-on",
    109: "barred spiral galaxy with faint outer arms",
    110: "dwarf elliptical galaxy, satellite of andromeda",
}


def get_caption(messier_num: int) -> str:
    """Generate a caption for a Messier object."""
    description = MESSIER_DESCRIPTIONS.get(
        messier_num,
        "deep space astronomical object, cosmic formation"
    )
    return f"{TRIGGER_WORD}, {description}, hubble space telescope photography, high detail"


def resize_image(input_path: Path, output_path: Path, size: int) -> bool:
    """Resize an image to a square, maintaining aspect ratio with padding or cropping."""
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Calculate dimensions for center crop to square
            width, height = img.size
            min_dim = min(width, height)

            # Center crop to square
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim

            img_cropped = img.crop((left, top, right, bottom))

            # Resize to target size
            img_resized = img_cropped.resize((size, size), Image.Resampling.LANCZOS)

            # Save as PNG for training (lossless)
            output_path = output_path.with_suffix('.png')
            img_resized.save(output_path, 'PNG', optimize=True)

            return True
    except Exception as e:
        print(f"  Error processing {input_path}: {e}")
        return False


def main():
    source_path = Path(SOURCE_DIR)
    output_path = Path(OUTPUT_DIR)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Preparing dataset from {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Trigger word: {TRIGGER_WORD}")
    print("-" * 50)

    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff'}
    images = [f for f in source_path.iterdir()
              if f.suffix.lower() in image_extensions]

    print(f"Found {len(images)} images")

    successful = 0

    for img_path in sorted(images):
        # Extract Messier number from filename (e.g., M1.webp -> 1)
        filename = img_path.stem
        try:
            messier_num = int(filename.replace('M', '').replace('m', ''))
        except ValueError:
            print(f"  Skipping {img_path.name} - cannot parse Messier number")
            continue

        # Output paths
        out_img_path = output_path / f"M{messier_num}.png"
        out_txt_path = output_path / f"M{messier_num}.txt"

        print(f"Processing M{messier_num}...", end=" ")

        # Resize and save image
        if resize_image(img_path, out_img_path, TARGET_SIZE):
            # Generate caption
            caption = get_caption(messier_num)
            out_txt_path.write_text(caption)
            print("OK")
            successful += 1
        else:
            print("FAILED")

    print("-" * 50)
    print(f"Dataset prepared: {successful}/{len(images)} images")
    print(f"Output location: {output_path.absolute()}")
    print(f"\nNext step: Set up training environment (see LORA_TRAINING_GUIDE.md)")


if __name__ == "__main__":
    main()