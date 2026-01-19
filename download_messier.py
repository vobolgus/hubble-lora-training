#!/usr/bin/env python3
"""Script to download all Messier catalog images from NASA Hubble website."""

import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

BASE_URL = "https://science.nasa.gov/mission/hubble/science/explore-the-night-sky/hubble-messier-catalog/"
OUTPUT_DIR = "messier_images"

# Known Messier numbers (not all numbers 1-110 have entries)
MESSIER_NUMBERS = [
    1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 24,
    27, 28, 30, 31, 32, 33, 35, 42, 43, 44, 45, 46, 48, 49, 51, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
    94, 95, 96, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 110
]

def get_main_image_url(page_url):
    """Extract the main image URL from a Messier page."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(page_url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Error fetching page: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    # Look for the main article image - typically in figure or main content area
    # Try various selectors that NASA uses

    # Method 1: Look for og:image meta tag (usually the main image)
    og_image = soup.find('meta', property='og:image')
    if og_image and og_image.get('content'):
        return og_image['content']

    # Method 2: Look for large images in the article
    for img in soup.find_all('img'):
        src = img.get('src', '') or img.get('data-src', '')
        if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.tif']):
            # Skip small thumbnails and icons
            if 'thumbnail' not in src.lower() and 'icon' not in src.lower():
                if src.startswith('http'):
                    return src
                else:
                    return urljoin(page_url, src)

    return None


def download_image(url, output_path):
    """Download an image from URL to the specified path."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=60, stream=True)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.RequestException as e:
        print(f"  Error downloading: {e}")
        return False


def get_file_extension(url):
    """Extract file extension from URL."""
    # Remove query parameters
    url_path = url.split('?')[0]
    # Get the extension
    ext = os.path.splitext(url_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.gif']:
        return ext
    return '.jpg'  # default


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Downloading Messier catalog images to {OUTPUT_DIR}/")
    print(f"Total objects to download: {len(MESSIER_NUMBERS)}")
    print("-" * 50)

    successful = 0
    failed = []

    for i, m_num in enumerate(MESSIER_NUMBERS, 1):
        page_url = f"{BASE_URL}messier-{m_num}/"
        print(f"[{i}/{len(MESSIER_NUMBERS)}] M{m_num}: ", end="", flush=True)

        image_url = get_main_image_url(page_url)

        if not image_url:
            print("Could not find image URL")
            failed.append(m_num)
            continue

        ext = get_file_extension(image_url)
        output_path = os.path.join(OUTPUT_DIR, f"M{m_num}{ext}")

        if download_image(image_url, output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"OK ({file_size:.1f} KB)")
            successful += 1
        else:
            failed.append(m_num)

        # Be polite to the server
        time.sleep(0.5)

    print("-" * 50)
    print(f"Download complete: {successful}/{len(MESSIER_NUMBERS)} successful")
    if failed:
        print(f"Failed: M{', M'.join(map(str, failed))}")


if __name__ == "__main__":
    main()