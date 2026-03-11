#!/usr/bin/env python3
"""Download Parakeet TDT 0.6B v3 (int8) and Silero VAD models."""

import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

MODELS_DIR = Path(__file__).resolve().parent / "models"

FILES = [
    # Parakeet TDT 0.6B v3 int8 (sherpa-onnx format)
    {
        "name": "encoder.int8.onnx",
        "url": "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main/encoder.int8.onnx",
    },
    {
        "name": "decoder.int8.onnx",
        "url": "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main/decoder.int8.onnx",
    },
    {
        "name": "joiner.int8.onnx",
        "url": "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main/joiner.int8.onnx",
    },
    {
        "name": "tokens.txt",
        "url": "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main/tokens.txt",
    },
    # Silero VAD
    {
        "name": "silero_vad.onnx",
        "url": "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    },
]


def download_file(url: str, dest: Path):
    """Download a file with progress bar."""
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for file_info in FILES:
        dest = MODELS_DIR / file_info["name"]
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  {file_info['name']} already exists, skipping.")
            continue
        print(f"Downloading {file_info['name']}...")
        download_file(file_info["url"], dest)
    print("\nAll models downloaded to:", MODELS_DIR)


if __name__ == "__main__":
    main()
