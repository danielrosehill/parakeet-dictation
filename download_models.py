#!/usr/bin/env python3
"""Download model files for Parakeet Dictation.

Usage:
    python download_models.py              # download default (desktop) profile + VAD
    python download_models.py desktop      # download desktop profile
    python download_models.py laptop       # download laptop profile
    python download_models.py streaming    # download streaming profile
    python download_models.py all          # download all profiles
"""

import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
MODELS_JSON = APP_DIR / "models.json"


def download_file(url: str, dest: Path):
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))


def download_vad(config: dict):
    vad = config["vad"]
    dest = MODELS_DIR / vad["filename"]
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {vad['filename']} already exists.")
        return
    print(f"Downloading VAD model...")
    download_file(vad["url"], dest)


def download_profile(config: dict, profile_id: str):
    profiles = config["profiles"]
    if profile_id not in profiles:
        print(f"Unknown profile: {profile_id}")
        print(f"Available: {', '.join(profiles.keys())}")
        sys.exit(1)

    profile = profiles[profile_id]
    profile_dir = MODELS_DIR / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {profile['name']} ({profile['size_mb']} MB) ===")

    for key, info in profile["files"].items():
        dest = profile_dir / info["filename"]
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  {info['filename']} already exists.")
            continue
        print(f"Downloading {info['filename']}...")
        download_file(info["url"], dest)


def main():
    with open(MODELS_JSON) as f:
        config = json.load(f)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    targets = sys.argv[1:] if len(sys.argv) > 1 else ["desktop"]
    if "all" in targets:
        targets = list(config["profiles"].keys())

    # Always download VAD
    download_vad(config)

    for t in targets:
        download_profile(config, t)

    print("\nDone. Models are in:", MODELS_DIR)


if __name__ == "__main__":
    main()
