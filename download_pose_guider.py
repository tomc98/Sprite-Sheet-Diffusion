#!/usr/bin/env python3
"""
Download the pose_guider.pth model from HuggingFace
This model is required for the diffusion pipeline but wasn't included in the Google Drive models.
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

def main():
    print("=" * 60)
    print("Downloading pose_guider.pth from HuggingFace")
    print("=" * 60)
    
    # Target path
    model_dir = Path(__file__).parent / "ModelTraining" / "pretrained_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    pose_guider_path = model_dir / "pose_guider.pth"
    
    # Check if already exists
    if pose_guider_path.exists():
        print(f"✓ pose_guider.pth already exists at {pose_guider_path}")
        print(f"  Size: {pose_guider_path.stat().st_size / 1024 / 1024:.2f} MB")
        response = input("Download again? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Download URL
    url = "https://huggingface.co/patrolli/AnimateAnyone/resolve/main/pose_guider.pth"
    
    print(f"Downloading from: {url}")
    print(f"Destination: {pose_guider_path}")
    
    try:
        download_file(url, pose_guider_path)
        print(f"✓ Successfully downloaded pose_guider.pth")
        print(f"  Size: {pose_guider_path.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()