#!/usr/bin/env python3
"""
Download required HuggingFace models for Sprite Sheet Diffusion
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def download_models():
    """Download required models from HuggingFace"""
    
    base_dir = Path(__file__).parent / "ModelTraining" / "pretrained_model"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_download = [
        {
            "repo_id": "stabilityai/sd-vae-ft-mse",
            "local_dir": base_dir / "sd-vae-ft-mse",
            "description": "VAE model for image encoding/decoding"
        },
        {
            "repo_id": "runwayml/stable-diffusion-v1-5", 
            "local_dir": base_dir / "stable-diffusion-v1-5",
            "description": "Stable Diffusion v1.5 base model"
        },
        {
            "repo_id": "openai/clip-vit-large-patch14",
            "local_dir": base_dir / "clip-vit-large-patch14",
            "description": "CLIP model for text/image embeddings"
        }
    ]
    
    print("=" * 60)
    print("Downloading required HuggingFace models")
    print("=" * 60)
    print()
    
    for model in models_to_download:
        print(f"Downloading: {model['repo_id']}")
        print(f"Description: {model['description']}")
        print(f"Target directory: {model['local_dir']}")
        
        try:
            # Check if already exists
            if model['local_dir'].exists() and any(model['local_dir'].iterdir()):
                print(f"✓ Model already exists, skipping...")
            else:
                print("Downloading... (this may take several minutes)")
                snapshot_download(
                    repo_id=model['repo_id'],
                    local_dir=str(model['local_dir']),
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                print(f"✓ Successfully downloaded {model['repo_id']}")
                
        except Exception as e:
            print(f"✗ Error downloading {model['repo_id']}: {e}")
            print("Please try again or download manually from HuggingFace")
            return False
        
        print()
    
    print("=" * 60)
    print("✓ All models downloaded successfully!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)