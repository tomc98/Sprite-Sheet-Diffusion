#!/usr/bin/env python3
"""Download missing pretrained models from HuggingFace"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil

def download_models():
    base_dir = Path("/root/Sprite-Sheet-Diffusion/ModelTraining/pretrained_model")
    
    models_to_download = [
        {
            "repo_id": "stabilityai/sd-vae-ft-mse",
            "local_dir": base_dir / "sd-vae-ft-mse",
            "files": ["diffusion_pytorch_model.bin", "config.json"]
        },
        {
            "repo_id": "runwayml/stable-diffusion-v1-5",
            "local_dir": base_dir / "stable-diffusion-v1-5",
            "files": ["unet/diffusion_pytorch_model.bin", "unet/config.json", 
                     "vae/diffusion_pytorch_model.bin", "vae/config.json",
                     "text_encoder/pytorch_model.bin", "text_encoder/config.json",
                     "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", 
                     "tokenizer/merges.txt", "model_index.json"]
        },
        {
            "repo_id": "openai/clip-vit-large-patch14",
            "local_dir": base_dir / "image_encoder",
            "files": ["pytorch_model.bin", "config.json"]
        }
    ]
    
    for model_info in models_to_download:
        print(f"\n📥 Downloading {model_info['repo_id']}...")
        local_dir = model_info["local_dir"]
        
        try:
            # Download specific files
            for file in model_info["files"]:
                print(f"  - Downloading {file}")
                try:
                    from huggingface_hub import hf_hub_download
                    
                    # Create subdirectories if needed
                    file_path = Path(file)
                    target_dir = local_dir / file_path.parent if file_path.parent != Path(".") else local_dir
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Download file
                    downloaded_file = hf_hub_download(
                        repo_id=model_info["repo_id"],
                        filename=file,
                        cache_dir="/tmp/hf_cache"
                    )
                    
                    # Copy to target location
                    target_file = local_dir / file
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    if Path(downloaded_file).exists():
                        shutil.copy2(downloaded_file, target_file)
                        print(f"    ✓ {file} downloaded")
                    
                except Exception as e:
                    print(f"    ✗ Failed to download {file}: {e}")
                    
        except Exception as e:
            print(f"  ✗ Error downloading {model_info['repo_id']}: {e}")
    
    print("\n✅ Model download complete!")

if __name__ == "__main__":
    download_models()