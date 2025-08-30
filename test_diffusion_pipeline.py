#!/usr/bin/env python3
"""
Test script to verify the diffusion pipeline can be loaded and used
"""

import os
import sys
import torch
from pathlib import Path

# Add ModelTraining to path
sys.path.append(str(Path(__file__).parent / "ModelTraining"))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from models.pose_guider_org import PoseGuiderOrg
        print("✓ PoseGuiderOrg imported")
    except ImportError as e:
        print(f"✗ Failed to import PoseGuiderOrg: {e}")
        return False
    
    try:
        from models.unet_2d_condition import UNet2DConditionModel
        print("✓ UNet2DConditionModel imported")
    except ImportError as e:
        print(f"✗ Failed to import UNet2DConditionModel: {e}")
        return False
    
    try:
        from models.unet_3d import UNet3DConditionModel
        print("✓ UNet3DConditionModel imported")
    except ImportError as e:
        print(f"✗ Failed to import UNet3DConditionModel: {e}")
        return False
    
    try:
        from pipelines.pipeline_pose2img import Pose2ImagePipeline
        print("✓ Pose2ImagePipeline imported")
    except ImportError as e:
        print(f"✗ Failed to import Pose2ImagePipeline: {e}")
        return False
    
    try:
        from openpose import OpenposeDetector
        print("✓ OpenposeDetector imported")
    except ImportError as e:
        print(f"✗ Failed to import OpenposeDetector: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if models can be loaded"""
    print("\nTesting model loading...")
    
    base_path = Path(__file__).parent / "ModelTraining" / "pretrained_model"
    
    # Check model files
    models = {
        "denoising_unet.pth": base_path / "denoising_unet.pth",
        "reference_unet.pth": base_path / "reference_unet.pth",
        "motion_module.pth": base_path / "motion_module.pth",
    }
    
    for name, path in models.items():
        if path.exists():
            print(f"✓ {name} found at {path}")
            # Try to load it
            try:
                state_dict = torch.load(path, map_location="cpu")
                print(f"  Loaded successfully, keys: {len(state_dict.keys())}")
            except Exception as e:
                print(f"  Failed to load: {e}")
        else:
            print(f"✗ {name} not found at {path}")
    
    # Check HuggingFace models
    hf_models = [
        "sd-vae-ft-mse",
        "stable-diffusion-v1-5",
        "clip-vit-large-patch14"
    ]
    
    print("\nChecking HuggingFace models...")
    for model in hf_models:
        model_path = base_path / model
        if model_path.exists() and model_path.is_dir():
            files = list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
            print(f"✓ {model}: {len(files)} model files")
        else:
            print(f"✗ {model} not found")

def test_simple_pipeline():
    """Test creating a simple pipeline"""
    print("\nTesting pipeline creation...")
    
    try:
        from diffusers import AutoencoderKL, DDIMScheduler
        from transformers import CLIPVisionModelWithProjection
        
        # Test VAE loading
        vae_path = Path(__file__).parent / "ModelTraining" / "pretrained_model" / "sd-vae-ft-mse"
        if vae_path.exists():
            print("Attempting to load VAE...")
            vae = AutoencoderKL.from_pretrained(str(vae_path))
            print("✓ VAE loaded successfully")
        else:
            print(f"✗ VAE path not found: {vae_path}")
            
    except Exception as e:
        print(f"✗ Failed to create pipeline: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=" * 60)
    print("Sprite Sheet Diffusion Pipeline Test")
    print("=" * 60)
    
    # Check CUDA
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    print()
    
    # Run tests
    if test_imports():
        print("\n✓ All imports successful!")
    else:
        print("\n✗ Some imports failed")
    
    test_model_loading()
    test_simple_pipeline()
    
    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    main()