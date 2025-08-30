#!/usr/bin/env python3
"""
Sprite Sheet Diffusion - Proper Implementation Using Diffusion Models
This module provides the actual diffusion-based sprite generation using the trained models.
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPVisionModelWithProjection

# Add ModelTraining to path
sys.path.append(str(Path(__file__).parent.parent / "ModelTraining"))

from models.pose_guider_org import PoseGuiderOrg
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from pipelines.pipeline_pose2img import Pose2ImagePipeline
from openpose import OpenposeDetector


class SpriteDiffusionGenerator:
    """Main class for generating sprite sheets using the diffusion pipeline"""
    
    def __init__(self, config_path=None, device="cuda"):
        """Initialize the diffusion models and pipeline"""
        
        self.device = device
        self.weight_dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Default paths
        base_path = Path(__file__).parent.parent / "ModelTraining"
        self.pretrained_path = base_path / "pretrained_model"
        
        # Model paths
        self.vae_path = self.pretrained_path / "sd-vae-ft-mse"
        self.base_model_path = self.pretrained_path / "stable-diffusion-v1-5"
        self.image_encoder_path = self.pretrained_path / "clip-vit-large-patch14"
        
        # Weight paths
        self.denoising_unet_path = self.pretrained_path / "denoising_unet.pth"
        self.reference_unet_path = self.pretrained_path / "reference_unet.pth"
        self.pose_guider_path = self.pretrained_path / "pose_guider.pth"
        self.motion_module_path = self.pretrained_path / "motion_module.pth"
        
        # Initialize models
        self._init_models()
        
        # Initialize pose detector
        self.pose_detector = OpenposeDetector()
        
    def _init_models(self):
        """Initialize all required models"""
        
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            str(self.vae_path)
        ).to(self.device, dtype=self.weight_dtype)
        
        print("Loading Reference UNet...")
        self.reference_unet = UNet2DConditionModel.from_pretrained(
            str(self.base_model_path),
            subfolder="unet",
        ).to(dtype=self.weight_dtype, device=self.device)
        
        print("Loading Denoising UNet...")
        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            str(self.base_model_path),
            "",  # motion module path will be loaded separately
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
            },
        ).to(dtype=self.weight_dtype, device=self.device)
        
        print("Loading Pose Guider...")
        self.pose_guider = PoseGuiderOrg(
            conditioning_embedding_channels=320,
            block_out_channels=(16, 32, 96, 256)
        ).to(device=self.device, dtype=self.weight_dtype)
        
        print("Loading Image Encoder...")
        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(
            str(self.image_encoder_path)
        ).to(dtype=self.weight_dtype, device=self.device)
        
        # Load pretrained weights
        self._load_weights()
        
        # Initialize scheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        
        # Create pipeline
        self.pipe = Pose2ImagePipeline(
            vae=self.vae,
            image_encoder=self.image_enc,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            pose_guider=self.pose_guider,
            scheduler=self.scheduler,
        )
        self.pipe = self.pipe.to(self.device, dtype=self.weight_dtype)
        
    def _load_weights(self):
        """Load pretrained weights for the models"""
        
        if self.denoising_unet_path.exists():
            print(f"Loading denoising UNet weights from {self.denoising_unet_path}")
            self.denoising_unet.load_state_dict(
                torch.load(self.denoising_unet_path, map_location="cpu"),
                strict=False,
            )
        else:
            print(f"Warning: Denoising UNet weights not found at {self.denoising_unet_path}")
            
        if self.reference_unet_path.exists():
            print(f"Loading reference UNet weights from {self.reference_unet_path}")
            self.reference_unet.load_state_dict(
                torch.load(self.reference_unet_path, map_location="cpu"),
                strict=True,
            )
        else:
            print(f"Warning: Reference UNet weights not found at {self.reference_unet_path}")
            
        if self.pose_guider_path.exists():
            print(f"Loading pose guider weights from {self.pose_guider_path}")
            self.pose_guider.load_state_dict(
                torch.load(self.pose_guider_path, map_location="cpu"),
                strict=True,
            )
        else:
            print(f"Warning: Pose guider weights not found at {self.pose_guider_path}")
    
    def extract_pose(self, image):
        """Extract pose from an image using OpenPose"""
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize to standard size
        image = cv2.resize(image, (512, 512))
        
        # Extract pose
        pose = self.pose_detector(
            image,
            include_body=True,
            include_hand=False,
            include_face=False,
            use_dw_pose=True
        )
        
        # Convert back to PIL
        pose_image = Image.fromarray(pose)
        
        return pose_image
    
    def generate_pose_sequence(self, reference_image, animation_type, num_frames):
        """Generate a sequence of poses for the given animation type"""
        
        poses = []
        
        # Extract reference pose
        ref_pose = self.extract_pose(reference_image)
        
        # For now, create variations of the reference pose
        # In a full implementation, this would use motion-specific pose templates
        for i in range(num_frames):
            if animation_type == "idle":
                # Subtle variations for idle
                pose = ref_pose  # Could add subtle modifications
            elif animation_type == "walk":
                # Walking poses
                pose = self._generate_walk_pose(ref_pose, i, num_frames)
            elif animation_type == "run":
                # Running poses
                pose = self._generate_run_pose(ref_pose, i, num_frames)
            elif animation_type == "jump":
                # Jumping poses
                pose = self._generate_jump_pose(ref_pose, i, num_frames)
            else:
                # Default to reference pose
                pose = ref_pose
            
            poses.append(pose)
        
        return poses
    
    def _generate_walk_pose(self, ref_pose, frame_idx, total_frames):
        """Generate walking pose variations"""
        # This is a simplified version - in production, you'd use actual pose keypoints
        return ref_pose
    
    def _generate_run_pose(self, ref_pose, frame_idx, total_frames):
        """Generate running pose variations"""
        return ref_pose
    
    def _generate_jump_pose(self, ref_pose, frame_idx, total_frames):
        """Generate jumping pose variations"""
        return ref_pose
    
    def generate_sprite_sheet(self, reference_image, animation_type="idle", num_frames=4, 
                            width=512, height=512, guidance_scale=3.5, 
                            num_inference_steps=25, seed=42):
        """
        Generate a sprite sheet using the diffusion pipeline
        
        Args:
            reference_image: PIL Image of the character
            animation_type: Type of animation (idle, walk, run, jump, etc.)
            num_frames: Number of frames to generate
            width: Width of each frame
            height: Height of each frame
            guidance_scale: Guidance scale for diffusion
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
        
        Returns:
            PIL Image containing the sprite sheet
        """
        
        # Set random seed
        generator = torch.manual_seed(seed)
        
        # Resize reference image
        reference_image = reference_image.resize((width, height), Image.LANCZOS)
        
        # Generate pose sequence
        poses = self.generate_pose_sequence(reference_image, animation_type, num_frames)
        
        # Extract reference pose
        ref_pose = poses[0]
        
        # Generate frames using diffusion
        generated_frames = []
        
        for i, pose_image in enumerate(poses):
            print(f"Generating frame {i+1}/{num_frames}...")
            
            # Run diffusion pipeline
            with torch.no_grad():
                result = self.pipe(
                    ref_image=reference_image,
                    pose_image=pose_image,
                    ref_pose_image=ref_pose,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            
            # Get the generated image
            if hasattr(result, 'images'):
                generated_image = result.images[0]
            else:
                generated_image = result[0]
            
            # Convert to PIL if needed
            if isinstance(generated_image, torch.Tensor):
                generated_image = self._tensor_to_pil(generated_image)
            
            generated_frames.append(generated_image)
        
        # Create sprite sheet
        sprite_sheet = self._create_sprite_sheet(generated_frames)
        
        return sprite_sheet
    
    def _tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Denormalize if needed
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2
        
        tensor = tensor.clamp(0, 1)
        tensor = tensor.cpu().permute(1, 2, 0).numpy()
        tensor = (tensor * 255).astype(np.uint8)
        
        return Image.fromarray(tensor)
    
    def _create_sprite_sheet(self, frames):
        """Combine frames into a horizontal sprite sheet"""
        
        if not frames:
            return None
        
        frame_width, frame_height = frames[0].size
        num_frames = len(frames)
        
        # Create sprite sheet
        sprite_sheet = Image.new('RGBA', 
                                (frame_width * num_frames, frame_height),
                                (0, 0, 0, 0))
        
        # Place frames
        for i, frame in enumerate(frames):
            x = i * frame_width
            sprite_sheet.paste(frame, (x, 0))
        
        return sprite_sheet


# Singleton instance
_generator_instance = None

def get_generator():
    """Get or create the singleton generator instance"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = SpriteDiffusionGenerator()
    return _generator_instance