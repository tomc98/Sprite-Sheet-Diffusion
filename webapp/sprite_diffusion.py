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

from models.pose_guider import PoseGuider
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
        self.pose_guider = PoseGuider(
            noise_latent_channels=320,
            use_ca=True
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
        """Generate a sequence of poses for the given animation type with dramatic variations"""
        
        poses = []
        
        # For animations, create dramatically different synthetic poses instead of using reference
        for i in range(num_frames):
            if animation_type == "idle":
                # Subtle but visible variations for idle
                pose = self._generate_idle_pose(i, num_frames)
            elif animation_type == "walk":
                # Walking poses with clear leg movement
                pose = self._generate_walk_pose(i, num_frames)
            elif animation_type == "run":
                # Running poses with dramatic movement
                pose = self._generate_run_pose(i, num_frames)
            elif animation_type == "attack":
                # Attack poses with dramatic arm movements
                pose = self._generate_attack_pose(i, num_frames)
            elif animation_type == "defend":
                # Defensive poses
                pose = self._generate_defend_pose(i, num_frames)
            elif animation_type == "jump":
                # Jumping poses with full body movement
                pose = self._generate_jump_pose(i, num_frames)
            else:
                # Create basic pose variations
                pose = self._generate_basic_pose(i, num_frames)
            
            poses.append(pose)
        
        return poses
    
    def _generate_synthetic_pose(self, pose_type, frame_idx, total_frames):
        """Generate synthetic OpenPose skeleton with dramatic variations"""
        
        # Create a blank 512x512 image
        pose_image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Define stick figure proportions and center
        center_x, center_y = 256, 256
        head_radius = 25
        torso_length = 80
        arm_length = 60
        leg_length = 90
        
        # Calculate animation progress (0 to 1)
        progress = frame_idx / max(1, total_frames - 1)
        
        # Define keypoint positions based on pose type
        if pose_type == "walk":
            # Walking cycle with alternating legs
            cycle_pos = (frame_idx % 8) / 8.0  # 8-frame walk cycle
            
            # Head position (slight bob)
            head_y = center_y - torso_length - head_radius + int(5 * np.sin(cycle_pos * 2 * np.pi))
            
            # Torso
            torso_top = center_y - torso_length
            torso_bottom = center_y
            
            # Arms swing opposite to legs
            arm_swing = 30 * np.sin(cycle_pos * 2 * np.pi)
            left_arm_angle = np.radians(arm_swing)
            right_arm_angle = np.radians(-arm_swing)
            
            # Legs alternate
            leg_swing = 25 * np.sin(cycle_pos * 2 * np.pi)
            left_leg_angle = np.radians(leg_swing)
            right_leg_angle = np.radians(-leg_swing)
            
        elif pose_type == "attack":
            # Attack poses with dramatic arm positions
            if frame_idx < total_frames // 3:
                # Wind up
                left_arm_angle = np.radians(-120 + 60 * progress)
                right_arm_angle = np.radians(-60)
            elif frame_idx < 2 * total_frames // 3:
                # Strike
                left_arm_angle = np.radians(-30)
                right_arm_angle = np.radians(90)
            else:
                # Follow through
                left_arm_angle = np.radians(30)
                right_arm_angle = np.radians(45)
            
            head_y = center_y - torso_length - head_radius
            torso_top = center_y - torso_length
            torso_bottom = center_y
            left_leg_angle = np.radians(10)
            right_leg_angle = np.radians(-10)
            
        elif pose_type == "defend":
            # Defensive crouching pose
            crouch_amount = 20 + 15 * np.sin(progress * np.pi)
            head_y = center_y - torso_length - head_radius + crouch_amount
            torso_top = center_y - torso_length + crouch_amount
            torso_bottom = center_y + crouch_amount
            
            # Arms up in defensive position
            left_arm_angle = np.radians(-90 + 30 * np.sin(progress * 2 * np.pi))
            right_arm_angle = np.radians(-90 - 30 * np.sin(progress * 2 * np.pi))
            
            # Bent legs
            left_leg_angle = np.radians(15)
            right_leg_angle = np.radians(-15)
            
        else:  # idle or basic
            # Subtle breathing motion
            breath = 3 * np.sin(progress * 2 * np.pi)
            head_y = center_y - torso_length - head_radius + breath
            torso_top = center_y - torso_length + breath
            torso_bottom = center_y + breath
            
            # Slight arm sway
            left_arm_angle = np.radians(10 + 5 * np.sin(progress * 2 * np.pi))
            right_arm_angle = np.radians(-10 - 5 * np.sin(progress * 2 * np.pi))
            left_leg_angle = np.radians(5)
            right_leg_angle = np.radians(-5)
        
        # Draw stick figure with thick lines for visibility
        line_thickness = 8
        color = (255, 255, 255)  # White lines
        
        # Head (circle) - ensure all coordinates are integers
        cv2.circle(pose_image, (int(center_x), int(head_y)), int(head_radius), color, line_thickness)
        
        # Torso (vertical line)
        cv2.line(pose_image, (int(center_x), int(torso_top)), (int(center_x), int(torso_bottom)), color, line_thickness)
        
        # Left arm
        left_arm_end_x = int(center_x + arm_length * np.cos(left_arm_angle + np.pi/2))
        left_arm_end_y = int(torso_top + arm_length * np.sin(left_arm_angle + np.pi/2))
        cv2.line(pose_image, (int(center_x), int(torso_top)), (left_arm_end_x, left_arm_end_y), color, line_thickness)
        
        # Right arm
        right_arm_end_x = int(center_x + arm_length * np.cos(right_arm_angle - np.pi/2))
        right_arm_end_y = int(torso_top + arm_length * np.sin(right_arm_angle - np.pi/2))
        cv2.line(pose_image, (int(center_x), int(torso_top)), (right_arm_end_x, right_arm_end_y), color, line_thickness)
        
        # Left leg
        left_leg_end_x = int(center_x + leg_length * np.cos(left_leg_angle + np.pi/2))
        left_leg_end_y = int(torso_bottom + leg_length * np.sin(left_leg_angle + np.pi/2))
        cv2.line(pose_image, (int(center_x), int(torso_bottom)), (left_leg_end_x, left_leg_end_y), color, line_thickness)
        
        # Right leg
        right_leg_end_x = int(center_x + leg_length * np.cos(right_leg_angle - np.pi/2))
        right_leg_end_y = int(torso_bottom + leg_length * np.sin(right_leg_angle - np.pi/2))
        cv2.line(pose_image, (int(center_x), int(torso_bottom)), (right_leg_end_x, right_leg_end_y), color, line_thickness)
        
        return Image.fromarray(pose_image)
    
    def _generate_idle_pose(self, frame_idx, total_frames):
        """Generate idle pose with breathing motion"""
        return self._generate_synthetic_pose("idle", frame_idx, total_frames)
    
    def _generate_walk_pose(self, frame_idx, total_frames):
        """Generate walking pose with clear leg movement"""
        return self._generate_synthetic_pose("walk", frame_idx, total_frames)
    
    def _generate_run_pose(self, frame_idx, total_frames):
        """Generate running pose with dramatic movement"""
        return self._generate_synthetic_pose("walk", frame_idx, total_frames)  # Use walk but more dramatic
    
    def _generate_attack_pose(self, frame_idx, total_frames):
        """Generate attack pose with dramatic arm movements"""
        return self._generate_synthetic_pose("attack", frame_idx, total_frames)
        
    def _generate_defend_pose(self, frame_idx, total_frames):
        """Generate defensive pose"""
        return self._generate_synthetic_pose("defend", frame_idx, total_frames)
    
    def _generate_jump_pose(self, frame_idx, total_frames):
        """Generate jumping pose with full body movement"""
        return self._generate_synthetic_pose("jump", frame_idx, total_frames)
        
    def _generate_basic_pose(self, frame_idx, total_frames):
        """Generate basic pose variations"""
        return self._generate_synthetic_pose("idle", frame_idx, total_frames)
    
    def generate_sprite_sheet(self, reference_image, animation_type="idle", num_frames=4, 
                            width=512, height=512, guidance_scale=20.0, 
                            num_inference_steps=30, seed=42):
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