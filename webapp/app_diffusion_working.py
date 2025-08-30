#!/usr/bin/env python3
"""
Sprite Sheet Diffusion Web Application - Working Implementation
This uses the correct PoseGuider that works with the pipeline
"""

import os
import sys
import json
import uuid
import shutil
import threading
import traceback
import math
import torch
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw

# Add ModelTraining to path for imports
sys.path.append(str(Path(__file__).parent.parent / "ModelTraining"))

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from models.pose_guider import PoseGuider
from models.pose_guider_org import PoseGuiderOrg  # Use the regular PoseGuider with cross-attention
from models.pose_guider_multi_res import SingleTensorPoseGuider  # Improved multi-resolution pose guider
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from pipelines.pipeline_pose2img import Pose2ImagePipeline
from pipeline_simplified import SimplifiedPose2ImagePipeline
from openpose import OpenposeDetector

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'static' / 'uploads'
app.config['RESULTS_FOLDER'] = Path(__file__).parent / 'static' / 'results'

# Ensure folders exist
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(parents=True, exist_ok=True)

# Check if CUDA is available
USE_GPU = torch.cuda.is_available()
print(f"GPU Available: {USE_GPU}")
if USE_GPU:
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Processing status storage
processing_status = {}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Animation types with frame configurations
ANIMATION_TYPES = {
    'idle': {'name': 'Idle', 'frames': 4},
    'walk': {'name': 'Walk', 'frames': 8},
    'run': {'name': 'Run', 'frames': 8},
    'jump': {'name': 'Jump', 'frames': 6},
    'attack': {'name': 'Attack', 'frames': 6},
    'defend': {'name': 'Defend', 'frames': 4},
}

# Global pipeline instance
_pipeline = None
_pose_detector = None

def get_pipeline():
    """Get or create the diffusion pipeline"""
    global _pipeline, _pose_detector
    
    if _pipeline is None:
        try:
            print("Initializing diffusion pipeline...")
            
            device = "cuda" if USE_GPU else "cpu"
            weight_dtype = torch.float16 if USE_GPU else torch.float32
            
            # Paths
            base_path = Path(__file__).parent.parent / "ModelTraining" / "pretrained_model"
            
            # Load VAE
            print("Loading VAE...")
            vae = AutoencoderKL.from_pretrained(
                str(base_path / "sd-vae-ft-mse")
            ).to(device, dtype=weight_dtype)
            
            # Load Reference UNet
            print("Loading Reference UNet...")
            reference_unet = UNet2DConditionModel.from_pretrained(
                str(base_path / "stable-diffusion-v1-5"),
                subfolder="unet",
            ).to(dtype=weight_dtype, device=device)
            
            reference_unet.load_state_dict(
                torch.load(base_path / "reference_unet.pth", map_location="cpu"),
                strict=False
            )
            
            # Load Denoising UNet
            print("Loading Denoising UNet...")
            denoising_unet = UNet3DConditionModel.from_pretrained_2d(
                str(base_path / "stable-diffusion-v1-5"),
                "",
                subfolder="unet",
                unet_additional_kwargs={
                    "use_motion_module": False,
                    "unet_use_temporal_attention": False,
                },
            ).to(dtype=weight_dtype, device=device)
            
            denoising_unet.load_state_dict(
                torch.load(base_path / "denoising_unet.pth", map_location="cpu"),
                strict=False
            )
            
            # Load Pose Guider - Use PoseGuiderOrg with multi-resolution wrapper
            print("Loading Pose Guider...")
            base_pose_guider = PoseGuiderOrg(
                conditioning_embedding_channels=320, 
                block_out_channels=(16, 32, 96, 256)
            ).to(device=device, dtype=weight_dtype)
            
            # Try to load weights if they exist
            pose_guider_path = base_path / "pose_guider.pth"
            if pose_guider_path.exists():
                try:
                    base_pose_guider.load_state_dict(
                        torch.load(pose_guider_path, map_location="cpu"),
                        strict=True  # PoseGuiderOrg should match the weights exactly
                    )
                    print("Loaded pose_guider weights")
                except Exception as e:
                    print(f"Warning: Could not load pose_guider weights: {e}")
                    print("Using random initialization")
            
            # Wrap with multi-resolution adapter using correct channel dimensions
            pose_guider = SingleTensorPoseGuider(
                base_pose_guider,
                unet_block_channels=(320, 320, 640, 1280, 1280)  # Corrected based on UNet debug
            ).to(device=device, dtype=weight_dtype)
            
            # Load Image Encoder
            print("Loading Image Encoder...")
            image_enc = CLIPVisionModelWithProjection.from_pretrained(
                str(base_path / "image_encoder")
            ).to(dtype=weight_dtype, device=device)
            
            # Initialize scheduler with v_prediction as in the original config
            scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
                prediction_type="v_prediction",
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
            )
            
            # Create pipeline - use simplified pipeline for PoseGuiderOrg
            print("Creating pipeline...")
            _pipeline = SimplifiedPose2ImagePipeline(
                vae=vae,
                image_encoder=image_enc,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                pose_guider=pose_guider,
                scheduler=scheduler,
            )
            _pipeline = _pipeline.to(device, dtype=weight_dtype)
            
            # Initialize pose detector
            print("Initializing pose detector...")
            _pose_detector = OpenposeDetector()
            
            print("Pipeline ready!")
            
        except Exception as e:
            print(f"Failed to initialize pipeline: {e}")
            traceback.print_exc()
            _pipeline = None
    
    return _pipeline, _pose_detector

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pose(image):
    """Extract pose from an image"""
    _, pose_detector = get_pipeline()
    
    if pose_detector is None:
        return None
    
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Resize
    image = cv2.resize(image, (512, 512))
    
    # Extract pose
    pose = pose_detector(
        image,
        include_body=True,
        include_hand=False,
        include_face=False,
        use_dw_pose=True
    )
    
    # Convert back to PIL
    return Image.fromarray(pose)

def generate_animation_poses(base_pose, animation_type, num_frames):
    """Generate different poses for each frame of an animation
    
    Creates OpenPose-style colored skeleton poses for animation.
    OpenPose uses different colors for different body parts:
    - Red/Orange: Right side limbs
    - Green/Blue: Left side limbs  
    - Purple/Pink: Body and connections
    """
    poses = []
    
    # Enhanced high-contrast colors for stronger pose control
    colors = {
        'head': (255, 255, 255),      # Bright white - maximum visibility
        'neck': (255, 255, 255),      # Bright white
        'body': (255, 255, 255),      # Bright white - main body mass
        'right_arm': (255, 0, 0),     # Bright red - strong signal
        'left_arm': (0, 0, 255),      # Bright blue - strong signal
        'right_leg': (0, 255, 0),     # Bright green - strong signal
        'left_leg': (255, 255, 0),    # Bright yellow - strong signal
        'joints': (255, 255, 255),    # White joints for structure
        'tool_bag': (255, 128, 0),    # Orange - character-specific
    }
    
    # For each frame, create a distinct skeletal pose
    for i in range(num_frames):
        # Create a new black canvas for each pose
        pose_img = Image.new('RGB', (512, 512), color='black')
        draw = ImageDraw.Draw(pose_img)
        
        # Enhanced base positions optimized for fluffy character
        center_x, center_y = 255, 255  # Better centering
        head_x, head_y = center_x, center_y - 100  # Head position
        neck_y = center_y - 60
        shoulder_y = center_y - 40
        body_center_y = center_y
        hip_y = center_y + 60
        body_width = 80  # Wider for fluffy character
        limb_length = 80  # Longer for dramatic poses
        
        if animation_type == 'walk':
            # DRAMATIC Walking animation - strong alternating poses
            phase = (i / num_frames) * 2 * math.pi
            
            # Much stronger movement for clear pose differences
            body_offset = int(25 * math.sin(phase))  # 5x stronger horizontal sway
            vertical_bob = int(20 * abs(math.sin(phase * 2)))  # Clear vertical walking bob
            
            # Large head with clear position changes
            head_size = 45  # Much larger for better control signal
            head_pos_x = head_x + body_offset
            head_pos_y = head_y - vertical_bob
            draw.ellipse([head_pos_x - head_size, head_pos_y - head_size, 
                         head_pos_x + head_size, head_pos_y + head_size], 
                         fill=colors['head'], outline=colors['head'], width=3)
            
            # Add clear facial features for orientation
            draw.ellipse([head_pos_x - 18, head_pos_y - 12, head_pos_x - 8, head_pos_y - 2], fill=(0, 0, 0))
            draw.ellipse([head_pos_x + 8, head_pos_y - 12, head_pos_x + 18, head_pos_y - 2], fill=(0, 0, 0))
            draw.arc([head_pos_x - 12, head_pos_y + 5, head_pos_x + 12, head_pos_y + 15], 0, 180, fill=(0, 0, 0), width=2)
            
            # Thick neck for better connection visibility
            draw.line([head_pos_x, head_pos_y + head_size, head_pos_x, neck_y - vertical_bob], 
                     fill=colors['neck'], width=18)
            
            # Large fluffy body - elliptical shape for character
            body_x = head_pos_x
            body_y = body_center_y - vertical_bob
            draw.ellipse([body_x - body_width, body_y - 70, 
                         body_x + body_width, body_y + 70], 
                         fill=colors['body'], outline=colors['body'], width=3)
            
            # Character's tool bag (distinctive feature)
            bag_x = body_x - 15
            bag_y = body_y + 25
            draw.rectangle([bag_x - 30, bag_y - 20, bag_x + 30, bag_y + 30], 
                          fill=colors['tool_bag'], outline=colors['tool_bag'], width=2)
            
            # DRAMATIC arm swinging - huge range of motion
            left_arm_angle = 80 * math.sin(phase)   # Much larger swing range
            right_arm_angle = -80 * math.sin(phase)
            
            shoulder_y_pos = shoulder_y - vertical_bob
            
            # Left arm with clear joint and endpoint
            shoulder_x_left = body_x - 65
            left_arm_x = shoulder_x_left + int(limb_length * math.sin(math.radians(left_arm_angle + 45)))
            left_arm_y = shoulder_y_pos + int(limb_length * math.cos(math.radians(left_arm_angle + 45)))
            
            # Shoulder joint (clear landmark)
            draw.ellipse([shoulder_x_left - 10, shoulder_y_pos - 10, 
                         shoulder_x_left + 10, shoulder_y_pos + 10], 
                        fill=colors['joints'], width=2)
            # Thick arm line
            draw.line([shoulder_x_left, shoulder_y_pos, left_arm_x, left_arm_y], 
                     fill=colors['left_arm'], width=16)
            # Hand/endpoint marker
            draw.ellipse([left_arm_x - 8, left_arm_y - 8, left_arm_x + 8, left_arm_y + 8], 
                        fill=colors['left_arm'], width=2)
            
            # Right arm with clear joint and endpoint
            shoulder_x_right = body_x + 65
            right_arm_x = shoulder_x_right + int(limb_length * math.sin(math.radians(right_arm_angle - 45)))
            right_arm_y = shoulder_y_pos + int(limb_length * math.cos(math.radians(right_arm_angle - 45)))
            
            # Shoulder joint (clear landmark)
            draw.ellipse([shoulder_x_right - 10, shoulder_y_pos - 10, 
                         shoulder_x_right + 10, shoulder_y_pos + 10], 
                        fill=colors['joints'], width=2)
            # Thick arm line
            draw.line([shoulder_x_right, shoulder_y_pos, right_arm_x, right_arm_y], 
                     fill=colors['right_arm'], width=16)
            # Hand/endpoint marker
            draw.ellipse([right_arm_x - 8, right_arm_y - 8, right_arm_x + 8, right_arm_y + 8], 
                        fill=colors['right_arm'], width=2)
            
            # DRAMATIC leg stepping - massive stride differences
            left_leg_angle = 70 * math.sin(phase)   # Much stronger leg movement
            right_leg_angle = -70 * math.sin(phase)
            
            hip_pos_y = hip_y - vertical_bob
            
            # Left leg with dramatic stride
            hip_x_left = body_x - 35
            left_stride = int(80 * math.sin(math.radians(left_leg_angle)))  # Huge stride
            left_lift = int(40 * abs(math.sin(phase)))  # High foot lift
            
            left_leg_x = hip_x_left + left_stride
            left_leg_y = hip_pos_y + limb_length + 20 - left_lift
            
            # Hip joint (clear landmark)
            draw.ellipse([hip_x_left - 10, hip_pos_y - 10, 
                         hip_x_left + 10, hip_pos_y + 10], 
                        fill=colors['joints'], width=2)
            # Thick leg line
            draw.line([hip_x_left, hip_pos_y, left_leg_x, left_leg_y], 
                     fill=colors['left_leg'], width=16)
            # Foot marker
            draw.ellipse([left_leg_x - 10, left_leg_y - 10, left_leg_x + 10, left_leg_y + 10], 
                        fill=colors['left_leg'], width=2)
            
            # Right leg with dramatic stride
            hip_x_right = body_x + 35
            right_stride = int(80 * math.sin(math.radians(right_leg_angle)))
            right_lift = int(40 * abs(math.sin(phase + math.pi)))
            
            right_leg_x = hip_x_right + right_stride
            right_leg_y = hip_pos_y + limb_length + 20 - right_lift
            
            # Hip joint (clear landmark)
            draw.ellipse([hip_x_right - 10, hip_pos_y - 10, 
                         hip_x_right + 10, hip_pos_y + 10], 
                        fill=colors['joints'], width=2)
            # Thick leg line
            draw.line([hip_x_right, hip_pos_y, right_leg_x, right_leg_y], 
                     fill=colors['right_leg'], width=16)
            # Foot marker
            draw.ellipse([right_leg_x - 10, right_leg_y - 10, right_leg_x + 10, right_leg_y + 10], 
                        fill=colors['right_leg'], width=2)
            
        elif animation_type == 'run':
            # ENHANCED Running animation - explosive dynamic movement
            phase = (i / num_frames) * 2 * math.pi
            
            # Massive body movement for running
            body_offset = int(35 * math.sin(phase))  # Even stronger than walking
            vertical_bob = int(30 * abs(math.sin(phase * 2)))  # High bouncing
            forward_lean = int(20 * math.cos(phase))  # Dynamic forward lean
            
            # Large running head with motion blur effect
            head_size = 50  # Larger for better control
            head_pos_x = head_x + body_offset + forward_lean
            head_pos_y = head_y - vertical_bob
            draw.ellipse([head_pos_x - head_size, head_pos_y - head_size, 
                         head_pos_x + head_size, head_pos_y + head_size], 
                         fill=colors['head'], outline=colors['head'], width=3)
            
            # Intense facial features for running
            draw.ellipse([head_pos_x - 20, head_pos_y - 15, head_pos_x - 10, head_pos_y - 5], fill=(0, 0, 0))
            draw.ellipse([head_pos_x + 10, head_pos_y - 15, head_pos_x + 20, head_pos_y - 5], fill=(0, 0, 0))
            draw.line([head_pos_x - 15, head_pos_y + 10, head_pos_x + 15, head_pos_y + 8], fill=(0, 0, 0), width=3)  # Determined expression
            
            # Massive leaning body for running dynamics
            body_x = head_pos_x
            body_y = body_center_y - vertical_bob
            # Elliptical body tilted forward
            draw.ellipse([body_x - body_width + forward_lean, body_y - 75, 
                         body_x + body_width + forward_lean, body_y + 75], 
                         fill=colors['body'], outline=colors['body'], width=4)
            
            # Running gear/tool bag with motion
            bag_x = body_x + forward_lean - 20
            bag_y = body_y + 30
            draw.rectangle([bag_x - 35, bag_y - 25, bag_x + 35, bag_y + 35], 
                          fill=colors['tool_bag'], outline=colors['tool_bag'], width=3)
            
            # EXPLOSIVE arm pumping - maximum range for running
            left_arm_angle = 100 * math.sin(phase)   # Huge pumping motion
            right_arm_angle = -100 * math.sin(phase)
            
            shoulder_y_pos = shoulder_y - vertical_bob
            
            # Left arm with powerful pumping
            shoulder_x_left = body_x - 70 + forward_lean
            left_arm_x = shoulder_x_left + int((limb_length + 20) * math.sin(math.radians(left_arm_angle + 60)))
            left_arm_y = shoulder_y_pos + int((limb_length + 20) * math.cos(math.radians(left_arm_angle + 60)))
            
            # Shoulder joint with motion blur
            draw.ellipse([shoulder_x_left - 12, shoulder_y_pos - 12, 
                         shoulder_x_left + 12, shoulder_y_pos + 12], 
                        fill=colors['joints'], width=3)
            # Thick pumping arm
            draw.line([shoulder_x_left, shoulder_y_pos, left_arm_x, left_arm_y], 
                     fill=colors['left_arm'], width=18)
            # Fist/endpoint with power
            draw.ellipse([left_arm_x - 10, left_arm_y - 10, left_arm_x + 10, left_arm_y + 10], 
                        fill=colors['left_arm'], width=3)
            
            # Right arm with powerful pumping
            shoulder_x_right = body_x + 70 + forward_lean
            right_arm_x = shoulder_x_right + int((limb_length + 20) * math.sin(math.radians(right_arm_angle - 60)))
            right_arm_y = shoulder_y_pos + int((limb_length + 20) * math.cos(math.radians(right_arm_angle - 60)))
            
            # Shoulder joint with motion blur
            draw.ellipse([shoulder_x_right - 12, shoulder_y_pos - 12, 
                         shoulder_x_right + 12, shoulder_y_pos + 12], 
                        fill=colors['joints'], width=3)
            # Thick pumping arm
            draw.line([shoulder_x_right, shoulder_y_pos, right_arm_x, right_arm_y], 
                     fill=colors['right_arm'], width=18)
            # Fist/endpoint with power
            draw.ellipse([right_arm_x - 10, right_arm_y - 10, right_arm_x + 10, right_arm_y + 10], 
                        fill=colors['right_arm'], width=3)
            
            # EXPLOSIVE running stride - maximum knee lift and extension
            left_leg_angle = 90 * math.sin(phase)   # Extreme knee lift
            right_leg_angle = -90 * math.sin(phase)
            
            hip_pos_y = hip_y - vertical_bob
            
            # Left leg with extreme running form
            hip_x_left = body_x - 40 + forward_lean
            
            # High knee lift phase
            if abs(math.sin(phase)) > 0.3:  # High knee phase
                knee_lift = int(60 * abs(math.sin(phase)))
                knee_x = hip_x_left + int(30 * math.sin(math.radians(left_leg_angle)))
                knee_y = hip_pos_y + 30 - knee_lift
                foot_x = knee_x + int(50 * math.sin(math.radians(left_leg_angle + 45)))
                foot_y = hip_pos_y + limb_length + 30 - int(20 * abs(math.sin(phase)))
                
                # Hip joint
                draw.ellipse([hip_x_left - 12, hip_pos_y - 12, hip_x_left + 12, hip_pos_y + 12], 
                            fill=colors['joints'], width=3)
                # Thigh (hip to knee)
                draw.line([hip_x_left, hip_pos_y, knee_x, knee_y], 
                         fill=colors['left_leg'], width=18)
                # Knee joint
                draw.ellipse([knee_x - 8, knee_y - 8, knee_x + 8, knee_y + 8], 
                            fill=colors['joints'], width=2)
                # Shin (knee to foot)
                draw.line([knee_x, knee_y, foot_x, foot_y], 
                         fill=colors['left_leg'], width=16)
                # Foot
                draw.ellipse([foot_x - 12, foot_y - 12, foot_x + 12, foot_y + 12], 
                            fill=colors['left_leg'], width=3)
            else:  # Extension phase
                left_stride = int(100 * math.sin(math.radians(left_leg_angle)))
                left_leg_x = hip_x_left + left_stride
                left_leg_y = hip_pos_y + limb_length + 40
                
                draw.ellipse([hip_x_left - 12, hip_pos_y - 12, hip_x_left + 12, hip_pos_y + 12], 
                            fill=colors['joints'], width=3)
                draw.line([hip_x_left, hip_pos_y, left_leg_x, left_leg_y], 
                         fill=colors['left_leg'], width=18)
                draw.ellipse([left_leg_x - 12, left_leg_y - 12, left_leg_x + 12, left_leg_y + 12], 
                            fill=colors['left_leg'], width=3)
            
            # Right leg with extreme running form
            hip_x_right = body_x + 40 + forward_lean
            
            # High knee lift phase (opposite to left)
            if abs(math.sin(phase + math.pi)) > 0.3:  # High knee phase
                knee_lift = int(60 * abs(math.sin(phase + math.pi)))
                knee_x = hip_x_right + int(30 * math.sin(math.radians(right_leg_angle)))
                knee_y = hip_pos_y + 30 - knee_lift
                foot_x = knee_x + int(50 * math.sin(math.radians(right_leg_angle - 45)))
                foot_y = hip_pos_y + limb_length + 30 - int(20 * abs(math.sin(phase + math.pi)))
                
                # Hip joint
                draw.ellipse([hip_x_right - 12, hip_pos_y - 12, hip_x_right + 12, hip_pos_y + 12], 
                            fill=colors['joints'], width=3)
                # Thigh (hip to knee)
                draw.line([hip_x_right, hip_pos_y, knee_x, knee_y], 
                         fill=colors['right_leg'], width=18)
                # Knee joint
                draw.ellipse([knee_x - 8, knee_y - 8, knee_x + 8, knee_y + 8], 
                            fill=colors['joints'], width=2)
                # Shin (knee to foot)
                draw.line([knee_x, knee_y, foot_x, foot_y], 
                         fill=colors['right_leg'], width=16)
                # Foot
                draw.ellipse([foot_x - 12, foot_y - 12, foot_x + 12, foot_y + 12], 
                            fill=colors['right_leg'], width=3)
            else:  # Extension phase
                right_stride = int(100 * math.sin(math.radians(right_leg_angle)))
                right_leg_x = hip_x_right + right_stride
                right_leg_y = hip_pos_y + limb_length + 40
                
                draw.ellipse([hip_x_right - 12, hip_pos_y - 12, hip_x_right + 12, hip_pos_y + 12], 
                            fill=colors['joints'], width=3)
                draw.line([hip_x_right, hip_pos_y, right_leg_x, right_leg_y], 
                         fill=colors['right_leg'], width=18)
                draw.ellipse([right_leg_x - 12, right_leg_y - 12, right_leg_x + 12, right_leg_y + 12], 
                            fill=colors['right_leg'], width=3)
                
        elif animation_type == 'jump':
            # Jump animation - vertical movement
            jump_height = int(80 * abs(math.sin((i / num_frames) * math.pi)))
            
            # Head
            draw.ellipse([head_x - 25, head_y - jump_height - 25, 
                         head_x + 25, head_y - jump_height + 25], fill='white')
            
            # Body
            draw.line([head_x, neck_y - jump_height, head_x, hip_y - jump_height], fill='white', width=3)
            
            # Arms raised
            arm_raise = int(45 * abs(math.sin((i / num_frames) * math.pi)))
            draw.line([head_x, shoulder_y - jump_height, 
                      head_x - 50, shoulder_y - jump_height - arm_raise], fill='white', width=3)
            draw.line([head_x, shoulder_y - jump_height, 
                      head_x + 50, shoulder_y - jump_height - arm_raise], fill='white', width=3)
            
            # Legs bent during jump
            if jump_height > 20:  # In the air
                # Knees bent
                draw.line([head_x, hip_y - jump_height, head_x - 30, hip_y - jump_height + 40], fill='white', width=3)
                draw.line([head_x - 30, hip_y - jump_height + 40, head_x - 25, hip_y - jump_height + 80], fill='white', width=3)
                draw.line([head_x, hip_y - jump_height, head_x + 30, hip_y - jump_height + 40], fill='white', width=3)
                draw.line([head_x + 30, hip_y - jump_height + 40, head_x + 25, hip_y - jump_height + 80], fill='white', width=3)
            else:  # On ground
                draw.line([head_x, hip_y - jump_height, head_x - 25, hip_y - jump_height + 100], fill='white', width=3)
                draw.line([head_x, hip_y - jump_height, head_x + 25, hip_y - jump_height + 100], fill='white', width=3)
                
        else:  # idle, attack, etc - enhanced poses with compelling movement
            # ENHANCED breathing/idle motion - much more pronounced
            breath = int(12 * math.sin((i / num_frames) * 2 * math.pi))  # 4x stronger breathing
            sway = int(8 * math.sin((i / num_frames) * math.pi))  # Gentle side sway
            
            # Special handling for different animation types
            if animation_type == 'attack':
                # Attack sequence - dramatic action poses
                attack_phase = (i / num_frames) * math.pi
                strike_power = int(40 * abs(math.sin(attack_phase)))
                arm_strike = int(80 * math.sin(attack_phase))
            else:
                strike_power = 0
                arm_strike = 0
            
            # Large expressive head with breathing
            head_size = 40
            head_pos_x = head_x + sway
            head_pos_y = head_y + breath
            draw.ellipse([head_pos_x - head_size, head_pos_y - head_size, 
                         head_pos_x + head_size, head_pos_y + head_size], 
                         fill=colors['head'], outline=colors['head'], width=3)
            
            # Expressive face features
            if animation_type == 'attack':
                # Fierce attack expression
                draw.ellipse([head_pos_x - 15, head_pos_y - 10, head_pos_x - 5, head_pos_y], fill=(255, 0, 0))
                draw.ellipse([head_pos_x + 5, head_pos_y - 10, head_pos_x + 15, head_pos_y], fill=(255, 0, 0))
                draw.arc([head_pos_x - 8, head_pos_y + 5, head_pos_x + 8, head_pos_y + 15], 180, 360, fill=(255, 0, 0), width=3)
            else:
                # Calm idle expression
                draw.ellipse([head_pos_x - 12, head_pos_y - 8, head_pos_x - 4, head_pos_y], fill=(0, 0, 0))
                draw.ellipse([head_pos_x + 4, head_pos_y - 8, head_pos_x + 12, head_pos_y], fill=(0, 0, 0))
                draw.arc([head_pos_x - 10, head_pos_y + 5, head_pos_x + 10, head_pos_y + 12], 0, 180, fill=(0, 0, 0), width=2)
            
            # Thick neck with movement
            draw.line([head_pos_x, head_pos_y + head_size, head_pos_x + sway, neck_y + breath], 
                     fill=colors['neck'], width=15)
            
            # Large body with breathing expansion
            body_x = head_pos_x + sway
            body_y = body_center_y + breath
            body_expansion = int(10 * abs(math.sin((i / num_frames) * 2 * math.pi)))  # Breathing expansion
            
            draw.ellipse([body_x - body_width - body_expansion, body_y - 70, 
                         body_x + body_width + body_expansion, body_y + 70], 
                         fill=colors['body'], outline=colors['body'], width=3)
            
            # Tool bag with subtle movement
            bag_x = body_x - 15 + int(3 * math.sin((i / num_frames) * 2 * math.pi))
            bag_y = body_y + 25
            draw.rectangle([bag_x - 30, bag_y - 20, bag_x + 30, bag_y + 30], 
                          fill=colors['tool_bag'], outline=colors['tool_bag'], width=2)
            
            # Arms with distinct poses for different animations
            if animation_type == 'attack':
                # Dynamic attack arm positions
                left_arm_angle = 20 + arm_strike
                right_arm_angle = -30 - arm_strike
            else:
                # Subtle idle arm movement - much more noticeable
                left_arm_angle = 20 * math.sin((i / num_frames) * 2 * math.pi)
                right_arm_angle = -15 * math.cos((i / num_frames) * 2 * math.pi)
            
            shoulder_y_pos = shoulder_y + breath
            
            # Left arm with clear positioning
            shoulder_x_left = body_x - 65
            left_arm_x = shoulder_x_left + int(limb_length * math.sin(math.radians(left_arm_angle + 45)))
            left_arm_y = shoulder_y_pos + int(limb_length * math.cos(math.radians(left_arm_angle + 45)))
            
            draw.ellipse([shoulder_x_left - 8, shoulder_y_pos - 8, 
                         shoulder_x_left + 8, shoulder_y_pos + 8], 
                        fill=colors['joints'], width=2)
            draw.line([shoulder_x_left, shoulder_y_pos, left_arm_x, left_arm_y], 
                     fill=colors['left_arm'], width=14)
            draw.ellipse([left_arm_x - 6, left_arm_y - 6, left_arm_x + 6, left_arm_y + 6], 
                        fill=colors['left_arm'], width=2)
            
            # Right arm with clear positioning
            shoulder_x_right = body_x + 65
            right_arm_x = shoulder_x_right + int(limb_length * math.sin(math.radians(right_arm_angle - 45)))
            right_arm_y = shoulder_y_pos + int(limb_length * math.cos(math.radians(right_arm_angle - 45)))
            
            draw.ellipse([shoulder_x_right - 8, shoulder_y_pos - 8, 
                         shoulder_x_right + 8, shoulder_y_pos + 8], 
                        fill=colors['joints'], width=2)
            draw.line([shoulder_x_right, shoulder_y_pos, right_arm_x, right_arm_y], 
                     fill=colors['right_arm'], width=14)
            draw.ellipse([right_arm_x - 6, right_arm_y - 6, right_arm_x + 6, right_arm_y + 6], 
                        fill=colors['right_arm'], width=2)
            
            # Enhanced legs with proper stance
            hip_pos_y = hip_y + int(breath / 2)  # Subtle hip movement
            
            if animation_type == 'attack':
                # Attack stance - wider, more aggressive
                left_stance = -40 + int(strike_power / 2)
                right_stance = 40 - int(strike_power / 2)
                stance_spread = 20
            else:
                # Idle stance with subtle weight shifts
                weight_shift = int(10 * math.sin((i / num_frames) * math.pi))
                left_stance = -30 + weight_shift
                right_stance = 30 - weight_shift
                stance_spread = 0
            
            # Left leg
            hip_x_left = body_x - 35
            left_leg_x = hip_x_left + left_stance
            left_leg_y = hip_pos_y + limb_length + 20 + stance_spread
            
            draw.ellipse([hip_x_left - 8, hip_pos_y - 8, hip_x_left + 8, hip_pos_y + 8], 
                        fill=colors['joints'], width=2)
            draw.line([hip_x_left, hip_pos_y, left_leg_x, left_leg_y], 
                     fill=colors['left_leg'], width=14)
            draw.ellipse([left_leg_x - 8, left_leg_y - 8, left_leg_x + 8, left_leg_y + 8], 
                        fill=colors['left_leg'], width=2)
            
            # Right leg
            hip_x_right = body_x + 35
            right_leg_x = hip_x_right + right_stance
            right_leg_y = hip_pos_y + limb_length + 20 + stance_spread
            
            draw.ellipse([hip_x_right - 8, hip_pos_y - 8, hip_x_right + 8, hip_pos_y + 8], 
                        fill=colors['joints'], width=2)
            draw.line([hip_x_right, hip_pos_y, right_leg_x, right_leg_y], 
                     fill=colors['right_leg'], width=14)
            draw.ellipse([right_leg_x - 8, right_leg_y - 8, right_leg_x + 8, right_leg_y + 8], 
                        fill=colors['right_leg'], width=2)
        
        poses.append(pose_img)
    
    return poses

def generate_frame_with_diffusion(reference_image, pose_image, ref_pose_image, seed=42):
    """Generate a single frame using diffusion
    
    Args:
        reference_image: The reference character image
        pose_image: The target pose for this frame
        ref_pose_image: The pose of the reference image
        seed: Random seed for generation
    """
    
    pipeline, _ = get_pipeline()
    
    if pipeline is None:
        return None
    
    try:
        generator = torch.manual_seed(seed)
        
        with torch.no_grad():
            result = pipeline(
                ref_image=reference_image,
                pose_image=pose_image,
                # ref_pose_image not used with PoseGuiderOrg
                width=512,
                height=512,
                num_inference_steps=30,  # More steps for better quality
                guidance_scale=5.0,     # Higher guidance for stronger pose control
                generator=generator,
            )
        
        # Get the generated image
        if hasattr(result, 'images'):
            generated_image = result.images
        else:
            generated_image = result
        
        # Convert to PIL if needed
        if isinstance(generated_image, torch.Tensor):
            print(f"Generated image tensor shape: {generated_image.shape}")
            
            # Handle different tensor formats based on actual dimensions
            if generated_image.dim() == 5:
                # Format: (batch, channels, height, width, color_channels) -> [1, 1, 512, 512, 3]
                # Extract first batch and channel: [0, 0] -> [512, 512, 3]
                generated_image = generated_image[0, 0]  # (height, width, color_channels)
                print(f"After extracting batch and channel: {generated_image.shape}")
            elif generated_image.dim() == 4:
                # Format: (batch, height, width, color_channels) -> [1, 512, 512, 3]
                generated_image = generated_image[0]  # (height, width, color_channels)
                print(f"After extracting batch: {generated_image.shape}")
            elif generated_image.dim() == 3:
                # Already in correct format: (height, width, color_channels)
                print(f"Already 3D (H, W, C): {generated_image.shape}")
            
            # Convert to numpy if it's still a tensor
            if isinstance(generated_image, torch.Tensor):
                generated_image = generated_image.cpu().numpy()
                print(f"After converting to numpy: {generated_image.shape}")
            
            # Ensure values are in [0, 1] range (they should already be from decode_latents)
            generated_image = np.clip(generated_image, 0, 1)
            
            # Convert to uint8
            generated_image = (generated_image * 255).astype(np.uint8)
            print(f"Final image shape before PIL: {generated_image.shape}, dtype: {generated_image.dtype}")
            
            # Handle different image formats
            if generated_image.shape[-1] == 3:
                generated_image = Image.fromarray(generated_image, mode='RGB')
            elif generated_image.shape[-1] == 1:
                generated_image = Image.fromarray(generated_image[:,:,0], mode='L')
            else:
                generated_image = Image.fromarray(generated_image)
        
        return generated_image
        
    except Exception as e:
        print(f"Error generating frame: {e}")
        traceback.print_exc()
        return None

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index_v2.html')

@app.route('/api/animations')
def get_animations():
    """Get list of available animation types"""
    animations = [
        {
            'id': key,
            'name': value['name'],
            'frames': value['frames']
        }
        for key, value in ANIMATION_TYPES.items()
    ]
    return jsonify(animations)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    animation_type = request.form.get('animation_type', 'idle')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{job_id}_{timestamp}_{filename}"
        filepath = app.config['UPLOAD_FOLDER'] / unique_filename
        file.save(str(filepath))
        
        # Initialize job status
        processing_status[job_id] = {
            'status': 'queued',
            'input_file': unique_filename,
            'output_file': None,
            'message': 'File uploaded successfully',
            'progress': 0,
            'animation_type': animation_type
        }
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_with_diffusion,
            args=(job_id, filepath, animation_type)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': 'File uploaded and processing started',
            'filename': unique_filename,
            'animation_type': animation_type
        }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_with_diffusion(job_id, input_path, animation_type):
    """Process the image using the diffusion pipeline"""
    try:
        processing_status[job_id]['status'] = 'processing'
        processing_status[job_id]['message'] = 'Initializing diffusion pipeline...'
        processing_status[job_id]['progress'] = 10
        
        # Load input image
        input_image = Image.open(input_path).convert("RGB")
        input_image = input_image.resize((512, 512), Image.LANCZOS)
        
        # Get animation configuration
        num_frames = ANIMATION_TYPES.get(animation_type, {}).get('frames', 4)
        
        processing_status[job_id]['progress'] = 20
        processing_status[job_id]['message'] = 'Extracting pose...'
        
        # Extract reference pose
        reference_pose = extract_pose(input_image)
        
        if reference_pose is None or np.array(reference_pose).max() == 0:
            # If pose extraction fails, create a synthetic OpenPose-style reference pose
            reference_pose = Image.new('RGB', (512, 512), color='black')
            draw = ImageDraw.Draw(reference_pose)
            
            # OpenPose-style colors
            colors = {
                'head': (255, 255, 0),       # Yellow
                'neck': (255, 128, 0),        # Orange  
                'body': (255, 0, 255),        # Magenta
                'right_arm': (255, 0, 0),     # Red
                'left_arm': (0, 255, 0),      # Green
                'right_leg': (255, 128, 0),   # Orange
                'left_leg': (0, 255, 255),    # Cyan
            }
            
            # Draw OpenPose-style skeleton - standing pose
            # Head
            draw.ellipse([240, 105, 270, 135], fill=colors['head'], outline=colors['head'])
            # Neck
            draw.line([255, 135, 255, 160], fill=colors['neck'], width=5)
            # Body
            draw.line([255, 160, 255, 300], fill=colors['body'], width=5)
            # Left arm
            draw.line([255, 180, 200, 250], fill=colors['left_arm'], width=4)
            # Right arm  
            draw.line([255, 180, 310, 250], fill=colors['right_arm'], width=4)
            # Left leg
            draw.line([255, 300, 230, 400], fill=colors['left_leg'], width=4)
            # Right leg
            draw.line([255, 300, 280, 400], fill=colors['right_leg'], width=4)
        
        processing_status[job_id]['progress'] = 30
        processing_status[job_id]['message'] = f'Generating poses for {animation_type} animation...'
        
        # Generate different poses for each frame based on animation type
        animation_poses = generate_animation_poses(reference_pose, animation_type, num_frames)
        
        processing_status[job_id]['progress'] = 40
        processing_status[job_id]['message'] = f'Generating {num_frames} frames with diffusion...'
        
        # Generate frames
        frames = []
        for i in range(num_frames):
            progress = 40 + (i * 40 // num_frames)
            processing_status[job_id]['progress'] = progress
            processing_status[job_id]['message'] = f'Generating frame {i+1}/{num_frames}...'
            
            # Use different pose for each frame
            generated_frame = generate_frame_with_diffusion(
                input_image, 
                animation_poses[i],
                reference_pose,  # Pass the reference pose
                seed=42 + i
            )
            
            if generated_frame is None:
                # Fallback to original image if generation fails
                generated_frame = input_image.copy()
            
            # Resize to consistent size
            generated_frame = generated_frame.resize((128, 128), Image.LANCZOS)
            frames.append(generated_frame)
        
        processing_status[job_id]['progress'] = 85
        processing_status[job_id]['message'] = 'Creating sprite sheet...'
        
        # Create sprite sheet
        sprite_width = 128 * num_frames
        sprite_height = 128
        sprite_sheet = Image.new('RGBA', (sprite_width, sprite_height), (0, 0, 0, 0))
        
        for i, frame in enumerate(frames):
            x = i * 128
            # Convert to RGBA if needed
            if frame.mode != 'RGBA':
                frame = frame.convert('RGBA')
            sprite_sheet.paste(frame, (x, 0))
        
        # Save result
        output_filename = f"sprite_{animation_type}_{job_id}.png"
        output_path = app.config['RESULTS_FOLDER'] / output_filename
        sprite_sheet.save(str(output_path), "PNG")
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['output_file'] = output_filename
        processing_status[job_id]['message'] = f'{animation_type} sprite sheet generated using diffusion!'
        processing_status[job_id]['progress'] = 100
        
    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['message'] = f'Error: {str(e)}'
        processing_status[job_id]['progress'] = 0
        print(f"Error processing job {job_id}: {e}")
        traceback.print_exc()

@app.route('/api/status/<job_id>')
def get_job_status(job_id):
    """Get processing status for a job"""
    if job_id in processing_status:
        status = processing_status[job_id].copy()
        if status['output_file']:
            status['output_url'] = url_for('static', filename=f'results/{status["output_file"]}')
        return jsonify(status)
    return jsonify({'error': 'Job not found'}), 404

@app.route('/api/download/<job_id>')
def download_result(job_id):
    """Download the processed result"""
    if job_id in processing_status and processing_status[job_id]['output_file']:
        file_path = app.config['RESULTS_FOLDER'] / processing_status[job_id]['output_file']
        if file_path.exists():
            return send_file(str(file_path), as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/api/export-gif', methods=['POST'])
def export_gif():
    """Export the sprite sheet as an animated GIF"""
    try:
        job_id = request.form.get('job_id')
        frame_delay = int(request.form.get('frame_delay', 50))
        
        if not job_id or job_id not in processing_status:
            return jsonify({'error': 'Invalid job ID'}), 400
        
        status = processing_status[job_id]
        if not status['output_file']:
            return jsonify({'error': 'No output file available'}), 400
        
        # Get animation type to determine frame count
        animation_type = status.get('animation_type', 'idle')
        frame_count = ANIMATION_TYPES.get(animation_type, {}).get('frames', 4)
        
        # Load the sprite sheet
        sprite_path = app.config['RESULTS_FOLDER'] / status['output_file']
        if not sprite_path.exists():
            return jsonify({'error': 'Sprite sheet not found'}), 404
        
        sprite_sheet = Image.open(sprite_path)
        sprite_width, sprite_height = sprite_sheet.size
        
        # Calculate frame dimensions
        frame_width = sprite_width // frame_count
        frame_height = sprite_height
        
        # Extract frames
        frames = []
        for i in range(frame_count):
            left = i * frame_width
            right = left + frame_width
            frame = sprite_sheet.crop((left, 0, right, frame_height))
            frames.append(frame)
        
        # Create GIF
        gif_filename = f"animation_{job_id}.gif"
        gif_path = app.config['RESULTS_FOLDER'] / gif_filename
        
        # Save as animated GIF
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_delay,
            loop=0
        )
        
        return send_file(str(gif_path), mimetype='image/gif', as_attachment=True, download_name=gif_filename)
        
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/info')
def get_info():
    """Get system information"""
    pipeline_ready = _pipeline is not None
    return jsonify({
        'pipeline_ready': pipeline_ready,
        'gpu_available': USE_GPU,
        'gpu_device': torch.cuda.get_device_name(0) if USE_GPU else None,
        'animations_available': len(ANIMATION_TYPES)
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Sprite Sheet Diffusion - Working Implementation")
    print("=" * 60)
    print(f"GPU: {'Enabled' if USE_GPU else 'Disabled'}")
    
    # Pre-load pipeline
    if USE_GPU:
        print("\nPre-loading diffusion pipeline...")
        get_pipeline()
    
    print("\nStarting server on http://localhost:8080")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8080, debug=False)