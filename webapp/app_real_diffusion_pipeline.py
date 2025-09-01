#!/usr/bin/env python3
"""
Real Sprite Sheet Diffusion Web Application
Uses the actual diffusion pipeline from the research paper
"""

import os
import sys
import json
import uuid
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import threading
import traceback
import math

# Add ModelTraining to path for imports
sys.path.append(str(Path(__file__).parent.parent / "ModelTraining"))

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from models.unet_2d_condition import UNet2DConditionModel  
from models.unet_3d import UNet3DConditionModel
from models.pose_guider_org import PoseGuiderOrg
from models.pose_guider_multi_res import SingleTensorPoseGuider
from pipelines.pipeline_pose2img import Pose2ImagePipeline

app = Flask(__name__)
CORS(app)

# Configuration  
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'static' / 'uploads'
app.config['RESULTS_FOLDER'] = Path(__file__).parent / 'static' / 'results'

# Ensure folders exist
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(parents=True, exist_ok=True)

# Global variables for models
diffusion_pipeline = None
processing_status = {}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Animation types
ANIMATION_TYPES = {
    'idle': {'name': 'Idle', 'description': 'Standing idle animation', 'frames': 4},
    'walk': {'name': 'Walk', 'description': 'Walking cycle animation', 'frames': 8},
    'run': {'name': 'Run', 'description': 'Running cycle animation', 'frames': 6},
    'jump': {'name': 'Jump', 'description': 'Jump animation sequence', 'frames': 6},
    'attack': {'name': 'Attack', 'description': 'Attack animation', 'frames': 4},
}

def load_real_diffusion_pipeline():
    """Load the complete diffusion pipeline as described in the paper"""
    global diffusion_pipeline
    
    print("🔄 Loading complete diffusion pipeline...")
    
    try:
        # Paths to pretrained models
        model_base = Path(__file__).parent.parent / "ModelTraining" / "pretrained_model"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use fp32 to avoid bias dtype mismatch as noted by senior dev
        weight_dtype = torch.float32
        
        print("  📦 Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            str(model_base / "sd-vae-ft-mse")
        ).to(device, dtype=weight_dtype)
        
        print("  📦 Loading Image Encoder...")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(model_base / "clip-vit-large-patch14")
        ).to(device, dtype=weight_dtype)
        
        print("  📦 Loading Reference UNet...")
        reference_unet = UNet2DConditionModel.from_pretrained(
            str(model_base / "stable-diffusion-v1-5" / "unet")
        ).to(device, dtype=weight_dtype)
        
        print("  📦 Loading Denoising UNet...")
        # Load with proper parameters as shown in working app
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            str(model_base / "stable-diffusion-v1-5"),
            "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
            },
        ).to(device, dtype=weight_dtype)
        
        # Load custom weights BEFORE motion module to avoid dtype conflicts
        print("  🔧 Loading custom model weights...")
        if (model_base / "reference_unet.pth").exists():
            reference_state = torch.load(str(model_base / "reference_unet.pth"), map_location="cpu")
            reference_unet.load_state_dict(reference_state, strict=False)
            print("    ✅ Reference UNet custom weights loaded")
        
        if (model_base / "denoising_unet.pth").exists():
            denoising_state = torch.load(str(model_base / "denoising_unet.pth"), map_location="cpu")
            denoising_unet.load_state_dict(denoising_state, strict=False)
            print("    ✅ Denoising UNet custom weights loaded")
        
        print("  📦 Loading Pose Guider with TENSOR DIMENSION FIX...")
        # Use the FIXED approach with SingleTensorPoseGuider wrapper
        base_pose_guider = PoseGuiderOrg(
            conditioning_embedding_channels=320,
            block_out_channels=(16, 32, 96, 256),
        ).to(device, dtype=weight_dtype)
        
        if (model_base / "pose_guider.pth").exists():
            pose_state = torch.load(str(model_base / "pose_guider.pth"), map_location="cpu")
            # Convert to fp32
            for k in pose_state:
                if pose_state[k].dtype == torch.float16:
                    pose_state[k] = pose_state[k].float()
            base_pose_guider.load_state_dict(pose_state, strict=True)
            print("    ✅ Base Pose Guider weights loaded")
        
        # Apply the TENSOR DIMENSION FIX using SingleTensorPoseGuider wrapper
        pose_guider = SingleTensorPoseGuider(
            base_pose_guider, 
            unet_block_channels=(320, 320, 640, 1280, 1280)
        ).to(device)
        print("    ✅ SingleTensorPoseGuider TENSOR FIX applied!")
        
        print("  📦 Setting up scheduler...")
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1
        )
        
        print("  🔧 Creating pipeline...")
        # Create the actual pipeline from the paper
        diffusion_pipeline = Pose2ImagePipeline(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler
        )
        
        print("✅ Complete diffusion pipeline loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        traceback.print_exc()
        return False

def create_openpose_image(pose_type, frame_idx, total_frames, size=(512, 512)):
    """Create OpenPose-style conditioning image for the diffusion pipeline"""
    
    # Create black background
    pose_img = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(pose_img)
    
    # Character center position
    cx, cy = size[0] // 2, size[1] // 2 + 50
    
    # Animation parameter
    t = frame_idx / total_frames * 2 * np.pi
    
    # OpenPose keypoint colors (matching the training data format)
    colors = {
        'head': (255, 255, 0),      # Yellow
        'neck': (255, 128, 0),      # Orange  
        'body': (255, 0, 255),      # Magenta
        'left_arm': (0, 255, 0),    # Green
        'right_arm': (255, 0, 0),   # Red
        'left_leg': (0, 255, 255),  # Cyan
        'right_leg': (255, 128, 0), # Orange
    }
    
    # Generate pose keypoints based on animation type
    keypoints = generate_pose_keypoints(pose_type, frame_idx, total_frames, cx, cy)
    
    # Draw pose skeleton
    draw_pose_skeleton(draw, keypoints, colors)
    
    return pose_img

def generate_pose_keypoints(pose_type, frame_idx, total_frames, cx, cy):
    """Generate pose keypoints for different animation types"""
    
    t = frame_idx / total_frames * 2 * np.pi
    keypoints = {}
    
    if pose_type == 'walk':
        # Walking cycle keypoints
        body_sway = np.sin(t) * 8
        head_bob = abs(np.sin(t * 2)) * 4
        
        keypoints['head'] = (cx + body_sway//2, cy - 120 - head_bob)
        keypoints['neck'] = (cx + body_sway//2, cy - 100)
        keypoints['left_shoulder'] = (cx - 20 + body_sway//2, cy - 80)
        keypoints['right_shoulder'] = (cx + 20 + body_sway//2, cy - 80)
        
        # Arms with opposite swing to legs
        arm_swing = np.sin(t + np.pi) * 20
        keypoints['left_elbow'] = (cx - 30 + body_sway + arm_swing, cy - 40)
        keypoints['right_elbow'] = (cx + 30 + body_sway - arm_swing, cy - 40)
        keypoints['left_wrist'] = (cx - 35 + body_sway + arm_swing, cy - 10)
        keypoints['right_wrist'] = (cx + 35 + body_sway - arm_swing, cy - 10)
        
        # Torso
        keypoints['spine'] = (cx + body_sway, cy - 40)
        keypoints['hip'] = (cx + body_sway, cy + 20)
        
        # Legs with walking cycle
        leg_cycle = np.sin(t) * 30
        keypoints['left_hip'] = (cx - 15 + body_sway, cy + 20)
        keypoints['right_hip'] = (cx + 15 + body_sway, cy + 20)
        keypoints['left_knee'] = (cx - 20 + body_sway + leg_cycle//2, cy + 60)
        keypoints['right_knee'] = (cx + 20 + body_sway - leg_cycle//2, cy + 60)
        keypoints['left_ankle'] = (cx - 25 + body_sway + leg_cycle, cy + 100)
        keypoints['right_ankle'] = (cx + 25 + body_sway - leg_cycle, cy + 100)
        
    elif pose_type == 'jump':
        # Jump sequence
        if frame_idx < total_frames // 3:
            # Crouch phase
            crouch = 0.7
            keypoints['head'] = (cx, cy - int(120 * crouch))
            keypoints['hip'] = (cx, cy + int(40 * crouch))
            # Bent legs
            keypoints['left_knee'] = (cx - 30, cy + 20)
            keypoints['right_knee'] = (cx + 30, cy + 20)
            keypoints['left_ankle'] = (cx - 35, cy + 40)
            keypoints['right_ankle'] = (cx + 35, cy + 40)
            # Arms back
            keypoints['left_wrist'] = (cx - 50, cy - 30)
            keypoints['right_wrist'] = (cx + 50, cy - 30)
            
        elif frame_idx < 2 * total_frames // 3:
            # Air phase
            keypoints['head'] = (cx, cy - 140)
            keypoints['hip'] = (cx, cy + 10)
            # Extended legs
            keypoints['left_knee'] = (cx - 20, cy + 50)
            keypoints['right_knee'] = (cx + 20, cy + 50)
            keypoints['left_ankle'] = (cx - 25, cy + 90)
            keypoints['right_ankle'] = (cx + 25, cy + 90)
            # Arms up
            keypoints['left_wrist'] = (cx - 60, cy - 50)
            keypoints['right_wrist'] = (cx + 60, cy - 50)
        else:
            # Landing phase
            crouch = 0.8
            keypoints['head'] = (cx, cy - int(110 * crouch))
            keypoints['hip'] = (cx, cy + int(30 * crouch))
            # Slightly bent legs
            keypoints['left_knee'] = (cx - 25, cy + 30)
            keypoints['right_knee'] = (cx + 25, cy + 30)
            keypoints['left_ankle'] = (cx - 30, cy + 70)
            keypoints['right_ankle'] = (cx + 30, cy + 70)
            # Arms stabilizing
            keypoints['left_wrist'] = (cx - 40, cy - 20)
            keypoints['right_wrist'] = (cx + 40, cy - 20)
    
    elif pose_type == 'idle':
        # Subtle idle breathing
        breath = np.sin(t * 2) * 2
        sway = np.sin(t) * 3
        
        keypoints['head'] = (cx + sway, cy - 120 + breath)
        keypoints['neck'] = (cx + sway, cy - 100 + breath)
        keypoints['hip'] = (cx + sway, cy + 20)
        keypoints['left_ankle'] = (cx - 20, cy + 100)
        keypoints['right_ankle'] = (cx + 20, cy + 100)
        keypoints['left_wrist'] = (cx - 30 + sway, cy + breath)
        keypoints['right_wrist'] = (cx + 30 + sway, cy + breath)
    
    # Fill in missing standard keypoints
    if 'neck' not in keypoints and 'head' in keypoints:
        keypoints['neck'] = (keypoints['head'][0], keypoints['head'][1] + 20)
    if 'spine' not in keypoints and 'neck' in keypoints and 'hip' in keypoints:
        keypoints['spine'] = (keypoints['neck'][0], (keypoints['neck'][1] + keypoints['hip'][1]) // 2)
    
    return keypoints

def draw_pose_skeleton(draw, keypoints, colors):
    """Draw the pose skeleton with OpenPose style"""
    
    # Draw connections between keypoints
    connections = [
        ('head', 'neck', colors['head']),
        ('neck', 'spine', colors['body']),
        ('spine', 'hip', colors['body']),
        ('neck', 'left_shoulder', colors['left_arm']),
        ('neck', 'right_shoulder', colors['right_arm']),
        ('left_shoulder', 'left_elbow', colors['left_arm']),
        ('left_elbow', 'left_wrist', colors['left_arm']),
        ('right_shoulder', 'right_elbow', colors['right_arm']),
        ('right_elbow', 'right_wrist', colors['right_arm']),
        ('hip', 'left_hip', colors['left_leg']),
        ('hip', 'right_hip', colors['right_leg']),
        ('left_hip', 'left_knee', colors['left_leg']),
        ('left_knee', 'left_ankle', colors['left_leg']),
        ('right_hip', 'right_knee', colors['right_leg']),
        ('right_knee', 'right_ankle', colors['right_leg']),
    ]
    
    # Draw connections
    for start, end, color in connections:
        if start in keypoints and end in keypoints:
            draw.line([keypoints[start], keypoints[end]], fill=color, width=6)
    
    # Draw keypoints as circles
    for point_name, (x, y) in keypoints.items():
        color = colors.get('body', (255, 255, 255))  # Default color
        if 'left_arm' in point_name or 'left_shoulder' in point_name or 'left_elbow' in point_name or 'left_wrist' in point_name:
            color = colors['left_arm']
        elif 'right_arm' in point_name or 'right_shoulder' in point_name or 'right_elbow' in point_name or 'right_wrist' in point_name:
            color = colors['right_arm']
        elif 'left' in point_name:
            color = colors['left_leg']
        elif 'right' in point_name:
            color = colors['right_leg']
        elif 'head' in point_name:
            color = colors['head']
            
        draw.ellipse([x-5, y-5, x+5, y+5], fill=color)

def generate_sprite_with_real_diffusion(reference_image, animation_type, num_frames):
    """Generate sprite using the WORKING diffusion method with SingleTensorPoseGuider fix"""
    
    if diffusion_pipeline is None:
        raise Exception("Diffusion pipeline not loaded")
    
    print(f"🎨 Generating {num_frames} frames using WORKING diffusion method...")
    
    # Process reference image with BLACK background (like training data)
    if reference_image.mode == 'RGBA':
        rgb_image = Image.new('RGB', reference_image.size, (0, 0, 0))  # BLACK background!
        rgb_image.paste(reference_image, mask=reference_image.split()[-1])
        reference_image = rgb_image
    
    reference_image = reference_image.resize((512, 512), Image.LANCZOS)
    
    # Create reference pose (standing/idle) 
    ref_pose = Image.new('RGB', (512, 512), (0, 0, 0))
    draw = ImageDraw.Draw(ref_pose)
    cx, cy = 256, 350
    draw.ellipse([cx-6, cy-76, cx+6, cy-64], fill=(255, 255, 0))
    draw.line([(cx, cy-64), (cx, cy+16)], fill=(255, 0, 255), width=4)
    draw.line([(cx, cy-44), (cx-20, cy-14)], fill=(0, 255, 0), width=2)
    draw.line([(cx, cy-44), (cx+20, cy-14)], fill=(255, 0, 0), width=2)
    draw.line([(cx, cy+16), (cx-16, cy+66)], fill=(0, 255, 255), width=2)
    draw.line([(cx, cy+16), (cx+16, cy+66)], fill=(255, 128, 0), width=2)
    
    generated_frames = []
    
    for frame_idx in range(num_frames):
        print(f"  Generating frame {frame_idx + 1}/{num_frames}...")
        
        # Create clean pose for this frame
        pose_image = create_openpose_image(animation_type, frame_idx, num_frames)
        pose_image.save(f"debug_pose_{animation_type}_{frame_idx:02d}.png")
        
        # Generate using the WORKING method (no exception handling - let it fail properly)
        with torch.no_grad():
            print(f"    🧠 Running WORKING diffusion inference...")
            
            result = diffusion_pipeline(
                ref_image=reference_image,
                pose_image=pose_image,
                ref_pose_image=ref_pose,  # Include reference pose as pipeline expects
                width=512,
                height=512, 
                num_inference_steps=15,  # Reasonable for testing
                guidance_scale=3.5,
                generator=torch.Generator(device='cuda').manual_seed(200 + frame_idx)
            )
            
            print(f"    🔄 Processing with WORKING method...")
            
            # Use the working output processing (images are already [0,1])
            if hasattr(result, 'images'):
                img_tensor = result.images[0, :, 0, :, :].permute(1, 2, 0).cpu().numpy()
            else:
                img_tensor = result[0, :, 0, :, :].permute(1, 2, 0).cpu().numpy()
            
            # Since range is [0,1], convert directly
            img_tensor = np.clip(img_tensor, 0, 1)
            generated_frame = Image.fromarray((img_tensor * 255).astype(np.uint8))
            
            generated_frames.append(generated_frame)
            print(f"    ✅ Frame {frame_idx + 1} completed!")
    
    return generated_frames

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index_v2.html')

@app.route('/api/animations')
def get_animations():
    """Get list of available animations"""
    animations = []
    for key, value in ANIMATION_TYPES.items():
        animations.append({
            'id': key,
            'name': value['name'],
            'description': value['description'],
            'frames': value['frames']
        })
    return jsonify(animations)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload with animation type"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    animation_type = request.form.get('animation_type', 'idle')
    if animation_type not in ANIMATION_TYPES:
        animation_type = 'idle'
    
    if file and allowed_file(file.filename):
        job_id = str(uuid.uuid4())
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{job_id}_{timestamp}_{filename}"
        filepath = app.config['UPLOAD_FOLDER'] / unique_filename
        file.save(str(filepath))
        
        processing_status[job_id] = {
            'status': 'queued',
            'input_file': unique_filename,
            'output_file': None,
            'message': 'File uploaded, starting real diffusion generation...',
            'progress': 0,
            'animation_type': animation_type
        }
        
        thread = threading.Thread(target=process_with_real_diffusion, 
                                 args=(job_id, filepath, animation_type))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': 'File uploaded, real diffusion pipeline starting',
            'filename': unique_filename,
            'animation_type': animation_type
        }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_with_real_diffusion(job_id, input_path, animation_type):
    """Process the image with the real diffusion pipeline"""
    try:
        processing_status[job_id]['status'] = 'processing'
        processing_status[job_id]['message'] = 'Loading image and initializing diffusion...'
        processing_status[job_id]['progress'] = 10
        
        input_image = Image.open(input_path).convert("RGBA")
        
        processing_status[job_id]['progress'] = 20
        processing_status[job_id]['message'] = f'Running diffusion pipeline for {animation_type} animation...'
        
        num_frames = ANIMATION_TYPES[animation_type]['frames']
        
        # Generate frames using the real diffusion pipeline
        frames = generate_sprite_with_real_diffusion(input_image, animation_type, num_frames)
        
        processing_status[job_id]['progress'] = 90
        processing_status[job_id]['message'] = 'Assembling final sprite sheet...'
        
        # Create sprite sheet
        frame_width, frame_height = frames[0].size
        sprite_sheet = Image.new('RGB', (frame_width * len(frames), frame_height), (255, 255, 255))
        
        for i, frame in enumerate(frames):
            x = i * frame_width
            sprite_sheet.paste(frame, (x, 0))
        
        # Add frame dividers
        draw = ImageDraw.Draw(sprite_sheet)
        for i in range(1, len(frames)):
            x = i * frame_width
            draw.line([(x, 0), (x, sprite_sheet.height)], fill=(200, 200, 200), width=2)
        
        output_filename = f"real_diffusion_sprite_{animation_type}_{job_id}.png"
        output_path = app.config['RESULTS_FOLDER'] / output_filename
        sprite_sheet.save(str(output_path), "PNG")
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['output_file'] = output_filename
        processing_status[job_id]['message'] = f'Real diffusion {animation_type} sprite sheet completed!'
        processing_status[job_id]['progress'] = 100
        
    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['message'] = f'Diffusion error: {str(e)}'
        processing_status[job_id]['progress'] = 0
        print(f"Error in real diffusion processing {job_id}: {e}")
        traceback.print_exc()

@app.route('/api/status/<job_id>')
def get_status(job_id):
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

@app.route('/api/test')
def test():
    """Test endpoint"""
    pipeline_status = "loaded" if diffusion_pipeline is not None else "not loaded"
    return jsonify({
        'status': 'ok',
        'diffusion_pipeline': pipeline_status,
        'animations_available': len(ANIMATION_TYPES),
        'cuda_available': torch.cuda.is_available(),
        'models_ready': pipeline_status == "loaded"
    })

if __name__ == '__main__':
    print("🚀 Starting REAL Sprite Sheet Diffusion Web Application")
    print("📄 This implementation follows the research paper methodology")
    
    # Load the complete diffusion pipeline
    if load_real_diffusion_pipeline():
        print("✅ Real diffusion pipeline loaded, starting web server...")
        app.run(host='0.0.0.0', port=8080, debug=False)
    else:
        print("❌ Failed to load diffusion pipeline, exiting...")
        sys.exit(1)
