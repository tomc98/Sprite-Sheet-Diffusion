#!/usr/bin/env python3
"""
CLEAN Sprite Sheet Diffusion Web Application
Uses the WORKING UNet2D + EMA VAE approach for clean output
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

# Add ModelTraining to path
sys.path.append(str(Path(__file__).parent.parent / "ModelTraining"))

from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'static' / 'uploads'
app.config['RESULTS_FOLDER'] = Path(__file__).parent / 'static' / 'results'

# Ensure folders exist
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(parents=True, exist_ok=True)

# Global variables
clean_pipeline = None
processing_status = {}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Animation types with pose descriptions
ANIMATION_TYPES = {
    'idle': {
        'name': 'Idle',
        'description': 'Standing idle animation with subtle movement',
        'frames': 4,
        'poses': [
            "standing neutral pose",
            "slightly leaning left", 
            "standing neutral pose",
            "slightly leaning right"
        ]
    },
    'walk': {
        'name': 'Walk',
        'description': 'Walking cycle animation',
        'frames': 8,
        'poses': [
            "standing neutral pose",
            "left foot stepping forward",
            "mid-walk stride position", 
            "right foot stepping forward",
            "neutral standing position",
            "right foot stepping forward again",
            "mid-walk stride opposite",
            "left foot stepping forward again"
        ]
    },
    'run': {
        'name': 'Run', 
        'description': 'Running cycle animation',
        'frames': 6,
        'poses': [
            "preparing to run",
            "left leg forward running",
            "mid-run airborne",
            "right leg forward running", 
            "mid-run airborne opposite",
            "completing run cycle"
        ]
    },
    'jump': {
        'name': 'Jump',
        'description': 'Jump animation sequence',
        'frames': 6,
        'poses': [
            "crouching to prepare jump",
            "beginning to jump upward",
            "mid-air jumping peak", 
            "starting to land",
            "landing with bent knees",
            "returning to standing"
        ]
    },
    'attack': {
        'name': 'Attack',
        'description': 'Attack animation',
        'frames': 4,
        'poses': [
            "preparing attack stance",
            "winding up attack",
            "striking forward attack",
            "recovering from attack"
        ]
    }
}

def load_clean_pipeline():
    """Load the WORKING UNet2D + EMA VAE pipeline"""
    global clean_pipeline
    
    print("🔄 Loading CLEAN sprite generation pipeline...")
    print("   Using UNet2D + EMA VAE (avoiding problematic UNet3D)")
    
    try:
        device = "cuda"
        dtype = torch.float32
        model_base = Path(__file__).parent.parent / "ModelTraining" / "pretrained_model"
        
        print("  📦 Loading EMA VAE for high quality...")
        vae_ema = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device, dtype=dtype)
        
        print("  📦 Loading Img2Img pipeline with clean UNet2D...")
        clean_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            str(model_base / "stable-diffusion-v1-5"),
            vae=vae_ema,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        print("✅ CLEAN pipeline loaded successfully!")
        print("   ✅ EMA VAE for high-quality decoding")
        print("   ✅ UNet2D for clean, stable generation")
        print("   ✅ Img2Img for character pose modification")
        return True
        
    except Exception as e:
        print(f"❌ Error loading clean pipeline: {e}")
        traceback.print_exc()
        return False

def generate_clean_character_pose(character_img, pose_description, frame_idx):
    """Generate clean character in specific pose"""
    
    prompt = f"cute fluffy creature character {pose_description}, game sprite art, clean character, black background, high quality, detailed"
    negative_prompt = "blurry, distorted, bad anatomy, multiple characters, text, watermark, noise, grainy, corrupted"
    
    with torch.no_grad():
        result = clean_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=character_img,
            strength=0.35,  # Light modification to preserve character identity
            width=512,
            height=512,
            num_inference_steps=20,  # Good quality
            guidance_scale=7.5,
            generator=torch.Generator(device='cuda').manual_seed(400 + frame_idx)
        ).images[0]
    
    return result

def generate_clean_sprite_sheet(reference_image, animation_type, num_frames):
    """Generate clean sprite sheet using working UNet2D method"""
    
    if clean_pipeline is None:
        raise Exception("Clean pipeline not loaded")
    
    print(f"🎨 Generating {num_frames} CLEAN frames using UNet2D method...")
    
    # Process reference image with black background
    if reference_image.mode == 'RGBA':
        char_black = Image.new('RGB', reference_image.size, (0, 0, 0))
        char_black.paste(reference_image, mask=reference_image.split()[-1])
        reference_image = char_black
    
    reference_image = reference_image.resize((512, 512), Image.LANCZOS)
    
    # Get pose descriptions for this animation
    poses = ANIMATION_TYPES[animation_type]['poses']
    
    clean_frames = []
    
    for frame_idx in range(num_frames):
        pose_desc = poses[frame_idx] if frame_idx < len(poses) else poses[-1]
        print(f"  Frame {frame_idx + 1}/{num_frames}: {pose_desc}")
        
        # Generate clean frame
        clean_frame = generate_clean_character_pose(reference_image, pose_desc, frame_idx)
        clean_frames.append(clean_frame)
        
        # Save individual frame
        clean_frame.save(f"clean_frame_{animation_type}_{frame_idx:02d}.png")
        print(f"    ✅ Clean frame {frame_idx + 1} generated!")
    
    return clean_frames

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
            'message': 'Starting CLEAN sprite generation...',
            'progress': 0,
            'animation_type': animation_type
        }
        
        # Start processing in background
        thread = threading.Thread(target=process_clean_sprite, 
                                 args=(job_id, filepath, animation_type))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': 'CLEAN sprite generation started',
            'filename': unique_filename,
            'animation_type': animation_type
        }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_clean_sprite(job_id, input_path, animation_type):
    """Process sprite using the CLEAN UNet2D method"""
    try:
        processing_status[job_id]['status'] = 'processing'
        processing_status[job_id]['message'] = 'Loading image for CLEAN generation...'
        processing_status[job_id]['progress'] = 10
        
        input_image = Image.open(input_path).convert("RGBA")
        
        processing_status[job_id]['progress'] = 25
        processing_status[job_id]['message'] = f'Generating CLEAN {animation_type} animation frames...'
        
        num_frames = ANIMATION_TYPES[animation_type]['frames']
        
        # Use CLEAN generation method
        frames = generate_clean_sprite_sheet(input_image, animation_type, num_frames)
        
        processing_status[job_id]['progress'] = 85
        processing_status[job_id]['message'] = 'Assembling CLEAN sprite sheet...'
        
        # Create horizontal sprite sheet
        frame_width, frame_height = frames[0].size
        sprite_sheet = Image.new('RGB', (frame_width * len(frames), frame_height), (0, 0, 0))
        
        for i, frame in enumerate(frames):
            x = i * frame_width
            sprite_sheet.paste(frame, (x, 0))
        
        # Add subtle frame dividers
        draw = ImageDraw.Draw(sprite_sheet)
        for i in range(1, len(frames)):
            x = i * frame_width
            draw.line([(x, 0), (x, sprite_sheet.height)], fill=(32, 32, 32), width=1)
        
        output_filename = f"clean_sprite_{animation_type}_{job_id}.png"
        output_path = app.config['RESULTS_FOLDER'] / output_filename
        sprite_sheet.save(str(output_path), "PNG")
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['output_file'] = output_filename
        processing_status[job_id]['message'] = f'CLEAN {animation_type} sprite sheet generated successfully!'
        processing_status[job_id]['progress'] = 100
        
    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['message'] = f'Clean generation error: {str(e)}'
        processing_status[job_id]['progress'] = 0
        print(f"Error in clean processing {job_id}: {e}")
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

@app.route('/api/export-gif', methods=['POST'])
def export_gif():
    """Export sprite sheet as animated GIF"""
    try:
        job_id = request.form.get('job_id')
        frame_delay = int(request.form.get('frame_delay', 100))
        
        if not job_id or job_id not in processing_status:
            return jsonify({'error': 'Invalid job ID'}), 400
        
        status = processing_status[job_id]
        if not status['output_file']:
            return jsonify({'error': 'No output file available'}), 400
        
        animation_type = status.get('animation_type', 'idle')
        frame_count = ANIMATION_TYPES.get(animation_type, {}).get('frames', 4)
        
        # Load sprite sheet
        sprite_path = app.config['RESULTS_FOLDER'] / status['output_file']
        sprite_sheet = Image.open(sprite_path)
        
        # Extract frames
        frame_width = sprite_sheet.width // frame_count
        frames = []
        for i in range(frame_count):
            left = i * frame_width
            frame = sprite_sheet.crop((left, 0, left + frame_width, sprite_sheet.height))
            frames.append(frame)
        
        # Create animated GIF
        gif_filename = f"clean_animation_{job_id}.gif"
        gif_path = app.config['RESULTS_FOLDER'] / gif_filename
        
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_delay,
            loop=0
        )
        
        return send_file(str(gif_path), mimetype='image/gif', as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test')
def test():
    """Test endpoint"""
    pipeline_status = "loaded" if clean_pipeline else "not loaded"
    return jsonify({
        'status': 'ok',
        'pipeline': pipeline_status,
        'method': 'CLEAN UNet2D + EMA VAE',
        'quality': 'High (no corruption)',
        'animations_available': len(ANIMATION_TYPES),
        'cuda_available': torch.cuda.is_available()
    })

if __name__ == '__main__':
    print("🚀 Starting CLEAN Sprite Sheet Generation Web Application")
    print("✨ Using WORKING UNet2D + EMA VAE method (no corruption)")
    
    if load_clean_pipeline():
        print("✅ Clean pipeline loaded, starting web server...")
        app.run(host='0.0.0.0', port=8080, debug=False)
    else:
        print("❌ Failed to load clean pipeline")
        sys.exit(1)