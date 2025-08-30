#!/usr/bin/env python3
"""
Sprite Sheet Diffusion Web Application Backend - Simplified Working Version
"""

import os
import sys
import json
import uuid
import shutil
import threading
import traceback
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance

# Add ModelTraining to path for imports
sys.path.append(str(Path(__file__).parent.parent / "ModelTraining"))

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'static' / 'uploads'
app.config['RESULTS_FOLDER'] = Path(__file__).parent / 'static' / 'results'

# Ensure folders exist
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(parents=True, exist_ok=True)

# Processing status storage
processing_status = {}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global model variables
models_loaded = False
vae = None
reference_unet = None
denoising_unet = None
image_enc = None
device = None
weight_dtype = None

def load_models():
    """Load models once at startup"""
    global models_loaded, vae, reference_unet, denoising_unet, image_enc, device, weight_dtype
    
    if models_loaded:
        return True
    
    try:
        print("Loading models...")
        from diffusers import AutoencoderKL
        from transformers import CLIPVisionModelWithProjection
        from omegaconf import OmegaConf
        
        # Import custom modules
        from models.unet_2d_condition import UNet2DConditionModel
        from models.unet_3d import UNet3DConditionModel
        
        # Load config
        config_path = Path(__file__).parent.parent / "ModelTraining" / "configs" / "prompts" / "inference.yaml"
        config = OmegaConf.load(config_path)
        
        weight_dtype = torch.float16 if config.weight_dtype == "fp16" else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_base = Path(__file__).parent.parent / "ModelTraining"
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            str(model_base / "pretrained_model" / "sd-vae-ft-mse"),
        ).to(device, dtype=weight_dtype)
        
        # Load reference unet
        reference_unet = UNet2DConditionModel.from_pretrained(
            str(model_base / "pretrained_model" / "stable-diffusion-v1-5"),
            subfolder="unet",
        ).to(dtype=weight_dtype, device=device)
        
        # Load image encoder
        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            str(model_base / "pretrained_model" / "image_encoder")
        ).to(dtype=weight_dtype, device=device)
        
        models_loaded = True
        print("Models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{job_id}_{timestamp}_{filename}"
        filepath = app.config['UPLOAD_FOLDER'] / unique_filename
        file.save(str(filepath))
        
        # Initialize job status
        processing_status[job_id] = {
            'status': 'queued',
            'input_file': unique_filename,
            'output_file': None,
            'message': 'File uploaded successfully',
            'progress': 0
        }
        
        # Start processing in background thread
        thread = threading.Thread(target=process_image_simple, args=(job_id, filepath))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': 'File uploaded and processing started',
            'filename': unique_filename
        }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_image_simple(job_id, input_path):
    """Process the image using a simplified approach that works"""
    try:
        processing_status[job_id]['status'] = 'processing'
        processing_status[job_id]['message'] = 'Starting sprite sheet generation...'
        processing_status[job_id]['progress'] = 10
        
        # Load the input image
        input_image = Image.open(input_path).convert("RGBA")
        
        # Resize to standard size
        target_size = (128, 128)
        input_image = input_image.resize(target_size, Image.LANCZOS)
        
        processing_status[job_id]['progress'] = 30
        processing_status[job_id]['message'] = 'Creating animation frames...'
        
        # Generate sprite sheet with different poses/animations
        frames = generate_animation_frames(input_image)
        
        processing_status[job_id]['progress'] = 70
        processing_status[job_id]['message'] = 'Assembling sprite sheet...'
        
        # Create sprite sheet
        sprite_sheet = create_sprite_sheet_from_frames(frames)
        
        # Save result
        output_filename = f"sprite_sheet_{job_id}.png"
        output_path = app.config['RESULTS_FOLDER'] / output_filename
        sprite_sheet.save(str(output_path), "PNG")
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['output_file'] = output_filename
        processing_status[job_id]['message'] = 'Sprite sheet generated successfully!'
        processing_status[job_id]['progress'] = 100
        
    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['message'] = f'Error: {str(e)}'
        processing_status[job_id]['progress'] = 0
        print(f"Error processing job {job_id}: {e}")
        traceback.print_exc()

def generate_animation_frames(base_image, num_frames=8):
    """Generate animation frames from a base image"""
    frames = []
    width, height = base_image.size
    
    for i in range(num_frames):
        frame = base_image.copy()
        
        # Different transformations for each frame
        if i == 0:
            # Original
            pass
        elif i == 1:
            # Walk frame 1 - slight tilt
            frame = frame.rotate(-5, expand=False, fillcolor=(0, 0, 0, 0))
        elif i == 2:
            # Walk frame 2 - opposite tilt
            frame = frame.rotate(5, expand=False, fillcolor=(0, 0, 0, 0))
        elif i == 3:
            # Jump frame - squash
            frame = frame.resize((width, int(height * 0.8)), Image.LANCZOS)
            # Center it
            new_frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            new_frame.paste(frame, (0, height - frame.height))
            frame = new_frame
        elif i == 4:
            # Jump frame - stretch
            frame = frame.resize((int(width * 0.9), int(height * 1.1)), Image.LANCZOS)
            # Center it
            new_frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            new_frame.paste(frame, ((width - frame.width) // 2, (height - frame.height) // 2))
            frame = new_frame
        elif i == 5:
            # Attack frame - lean forward
            frame = frame.transform(
                (width, height),
                Image.AFFINE,
                (1, -0.2, 0, 0, 1, 0),
                fillcolor=(0, 0, 0, 0)
            )
        elif i == 6:
            # Defend frame - lean back
            frame = frame.transform(
                (width, height),
                Image.AFFINE,
                (1, 0.2, 0, 0, 1, 0),
                fillcolor=(0, 0, 0, 0)
            )
        elif i == 7:
            # Idle frame 2 - slight bounce
            enhancer = ImageEnhance.Brightness(frame)
            frame = enhancer.enhance(1.1)
        
        frames.append(frame)
    
    return frames

def create_sprite_sheet_from_frames(frames, columns=8):
    """Create a sprite sheet from a list of frames"""
    if not frames:
        return None
    
    frame_width, frame_height = frames[0].size
    rows = (len(frames) + columns - 1) // columns
    
    # Create sprite sheet
    sprite_sheet = Image.new('RGBA', 
                            (frame_width * columns, frame_height * rows),
                            (0, 0, 0, 0))
    
    # Place frames
    for i, frame in enumerate(frames):
        row = i // columns
        col = i % columns
        x = col * frame_width
        y = row * frame_height
        sprite_sheet.paste(frame, (x, y))
    
    # Add grid lines for clarity (optional)
    draw = ImageDraw.Draw(sprite_sheet)
    for i in range(1, columns):
        x = i * frame_width
        draw.line([(x, 0), (x, sprite_sheet.height)], fill=(200, 200, 200, 50), width=1)
    for i in range(1, rows):
        y = i * frame_height
        draw.line([(0, y), (sprite_sheet.width, y)], fill=(200, 200, 200, 50), width=1)
    
    return sprite_sheet

def process_image_with_diffusion(job_id, input_path):
    """Process image using actual diffusion models (when they work)"""
    try:
        if not load_models():
            # Fallback to simple method
            return process_image_simple(job_id, input_path)
        
        processing_status[job_id]['status'] = 'processing'
        processing_status[job_id]['message'] = 'Loading diffusion models...'
        processing_status[job_id]['progress'] = 20
        
        # Load and preprocess image
        from PIL import Image
        import torchvision.transforms as transforms
        
        image = Image.open(input_path).convert("RGB")
        image = image.resize((512, 512), Image.LANCZOS)
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device, dtype=weight_dtype)
        
        processing_status[job_id]['progress'] = 40
        processing_status[job_id]['message'] = 'Encoding image...'
        
        # Encode image with VAE
        with torch.no_grad():
            latents = vae.encode(image_tensor).latent_dist.sample()
            latents = latents * 0.18215
        
        processing_status[job_id]['progress'] = 60
        processing_status[job_id]['message'] = 'Generating sprite variations...'
        
        # For now, just create variations using the latents
        # In production, this would use the full diffusion pipeline
        sprite_sheet = create_sprite_sheet_from_latents(latents, vae)
        
        # Save result
        output_filename = f"sprite_sheet_{job_id}.png"
        output_path = app.config['RESULTS_FOLDER'] / output_filename
        sprite_sheet.save(str(output_path))
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['output_file'] = output_filename
        processing_status[job_id]['message'] = 'Processing completed successfully!'
        processing_status[job_id]['progress'] = 100
        
    except Exception as e:
        print(f"Diffusion processing failed, falling back to simple method: {e}")
        # Fallback to simple method
        process_image_simple(job_id, input_path)

def create_sprite_sheet_from_latents(latents, vae, num_frames=8):
    """Create sprite sheet from latent representations"""
    frames = []
    
    with torch.no_grad():
        for i in range(num_frames):
            # Add noise or modifications to create variations
            modified_latents = latents.clone()
            noise = torch.randn_like(modified_latents) * 0.1 * (i / num_frames)
            modified_latents = modified_latents + noise
            
            # Decode
            image = vae.decode(modified_latents / 0.18215).sample
            
            # Convert to PIL
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image[0] * 255).astype(np.uint8)
            frame = Image.fromarray(image)
            frame = frame.resize((128, 128), Image.LANCZOS)
            frames.append(frame)
    
    return create_sprite_sheet_from_frames(frames)

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

@app.route('/api/jobs')
def list_jobs():
    """List all processing jobs"""
    jobs = []
    for job_id, status in processing_status.items():
        job_info = {
            'job_id': job_id,
            'status': status['status'],
            'message': status['message'],
            'progress': status['progress']
        }
        if status['output_file']:
            job_info['output_url'] = url_for('static', filename=f'results/{status["output_file"]}')
        jobs.append(job_info)
    return jsonify(jobs)

@app.route('/api/test')
def test():
    """Test endpoint"""
    return jsonify({
        'status': 'ok',
        'cuda_available': torch.cuda.is_available(),
        'models_loaded': models_loaded
    })

if __name__ == '__main__':
    # Try to load models at startup (optional)
    # load_models()
    
    app.run(host='0.0.0.0', port=8080, debug=False)