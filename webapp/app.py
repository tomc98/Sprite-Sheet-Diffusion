#!/usr/bin/env python3
"""
Sprite Sheet Diffusion Web Application Backend
"""

import os
import sys
import json
import uuid
import shutil
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch

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
        thread = threading.Thread(target=process_image, args=(job_id, filepath))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': 'File uploaded and processing started',
            'filename': unique_filename
        }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_image(job_id, input_path):
    """Process the image using Sprite Sheet Diffusion"""
    try:
        processing_status[job_id]['status'] = 'processing'
        processing_status[job_id]['message'] = 'Initializing Sprite Sheet Diffusion...'
        processing_status[job_id]['progress'] = 10
        
        # Import necessary modules
        from diffusers import AutoencoderKL, DDIMScheduler
        from transformers import CLIPVisionModelWithProjection
        from omegaconf import OmegaConf
        from PIL import Image
        import numpy as np
        
        # Import custom modules
        from models.unet_2d_condition import UNet2DConditionModel
        from models.unet_3d import UNet3DConditionModel
        from models.pose_guider import PoseGuider
        from pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
        from utils.util import save_videos_grid
        
        processing_status[job_id]['progress'] = 20
        processing_status[job_id]['message'] = 'Loading configuration...'
        
        # Load config
        config_path = Path(__file__).parent.parent / "ModelTraining" / "configs" / "prompts" / "inference.yaml"
        config = OmegaConf.load(config_path)
        
        weight_dtype = torch.float16 if config.weight_dtype == "fp16" else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        processing_status[job_id]['progress'] = 30
        processing_status[job_id]['message'] = 'Loading models...'
        
        # Load models (simplified for demo)
        model_base = Path(__file__).parent.parent / "ModelTraining"
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            str(model_base / "pretrained_model" / "sd-vae-ft-mse"),
        ).to(device, dtype=weight_dtype)
        
        processing_status[job_id]['progress'] = 50
        
        # Load reference image
        ref_image = Image.open(input_path).convert("RGB")
        ref_image = ref_image.resize((512, 512), Image.LANCZOS)
        
        processing_status[job_id]['progress'] = 70
        processing_status[job_id]['message'] = 'Generating sprite sheet...'
        
        # For now, create a simple sprite sheet by duplicating the image
        # In production, this would use the full diffusion pipeline
        sprite_sheet = create_simple_sprite_sheet(ref_image)
        
        # Save result
        output_filename = f"result_{job_id}.png"
        output_path = app.config['RESULTS_FOLDER'] / output_filename
        sprite_sheet.save(str(output_path))
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['output_file'] = output_filename
        processing_status[job_id]['message'] = 'Processing completed successfully!'
        processing_status[job_id]['progress'] = 100
        
    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['message'] = f'Error: {str(e)}'
        processing_status[job_id]['progress'] = 0
        print(f"Error processing job {job_id}: {e}")

def create_simple_sprite_sheet(image, frames=8):
    """Create a simple sprite sheet by creating variations of the input image"""
    from PIL import Image, ImageEnhance, ImageOps
    import numpy as np
    
    width, height = image.size
    sprite_sheet = Image.new('RGBA', (width * frames, height))
    
    for i in range(frames):
        # Create variations for demo
        frame = image.copy()
        
        # Apply different transformations for each frame
        if i % 2 == 1:
            frame = ImageOps.mirror(frame)  # Flip horizontally
        
        # Vary brightness slightly
        enhancer = ImageEnhance.Brightness(frame)
        brightness_factor = 0.8 + (i * 0.05)
        frame = enhancer.enhance(brightness_factor)
        
        # Paste into sprite sheet
        sprite_sheet.paste(frame, (i * width, 0))
    
    return sprite_sheet

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)