#!/usr/bin/env python3
"""
Sprite Sheet Diffusion Web Application - Using Actual Diffusion Models
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
from PIL import Image

# Add ModelTraining to path for imports
sys.path.append(str(Path(__file__).parent.parent / "ModelTraining"))

# Check if CUDA is available
USE_GPU = torch.cuda.is_available()
print(f"GPU Available: {USE_GPU}")
if USE_GPU:
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Import the diffusion generator
try:
    from sprite_diffusion import get_generator, SpriteDiffusionGenerator
    DIFFUSION_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load diffusion models: {e}")
    DIFFUSION_AVAILABLE = False
    # Fall back to simple transformations
    from app_v3 import (
        generate_idle_animation, generate_walk_animation, 
        generate_run_animation, generate_jump_animation,
        generate_attack_animation, generate_defend_animation,
        generate_crouch_animation, generate_death_animation,
        generate_celebrate_animation, generate_roll_animation,
        create_sprite_sheet_from_frames
    )

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

# Animation types with frame configurations
ANIMATION_TYPES = {
    'idle': {'name': 'Idle', 'description': 'Standing idle animation', 'frames': 4},
    'walk': {'name': 'Walk', 'description': 'Walking cycle animation', 'frames': 8},
    'run': {'name': 'Run', 'description': 'Running cycle animation', 'frames': 8},
    'jump': {'name': 'Jump', 'description': 'Jump animation sequence', 'frames': 6},
    'attack': {'name': 'Attack', 'description': 'Attack animation', 'frames': 6},
    'defend': {'name': 'Defend', 'description': 'Defensive stance', 'frames': 4},
    'crouch': {'name': 'Crouch', 'description': 'Crouching animation', 'frames': 4},
    'death': {'name': 'Death', 'description': 'Death animation', 'frames': 6},
    'celebrate': {'name': 'Celebrate', 'description': 'Victory celebration', 'frames': 8},
    'roll': {'name': 'Roll', 'description': 'Rolling animation', 'frames': 8},
}

# Global generator instance (lazy loaded)
_generator = None

def get_diffusion_generator():
    """Get or create the diffusion generator instance"""
    global _generator
    if _generator is None and DIFFUSION_AVAILABLE:
        try:
            print("Initializing diffusion models...")
            _generator = SpriteDiffusionGenerator(device="cuda" if USE_GPU else "cpu")
            print("Diffusion models loaded successfully!")
        except Exception as e:
            print(f"Failed to initialize diffusion models: {e}")
            traceback.print_exc()
    return _generator

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            'description': value['description'],
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
            'animation_type': animation_type,
            'using_diffusion': DIFFUSION_AVAILABLE
        }
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_with_diffusion if DIFFUSION_AVAILABLE else process_with_transforms,
            args=(job_id, filepath, animation_type)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': 'File uploaded and processing started',
            'filename': unique_filename,
            'animation_type': animation_type,
            'using_diffusion': DIFFUSION_AVAILABLE
        }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_with_diffusion(job_id, input_path, animation_type):
    """Process the image using the diffusion pipeline"""
    try:
        processing_status[job_id]['status'] = 'processing'
        processing_status[job_id]['message'] = 'Initializing diffusion models...'
        processing_status[job_id]['progress'] = 10
        
        # Get or initialize generator
        generator = get_diffusion_generator()
        
        if generator is None:
            raise Exception("Failed to initialize diffusion models")
        
        # Load input image
        input_image = Image.open(input_path).convert("RGB")
        
        # Get animation configuration
        anim_config = ANIMATION_TYPES[animation_type]
        num_frames = anim_config['frames']
        
        processing_status[job_id]['progress'] = 20
        processing_status[job_id]['message'] = f'Generating {animation_type} animation with {num_frames} frames...'
        
        # Generate sprite sheet using diffusion
        sprite_sheet = generator.generate_sprite_sheet(
            reference_image=input_image,
            animation_type=animation_type,
            num_frames=num_frames,
            width=128,  # Use smaller size for faster generation
            height=128,
            guidance_scale=3.5,
            num_inference_steps=25,
            seed=42
        )
        
        processing_status[job_id]['progress'] = 90
        processing_status[job_id]['message'] = 'Finalizing sprite sheet...'
        
        # Save result
        output_filename = f"sprite_{animation_type}_{job_id}.png"
        output_path = app.config['RESULTS_FOLDER'] / output_filename
        sprite_sheet.save(str(output_path), "PNG")
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['output_file'] = output_filename
        processing_status[job_id]['message'] = f'{anim_config["name"]} sprite sheet generated using diffusion!'
        processing_status[job_id]['progress'] = 100
        
    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['message'] = f'Error: {str(e)}'
        processing_status[job_id]['progress'] = 0
        print(f"Error processing job {job_id}: {e}")
        traceback.print_exc()

def process_with_transforms(job_id, input_path, animation_type):
    """Process the image using simple transformations (fallback)"""
    try:
        processing_status[job_id]['status'] = 'processing'
        processing_status[job_id]['message'] = f'Generating {animation_type} animation...'
        processing_status[job_id]['progress'] = 10
        
        # Load input image
        input_image = Image.open(input_path).convert("RGBA")
        input_image = input_image.resize((128, 128), Image.LANCZOS)
        
        processing_status[job_id]['progress'] = 30
        
        # Generate frames based on animation type
        if animation_type == 'idle':
            frames = generate_idle_animation(input_image)
        elif animation_type == 'walk':
            frames = generate_walk_animation(input_image)
        elif animation_type == 'run':
            frames = generate_run_animation(input_image)
        elif animation_type == 'jump':
            frames = generate_jump_animation(input_image)
        elif animation_type == 'attack':
            frames = generate_attack_animation(input_image)
        elif animation_type == 'defend':
            frames = generate_defend_animation(input_image)
        elif animation_type == 'crouch':
            frames = generate_crouch_animation(input_image)
        elif animation_type == 'death':
            frames = generate_death_animation(input_image)
        elif animation_type == 'celebrate':
            frames = generate_celebrate_animation(input_image)
        elif animation_type == 'roll':
            frames = generate_roll_animation(input_image)
        else:
            frames = generate_idle_animation(input_image)
        
        processing_status[job_id]['progress'] = 70
        
        # Create sprite sheet
        sprite_sheet = create_sprite_sheet_from_frames(frames)
        
        # Save result
        output_filename = f"sprite_{animation_type}_{job_id}.png"
        output_path = app.config['RESULTS_FOLDER'] / output_filename
        sprite_sheet.save(str(output_path), "PNG")
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['output_file'] = output_filename
        processing_status[job_id]['message'] = f'{ANIMATION_TYPES[animation_type]["name"]} sprite sheet generated!'
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
    return jsonify({
        'diffusion_available': DIFFUSION_AVAILABLE,
        'gpu_available': USE_GPU,
        'gpu_device': torch.cuda.get_device_name(0) if USE_GPU else None,
        'animations_available': len(ANIMATION_TYPES),
        'mode': 'diffusion' if DIFFUSION_AVAILABLE else 'transform'
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Sprite Sheet Diffusion Web Application")
    print("=" * 60)
    print(f"Mode: {'Diffusion Pipeline' if DIFFUSION_AVAILABLE else 'Simple Transformations'}")
    print(f"GPU: {'Enabled' if USE_GPU else 'Disabled'}")
    print("=" * 60)
    
    # Pre-load models if available
    if DIFFUSION_AVAILABLE and USE_GPU:
        print("Pre-loading diffusion models...")
        get_diffusion_generator()
    
    app.run(host='0.0.0.0', port=8080, debug=False)