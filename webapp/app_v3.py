#!/usr/bin/env python3
"""
Sprite Sheet Diffusion Web Application Backend - With Animation Selection
"""

import os
import sys
import json
import uuid
import shutil
import threading
import traceback
import math
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

# Animation types with frame configurations
ANIMATION_TYPES = {
    'idle': {
        'name': 'Idle',
        'description': 'Standing idle animation with subtle movement',
        'frames': 4,
        'generator': 'generate_idle_animation'
    },
    'walk': {
        'name': 'Walk',
        'description': 'Walking cycle animation',
        'frames': 8,
        'generator': 'generate_walk_animation'
    },
    'run': {
        'name': 'Run',
        'description': 'Running cycle animation',
        'frames': 8,
        'generator': 'generate_run_animation'
    },
    'jump': {
        'name': 'Jump',
        'description': 'Jump animation sequence',
        'frames': 6,
        'generator': 'generate_jump_animation'
    },
    'attack': {
        'name': 'Attack',
        'description': 'Attack/punch animation',
        'frames': 6,
        'generator': 'generate_attack_animation'
    },
    'defend': {
        'name': 'Defend',
        'description': 'Defensive stance animation',
        'frames': 4,
        'generator': 'generate_defend_animation'
    },
    'crouch': {
        'name': 'Crouch',
        'description': 'Crouching animation',
        'frames': 4,
        'generator': 'generate_crouch_animation'
    },
    'death': {
        'name': 'Death',
        'description': 'Death/defeat animation',
        'frames': 6,
        'generator': 'generate_death_animation'
    },
    'celebrate': {
        'name': 'Celebrate',
        'description': 'Victory celebration animation',
        'frames': 8,
        'generator': 'generate_celebrate_animation'
    },
    'roll': {
        'name': 'Roll',
        'description': 'Rolling/dodging animation',
        'frames': 8,
        'generator': 'generate_roll_animation'
    }
}

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
    
    # Get animation type from request
    animation_type = request.form.get('animation_type', 'idle')
    if animation_type not in ANIMATION_TYPES:
        animation_type = 'idle'
    
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
            'progress': 0,
            'animation_type': animation_type
        }
        
        # Start processing in background thread
        thread = threading.Thread(target=process_image_with_animation, 
                                 args=(job_id, filepath, animation_type))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': 'File uploaded and processing started',
            'filename': unique_filename,
            'animation_type': animation_type
        }), 200
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_image_with_animation(job_id, input_path, animation_type):
    """Process the image with selected animation type"""
    try:
        processing_status[job_id]['status'] = 'processing'
        processing_status[job_id]['message'] = f'Starting {ANIMATION_TYPES[animation_type]["name"]} animation generation...'
        processing_status[job_id]['progress'] = 10
        
        # Load the input image
        input_image = Image.open(input_path).convert("RGBA")
        
        # Resize to standard size
        target_size = (128, 128)
        input_image = input_image.resize(target_size, Image.LANCZOS)
        
        processing_status[job_id]['progress'] = 30
        processing_status[job_id]['message'] = f'Creating {animation_type} animation frames...'
        
        # Generate animation frames based on type
        generator_func = globals()[ANIMATION_TYPES[animation_type]['generator']]
        frames = generator_func(input_image)
        
        processing_status[job_id]['progress'] = 70
        processing_status[job_id]['message'] = 'Assembling sprite sheet...'
        
        # Create sprite sheet
        sprite_sheet = create_sprite_sheet_from_frames(frames)
        
        # Save result
        output_filename = f"sprite_{animation_type}_{job_id}.png"
        output_path = app.config['RESULTS_FOLDER'] / output_filename
        sprite_sheet.save(str(output_path), "PNG")
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['output_file'] = output_filename
        processing_status[job_id]['message'] = f'{ANIMATION_TYPES[animation_type]["name"]} sprite sheet generated successfully!'
        processing_status[job_id]['progress'] = 100
        
    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['message'] = f'Error: {str(e)}'
        processing_status[job_id]['progress'] = 0
        print(f"Error processing job {job_id}: {e}")
        traceback.print_exc()

# Animation generation functions
def generate_idle_animation(base_image):
    """Generate idle animation frames"""
    frames = []
    width, height = base_image.size
    
    for i in range(4):
        frame = base_image.copy()
        
        # Subtle breathing effect
        scale = 1.0 + math.sin(i * math.pi / 2) * 0.02
        new_height = int(height * scale)
        frame = frame.resize((width, new_height), Image.LANCZOS)
        
        # Center in original size
        final_frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        y_offset = (height - new_height) // 2
        final_frame.paste(frame, (0, y_offset))
        
        frames.append(final_frame)
    
    return frames

def generate_walk_animation(base_image):
    """Generate walking animation frames"""
    frames = []
    width, height = base_image.size
    
    for i in range(8):
        frame = base_image.copy()
        
        # Walking bob and tilt
        angle = math.sin(i * math.pi / 4) * 3  # Slight body sway
        y_offset = abs(math.sin(i * math.pi / 4)) * 5  # Vertical bob
        x_offset = math.sin(i * math.pi / 4) * 2  # Slight horizontal sway
        
        frame = frame.rotate(angle, expand=False, fillcolor=(0, 0, 0, 0))
        
        # Apply movement
        final_frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        final_frame.paste(frame, (int(x_offset), int(y_offset)))
        
        frames.append(final_frame)
    
    return frames

def generate_run_animation(base_image):
    """Generate running animation frames"""
    frames = []
    width, height = base_image.size
    
    for i in range(8):
        frame = base_image.copy()
        
        # More extreme movement than walking
        angle = math.sin(i * math.pi / 4) * 5  # Body lean while running
        y_offset = abs(math.sin(i * math.pi / 4)) * 10  # More vertical movement
        x_offset = math.sin(i * math.pi / 4) * 3  # Horizontal sway
        
        # Lean forward slightly
        frame = frame.transform(
            (width, height),
            Image.AFFINE,
            (1, -0.05, 0, 0, 1, 0),
            fillcolor=(0, 0, 0, 0)
        )
        
        frame = frame.rotate(angle, expand=False, fillcolor=(0, 0, 0, 0))
        
        final_frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        final_frame.paste(frame, (int(x_offset), int(y_offset)))
        
        frames.append(final_frame)
    
    return frames

def generate_jump_animation(base_image):
    """Generate jump animation frames"""
    frames = []
    width, height = base_image.size
    
    # Jump sequence: crouch, launch, apex, fall, land, recover
    sequences = [
        {'scale_y': 0.85, 'y_pos': 0.15},  # Crouch
        {'scale_y': 1.1, 'y_pos': -0.05},   # Launch
        {'scale_y': 1.0, 'y_pos': -0.3},    # Rising
        {'scale_y': 1.0, 'y_pos': -0.4},    # Apex
        {'scale_y': 1.0, 'y_pos': -0.2},    # Falling
        {'scale_y': 0.9, 'y_pos': 0.1},     # Landing
    ]
    
    for seq in sequences:
        frame = base_image.copy()
        
        # Apply vertical scaling
        new_height = int(height * seq['scale_y'])
        frame = frame.resize((width, new_height), Image.LANCZOS)
        
        # Position in frame
        final_frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        y_offset = int(height * seq['y_pos']) + (height - new_height) // 2
        final_frame.paste(frame, (0, y_offset))
        
        frames.append(final_frame)
    
    return frames

def generate_attack_animation(base_image):
    """Generate attack animation frames"""
    frames = []
    width, height = base_image.size
    
    # Attack sequence: prepare, strike, follow-through
    sequences = [
        {'angle': 5, 'x_shift': -10},    # Pull back
        {'angle': 0, 'x_shift': -5},     # Prepare
        {'angle': -15, 'x_shift': 20},   # Strike
        {'angle': -10, 'x_shift': 15},   # Follow through
        {'angle': -5, 'x_shift': 5},     # Recovery
        {'angle': 0, 'x_shift': 0},      # Return
    ]
    
    for seq in sequences:
        frame = base_image.copy()
        
        # Rotation for strike
        frame = frame.rotate(seq['angle'], expand=False, fillcolor=(0, 0, 0, 0))
        
        # Horizontal movement
        final_frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        final_frame.paste(frame, (seq['x_shift'], 0))
        
        frames.append(final_frame)
    
    return frames

def generate_defend_animation(base_image):
    """Generate defensive stance animation"""
    frames = []
    width, height = base_image.size
    
    for i in range(4):
        frame = base_image.copy()
        
        # Defensive lean back
        angle = 10 + math.sin(i * math.pi / 2) * 5
        frame = frame.rotate(angle, expand=False, fillcolor=(0, 0, 0, 0))
        
        # Slight crouch
        scale_y = 0.95 - math.sin(i * math.pi / 2) * 0.05
        new_height = int(height * scale_y)
        frame = frame.resize((width, new_height), Image.LANCZOS)
        
        final_frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        final_frame.paste(frame, (0, height - new_height))
        
        frames.append(final_frame)
    
    return frames

def generate_crouch_animation(base_image):
    """Generate crouching animation"""
    frames = []
    width, height = base_image.size
    
    # Crouch sequence
    scales = [1.0, 0.9, 0.75, 0.7]
    
    for scale in scales:
        frame = base_image.copy()
        
        # Vertical compression
        new_height = int(height * scale)
        new_width = int(width * (1 + (1 - scale) * 0.3))  # Slight widening
        frame = frame.resize((new_width, new_height), Image.LANCZOS)
        
        # Position at bottom
        final_frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        x_offset = (width - new_width) // 2
        y_offset = height - new_height
        final_frame.paste(frame, (x_offset, y_offset))
        
        frames.append(final_frame)
    
    return frames

def generate_death_animation(base_image):
    """Generate death/defeat animation"""
    frames = []
    width, height = base_image.size
    
    for i in range(6):
        frame = base_image.copy()
        
        # Progressive fall
        angle = -i * 15  # Rotate progressively
        opacity = int(255 * (1 - i * 0.15))  # Fade out
        
        frame = frame.rotate(angle, expand=False, fillcolor=(0, 0, 0, 0))
        
        # Apply opacity
        frame.putalpha(opacity)
        
        # Fall down
        y_offset = int(i * height * 0.1)
        
        final_frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        final_frame.paste(frame, (0, y_offset), frame)
        
        frames.append(final_frame)
    
    return frames

def generate_celebrate_animation(base_image):
    """Generate celebration animation"""
    frames = []
    width, height = base_image.size
    
    for i in range(8):
        frame = base_image.copy()
        
        # Jumping celebration
        y_offset = -abs(math.sin(i * math.pi / 4)) * 20
        rotation = math.sin(i * math.pi / 4) * 10
        
        # Slight size variation for excitement
        scale = 1.0 + abs(math.sin(i * math.pi / 4)) * 0.1
        new_size = (int(width * scale), int(height * scale))
        frame = frame.resize(new_size, Image.LANCZOS)
        
        frame = frame.rotate(rotation, expand=False, fillcolor=(0, 0, 0, 0))
        
        # Center in frame
        final_frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        x_offset = (width - new_size[0]) // 2
        y_offset = int(y_offset + (height - new_size[1]) // 2)
        final_frame.paste(frame, (x_offset, y_offset))
        
        frames.append(final_frame)
    
    return frames

def generate_roll_animation(base_image):
    """Generate rolling/dodging animation"""
    frames = []
    width, height = base_image.size
    
    for i in range(8):
        frame = base_image.copy()
        
        # 360 degree roll
        angle = i * 45
        frame = frame.rotate(angle, expand=False, fillcolor=(0, 0, 0, 0))
        
        # Move horizontally during roll
        x_offset = int(i * width * 0.1)
        
        # Compress during roll
        if i in [2, 3, 4, 5]:
            scale = 0.8
            new_height = int(height * scale)
            frame = frame.resize((width, new_height), Image.LANCZOS)
            y_offset = (height - new_height) // 2
        else:
            y_offset = 0
        
        final_frame = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        final_frame.paste(frame, (x_offset % width, y_offset))
        
        frames.append(final_frame)
    
    return frames

def create_sprite_sheet_from_frames(frames, columns=None):
    """Create a sprite sheet from a list of frames"""
    if not frames:
        return None
    
    frame_width, frame_height = frames[0].size
    
    # Always use single row for consistency with animation preview
    # This ensures the sprite sheet is always horizontal
    columns = len(frames)
    rows = 1
    
    # Create sprite sheet
    sprite_sheet = Image.new('RGBA', 
                            (frame_width * columns, frame_height * rows),
                            (0, 0, 0, 0))
    
    # Place frames horizontally
    for i, frame in enumerate(frames):
        x = i * frame_width
        y = 0
        sprite_sheet.paste(frame, (x, y))
    
    # Add subtle grid lines
    draw = ImageDraw.Draw(sprite_sheet)
    for i in range(1, columns):
        x = i * frame_width
        draw.line([(x, 0), (x, sprite_sheet.height)], fill=(200, 200, 200, 30), width=1)
    for i in range(1, rows):
        y = i * frame_height
        draw.line([(0, y), (sprite_sheet.width, y)], fill=(200, 200, 200, 30), width=1)
    
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

@app.route('/api/export-gif', methods=['POST'])
def export_gif():
    """Export the sprite sheet as an animated GIF"""
    try:
        job_id = request.form.get('job_id')
        frame_delay = int(request.form.get('frame_delay', 50))  # milliseconds
        
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
            duration=frame_delay,  # Duration in milliseconds
            loop=0  # Loop forever
        )
        
        # Send the GIF file
        return send_file(str(gif_path), mimetype='image/gif', as_attachment=True, download_name=gif_filename)
        
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/test')
def test():
    """Test endpoint"""
    return jsonify({
        'status': 'ok',
        'animations_available': len(ANIMATION_TYPES)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)