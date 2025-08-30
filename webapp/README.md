# Sprite Sheet Diffusion Web Application

A web application for generating game character sprite sheets with multiple animation types. Upload a single character image and generate sprite sheets for various animations like idle, walk, run, jump, attack, and more.

## Features

- **10 Animation Types**: Idle, Walk, Run, Jump, Attack, Defend, Crouch, Death, Celebrate, Roll
- **Live Animation Preview**: See your sprite animations play in real-time
- **Adjustable Speed Control**: Control animation playback speed (10-200ms per frame)
- **GIF Export**: Download your animations as animated GIF files
- **Drag & Drop Upload**: Easy file upload interface
- **Async Processing**: Non-blocking job queue system for smooth operation

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for AI model inference)
- 8GB+ RAM recommended
- ~10GB disk space for models

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tomc98/Sprite-Sheet-Diffusion.git
cd Sprite-Sheet-Diffusion
```

### 2. Set Up Python Environment

Using venv (recommended if conda is not available):
```bash
python3 -m venv ssd_env
source ssd_env/bin/activate  # On Windows: ssd_env\Scripts\activate
```

Or using conda:
```bash
conda create -n ssd_env python=3.8
conda activate ssd_env
```

### 3. Install Dependencies

```bash
# Install PyTorch (adjust for your CUDA version or use CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r ModelTraining/requirements.txt

# Install additional webapp dependencies
pip install flask flask-cors pillow werkzeug
```

### 4. Download Required Models

#### Option A: Download Pre-trained Models from Google Drive

Download the following models from the Google Drive links in the main README:
- `denoising_unet-30000.pth` (3.2GB)
- `reference_unet-30000.pth` (3.2GB)
- `motion_module.pth` (1.7GB)

Place them in `ModelTraining/pretrained_model/` directory.

#### Option B: Use the Automated Download Script

```bash
# This will download the models from Google Drive
python download_models.py
```

### 5. Download Additional Models (Required)

```bash
# Download required HuggingFace models
python download_hf_models.py
```

Or manually download:
- stabilityai/sd-vae-ft-mse
- runwayml/stable-diffusion-v1-5
- openai/clip-vit-large-patch14

## Running the Web Application

### 1. Start the Flask Server

```bash
cd webapp
python app_v3.py
```

The server will start on `http://localhost:8080`

### 2. Access the Application

Open your web browser and navigate to:
```
http://localhost:8080
```

### 3. For Remote/Cloud Servers

If running on a remote server (e.g., VAST.ai, AWS, etc.), use SSH tunneling:

```bash
ssh -L 8080:localhost:8080 user@your-server-ip
```

Then access the application at `http://localhost:8080` on your local machine.

## Usage

1. **Upload an Image**: 
   - Drag and drop a character image or click to browse
   - Supported formats: PNG, JPG, GIF, BMP
   - Recommended: Square images, 256x256 or larger

2. **Select Animation Type**:
   - Choose from 10 different animation types
   - Each animation has a different number of frames (4-8 frames)

3. **Generate Sprite Sheet**:
   - Click "Generate Sprite Sheet"
   - Wait for processing (usually 5-15 seconds)
   - The sprite sheet will appear with an animated preview

4. **Preview & Export**:
   - Watch the animation preview in real-time
   - Adjust speed with the slider (10-200ms per frame)
   - Use Play/Pause button to control playback
   - Download the sprite sheet as PNG
   - Export as animated GIF

## File Structure

```
webapp/
├── app_v3.py                 # Main Flask application
├── templates/
│   └── index_v2.html        # Web interface
├── static/
│   ├── uploads/             # Uploaded images (auto-created)
│   └── results/             # Generated sprite sheets (auto-created)
└── README.md                # This file
```

## Animation Types

| Animation | Frames | Description |
|-----------|--------|-------------|
| Idle | 4 | Standing animation with subtle breathing effect |
| Walk | 8 | Walking cycle with bob and tilt |
| Run | 8 | Fast running animation with dynamic movement |
| Jump | 6 | Complete jump sequence from crouch to landing |
| Attack | 6 | Combat attack animation with strike motion |
| Defend | 4 | Defensive stance with shield motion |
| Crouch | 4 | Crouching/ducking animation |
| Death | 6 | Defeat animation with progressive fall |
| Celebrate | 8 | Victory celebration with jumping |
| Roll | 8 | Rolling/dodging movement animation |

## API Endpoints

- `GET /` - Main web interface
- `GET /api/animations` - List available animation types
- `POST /api/upload` - Upload image and start processing
- `GET /api/status/<job_id>` - Check processing status
- `GET /api/download/<job_id>` - Download sprite sheet
- `POST /api/export-gif` - Export animation as GIF

## Troubleshooting

### Port Already in Use
```bash
# Kill existing process on port 8080
lsof -ti:8080 | xargs kill -9
```

### Memory Issues
- Reduce batch processing
- Use smaller input images (256x256)
- Ensure sufficient RAM (8GB+)

### Model Loading Errors
- Verify all model files are in `ModelTraining/pretrained_model/`
- Check file permissions
- Ensure sufficient disk space

### CUDA/GPU Issues
- Install CPU version of PyTorch if no GPU available
- Check CUDA compatibility with PyTorch version
- Verify GPU drivers are up to date

## Development

### Running in Debug Mode
```python
# In app_v3.py, change:
app.run(host='0.0.0.0', port=8080, debug=True)
```

### Custom Animation Types
Add new animations by:
1. Define in `ANIMATION_TYPES` dictionary
2. Create `generate_[name]_animation()` function
3. Implement frame generation logic

## Performance Tips

- Use GPU acceleration when available
- Process images in batches for efficiency
- Cache frequently used sprite sheets
- Optimize image sizes before upload

## Security Notes

- The application runs on port 8080 by default
- File uploads are restricted to 16MB
- Only image files are accepted
- Uploaded files are stored temporarily

## License

See the main repository LICENSE file.

## Support

For issues or questions:
- Create an issue on [GitHub](https://github.com/tomc98/Sprite-Sheet-Diffusion/issues)
- Check existing issues for solutions
- Refer to the main project documentation