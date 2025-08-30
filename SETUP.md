# Complete Setup Guide for Sprite Sheet Diffusion

This guide provides detailed instructions for setting up the Sprite Sheet Diffusion project, including both the model training environment and the web application.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Detailed Installation](#detailed-installation)
4. [Running the Web Application](#running-the-web-application)
5. [Model Training Setup](#model-training-setup)
6. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum
- **Storage**: 15GB free space
- **GPU**: Optional but recommended (NVIDIA with CUDA 11.8+)

### Recommended Requirements
- **RAM**: 16GB or more
- **GPU**: NVIDIA RTX 3060 or better with 8GB+ VRAM
- **Storage**: 25GB free space (for models and datasets)

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/tomc98/Sprite-Sheet-Diffusion.git
cd Sprite-Sheet-Diffusion

# 2. Create virtual environment
python3 -m venv ssd_env
source ssd_env/bin/activate  # On Windows: ssd_env\Scripts\activate

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r ModelTraining/requirements.txt

# 4. Download models
python download_models.py  # For Google Drive models
python download_hf_models.py  # For HuggingFace models

# 5. Run the web app
cd webapp
python app_v3.py

# 6. Open browser to http://localhost:8080
```

## Detailed Installation

### Step 1: Environment Setup

#### Option A: Using Python venv (Recommended)
```bash
# Create virtual environment
python3 -m venv ssd_env

# Activate environment
# Linux/macOS:
source ssd_env/bin/activate
# Windows:
ssd_env\Scripts\activate
```

#### Option B: Using Conda
```bash
# Create conda environment
conda create -n sprite-diffusion python=3.8
conda activate sprite-diffusion
```

### Step 2: Install PyTorch

Choose the appropriate command for your system:

#### With CUDA 11.8 (GPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### With CUDA 12.1 (GPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CPU Only
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Dependencies

```bash
# Core dependencies
pip install -r ModelTraining/requirements.txt

# Fix version conflicts if they occur
pip install numpy==1.24.4
pip install huggingface-hub==0.19.4
pip install mediapipe==0.10.11

# Web application dependencies
pip install flask flask-cors pillow werkzeug
```

### Step 4: Download Pre-trained Models

#### Required Models from Google Drive

The following models need to be downloaded from Google Drive (links in main README):

1. **Denoising UNet** (~3.2GB)
   - File: `denoising_unet-30000.pth`
   - Place in: `ModelTraining/pretrained_model/`

2. **Reference UNet** (~3.2GB)
   - File: `reference_unet-30000.pth`
   - Place in: `ModelTraining/pretrained_model/`

3. **Motion Module** (~1.7GB)
   - File: `motion_module.pth`
   - Place in: `ModelTraining/pretrained_model/`

#### Automated Download Script
```bash
# This script helps download from Google Drive
python download_models.py
```

#### Required Models from HuggingFace

```bash
# Download additional required models
python download_hf_models.py

# IMPORTANT: Also download the pose_guider model (not included in Google Drive)
python download_pose_guider.py
```

**Note:** The pose_guider.pth (~4.2MB) is essential for the diffusion pipeline but wasn't included in the Google Drive models. It must be downloaded separately from HuggingFace.

This downloads:
- `stabilityai/sd-vae-ft-mse` - VAE for image encoding
- `runwayml/stable-diffusion-v1-5` - Base diffusion model
- `openai/clip-vit-large-patch14` - CLIP embeddings

### Step 5: Create Symbolic Links (if needed)

```bash
cd ModelTraining/pretrained_model

# Create links for model files
ln -s denoising_unet-30000.pth denoising_unet.pth
ln -s reference_unet-30000.pth reference_unet.pth
```

## Running the Web Application

### Local Development

```bash
# Navigate to webapp directory
cd webapp

# Run the Flask application
python app_v3.py

# The app will be available at http://localhost:8080
```

### Production Deployment

For production, use a WSGI server:

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 app_v3:app
```

### Remote Server Access

If running on a remote server (VAST.ai, AWS, etc.):

```bash
# Create SSH tunnel from your local machine
ssh -L 8080:localhost:8080 -p [SSH_PORT] user@[SERVER_IP]

# Example for VAST:
ssh -L 8080:localhost:8080 -p 40180 root@171.6.52.112
```

Then access at `http://localhost:8080` on your local browser.

## Model Training Setup

### Preparing Dataset

1. Create dataset directory structure:
```bash
mkdir -p game_animation/characters/[character_name]/motions/[motion_name]/{poses,ground_truth}
```

2. Add character images:
- `main_reference.png` - Main character reference
- Motion-specific references in each motion folder

3. Generate dataset JSON:
```bash
cd ModelTraining
python create_json.py
```

### Training Configuration

Edit `ModelTraining/configs/prompts/test_cases.py` to define training cases.

### Start Training

```bash
cd ModelTraining
python train.py --config configs/training_config.yaml
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```python
# Reduce batch size in config
# Or use CPU mode by setting:
export CUDA_VISIBLE_DEVICES=""
```

#### 2. Missing Model Files
```bash
# Verify all models are downloaded
ls -la ModelTraining/pretrained_model/

# Re-download if missing
python download_models.py
python download_hf_models.py
```

#### 3. Port Already in Use
```bash
# Find and kill process using port 8080
lsof -ti:8080 | xargs kill -9

# Or use a different port
python app_v3.py --port 8081
```

#### 4. ImportError: No module named 'torch'
```bash
# Ensure virtual environment is activated
source ssd_env/bin/activate  # Linux/macOS
# or
ssd_env\Scripts\activate  # Windows

# Reinstall PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 5. HuggingFace Download Issues
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface
python download_hf_models.py

# Or set proxy if needed
export HF_ENDPOINT=https://hf-mirror.com
```

#### 6. NumPy Version Conflicts
```bash
# Force install compatible version
pip install numpy==1.24.4 --force-reinstall
```

## Testing Installation

Run the test script to verify everything is working:

```bash
# Test animation generation
python test_animations.py

# Test web API
python test_animation_preview.py
```

## Project Structure

```
Sprite-Sheet-Diffusion/
├── ModelTraining/
│   ├── pretrained_model/      # Pre-trained models go here
│   ├── configs/                # Training configurations
│   ├── requirements.txt        # Python dependencies
│   └── train.py               # Training script
├── webapp/
│   ├── app_v3.py              # Flask application
│   ├── templates/             # HTML templates
│   │   └── index_v2.html      # Web interface
│   ├── static/                # Static files
│   └── README.md              # Webapp documentation
├── download_models.py         # Google Drive model downloader
├── download_hf_models.py      # HuggingFace model downloader
├── test_animations.py         # Animation test script
├── SETUP.md                   # This file
└── README.md                  # Main project README
```

## Next Steps

1. **Try the Web App**: Upload an image and generate sprite sheets
2. **Explore Animations**: Test all 10 animation types
3. **Export GIFs**: Create animated GIFs from sprite sheets
4. **Train Custom Models**: Use your own dataset for training
5. **Customize Animations**: Modify animation generation functions

## Getting Help

- Check the [Issues](https://github.com/tomc98/Sprite-Sheet-Diffusion/issues) page
- Read the webapp [README](webapp/README.md) for detailed usage
- Review the main project documentation

## License

See the LICENSE file in the repository root.