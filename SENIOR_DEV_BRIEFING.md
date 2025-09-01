# Senior Dev Briefing: Sprite Sheet Diffusion Implementation

## Context & Objective

We're implementing the **Sprite Sheet Diffusion** research paper methodology for generating animated game character sprites using pose-guided diffusion models. The system is 95% complete but needs the final integration fix to achieve true pose-guided generation.

### Research Paper
- **Title**: "Sprite Sheet Diffusion: Generate Game Character for Animation"
- **arXiv**: https://arxiv.org/html/2412.03685v2
- **Authors**: Cheng-An Hsieh, Jing Zhang, Ava Yan
- **Key Innovation**: Adapts "Animate Anyone" for sprite sheet generation using pose conditioning

## Current System Status

### ✅ What's Working
- **All models downloaded and loaded** (9+ GB):
  - Custom fine-tuned weights: `denoising_unet.pth`, `reference_unet.pth`
  - Motion module: `motion_module.pth` 
  - Pose conditioning: `pose_guider.pth`
  - Base models: Stable Diffusion v1.5, VAE, CLIP
- **Environment fully set up**: Python 3.10, CUDA 11.8, RTX 3090
- **Web application functional**: Flask server on port 8080
- **Pipeline components loading**: All models initialize correctly

### ❌ Critical Issue
**The try/catch in the pose conditioning is falling back to the except block instead of succeeding in the try block.**

## Technical Architecture (From Paper)

### Two-Stage Training
1. **Stage 1 (Pose-to-Image)**: Train ReferenceNet + PoseGuider + denoising network
2. **Stage 2 (Pose-to-Sprite)**: Add Motion Module for temporal consistency

### Pipeline Components
1. **ReferenceNet**: Encodes character appearance using spatial-attention
2. **PoseGuider**: Encodes OpenPose conditioning into features
3. **Motion Module**: Ensures smooth frame transitions
4. **UNet3D**: Generates pose-conditioned character frames

## The Problem

### Location
`/root/Sprite-Sheet-Diffusion/ModelTraining/pipelines/pipeline_pose2img.py` lines 111-114:

```python
# Current code (your fix)
try:
    pose_fea = self.pose_guider(pose_cond_tensor, ref_pose_tensor)
except TypeError:
    pose_fea = self.pose_guider(pose_cond_tensor)
```

### Issue
- **Pipeline expects**: `pose_guider(pose_cond_tensor, ref_pose_tensor)` (2 args)
- **PoseGuiderOrg.forward()**: Only accepts 1 argument
- **Currently**: Falls back to except block, but we need the **try block to succeed**

### Evidence
The system generates this output (debugging fallback):
- Character + pose overlay instead of pure pose-guided diffusion
- Processing completes too quickly (~20s instead of proper diffusion time)

## Success Criteria

### Expected Output
The try block should succeed and generate:
- **8 distinct character poses** showing walking motion
- **Each frame different** - character in various walking positions
- **No pose overlays** - pure character variations
- **Longer processing time** - real diffusion inference (~2-3 minutes for 8 frames)

### Test Case
- **Input**: `/root/Generated_Image_August_27__2025_-_11_26PM-removebg-preview.png` (cute fluffy character)
- **Animation**: Walking cycle (8 frames)
- **Expected**: Character showing leg movement, body sway, natural walking motion

## Resources for Development

### Key Files to Examine
```
/root/Sprite-Sheet-Diffusion/
├── ModelTraining/
│   ├── models/pose_guider_org.py           # Core PoseGuider implementation
│   ├── models/pose_guider_multi_res.py     # Multi-resolution adapter  
│   ├── pipelines/pipeline_pose2img.py      # Main diffusion pipeline
│   ├── models/unet_3d.py                   # 3D UNet expecting pose features
│   └── pretrained_model/                   # All model weights (9GB)
├── webapp/app_real_diffusion_pipeline.py   # Current webapp implementation
├── FIX_SYNOPSIS.md                         # Senior dev's previous fixes
├── problem-statement.md                    # Detailed problem analysis
└── TENSOR_DIMENSION_FIX.md                 # Previous dimension fix docs
```

### Working Examples
- **Reference implementation**: `/root/Sprite-Sheet-Diffusion/webapp/app_diffusion_working.py`
- **Test scripts**: `/root/Sprite-Sheet-Diffusion/test_diffusion_pipeline.py`
- **Original repository**: https://github.com/chenganhsieh/Sprite-Sheet-Diffusion

### Environment Setup
```bash
cd /root/Sprite-Sheet-Diffusion
source ssd_env/bin/activate
export PYTHONPATH=/root/Sprite-Sheet-Diffusion/ModelTraining:$PYTHONPATH
```

## AI Prompt for Recursive Improvement

### Task
Fix the Sprite Sheet Diffusion pipeline so that the try block succeeds instead of falling back to the except block, enabling true pose-guided diffusion generation.

### Context
You are working on a research implementation of "Sprite Sheet Diffusion: Generate Game Character for Animation" (arXiv:2412.03685v2). The system uses a two-stage diffusion approach:

1. **ReferenceNet** encodes character appearance 
2. **PoseGuider** converts OpenPose skeletons to conditioning features
3. **UNet3D** with Motion Module generates pose-guided character frames

### Current Issue
The pipeline fails at pose conditioning due to API mismatch:
```python
# This should work but fails with "TypeError: takes 2 positional arguments but 3 were given"
pose_fea = self.pose_guider(pose_cond_tensor, ref_pose_tensor)
```

### Available Resources
- **All models loaded** and accessible in `/root/Sprite-Sheet-Diffusion/ModelTraining/pretrained_model/`
- **Working environment** with CUDA RTX 3090
- **Test character**: `/root/Generated_Image_August_27__2025_-_11_26PM-removebg-preview.png`
- **Pipeline code**: `/root/Sprite-Sheet-Diffusion/ModelTraining/pipelines/pipeline_pose2img.py`
- **PoseGuider**: `/root/Sprite-Sheet-Diffusion/ModelTraining/models/pose_guider_org.py`

### Success Validation
After each fix attempt, test using:

```bash
cd /root/Sprite-Sheet-Diffusion/webapp
source ../ssd_env/bin/activate
python3 << 'EOF'
import requests, json, time, shutil
from pathlib import Path

# Test the fix
shutil.copy2("/root/Generated_Image_August_27__2025_-_11_26PM-removebg-preview.png", "test.png")

with open("test.png", 'rb') as f:
    files = {'file': f}
    data = {'animation_type': 'walk'}
    response = requests.post('http://localhost:8080/api/upload', files=files, data=data)
    job_id = response.json()['job_id']
    
    # Monitor - should take 2-3 minutes for real diffusion
    start_time = time.time()
    while time.time() - start_time < 300:
        status = requests.get(f'http://localhost:8080/api/status/{job_id}').json()
        print(f"[{int(time.time()-start_time)}s] {status['progress']}% - {status['message']}")
        
        if status['status'] == 'completed':
            output_file = status['output_file'] 
            print(f"✅ Success: {output_file}")
            break
        elif status['status'] == 'error':
            print(f"❌ Error: {status['message']}")
            break
        time.sleep(10)
EOF
```

### Success Indicators
- **Processing time**: 2-3 minutes (not 20 seconds)
- **Output**: 8 visually different character poses in walking motion
- **No pose overlays**: Pure character variations, no debug stick figures
- **Log shows**: "Running diffusion inference..." messages for each frame

### Failure Indicators
- **Quick completion**: Under 30 seconds = fallback path
- **Same character repeated**: No pose variation = not using diffusion
- **Pose overlays visible**: Black squares with stick figures = debugging mode
- **TypeError in logs**: PoseGuider call failing

### Debugging Commands
```bash
# Check webapp logs
tail -f /root/Sprite-Sheet-Diffusion/webapp/webapp_fixed.log

# Verify models
python -c "
import sys; sys.path.append('/root/Sprite-Sheet-Diffusion/ModelTraining')
from models.pose_guider_org import PoseGuiderOrg
print('PoseGuiderOrg.forward signature:', PoseGuiderOrg.forward.__code__.co_varnames)
"

# Test pose guider directly
python -c "
import torch, sys
sys.path.append('/root/Sprite-Sheet-Diffusion/ModelTraining')
from models.pose_guider_org import PoseGuiderOrg

pg = PoseGuiderOrg(conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256))
test_tensor = torch.randn(1, 3, 512, 512)
try:
    result = pg(test_tensor)
    print('✅ Single arg works:', result.shape if hasattr(result, 'shape') else type(result))
except Exception as e:
    print('❌ Single arg failed:', e)

try:
    result = pg(test_tensor, test_tensor)  
    print('✅ Two args works:', result.shape if hasattr(result, 'shape') else type(result))
except Exception as e:
    print('❌ Two args failed:', e)
"
```

### Recursive Improvement Strategy
1. **Analyze** the PoseGuiderOrg.forward() method signature
2. **Test** different argument combinations to find what works
3. **Modify** either the pipeline call or PoseGuider interface
4. **Validate** using the test script above
5. **If still failing**: Analyze error, adjust approach, repeat
6. **Continue** until the try block succeeds and generates real pose variations

### Key Constraints
- **Cannot retrain models** - must work with existing `pose_guider.pth` weights
- **Must maintain compatibility** with the existing UNet3D expectations
- **Should follow paper methodology** - ReferenceNet + PoseGuider + Motion Module

### Reference Papers
- **Main paper**: "Sprite Sheet Diffusion: Generate Game Character for Animation"
- **Base work**: "Animate Anyone: Consistent and Controllable Image-to-Video Synthesis"
- **Architecture**: Built on Stable Diffusion v1.5 with custom attention mechanisms

Continue iterating until the diffusion pipeline generates 8 distinct character poses showing realistic walking motion, with each frame being a unique pose-guided generation rather than the same character with debugging overlays.