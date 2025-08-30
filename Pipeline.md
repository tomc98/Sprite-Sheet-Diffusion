# Sprite Sheet Diffusion Pipeline Documentation

This document provides a comprehensive overview of the Sprite Sheet Diffusion web application pipeline, from frontend interaction to backend diffusion processing.

## Complete System Architecture

```mermaid
graph TB
    subgraph "Frontend (Browser)"
        A[User Interface] --> B[File Upload]
        B --> C[Animation Type Selection]
        C --> D[Submit Request]
    end
    
    subgraph "Flask Web Server"
        D --> E[API Upload Endpoint]
        E --> F[File Validation Storage]
        F --> G[Generate Unique Job ID]
        G --> H[Start Background Thread]
        H --> I[Return Job ID to Frontend]
        
        I --> J[Frontend Polls API Status]
        J --> K[Check Job Status]
        K --> L{Job Complete?}
        L -->|No| J
        L -->|Yes| M[Return Results]
    end
    
    subgraph "Background Processing Thread"
        H --> N[Load Reference Image]
        N --> O[Generate Animation Frames]
        O --> P[Save Results]
        P --> Q[Update Job Status]
    end
    
    subgraph "Diffusion Pipeline Core"
        O --> R[Initialize Models]
        R --> S[For Each Animation Frame]
        S --> T[Generate Pose Image]
        T --> U[Run Diffusion Process]
        U --> V[Save Generated Frame]
        V --> W{More Frames?}
        W -->|Yes| S
        W -->|No| X[Combine into Sprite Sheet]
        X --> P
    end
```

## Detailed Backend Processing Pipeline

```mermaid
flowchart TD
    subgraph "1. Model Initialization (Done Once at Startup)"
        A1[Load Configuration] --> A2[Initialize VAE AutoencoderKL]
        A2 --> A3[Load Reference UNet2D]
        A3 --> A4[Load Denoising UNet3D]
        A4 --> A5[Load PoseGuiderOrg Model]
        A5 --> A6[Load CLIP Vision Encoder]
        A6 --> A7[Create DDIM Scheduler]
        A7 --> A8[Initialize OpenPose Detector]
        A8 --> A9[Create Simplified Pipeline]
        A9 --> A10[Move Models to GPU]
    end
    
    subgraph "2. Request Processing"
        B1[Receive Upload Request] --> B2[Validate File Format]
        B2 --> B3[Resize Image to 512x512]
        B3 --> B4[Save to Upload Directory]
        B4 --> B5[Generate UUID Job ID]
        B5 --> B6[Create Job Entry]
        B6 --> B7[Start Background Thread]
    end
    
    subgraph "3. Animation Frame Generation Loop"
        C1[Load Animation Type Poses] --> C2[For Each Frame Position]
        C2 --> C3[Generate Pose Skeleton]
        C3 --> C4[Create OpenPose-Style Image]
        C4 --> C5[Run Diffusion Generation]
        C5 --> C6[Save Individual Frame]
        C6 --> C7{More Frames?}
        C7 -->|Yes| C2
        C7 -->|No| C8[Combine All Frames]
        C8 --> C9[Create Final Sprite Sheet]
    end
    
    subgraph "4. Pose Generation Process"
        D1[Determine Animation Type] --> D2{Animation Type}
        D2 -->|walk| D3[Generate Walking Cycle]
        D2 -->|run| D4[Generate Running Cycle]
        D2 -->|idle| D5[Generate Idle Animation]
        D2 -->|jump| D6[Generate Jump Sequence]
        D2 -->|attack| D7[Generate Attack Animation]
        
        D3 --> D8[Create Stick Figure Pose]
        D4 --> D8
        D5 --> D8
        D6 --> D8
        D7 --> D8
        
        D8 --> D9[Apply OpenPose Colors]
        D9 --> D10[Convert to RGB Image]
    end
    
    subgraph "5. Diffusion Process Detail"
        E1[Prepare Reference Image] --> E2[Extract CLIP Embeddings]
        E2 --> E3[Encode Reference to Latents]
        E3 --> E4[Process Pose Image]
        E4 --> E5[Generate Pose Features]
        E5 --> E6[Setup Classifier-Free Guidance]
        E6 --> E7[Initialize Random Latents]
        
        E7 --> E8[Denoising Loop - 25 Steps]
        E8 --> E9[Reference UNet Forward Pass]
        E9 --> E10[Update Reference Features]
        E10 --> E11[Scale Latent Input]
        E11 --> E12[Denoising UNet Forward]
        E12 --> E13[Apply Pose Conditioning]
        E13 --> E14[Classifier-Free Guidance]
        E14 --> E15[Scheduler Step]
        E15 --> E16{More Steps?}
        E16 -->|Yes| E11
        E16 -->|No| E17[Decode Latents to Image]
    end
    
    A10 --> B1
    B7 --> C1
    C5 --> E1
    E17 --> C6
    C9 --> F1[Update Job Status to Complete]
```

## Model Architecture Details

```mermaid
graph LR
    subgraph "Input Processing"
        I1[Reference Image<br/>512x512x3] --> I2[CLIP Vision Encoder]
        I2 --> I3[Image Embeddings<br/>1x257x768]
        
        I4[Pose Image<br/>512x512x3] --> I5[PoseGuiderOrg]
        I5 --> I6[Pose Features<br/>List of Tensors]
    end
    
    subgraph "Latent Space Processing"
        L1[VAE Encoder] --> L2[Reference Latents<br/>1x4x64x64]
        L3[Random Noise<br/>1x4x1x64x64] --> L4[Denoising Process]
    end
    
    subgraph "UNet Architecture"
        U1[Reference UNet2D<br/>Spatial Attention] --> U2[Feature Writer]
        U2 --> U3[Cross-Attention Features]
        
        U4[Denoising UNet3D<br/>Temporal Consistency] --> U5[Feature Reader]
        U5 --> U6[Pose Conditioning]
        U6 --> U7[Noise Prediction]
    end
    
    subgraph "Output Generation"
        O1[VAE Decoder] --> O2[Final Image<br/>512x512x3]
    end
    
    I3 --> U1
    I6 --> U6
    L2 --> U1
    L4 --> U4
    U3 --> U5
    U7 --> L4
    L4 --> O1
```

## Pose Generation Coordinate System

```mermaid
graph TD
    subgraph "OpenPose Color Mapping"
        P1[Body Parts] --> P2[Head: Yellow #FFFF00]
        P1 --> P3[Neck: Orange #FF8000]
        P1 --> P4[Body: Magenta #FF00FF]
        P1 --> P5[Left Arm: Green #00FF00]
        P1 --> P6[Right Arm: Red #FF0000]
        P1 --> P7[Left Leg: Cyan #00FFFF]
        P1 --> P8[Right Leg: Orange #FF8000]
    end
    
    subgraph "Animation Cycles"
        A1[Walk Cycle - 8 Frames] --> A2[Frame 0: Standing]
        A2 --> A3[Frame 1: Left foot forward]
        A3 --> A4[Frame 2: Mid-stride]
        A4 --> A5[Frame 3: Right foot forward]
        A5 --> A6[Frame 4: Standing]
        A6 --> A7[Frame 5: Right foot forward]
        A7 --> A8[Frame 6: Mid-stride]
        A8 --> A9[Frame 7: Left foot forward]
    end
```

## File System Structure

```mermaid
graph TD
    subgraph "Directory Structure"
        D1[webapp/] --> D2[static/uploads/]
        D1 --> D3[static/results/]
        D1 --> D4[templates/]
        
        D2 --> D5[Original uploaded images]
        D3 --> D6[Generated sprite sheets]
        D4 --> D7[HTML templates]
    end
    
    subgraph "File Naming Convention"
        F1[Upload] --> F2[uuid_timestamp_originalname.ext]
        F3[Result] --> F4[sprite_animtype_uuid.png]
    end
```

## Error Handling Flow

```mermaid
graph TD
    E1[Process Request] --> E2{Validation OK?}
    E2 -->|No| E3[Return Error Response]
    E2 -->|Yes| E4[Start Processing]
    
    E4 --> E5{Model Load OK?}
    E5 -->|No| E6[Log Error, Return Fallback]
    E5 -->|Yes| E7[Generate Frames]
    
    E7 --> E8{Generation OK?}
    E8 -->|No| E9[Log Error, Create Error Image]
    E8 -->|Yes| E10[Save Results]
    
    E9 --> E11[Update Status: Error]
    E10 --> E12[Update Status: Complete]
```

## Current Technical Issues

### Tensor Size Mismatch
The current implementation has a dimension mismatch issue:
- **Problem**: `The size of tensor a (32) must match the size of tensor b (64) at non-singleton dimension 4`
- **Location**: UNet3D forward pass when adding pose features
- **Cause**: PoseGuiderOrg output dimensions don't match UNet3D expectations at different resolution levels

### Proposed Solutions
1. **Resize pose features** to match expected dimensions at each UNet block
2. **Use interpolation** to adapt single pose feature to multiple resolution levels
3. **Switch to regular PoseGuider** if compatible weights are available

## Performance Characteristics

```mermaid
graph LR
    subgraph "Processing Times"
        T1[Model Loading: ~30s] --> T2[Image Upload: ~1s]
        T2 --> T3[Single Frame: ~3s]
        T3 --> T4[8-frame Animation: ~24s]
        T4 --> T5[Sprite Sheet Assembly: ~1s]
    end
    
    subgraph "Memory Usage"
        M1[Models: ~7GB VRAM] --> M2[Processing: ~1GB VRAM]
        M2 --> M3[Total: ~8GB VRAM]
    end
```

## API Endpoints

```mermaid
sequenceDiagram
    participant Frontend
    participant Flask
    participant Background
    
    Frontend->>Flask: POST /api/upload (image + animation_type)
    Flask->>Flask: Validate & save file
    Flask->>Background: Start generation thread
    Flask-->>Frontend: Return job_id
    
    loop Status Polling
        Frontend->>Flask: GET /api/status/{job_id}
        Flask-->>Frontend: Return progress/status
    end
    
    Background->>Background: Generate animation frames
    Background->>Flask: Update job status
    
    Frontend->>Flask: GET /static/results/{sprite_sheet}
    Flask-->>Frontend: Return final sprite sheet
```

This pipeline handles the complete process from user interaction to final sprite sheet generation using the Sprite Sheet Diffusion model with pose conditioning and temporal consistency.