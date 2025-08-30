"""
Multi-Resolution Pose Guider for UNet3D
This creates pose features with correct channel dimensions for each UNet block
"""

from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin

from models.motion_module import zero_module
from models.resnet import InflatedConv3d


class MultiResPoseGuider(ModelMixin):
    """
    PoseGuider that outputs features with correct channel dimensions for each UNet3D block
    
    UNet3D expects:
    - Block 0 (conv_in output): 320 channels
    - Block 1 (first down): 640 channels  
    - Block 2 (second down): 1280 channels
    - Block 3 (third down): 1280 channels
    """
    
    def __init__(
        self,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
        unet_block_channels: Tuple[int] = (320, 640, 1280, 1280),
    ):
        super().__init__()
        
        self.unet_block_channels = unet_block_channels
        
        # Shared encoder - similar to PoseGuiderOrg
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )
        
        self.encoder_blocks = nn.ModuleList([])
        
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.encoder_blocks.append(
                nn.Sequential(
                    InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1),
                    nn.SiLU(),
                    InflatedConv3d(channel_in, channel_out, kernel_size=3, padding=1, stride=2),
                    nn.SiLU()
                )
            )
        
        # Feature dimension at this point: block_out_channels[-1] = 256
        
        # Multi-resolution output heads - one for each UNet block
        self.output_heads = nn.ModuleList([])
        
        for unet_channels in unet_block_channels:
            self.output_heads.append(
                zero_module(
                    InflatedConv3d(
                        block_out_channels[-1],  # 256
                        unet_channels,
                        kernel_size=3,
                        padding=1,
                    )
                )
            )
    
    def forward(self, conditioning):
        """
        Args:
            conditioning: [B, C, T, H, W] pose conditioning tensor
        
        Returns:
            List of pose features for each UNet block:
            - [B, 320, T, H, W] for block 0
            - [B, 640, T, H//2, W//2] for block 1  
            - [B, 1280, T, H//4, W//4] for block 2
            - [B, 1280, T, H//8, W//8] for block 3
        """
        # Encode pose features
        x = self.conv_in(conditioning)
        x = F.silu(x)
        
        # Store intermediate features at different resolutions
        features_at_resolutions = [x]  # Full resolution
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            features_at_resolutions.append(x)  # 1/2, 1/4, 1/8 resolutions
        
        # Generate outputs for each UNet block
        outputs = []
        
        for i, output_head in enumerate(self.output_heads):
            # Choose the appropriate resolution feature
            if i == 0:
                # Block 0: Full resolution (64x64 in latent space)
                feature = features_at_resolutions[0]
            elif i == 1:
                # Block 1: 1/2 resolution (32x32)
                feature = features_at_resolutions[1]
            elif i == 2:
                # Block 2: 1/4 resolution (16x16)
                feature = features_at_resolutions[2]
            else:
                # Block 3+: 1/8 resolution (8x8)
                feature = features_at_resolutions[3]
            
            # Generate output with correct channels for this UNet block
            output = output_head(feature)
            outputs.append(output)
        
        return outputs


class SingleTensorPoseGuider(ModelMixin):
    """
    Wrapper that converts PoseGuiderOrg single tensor output to multi-resolution format
    This is a simpler solution that doesn't require retraining
    """
    
    def __init__(
        self,
        pose_guider_org,
        unet_block_channels: Tuple[int] = (320, 320, 640, 1280, 1280),  # Corrected based on debug
    ):
        super().__init__()
        
        self.pose_guider_org = pose_guider_org
        self.unet_block_channels = unet_block_channels
        
        # Create channel adaptation layers for each UNet block
        self.channel_adapters = nn.ModuleList([])
        
        # The original PoseGuiderOrg outputs 320 channels
        orig_channels = 320
        
        for target_channels in unet_block_channels:
            if target_channels == orig_channels:
                # No adaptation needed
                self.channel_adapters.append(nn.Identity())
            else:
                # Adapt channels using 1x1 conv
                self.channel_adapters.append(
                    zero_module(
                        InflatedConv3d(
                            orig_channels,
                            target_channels,
                            kernel_size=1,
                            padding=0,
                        )
                    )
                )
    
    def forward(self, conditioning):
        """
        Args:
            conditioning: [B, C, T, H, W] pose conditioning tensor
        
        Returns:
            List of pose features for each UNet block with correct channel dimensions
        """
        # Get the original single feature output
        base_feature = self.pose_guider_org(conditioning)
        
        # Expected UNet spatial resolutions and channels based on actual UNet debug:
        # conv_in: 320ch@64x64, down0: 320ch@32x32, down1: 640ch@16x16, down2: 1280ch@8x8, down3: 1280ch@8x8
        expected_sizes = [64, 32, 16, 8, 8]
        
        outputs = []
        
        # Need to match the length of expected_sizes
        for i, target_size in enumerate(expected_sizes):
            # Resize spatial dimensions if needed
            if base_feature.shape[-1] != target_size or base_feature.shape[-2] != target_size:
                # Resize the spatial dimensions
                B, C, T, H, W = base_feature.shape
                resized_feature = F.interpolate(
                    base_feature.view(B * T, C, H, W),
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                )
                resized_feature = resized_feature.view(B, C, T, target_size, target_size)
            else:
                resized_feature = base_feature
            
            # Use the appropriate adapter (cycle through them if we have more expected sizes than adapters)
            adapter = self.channel_adapters[min(i, len(self.channel_adapters) - 1)]
            
            # Adapt channels
            adapted_feature = adapter(resized_feature)
            outputs.append(adapted_feature)
        
        return outputs