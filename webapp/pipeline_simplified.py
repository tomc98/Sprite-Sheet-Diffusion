"""
Simplified pipeline for PoseGuiderOrg that only takes one argument
Based on the original pipeline but adapted for single-image inference
"""

import torch
from typing import Optional, Union, List, Callable
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler
from diffusers.models import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModelWithProjection
from PIL import Image
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "ModelTraining"))

from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.mutual_self_attention import ReferenceAttentionControl

@dataclass
class Pose2ImagePipelineOutput(BaseOutput):
    images: Union[torch.Tensor, np.ndarray]

class SimplifiedPose2ImagePipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModelWithProjection,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider,
        scheduler: DDIMScheduler,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )
        
    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        width: int,
        height: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        
        if generator is not None:
            latents = torch.randn(shape, generator=generator, dtype=dtype).to(device)
        else:
            latents = torch.randn(shape, dtype=dtype).to(device)
        
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def prepare_extra_step_kwargs(self, generator, eta):
        extra_step_kwargs = {}
        if eta is not None:
            extra_step_kwargs["eta"] = eta
        if generator is not None:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def decode_latents(self, latents):
        video = []
        for frame_idx in range(latents.shape[2]):
            latents_frame = latents[:, :, frame_idx, :, :]
            frame = self.vae.decode(latents_frame / 0.18215).sample
            video.append(frame)
        video = torch.stack(video, dim=2)
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.cpu().permute(0, 2, 3, 4, 1).float().numpy()
        return video
    
    def __call__(
        self,
        ref_image,
        pose_image,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        device = self._execution_device
        
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        batch_size = 1
        
        # Prepare clip image embeds
        clip_image = self.clip_image_processor.preprocess(
            ref_image.resize((224, 224)), return_tensors="pt"
        ).pixel_values
        clip_image_embeds = self.image_encoder(
            clip_image.to(device, dtype=self.image_encoder.dtype)
        ).image_embeds
        image_prompt_embeds = clip_image_embeds.unsqueeze(1)
        
        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)
        
        if do_classifier_free_guidance:
            image_prompt_embeds = torch.cat(
                [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
            )
        
        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        
        num_channels_latents = self.denoising_unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            clip_image_embeds.dtype,
            device,
            generator,
        )
        latents = latents.unsqueeze(2)  # (bs, c, 1, h', w')
        latents_dtype = latents.dtype
        
        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)
        
        # Prepare pose condition image  
        pose_cond_tensor = self.cond_image_processor.preprocess(
            pose_image, height=height, width=width
        )
        
        pose_cond_tensor = pose_cond_tensor.unsqueeze(2)  # (bs, c, 1, h, w)
        pose_cond_tensor = pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        
        # PoseGuider returns features for each UNet block with correct dimensions
        # pose_fea is a list: [320ch@64x64, 640ch@32x32, 1280ch@16x16, 1280ch@8x8]
        pose_fea = self.pose_guider(pose_cond_tensor)
        
        if do_classifier_free_guidance:
            # Double each pose feature for classifier-free guidance
            pose_fea = [torch.cat([feat] * 2) for feat in pose_fea]
        
        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 1. Forward reference image
                if i == 0:
                    self.reference_unet(
                        ref_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        encoder_hidden_states=image_prompt_embeds,
                        return_dict=False,
                    )
                    
                    # 2. Update reference unet feature into denosing net
                    reference_control_reader.update(reference_control_writer)
                
                # 3.1 expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                
                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_prompt_embeds,
                    pose_cond_fea=pose_fea,
                    return_dict=False,
                )[0]
                
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
            reference_control_reader.clear()
            reference_control_writer.clear()
        
        # Post-processing
        image = self.decode_latents(latents)  # (b, c, 1, h, w)
        
        # Convert to tensor
        if output_type == "tensor":
            image = torch.from_numpy(image)
        
        if not return_dict:
            return image
        
        return Pose2ImagePipelineOutput(images=image)


# Import CLIPImageProcessor from transformers
from transformers import CLIPImageProcessor