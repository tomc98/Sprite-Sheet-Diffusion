import os
import argparse
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from accelerate import Accelerator
from ip_adapter.ip_adapter import ImageProjModel, IPAdapter, AttnProcessor, IPAttnProcessor
import itertools


# Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, size=512, image_root_path=""):
        super().__init__()
        self.size = size
        self.image_root_path = image_root_path
        self.data = json.load(open(json_file))

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
        ])
        
        self.sd_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        item = self.data[idx]
        reference_image = Image.open(os.path.join(self.image_root_path, item["reference_image"])).convert("RGB")
        pose_image = Image.open(os.path.join(self.image_root_path, item["pose_image"])).convert("RGB")
        target_image = Image.open(os.path.join(self.image_root_path, item["target_image"])).convert("RGB")

        reference_image_sd = self.sd_transform(reference_image)
        pose_image_sd = self.sd_transform(pose_image)
        target_image_sd = self.sd_transform(target_image)

        reference_image_clip = self.transform(reference_image)
        clip_image = self.clip_image_processor(
            images=reference_image_clip.permute(1, 2, 0).numpy(),
            return_tensors="pt",
            do_rescale=False  # 禁用重复的缩放操作
        ).pixel_values[0]

        return {
            "reference_image": reference_image_sd,
            "pose_image": pose_image_sd,
            "target_image": target_image_sd,
            "clip_image": clip_image,
        }

    def __len__(self):
        return len(self.data)

def collate_fn(data):
    reference_images = torch.stack([example["reference_image"] for example in data])
    pose_images = torch.stack([example["pose_image"] for example in data])
    target_images = torch.stack([example["target_image"] for example in data])
    clip_images = torch.stack([example["clip_image"] for example in data])

    return {
        "reference_images": reference_images,
        "pose_images": pose_images,
        "target_images": target_images,
        "clip_images": clip_images,
    }

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Training script for IP-Adapter with ControlNet.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--controlnet_model_path", type=str, required=True)
    parser.add_argument("--image_encoder_path", type=str, required=True)
    parser.add_argument("--data_json_file", type=str, required=True)
    parser.add_argument("--data_root_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_dir/")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    return parser.parse_args()

from tqdm import tqdm  # 导入 tqdm
import wandb  # 导入 wandb

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # 初始化 wandb
    if accelerator.is_main_process:
        wandb.init(
            project="IP-Adapter-ControlNet",  # 替换为你的项目名称
            config={
                "pretrained_model": args.pretrained_model_name_or_path,
                "controlnet_model": args.controlnet_model_path,
                "image_encoder": args.image_encoder_path,
                "batch_size": args.train_batch_size,
                "resolution": args.resolution,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_epochs": args.num_train_epochs,
            }
        )

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_path)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    
    vae = vae.to(accelerator.device)
    unet = unet.to(accelerator.device)
    controlnet = controlnet.to(accelerator.device)
    image_encoder = image_encoder.to(accelerator.device)

    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    ).to(accelerator.device)

    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            processor = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=1.0,
                num_tokens=4,
            )
            attn_procs[name] = processor.to(accelerator.device)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(list(unet.attn_processors.values()))

    optimizer = torch.optim.AdamW(
        itertools.chain(
            image_proj_model.parameters(),
            adapter_modules.parameters()
        ),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    train_dataset = MyDataset(args.data_json_file, size=args.resolution, image_root_path=args.data_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers
    )

    image_proj_model, adapter_modules, optimizer, train_dataloader = accelerator.prepare(
        image_proj_model, adapter_modules, optimizer, train_dataloader
    )

    # tqdm
    for epoch in range(args.num_train_epochs):
        progress_bar = tqdm(
            enumerate(train_dataloader), 
            desc=f"Epoch {epoch+1}/{args.num_train_epochs}",
            total=len(train_dataloader),
            disable=not accelerator.is_main_process  
        )
        for step, batch in progress_bar:
            with accelerator.accumulate(image_proj_model):
                reference_images = batch["reference_images"].to(accelerator.device, dtype=torch.float32)
                pose_images = batch["pose_images"].to(accelerator.device, dtype=torch.float32)
                target_images = batch["target_images"].to(accelerator.device, dtype=torch.float32)
                clip_images = batch["clip_images"].to(accelerator.device, dtype=torch.float32)

                with torch.no_grad():
                    clip_image_embeds = image_encoder(clip_images).image_embeds
                    image_prompt_embeds = image_proj_model(clip_image_embeds)

                with torch.no_grad():
                    latents = vae.encode(target_images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.size(0),), device=latents.device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    controlnet_output = controlnet(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=image_prompt_embeds,
                        controlnet_cond=pose_images,
                        return_dict=False,
                    )

                noise_pred = unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=image_prompt_embeds,
                    down_block_additional_residuals=controlnet_output[0],
                    mid_block_additional_residual=controlnet_output[1],
                ).sample

                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                progress_bar.set_postfix({"loss": loss.item()})

                if accelerator.is_main_process:
                    wandb.log({"loss": loss.item()})

                if step % args.save_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}-{step}")
                    os.makedirs(save_path, exist_ok=True)
                    state_dict = {
                        "image_proj": accelerator.unwrap_model(image_proj_model).state_dict(),
                        "ip_adapter": accelerator.unwrap_model(adapter_modules).state_dict(),
                    }
                    torch.save(state_dict, os.path.join(save_path, "ip_adapter.bin"))
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # end wandb
    if accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()
