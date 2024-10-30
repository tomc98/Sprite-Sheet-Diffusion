# wget https://huggingface.co/ZJYang/AniPortrait/resolve/main/denoising_unet.pth
# wget https://huggingface.co/ZJYang/AniPortrait/resolve/main/film_net_fp16.pt
# wget https://huggingface.co/ZJYang/AniPortrait/resolve/main/motion_module.pth
# wget https://huggingface.co/ZJYang/AniPortrait/resolve/main/pose_guider.pth
# wget https://huggingface.co/ZJYang/AniPortrait/resolve/main/reference_unet.pth
mkdir wav2vec2-base-960h
mkdir stable-diffusion-v1-5
mkdir sd-vae-ft-mse
mkdir image_encoder
cd stable-diffusion-v1-5
mkdir feature_extractor
cd feature_extractor
wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/feature_extractor/preprocessor_config.json
cd ../
wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/model_index.json
mkdir unet
cd unet
wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin
wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/unet/config.json
cd ../
wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-inference.yaml
cd ../wav2vec2-base-960h
wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json
wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/feature_extractor_config.json
wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/preprocessor_config.json
wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin
wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/README.md
wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/special_tokens_map.json
wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/tokenizer_config.json
wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json
cd ../sd-vae-ft-mse
wget https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json
wget https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin
wget https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors
cd ../image_encoder
wget https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/config.json
wget https://huggingface.co/lambdalabs/sd-image-variations-diffusers/resolve/main/image_encoder/pytorch_model.bin
