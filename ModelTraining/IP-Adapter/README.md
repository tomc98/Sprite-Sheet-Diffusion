# SD-IPCN Inference code

check ModelTraining/IP-Adapter/ip_adapter_controlnet_demo_new.ipynb


# SD-IPCN Training code

cd IP-Adaptor

```
accelerate launch --num_processes 8 --multi_gpu --mixed_precision "fp16"   tutorial_train.py   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5/"   --image_encoder_path="{image_encoder_path}"   --data_json_file="{data.json}"   --data_root_path="{image_path}"   --mixed_precision="fp16"   --resolution=512   --train_batch_size=8   --dataloader_num_workers=4   --learning_rate=1e-04   --weight_decay=0.01   --output_dir="{output_dir}"   --save_steps=10000
```


Adapted from https://github.com/TencentARC/IP-Adapter
