# Sprite Sheet Diffusion: Generate Game Character for Animation (SSD)

## Overview
Below is an image depicting the method used in the Sprite Sheet Diffusion approach:

![Method Overview](https://github.com/chenganhsieh/spritesheet-diffusion/blob/main/static/images/method.png)

## Installation 
Create a conda environment with the following command:
```bash
conda create -n ssd python=3.10
conda activate ssd
pip install -r requirements.txt
```
## Pretrained weight
Our fine-tune model weight is [here](https://drive.google.com/drive/folders/1VxbOv5PE441NsNStQlmqbIw0iyY9Mn9L?usp=sharing). 

## Usage

### Evaluation
* ID-Adpater
We implemented two evaluation scripts to automate the evaluation process and generate a summary of the results in a MARKDOWN file.
- `eval_img_quality.py`: This script evaluates the similarity between ground truth and generated motion frames using metrics such as Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Learned Perceptual Image Patch Similarity (LPIPS).
- `eval_sub_consistency`: This script assesses subject consistency within the generated sequence, measured based on the Subject Consistency Score proposed by [VBench](https://github.com/OpenGVLab/VBench).
* Animate Anyone
Edit the configuration file located at `configs/prompts/inference.yaml` to specify your model path and data path.
```bash
python inference.py
```



#### Prerequisites
Please organize your results using the following folder structure:

```
results/
└── Exp_A/
    ├── character_X/
    │   └── motions/
    │       ├── run/
    │       │   ├── ground_truth/
    │       │   └── predict/
    │       └── walk/
    │           ├── ground_truth/
    │           └── predict/
    ├── character_Y/
    │   └── motions/
    │       ├── sit/
    │       │   ├── ground_truth/
    │       │   └── predict/
    │       └── stand/
    │           ├── ground_truth/
    │           └── predict/
    └── ...
```

#### Image Quality Evaluation
```bash
python eval_img_quality.py \
    --result_dir <path_to_experiment_results_directory> \
    --reference_img <reference_image_folder_name> \
    --generated_img <generated_image_folder_name> \
    --output_file <output_markdown_file_path>
```

#### Subject Consistency Evaluation
```bash
python eval_sub_consistency.py \
    --result_dir <ath_to_experiment_results_directory> \
    --generated_img <generated_image_folder_name> \
    --output_video_dir <path_to_temporary_video_directory> \
    --eval_dir <path_to_evaluation_results_directory> \
    --tag <evaluation_tag_for_naming>
```


## Example
TBD

## Contact
* Cheng-An Hsieh
* Jing Zhang
* Ava Yan

## License
TBD

## Acknowledgments
Our codebase is directly built on top of [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone). We would like to thank Huawei Wei and team for open-sourcing their code.

## Citation
If you find our work useful, please consider citing:
```bibtex
@article{hsieh2024sprite,
  title={Sprite Sheet Diffusion: Generate Game Character for Animation},
  author={Hsieh, Cheng-An and Zhang, Jing and Yan, Ava},
  journal={arXiv preprint arXiv:2412.03685},
  year={2024}
}
