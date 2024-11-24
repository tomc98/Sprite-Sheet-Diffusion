# https://github.com/google-research/google-research/tree/master/cmmd
import os
import numpy as np
import cv2
import lpips
import argparse
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
from ssim import SSIM, Image


def ssim_score(input_image_path: str, target_image_path: str) -> float:
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    Higher SSIM score means the images are more similar.
    """
    input_image = Image.open(input_image_path)
    target_image = Image.open(target_image_path)
    score = SSIM(input_image).cw_ssim_value(target_image)
    # print("SSIM Score: ", score)
    return score

def psnr_score(input_image_path: str, target_image_path: str) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Higher PSNR score means the images are more similar.
    """
    input_image = cv2.imread(input_image_path)
    target_image = cv2.imread(target_image_path)
    score = cv2.PSNR(input_image, target_image)
    # print("PSNR Score: ", score)
    return score

def lpips_score(input_image_path: str, target_image_path: str, loss_fn: object, transform: object) -> float:
    """
    Calculate the Learned Perceptual Image Patch Similarity (LPIPS) between two images.
    Lower LPIPS score means the images are more similar.
    """
    input_image = transform(Image.open(input_image_path)).unsqueeze(0)
    target_image = transform(Image.open(target_image_path)).unsqueeze(0)
    score = loss_fn(input_image, target_image).item()
    # print("LPIPS Score: ", score)
    return score


def evaluate(input_images_folder: str, target_images_folder: str, lpips_loss_fn: object, lpips_transform: object) -> dict:
    """
    Evaluate the similarity between input and target images using SSIM, PSNR, and LPIPS metrics.
    """
    metrics = {
        "SSIM": [],
        "PSNR": [],
        "LPIPS": []
    }

    input_files = sorted(os.listdir(input_images_folder))
    target_files = sorted(os.listdir(target_images_folder))

    if len(input_files) != len(target_files):
        raise ValueError("Mismatch in the number of images between input and target folders.")

    with tqdm(total=len(input_files), desc="Processing Images", leave=False) as image_pbar:
        for input_file, target_file in zip(input_files, target_files):
            input_path = os.path.join(input_images_folder, input_file)
            target_path = os.path.join(target_images_folder, target_file)

            metrics["SSIM"].append(ssim_score(input_path, target_path))
            metrics["PSNR"].append(psnr_score(input_path, target_path))
            metrics["LPIPS"].append(lpips_score(input_path, target_path, lpips_loss_fn, lpips_transform))
            
            image_pbar.update(1)

    return metrics

def summarize_metrics(metrics: dict) -> dict:
    """
    Calculate the average, standard deviation, minimum, and maximum of the given metrics.
    """
    summary = {}
    for metric, scores in metrics.items():
        summary[metric] = {
            "Avg": np.mean(scores),
            "Std": np.std(scores),
            "Min": np.min(scores),
            "Max": np.max(scores),
        }
    return summary

def save_results(overall_summary: dict, character_motion_summaries: dict, detailed_scores: dict, output_file_path: str) -> None:
    """
    Save the evaluation results to a markdown file.
    """
    overall_summary_df = pd.DataFrame(overall_summary).T
    overall_summary_df.columns = ["Avg", "Std", "Min", "Max"]

    with open(output_file_path, "w") as f:
        # Write overall evaluation results
        f.write("# Overall Evaluation Results\n\n")
        f.write(overall_summary_df.to_markdown())
        f.write("\n\n")

        # Write detailed evaluation results for each character motion
        for character_motion, summary in character_motion_summaries.items():
            f.write(f"## {character_motion}\n\n")
            summary_df = pd.DataFrame(summary).T
            summary_df.columns = ["Avg", "Std", "Min", "Max"]
            f.write(summary_df.to_markdown())
            f.write("\n\n")

            # Write detailed scores for each frame in the motion
            detailed_scores_df = pd.DataFrame(detailed_scores[character_motion])
            f.write(detailed_scores_df.to_markdown(index=False))
            f.write("\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image similarity metrics and save results.")
    parser.add_argument("--result_dir", type=str, required=True, help="Path to the experiment results directory.")
    parser.add_argument("--reference_img", type=str, required=True, help="Reference image folder name.")
    parser.add_argument("--generated_img", type=str, required=True, help="Generated image folder name.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output markdown file.")
    args = parser.parse_args()

    result_dir = args.result_dir
    output_file_path = args.output_file
    reference_img_folder = args.reference_img
    generated_img_folder = args.generated_img

    characters = os.listdir(result_dir)

    overall_metrics = {
        "SSIM": [],
        "PSNR": [],
        "LPIPS": []
    }

    character_motion_summaries = {}
    detailed_scores = {}

    # Initialize LPIPS loss function and transformation
    lpips_loss_fn = lpips.LPIPS(net='alex')
    lpips_transform = transforms.Compose([transforms.ToTensor()])

    with tqdm(characters, desc="Processing Characters") as character_pbar:
        for character in characters:
            character_path = os.path.join(result_dir, character)
            if not os.path.isdir(character_path):
                character_pbar.update(1)
                continue

            motions_path = os.path.join(character_path, "motions")
            motions = os.listdir(motions_path)

            with tqdm(motions, desc=f"Processing Motions for {character}", leave=False) as motion_pbar:
                for motion in motions:
                    ground_truth_path = os.path.join(motions_path, motion, reference_img_folder)
                    generated_img_path = os.path.join(motions_path, motion, generated_img_folder)

                    if not os.path.exists(ground_truth_path) or not os.path.exists(generated_img_path):
                        print(f"Skipping motion '{motion}' for character '{character}' due to missing folders.")
                        motion_pbar.update(1)
                        continue

                    print(f"Evaluating '{motion}' motion for character '{character}'...")
                    metrics = evaluate(ground_truth_path, generated_img_path, lpips_loss_fn, lpips_transform)

                    for metric in overall_metrics.keys():
                        overall_metrics[metric].extend(metrics[metric])

                    summary = summarize_metrics(metrics)
                    character_motion_key = f"{character} - {motion}"
                    character_motion_summaries[character_motion_key] = summary

                    detailed_scores[character_motion_key] = {
                        "Frame": [f"Frame {i+1}" for i in range(len(metrics["SSIM"]))],
                        "SSIM": metrics["SSIM"],
                        "PSNR": metrics["PSNR"],
                        "LPIPS": metrics["LPIPS"]
                    }

                    motion_pbar.update(1)

            character_pbar.update(1)

    overall_summary = summarize_metrics(overall_metrics)
    save_results(overall_summary, character_motion_summaries, detailed_scores, output_file_path)

    print(f"Evaluation completed. Results saved to {output_file_path}")