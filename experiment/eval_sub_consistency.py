import cv2
import os
import argparse
from tqdm import tqdm
import torch
from vbench import VBench
import json
import pandas as pd

def frames_to_video(frame_dir: str, output_video: str, frame_rate: int = 30) -> None:
    """
    Covert motion frames to a video file.
    """
    # Check if the video already exists
    if os.path.exists(output_video):
        print(f"Video already exists: {output_video}")
        return
    
    frames_list = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir)])

    # Read the first frame to get video dim
    first_frame = cv2.imread(frames_list[0])
    height, width, _ = first_frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # Write frames to video
    print(f"Writing frames from {frame_dir} to video...")
    for frame_file in frames_list:
        frame = cv2.imread(frame_file)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as: {output_video}")

def save_results(json_file: str, readme_file: str) -> None:
    """
    Save evaluation results to a MARKDOWN file.
    """
    # Load the subject consistency result
    with open(json_file, 'r') as file:
        data = json.load(file)

    subject_consistency = data["subject_consistency"][1]

    df = pd.DataFrame(subject_consistency)
    df['Character-Motion'] = df['video_path'].apply(lambda x: x.split('/')[-1].replace('.mp4', ''))
    df = df[['Character-Motion', 'video_results']]

    # Calculate overall statistics
    overall_stats = {
        "Subject Consistency": {
            "Avg": df["video_results"].mean(),
            "Std": df["video_results"].std(),
            "Min": df["video_results"].min(),
            "Max": df["video_results"].max()
        }
    }

    with open(readme_file, 'w') as readme:
        # Write overall statistics
        readme.write("# Subject Consistency Evaluation\n\n")
        readme.write("## Overall Statistics\n\n")
        readme.write("| Subject Consistency | Avg         | Std         | Min         | Max         |\n")
        readme.write("|---------------------|-------------|-------------|-------------|-------------|\n")
        readme.write(f"|                     | {overall_stats['Subject Consistency']['Avg']:<11.6f} | {overall_stats['Subject Consistency']['Std']:<11.6f} | {overall_stats['Subject Consistency']['Min']:<11.6f} | {overall_stats['Subject Consistency']['Max']:<11.6f} |\n")
        
        # Write detailed results
        readme.write("\n## Detailed Results\n\n")
        readme.write("| Character-Motion                 | Result        |\n")
        readme.write("|----------------------------------|---------------|\n")
        for _, row in df.iterrows():
            readme.write(f"| {row['Character-Motion']:<32} | {row['video_results']:<11.6f} |\n")
    
    print(f"Results saved to: {readme_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate subject consistency in generated frames.")
    parser.add_argument("--result_dir", type=str, required=True, help="Path to the experiment results directory.")
    parser.add_argument("--generated_img", type=str, required=True, help="Generated image folder name.")
    parser.add_argument("--output_video_dir", type=str, required=True, help="Directory to save output videos.")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory to save evaluation results.")
    parser.add_argument("--tag", type=str, required=True, help="Tag for the evaluation results.")
    args = parser.parse_args()

    generated_img_folder = args.generated_img
    output_video_dir = args.output_video_dir
    result_dir = args.result_dir
    tag = args.tag
    eval_dir = args.eval_dir

    os.makedirs(output_video_dir, exist_ok=True)
    characters = os.listdir(result_dir)

    evaluate_lists = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_sub_consistency = VBench(device, "VBench_full_info.json", eval_dir)

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
                    generated_img_path = os.path.join(motions_path, motion, generated_img_folder)

                    if not os.path.exists(generated_img_path):
                        print(f"Skipping motion '{motion}' for character '{character}' due to missing folders.")
                        motion_pbar.update(1)
                        continue

                    output_video_path = os.path.join(output_video_dir, f"{character}_{motion}.mp4")
                    frames_to_video(generated_img_path, output_video_path)
                    # evaluate_lists.append(output_video_path)

                    motion_pbar.update(1)

            character_pbar.update(1)

    eval_sub_consistency.evaluate(output_video_dir, name=tag, dimension_list=["subject_consistency"], mode="custom_input")

    # Save evaluation results to MARKDOWN file
    save_results(os.path.join(eval_dir, f"{tag}_eval_results.json"), os.path.join(eval_dir, f"RESULTS_frame_consistency_{tag}.md"))



