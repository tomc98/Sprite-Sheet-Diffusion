import os
import json
import cv2
import numpy as np
from PIL import Image
from openpose import OpenposeDetector

def generate_dataset_json(dataset_dir, output_json):
    pose_detector =  OpenposeDetector()
    dataset = {"characters": []}

    characters_dir = os.path.join(dataset_dir, "characters")
    if not os.path.isdir(characters_dir):
        raise ValueError(f"Characters directory not found at {characters_dir}")

    for character_name in sorted(os.listdir(characters_dir)):
        character_path = os.path.join(characters_dir, character_name)
        if not os.path.isdir(character_path):
            print(f"Skipping non-directory: {character_path}")
            continue

        print(f"Processing character: {character_name}")

        # Path to main_reference.png
        main_reference_path = os.path.join(character_path, "main_reference.png")
        if not os.path.isfile(main_reference_path):
            print(f"Warning: main_reference.png not found for character {character_name}")
            main_reference = None
        else:
            main_reference = main_reference_path
            main_reference_pose_path = os.path.join(character_path, "main_reference_pose.png")
            ref_image_pil = Image.open(main_reference_path).convert("RGB")
            ref_image_np = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)
            ref_image_np = cv2.resize(ref_image_np, (512, 512))
            
            ref_pose = pose_detector(ref_image_np,
                                 include_body=True,
                                 include_hand=False,
                                 include_face=False,
                                 use_dw_pose=True)
            ref_pose_image = Image.fromarray(ref_pose)
            ref_pose_image.save(main_reference_pose_path) 
            
            

        character_dict = {
            "name": character_name,
            "main_reference": main_reference,
            "main_reference_pose": main_reference_pose_path,
            "motions": []
        }

        motions_dir = os.path.join(character_path, "motions")
        if not os.path.isdir(motions_dir):
            print(f"Warning: motions directory not found for character {character_name}")
            dataset["characters"].append(character_dict)
            continue

        for motion_name in sorted(os.listdir(motions_dir)):
            motion_path = os.path.join(motions_dir, motion_name)
            if not os.path.isdir(motion_path):
                print(f"Skipping non-directory: {motion_path}")
                continue

            print(f"  Processing motion: {motion_name}")

            # Reference image
            reference_path = os.path.join(motion_path, "reference.png")
            if not os.path.isfile(reference_path):
                print(f"    Warning: reference.png not found for motion {motion_name} of character {character_name}")
                reference = None
            else:
                reference = reference_path

            # Pose images
            poses_dir = os.path.join(motion_path, "poses")
            if os.path.isdir(poses_dir):
                poses = sorted([
                    os.path.join(poses_dir, fname)
                    for fname in os.listdir(poses_dir)
                    if fname.lower().endswith(".png")
                ])
            else:
                print(f"    Warning: poses directory not found for motion {motion_name} of character {character_name}")
                poses = []

            # Ground truth frames
            ground_truth_dir = os.path.join(motion_path, "ground_truth")
            if os.path.isdir(ground_truth_dir):
                ground_truth = sorted([
                    os.path.join(ground_truth_dir, fname)
                    for fname in os.listdir(ground_truth_dir)
                    if fname.lower().endswith(".png")
                ])
            else:
                print(f"    Warning: ground_truth directory not found for motion {motion_name} of character {character_name}")
                ground_truth = []

            motion_dict = {
                "motion_name": motion_name,
                "reference": reference,
                "poses": poses,
                "ground_truth": ground_truth
            }

            character_dict["motions"].append(motion_dict)

        dataset["characters"].append(character_dict)

    # Write to JSON file
    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Dataset JSON has been created at {output_json}")

if __name__ == "__main__":
    dataset_directory = "game_animation"  # Replace with your dataset directory path if different
    output_json_file = "game_animation.json"  # Desired output JSON file name
    generate_dataset_json(dataset_directory, output_json_file)
