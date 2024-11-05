import os, io, csv, math, random, pdb
import cv2
import numpy as np
import json
from PIL import Image
from einops import rearrange

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from transformers import CLIPImageProcessor

import torch.distributed as dist

from utils.draw_util import FaceMeshVisualizer

def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)



class GameDatasetValid(Dataset):
    def __init__(
            self,
            json_path,
            extra_json_path=None,
            sample_size=[512, 512],
            is_image=True,
            sample_stride_aug=False
    ):
        zero_rank_print(f"loading annotations from {json_path} ...")
        self.data_dic_name_list, self.data_dic = self.get_data(json_path, extra_json_path)
        
        self.length = len(self.data_dic_name_list)
        zero_rank_print(f"data scale: {self.length}")
        
        self.sample_stride_aug = sample_stride_aug
        self.sample_size = sample_size
        self.is_image = is_image
        
        self.resize = transforms.Resize((sample_size[0], sample_size[1]))


        self.pixel_transforms = transforms.Compose([
            transforms.Resize([sample_size[1], sample_size[0]]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        self.clip_image_processor = CLIPImageProcessor()
        

    def get_data(self, json_name, extra_json_name, augment_num=1):
        zero_rank_print(f"start loading data: {json_name}")
        with open(json_name,'r') as f:
            data_dic = json.load(f)

        data_dic_name_list = []
        for character in data_dic['characters']:
            character_name = character['name']
            main_reference_pose = character['main_reference_pose']
            for motion in character['motions']:
                motion_name = motion['motion_name']
                # Create a unique identifier for each (character, motion)
                unique_id = f"{character_name}_{motion_name}"
                data_dic_name_list.append(unique_id)
                # Store the corresponding data
                data_dic[unique_id] = {
                    'character_name': character_name,
                    'motion_name': motion_name,
                    'reference': motion['reference'],
                    'poses': motion['poses'],
                    'ground_truth': motion['ground_truth'],
                    'reference_pose': main_reference_pose,
                }


        # Handle extra JSON for augmentation if provided
        if extra_json_name is not None:
            zero_rank_print(f"Start loading data: {extra_json_name}")
            with open(extra_json_name, 'r') as f:
                extra_data_dic = json.load(f)
            data_dic.update(extra_data_dic)
            for character in extra_data_dic['characters']:
                character_name = character['name']
                for motion in character['motions']:
                    motion_name = motion['motion_name']
                    unique_id = f"{character_name}_{motion_name}"
                    data_dic_name_list.append(unique_id)

        random.shuffle(data_dic_name_list)
        zero_rank_print("finish loading")
        return data_dic_name_list, data_dic

    def __len__(self):
        return len(self.data_dic_name_list)
    
    def get_batch_wo_pose(self, index):
        unique_id = self.data_dic_name_list[index]
        sample_info = self.data_dic[unique_id]

        # Ground Truth Images
        ground_truth_paths = sample_info['ground_truth']
        pixel_values = [cv2.imread(path) for path in ground_truth_paths]
        pixel_values = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in pixel_values]
        pixel_values = [self.contrast_normalization(img) for img in pixel_values]
        pixel_values = np.array(pixel_values)
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.0  # Normalize to [0,1]

        # Pose Images for Ground Truth
        poses_paths = sample_info['poses']
        pixel_values_pose = [cv2.imread(path) for path in poses_paths]
        pixel_values_pose = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in pixel_values_pose]
        pixel_values_pose = [self.contrast_normalization(img) for img in pixel_values_pose]
        pixel_values_pose = np.array(pixel_values_pose)
        pixel_values_pose = torch.from_numpy(pixel_values_pose).permute(0, 3, 1, 2).contiguous()
        pixel_values_pose = pixel_values_pose / 255.0  # Normalize to [0,1]

        # Reference Image processed by CLIP
        reference_path = sample_info['reference']
        ref_img = cv2.imread(reference_path)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img = self.contrast_normalization(ref_img)
        ref_img_pil = Image.fromarray(ref_img)
        clip_ref_image = self.clip_image_processor(images=ref_img_pil, return_tensors="pt").pixel_values

        # Main Reference Image (main_reference.png)
        character_name = sample_info['character_name']
        main_reference_path = os.path.join("dataset", "characters", character_name, "main_reference.png")
        if not os.path.isfile(main_reference_path):
            raise FileNotFoundError(f"main_reference.png not found for character {character_name}")
        pixel_values_ref_img = ref_img
        pixel_values_ref_img = torch.from_numpy(pixel_values_ref_img).permute(2, 0, 1).contiguous()
        pixel_values_ref_img = pixel_values_ref_img / 255.0  # Normalize to [0,1]

        # Reference Pose Image
        # Assuming the reference pose is the first pose in the list
        ref_pose_path = sample_info['reference_pose']
        if ref_pose_path:
            ref_img = cv2.imread(ref_pose_path)
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            pixel_values_ref_pose = torch.from_numpy(ref_img).permute(2, 0, 1).contiguous()
            pixel_values_ref_pose = pixel_values_ref_pose / 255.0  # Normalize to [0,1]
        else:
            # If no pose is available, initialize with zeros
            pixel_values_ref_pose = torch.zeros(3, self.sample_size[0], self.sample_size[1])

        # Drop Image Embeds (randomly set to 1 with 10% probability)
        drop_image_embeds = 1 if random.random() < 0.1 else 0

        return {
            'pixel_values': pixel_values,  # Ground Truth Images
            'pixel_values_pose': pixel_values_pose,  # Pose for Ground Truth Images
            'clip_ref_image': clip_ref_image,  # CLIP-processed Reference Image
            'pixel_values_ref_img': pixel_values_ref_img,  # main_reference.png
            'drop_image_embeds': drop_image_embeds,  # Drop Image Embeds Flag
            'pixel_values_ref_pose': pixel_values_ref_pose  # Reference Pose Image
        }
    def contrast_normalization(self, image, lower_bound=0, upper_bound=255):
        image = image.astype(np.float32)
        normalized_image = image  * (upper_bound - lower_bound) / 255 + lower_bound
        normalized_image = normalized_image.astype(np.uint8)

        return normalized_image

    def __getitem__(self, idx):
        sample = self.get_batch_wo_pose(idx)

        # Apply pixel transformations
        pixel_values = self.pixel_transforms(sample['pixel_values'])  # Ground Truth Images
        pixel_values_pose = self.pixel_transforms(sample['pixel_values_pose'])  # Pose Images

        # Process reference images
        pixel_values_ref_img = self.pixel_transforms(sample['pixel_values_ref_img'])  # main_reference.png
        pixel_values_ref_pose = self.pixel_transforms(sample['pixel_values_ref_pose'])  # Reference Pose Image

        # Create the sample dictionary with the required variables
        final_sample = {
            'pixel_values': pixel_values,  # Ground Truth Images
            'pixel_values_pose': pixel_values_pose,  # Pose for Ground Truth Images
            'clip_ref_image': sample['clip_ref_image'],  # CLIP-processed Reference Image
            'pixel_values_ref_img': pixel_values_ref_img,  # main_reference.png
            'drop_image_embeds': sample['drop_image_embeds'],  # Drop Image Embeds Flag
            'pixel_values_ref_pose': pixel_values_ref_pose  # Reference Pose Image
        }
        
        return final_sample


class GameDataset(Dataset):
    def __init__(
            self,
            json_path,
            extra_json_path=None,
            sample_size=[512, 512],
            is_image=True, 
            sample_stride_aug=False
    ):
        zero_rank_print(f"Loading annotations from {json_path} ...")
        self.data_dic_name_list, self.data_dic = self.get_data(json_path, extra_json_path)
        
        self.length = len(self.data_dic_name_list)
        zero_rank_print(f"Data scale: {self.length}")

        self.sample_size = sample_size
        self.sample_stride_aug = sample_stride_aug
        self.is_image = is_image

        # Define transformations
        self.pixel_transforms = transforms.Compose([
            transforms.Resize((self.sample_size[0], self.sample_size[1])),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5],
                                 inplace=True),
        ])

        # Initialize CLIP processor
        self.clip_image_processor = CLIPImageProcessor()

    def get_data(self, json_name, extra_json_name, augment_num=1):
        zero_rank_print(f"Start loading data: {json_name}")
        with open(json_name, 'r') as f:
            data_dic = json.load(f)

        data_dic_name_list = []
        for character in data_dic['characters']:
            character_name = character['name']
            main_reference_pose = character['main_reference_pose']
            for motion in character['motions']:
                motion_name = motion['motion_name']
                # Create a unique identifier for each (character, motion)
                unique_id = f"{character_name}_{motion_name}"
                data_dic_name_list.append(unique_id)
                # Store the corresponding data
                data_dic[unique_id] = {
                    'character_name': character_name,
                    'motion_name': motion_name,
                    'reference': motion['reference'],
                    'poses': motion['poses'],
                    'ground_truth': motion['ground_truth'],
                    'reference_pose': main_reference_pose,
                }

        # Handle extra JSON for augmentation if provided
        if extra_json_name is not None:
            zero_rank_print(f"Start loading data: {extra_json_name}")
            with open(extra_json_name, 'r') as f:
                extra_data_dic = json.load(f)
            data_dic.update(extra_data_dic)
            for character in extra_data_dic['characters']:
                character_name = character['name']
                for motion in character['motions']:
                    motion_name = motion['motion_name']
                    unique_id = f"{character_name}_{motion_name}"
                    data_dic_name_list.append(unique_id)

        # Remove any invalid entries if necessary (based on your original logic)
        # This section can be customized based on your dataset's specifics
        # For example, removing entries without ground_truth frames
        valid_data_dic_name_list = []
        for unique_id in data_dic_name_list:
            if len(data_dic[unique_id]['ground_truth']) >= 1:
                valid_data_dic_name_list.append(unique_id)
        
        # Update the data_dic_name_list to include only valid entries
        data_dic_name_list = valid_data_dic_name_list

        # Shuffle the data
        random.shuffle(data_dic_name_list)
        zero_rank_print("Finish loading")
        return data_dic_name_list, data_dic

    def __len__(self):
        return len(self.data_dic_name_list)

    
    def get_batch_wo_pose(self, index):
        unique_id = self.data_dic_name_list[index]
        sample_info = self.data_dic[unique_id]

        # Ground Truth Images
        ground_truth_paths = sample_info['ground_truth']
        pixel_values = [cv2.imread(path) for path in ground_truth_paths]
        pixel_values = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in pixel_values]
        pixel_values = [self.contrast_normalization(img) for img in pixel_values]
        pixel_values = np.array(pixel_values)
        pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.0  # Normalize to [0,1]

        # Pose Images for Ground Truth
        poses_paths = sample_info['poses']
        pixel_values_pose = [cv2.imread(path) for path in poses_paths]
        pixel_values_pose = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in pixel_values_pose]
        pixel_values_pose = [self.contrast_normalization(img) for img in pixel_values_pose]
        pixel_values_pose = np.array(pixel_values_pose)
        pixel_values_pose = torch.from_numpy(pixel_values_pose).permute(0, 3, 1, 2).contiguous()
        pixel_values_pose = pixel_values_pose / 255.0  # Normalize to [0,1]

        # Reference Image processed by CLIP
        reference_path = sample_info['reference']
        ref_img = cv2.imread(reference_path)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img = self.contrast_normalization(ref_img)
        ref_img_pil = Image.fromarray(ref_img)
        clip_ref_image = self.clip_image_processor(images=ref_img_pil, return_tensors="pt").pixel_values

        # Main Reference Image (main_reference.png)
        character_name = sample_info['character_name']
        main_reference_path = os.path.join("dataset", "characters", character_name, "main_reference.png")
        if not os.path.isfile(main_reference_path):
            raise FileNotFoundError(f"main_reference.png not found for character {character_name}")
        pixel_values_ref_img = ref_img
        pixel_values_ref_img = torch.from_numpy(pixel_values_ref_img).permute(2, 0, 1).contiguous()
        pixel_values_ref_img = pixel_values_ref_img / 255.0  # Normalize to [0,1]

        # Reference Pose Image
        # Assuming the reference pose is the first pose in the list
        ref_pose_path = sample_info['reference_pose']
        if ref_pose_path:
            ref_img = cv2.imread(ref_pose_path)
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            pixel_values_ref_pose = torch.from_numpy(ref_img).permute(2, 0, 1).contiguous()
            pixel_values_ref_pose = pixel_values_ref_pose / 255.0  # Normalize to [0,1]
        else:
            # If no pose is available, initialize with zeros
            pixel_values_ref_pose = torch.zeros(3, self.sample_size[0], self.sample_size[1])

        # Drop Image Embeds (randomly set to 1 with 10% probability)
        drop_image_embeds = 1 if random.random() < 0.1 else 0

        return {
            'pixel_values': pixel_values,  # Ground Truth Images
            'pixel_values_pose': pixel_values_pose,  # Pose for Ground Truth Images
            'clip_ref_image': clip_ref_image,  # CLIP-processed Reference Image
            'pixel_values_ref_img': pixel_values_ref_img,  # main_reference.png
            'drop_image_embeds': drop_image_embeds,  # Drop Image Embeds Flag
            'pixel_values_ref_pose': pixel_values_ref_pose  # Reference Pose Image
        }
        
    def contrast_normalization(self, image, lower_bound=0, upper_bound=255):
        image = image.astype(np.float32)
        normalized_image = image  * (upper_bound - lower_bound) / 255 + lower_bound
        normalized_image = normalized_image.astype(np.uint8)

        return normalized_image

    def __getitem__(self, idx):
        sample = self.get_batch_wo_pose(idx)

        # Apply pixel transformations
        pixel_values = self.pixel_transforms(sample['pixel_values'])  # Ground Truth Images
        pixel_values_pose = self.pixel_transforms(sample['pixel_values_pose'])  # Pose Images

        # Process reference images
        pixel_values_ref_img = self.pixel_transforms(sample['pixel_values_ref_img'])  # main_reference.png
        pixel_values_ref_pose = self.pixel_transforms(sample['pixel_values_ref_pose'])  # Reference Pose Image

        # Create the sample dictionary with the required variables
        final_sample = {
            'pixel_values': pixel_values,  # Ground Truth Images
            'pixel_values_pose': pixel_values_pose,  # Pose for Ground Truth Images
            'clip_ref_image': sample['clip_ref_image'],  # CLIP-processed Reference Image
            'pixel_values_ref_img': pixel_values_ref_img,  # main_reference.png
            'drop_image_embeds': sample['drop_image_embeds'],  # Drop Image Embeds Flag
            'pixel_values_ref_pose': pixel_values_ref_pose  # Reference Pose Image
        }
        
        return final_sample

def collate_fn(data): 
    pixel_values = torch.stack([example["pixel_values"] for example in data])
    pixel_values_pose = torch.stack([example["pixel_values_pose"] for example in data])
    clip_ref_image = torch.cat([example["clip_ref_image"] for example in data])
    pixel_values_ref_img = torch.stack([example["pixel_values_ref_img"] for example in data])
    drop_image_embeds = [example["drop_image_embeds"] for example in data]
    drop_image_embeds = torch.Tensor(drop_image_embeds)
    pixel_values_ref_pose = torch.stack([example["pixel_values_ref_pose"] for example in data])

    return {
        "pixel_values": pixel_values,
        "pixel_values_pose": pixel_values_pose,
        "clip_ref_image": clip_ref_image,
        "pixel_values_ref_img": pixel_values_ref_img,
        "drop_image_embeds": drop_image_embeds,
        "pixel_values_ref_pose": pixel_values_ref_pose,
    }

