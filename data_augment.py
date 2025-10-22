import os
import json
import shutil
import random
import cv2
import numpy as np
import albumentations as A

from tqdm import tqdm
from timeit import default_timer as timer


def augment_split(input_dir, output_dir, target_count, split_name):
    """
    Augments a single split (train/val/test) to reach target_count images
    
    Args:
        input_dir: Input directory for the split (e.g., 'data/train')
        output_dir: Output directory for the augmented split (e.g., 'data_aug/train')
        target_count: Target number of images for this split
        split_name: Name of the split (for progress messages)
    """
    # Paths for annotations and images
    images_dir = os.path.join(input_dir, 'images')
    json_path = os.path.join(images_dir, '_annotations.coco.json')
    
    # Create output directories
    output_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Load COCO annotations
    if not os.path.exists(json_path):
        print(f"Annotation file not found: {json_path}")
        return
    with open(json_path) as f:
        coco_data = json.load(f)
    
    # Copy original images to output directory
    for img_info in coco_data['images']:
        src = os.path.join(images_dir, img_info['file_name'])
        dst = os.path.join(output_images_dir, img_info['file_name'])
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Image not found: {src}")
    
    # Calculate augmentation needs
    num_originals = len(coco_data['images'])
    if num_originals == 0:
        print(f"No images found in {input_dir}. Skipping augmentation.")
        return
    
    # Prepare new dataset containers
    new_images = []
    new_annotations = []
    annotation_id_counter = 1
    
    # Create augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.RandomGamma(p=0.3),
        A.CLAHE(p=0.3),
        A.RandomScale(scale_limit=(-0.5, 1.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    ], bbox_params=A.BboxParams(format='coco', min_visibility=0.2))
    
    # Process each original image
    for img_info in tqdm(coco_data['images'], desc=f"Augmenting {split_name}"):
        # Load original image
        img_path = os.path.join(images_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            continue
        
        # Get annotations for this image
        annotations = [a for a in coco_data['annotations'] 
                      if a['image_id'] == img_info['id']]
        bboxes = [a['bbox'] for a in annotations]
        category_ids = [a['category_id'] for a in annotations]
        
        # Add original image to new dataset
        new_image_id = len(new_images) + 1
        new_images.append({
            **img_info,
            "id": new_image_id
        })
        
        # Copy annotations for original image
        for ann in annotations:
            new_annotations.append({
                **ann,
                "id": annotation_id_counter,
                "image_id": new_image_id
            })
            annotation_id_counter += 1
    
    # Calculate how many more images we need to generate
    num_needed = target_count - len(new_images)
    if num_needed <= 0:
        print(f"{split_name} already has {len(new_images)} images (target: {target_count})")
    else:
        # Calculate augmentations per image
        aug_per_image = num_needed // num_originals
        remainder = num_needed % num_originals
        
        # Generate augmented images
        for idx, img_info in enumerate(tqdm(coco_data['images'], desc="Creating augmentations")):
            img_path = os.path.join(images_dir, img_info['file_name'])
            if not os.path.exists(img_path):
                continue
                
            try:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                continue
            
            # Get annotations
            annotations = [a for a in coco_data['annotations'] 
                          if a['image_id'] == img_info['id']]
            bboxes = [a['bbox'] for a in annotations]
            category_ids = [a['category_id'] for a in annotations]
            
            # Determine how many augmentations to create for this image
            num_augmentations = aug_per_image + (1 if idx < remainder else 0)
            
            for aug_idx in range(num_augmentations):
                try:
                    # Apply transformations
                    transformed = transform(
                        image=image, 
                        bboxes=bboxes, 
                        category_ids=category_ids
                    )
                except Exception as e:
                    print(f"Skipping augmentation due to error: {str(e)}")
                    continue
                    
                # Skip if no bounding boxes survived
                if not transformed['bboxes']:
                    continue
                    
                # Create new image filename
                base, ext = os.path.splitext(img_info['file_name'])
                new_filename = f"{base}_aug{aug_idx}{ext}"
                new_filepath = os.path.join(output_images_dir, new_filename)
                
                # Save augmented image
                try:
                    transformed_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(new_filepath, transformed_image)
                except Exception as e:
                    print(f"Failed to save augmented image: {str(e)}")
                    continue
                
                # Create new image entry
                new_image_id = len(new_images) + 1
                new_images.append({
                    "id": new_image_id,
                    "file_name": new_filename,
                    "height": transformed_image.shape[0],
                    "width": transformed_image.shape[1],
                    "license": img_info.get('license', 1),
                    "date_captured": img_info.get('date_captured', ''),
                })
                
                # Create new annotations
                for bbox, cat_id in zip(transformed['bboxes'], transformed['category_ids']):
                    area = bbox[2] * bbox[3]  # width * height
                    new_annotations.append({
                        "id": annotation_id_counter,
                        "image_id": new_image_id,
                        "category_id": cat_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    annotation_id_counter += 1
    
    # Create COCO data structure
    new_coco = {
        "info": coco_data["info"],
        "licenses": coco_data["licenses"],
        "categories": coco_data["categories"],
        "images": new_images,
        "annotations": new_annotations
    }
    
    # Save new annotations in the images folder
    output_json = os.path.join(output_images_dir, '_annotations.coco.json')
    with open(output_json, 'w') as f:
        json.dump(new_coco, f)
    
    print(f"Created {split_name} set with {len(new_images)} images and {len(new_annotations)} annotations")

def augment_dataset():
    """Augments the entire dataset to target counts per split"""
    # Define splits and target counts
    splits = [
        ('train', 5000),
        ('val', 1000),
        ('test', 1000)
    ]
    
    # Create output base directory
    base_output_dir = 'data_aug'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Process each split
    for split_name, target_count in splits:
        input_dir = os.path.join('data', split_name)
        output_dir = os.path.join(base_output_dir, split_name)
        
        print(f"\n{'='*50}")
        print(f"Processing {split_name.upper()} split")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Target count: {target_count} images")
        print(f"{'='*50}")
        
        augment_split(input_dir, output_dir, target_count, split_name)
    
    print("\nAugmentation complete! Final dataset saved in: data_aug")
    print("Directory structure:")
    print("data_aug/")
    print("├── train/")
    print("│   └── images/")
    print("│       ├── *.jpg (original + augmented)")
    print("│       └── _annotations.coco.json")
    print("├── val/")
    print("│   └── images/")
    print("└── test/")
    print("    └── images/")

if __name__ == "__main__":
    # Install required packages if missing
    try:
        start_time = timer()
        import albumentations
        print("Starting the process")
        augment_dataset()
        end_time = timer()
        print(f"Total time: {end_time-start_time:.2f}")
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.run(['pip', 'install', 'albumentations', 'opencv-python', 'tqdm'])
