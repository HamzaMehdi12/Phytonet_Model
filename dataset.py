import os 
import torch 
from torch.utils.data import Dataset 
from PIL import Image 
import json 
import numpy as np 
import torchvision.transforms as T 

class BotanicalDataset(Dataset): 
    def __init__(self, root_dir, img_size=224, mode='train', transform=None): 
        self.root_dir = root_dir 
        self.img_size = img_size 
        self.mode = mode
        
        # CRITICAL: Label remapping MUST be defined first
        # Dataset labels: {0: stem-tomato, 1: stem, 2: tomato}
        # Model labels: {0: stem, 1: tomato}
        self.label_mapping = {
            0: -1,  # Skip stem-tomato (not used)
            1: 0,   # Stem -> class 0
            2: 1    # Tomato -> class 1
        }
        
        # Set up transform
        if transform is not None: 
            self.transform = transform 
        else: 
            self.transform = T.Compose([
                T.Resize((img_size, img_size)), 
                T.ToTensor(), 
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Find annotation file 
        self.annotation_file = self._find_annotation_file() 
        
        # Load annotations 
        with open(self.annotation_file, 'r') as f: 
            data = json.load(f) 
        
        # Create mappings 
        self.image_id_to_info = {img['id']: img for img in data['images']} 
        self.category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']} 
        self.category_name_to_id = {cat['name']: cat['id'] for cat in data['categories']} 
        
        # Group annotations by image id 
        self.image_annotations = {} 
        
        for ann in data['annotations']: 
            image_id = ann['image_id'] 
            if image_id not in self.image_annotations: 
                self.image_annotations[image_id] = [] 
            self.image_annotations[image_id].append(ann) 
            
        # Create list of image IDs 
        self.image_ids = list(self.image_annotations.keys())
        
        print(f"Loaded {len(self.image_ids)} images from {root_dir}")
        self._print_label_distribution()
    
    def _find_annotation_file(self): 
        possible_paths = [
            os.path.join(self.root_dir, '_annotations.coco.json'), 
            os.path.join(self.root_dir, 'annotations.coco.json'), 
            os.path.join(self.root_dir, 'images', '_annotations.coco.json'), 
            os.path.join(self.root_dir, 'images', 'annotations.coco.json') 
        ] 
        
        for path in possible_paths: 
            if os.path.exists(path): 
                print(f"Found annotation file: {path}")
                return path 
        raise FileNotFoundError(f"No annotation file found in: {self.root_dir}")
    
    def _print_label_distribution(self):
        """Print original and remapped label distribution"""
        original_counts = {}
        remapped_counts = {0: 0, 1: 0}
        
        for annotations in self.image_annotations.values():
            for ann in annotations:
                orig_label = ann['category_id']
                original_counts[orig_label] = original_counts.get(orig_label, 0) + 1
                
                # Count remapped labels
                if orig_label in self.label_mapping:
                    new_label = self.label_mapping[orig_label]
                    if new_label >= 0:
                        remapped_counts[new_label] += 1
        
        print("\n" + "="*60)
        print("LABEL DISTRIBUTION:")
        print("="*60)
        print("Original labels (in dataset files):")
        for label, count in sorted(original_counts.items()):
            cat_name = self.category_id_to_name.get(label, f"Unknown-{label}")
            print(f"  {label} ({cat_name}): {count} boxes")
        
        print("\nRemapped labels (used for training):")
        print(f"  0 (stem):   {remapped_counts[0]} boxes")
        print(f"  1 (tomato): {remapped_counts[1]} boxes")
        print("="*60 + "\n")
    
    def __len__(self): 
        return len(self.image_ids) 
    
    def __getitem__(self, idx): 
        image_id = self.image_ids[idx] 
        image_info = self.image_id_to_info[image_id] 
        annotations = self.image_annotations[image_id] 
        
        # Load image 
        image_path = os.path.join(self.root_dir, 'images', image_info['file_name']) 
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # Process boxes and labels with remapping
        boxes = [] 
        labels = []
        
        for ann in annotations: 
            original_label = ann['category_id']
            
            # CRITICAL: Apply label remapping here!
            # Skip class 0 (stem-tomato), remap 1->0 (stem), 2->1 (tomato)
            if original_label not in self.label_mapping:
                continue  # Skip unknown labels
            
            new_label = self.label_mapping[original_label]
            if new_label < 0:  # Skip class 0 (stem-tomato)
                continue
            
            # Get bounding box 
            x, y, w, h = ann['bbox'] 
            
            # Convert to [x1, y1, x2, y2] format 
            x1 = x 
            y1 = y 
            x2 = x + w 
            y2 = y + h 
            
            # Normalize coordinates to [0, 1]
            x1 = x1 / orig_width 
            y1 = y1 / orig_height 
            x2 = x2 / orig_width 
            y2 = y2 / orig_height
            
            # Clamp to valid range
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))
            
            # Only add valid boxes
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(new_label)  # Use remapped label!
        
        # Apply image transformations
        if hasattr(self.transform, '__class__') and 'Compose' in str(self.transform.__class__):
            # Check if it's albumentations
            if hasattr(self.transform, 'processors'):
                # Albumentations transform
                image_np = np.array(image)
                
                # Convert normalized boxes to pixel coordinates for albumentations
                pixel_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    pixel_boxes.append([
                        x1 * orig_width,
                        y1 * orig_height,
                        x2 * orig_width,
                        y2 * orig_height
                    ])
                
                try:
                    transformed = self.transform(
                        image=image_np,
                        bboxes=pixel_boxes,
                        labels=labels
                    )
                    
                    image_tensor = transformed['image']
                    pixel_boxes = transformed['bboxes']
                    labels = transformed['labels']
                    
                    # Convert back to normalized [0, 1]
                    boxes = []
                    for box in pixel_boxes:
                        x1, y1, x2, y2 = box
                        boxes.append([
                            x1 / self.img_size,
                            y1 / self.img_size,
                            x2 / self.img_size,
                            y2 / self.img_size
                        ])
                except Exception as e:
                    print(f"Transform error: {e}")
                    # Fallback to basic transform
                    image_tensor = T.Compose([
                        T.Resize((self.img_size, self.img_size)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])(image)
            else:
                # Regular torchvision transform
                image_tensor = self.transform(image)
        else:
            # Fallback
            image_tensor = self.transform(image)
        
        # Convert to tensors 
        if len(boxes) > 0: 
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else: 
            boxes = torch.zeros((0, 4), dtype=torch.float32) 
            labels = torch.zeros((0,), dtype=torch.long)
        
        # Clamp boxes one more time
        if len(boxes) > 0:
            boxes = boxes.clamp(0, 1)
        
        target = {
            'boxes': boxes, 
            'labels': labels,  # These are now 0 and 1!
            'image_path': image_path
        }
        
        return image_tensor, target


# Testing function
if __name__ == "__main__":
    print("Testing BotanicalDataset...")
    
    # Test with your data directory
    dataset = BotanicalDataset(
        root_dir='data/train',
        img_size=224,
        mode='train'
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Check first few samples
    print("\nChecking first 5 samples:")
    for i in range(min(5, len(dataset))):
        img, target = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Image shape: {img.shape}")
        print(f"  Boxes: {target['boxes'].shape}")
        print(f"  Labels: {target['labels'].tolist()}")
        
        # Verify labels are 0 or 1
        if len(target['labels']) > 0:
            unique_labels = target['labels'].unique().tolist()
            if all(l in [0, 1] for l in unique_labels):
                print(f"  ✓ Labels correctly remapped: {unique_labels}")
            else:
                print(f"  ✗ WARNING: Unexpected labels: {unique_labels}")
    
    print("\n" + "="*60)
    print("Dataset test complete!")