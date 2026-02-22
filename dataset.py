import os
import json
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class BotanicalDataset(Dataset):
    """
    COCO-style object detection dataset

    COCO labels:
        1 -> Stem
        2 -> Tomato

    Model labels:
        0 -> Stem
        1 -> Tomato
    """

    def __init__(self, root_dir, img_size=224, mode="train", transform=None, use_mosaic=True, mosaic_prob=0.5, current_epoch=0):
        self.root_dir = root_dir
        self.img_size = img_size
        self.mode = mode
        self.transform = transform
        self.use_mosaic = use_mosaic and (mode == "train")  # Only use mosaic in training
        self.mosaic_prob = mosaic_prob
        self.current_epoch = current_epoch  # Track current epoch for mosaic scheduling

        # ---------------------------------------------------------
        # Load annotations
        # ---------------------------------------------------------
        self.annotation_file = self._find_annotation_file()
        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        self.image_id_to_info = {img["id"]: img for img in data["images"]}
        self.category_id_to_name = {c["id"]: c["name"] for c in data["categories"]}

        self.image_annotations = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            self.image_annotations.setdefault(img_id, []).append(ann)

        self.image_ids = list(self.image_annotations.keys())

        print(f"Loaded {len(self.image_ids)} images from {root_dir}")
        self._print_label_distribution()

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------
    def _find_annotation_file(self):
        paths = [
            os.path.join(self.root_dir, "_annotations.coco.json"),
            os.path.join(self.root_dir, "annotations.coco.json"),
            os.path.join(self.root_dir, "images", "_annotations.coco.json"),
            os.path.join(self.root_dir, "images", "annotations.coco.json"),
        ]
        for p in paths:
            if os.path.exists(p):
                print(f"Found annotation file: {p}")
                return p
        raise FileNotFoundError("COCO annotation JSON not found")

    def is_albumentations(self):
        return (
            self.transform is not None
            and "albumentations" in str(type(self.transform)).lower()
        )

    def _print_label_distribution(self):
        counts = {0: 0, 1: 0}

        for anns in self.image_annotations.values():
            for ann in anns:
                cid = ann["category_id"]
                if cid in [1, 2]:
                    counts[cid - 1] += 1

        print("\n" + "=" * 60)
        print("LABEL DISTRIBUTION (MODEL LABELS)")
        print("=" * 60)
        print(f"0 (Stem):   {counts[0]} boxes")
        print(f"1 (Tomato): {counts[1]} boxes")
        print("=" * 60 + "\n")

    # ---------------------------------------------------------
    # Dataset API
    # ---------------------------------------------------------
    def __len__(self):
        return len(self.image_ids)

    def load_mosaic(self, index):
        """
        Mosaic augmentation - combines 4 images into one
        YOLOv4/v5/v8 technique for better small object detection
        Returns mosaic image and combined targets
        """
        # Select 3 additional random images
        indices = [index] + random.choices(range(len(self)), k=3)
        
        # Create mosaic canvas (2x img_size)
        mosaic_size = self.img_size * 2
        mosaic_img = Image.new('RGB', (mosaic_size, mosaic_size), (114, 114, 114))
        mosaic_boxes = []
        mosaic_labels = []
        
        # Define quadrants (top-left, top-right, bottom-left, bottom-right)
        positions = [
            (0, 0),                          # top-left
            (self.img_size, 0),              # top-right
            (0, self.img_size),              # bottom-left
            (self.img_size, self.img_size)   # bottom-right
        ]
        
        for i, img_idx in enumerate(indices):
            # Load image without mosaic (avoid recursion)
            img_id = self.image_ids[img_idx]
            info = self.image_id_to_info[img_id]
            anns = self.image_annotations[img_id]
            
            img_path = os.path.join(self.root_dir, "images", info["file_name"])
            image = Image.open(img_path).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            
            # Parse boxes
            orig_w, orig_h = self.img_size, self.img_size
            boxes = []
            labels = []
            
            for ann in anns:
                cid = ann["category_id"]
                if cid not in [1, 2]:
                    continue
                label = cid - 1
                
                x, y, w, h = ann["bbox"]
                x1 = x / info["width"]
                y1 = y / info["height"]
                x2 = (x + w) / info["width"]
                y2 = (y + h) / info["height"]
                
                x1 = max(0, min(1, x1))
                y1 = max(0, min(1, y1))
                x2 = max(0, min(1, x2))
                y2 = max(0, min(1, y2))
                
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(label)
            
            # Place image in quadrant
            paste_x, paste_y = positions[i]
            mosaic_img.paste(image, (paste_x, paste_y))
            
            # Adjust boxes to mosaic coordinates
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                # Scale to image size
                x1 = x1 * self.img_size + paste_x
                y1 = y1 * self.img_size + paste_y
                x2 = x2 * self.img_size + paste_x
                y2 = y2 * self.img_size + paste_y
                
                # Normalize to mosaic size
                mosaic_boxes.append([
                    x1 / mosaic_size,
                    y1 / mosaic_size,
                    x2 / mosaic_size,
                    y2 / mosaic_size
                ])
                mosaic_labels.append(label)
        
        # Resize back to img_size
        mosaic_img = mosaic_img.resize((self.img_size, self.img_size))
        
        return mosaic_img, mosaic_boxes, mosaic_labels

    def __getitem__(self, idx):
        # Mosaic augmentation with epoch-based scheduling
        # Enable mosaic earlier at epoch 10 for faster learning
        if self.current_epoch < 10:
            use_mosaic_this_epoch = False
        elif self.current_epoch < 40:
            # Ramp up mosaic from 0% to 50% between epoch 10-40
            ramp_prob = (self.current_epoch - 10) / 30 * self.mosaic_prob
            use_mosaic_this_epoch = self.use_mosaic and random.random() < ramp_prob
        else:
            # Full mosaic after epoch 40
            use_mosaic_this_epoch = self.use_mosaic and random.random() < self.mosaic_prob
        
        if use_mosaic_this_epoch:
            image, boxes, labels = self.load_mosaic(idx)
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
            
            # Apply transforms if any
            if self.transform and self.is_albumentations():
                image_np = np.array(image)
                pixel_boxes = [
                    [b[0] * self.img_size, b[1] * self.img_size, 
                     b[2] * self.img_size, b[3] * self.img_size]
                    for b in boxes.tolist()
                ]
                
                transformed = self.transform(
                    image=image_np,
                    bboxes=pixel_boxes,
                    labels=labels.tolist()
                )
                
                image = transformed["image"]
                if len(transformed["bboxes"]) > 0:
                    boxes = torch.tensor([
                        [b[0] / self.img_size, b[1] / self.img_size,
                         b[2] / self.img_size, b[3] / self.img_size]
                        for b in transformed["bboxes"]
                    ], dtype=torch.float32)
                    labels = torch.tensor(transformed["labels"], dtype=torch.int64)
                else:
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.int64)
            else:
                # Convert to tensor
                if isinstance(image, Image.Image):
                    image = T.ToTensor()(image)
            
            return image, {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([idx]),
                "image_path": "mosaic"
            }
        
        # Normal loading (original code)
        img_id = self.image_ids[idx]
        info = self.image_id_to_info[img_id]
        anns = self.image_annotations[img_id]

        img_path = os.path.join(self.root_dir, "images", info["file_name"])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        boxes = []
        labels = []

        # -----------------------------------------------------
        # Parse annotations
        # -----------------------------------------------------
        for ann in anns:
            cid = ann["category_id"]

            # COCO (1,2) → Model (0,1)
            if cid not in [1, 2]:
                continue

            label = cid - 1  # <-- CRITICAL FIX

            x, y, w, h = ann["bbox"]

            x1 = x / orig_w
            y1 = y / orig_h
            x2 = (x + w) / orig_w
            y2 = (y + h) / orig_h

            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(label)
        
        if len(boxes) == 0:
            boxes = []
            labels = []

        # -----------------------------------------------------
        # Transforms
        # -----------------------------------------------------
        if self.is_albumentations():
            image_np = np.array(image)

            pixel_boxes = [
                [b[0] * orig_w, b[1] * orig_h, b[2] * orig_w, b[3] * orig_h]
                for b in boxes
            ]

            transformed = self.transform(
                image=image_np,
                bboxes=pixel_boxes,
                labels=labels
            )

            image_tensor = transformed["image"]
            pixel_boxes = transformed["bboxes"]
            labels = transformed["labels"]

            boxes = [
                [
                    b[0] / self.img_size,
                    b[1] / self.img_size,
                    b[2] / self.img_size,
                    b[3] / self.img_size,
                ]
                for b in pixel_boxes
            ]

        else:
            if self.transform is None:
                self.transform = T.Compose([
                    T.Resize((self.img_size, self.img_size)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])

            image_tensor = self.transform(image)

        # -----------------------------------------------------
        # To tensors
        # -----------------------------------------------------
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32).clamp(0, 1)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        return image_tensor, {
            "boxes": boxes,
            "labels": labels,
            "image_path": img_path,
        }


# -------------------------------------------------------------------------
# Debug test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing BotanicalDataset (COCO 1→2 → Model 0→1)")

    dataset = BotanicalDataset(
        root_dir="data/train",
        img_size=224,
        mode="train",
    )

    all_labels = []
    for i in range(min(10, len(dataset))):
        _, target = dataset[i]
        all_labels.extend(target["labels"].tolist())

    print("\nUnique labels found:", sorted(set(all_labels)))
    assert set(all_labels).issubset({0, 1}), "❌ Invalid labels detected"
    print("✅ Labels are correct (0:Stem, 1:Tomato)")
