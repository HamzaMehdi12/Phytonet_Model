import os
import json
import torch
import numpy as np
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

    def __init__(self, root_dir, img_size=224, mode="train", transform=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.mode = mode
        self.transform = transform

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

    def __getitem__(self, idx):
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
            boxes.append([0.0, 0.0, 0.01, 0.01])
            labels.append(0)

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
