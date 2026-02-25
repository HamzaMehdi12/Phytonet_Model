import os
import json
import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class BotanicalDataset(Dataset):
    """
    COCO-style object detection dataset.

    COCO labels:  1 → Stem   2 → Tomato
    Model labels: 0 → Stem   1 → Tomato

    FIXES vs previous version
    ──────────────────────────
    FIX D1: image_ids now includes ALL images (not just annotated ones).
            Unannotated images act as hard-negative examples — the model must
            learn to output no boxes, not just ignore them.

    FIX D7: normal __getitem__ now returns 'image_id' key to match the
            mosaic path, giving a consistent dict contract throughout.

    FIX D8: load_mosaic now uses actual image dimensions (image.size) to
            normalise box coords, not COCO metadata which may be stale or wrong.
            (normal __getitem__ already did this correctly with orig_w/orig_h.)

    NOTED (not bugs):
    • use_mosaic=False in train.py → mosaic code is dormant but correct.
    • category_id_to_name built but unused — left for future use.
    • Non-albumentations transform fallback mutates self.transform once;
      harmless with per-worker copies but cleaned up for clarity.
    """

    def __init__(self, root_dir, img_size=224, mode="train", transform=None,
                 use_mosaic=True, mosaic_prob=0.5, current_epoch=0):
        self.root_dir     = root_dir
        self.img_size     = img_size
        self.mode         = mode
        self.transform    = transform
        self.use_mosaic   = use_mosaic and (mode == "train")
        self.mosaic_prob  = mosaic_prob
        self.current_epoch = current_epoch

        self.annotation_file = self._find_annotation_file()
        with open(self.annotation_file, "r") as f:
            data = json.load(f)

        self.image_id_to_info    = {img["id"]: img for img in data["images"]}
        self.category_id_to_name = {c["id"]: c["name"] for c in data["categories"]}

        self.image_annotations = {}
        for ann in data["annotations"]:
            self.image_annotations.setdefault(ann["image_id"], []).append(ann)

        # FIX D1: include ALL images (annotated or not).
        # Unannotated images are valid hard-negative training samples.
        # Previously: self.image_ids = list(self.image_annotations.keys())
        # which silently excluded every image with no boxes.
        self.image_ids = [img["id"] for img in data["images"]]

        # Build default transform once (not per-sample)
        self._default_transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"Loaded {len(self.image_ids)} images from {root_dir} "
              f"({len(self.image_annotations)} annotated)")
        self._print_label_distribution()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _find_annotation_file(self):
        for p in [
            os.path.join(self.root_dir, "_annotations.coco.json"),
            os.path.join(self.root_dir, "annotations.coco.json"),
            os.path.join(self.root_dir, "images", "_annotations.coco.json"),
            os.path.join(self.root_dir, "images", "annotations.coco.json"),
        ]:
            if os.path.exists(p):
                print(f"Found annotation file: {p}")
                return p
        raise FileNotFoundError("COCO annotation JSON not found in " + self.root_dir)

    def is_albumentations(self):
        return (self.transform is not None
                and "albumentations" in str(type(self.transform)).lower())

    def _print_label_distribution(self):
        counts = {0: 0, 1: 0}
        for anns in self.image_annotations.values():
            for ann in anns:
                cid = ann["category_id"]
                if cid in [1, 2]:
                    counts[cid - 1] += 1
        print("=" * 50)
        print("LABEL DISTRIBUTION")
        print(f"  0 (Stem):   {counts[0]}")
        print(f"  1 (Tomato): {counts[1]}")
        print("=" * 50)

    def __len__(self):
        return len(self.image_ids)

    # ------------------------------------------------------------------
    # Mosaic augmentation
    # ------------------------------------------------------------------

    def load_mosaic(self, index):
        """
        4-image mosaic (YOLOv4/v5/v8 style).

        FIX D8: box normalisation now uses actual loaded image size
                (image.size) instead of COCO metadata dims.
        """
        indices   = [index] + random.choices(range(len(self)), k=3)
        mosaic_sz = self.img_size * 2
        mosaic_img = Image.new("RGB", (mosaic_sz, mosaic_sz), (114, 114, 114))
        mosaic_boxes  = []
        mosaic_labels = []

        positions = [
            (0,              0),
            (self.img_size,  0),
            (0,              self.img_size),
            (self.img_size,  self.img_size),
        ]

        for i, img_idx in enumerate(indices):
            img_id = self.image_ids[img_idx]
            info   = self.image_id_to_info[img_id]
            anns   = self.image_annotations.get(img_id, [])

            img_path = os.path.join(self.root_dir, "images", info["file_name"])
            if not os.path.exists(img_path):
                continue
            image  = Image.open(img_path).convert("RGB")

            # FIX D8: use actual dims, not metadata
            orig_w, orig_h = image.size
            image = image.resize((self.img_size, self.img_size))

            paste_x, paste_y = positions[i]
            mosaic_img.paste(image, (paste_x, paste_y))

            for ann in anns:
                cid = ann["category_id"]
                if cid not in [1, 2]:
                    continue
                label = cid - 1
                x, y, w, h = ann["bbox"]
                x1 = max(0., min(1., x / orig_w))
                y1 = max(0., min(1., y / orig_h))
                x2 = max(0., min(1., (x + w) / orig_w))
                y2 = max(0., min(1., (y + h) / orig_h))
                if x2 <= x1 or y2 <= y1:
                    continue
                # Map to mosaic canvas (pixels), then normalise to [0,1]
                mosaic_boxes.append([
                    (x1 * self.img_size + paste_x) / mosaic_sz,
                    (y1 * self.img_size + paste_y) / mosaic_sz,
                    (x2 * self.img_size + paste_x) / mosaic_sz,
                    (y2 * self.img_size + paste_y) / mosaic_sz,
                ])
                mosaic_labels.append(label)

        # Resize canvas back to img_size — box normalisation stays valid
        mosaic_img = mosaic_img.resize((self.img_size, self.img_size))
        return mosaic_img, mosaic_boxes, mosaic_labels

    # ------------------------------------------------------------------
    # Main __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        # Mosaic scheduling
        if self.current_epoch < 15:
            use_mosaic_now = False
        elif self.current_epoch < 45:
            ramp = (self.current_epoch - 15) / 30 * self.mosaic_prob
            use_mosaic_now = self.use_mosaic and random.random() < ramp
        else:
            use_mosaic_now = self.use_mosaic and random.random() < self.mosaic_prob

        if use_mosaic_now:
            image, boxes, labels = self.load_mosaic(idx)
            if not boxes:
                boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
                labels_t = torch.zeros((0,),   dtype=torch.int64)
            else:
                boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
                labels_t = torch.tensor(labels, dtype=torch.int64)

            if self.is_albumentations():
                img_np      = np.array(image)
                pixel_boxes = [[b[0]*self.img_size, b[1]*self.img_size,
                                 b[2]*self.img_size, b[3]*self.img_size]
                                for b in boxes_t.tolist()]
                tr = self.transform(image=img_np, bboxes=pixel_boxes,
                                    labels=labels_t.tolist())
                image_tensor = tr["image"]
                if tr["bboxes"]:
                    boxes_t  = torch.tensor([[b[0]/self.img_size, b[1]/self.img_size,
                                              b[2]/self.img_size, b[3]/self.img_size]
                                             for b in tr["bboxes"]], dtype=torch.float32)
                    labels_t = torch.tensor(tr["labels"], dtype=torch.int64)
                else:
                    boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
                    labels_t = torch.zeros((0,),   dtype=torch.int64)
            else:
                image_tensor = T.ToTensor()(image)

            return image_tensor, {
                "boxes":      boxes_t,
                "labels":     labels_t,
                "image_id":   torch.tensor([idx]),   # consistent with normal path
                "image_path": "mosaic",
            }

        # ── Normal loading ──────────────────────────────────────────────
        img_id = self.image_ids[idx]
        info   = self.image_id_to_info[img_id]
        anns   = self.image_annotations.get(img_id, [])   # may be empty → []

        img_path = os.path.join(self.root_dir, "images", info["file_name"])
        if not os.path.exists(img_path):
            # Return blank image + empty boxes rather than crashing
            image = Image.new("RGB", (self.img_size, self.img_size), (114, 114, 114))
        else:
            image = Image.open(img_path).convert("RGB")

        orig_w, orig_h = image.size
        boxes, labels  = [], []

        for ann in anns:
            cid = ann["category_id"]
            if cid not in [1, 2]:
                continue
            label = cid - 1
            x, y, w, h = ann["bbox"]
            x1 = max(0., min(1., x / orig_w))
            y1 = max(0., min(1., y / orig_h))
            x2 = max(0., min(1., (x + w) / orig_w))
            y2 = max(0., min(1., (y + h) / orig_h))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(label)

        # ── Transforms ─────────────────────────────────────────────────
        if self.is_albumentations():
            img_np      = np.array(image)
            pixel_boxes = [[b[0]*orig_w, b[1]*orig_h,
                            b[2]*orig_w, b[3]*orig_h] for b in boxes]
            tr = self.transform(image=img_np, bboxes=pixel_boxes, labels=labels)
            image_tensor = tr["image"]
            pixel_boxes  = tr["bboxes"]
            labels       = tr["labels"]
            boxes = [[b[0]/self.img_size, b[1]/self.img_size,
                      b[2]/self.img_size, b[3]/self.img_size]
                     for b in pixel_boxes]
        else:
            image_tensor = self._default_transform(image)

        # ── To tensors ─────────────────────────────────────────────────
        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32).clamp(0, 1)
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.long)

        return image_tensor, {
            "boxes":      boxes_t,
            "labels":     labels_t,
            "image_id":   torch.tensor([img_id]),   # FIX D7: always present
            "image_path": img_path,
        }


# ── debug ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = BotanicalDataset(root_dir="data/train", img_size=224, mode="train")
    all_labels = []
    for i in range(min(10, len(ds))):
        _, t = ds[i]
        all_labels.extend(t["labels"].tolist())
        assert "image_id" in t, "image_id missing"
    unique = sorted(set(all_labels))
    print(f"Unique labels: {unique}")
    assert set(unique).issubset({0, 1}), "❌ Invalid labels"
    print("✅ Labels and image_id contract OK")
