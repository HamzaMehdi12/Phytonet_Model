import os
import time
import argparse
import math
import gc
import json
import random
from copy import deepcopy

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.ops import nms, box_iou
from torchmetrics.detection import MeanAveragePrecision
from thop import profile

import albumentations as A
from albumentations.pytorch import ToTensorV2

from phytonet import (HighAccuracyPhytoSparseNet, HighAccuracyPhytoSparseNetStrong,
                       PhytoNetEdge)
from botanical_loss import DetectionLoss, MultiHeadDetectionLoss
from dataset import BotanicalDataset


# ─────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────

class Config:
    """Centralized configuration for anchors and training params."""
    
    # Anchors tuned for stems (~10-25px) and tomatoes (~30-130px) at 224px input
    # Each head gets 3 anchors sized for its receptive field
    ANCHORS_SMALL = [[10, 6], [15, 9], [22, 14]]      # stride 8  → 28×28
    ANCHORS_MEDIUM = [[28, 18], [38, 25], [55, 35]]   # stride 16 → 14×14
    ANCHORS_LARGE = [[70, 45], [95, 60], [130, 80]]   # stride 32 → 7×7
    
    # Combined anchors for legacy models
    ANCHORS_ALL = ANCHORS_SMALL + ANCHORS_MEDIUM + ANCHORS_LARGE
    
    # Loss weights per head (small objects dominate in botanical data)
    HEAD_WEIGHTS = (0.5, 0.35, 0.15)  # (small, medium, large)
    
    # Class weights: stems are ~5× rarer and harder to detect
    CLASS_WEIGHTS = [4.0, 1.0]  # [stem, tomato]


# ─────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def create_augmentations(img_size=224):
    """Create training augmentation pipeline."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        A.RandomScale(scale_limit=0.3, p=0.5),
        A.Perspective(scale=(0.02, 0.08), p=0.3),
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc', 
        label_fields=['labels'],
        min_visibility=0.3
    ))


def create_val_transforms(img_size=224):
    """Create validation/test transform pipeline (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.1
    ))


# ─────────────────────────────────────────────────────────────
#  Model-output → loss-ready conversions
# ─────────────────────────────────────────────────────────────

def tensor_to_pred_dict(tensor, num_classes=2):
    """
    Convert raw model output tensor to prediction dictionary.
    
    Args:
        tensor: [B, C, H, W] where C = num_anchors * (5 + num_classes)
        num_classes: number of object classes
    
    Returns:
        dict with pred_boxes[B,N,4], pred_obj[B,N], pred_cls[B,N,C]
    """
    B, C, H, W = tensor.shape
    vals_per_anchor = 5 + num_classes
    num_anchors = C // vals_per_anchor
    
    if C % vals_per_anchor != 0:
        raise ValueError(f"Channels {C} not divisible by {vals_per_anchor}")
    
    # Reshape: [B, A, vals, H, W] -> [B, A, H, W, vals] -> [B, A*H*W, vals]
    pred = tensor.view(B, num_anchors, vals_per_anchor, H, W)
    pred = pred.permute(0, 1, 3, 4, 2).contiguous()
    pred = pred.view(B, num_anchors * H * W, vals_per_anchor)
    
    return {
        'pred_boxes': pred[..., :4],
        'pred_obj': pred[..., 4],
        'pred_cls': pred[..., 5:]
    }


def prepare_predictions_for_loss(model_output, num_classes=2):
    """
    Convert raw model output to prediction dictionaries for loss computation.
    
    Supports:
        - PhytoNetEdge: dict with 'small', 'medium', 'large' keys
        - Legacy 2-head: dict with 'large', 'medium' keys
        - Single tensor: direct conversion
    """
    if isinstance(model_output, dict):
        # Already in correct format
        if {'pred_boxes', 'pred_cls', 'pred_obj'}.issubset(model_output):
            return model_output
        
        # Multi-head model
        out = {}
        for key in ('small', 'medium', 'large'):
            if key in model_output:
                out[key] = tensor_to_pred_dict(model_output[key], num_classes)
        
        if out:
            return out
        
        raise TypeError(f"Dict with unexpected keys: {list(model_output.keys())}")
    
    if isinstance(model_output, torch.Tensor):
        return tensor_to_pred_dict(model_output, num_classes)
    
    raise TypeError(f"Expected dict or Tensor, got {type(model_output)}")


def prepare_targets_for_loss(raw_targets, output_shape, img_size=224,
                              anchors=None, num_classes=2, head_name=None):
    """
    Build target tensors for ONE detection head.
    
    CRITICAL: anchors MUST match the head's expected anchor count.
    
    Args:
        raw_targets: list of dicts with 'boxes' and 'labels'
        output_shape: (B, C, H, W) shape of this head's output
        img_size: input image size
        anchors: list of [w, h] anchor sizes for THIS head only
        num_classes: number of object classes
        head_name: 'small', 'medium', or 'large' for debugging
    
    Returns:
        dict with 'obj', 'cls', 'boxes' target tensors
    """
    if anchors is None:
        raise ValueError("anchors must be explicitly provided for each head")
    
    device = raw_targets[0]['boxes'].device
    batch_size = len(raw_targets)
    _, C, H, W = output_shape
    
    anchors_t = torch.tensor(anchors, dtype=torch.float32, device=device)
    num_anchors = anchors_t.shape[0]
    anchors_norm = anchors_t / float(img_size)
    
    # Validate channel count
    expected_channels = num_anchors * (5 + num_classes)
    if C != expected_channels:
        raise ValueError(f"Channel mismatch in {head_name}: output has {C} channels, "
                        f"expected {expected_channels} for {num_anchors} anchors")
    
    # Initialize targets
    N = num_anchors * H * W
    target_obj = torch.zeros(batch_size, N, device=device)
    target_cls = torch.zeros(batch_size, N, num_classes, device=device)
    target_boxes = torch.zeros(batch_size, N, 4, device=device)
    
    total_pos = stem_pos = tom_pos = 0
    
    for b in range(batch_size):
        gt_boxes = raw_targets[b]['boxes']
        gt_labels = raw_targets[b]['labels']
        
        if len(gt_boxes) == 0:
            continue
        
        # Box center and size (normalized)
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
        gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        
        # Grid cell indices
        gx = (gt_cx * W).long().clamp(0, W - 1)
        gy = (gt_cy * H).long().clamp(0, H - 1)
        
        for i in range(len(gt_boxes)):
            gxi, gyi = gx[i].item(), gy[i].item()
            gwi, ghi = gt_w[i], gt_h[i]
            lbl = int(gt_labels[i].item())
            
            # Compute IoU with all anchors
            ious = []
            for an in anchors_norm:
                aw, ah = an[0], an[1]
                inter = torch.min(gwi, aw) * torch.min(ghi, ah)
                union = gwi * ghi + aw * ah - inter
                ious.append((inter / (union + 1e-6)).item())
            
            ious_t = torch.tensor(ious, device=device)
            best_anchor = int(ious_t.argmax().item())
            
            # Anchor matching strategy based on class
            # Stems are thin → need lower threshold, more anchors
            # Tomatoes are rounder → can be stricter
            if lbl == 0:  # stem
                top_k, thresh = 2, 0.15
            else:  # tomato
                top_k, thresh = 1, 0.25
            
            _, top_idx = torch.topk(ious_t, min(top_k, len(ious_t)))
            matching = [int(a.item()) for a in top_idx if ious_t[a] >= thresh]
            
            # Always include best anchor
            if best_anchor not in matching:
                matching.append(best_anchor)
            
            # Assign targets
            lbl_c = max(0, min(lbl, num_classes - 1))
            for ai in matching:
                idx = ai * H * W + gyi * W + gxi
                target_obj[b, idx] = 1.0
                target_cls[b, idx, lbl_c] = 1.0
                target_boxes[b, idx] = gt_boxes[i]
                
                total_pos += 1
                if lbl == 0:
                    stem_pos += 1
                else:
                    tom_pos += 1
    
    # Debug output
    if batch_size > 0 and head_name:
        print(f"[{H}×{W}] pos/img={total_pos/batch_size:.1f} "
              f"(stem={stem_pos/batch_size:.1f} "
              f"tom={tom_pos/batch_size:.1f})", end="  ")
    
    return {'obj': target_obj, 'cls': target_cls, 'boxes': target_boxes}


# ─────────────────────────────────────────────────────────────
#  Decode / NMS
# ─────────────────────────────────────────────────────────────

def decode_single_head(pred, anchors, img_size=224, conf_thresh=0.25,
                       box_scale=1.0, num_classes=2):
    """
    Decode predictions from a single detection head.
    
    Args:
        pred: [C, H, W] raw predictions for one image
        anchors: list of [w, h] anchor sizes
        img_size: input image size
        conf_thresh: confidence threshold
        box_scale: box size multiplier
        num_classes: number of classes
    
    Returns:
        boxes [N, 4], scores [N], class_ids [N] (all normalized to [0,1])
    """
    device = pred.device
    anchors_t = torch.tensor(anchors, dtype=torch.float32, device=device)
    A = len(anchors)
    C, H, W = pred.shape
    
    vals_per_anchor = 5 + num_classes
    if C != A * vals_per_anchor:
        raise ValueError(f"Channel mismatch: {C} != {A} * {vals_per_anchor}")
    
    # Reshape to [A, H, W, vals]
    pred = pred.view(A, vals_per_anchor, H, W).permute(0, 2, 3, 1).contiguous()
    
    # Grid coordinates
    gy, gx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    gx = gx.view(1, H, W, 1).expand(A, H, W, 1)
    gy = gy.view(1, H, W, 1).expand(A, H, W, 1)
    
    # Decode boxes
    cx = (torch.sigmoid(pred[..., 0:1]) + gx) / W
    cy = (torch.sigmoid(pred[..., 1:2]) + gy) / H
    
    an = anchors_t / float(img_size)
    aw = an[:, 0].view(A, 1, 1, 1)
    ah = an[:, 1].view(A, 1, 1, 1)
    
    bw = torch.exp(pred[..., 2:3].clamp(-5, 5)) * aw * box_scale
    bh = torch.exp(pred[..., 3:4].clamp(-5, 5)) * ah * box_scale
    
    x1 = (cx - bw / 2).reshape(-1)
    y1 = (cy - bh / 2).reshape(-1)
    x2 = (cx + bw / 2).reshape(-1)
    y2 = (cy + bh / 2).reshape(-1)
    boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)
    
    # Decode scores
    obj_prob = torch.sigmoid(pred[..., 4]).reshape(-1)
    cls_prob = torch.sigmoid(pred[..., 5:]).reshape(-1, num_classes)
    cls_scores, cls_ids = cls_prob.max(dim=-1)
    
    # Combined score (geometric mean)
    scores = torch.sqrt(obj_prob * cls_scores)
    
    # Filter by confidence
    keep = scores > conf_thresh
    
    return boxes[keep], scores[keep], cls_ids[keep]


def decode_and_merge_heads(model_output, args, conf_thresh):
    """
    Decode all detection heads and merge with cross-head NMS.
    
    Args:
        model_output: dict with per-image tensors {key: [C,H,W]}
        args: arguments with anchor configs
        conf_thresh: confidence threshold
    
    Returns:
        boxes [N, 4], scores [N], class_ids [N] (normalized to [0,1])
    """
    all_boxes, all_scores, all_cls = [], [], []
    device = None
    
    # Map heads to their anchors
    head_config = {
        'small': Config.ANCHORS_SMALL,
        'medium': Config.ANCHORS_MEDIUM,
        'large': Config.ANCHORS_LARGE,
    }
    
    for head_name, tensor in model_output.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        
        device = tensor.device
        anchors = head_config.get(head_name, Config.ANCHORS_ALL[:3])
        
        boxes, scores, cls_ids = decode_single_head(
            tensor,
            anchors=anchors,
            img_size=args.img_size,
            conf_thresh=conf_thresh,
            box_scale=args.box_scale,
            num_classes=2
        )
        
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_cls.append(cls_ids)
    
    # Handle empty predictions
    if not any(len(b) > 0 for b in all_boxes):
        return (torch.empty((0, 4), device=device),
                torch.empty((0,), device=device),
                torch.empty((0,), dtype=torch.int64, device=device))
    
    # Concatenate all predictions
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    class_ids = torch.cat(all_cls)
    
    # Per-class NMS
    abs_boxes = boxes * args.img_size
    final_boxes, final_scores, final_cls = [], [], []
    
    for c in class_ids.unique():
        mask = class_ids == c
        c_boxes = abs_boxes[mask]
        c_scores = scores[mask]
        
        keep = nms(c_boxes, c_scores, args.iou_thresh)[:300]
        
        final_boxes.append(c_boxes[keep])
        final_scores.append(c_scores[keep])
        final_cls.append(torch.full((len(keep),), int(c.item()), 
                                    dtype=torch.int64, device=device))
    
    if not final_boxes:
        return (torch.empty((0, 4), device=device),
                torch.empty((0,), device=device),
                torch.empty((0,), dtype=torch.int64, device=device))
    
    boxes = torch.cat(final_boxes) / float(args.img_size)
    scores = torch.cat(final_scores)
    class_ids = torch.cat(final_cls)
    
    # Filter invalid boxes
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    
    return boxes[valid], scores[valid], class_ids[valid]


# ─────────────────────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────────────────────

def draw_dashed_rectangle(img, pt1, pt2, color, thickness, dash=10):
    """Draw a dashed rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def save_detection_image(img_tensor, target, preds, path, class_names, args):
    """Save visualization of detections vs ground truth."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = ((img_tensor.cpu() * std + mean).clamp(0, 1)
               .numpy().transpose(1, 2, 0) * 255).astype(np.uint8).copy()
        h, w = img.shape[:2]
        
        boxes, scores, cls_ids = preds
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.numpy()
        if isinstance(cls_ids, torch.Tensor):
            cls_ids = cls_ids.numpy()
        
        # Draw predictions (green)
        for i in range(len(boxes)):
            if float(scores[i]) < args.eval_conf_thresh:
                continue
            x1 = int(max(0, boxes[i][0] * w))
            y1 = int(max(0, boxes[i][1] * h))
            x2 = int(min(w - 1, boxes[i][2] * w))
            y2 = int(min(h - 1, boxes[i][3] * h))
            if x2 <= x1 or y2 <= y1:
                continue
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lbl = f"{class_names.get(int(cls_ids[i]), '?')}: {float(scores[i]):.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(img, lbl, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45, (255, 255, 255), 1)
        
        # Draw ground truth (red dashed)
        if target:
            gb = target['boxes']
            gl = target['labels']
            if isinstance(gb, torch.Tensor):
                gb = gb.numpy()
            if isinstance(gl, torch.Tensor):
                gl = gl.numpy()
            
            for i in range(len(gb)):
                x1 = int(max(0, gb[i][0] * w))
                y1 = int(max(0, gb[i][1] * h))
                x2 = int(min(w - 1, gb[i][2] * w))
                y2 = int(min(h - 1, gb[i][3] * h))
                if x2 <= x1 or y2 <= y1:
                    continue
                
                draw_dashed_rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                lbl = f"GT:{class_names.get(int(gl[i]), '?')}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img, (x1, y2), (x1 + tw, y2 + th + 6), (0, 0, 255), -1)
                cv2.putText(img, lbl, (x1, y2 + th), cv2.FONT_HERSHEY_SIMPLEX,
                           0.45, (255, 255, 255), 1)
        
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"save_detection_image error: {e}")


def plot_training_curves(train_loss, val_metrics, output_dir):
    """Plot training loss and validation metrics."""
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    if val_metrics:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, [m.get('map', 0) for m in val_metrics], 'r-', label='mAP')
        plt.plot(epochs, [m.get('map_50', 0) for m in val_metrics], 'g-', label='mAP@50')
        plt.legend()
        plt.ylim(0, 1)
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()


def plot_confusion_matrix(cm, class_names, path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def log_model_info(model, img_size, device, output_dir):
    """Log model architecture information."""
    try:
        x = torch.randn(1, 3, img_size, img_size).to(device)
        flops, _ = profile(model, inputs=(x,), verbose=False)
        
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        size_fp32 = total * 4 / 1024**2
        size_int8 = total * 1 / 1024**2
        
        info = {
            'total_params': total,
            'trainable_params': trainable,
            'frozen_params': frozen,
            'flops': flops,
            'size_fp32_mb': round(size_fp32, 2),
            'size_int8_mb': round(size_int8, 2),
            'img_size': img_size,
        }
        
        with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n{'─' * 46}")
        print(f"  PhytoNetEdge – Model Summary")
        print(f"{'─' * 46}")
        print(f"  Total params      : {total:>12,}")
        print(f"  Trainable params  : {trainable:>12,}")
        print(f"  Frozen params     : {frozen:>12,}")
        print(f"  GFLOPs @ {img_size}px   : {flops/1e9:>11.3f}")
        print(f"  Size (FP32)       : {size_fp32:>10.2f} MB")
        print(f"  Size (INT8 est.)  : {size_int8:>10.2f} MB")
        print(f"{'─' * 46}\n")
        
        return info
    except Exception as e:
        print(f"log_model_info: {e}")
        return {}
# ─────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────

def validate_model(model, dataloader, device, class_names, args, epoch, phase='val'):
    """
    Validate model on a dataset.
    
    Uses decode_and_merge_heads() to properly combine predictions from
    all detection heads (small/medium/large).
    """
    model.eval()
    
    val_metrics = {
        'map': 0., 'map_50': 0., 'map_75': 0., 'mar_100': 0.,
        'overall_precision': 0., 'overall_recall': 0., 'overall_f1': 0.
    }
    for cn in class_names.values():
        val_metrics.update({
            f'{cn}_precision': 0., f'{cn}_recall': 0., f'{cn}_f1': 0.,
            f'{cn}_tp': 0, f'{cn}_fp': 0, f'{cn}_fn': 0
        })
    
    all_true, all_pred = [], []
    map_metric = MeanAveragePrecision(class_metrics=True)
    map_metric.warn_on_many_detections = False
    
    try:
        with torch.no_grad():
            for idx, (imgs, targets) in enumerate(
                    tqdm(dataloader, desc=f"[{phase}] epoch {epoch}", leave=False)):
                imgs = imgs.to(device)
                out = model(imgs)
                
                for b in range(imgs.shape[0]):
                    # Extract per-image tensors for each head
                    if isinstance(out, dict):
                        per_img = {k: v[b] for k, v in out.items()
                                   if isinstance(v, torch.Tensor)}
                    else:
                        per_img = {'single': out[b]}
                    
                    # Decode and merge all heads
                    boxes, scores, cls_ids = decode_and_merge_heads(
                        per_img, args, conf_thresh=args.eval_conf_thresh)

                    boxes = boxes.cpu()
                    scores = scores.cpu()
                    cls_ids = cls_ids.cpu()
                    gt_boxes = targets[b]['boxes'].cpu()
                    gt_labels = targets[b]['labels'].cpu()
                    
                    if len(boxes) == 0:
                        boxes = torch.empty((0, 4))
                        scores = torch.empty((0,))
                        cls_ids = torch.empty((0,), dtype=torch.int64)
                    
                    # Update mAP metric
                    map_metric.update(
                        [{'boxes': boxes, 'scores': scores, 'labels': cls_ids}],
                        [{'boxes': gt_boxes, 'labels': gt_labels}]
                    )
                    
                    # IoU-based matching for confusion matrix
                    if len(boxes) > 0 and len(gt_boxes) > 0:
                        ious = box_iou(boxes, gt_boxes)
                        matched = set()
                        for g in range(len(gt_boxes)):
                            best_iou, best_p = ious[:, g].max(0)
                            if best_iou > 0.5:
                                all_true.append(gt_labels[g].item())
                                all_pred.append(cls_ids[best_p].item())
                                matched.add(best_p.item())
                        for p in range(len(boxes)):
                            if p not in matched:
                                all_true.append(-1)  # False positive
                                all_pred.append(cls_ids[p].item())
                    elif len(boxes) > 0:
                        for c in cls_ids:
                            all_true.append(-1)
                            all_pred.append(c.item())
                    
                    # Save first batch visualization
                    if idx == 0 and b == 0:
                        save_detection_image(
                            imgs[b].cpu(), targets[b],
                            (boxes, scores, cls_ids),
                            os.path.join(args.output_dir, 'detections', phase, 
                                        f'epoch_{epoch}.jpg'),
                            class_names, args
                        )
        
        # Compute mAP
        res = map_metric.compute()
        val_metrics.update({
            'map': res['map'].item(),
            'map_50': res['map_50'].item(),
            'map_75': res['map_75'].item(),
            'mar_100': res['mar_100'].item()
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
    
    # Compute per-class metrics from confusion matrix
    valid = [i for i, l in enumerate(all_true) if l != -1]
    if valid:
        ft = [all_true[i] for i in valid]
        fp_l = [all_pred[i] for i in valid]
        cm = confusion_matrix(ft, fp_l, labels=list(class_names.keys()))
        val_metrics['confusion_matrix'] = cm.tolist()
        
        # Count false positives
        fp_counts = {}
        for i, l in enumerate(all_true):
            if l == -1:
                fp_counts[all_pred[i]] = fp_counts.get(all_pred[i], 0) + 1
        
        total_tp = total_fp = total_fn = 0
        for idx_c, cn in class_names.items():
            if idx_c < cm.shape[0]:
                tp = cm[idx_c, idx_c]
                fp_c = cm[:, idx_c].sum() - tp + fp_counts.get(idx_c, 0)
                fn = cm[idx_c, :].sum() - tp
                
                pr = tp / (tp + fp_c) if (tp + fp_c) > 0 else 0.
                rc = tp / (tp + fn) if (tp + fn) > 0 else 0.
                f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.
                
                val_metrics.update({
                    f'{cn}_precision': pr, f'{cn}_recall': rc, f'{cn}_f1': f1,
                    f'{cn}_tp': int(tp), f'{cn}_fp': int(fp_c), f'{cn}_fn': int(fn)
                })
                
                total_tp += tp
                total_fp += fp_c
                total_fn += fn
        
        op = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.
        ore = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.
        of1 = 2 * op * ore / (op + ore) if (op + ore) > 0 else 0.
        
        val_metrics.update({
            'overall_precision': op,
            'overall_recall': ore,
            'overall_f1': of1
        })
        
        if phase == 'val':
            plot_confusion_matrix(
                cm, list(class_names.values()),
                os.path.join(args.output_dir, f'cm_epoch_{epoch}.png')
            )
    
    return val_metrics


# ─────────────────────────────────────────────────────────────
#  Main Training Loop
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='PhytoNetEdge Training')
    parser.add_argument('--train_dir', default='Tomato_d/train')
    parser.add_argument('--val_dir', default='Tomato_d/valid')
    parser.add_argument('--test_dir', default='Tomato_d/test')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--model', default='edge', choices=['base', 'strong', 'edge'])
    parser.add_argument('--box_scale', type=float, default=1.0)
    parser.add_argument('--conf_thresh', type=float, default=0.25)
    parser.add_argument('--eval_conf_thresh', type=float, default=0.10)  # Increased from 0.05
    parser.add_argument('--iou_thresh', type=float, default=0.45)
    parser.add_argument('--output_dir', default='weights')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--accumulate', type=int, default=2)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()
    
    # Store anchor configs in args
    args.anchors_small = Config.ANCHORS_SMALL
    args.anchors_medium = Config.ANCHORS_MEDIUM
    args.anchors_large = Config.ANCHORS_LARGE
    args.anchors = Config.ANCHORS_ALL
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | LR={args.lr} batch={args.batch_size} "
          f"eval_conf={args.eval_conf_thresh}")
    
    # Create directories
    for d in [args.output_dir,
              f"{args.output_dir}/checkpoints",
              f"{args.output_dir}/detections/val",
              f"{args.output_dir}/detections/test"]:
        os.makedirs(d, exist_ok=True)
    
    class_names = {0: "stem", 1: "tomato"}
    
    if args.use_wandb:
        import wandb
        wandb.init(project='tomato-detection', config=vars(args))
    
    # ── Datasets ──
    train_ds = BotanicalDataset(
        args.train_dir, img_size=args.img_size, mode='train',
        transform=create_augmentations(args.img_size), use_mosaic=True
    )
    val_ds = BotanicalDataset(
        args.val_dir, img_size=args.img_size, mode='val',
        transform=create_val_transforms(args.img_size)
    )
    test_ds = BotanicalDataset(
        args.test_dir, img_size=args.img_size, mode='test',
        transform=create_val_transforms(args.img_size)
    )
    
    # Debug: class distribution
    label_counts = {0: 0, 1: 0}
    for i in range(min(200, len(train_ds))):
        _, t = train_ds[i]
        for l in t['labels']:
            label_counts[int(l)] += 1
    print(f"Class dist (first 200): stems={label_counts[0]} tomatoes={label_counts[1]}")
    
    # Weighted sampler - stems up-sampled
    sample_weights = []
    for img_id in train_ds.image_ids:
        anns = train_ds.image_annotations.get(img_id, [])
        n_stem = sum(1 for a in anns if a.get('category_id') == 1)
        n_tom = sum(1 for a in anns if a.get('category_id') == 0)
        # Weight images with stems higher
        sample_weights.append(1.0 + 8.0 * n_stem + 1.0 * n_tom)
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2
    )
    test_loader = DataLoader(
        test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2
    )
    
    # ── Model ──
    if args.model == 'edge':
        model = PhytoNetEdge(num_classes=2, num_anchors=3).to(device)
        is_three_head = True
    elif args.model == 'strong':
        model = HighAccuracyPhytoSparseNetStrong(num_classes=2).to(device)
        is_three_head = False
    else:
        model = HighAccuracyPhytoSparseNet(num_classes=2).to(device)
        is_three_head = False
    
    log_model_info(model, args.img_size, device, args.output_dir)
    
    # Verify model output shapes
    with torch.no_grad():
        _test = model(torch.randn(1, 3, args.img_size, args.img_size).to(device))
    if isinstance(_test, dict):
        print(f"Model heads: {{{', '.join(f'{k}: {tuple(v.shape)}' for k, v in _test.items())}}}")
    else:
        print(f"Model output shape: {_test.shape}")
    
    # ── Loss ──
    class_weights_tensor = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32, device=device)
    
    if is_three_head:
        # Use MultiHeadDetectionLoss wrapper for cleaner code
        multi_loss = MultiHeadDetectionLoss(
            anchors_small=Config.ANCHORS_SMALL,
            anchors_medium=Config.ANCHORS_MEDIUM,
            anchors_large=Config.ANCHORS_LARGE,
            num_classes=2,
            img_size=args.img_size,
            box_scale=args.box_scale,
            class_weights=class_weights_tensor,
            head_weights=Config.HEAD_WEIGHTS
        )
    else:
        loss_fn = DetectionLoss(
            anchors=Config.ANCHORS_ALL,
            alpha=0.25, gamma=2.0,
            lambda_box=8.0, lambda_obj=6.0, lambda_cls=1.5,
            class_weights=class_weights_tensor,
            num_classes=2, img_size=args.img_size,
            box_scale=args.box_scale, head_name='single'
        )
    
    # ── Optimizer ──
    base_lr = max(1e-6, min(args.lr, 1e-2))
    
    if is_three_head and hasattr(PhytoNetEdge, 'backbone_params'):
        # Differential LR: backbone gets 10× lower LR
        optimizer = optim.AdamW([
            {'params': PhytoNetEdge.backbone_params(model), 'lr': base_lr * 0.1},
            {'params': PhytoNetEdge.head_params(model), 'lr': base_lr * 4.0}
        ], weight_decay=1e-4, betas=(0.9, 0.999))
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=1e-4, betas=(0.9, 0.999)
        )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=[base_lr * 0.1, base_lr * 4.0],  # [backbone, heads]
                    epochs=args.epochs,
                    steps_per_epoch=len(train_loader),
                    pct_start=0.1,  # 10% warmup
                    anneal_strategy='cos'
                )
    
    amp_enabled = args.amp and torch.cuda.is_available()
    scaler = GradScaler(enabled=amp_enabled, init_scale=4096)
    
    # EMA model
    ema_model = deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    
    best_map50, best_epoch, patience_counter = 0., 0, 0
    train_loss_history, val_metrics_history = [], []
    start_epoch, WARMUP = 1, 1
    
    # ── Resume from checkpoint ──
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('epoch_') and f.endswith('.pth')])
    if ckpts:
        try:
            ck = torch.load(os.path.join(ckpt_dir, ckpts[-1]), map_location=device, weights_only=False)
            model.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            scheduler.last_epoch = ck['epoch'] * len(train_loader)
            start_epoch = ck['epoch'] + 1
            best_map50 = ck.get('best_map50', 0.)
            print(f"Resumed from epoch {ck['epoch']} best_mAP50={best_map50:.4f}")
        except Exception as e:
            print(f"Checkpoint load failed ({e}), starting fresh")
    
    # ─────────────────────────────────────────────────────────
    #  Training Loop
    # ─────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            train_ds.current_epoch = epoch
            
            # Warmup
            if epoch <= WARMUP:
                wf = 0.1 + 0.9 * epoch / WARMUP
                for i, pg in enumerate(optimizer.param_groups):
                    scale = 0.1 if (is_three_head and i == 0) else 1.0
                    pg['lr'] = base_lr * scale * wf
                print(f"Warmup LR = {base_lr * wf:.2e}")
            
            model.train()
            e_loss = e_obj = e_cls = e_box = 0.
            optimizer.zero_grad()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
            
            for bidx, (imgs, targets) in enumerate(pbar):
                imgs = imgs.to(device, non_blocking=True)
                dev_tgts = [{'boxes': t['boxes'].to(device),
                             'labels': t['labels'].to(device)} for t in targets]
                
                with autocast('cuda', enabled=amp_enabled):
                    outputs = model(imgs)
                    pred_dict = prepare_predictions_for_loss(outputs)
                    
                    if is_three_head and isinstance(outputs, dict) and 'small' in outputs:
                        # Build targets for each head
                        tgt_s = prepare_targets_for_loss(
                            dev_tgts, tuple(outputs['small'].shape),
                            img_size=args.img_size, anchors=Config.ANCHORS_SMALL,
                            num_classes=2, head_name='small'
                        )
                        tgt_m = prepare_targets_for_loss(
                            dev_tgts, tuple(outputs['medium'].shape),
                            img_size=args.img_size, anchors=Config.ANCHORS_MEDIUM,
                            num_classes=2, head_name='medium'
                        )
                        tgt_l = prepare_targets_for_loss(
                            dev_tgts, tuple(outputs['large'].shape),
                            img_size=args.img_size, anchors=Config.ANCHORS_LARGE,
                            num_classes=2, head_name='large'
                        )
                        
                        # Compute combined loss
                        loss, obj_loss, cls_loss, box_loss = multi_loss(
                            pred_dict,
                            {'small': tgt_s, 'medium': tgt_m, 'large': tgt_l}
                        )
                    else:
                        # Single head / legacy model
                        shape = (tuple(outputs.shape) if isinstance(outputs, torch.Tensor)
                                else tuple(next(iter(outputs.values())).shape))
                        tgt = prepare_targets_for_loss(
                            dev_tgts, shape, img_size=args.img_size,
                            anchors=Config.ANCHORS_ALL, num_classes=2, head_name='single'
                        )
                        loss, obj_loss, cls_loss, box_loss = loss_fn(pred_dict, tgt)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Bad loss at batch {bidx}, skipping")
                    optimizer.zero_grad()
                    if amp_enabled:
                        scaler.update()
                    continue
                
                scaled = loss / args.accumulate
                if amp_enabled:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                
                e_loss += loss.item()
                e_obj += obj_loss.item()
                e_cls += cls_loss.item()
                e_box += box_loss.item()
                
                if (bidx + 1) % args.accumulate == 0:
                    if amp_enabled:
                        scaler.unscale_(optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        optimizer.zero_grad()
                        if amp_enabled:
                            scaler.update()
                        continue
                    
                    if amp_enabled:
                        old_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        did_step = scaler.get_scale() >= old_scale
                    else:
                        optimizer.step()
                        did_step = True
                    
                    if did_step:
                        ema_decay = min(0.9999, (1 + epoch) / (10 + epoch))
                        with torch.no_grad():
                            for ep, mp in zip(ema_model.parameters(), model.parameters()):
                                ep.data.mul_(ema_decay).add_(mp.data, alpha=1 - ema_decay)
                    
                    optimizer.zero_grad()
                
                n = bidx + 1
                pbar.set_postfix({
                    'loss': f'{e_loss/n:.3f}',
                    'obj': f'{e_obj/n:.3f}',
                    'cls': f'{e_cls/n:.3f}',
                    'box': f'{e_box/n:.3f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'
                })
            
            scheduler.step()
            
            nb = len(train_loader)
            avg = e_loss / nb
            train_loss_history.append(avg)
            print(f"\nEpoch {epoch} ({time.time()-t0:.0f}s) | "
                  f"loss={avg:.4f} obj={e_obj/nb:.4f} cls={e_cls/nb:.4f} box={e_box/nb:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")
            
            # Validation
            val_m = validate_model(model, val_loader, device, class_names, args, epoch, 'val')
            val_metrics_history.append(val_m)
            print(f"  mAP={val_m['map']:.4f} mAP@50={val_m['map_50']:.4f} "
                  f"stem_R={val_m.get('stem_recall', 0.):.3f} "
                  f"tom_R={val_m.get('tomato_recall', 0.):.3f} "
                  f"F1={val_m['overall_f1']:.4f}")
            
            if args.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch, 'train/loss': avg,
                    'val/map': val_m['map'], 'val/map_50': val_m['map_50'],
                    'val/stem_recall': val_m.get('stem_recall', 0.),
                    'val/tomato_recall': val_m.get('tomato_recall', 0.),
                    'lr': optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_m['map_50'] > best_map50:
                best_map50, best_epoch, patience_counter = val_m['map_50'], epoch, 0
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                print(f"  ✅ New best mAP@50={best_map50:.4f}")
            else:
                patience_counter += 1
            
            # Save checkpoint
            if epoch % 5 == 0 or epoch == args.epochs:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_map50': best_map50
                }, os.path.join(ckpt_dir, f'epoch_{epoch}.pth'))
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            torch.cuda.empty_cache()
            gc.collect()
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        if args.use_wandb:
            import wandb
            wandb.alert(title="Training Failed", text=str(e))
    
    # ── Test ──
    bp = os.path.join(args.output_dir, 'best_model.pth')
    if os.path.exists(bp):
        model.load_state_dict(torch.load(bp, map_location=device, weights_only=False))
    
    test_m = validate_model(model, test_loader, device, class_names, args, best_epoch, 'test')
    print(f"\nTest | mAP={test_m['map']:.4f} mAP@50={test_m['map_50']:.4f} "
          f"P={test_m['overall_precision']:.4f} R={test_m['overall_recall']:.4f}")
    
    # Save results
    plot_training_curves(train_loss_history, val_metrics_history, args.output_dir)
    
    with open(os.path.join(args.output_dir, 'training_metrics.json'), 'w') as f:
        json.dump({
            'train_loss': train_loss_history,
            'val_metrics': val_metrics_history,
            'test_metrics': test_m,
            'best_epoch': best_epoch,
            'best_map50': best_map50
        }, f, indent=2, cls=NumpyEncoder)
    
    # Quantization
    try:
        qm = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        torch.save(qm.state_dict(), os.path.join(args.output_dir, 'quantized_model.pth'))
        print("Quantized model saved")
    except Exception as e:
        print(f"Quantization failed: {e}")
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
