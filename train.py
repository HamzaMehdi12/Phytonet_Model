import os
import time
import argparse
import random
from PIL import Image
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
import cv2
import math
import gc
import seaborn as sns
import wandb
import shutil
import albumentations as A

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision.ops import nms
from phytonet import HighAccuracyPhytoSparseNet, HighAccuracyPhytoSparseNetStrong
from botanical_loss import DetectionLoss
from dataset import BotanicalDataset
from torchmetrics.detection import MeanAveragePrecision
from thop import profile
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def prepare_targets_for_loss(raw_targets, model_output_shape, img_size=224, 
                            anchors=[[11, 8], [17, 10], [23, 15], [29, 16], [35, 21],
                                     [65, 24], [49, 60], [95, 50], [137, 71]],
                            num_classes=2):
    """
    Target assignment with class-aware IoU thresholds.
    Stems are small → need lower threshold to get positive samples.
    """
    device = raw_targets[0]['boxes'].device
    batch_size = len(raw_targets)
    
    if isinstance(model_output_shape, torch.Size):
        _, C, H, W = model_output_shape
    else:
        _, C, H, W = model_output_shape
    
    anchors_t = torch.tensor(anchors, dtype=torch.float32, device=device)
    A = anchors_t.shape[0]

    # Use the actual img_size (current_size) for anchor normalization and grid assignment
    grid_img_size = img_size

    target_obj = torch.zeros(batch_size, A * H * W, device=device)
    target_cls = torch.zeros(batch_size, A * H * W, num_classes, device=device)
    target_boxes = torch.zeros(batch_size, A * H * W, 4, device=device)
    
    total_positives = 0
    stem_positives = 0
    tomato_positives = 0
    
    for b in range(batch_size):
        gt_boxes = raw_targets[b]['boxes']
        gt_labels = raw_targets[b]['labels']
        if len(gt_boxes) == 0:
            continue
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
        gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        grid_x = (gt_cx * W).long().clamp(0, W-1)
        grid_y = (gt_cy * H).long().clamp(0, H-1)
        anchors_norm = anchors_t / float(grid_img_size)
        for i, (gx, gy, gw, gh, label) in enumerate(zip(grid_x, grid_y, gt_w, gt_h, gt_labels)):
            anchor_ious = []
            for anchor in anchors_norm:
                aw, ah = anchor[0], anchor[1]
                inter_w = torch.min(gw, aw)
                inter_h = torch.min(gh, ah)
                inter = inter_w * inter_h
                union = gw * gh + aw * ah - inter
                iou = inter / (union + 1e-6)
                anchor_ious.append(iou.item())
            
            anchor_ious = torch.tensor(anchor_ious, device=device)
            
            label_idx = int(label.item())
            
            # Best anchor is ALWAYS assigned regardless of IoU
            best_anchor = anchor_ious.argmax()
            
            # Class-specific anchor assignment to reduce FP flood
            if label_idx == 0:  # stem - controlled positives
                top_k = 2
                iou_thresh = 0.25
            else:  # tomato - stricter to cut FP flood
                top_k = 1
                iou_thresh = 0.40
            
            # Get top-k anchors (includes best)
            _, top_anchors = torch.topk(anchor_ious, min(top_k, len(anchor_ious)))
            
            # Use top-k anchors that pass IoU threshold
            matching_anchors = [a for a in top_anchors if anchor_ious[a] >= iou_thresh]
            
            # Always include best anchor even if it fails threshold
            if best_anchor not in matching_anchors:
                matching_anchors = list(matching_anchors) + [best_anchor]
            
            # NO SPATIAL NEIGHBORS - assign only to center cell
            spatial_cells = [(int(gy), int(gx))]  # Center cell ONLY
            
            for anchor_idx in matching_anchors:
                anchor_idx = int(anchor_idx.item())
                label_idx_clamped = max(0, min(label_idx, num_classes - 1))
                
                for sy, sx in spatial_cells:
                    idx_center = anchor_idx * H * W + int(sy) * W + int(sx)
                    target_obj[b, idx_center] = 1.0
                    target_cls[b, idx_center, label_idx_clamped] = 1.0
                    target_boxes[b, idx_center] = gt_boxes[i]
                    total_positives += 1
                
                    if label_idx == 0:
                        stem_positives += 1
                    else:
                        tomato_positives += 1
    
    return {
        'obj': target_obj,
        'cls': target_cls,
        'boxes': target_boxes
    }

def prepare_predictions_for_loss(model_output, num_classes=2):
    """
    Convert model output to format expected by DetectionLoss.
    
    Handles three cases:
    1. Dict with 'pred_boxes', 'pred_cls', 'pred_obj' -> return as-is
    2. Dict with 'large', 'medium' -> extract 'large' tensor
    3. Tensor [B, C, H, W] -> convert to prediction dict
    """
    # Case 1: Already in correct prediction format
    if isinstance(model_output, dict):
        required_keys = {'pred_boxes', 'pred_cls', 'pred_obj'}
        if required_keys.issubset(model_output.keys()):
            return model_output
        # Case 2: Multi-scale output dict (both heads)
        if 'large' in model_output and 'medium' in model_output:
            # Return dict of both heads
            return {
                'large': prepare_predictions_for_loss(model_output['large'], num_classes),
                'medium': prepare_predictions_for_loss(model_output['medium'], num_classes)
            }
        # If only one head present, fallback to tensor logic below
        if 'large' in model_output:
            model_output = model_output['large']
        elif 'medium' in model_output:
            model_output = model_output['medium']
        else:
            raise TypeError(f"Dict with unexpected keys: {model_output.keys()}")
    
    # Case 3: Tensor format - convert to dict
    if not isinstance(model_output, torch.Tensor):
        raise TypeError(f"Expected Tensor or dict, got {type(model_output)}")
    
    B, C, H, W = model_output.shape
    
    # Calculate number of anchors
    values_per_anchor = 5 + num_classes
    A = C // values_per_anchor
    
    if C % values_per_anchor != 0:
        raise ValueError(f"Channel dimension {C} is not divisible by {values_per_anchor}")
    
    # Reshape: [B, A*(5+C), H, W] -> [B, A, 5+C, H, W]
    pred = model_output.view(B, A, values_per_anchor, H, W)
    
    # Permute to: [B, A, H, W, 5+C]
    pred = pred.permute(0, 1, 3, 4, 2).contiguous()
    
    # Flatten spatial and anchor dimensions: [B, A*H*W, 5+C]
    pred = pred.view(B, A * H * W, values_per_anchor)
    
    # Extract components
    pred_boxes = pred[..., :4]  # [B, A*H*W, 4]
    pred_obj = pred[..., 4]     # [B, A*H*W]
    pred_cls = pred[..., 5:]    # [B, A*H*W, num_classes]
    
    return {
        'pred_boxes': pred_boxes,
        'pred_obj': pred_obj,
        'pred_cls': pred_cls
    }

def convert_dict_to_tensor(pred_dict, num_classes=2, H=7, W=7):
    """
    Convert prediction dict back to tensor format for decode_predictions_advanced.
    
    Args:
        pred_dict: Dict with 'pred_boxes', 'pred_cls', 'pred_obj'
        num_classes: Number of classes
        H, W: Grid height and width
    
    Returns:
        Tensor of shape [B, C, H, W] where C = A*(5+num_classes)
    """
    pred_boxes = pred_dict['pred_boxes']  # [B, A*H*W, 4]
    pred_obj = pred_dict['pred_obj'].unsqueeze(-1)  # [B, A*H*W, 1]
    pred_cls = pred_dict['pred_cls']  # [B, A*H*W, num_classes]
    
    B = pred_boxes.shape[0]
    A = pred_boxes.shape[1] // (H * W)
    
    # Concatenate: [B, A*H*W, 5+num_classes]
    pred = torch.cat([pred_boxes, pred_obj, pred_cls], dim=-1)
    
    # Reshape to [B, A, H, W, 5+num_classes]
    pred = pred.view(B, A, H, W, 5 + num_classes)
    
    # Permute to [B, A, 5+num_classes, H, W]
    pred = pred.permute(0, 1, 4, 2, 3).contiguous()
    
    # Reshape to [B, C, H, W]
    pred = pred.reshape(B, A * (5 + num_classes), H, W)
    
    return pred

def setup_wandb(args):
    """Initialize Weights & Biases for experiment tracking"""
    wandb.init(
        project="tomato-detection",
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "img_size": args.img_size,
            "architecture": "HighAccuracyPhytoSparseNet",
            "loss": "DetectionLoss"
        }
    )
    return wandb

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)

def create_diverse_augmentations(img_size=224):
    """Minimal augmentations: horizontal flip, resize, normalize, tensor."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def create_final_summary(model_info, train_loss_history, val_metrics_history, test_metrics, output_dir):
    """Create a comprehensive training summary"""
    summary = {
        'model_info': model_info,
        'training_summary': {
            'final_train_loss': train_loss_history[-1] if train_loss_history else 'N/A',
            'best_val_map': max([m.get('map', 0) for m in val_metrics_history]) if val_metrics_history else 'N/A',
            'best_val_map50': max([m.get('map_50', 0) for m in val_metrics_history]) if val_metrics_history else 'N/A',
            'test_map': test_metrics.get('map', 'N/A'),
            'test_map50': test_metrics.get('map_50', 'N/A'),
            'test_precision': test_metrics.get('overall_precision', 'N/A'),
            'test_recall': test_metrics.get('overall_recall', 'N/A'),
            'test_f1': test_metrics.get('overall_f1', 'N/A'),
        },
        'training_curves': {
            'train_loss': train_loss_history,
            'val_map': [m.get('map', 0) for m in val_metrics_history] if val_metrics_history else [],
            'val_map50': [m.get('map_50', 0) for m in val_metrics_history] if val_metrics_history else [],
        }
    }
    
    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    print(f"Training summary saved to {summary_path}")
    return summary

def decode_predictions_advanced(pred, conf_thresh=0.35, iou_thresh=0.45,
                                anchors=[[11, 8], [17, 10], [23, 15], [29, 16], [35, 21],
                                         [65, 24], [49, 60], [95, 50], [137, 71]],
                                img_size=224, max_detections=300, use_class_thresholds=True, box_scale=1.0,
                                apply_nms=True):
    """Decode network output to normalized boxes [0..1], scores and class ids.
    
    CRITICAL: Model outputs RAW logits (tx, ty, tw, th, to_logit, cls_logits)
    We must apply sigmoid to tx, ty, to and sigmoid to cls (BCE loss)
    """
    device = pred.device
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    A = anchors.shape[0]

    C, H, W = pred.shape
    
    # Try to infer number of classes
    if (C % A) == 0:
        num_classes = (C // A) - 5
    else:
        print(f"Warning: Cannot perfectly divide channels {C} by anchors {A}")
        return (torch.empty((0, 4), device=device), 
                torch.empty((0,), device=device), 
                torch.empty((0,), dtype=torch.int64, device=device))
    
    if num_classes < 1:
        print(f"Invalid num_classes={num_classes}, returning empty predictions")
        return (torch.empty((0, 4), device=device), 
                torch.empty((0,), device=device), 
                torch.empty((0,), dtype=torch.int64, device=device))

    # Reshape: [C, H, W] -> [A, (5+classes), H, W] -> [A, H, W, (5+classes)]
    pred = pred.view(A, 5 + num_classes, H, W).permute(0, 2, 3, 1).contiguous()

    # Create grid coordinates
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid_x = grid_x.view(1, H, W, 1).expand(A, H, W, 1).float()
    grid_y = grid_y.view(1, H, W, 1).expand(A, H, W, 1).float()

    # Extract raw predictions
    tx = pred[..., 0:1]  # center x offset (logit)
    ty = pred[..., 1:2]  # center y offset (logit)
    tw = pred[..., 2:3]  # width (log scale)
    th = pred[..., 3:4]  # height (log scale)
    to = pred[..., 4:5]  # objectness (logit)
    tcls = pred[..., 5:5+num_classes]  # class logits

    # Decode center positions (apply sigmoid, then normalize by grid)
    cx = (torch.sigmoid(tx) + grid_x) / W
    cy = (torch.sigmoid(ty) + grid_y) / H

    # Normalize anchors to [0,1] scale
    anchors_norm = anchors / float(img_size)
    aw = anchors_norm[:, 0].view(A, 1, 1, 1)
    ah = anchors_norm[:, 1].view(A, 1, 1, 1)

    # Decode width/height (apply exp with clamping, scale by anchor size)
    tw_clamped = tw.clamp(min=-10.0, max=10.0)
    th_clamped = th.clamp(min=-10.0, max=10.0)

    # CRITICAL: Match scale factor in loss function
    bw = torch.exp(tw_clamped) * aw * box_scale  # MUST match botanical_loss.py!
    bh = torch.exp(th_clamped) * ah * box_scale

    # Convert center + size to corners [x1, y1, x2, y2]
    x1 = (cx - bw / 2.0).reshape(-1)
    y1 = (cy - bh / 2.0).reshape(-1)
    x2 = (cx + bw / 2.0).reshape(-1)
    y2 = (cy + bh / 2.0).reshape(-1)

    boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)

    # Get objectness and class probabilities (BCE loss expects sigmoid per class)

    obj_prob = torch.sigmoid(to).reshape(-1)
    cls_prob = torch.sigmoid(tcls).reshape(-1, num_classes)
    cls_scores, cls_ids = cls_prob.max(dim=-1)

    # Combined confidence score (balanced: improves early recall)
    scores = torch.sqrt(obj_prob * cls_scores)
    
    # CRITICAL FIX: Use LOWER confidence threshold for initial filtering
    # This ensures we don't filter out true positives too early
    class_ids = cls_ids.reshape(-1)
    
    if use_class_thresholds:
        # Class-specific thresholds (precision-focused)
        # Stems: moderate threshold to avoid FP flood while keeping recall
        # Tomatoes: stricter threshold to suppress tomato FP flood
        class_thresholds = {
            0: conf_thresh * 0.25,  # stem - 0.09 @ conf=0.35
            1: conf_thresh * 1.2,   # tomato - 0.42 @ conf=0.35
        }
        
        # Apply class-specific thresholds
        adjusted_thresh = torch.tensor([
            class_thresholds.get(int(cls_id.item()), conf_thresh) 
            for cls_id in class_ids
        ], device=device)
        keep_mask = scores > adjusted_thresh
    else:
        # Uniform threshold for evaluation (preserve recall for mAP)
        keep_mask = scores > conf_thresh
    
    if keep_mask.sum() == 0:
        return (torch.empty((0, 4), device=device), 
                torch.empty((0,), device=device), 
                torch.empty((0,), dtype=torch.int64, device=device))

    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]

    if apply_nms:
        # Convert to pixel coordinates for NMS
        abs_boxes = boxes * img_size

        # Class-specific NMS
        final_boxes = []
        final_scores = []
        final_classes = []

        unique_classes = class_ids.unique()
        for c in unique_classes:
            cls_mask = (class_ids == c)
            cls_boxes = abs_boxes[cls_mask]
            cls_scores = scores[cls_mask]
            
            if cls_boxes.numel() == 0:
                continue
            
            # Use provided iou_thresh for all classes
            keep = nms(cls_boxes, cls_scores, iou_thresh)
            keep = keep[:max_detections]
            
            final_boxes.append(cls_boxes[keep])
            final_scores.append(cls_scores[keep])
            final_classes.append(torch.full((len(keep),), int(c.item()), 
                                           dtype=torch.int64, device=device))
    else:
        # No NMS: keep top-k by score
        if scores.numel() > max_detections:
            topk = torch.topk(scores, max_detections)
            keep = topk.indices
            boxes = boxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

        final_boxes = [boxes * img_size]
        final_scores = [scores]
        final_classes = [class_ids]

    if len(final_boxes) == 0:
        return (torch.empty((0, 4), device=device), 
                torch.empty((0,), device=device), 
                torch.empty((0,), dtype=torch.int64, device=device))

    # Concatenate all classes
    final_boxes = torch.cat(final_boxes, dim=0)
    final_scores = torch.cat(final_scores, dim=0)
    final_classes = torch.cat(final_classes, dim=0)

    # Convert back to normalized coordinates
    final_boxes = final_boxes / float(img_size)

    # Final check for valid boxes
    valid_mask = (final_boxes[:, 2] > final_boxes[:, 0]) & (final_boxes[:, 3] > final_boxes[:, 1])
    final_boxes = final_boxes[valid_mask]
    final_scores = final_scores[valid_mask]
    final_classes = final_classes[valid_mask]

    return final_boxes, final_scores, final_classes

def draw_dashed_rectangle(img, pt1, pt2, color, thickness, dash_length=10):
    """Draw a dashed rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    for x in range(x1, x2, dash_length * 2):
        end_x = min(x + dash_length, x2)
        cv2.line(img, (x, y1), (end_x, y1), color, thickness)
    
    for x in range(x1, x2, dash_length * 2):
        end_x = min(x + dash_length, x2)
        cv2.line(img, (x, y2), (end_x, y2), color, thickness)
    
    for y in range(y1, y2, dash_length * 2):
        end_y = min(y + dash_length, y2)
        cv2.line(img, (x1, y), (x1, end_y), color, thickness)
    
    for y in range(y1, y2, dash_length * 2):
        end_y = min(y + dash_length, y2)
        cv2.line(img, (x2, y), (x2, end_y), color, thickness)


def save_detection_image(image_tensor, target, predictions, output_path, class_names,
                         conf_thresh=0.35, img_size=224):
    try:
        if isinstance(image_tensor, torch.Tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_np = image_tensor.cpu() * std + mean
            img_np = img_np.clamp(0, 1).numpy().transpose(1, 2, 0) * 255
            img_np = img_np.astype(np.uint8)
        else:
            img_np = image_tensor

        img_draw = img_np.copy()
        height, width = img_draw.shape[:2]

        pred_boxes, pred_scores, pred_classes = predictions

        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
        if isinstance(pred_scores, torch.Tensor):
            pred_scores = pred_scores.cpu().numpy()
        if isinstance(pred_classes, torch.Tensor):
            pred_classes = pred_classes.cpu().numpy()

        if len(pred_boxes) > 0:
            for i in range(len(pred_boxes)):
                if len(pred_boxes[i]) != 4:
                    continue
                score = float(pred_scores[i])
                if score < conf_thresh:
                    continue

                bx0 = float(pred_boxes[i][0])
                by0 = float(pred_boxes[i][1])
                bx1 = float(pred_boxes[i][2])
                by1 = float(pred_boxes[i][3])

                x1 = int(max(0, bx0 * width))
                y1 = int(max(0, by0 * height))
                x2 = int(min(width - 1, bx1 * width))
                y2 = int(min(height - 1, by1 * height))

                if x2 <= x1 or y2 <= y1:
                    continue

                cls = int(pred_classes[i])
                color = (0, 255, 0)

                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

                cls_name = class_names.get(cls, f"Class {cls}")
                label = f"{cls_name}: {score:.2f}"

                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img_draw, (x1, y1 - text_height - 6), (x1 + text_width, y1), color, -1)
                cv2.putText(img_draw, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if target is not None and 'boxes' in target and 'labels' in target:
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
            if isinstance(gt_labels, torch.Tensor):
                gt_labels = gt_labels.cpu().numpy()

            for i in range(len(gt_boxes)):
                if len(gt_boxes[i]) != 4:
                    continue
                bx0, by0, bx1, by1 = gt_boxes[i]
                x1 = int(max(0, bx0 * width))
                y1 = int(max(0, by0 * height))
                x2 = int(min(width - 1, bx1 * width))
                y2 = int(min(height - 1, by1 * height))

                if x2 <= x1 or y2 <= y1:
                    continue

                color = (0, 0, 255)
                draw_dashed_rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

                cls_name = class_names.get(int(gt_labels[i]), f"Class {int(gt_labels[i])}")
                label = f"GT: {cls_name}"
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img_draw, (x1, y2), (x1 + text_width, y2 + text_height + 6), color, -1)
                cv2.putText(img_draw, label, (x1, y2 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imwrite(output_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
        print(f"Saved detection image to {output_path}")

    except Exception as e:
        print(f"Error in save_detection_image: {e}")
        blank_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        cv2.imwrite(output_path, blank_img)


def plot_training_curves(train_loss_history, val_metrics_history, output_dir):
    """Plot training loss and validation metrics"""
    epochs = range(1, len(train_loss_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_history, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()
    
    if val_metrics_history and len(val_metrics_history) > 0:
        map_scores = [m.get('map', 0) for m in val_metrics_history]
        map50_scores = [m.get('map_50', 0) for m in val_metrics_history]
        map75_scores = [m.get('map_75', 0) for m in val_metrics_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, map_scores, 'r-', label='mAP')
        plt.plot(epochs, map50_scores, 'g-', label='mAP@50')
        plt.plot(epochs, map75_scores, 'b-', label='mAP@75')
        plt.title('Validation Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
        plt.savefig(os.path.join(output_dir, 'validation_metrics.png'))
        plt.close()
        
        if 'overall_precision' in val_metrics_history[0]:
            precision_scores = [m.get('overall_precision', 0) for m in val_metrics_history]
            recall_scores = [m.get('overall_recall', 0) for m in val_metrics_history]
            f1_scores = [m.get('overall_f1', 0) for m in val_metrics_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, precision_scores, 'r-', label='Precision')
            plt.plot(epochs, recall_scores, 'g-', label='Recall')
            plt.plot(epochs, f1_scores, 'b-', label='F1 Score')
            plt.title('Validation PRF Metrics')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(output_dir, 'validation_prf_metrics.png'))
            plt.close()

def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def validate_model(model, dataloader, device, class_names, args, epoch, phase='val'):
    """Comprehensive validation function with error handling"""
    model.eval()
    
    val_metrics = {
        'map': 0.0,
        'map_50': 0.0,
        'map_75': 0.0,
        'mar_100': 0.0,
        'overall_precision': 0.0,
        'overall_recall': 0.0,
        'overall_f1': 0.0,
    }
    
    for cls_name in class_names.values():
        val_metrics[f'{cls_name}_precision'] = 0.0
        val_metrics[f'{cls_name}_recall'] = 0.0
        val_metrics[f'{cls_name}_f1'] = 0.0
        val_metrics[f'{cls_name}_tp'] = 0
        val_metrics[f'{cls_name}_fp'] = 0
        val_metrics[f'{cls_name}_fn'] = 0
    
    all_true_labels = []
    all_pred_labels = []
    
    try:
        from torchmetrics.detection import MeanAveragePrecision
        # Use a uniform IoU threshold for mAP@50
        map_metric = MeanAveragePrecision(
            class_metrics=True,
            iou_thresholds=[0.5]
        )
        # Suppress warning about >100 detections
        map_metric.warn_on_many_detections = False
        
        with torch.no_grad():
            for idx, (imgs, targets) in enumerate(dataloader):
                imgs = imgs.to(device)
                # Get model output for the whole batch
                model_output = model(imgs)
                # Handle multi-scale dict output
                if isinstance(model_output, dict):
                    if 'large' in model_output:
                        output_tensor = model_output['large']
                    elif 'pred_boxes' in model_output:
                        output_tensor = convert_dict_to_tensor(
                            model_output,
                            num_classes=2,
                            H=7,
                            W=7
                        )
                    else:
                        print(f"Unexpected dict keys: {model_output.keys()}")
                        continue
                elif isinstance(model_output, torch.Tensor):
                    output_tensor = model_output
                else:
                    print(f"Unexpected model output type: {type(model_output)}")
                    continue

                # Loop over all images in the batch
                batch_size = imgs.shape[0]
                for b in range(batch_size):
                    if output_tensor.dim() == 4:
                        single_output = output_tensor[b]
                    else:
                        single_output = output_tensor
                    # Decode predictions for this image
                    boxes, scores, class_ids = decode_predictions_advanced(
                        single_output,
                        conf_thresh=args.eval_conf_thresh,
                        iou_thresh=args.iou_thresh,
                        anchors=args.anchors,
                        img_size=args.img_size,
                        max_detections=300,  # Set to 300 for evaluation
                        use_class_thresholds=False,
                        box_scale=args.box_scale
                    )
                    boxes = boxes.cpu()
                    scores = scores.cpu()
                    class_ids = class_ids.cpu()
                    gt_boxes = targets[b]['boxes'].cpu()
                    gt_labels = targets[b]['labels'].cpu()

                if len(boxes) == 0:
                    boxes = torch.empty((0, 4))
                    scores = torch.empty((0,))
                    class_ids = torch.empty((0,), dtype=torch.int64)
                
                preds = [{
                    "boxes": boxes, 
                    "scores": scores, 
                    "labels": class_ids
                }]
                
                targets_dict = [{
                    "boxes": gt_boxes, 
                    "labels": gt_labels
                }]
                
                try:
                    map_metric.update(preds, targets_dict)
                except Exception as e:
                    print(f"Error updating metrics: {e}")
                    continue
                
                # FIXED: Match predictions to GT boxes via IoU, not positionally
                # This computes per-class TP/FP/FN for confusion matrix
                if len(boxes) > 0 and len(gt_boxes) > 0:
                    # Compute IoU between all pred and GT boxes
                    from torchvision.ops import box_iou
                    ious = box_iou(boxes, gt_boxes)  # [num_preds, num_gts]
                    
                    # Match each GT to best pred (if IoU > 0.5)
                    matched_preds = set()
                    for gt_idx in range(len(gt_boxes)):
                        gt_label = gt_labels[gt_idx].item()
                        best_iou, best_pred_idx = ious[:, gt_idx].max(dim=0)
                        
                        if best_iou > 0.5:
                            pred_label = class_ids[best_pred_idx].item()
                            all_true_labels.append(gt_label)
                            all_pred_labels.append(pred_label)
                            matched_preds.add(best_pred_idx.item())
                    
                    # Unmatched predictions = false positives (add to lists)
                    for pred_idx in range(len(boxes)):
                        if pred_idx not in matched_preds:
                            # FP - predict class but no matching GT
                            pred_label = class_ids[pred_idx].item()
                            all_pred_labels.append(pred_label)
                            all_true_labels.append(-1)  # No GT match
                
                elif len(gt_boxes) == 0 and len(boxes) > 0:
                    # All predictions are FP
                    for pred_label in class_ids:
                        all_pred_labels.append(pred_label.item())
                        all_true_labels.append(-1)
                
                # Save detection image for first batch
                if idx == 0 and phase == 'val':
                    save_detection_image(
                        imgs[0].cpu(), 
                        {
                            'boxes': gt_boxes,
                            'labels': gt_labels,
                            'image_path': targets[0].get('image_path', '')
                        },
                        (boxes, scores, class_ids),
                        os.path.join(args.output_dir, 'detections', phase, f'epoch_{epoch}.jpg'),
                        class_names,
                        conf_thresh=args.eval_conf_thresh,
                        img_size=args.img_size
                    )
        
        try:
            map_result = map_metric.compute()
            
            val_metrics.update({
                'map': map_result['map'].item(),
                'map_50': map_result['map_50'].item(),
                'map_75': map_result['map_75'].item(),
                'mar_100': map_result['mar_100'].item(),
            })
        except Exception as e:
            print(f"Error computing metrics: {e}")
        
        # Confusion matrix calculation
        try:
            from sklearn.metrics import confusion_matrix
            if len(all_true_labels) > 0 and len(all_pred_labels) > 0:
                # Filter out FP entries (where true_label = -1)
                # We'll count FP separately, not in confusion matrix
                valid_indices = [i for i, label in enumerate(all_true_labels) if label != -1]
                
                if len(valid_indices) > 0:
                    filtered_true = [all_true_labels[i] for i in valid_indices]
                    filtered_pred = [all_pred_labels[i] for i in valid_indices]
                    
                    cm = confusion_matrix(filtered_true, filtered_pred, labels=list(class_names.keys()))
                    val_metrics['confusion_matrix'] = cm.tolist()
                    
                    # Count FPs (unmatched predictions)
                    fp_counts = {}
                    for i, label in enumerate(all_true_labels):
                        if label == -1:  # False positive
                            pred_class = all_pred_labels[i]
                            fp_counts[pred_class] = fp_counts.get(pred_class, 0) + 1
                    
                    for i, cls_name in class_names.items():
                        if i < cm.shape[0] and i < cm.shape[1]:
                            tp = cm[i, i]
                            fp_from_cm = cm[:, i].sum() - tp  # Misclassifications
                            fp_unmatched = fp_counts.get(i, 0)  # Unmatched predictions
                            fp = fp_from_cm + fp_unmatched  # Total FP
                            fn = cm[i, :].sum() - tp
                            
                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                            
                            val_metrics[f'{cls_name}_precision'] = precision
                            val_metrics[f'{cls_name}_recall'] = recall
                            val_metrics[f'{cls_name}_f1'] = f1
                            val_metrics[f'{cls_name}_tp'] = int(tp)
                            val_metrics[f'{cls_name}_fp'] = int(fp)
                            val_metrics[f'{cls_name}_fn'] = int(fn)
                    
                    total_tp = sum([val_metrics[f'{cls_name}_tp'] for cls_name in class_names.values()])
                    total_fp = sum([val_metrics[f'{cls_name}_fp'] for cls_name in class_names.values()])
                    total_fn = sum([val_metrics[f'{cls_name}_fn'] for cls_name in class_names.values()])
                    
                    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
                    
                    val_metrics['overall_precision'] = overall_precision
                    val_metrics['overall_recall'] = overall_recall
                    val_metrics['overall_f1'] = overall_f1
                    
                    if phase == 'val':
                        plot_confusion_matrix(
                            cm, 
                            list(class_names.values()), 
                            os.path.join(args.output_dir, f'confusion_matrix_epoch_{epoch}.png')
                        )
        except Exception as e:
            print(f"Error calculating confusion matrix: {e}")
        
    except Exception as e:
        print(f"Error in validation: {e}")
        import traceback
        traceback.print_exc()
    
    return val_metrics

def log_model_info(model, input_size, device, output_dir):
    """Calculate and log model information"""
    try:
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        input_tensor = torch.randn(1, 3, input_size, input_size).to(device)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'model_size_mb': size_mb,
            'flops': flops,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_size': input_size,
            'device': str(device)
        }
        
        model_info_path = os.path.join(output_dir, 'model_info.json')
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model information saved to {model_info_path}")
        print(f"Model Size: {size_mb:.2f} MB")
        print(f"FLOPs: {flops:,}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
        return model_info
    except Exception as e:
        print(f"Error calculating model info: {e}")
        return None

def stabilize_gradients(model, max_norm=1.0, debug=False):
    """Enhanced gradient stabilization with detailed debugging"""
    total_norm = 0
    has_nan_inf = False
    extreme_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                has_nan_inf = True
                if debug:
                    print(f"NaN/Inf gradient in {name}, zeroing")
                param.grad.data = torch.zeros_like(param.grad.data)
                continue
                
            grad_max = grad.abs().max().item()
            if grad_max > 1e3:
                extreme_grad_count += 1
                if debug and extreme_grad_count < 5:
                    print(f"Extreme gradient in {name}: {grad_max:.6f}")
                scaling_factor = min(1.0, 1e3 / grad_max)
                param.grad.data.mul_(scaling_factor)
    
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().max().item() < 1e6:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** (0.5) if total_norm > 0 else 0
    
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-10)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    
    if debug and extreme_grad_count > 0:
        print(f"Found {extreme_grad_count} parameters with extreme gradients")
    
    return has_nan_inf, total_norm


def create_optimizer_and_scheduler(model, args):
    """Create optimizer with proper learning rate and warm-up cosine annealing
    
    CRITICAL FIX: User was using 1e-5 which is WAY TOO LOW for AdamW.
    For AdamW on object detection, typical LR range is 1e-4 to 5e-4.
    """
    # CRITICAL: Don't scale LR by batch size for AdamW - it's already adaptive
    # AdamW handles per-parameter adaptive learning rates
    base_lr = args.lr
    
    # Clamp to reasonable range to prevent user mistakes
    if base_lr < 1e-6:
        print(f"\n{'='*60}")
        print(f"WARNING: LR {base_lr:.2e} is TOO LOW!")
        print(f"For AdamW optimizer, recommended range is 1e-4 to 5e-4")
        print(f"Using minimum safe LR of 1e-4")
        print(f"{'='*60}\n")
        base_lr = 1e-4
    elif base_lr > 1e-2:
        print(f"\n{'='*60}")
        print(f"WARNING: LR {base_lr:.2e} is TOO HIGH!")
        print(f"Clamping to 5e-3 for stability")
        print(f"{'='*60}\n")
        base_lr = 5e-3

    print(f"\nOptimizer Configuration:")
    print(f"  Base LR: {base_lr:.2e}")
    print(f"  Weight Decay: 1e-4 (reduced for better generalization)")
    print(f"  Betas: (0.9, 0.999)")
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=base_lr, 
        weight_decay=1e-4,  # Reduced from 1e-3 for better generalization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    # Cosine annealing with warm restarts
    # FIXED: Use longer cycle and higher eta_min to prevent premature LR decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=100,              # Full cycle over 100 epochs (was 40, too short)
        eta_min=base_lr * 0.2   # Min LR is 20% of base (was 10%, too low)
    )
    
    return optimizer, scheduler

def calculate_class_weights(dataset):
    """Calculate class weights with smoothing"""
    class_counts = [1, 1]
    
    for i in range(len(dataset)):
        _, target = dataset[i]
        labels = target['labels']
        for label in labels:
            if label < len(class_counts):
                class_counts[label] += 1
    
    total = sum(class_counts)
    class_weights = [total / (count + 1) for count in class_counts]
    
    class_weights = [w / sum(class_weights) for w in class_weights]
    print(f"Class weights: {class_weights}")
    return torch.tensor(class_weights, dtype=torch.float32)

def adjust_weights(epoch, loss_fn, conf_thresh, device):
    """Keep loss weights constant - no complex schedules"""
    # DO NOTHING - let the model learn naturally
    return loss_fn

def main():

    parser = argparse.ArgumentParser(description='Advanced Detection Training')
    parser.add_argument('--train_dir', default='data_t/train', help='Training dataset directory')
    parser.add_argument('--val_dir', default='data_t/valid', help='Validation dataset directory')
    parser.add_argument('--test_dir', default='data_t/test', help='Test dataset directory')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (BALANCED 1e-3 - faster convergence, not too aggressive)')
    parser.add_argument('--img_size', type=int, default=224, help='Image Size')
    parser.add_argument('--model', type=str, default='base', choices=['base', 'strong'], help='Model variant')
    parser.add_argument('--box_scale', type=float, default=1.3, help='Box scale (increase to correct under-sized boxes)')
    parser.add_argument('--conf_thresh', type=float, default=0.35, help='Confidence Threshold (RAISED 0.25→0.35 to filter weak predictions)')
    parser.add_argument('--eval_conf_thresh', type=float, default=0.25, help='Eval Confidence Threshold (set to 0.25 for evaluation)')
    parser.add_argument('--iou_thresh', type=float, default=0.35, help='IOU Threshold (LOWERED 0.45→0.35 for maximum NMS suppression)')
    parser.add_argument('--output_dir', default='weights', help='Output directory')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--accumulate', type=int, default=2, help='Gradient accumulation steps (REDUCED for faster updates)')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')

    args = parser.parse_args()

    # CRITICAL: Print actual arguments being used
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION (BALANCED SCENARIO A+)")
    print(f"{'='*60}")
    print(f"Learning Rate: {args.lr:.2e} (BALANCED 1e-3)")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.img_size}")
    print(f"Model Variant: {args.model}")
    print(f"Box Scale: {args.box_scale}")

    args = parser.parse_args()
    
    # CRITICAL: Print actual arguments being used
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION (BALANCED SCENARIO A+)")
    print(f"{'='*60}")
    print(f"Learning Rate: {args.lr:.2e} (BALANCED 1e-3)")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.img_size}")
    print(f"Model Variant: {args.model}")
    print(f"Box Scale: {args.box_scale}")
    print(f"Epochs: {args.epochs}")
    print(f"Conf Thresh: {args.conf_thresh}")
    print(f"Eval Conf Thresh: {args.eval_conf_thresh}")
    print(f"IOU Thresh: {args.iou_thresh} (MAX NMS)")
    print(f"AMP Enabled: {args.amp}")
    print(f"Gradient Accumulation: {args.accumulate}")
    print(f"Effective Batch Size: {args.batch_size * args.accumulate}")
    print(f"{'='*60}\n")
    
    # Parse anchors - K-MEANS OPTIMIZED FOR TOMATO_D (24,850 boxes)
    args.anchors = [[11, 8], [17, 10], [23, 15], [29, 16], [35, 21],
                    [65, 24], [49, 60], [95, 50], [137, 71]]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # DO NOT delete output_dir - allows resuming from checkpoints!
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'detections', 'val'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'detections', 'test'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
    
    class_names = {0: "stem", 1: "tomato"}
    
    if args.use_wandb:
        wandb_logger = setup_wandb(args)
    


    print("\n[DEBUG] Loading datasets...")
    train_ds = BotanicalDataset(
        args.train_dir, 
        img_size=args.img_size, 
        mode='train', 
        transform=create_diverse_augmentations(args.img_size),
        use_mosaic=False
    )
    val_ds = BotanicalDataset(args.val_dir, img_size=args.img_size, mode='val')
    test_ds = BotanicalDataset(args.test_dir, img_size=args.img_size, mode='test')

    # Debug: Print class distribution
    label_counts = {0: 0, 1: 0}
    for i in range(min(200, len(train_ds))):
        _, t = train_ds[i]
        for l in t['labels']:
            label_counts[int(l)] += 1
    print(f"[DEBUG] Class distribution in first 200 train samples: stems={label_counts[0]}, tomatoes={label_counts[1]}")

    class_weights = calculate_class_weights(train_ds).to(device)

    # Weighted sampling: per-instance for stems (stronger boost)
    sample_weights = []
    for img_id in train_ds.image_ids:
        anns = train_ds.image_annotations.get(img_id, [])
        stem_count = sum(1 for ann in anns if ann.get("category_id") == 1)
        tomato_count = sum(1 for ann in anns if ann.get("category_id") == 0)
        # Each stem increases weight by 12x, tomato by 1x
        weight = 1.0 + 12.0 * stem_count + 1.0 * tomato_count
        sample_weights.append(weight)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        sampler=sampler,
        collate_fn=collate_fn, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)

    if args.model == 'strong':
        model = HighAccuracyPhytoSparseNetStrong(num_classes=2).to(device)
    else:
        model = HighAccuracyPhytoSparseNet(num_classes=2).to(device)
    model_info = log_model_info(model, args.img_size, device, args.output_dir)

    # Test forward pass and print output shapes
    print("[DEBUG] Running test forward pass...")
    test_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    if isinstance(test_output, dict):
        print(f"[DEBUG] Model output keys: {test_output.keys()}")
        for key, value in test_output.items():
            print(f"  [DEBUG] {key}: {value.shape}")
    else:
        print(f"[DEBUG] Model output shape: {test_output.shape}")

    # Create loss function with STRONG STEM BOOST
    class_weights_tensor = torch.tensor([12.0, 1.0], dtype=torch.float32).to(device)  # stem=12x, tomato=1x
    loss_fn = DetectionLoss(
        alpha=0.25,
        gamma=2.0,
        lambda_box=8.0,
        lambda_obj=2.0,
        lambda_cls=6.0,
        class_weights=class_weights_tensor,
        num_classes=2,
        anchors=args.anchors,
        img_size=args.img_size,
        box_scale=args.box_scale
    )
    
    print(f"\n{'='*60}")
    print(f"Loss Function Configuration (BALANCED SCENARIO A+):")
    print(f"  lambda_box: {loss_fn.lambda_box} (box localization - BOOSTED)")
    print(f"  lambda_obj: {loss_fn.lambda_obj} (objectness - reduced)")
    print(f"  lambda_cls: {loss_fn.lambda_cls} (classification - BALANCED at 6.0)")
    print(f"  focal alpha: {loss_fn.alpha}")
    print(f"  focal gamma: {loss_fn.gamma}")
    print(f"  class_weights: stem={class_weights_tensor[0]:.1f} (BALANCED 8→6), tomato={class_weights_tensor[1]:.1f}")
    print(f"{'='*60}\n")

    optimizer, scheduler = create_optimizer_and_scheduler(model, args)

    amp_enabled = args.amp and torch.cuda.is_available()
    
    # EMA (Exponential Moving Average) for stable model weights
    from copy import deepcopy
    ema_model = deepcopy(model).eval()
    ema_decay = 0.9999
    for param in ema_model.parameters():
        param.requires_grad = False
    
    best_map50 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    train_loss_history = []
    val_metrics_history = []
    
    # RESUME FROM CHECKPOINT if it exists
    start_epoch = 1
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    latest_checkpoint_path = None
    
    # Find latest checkpoint
    checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    if os.path.exists(checkpoints_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoints_dir) if f.startswith('epoch_') and f.endswith('.pth')])
        if checkpoints:
            latest_checkpoint_path = os.path.join(checkpoints_dir, checkpoints[-1])
    
    if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
        try:
            print(f"\n{'='*60}")
            print(f"RESUMING FROM CHECKPOINT: {latest_checkpoint_path}")
            print(f"{'='*60}\n")
            
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_map50 = checkpoint.get('best_map50', 0.0)
            best_epoch = checkpoint.get('epoch', 0)
            
            print(f"Resuming from epoch {start_epoch}")
            print(f"Best mAP@50 so far: {best_map50:.4f}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from epoch 1")
            start_epoch = 1
    

    try:
        scaler = GradScaler(enabled=amp_enabled, init_scale=2.**12)  # 4096 - reduce overflow
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            train_ds.current_epoch = epoch
            # Warmup LR for first 3 epochs
            if epoch <= 3:
                warmup_factor = (2.718 ** (2.0 * epoch / 3.0)) / (2.718 ** 2.0)
                warmup_factor = max(0.1, min(1.0, warmup_factor))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr * warmup_factor
                print(f"Exponential Warmup: LR = {args.lr * warmup_factor:.6f} (factor={warmup_factor:.3f})")

            model.train()
            epoch_loss = 0.0
            epoch_obj = 0.0
            epoch_cls = 0.0
            epoch_box = 0.0
            num_batches = len(train_loader)
            train_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch}")
            optimizer.zero_grad()
            for batch_idx, (imgs, targets) in train_bar:
                imgs = imgs.to(device)
                # Prepare targets for loss
                model_output_for_shape = model(imgs)
                if isinstance(model_output_for_shape, dict):
                    if 'large' in model_output_for_shape:
                        output_tensor_shape = model_output_for_shape['large']
                    elif 'pred_boxes' in model_output_for_shape:
                        output_tensor_shape = convert_dict_to_tensor(model_output_for_shape, num_classes=2, H=7, W=7)
                    else:
                        raise TypeError(f"Dict with unexpected keys: {model_output_for_shape.keys()}")
                elif isinstance(model_output_for_shape, torch.Tensor):
                    output_tensor_shape = model_output_for_shape
                else:
                    raise TypeError(f"Unexpected model output type: {type(model_output_for_shape)}")
                # Move all target tensors to the same device as imgs/model
                targets_for_loss = prepare_targets_for_loss(targets, output_tensor_shape.shape, img_size=args.img_size, anchors=args.anchors, num_classes=2)
                for k in targets_for_loss:
                    if isinstance(targets_for_loss[k], torch.Tensor):
                        targets_for_loss[k] = targets_for_loss[k].to(imgs.device)
                # Forward pass

                with autocast(enabled=amp_enabled):
                    model_output = model(imgs)
                    preds_for_loss = prepare_predictions_for_loss(model_output, num_classes=2)
                    # If preds_for_loss is a dict with 'large', extract it
                    if isinstance(preds_for_loss, dict) and 'large' in preds_for_loss:
                        preds_for_loss = preds_for_loss['large']
                    loss, obj_loss, cls_loss, box_loss = loss_fn(preds_for_loss, targets_for_loss)

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Invalid loss detected in batch {batch_idx}, skipping")
                    optimizer.zero_grad()
                    continue

                scaled_loss = loss / args.accumulate
                if amp_enabled:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                epoch_loss += loss.item()
                epoch_obj += obj_loss.item()
                epoch_cls += cls_loss.item()
                epoch_box += box_loss.item()

                if (batch_idx + 1) % args.accumulate == 0:
                    if amp_enabled:
                        scaler.unscale_(optimizer)
                    has_bad_grads, grad_norm = stabilize_gradients(model, max_norm=1.0, debug=False)
                    if has_bad_grads:
                        print(f"WARNING: Invalid gradients detected in batch {batch_idx}, zeroing them out")
                        optimizer.zero_grad()
                        if amp_enabled:
                            scaler.update()
                        continue
                    if math.isnan(grad_norm) or math.isinf(grad_norm):
                        print(f"Invalid gradient norm detected in batch {batch_idx}, zeroing gradients")
                        optimizer.zero_grad()
                        if amp_enabled:
                            scaler.update()
                        continue
                    if amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                        did_step = True
                    else:
                        optimizer.step()
                        did_step = True
                    if did_step:
                        with torch.no_grad():
                            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                                ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
                    optimizer.zero_grad()

                avg_loss = epoch_loss / (batch_idx + 1)
                train_bar.set_postfix({
                    'loss': f'{avg_loss:.3f}',
                    'obj': f'{epoch_obj/(batch_idx+1):.3f}',
                    'cls': f'{epoch_cls/(batch_idx+1):.3f}',
                    'box': f'{epoch_box/(batch_idx+1):.3f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'
                })

            if epoch > 3:
                try:
                    scheduler.step()
                except Exception as e:
                    print("Error in scheduler step!")
                    raise Exception(e)

            avg_loss = epoch_loss / num_batches
            avg_obj = epoch_obj / num_batches
            avg_cls = epoch_cls / num_batches
            avg_box = epoch_box / num_batches
            train_loss_history.append(avg_loss)

            print(f"\nEpoch {epoch} Training Summary ({time.time()-epoch_start:.1f}s)")
            print(f"Total Loss: {avg_loss:.4f} | Obj: {avg_obj:.4f} | Cls: {avg_cls:.4f} | Box: {avg_box:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.7f}")

            val_metrics = validate_model(ema_model, val_loader, device, class_names, args, epoch, 'val')
            val_metrics_history.append(val_metrics)

            print(f"\nValidation @ Epoch {epoch}")
            print(f"mAP: {val_metrics['map']:.4f} | mAP@50: {val_metrics['map_50']:.4f} | mAP@75: {val_metrics['map_75']:.4f}")
            print(f"Precision: {val_metrics['overall_precision']:.4f} | Recall: {val_metrics['overall_recall']:.4f} | F1: {val_metrics['overall_f1']:.4f}")

            if args.use_wandb:
                log_data = {
                    'epoch': epoch,
                    'train/loss': avg_loss,
                    'train/obj_loss': avg_obj,
                    'train/cls_loss': avg_cls,
                    'train/box_loss': avg_box,
                    'lr': optimizer.param_groups[0]['lr'],
                    'val/map': val_metrics['map'],
                    'val/map_50': val_metrics['map_50'],
                    'val/map_75': val_metrics['map_75'],
                    'val/precision': val_metrics['overall_precision'],
                    'val/recall': val_metrics['overall_recall'],
                    'val/f1': val_metrics['overall_f1'],
                }
                for cls_name in class_names.values():
                    log_data[f'val/{cls_name}_precision'] = val_metrics[f'{cls_name}_precision']
                    log_data[f'val/{cls_name}_recall'] = val_metrics[f'{cls_name}_recall']
                    log_data[f'val/{cls_name}_f1'] = val_metrics[f'{cls_name}_f1']
                wandb.log(log_data)

            if val_metrics['map_50'] > best_map50:
                best_map50 = val_metrics['map_50']
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                print(f"Saved new best model at epoch {epoch} with mAP@50: {val_metrics['map_50']:.4f}")
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == args.epochs:
                ckpt_path = os.path.join(args.output_dir, 'checkpoints', f'epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'metrics': val_metrics,
                    'best_map50': best_map50,
                }, ckpt_path)
                print(f"Saved checkpoint at epoch {epoch}")

            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            torch.cuda.empty_cache()
            gc.collect()
            print(f"{'-'*60}")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        if args.use_wandb:
            wandb.alert(title="Training Failed", text=str(e))
    
    try:
        plot_training_curves(train_loss_history, val_metrics_history, args.output_dir)
        print("Training curves plotted successfully")
    except Exception as e:
        print(f"Error plotting training curves: {e}")
    
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
            print(f"Loaded best model from epoch {best_epoch} for testing")
        except Exception as e:
            print(f"Error loading best model: {e}")
            print("Using current model weights instead")
    
    test_metrics = validate_model(model, test_loader, device, class_names, args, best_epoch, 'test')

    try:
        test_iter = iter(test_loader)
        imgs, targets = next(test_iter)
        
        imgs = imgs.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(imgs)

        if isinstance(outputs, dict):
            if 'large' in outputs:
                # Multi-scale output - use large head
                output_tensor = outputs['large']
            elif 'pred_boxes' in outputs:
                # Prediction dict - convert to tensor
                output_tensor = convert_dict_to_tensor(outputs, num_classes=2, H=7, W=7)
            else:
                raise ValueError(f"Unexpected dict keys: {outputs.keys()}")
        elif isinstance(outputs, torch.Tensor):
            output_tensor = outputs
        else:
            raise TypeError(f"Unexpected model output type: {type(outputs)}")
        
        # Extract first image if batch
        if output_tensor.dim() == 4:
            output_tensor = output_tensor[0]

        boxes, scores, class_ids = decode_predictions_advanced(
            output_tensor,  # Use the processed tensor, NOT outputs[0]
            conf_thresh=args.eval_conf_thresh,
            iou_thresh=args.iou_thresh,
            anchors=args.anchors,
            img_size=args.img_size,
            max_detections=300,  # Set to 300 for evaluation
            box_scale=args.box_scale
        )
        
        boxes = boxes.cpu()
        scores = scores.cpu()
        class_ids = class_ids.cpu()
        
        save_detection_image(
            imgs[0].cpu(), 
            {
                'boxes': targets[0]['boxes'],
                'labels': targets[0]['labels'],
                'image_path': targets[0].get('image_path', '')
            },
            (boxes, scores, class_ids),
            os.path.join(args.output_dir, 'detections', 'test', f'test_result.jpg'),
            class_names,
            conf_thresh=args.eval_conf_thresh,
            img_size=args.img_size
        )
    except Exception as e:
        print(f"Error saving test image: {e}")

    print(f"\nFinal Test Evaluation")
    print(f"mAP: {test_metrics['map']:.4f} | mAP@50: {test_metrics['map_50']:.4f} | mAP@75: {test_metrics['map_75']:.4f}")
    print(f"Precision: {test_metrics['overall_precision']:.4f} | Recall: {test_metrics['overall_recall']:.4f} | F1: {test_metrics['overall_f1']:.4f}")


    print("\nQuantizing model for edge deployment...")
    quantized_model_size_mb = None
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        quantized_path = os.path.join(args.output_dir, 'quantized_model.pth')
        torch.save(quantized_model.state_dict(), quantized_path)
        print("Quantized model saved successfully")
        # Calculate quantized model size
        quantized_model_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
        print(f"[DEBUG] Quantized model size: {quantized_model_size_mb:.2f} MB")
    except Exception as e:
        print(f"Error during quantization: {e}")
        print("Saving regular model instead")
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))

    # Save quantized model size for README.md
    if quantized_model_size_mb is not None:
        with open(os.path.join(args.output_dir, 'quantized_model_size.txt'), 'w') as f:
            f.write(f"Quantized model size: {quantized_model_size_mb:.2f} MB\n")
        print(f"[INFO] Quantized model size written to quantized_model_size.txt")
    
    metrics_path = os.path.join(args.output_dir, 'metrics', 'training_metrics.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump({
            'train_loss': train_loss_history,
            'val_metrics': val_metrics_history,
            'test_metrics': test_metrics,
            'best_epoch': best_epoch,
            'best_map50': best_map50,
        }, f, indent=2, cls=NumpyEncoder)

    print(f"All metrics saved to {metrics_path}")

    try:
        all_true_labels_test = []
        all_pred_labels_test = []
        
        model.eval()
        with torch.no_grad():
            for idx, (imgs, targets) in enumerate(test_loader):
                imgs = imgs.to(device)
                outputs = model(imgs)
                
                # Reuse output handling logic to avoid tensor mismatch
                if isinstance(outputs, dict):
                    if 'large' in outputs:
                        output_tensor = outputs['large']
                    elif 'pred_boxes' in outputs:
                        output_tensor = convert_dict_to_tensor(outputs, num_classes=2, H=7, W=7)
                    else:
                        print(f"Unexpected dict keys: {outputs.keys()}")
                        continue
                elif isinstance(outputs, torch.Tensor):
                    output_tensor = outputs
                else:
                    print(f"Unexpected model output type: {type(outputs)}")
                    continue

                if output_tensor.dim() == 4:
                    output_tensor = output_tensor[0]

                boxes, scores, class_ids = decode_predictions_advanced(
                    output_tensor,
                    conf_thresh=args.eval_conf_thresh,
                    iou_thresh=args.iou_thresh,
                    anchors=args.anchors,
                    img_size=args.img_size,
                    max_detections=300,  # Set to 300 for evaluation
                    box_scale=args.box_scale
                )
                
                gt_boxes = targets[0]['boxes'].cpu()
                gt_labels = targets[0]['labels'].cpu()
                pred_boxes = boxes.cpu()
                pred_labels = class_ids.cpu()

                # IoU-based matching for confusion matrix
                from torchvision.ops import box_iou
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(pred_boxes, gt_boxes)  # [num_preds, num_gts]
                    matched_preds = set()
                    for gt_idx in range(len(gt_boxes)):
                        gt_label = gt_labels[gt_idx].item()
                        best_iou, best_pred_idx = ious[:, gt_idx].max(dim=0)
                        if best_iou > 0.5:
                            pred_label = pred_labels[best_pred_idx].item()
                            all_true_labels_test.append(gt_label)
                            all_pred_labels_test.append(pred_label)
                            matched_preds.add(best_pred_idx.item())
                    # Unmatched predictions = false positives
                    for pred_idx in range(len(pred_boxes)):
                        if pred_idx not in matched_preds:
                            pred_label = pred_labels[pred_idx].item()
                            all_pred_labels_test.append(pred_label)
                            all_true_labels_test.append(-1)
                elif len(gt_boxes) == 0 and len(pred_boxes) > 0:
                    # All predictions are FP
                    for pred_label in pred_labels:
                        all_pred_labels_test.append(pred_label.item())
                        all_true_labels_test.append(-1)
        
        if len(all_true_labels_test) > 0 and len(all_pred_labels_test) > 0:
            cm_test = confusion_matrix(all_true_labels_test, all_pred_labels_test, labels=list(class_names.keys()))
            plot_confusion_matrix(
                cm_test, 
                list(class_names.values()), 
                os.path.join(args.output_dir, 'confusion_matrix_test.png')
            )
            
            test_metrics['confusion_matrix'] = cm_test.tolist()
        
    except Exception as e:
        print(f"Error creating test confusion matrix: {e}")

    try:
        if test_metrics:
            test_fig, test_ax = plt.subplots(figsize=(10, 6))
            test_categories = ['mAP', 'mAP@50', 'mAP@75']
            test_values = [test_metrics['map'], test_metrics['map_50'], test_metrics['map_75']]
            
            bars = test_ax.bar(test_categories, test_values)
            test_ax.set_title('Test Metrics')
            test_ax.set_ylabel('Score')
            test_ax.set_ylim(0, 1)
            
            for bar, value in zip(bars, test_values):
                height = bar.get_height()
                test_ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'test_metrics.png'))
            plt.close()
            print("Test metrics plotted successfully")
    except Exception as e:
        print(f"Error plotting test metrics: {e}")
    
    final_summary = create_final_summary(model_info, train_loss_history, val_metrics_history, test_metrics, args.output_dir)

    final_summ_path = os.path.join(args.output_dir, 'final_summary', 'training_metrics.json')
    os.makedirs(os.path.dirname(final_summ_path), exist_ok=True)
    with open(final_summ_path, 'w') as f:
        json.dump(final_summary, f, cls=NumpyEncoder, indent=2)
        
    print(f"Training complete! Final summary saved.")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
