import os
import time
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
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
from phytonet import HighAccuracyPhytoSparseNet
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
                             anchors=[[10,12], [16,18], [24,28], [32,36], [48,52],
                                      [64,68], [80,84], [96,100], [112,116]],
                             num_classes=2):
    """
    Convert raw targets (boxes, labels) to format expected by DetectionLoss.
    
    Args:
        raw_targets: List of dicts with 'boxes' [N,4] and 'labels' [N]
        model_output_shape: Shape of model output [B, C, H, W]
        img_size: Image size
        anchors: List of anchor boxes
        num_classes: Number of classes
    
    Returns:
        dict with 'boxes', 'cls', 'obj' tensors in grid format
    """
    device = raw_targets[0]['boxes'].device
    batch_size = len(raw_targets)
    _, C, H, W = model_output_shape
    
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    A = anchors.shape[0]
    
    # Initialize target tensors
    target_obj = torch.zeros(batch_size, A * H * W, device=device)
    target_cls = torch.zeros(batch_size, A * H * W, num_classes, device=device)
    target_boxes = torch.zeros(batch_size, A * H * W, 4, device=device)
    
    for b in range(batch_size):
        gt_boxes = raw_targets[b]['boxes']  # [N, 4] normalized
        gt_labels = raw_targets[b]['labels']  # [N]
        
        if len(gt_boxes) == 0:
            continue
        
        # Convert to center format
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
        gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
        
        # Find grid cells
        grid_x = (gt_cx * W).long().clamp(0, W-1)
        grid_y = (gt_cy * H).long().clamp(0, H-1)
        
        # Assign to anchors
        for i, (gx, gy, gw, gh, label) in enumerate(zip(grid_x, grid_y, gt_w, gt_h, gt_labels)):
            # Find best anchor
            anchor_ious = []
            for anchor in anchors:
                aw, ah = anchor[0] / img_size, anchor[1] / img_size
                inter_w = min(gw, aw)
                inter_h = min(gh, ah)
                inter = inter_w * inter_h
                union = gw * gh + aw * ah - inter
                iou = inter / (union + 1e-6)
                anchor_ious.append(iou)
            
            best_anchor = torch.tensor(anchor_ious).argmax().item()
            
            # Calculate index in flattened grid
            idx = best_anchor * H * W + gy * W + gx
            
            # Convert label to 0-indexed if needed
            label_idx = int(label.item())
            # Clamp label to valid range [0, num_classes-1]
            label_idx = max(0, min(label_idx, num_classes - 1))
            
            # Set targets
            target_obj[b, idx] = 1.0
            target_cls[b, idx, label_idx] = 1.0
            target_boxes[b, idx] = gt_boxes[i]
    
    return {
        'obj': target_obj,
        'cls': target_cls,
        'boxes': target_boxes
    }

def prepare_predictions_for_loss(model_output, num_classes=2):
    """
    Convert model output tensor to format expected by DetectionLoss.
    
    Args:
        model_output: Either a dict with keys or Tensor of shape [B, C, H, W] where C = A*(5+num_classes)
        num_classes: Number of classes
    
    Returns:
        dict with 'pred_boxes', 'pred_cls', 'pred_obj' tensors
    """
    # If already a dict with correct keys, return as is
    if isinstance(model_output, dict):
        if 'pred_boxes' in model_output and 'pred_cls' in model_output and 'pred_obj' in model_output:
            return model_output
    
    # Otherwise convert tensor to dict
    B, C, H, W = model_output.shape
    A = C // (5 + num_classes)
    
    # Reshape: [B, A, 5+num_classes, H, W] -> [B, A, H, W, 5+num_classes]
    pred = model_output.view(B, A, 5 + num_classes, H, W)
    pred = pred.permute(0, 1, 3, 4, 2).contiguous()
    
    # Flatten spatial and anchor dimensions: [B, A*H*W, ...]
    pred = pred.view(B, A * H * W, 5 + num_classes)
    
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
        Tensor of shape [B, C, H, W]
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
    pred = pred.permute(0, 1, 4, 2, 3)
    
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
    """More diverse augmentations to help with generalization"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),
        A.Perspective(scale=0.1, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
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

def decode_predictions_advanced(pred, conf_thresh=0.5, iou_thresh=0.3,
                                anchors=[[10,12], [16,18], [24,28], [32,36], [48,52],
                                         [64,68], [80,84], [96,100], [112,116]],
                                img_size=224, max_detections=300):
    """Decode network output to normalized boxes [0..1], scores and class ids."""
    device = pred.device
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    A = anchors.shape[0]

    C, H, W = pred.shape
    
    # Try to infer number of classes
    if (C % A) == 0:
        num_classes = (C // A) - 5
    else:
        print(f"Warning: Cannot perfectly divide channels {C} by anchors {A}")
        print(f"C % A = {C % A}, trying to estimate num_classes")
        # Try common values
        for nc in [2, 3, 1]:  # Try 2, 3, or 1 class
            if C == A * (5 + nc):
                num_classes = nc
                break
        else:
            print(f"Could not infer num_classes. C={C}, A={A}, H={H}, W={W}")
            # Return empty predictions
            return (torch.empty((0, 4), device=device), 
                    torch.empty((0,), device=device), 
                    torch.empty((0,), dtype=torch.int64, device=device))
    
    if num_classes < 1:
        print(f"Invalid num_classes={num_classes}, returning empty predictions")
        return (torch.empty((0, 4), device=device), 
                torch.empty((0,), device=device), 
                torch.empty((0,), dtype=torch.int64, device=device))

    pred = pred.view(A, 5 + num_classes, H, W).permute(0, 2, 3, 1).contiguous()

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid_x = grid_x.view(1, H, W, 1).expand(A, H, W, 1).float()
    grid_y = grid_y.view(1, H, W, 1).expand(A, H, W, 1).float()

    tx = pred[..., 0:1]
    ty = pred[..., 1:2]
    tw = pred[..., 2:3]
    th = pred[..., 3:4]
    to = pred[..., 4:5]
    tcls = pred[..., 5:5+num_classes]

    cx = (torch.sigmoid(tx) + grid_x) / W
    cy = (torch.sigmoid(ty) + grid_y) / H

    anchors_norm = anchors / float(img_size)
    aw = anchors_norm[:, 0].view(A, 1, 1, 1)
    ah = anchors_norm[:, 1].view(A, 1, 1, 1)

    tw_clamped = tw.clamp(min=-10.0, max=10.0)
    th_clamped = th.clamp(min=-10.0, max=10.0)

    bw = torch.exp(tw_clamped) * aw
    bh = torch.exp(th_clamped) * ah

    x1 = (cx - bw / 2.0).reshape(-1)
    y1 = (cy - bh / 2.0).reshape(-1)
    x2 = (cx + bw / 2.0).reshape(-1)
    y2 = (cy + bh / 2.0).reshape(-1)

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    obj_prob = torch.sigmoid(to).reshape(-1)
    cls_prob = torch.softmax(tcls, dim=-1).reshape(-1, num_classes)

    cls_scores, cls_ids = cls_prob.max(dim=-1)

    eps = 1e-6
    obj_logits = torch.log(obj_prob.clamp(min=eps) / (1 - obj_prob.clamp(min=eps) + eps))
    cls_logits = torch.log(cls_scores.clamp(min=eps) / (1 - cls_scores.clamp(min=eps) + eps))
    fused_logits = 0.6 * obj_logits + 0.4 * cls_logits
    scores = torch.sigmoid(fused_logits)

    class_ids = cls_ids.reshape(-1)
    class_thresholds = torch.tensor([conf_thresh, conf_thresh]).to(device)  # [stem_thresh, tomato_thresh]
    adjusted_thresh = class_thresholds[class_ids.long()]
    keep_mask = scores > adjusted_thresh
    if keep_mask.sum() == 0:
        return torch.empty((0, 4), device=device), torch.empty((0,), device=device), torch.empty((0,), dtype=torch.int64, device=device)

    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]

    abs_boxes = boxes.clone()
    abs_boxes[:, 0] = abs_boxes[:, 0] * img_size
    abs_boxes[:, 1] = abs_boxes[:, 1] * img_size
    abs_boxes[:, 2] = abs_boxes[:, 2] * img_size
    abs_boxes[:, 3] = abs_boxes[:, 3] * img_size

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
        class_nms_thresholds = {0: 0.3, 1: 0.3}  # Lower = more aggressive NMS
        class_iou_thresh = class_nms_thresholds.get(int(c.item()), iou_thresh)
        keep = nms(cls_boxes, cls_scores, class_iou_thresh)
        keep = keep[:max_detections]
        final_boxes.append(cls_boxes[keep])
        final_scores.append(cls_scores[keep])
        final_classes.append(torch.full((len(keep),), int(c.item()), dtype=torch.int64, device=device))

    if len(final_boxes) == 0:
        return torch.empty((0, 4), device=device), torch.empty((0,), device=device), torch.empty((0,), dtype=torch.int64, device=device)

    final_boxes = torch.cat(final_boxes, dim=0)
    final_scores = torch.cat(final_scores, dim=0)
    final_classes = torch.cat(final_classes, dim=0)

    final_boxes = final_boxes / float(img_size)

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
                         conf_thresh=0.5, img_size=224):
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
        map_metric = MeanAveragePrecision(class_metrics=True)
        
        with torch.no_grad():
            for idx, (imgs, targets) in enumerate(dataloader):
                imgs = imgs.to(device)
                outputs = model(imgs)
                
                # Convert dict to tensor if needed for decoding
                if isinstance(outputs, dict):
                    output_tensor = convert_dict_to_tensor(outputs, num_classes=2, H=7, W=7)[0]
                else:
                    output_tensor = outputs[0]
                
                boxes, scores, class_ids = decode_predictions_advanced(
                    output_tensor, 
                    conf_thresh=args.conf_thresh,
                    anchors=args.anchors,
                    img_size=args.img_size
                )
                
                boxes = boxes.cpu()
                scores = scores.cpu()
                class_ids = class_ids.cpu()
                
                gt_boxes = targets[0]['boxes'].cpu()
                gt_labels = targets[0]['labels'].cpu()
                
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
                
                batch_true_labels = gt_labels.numpy()
                batch_pred_labels = class_ids.numpy()
                
                min_length = min(len(batch_true_labels), len(batch_pred_labels))
                if min_length > 0:
                    all_true_labels.extend(batch_true_labels[:min_length])
                    all_pred_labels.extend(batch_pred_labels[:min_length])
                
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
                        conf_thresh=args.conf_thresh,
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
        
        try:
            if len(all_true_labels) > 0 and len(all_pred_labels) > 0:
                cm = confusion_matrix(all_true_labels, all_pred_labels, labels=list(class_names.keys()))
                val_metrics['confusion_matrix'] = cm.tolist()
                
                for i, cls_name in class_names.items():
                    if i < cm.shape[0] and i < cm.shape[1]:
                        tp = cm[i, i]
                        fp = cm[:, i].sum() - tp
                        fn = cm[i, :].sum() - tp
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        val_metrics[f'{cls_name}_precision'] = precision
                        val_metrics[f'{cls_name}_recall'] = recall
                        val_metrics[f'{cls_name}_f1'] = f1
                        val_metrics[f'{cls_name}_tp'] = tp
                        val_metrics[f'{cls_name}_fp'] = fp
                        val_metrics[f'{cls_name}_fn'] = fn
                
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

def stabilize_gradients(model, max_norm=0.5, debug=False):
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
    """Create optimizer with warm-up and cosine annealing"""
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)  
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    return optimizer, scheduler

def calculate_class_weights(dataset):
    """Calculate class weights with smoothing"""
    class_counts = [1, 1, 1]
    
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
    """Dynamically adjust loss weights based on training progress"""
    
    if epoch == 30:
        print("\n" + "="*60)
        print(f"EPOCH {epoch}: Adjusting loss weights (Phase 2)")
        print("="*60)
        loss_fn.lambda_cls = 4.5
        loss_fn.gamma = 2.5
        conf_thresh = 0.4
        print(f"  lambda_cls: {loss_fn.lambda_cls}")
        print(f"  gamma: {loss_fn.gamma}")
        print(f"  conf_thresh: {conf_thresh}")
        print("="*60 + "\n")
    
    elif epoch == 60:
        print("\n" + "="*60)
        print(f"EPOCH {epoch}: Adjusting loss weights (Phase 3)")
        print("="*60)
        loss_fn.lambda_cls = 5.5
        loss_fn.gamma = 3.0
        conf_thresh = 0.6
        print(f"  lambda_cls: {loss_fn.lambda_cls}")
        print(f"  gamma: {loss_fn.gamma}")
        print(f"  conf_thresh: {conf_thresh}")
        print("="*60 + "\n")
    
    elif epoch == 90:
        print("\n" + "="*60)
        print(f"EPOCH {epoch}: Adjusting loss weights (Phase 4)")
        print("="*60)
        loss_fn.lambda_cls = 6.5
        loss_fn.gamma = 3.5
        conf_thresh = 0.7
        print(f"  lambda_cls: {loss_fn.lambda_cls}")
        print(f"  gamma: {loss_fn.gamma}")
        print(f"  conf_thresh: {conf_thresh}")
        print("="*60 + "\n")
    
    elif epoch == 150:
        print("\n" + "="*60)
        print(f"EPOCH {epoch}: Adjusting loss weights (Phase 4)")
        print("="*60)
        loss_fn.lambda_cls = 8.0
        loss_fn.gamma = 4.0
        conf_thresh = 0.8
        print(f"  lambda_cls: {loss_fn.lambda_cls}")
        print(f"  gamma: {loss_fn.gamma}")
        print(f"  conf_thresh: {conf_thresh}")
        print("="*60 + "\n")
    
    elif epoch == 200:
        print("\n" + "="*60)
        print(f"EPOCH {epoch}: Adjusting loss weights (Phase 4)")
        print("="*60)
        loss_fn.lambda_cls = 9.0
        loss_fn.gamma = 5.0
        conf_thresh = 0.9
        print(f"  lambda_cls: {loss_fn.lambda_cls}")
        print(f"  gamma: {loss_fn.gamma}")
        print(f"  conf_thresh: {conf_thresh}")
        print("="*60 + "\n")
    
    return loss_fn

def main():
    parser = argparse.ArgumentParser(description='Advanced Detection Training')
    parser.add_argument('--train_dir', default='data/train', help='Training dataset directory')
    parser.add_argument('--val_dir', default='data/val', help='Validation dataset directory')
    parser.add_argument('--test_dir', default='data/test', help='Test dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Image Size')
    parser.add_argument('--conf_thresh', type=float, default=0.2, help='Confidence Threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.3, help='IOU Threshold')
    parser.add_argument('--output_dir', default='weights', help='Output directory')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--accumulate', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')

    args = parser.parse_args()
    
    # Parse anchors properly
    args.anchors = [[10,12], [16,18], [24,28], [32,36], [48,52], 
                    [64,68], [80,84], [96,100], [112,116]]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'detections', 'val'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'detections', 'test'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'metrics'), exist_ok=True)
    
    class_names = {0: "stem", 1: "tomato"}
    
    if args.use_wandb:
        wandb_logger = setup_wandb(args)
    
    train_ds = BotanicalDataset(
        args.train_dir, 
        img_size=args.img_size, 
        mode='train', 
        transform=create_diverse_augmentations(args.img_size)
    )
    val_ds = BotanicalDataset(args.val_dir, img_size=args.img_size, mode='val')
    test_ds = BotanicalDataset(args.test_dir, img_size=args.img_size, mode='test')

    class_weights = calculate_class_weights(train_ds).to(device)

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = HighAccuracyPhytoSparseNet(num_classes=2).to(device)
    model_info = log_model_info(model, args.img_size, device, args.output_dir)

    # Test forward pass
    test_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    
    # Check if model returns dict or tensor
    if isinstance(test_output, dict):
        print(f"Model output keys: {test_output.keys()}")
        for key, value in test_output.items():
            print(f"  {key}: {value.shape}")
    else:
        print(f"Model output shape: {test_output.shape}")

    # Create loss function
    class_weights_tensor = torch.tensor([3.0, 3.0], dtype=torch.float32).to(device)  # [stem_weight, tomato_weight]
    loss_fn = DetectionLoss(alpha=0.25,
                            gamma=2.0,
                            lambda_box=2.0,
                            lambda_cls=4.0,
                            lambda_obj=2.0,
                            class_weights=class_weights_tensor
                            )

    optimizer, scheduler = create_optimizer_and_scheduler(model, args)

    amp_enabled = args.amp and torch.cuda.is_available()
    
    best_map50 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    train_loss_history = []
    val_metrics_history = []
    
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            loss_fn = adjust_weights(epoch, loss_fn, args.conf_thresh, device)
            
            model.train()
            
            epoch_loss = 0.0
            epoch_obj = 0.0
            epoch_cls = 0.0
            epoch_box = 0.0
            
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
            
            optimizer.zero_grad()
            
            scaler = GradScaler(enabled=amp_enabled, init_scale=2.**8)

            for batch_idx, (imgs, targets) in enumerate(train_bar):
                imgs = imgs.to(device, non_blocking=True)
                
                device_targets = []
                for target in targets:
                    device_target = {
                        'boxes': target['boxes'].to(device),
                        'labels': target['labels'].to(device),
                        'image_path': target.get('image_path', '')
                    }
                    device_targets.append(device_target)
                
                with autocast(enabled=amp_enabled):
                    outputs = model(imgs)
                    
                    # Convert to dict format (handles both dict and tensor inputs)
                    pred_dict = prepare_predictions_for_loss(outputs, num_classes=2)
                    
                    # Get shape for target preparation
                    if isinstance(outputs, dict):
                        # Estimate shape from dict
                        batch_size = pred_dict['pred_boxes'].shape[0]
                        num_cells = pred_dict['pred_boxes'].shape[1]
                        H = W = int((num_cells / 9) ** 0.5)  # Assuming 9 anchors
                        output_shape = (batch_size, 9 * 7, H, W)  # 9 anchors, 7 = 5 + 2 classes
                    else:
                        output_shape = outputs.shape
                    
                    target_dict = prepare_targets_for_loss(device_targets, output_shape, 
                                                           img_size=args.img_size, 
                                                           anchors=args.anchors,
                                                           num_classes=2)
                    
                    # Call loss function with dict inputs
                    loss, cls_loss, obj_loss, box_loss = loss_fn(pred_dict, target_dict)
                
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Invalid loss detected in batch {batch_idx}, skipping")
                    optimizer.zero_grad()
                    continue
                
                scaled_loss = loss / args.accumulate
                
                if amp_enabled:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                epoch_loss += loss.item()
                epoch_obj += obj_loss.item()
                epoch_cls += cls_loss.item()
                epoch_box += box_loss.item()
                
                avg_loss = epoch_loss / (batch_idx + 1)
                train_bar.set_postfix({
                    'loss': f'{avg_loss:.3f}',
                    'obj': f'{epoch_obj/(batch_idx+1):.3f}',
                    'cls': f'{epoch_cls/(batch_idx+1):.3f}',
                    'box': f'{epoch_box/(batch_idx+1):.3f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'
                })
                
                if (batch_idx + 1) % args.accumulate == 0:
                    if amp_enabled:
                        scaler.unscale_(optimizer)
                    
                    has_bad_grads, grad_norm = stabilize_gradients(model, max_norm=0.5, debug=False)

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
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    
            try:
                scheduler.step()
            except Exception as e:
                print("Error in scheduler step!")
                raise Exception(e)
            
            num_batches = len(train_loader)
            avg_loss = epoch_loss / num_batches
            avg_obj = epoch_obj / num_batches
            avg_cls = epoch_cls / num_batches
            avg_box = epoch_box / num_batches
            
            train_loss_history.append(avg_loss)
            
            print(f"\nEpoch {epoch} Training Summary ({time.time()-epoch_start:.1f}s)")
            print(f"Total Loss: {avg_loss:.4f} | Obj: {avg_obj:.4f} | Cls: {avg_cls:.4f} | Box: {avg_box:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.7f}")

            val_metrics = validate_model(model, val_loader, device, class_names, args, epoch, 'val')
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
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"Loaded best model from epoch {best_epoch} for testing")
        except Exception as e:
            print(f"Error loading best model: {e}")
            print("Using current model weights instead")
    
    test_metrics = validate_model(model, test_loader, device, class_names, args, best_epoch, 'test')

    try:
        test_iter = iter(test_loader)
        imgs, targets = next(test_iter)
        
        imgs = imgs.to(device)
        with torch.no_grad():
            outputs = model(imgs)
        
        boxes, scores, class_ids = decode_predictions_advanced(
            outputs[0], 
            conf_thresh=args.conf_thresh,
            anchors=args.anchors,
            img_size=args.img_size
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
            conf_thresh=args.conf_thresh,
            img_size=args.img_size
        )
    except Exception as e:
        print(f"Error saving test image: {e}")

    print(f"\nFinal Test Evaluation")
    print(f"mAP: {test_metrics['map']:.4f} | mAP@50: {test_metrics['map_50']:.4f} | mAP@75: {test_metrics['map_75']:.4f}")
    print(f"Precision: {test_metrics['overall_precision']:.4f} | Recall: {test_metrics['overall_recall']:.4f} | F1: {test_metrics['overall_f1']:.4f}")

    print("\nQuantizing model for edge deployment...")
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        torch.save(quantized_model.state_dict(), os.path.join(args.output_dir, 'quantized_model.pth'))
        print("Quantized model saved successfully")
    except Exception as e:
        print(f"Error during quantization: {e}")
        print("Saving regular model instead")
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    
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
                
                boxes, scores, class_ids = decode_predictions_advanced(
                    outputs[0], 
                    conf_thresh=args.conf_thresh,
                    anchors=args.anchors,
                    img_size=args.img_size
                )
                
                class_ids = class_ids.cpu().numpy()
                gt_labels = targets[0]['labels'].cpu().numpy()
                
                min_length = min(len(gt_labels), len(class_ids))
                if min_length > 0:
                    all_true_labels_test.extend(gt_labels[:min_length])
                    all_pred_labels_test.extend(class_ids[:min_length])
        
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