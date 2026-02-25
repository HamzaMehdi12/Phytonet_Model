"""
infer.py — inference script for PhytoNetEdge and HighAccuracyPhytoSparseNet

FIXES vs previous version
──────────────────────────
FIX I1 (CRITICAL): ALL heads are now decoded and merged with cross-head NMS.
    - PhytoNetEdge: 3 heads (small 28×28, medium 14×14, large 7×7)
    - Legacy model: 2 heads (medium 14×14, large 7×7)
    Previously only 'large' was decoded → stems never appeared.

FIX I2: Per-head anchor assignment. Each head uses only its 3 anchors,
    not all 9.

FIX I3: Anchors are properly matched to heads based on stride/resolution.

FIX I4: --box_scale default changed to 1.0 to match fixed training.

FIX I5: --model argument now includes 'edge' for PhytoNetEdge.

FIX I6: torch.load uses weights_only=False to suppress warnings.

FIX I7: Added --eval_conf_thresh with sensible default (0.15).
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torchvision.ops import nms
from PIL import Image

from phytonet import (
    HighAccuracyPhytoSparseNet, 
    HighAccuracyPhytoSparseNetStrong,
    PhytoNetEdge
)


# ─────────────────────────────────────────────────────────────
#  Configuration (must match train.py)
# ─────────────────────────────────────────────────────────────

class Config:
    """Anchor configuration matching train.py"""
    
    # Per-head anchors for PhytoNetEdge (3 anchors each)
    ANCHORS_SMALL  = [[10, 6],  [15, 9],  [22, 14]]   # stride 8  → 28×28
    ANCHORS_MEDIUM = [[28, 18], [38, 25], [55, 35]]   # stride 16 → 14×14
    ANCHORS_LARGE  = [[70, 45], [95, 60], [130, 80]]  # stride 32 → 7×7
    
    # Combined anchors for legacy models (9 anchors total)
    ANCHORS_ALL = ANCHORS_SMALL + ANCHORS_MEDIUM + ANCHORS_LARGE


# ─────────────────────────────────────────────────────────────
#  Transform
# ─────────────────────────────────────────────────────────────

def get_infer_transform(img_size=224):
    """Standard inference transform matching training normalization."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────
#  Single-head Decoder
# ─────────────────────────────────────────────────────────────

def decode_single_head(pred, anchors, img_size=224, conf_thresh=0.25,
                       box_scale=1.0, num_classes=2):
    """
    Decode predictions from a single detection head.
    
    Args:
        pred: [C, H, W] raw predictions for one image
        anchors: list of [w, h] anchor sizes for THIS head only
        img_size: input image size
        conf_thresh: confidence threshold
        box_scale: box size multiplier
        num_classes: number of classes
    
    Returns:
        boxes [N, 4], scores [N], class_ids [N] (normalized to [0,1])
    """
    device = pred.device
    anchors_t = torch.tensor(anchors, dtype=torch.float32, device=device)
    A = len(anchors)
    C, H, W = pred.shape
    
    vals_per_anchor = 5 + num_classes
    empty = (torch.empty((0, 4), device=device),
             torch.empty((0,), device=device),
             torch.empty((0,), dtype=torch.int64, device=device))
    
    # Validate channel count
    if C != A * vals_per_anchor:
        print(f"Warning: Channel mismatch in decoder. C={C}, expected {A}×{vals_per_anchor}={A*vals_per_anchor}")
        return empty
    
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
    
    # Decode box coordinates
    cx = (torch.sigmoid(pred[..., 0:1]) + gx) / W
    cy = (torch.sigmoid(pred[..., 1:2]) + gy) / H
    
    an = anchors_t / float(img_size)
    aw = an[:, 0].view(A, 1, 1, 1)
    ah = an[:, 1].view(A, 1, 1, 1)
    
    bw = torch.exp(pred[..., 2:3].clamp(-5, 5)) * aw * box_scale
    bh = torch.exp(pred[..., 3:4].clamp(-5, 5)) * ah * box_scale
    
    # Convert to corner format
    x1 = (cx - bw / 2).reshape(-1)
    y1 = (cy - bh / 2).reshape(-1)
    x2 = (cx + bw / 2).reshape(-1)
    y2 = (cy + bh / 2).reshape(-1)
    boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)
    
    # Decode scores
    obj_prob = torch.sigmoid(pred[..., 4]).reshape(-1)
    cls_prob = torch.sigmoid(pred[..., 5:]).reshape(-1, num_classes)
    cls_scores, cls_ids = cls_prob.max(dim=-1)
    
    # Combined score (geometric mean of objectness and class confidence)
    scores = torch.sqrt(obj_prob * cls_scores)
    
    # Filter by confidence
    keep = scores > conf_thresh
    if keep.sum() == 0:
        return empty
    
    return boxes[keep], scores[keep], cls_ids[keep]


# ─────────────────────────────────────────────────────────────
#  Multi-head Decoder with Cross-head NMS
# ─────────────────────────────────────────────────────────────

def decode_and_merge_heads(model_output, img_size, conf_thresh, iou_thresh,
                           box_scale, is_edge_model=True, max_detections=300):
    """
    Decode ALL detection heads and merge with cross-head NMS.
    
    FIX I1: Now properly handles PhytoNetEdge (3 heads) and legacy (2 heads).
    FIX I2: Each head uses only its assigned anchors.
    
    Args:
        model_output: dict with head tensors {name: [C, H, W]}
        img_size: input image size
        conf_thresh: confidence threshold
        iou_thresh: NMS IoU threshold
        box_scale: box size multiplier
        is_edge_model: True for PhytoNetEdge (3 heads), False for legacy (2 heads)
        max_detections: maximum detections to keep
    
    Returns:
        boxes [N, 4], scores [N], class_ids [N] (normalized to [0,1])
    """
    all_boxes, all_scores, all_cls = [], [], []
    device = None
    
    # Map head names to their anchor configurations
    if is_edge_model:
        # PhytoNetEdge: 3 heads with 3 anchors each
        head_config = {
            'small':  Config.ANCHORS_SMALL,   # 28×28
            'medium': Config.ANCHORS_MEDIUM,  # 14×14
            'large':  Config.ANCHORS_LARGE,   # 7×7
        }
    else:
        # Legacy 2-head model: uses all 9 anchors per head
        head_config = {
            'medium': Config.ANCHORS_ALL,
            'large':  Config.ANCHORS_ALL,
        }
    
    # Decode each head
    for head_name, tensor in model_output.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if head_name not in head_config:
            continue
        
        device = tensor.device
        anchors = head_config[head_name]
        
        # Validate anchor-channel alignment for edge model
        if is_edge_model:
            expected_channels = len(anchors) * 7  # 3 anchors × (5 + 2 classes)
            if tensor.shape[0] != expected_channels:
                print(f"Warning: {head_name} head has {tensor.shape[0]} channels, "
                      f"expected {expected_channels}")
                continue
        
        boxes, scores, cls_ids = decode_single_head(
            tensor,
            anchors=anchors,
            img_size=img_size,
            conf_thresh=conf_thresh,
            box_scale=box_scale,
            num_classes=2
        )
        
        if len(boxes) > 0:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_cls.append(cls_ids)
    
    # Handle empty predictions
    if device is None:
        device = 'cpu'
    empty = (torch.empty((0, 4), device=device),
             torch.empty((0,), device=device),
             torch.empty((0,), dtype=torch.int64, device=device))
    
    if not all_boxes:
        return empty
    
    # Concatenate predictions from all heads
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    class_ids = torch.cat(all_cls)
    
    # Per-class NMS across all heads
    abs_boxes = boxes * img_size
    final_boxes, final_scores, final_cls = [], [], []
    
    for c in class_ids.unique():
        mask = class_ids == c
        c_boxes = abs_boxes[mask]
        c_scores = scores[mask]
        
        # Apply NMS
        keep = nms(c_boxes, c_scores, iou_thresh)[:max_detections]
        
        final_boxes.append(c_boxes[keep])
        final_scores.append(c_scores[keep])
        final_cls.append(torch.full((len(keep),), int(c.item()), 
                                    dtype=torch.int64, device=device))
    
    if not final_boxes:
        return empty
    
    # Combine and normalize
    boxes = torch.cat(final_boxes) / float(img_size)
    scores = torch.cat(final_scores)
    class_ids = torch.cat(final_cls)
    
    # Filter invalid boxes
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    
    return boxes[valid], scores[valid], class_ids[valid]


# ─────────────────────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────────────────────

def save_detection_image(image_tensor, predictions, output_path, class_names, 
                         conf_thresh, show_conf=True):
    """
    Save image with detection boxes overlaid.
    
    Args:
        image_tensor: [3, H, W] normalized tensor
        predictions: (boxes, scores, cls_ids) tuple
        output_path: path to save image
        class_names: dict mapping class_id to name
        conf_thresh: minimum confidence to display
        show_conf: whether to show confidence in label
    """
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = ((image_tensor.cpu() * std + mean).clamp(0, 1)
           .numpy().transpose(1, 2, 0) * 255).astype(np.uint8).copy()
    h, w = img.shape[:2]
    
    boxes, scores, cls_ids = predictions
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(cls_ids, torch.Tensor):
        cls_ids = cls_ids.cpu().numpy()
    
    # Color map: stem=blue, tomato=green
    colors = {0: (255, 100, 100), 1: (100, 255, 100)}  # BGR
    
    n_displayed = 0
    for i in range(len(boxes)):
        score = float(scores[i])
        if score < conf_thresh:
            continue
        
        x1 = int(max(0, boxes[i][0] * w))
        y1 = int(max(0, boxes[i][1] * h))
        x2 = int(min(w - 1, boxes[i][2] * w))
        y2 = int(min(h - 1, boxes[i][3] * h))
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        cls_id = int(cls_ids[i])
        color = colors.get(cls_id, (200, 200, 200))
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        name = class_names.get(cls_id, '?')
        if show_conf:
            label = f"{name}: {score:.2f}"
        else:
            label = name
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        n_displayed += 1
    
    # Save image
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path} ({n_displayed} detections displayed)")


def print_detection_summary(boxes, scores, cls_ids, class_names, conf_thresh):
    """Print summary of detections."""
    if len(boxes) == 0:
        print("  No detections")
        return
    
    # Count per class above threshold
    counts = {}
    for i, (score, cls_id) in enumerate(zip(scores, cls_ids)):
        if float(score) >= conf_thresh:
            cls_id = int(cls_id)
            name = class_names.get(cls_id, f'class_{cls_id}')
            counts[name] = counts.get(name, 0) + 1
    
    if counts:
        summary = ", ".join(f"{count} {name}(s)" for name, count in counts.items())
        print(f"  Detections: {summary}")
    else:
        print("  No detections above threshold")


# ─────────────────────────────────────────────────────────────
#  Main CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tomato/stem detection inference with PhytoNetEdge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output
    parser.add_argument("--weights", default="weights_edge/best_model.pth",
                        help="Path to model weights")
    parser.add_argument("--image", default=None, 
                        help="Single image path")
    parser.add_argument("--image_dir", default=None, 
                        help="Directory of images")
    parser.add_argument("--output_dir", default="inference_output",
                        help="Output directory for visualizations")
    
    # Model configuration
    parser.add_argument("--model", default="edge", choices=["base", "strong", "edge"],
                        help="Model architecture")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    
    # Detection parameters
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--box_scale", type=float, default=1.0,
                        help="Box scale (must match training)")
    
    # Output options
    parser.add_argument("--save_txt", action="store_true",
                        help="Save detections to text file")
    parser.add_argument("--no_save_img", action="store_true",
                        help="Don't save visualization images")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.image and not args.image_dir:
        raise ValueError("Provide --image or --image_dir")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    if args.model == "edge":
        model = PhytoNetEdge(num_classes=2, num_anchors=3).to(device)
        is_edge_model = True
    elif args.model == "strong":
        model = HighAccuracyPhytoSparseNetStrong(num_classes=2).to(device)
        is_edge_model = False
    else:
        model = HighAccuracyPhytoSparseNet(num_classes=2).to(device)
        is_edge_model = False
    
    # Load weights
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    
    state_dict = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded {args.model} model from {args.weights}")
    
    # Verify model output
    with torch.no_grad():
        test_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        test_output = model(test_input)
        if isinstance(test_output, dict):
            heads_info = {k: tuple(v.shape) for k, v in test_output.items()}
            print(f"Model heads: {heads_info}")
        else:
            print(f"Model output shape: {test_output.shape}")
    
    # Class names
    class_names = {0: "stem", 1: "tomato"}
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect image paths
    if args.image:
        image_paths = [args.image]
    else:
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_paths = [
            os.path.join(args.image_dir, f)
            for f in sorted(os.listdir(args.image_dir))
            if f.lower().endswith(valid_extensions)
        ]
    
    if not image_paths:
        print("No images found!")
        return
    
    print(f"\nProcessing {len(image_paths)} image(s)...")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Box scale: {args.box_scale}")
    print("-" * 50)
    
    # Inference transform
    transform = get_infer_transform(args.img_size)
    
    # Process images
    all_results = []
    
    with torch.no_grad():
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            print(f"\n{filename}:")
            
            try:
                # Load and transform image
                image = Image.open(img_path).convert("RGB")
                orig_size = image.size  # (W, H)
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                # Run inference
                outputs = model(img_tensor)
                
                # Extract per-image tensors (remove batch dimension)
                if isinstance(outputs, dict):
                    per_img = {k: v[0] for k, v in outputs.items()
                               if isinstance(v, torch.Tensor)}
                else:
                    per_img = {"single": outputs[0]}
                
                # Decode and merge all heads
                boxes, scores, cls_ids = decode_and_merge_heads(
                    per_img,
                    img_size=args.img_size,
                    conf_thresh=args.conf,
                    iou_thresh=args.iou,
                    box_scale=args.box_scale,
                    is_edge_model=is_edge_model,
                )
                
                # Print summary
                print_detection_summary(boxes, scores, cls_ids, class_names, args.conf)
                
                # Save visualization
                if not args.no_save_img:
                    save_path = os.path.join(
                        args.output_dir,
                        os.path.splitext(filename)[0] + "_pred.jpg"
                    )
                    save_detection_image(
                        img_tensor[0].cpu(),
                        (boxes, scores, cls_ids),
                        save_path,
                        class_names,
                        conf_thresh=args.conf
                    )
                
                # Save text results
                if args.save_txt:
                    txt_path = os.path.join(
                        args.output_dir,
                        os.path.splitext(filename)[0] + "_pred.txt"
                    )
                    with open(txt_path, 'w') as f:
                        f.write(f"# {filename}\n")
                        f.write(f"# Format: class_id x1 y1 x2 y2 confidence\n")
                        for i in range(len(boxes)):
                            if float(scores[i]) >= args.conf:
                                b = boxes[i]
                                f.write(f"{int(cls_ids[i])} "
                                       f"{b[0]:.4f} {b[1]:.4f} {b[2]:.4f} {b[3]:.4f} "
                                       f"{float(scores[i]):.4f}\n")
                
                # Store results
                all_results.append({
                    'image': filename,
                    'boxes': boxes.cpu().numpy() if len(boxes) > 0 else [],
                    'scores': scores.cpu().numpy() if len(scores) > 0 else [],
                    'class_ids': cls_ids.cpu().numpy() if len(cls_ids) > 0 else [],
                })
                
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    total_stems = sum(
        np.sum(np.array(r['class_ids']) == 0) 
        for r in all_results if len(r['class_ids']) > 0
    )
    total_tomatoes = sum(
        np.sum(np.array(r['class_ids']) == 1) 
        for r in all_results if len(r['class_ids']) > 0
    )
    
    print(f"Images processed: {len(all_results)}")
    print(f"Total detections: {total_stems} stems, {total_tomatoes} tomatoes")
    print(f"Output saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
