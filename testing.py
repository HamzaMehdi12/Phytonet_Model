"""
Run this BEFORE training to understand your dataset
and verify anchors match your objects.

Usage: python debug_anchors.py
"""

import torch
import numpy as np
import json
import os
from dataset import BotanicalDataset

def analyze_dataset(data_dir, img_size=224, num_samples=None):
    """Analyze box sizes and aspect ratios in dataset"""
    
    ds = BotanicalDataset(data_dir, img_size=img_size, mode='train')
    
    if num_samples is None:
        num_samples = len(ds)
    
    # Per-class stats
    stats = {
        0: {'widths': [], 'heights': [], 'areas': [], 'ratios': [], 'count': 0},  # stem
        1: {'widths': [], 'heights': [], 'areas': [], 'ratios': [], 'count': 0},  # tomato
    }
    
    class_names = {0: 'stem', 1: 'tomato'}
    
    for i in range(min(num_samples, len(ds))):
        _, target = ds[i]
        boxes = target['boxes']  # Normalized [0,1]
        labels = target['labels']
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            
            # Convert to pixel size
            w = (x2 - x1) * img_size
            h = (y2 - y1) * img_size
            area = w * h
            ratio = w / (h + 1e-6)  # width/height ratio
            
            cls = int(label.item())
            stats[cls]['widths'].append(w)
            stats[cls]['heights'].append(h)
            stats[cls]['areas'].append(area)
            stats[cls]['ratios'].append(ratio)
            stats[cls]['count'] += 1
    
    # Print analysis
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS - Box Size & Shape Distribution")
    print("=" * 70)
    
    for cls_id, cls_name in class_names.items():
        s = stats[cls_id]
        if s['count'] == 0:
            print(f"\n⚠️  {cls_name}: NO SAMPLES FOUND")
            continue
        
        widths = np.array(s['widths'])
        heights = np.array(s['heights'])
        ratios = np.array(s['ratios'])
        
        print(f"\n{'─' * 70}")
        print(f"  Class {cls_id}: {cls_name} ({s['count']} boxes)")
        print(f"{'─' * 70}")
        print(f"  Width  (px): min={widths.min():.1f}  mean={widths.mean():.1f}  "
              f"median={np.median(widths):.1f}  max={widths.max():.1f}")
        print(f"  Height (px): min={heights.min():.1f}  mean={heights.mean():.1f}  "
              f"median={np.median(heights):.1f}  max={heights.max():.1f}")
        print(f"  W/H Ratio:   min={ratios.min():.2f}  mean={ratios.mean():.2f}  "
              f"median={np.median(ratios):.2f}  max={ratios.max():.2f}")
        
        # Shape description
        med_ratio = np.median(ratios)
        if med_ratio < 0.5:
            shape = "TALL/THIN (vertical)"
        elif med_ratio > 2.0:
            shape = "WIDE (horizontal)"
        else:
            shape = "roughly SQUARE"
        print(f"  Shape: {shape}")
        
        # Percentiles for anchor design
        print(f"\n  Percentiles (W x H in pixels):")
        for p in [10, 25, 50, 75, 90]:
            pw = np.percentile(widths, p)
            ph = np.percentile(heights, p)
            print(f"    P{p:>2}: [{pw:>6.1f} x {ph:>6.1f}]")
    
    print(f"\n{'=' * 70}")
    
    # Now check IoU of current vs proposed anchors
    print("\n" + "=" * 70)
    print("ANCHOR IoU ANALYSIS")
    print("=" * 70)
    
    old_anchors = [[10,12], [16,18], [24,28], [32,36], [48,52], 
                   [64,68], [80,84], [96,100], [112,116]]
    
    new_anchors = [[6,20], [8,32], [12,48], [20,20], [32,32],
                   [48,48], [40,60], [60,40], [80,80]]
    
    def compute_anchor_ious(anchors, stats, img_size):
        """Compute average IoU between anchors and GT boxes"""
        class_names = {0: 'stem', 1: 'tomato'}
        
        for cls_id, cls_name in class_names.items():
            s = stats[cls_id]
            if s['count'] == 0:
                continue
            
            widths = np.array(s['widths']) / img_size  # normalize
            heights = np.array(s['heights']) / img_size
            
            print(f"\n  {cls_name}:")
            best_ious = []
            
            for idx in range(len(widths)):
                gw, gh = widths[idx], heights[idx]
                max_iou = 0
                best_anchor = None
                
                for a_idx, (aw, ah) in enumerate(anchors):
                    aw_n = aw / img_size
                    ah_n = ah / img_size
                    
                    inter_w = min(gw, aw_n)
                    inter_h = min(gh, ah_n)
                    inter = inter_w * inter_h
                    union = gw * gh + aw_n * ah_n - inter
                    iou = inter / (union + 1e-6)
                    
                    if iou > max_iou:
                        max_iou = iou
                        best_anchor = a_idx
                
                best_ious.append(max_iou)
            
            best_ious = np.array(best_ious)
            print(f"    Mean best IoU:   {best_ious.mean():.4f}")
            print(f"    Median best IoU: {np.median(best_ious):.4f}")
            print(f"    Min best IoU:    {best_ious.min():.4f}")
            print(f"    % with IoU>0.3:  {(best_ious > 0.3).mean()*100:.1f}%")
            print(f"    % with IoU>0.15: {(best_ious > 0.15).mean()*100:.1f}%")
            print(f"    % with IoU>0.1:  {(best_ious > 0.1).mean()*100:.1f}%")
    
    print("\n  --- OLD Anchors (square) ---")
    compute_anchor_ious(old_anchors, stats, img_size)
    
    print("\n  --- NEW Anchors (with thin stems) ---")
    compute_anchor_ious(new_anchors, stats, img_size)
    
    print(f"\n{'=' * 70}")
    
    return stats


if __name__ == "__main__":
    # Adjust path as needed
    train_dir = "Tomato_d/train"
    
    if not os.path.exists(train_dir):
        print(f"❌ Directory not found: {train_dir}")
        print("Change train_dir to your actual data path")
    else:
        stats = analyze_dataset(train_dir, img_size=224)
