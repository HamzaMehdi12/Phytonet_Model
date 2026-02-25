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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.ops import nms, box_iou
from torchmetrics.detection import MeanAveragePrecision
from thop import profile

import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb

from phytonet import HighAccuracyPhytoSparseNet, HighAccuracyPhytoSparseNetStrong
from botanical_loss import DetectionLoss
from dataset import BotanicalDataset


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
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def calculate_class_weights(dataset):
    counts = [1, 1]
    for i in range(len(dataset)):
        _, t = dataset[i]
        for lbl in t['labels']:
            if int(lbl) < len(counts):
                counts[int(lbl)] += 1
    total = sum(counts)
    w = [total / (c + 1) for c in counts]
    w = [x / sum(w) for x in w]
    print(f"Observed class balance weights: stem={w[0]:.3f} tomato={w[1]:.3f}")
    return torch.tensor(w, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
#  Model-output → loss-ready conversions
# ─────────────────────────────────────────────────────────────

def tensor_to_pred_dict(tensor, num_classes=2):
    """[B, C, H, W]  →  {pred_boxes, pred_obj, pred_cls}"""
    B, C, H, W = tensor.shape
    vpA = 5 + num_classes
    A   = C // vpA
    if C % vpA != 0:
        raise ValueError(f"Channels {C} not divisible by {vpA}")
    pred = tensor.view(B, A, vpA, H, W).permute(0, 1, 3, 4, 2).contiguous()
    pred = pred.view(B, A * H * W, vpA)
    return {'pred_boxes': pred[..., :4],
            'pred_obj':   pred[..., 4],
            'pred_cls':   pred[..., 5:]}


def prepare_predictions_for_loss(model_output, num_classes=2):
    """
    Convert raw model output to the pred-dict (or dict-of-pred-dicts) used
    by DetectionLoss.

    FIX (Bug 4 partial): previously this function stripped the 'medium' head
    and only kept 'large', so the 14×14 head never received any gradient.
    Now BOTH heads are returned and trained.
    """
    if isinstance(model_output, dict):
        if {'pred_boxes', 'pred_cls', 'pred_obj'}.issubset(model_output):
            return model_output  # already the right format
        # Both heads present → return both so DetectionLoss can train both
        if 'large' in model_output and 'medium' in model_output:
            return {
                'large':  prepare_predictions_for_loss(model_output['large'],  num_classes),
                'medium': prepare_predictions_for_loss(model_output['medium'], num_classes),
            }
        # Single head
        key = 'large' if 'large' in model_output else 'medium'
        if key in model_output:
            return prepare_predictions_for_loss(model_output[key], num_classes)
        raise TypeError(f"Dict with unexpected keys: {list(model_output.keys())}")
    if isinstance(model_output, torch.Tensor):
        return tensor_to_pred_dict(model_output, num_classes)
    raise TypeError(f"Expected dict or Tensor, got {type(model_output)}")


def prepare_targets_for_loss(raw_targets, output_shape, img_size=224,
                              anchors=None, num_classes=2):
    """
    Build target tensors for ONE detection head whose raw output has shape
    output_shape = (B, C, H, W).
    """
    if anchors is None:
        anchors = [[11, 8], [17, 10], [23, 15], [29, 16], [35, 21],
                   [65, 24], [49, 60], [95, 50], [137, 71]]

    device     = raw_targets[0]['boxes'].device
    batch_size = len(raw_targets)
    _, C, H, W = output_shape

    anchors_t    = torch.tensor(anchors, dtype=torch.float32, device=device)
    num_anchors  = anchors_t.shape[0]
    anchors_norm = anchors_t / float(img_size)

    target_obj   = torch.zeros(batch_size, num_anchors * H * W, device=device)
    target_cls   = torch.zeros(batch_size, num_anchors * H * W, num_classes, device=device)
    target_boxes = torch.zeros(batch_size, num_anchors * H * W, 4, device=device)

    total_pos = stem_pos = tom_pos = 0

    for b in range(batch_size):
        gt_boxes  = raw_targets[b]['boxes']
        gt_labels = raw_targets[b]['labels']
        if len(gt_boxes) == 0:
            continue

        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
        gt_w  =  gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h  =  gt_boxes[:, 3] - gt_boxes[:, 1]
        gx    = (gt_cx * W).long().clamp(0, W - 1)
        gy    = (gt_cy * H).long().clamp(0, H - 1)

        for i, (gxi, gyi, gwi, ghi, lbl) in enumerate(
                zip(gx, gy, gt_w, gt_h, gt_labels)):

            ious = []
            for an in anchors_norm:
                aw, ah = an[0], an[1]
                inter  = torch.min(gwi, aw) * torch.min(ghi, ah)
                union  = gwi * ghi + aw * ah - inter
                ious.append((inter / (union + 1e-6)).item())

            ious_t     = torch.tensor(ious, device=device)
            label_idx  = int(lbl.item())
            best_anch  = int(ious_t.argmax().item())

            if label_idx == 0:    # stem – low threshold, more anchors
                top_k, thresh = 2, 0.25
            else:                  # tomato
                top_k, thresh = 1, 0.40

            _, top_idx = torch.topk(ious_t, min(top_k, len(ious_t)))
            matching   = [int(a.item()) for a in top_idx if ious_t[a] >= thresh]
            if best_anch not in matching:
                matching.append(best_anch)

            lbl_c = max(0, min(label_idx, num_classes - 1))
            for ai in matching:
                idx = ai * H * W + int(gyi) * W + int(gxi)
                target_obj[b, idx]       = 1.0
                target_cls[b, idx, lbl_c] = 1.0
                target_boxes[b, idx]     = gt_boxes[i]
                total_pos += 1
                if label_idx == 0: stem_pos += 1
                else:              tom_pos  += 1

    if batch_size > 0:
        print(f"[{H}×{W}] pos/img={total_pos/batch_size:.1f} "
              f"(stem={stem_pos/batch_size:.1f} "
              f"tom={tom_pos/batch_size:.1f})", end="  ")

    return {'obj': target_obj, 'cls': target_cls, 'boxes': target_boxes}


# ─────────────────────────────────────────────────────────────
#  Decode / NMS
# ─────────────────────────────────────────────────────────────

def decode_predictions_advanced(pred, conf_thresh=0.25, iou_thresh=0.45,
                                 anchors=None, img_size=224, max_detections=300,
                                 use_class_thresholds=False, box_scale=1.0,
                                 apply_nms=True):
    if anchors is None:
        anchors = [[11, 8], [17, 10], [23, 15], [29, 16], [35, 21],
                   [65, 24], [49, 60], [95, 50], [137, 71]]

    device  = pred.device
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    A       = anchors.shape[0]
    C, H, W = pred.shape
    empty   = (torch.empty((0, 4), device=device),
               torch.empty((0,), device=device),
               torch.empty((0,), dtype=torch.int64, device=device))

    if C % A != 0: return empty
    num_classes = C // A - 5
    if num_classes < 1: return empty

    pred   = pred.view(A, 5 + num_classes, H, W).permute(0, 2, 3, 1).contiguous()
    gy, gx = torch.meshgrid(torch.arange(H, device=device),
                             torch.arange(W, device=device), indexing='ij')
    gx = gx.view(1, H, W, 1).expand(A, H, W, 1).float()
    gy = gy.view(1, H, W, 1).expand(A, H, W, 1).float()

    cx = (torch.sigmoid(pred[..., 0:1]) + gx) / W
    cy = (torch.sigmoid(pred[..., 1:2]) + gy) / H
    an = anchors / float(img_size)
    aw = an[:, 0].view(A, 1, 1, 1)
    ah = an[:, 1].view(A, 1, 1, 1)
    bw = torch.exp(pred[..., 2:3].clamp(-10, 10)) * aw * box_scale
    bh = torch.exp(pred[..., 3:4].clamp(-10, 10)) * ah * box_scale
    x1 = (cx - bw/2).reshape(-1)
    y1 = (cy - bh/2).reshape(-1)
    x2 = (cx + bw/2).reshape(-1)
    y2 = (cy + bh/2).reshape(-1)
    boxes     = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)
    obj_prob  = torch.sigmoid(pred[..., 4]).reshape(-1)
    cls_prob  = torch.sigmoid(pred[..., 5:5+num_classes]).reshape(-1, num_classes)
    cls_s, cls_ids = cls_prob.max(dim=-1)
    scores    = torch.sqrt(obj_prob * cls_s)
    class_ids = cls_ids.reshape(-1)

    if use_class_thresholds:
        ct = {0: conf_thresh * 1.0, 1: conf_thresh * 1.2}
        thr = torch.tensor([ct.get(int(c.item()), conf_thresh) for c in class_ids], device=device)
        keep = scores > thr
    else:
        keep = scores > conf_thresh

    if keep.sum() == 0: return empty
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

    if apply_nms:
        abs_b = boxes * img_size
        fb, fs, fc = [], [], []
        for c in class_ids.unique():
            m    = class_ids == c
            ki   = nms(abs_b[m], scores[m], iou_thresh)[:max_detections]
            fb.append(abs_b[m][ki])
            fs.append(scores[m][ki])
            fc.append(torch.full((len(ki),), int(c.item()), dtype=torch.int64, device=device))
        if not fb: return empty
        boxes     = torch.cat(fb) / float(img_size)
        scores    = torch.cat(fs)
        class_ids = torch.cat(fc)
    else:
        if scores.numel() > max_detections:
            ki = torch.topk(scores, max_detections).indices
            boxes, scores, class_ids = boxes[ki], scores[ki], class_ids[ki]

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid], scores[valid], class_ids[valid]


def decode_and_merge_heads(model_output, args, conf_thresh):
    """
    FIX (Bug 5): decode BOTH heads and merge with cross-head NMS.
    Previously only the 'large' (7×7) head was decoded during validation,
    meaning stems — which are small objects assigned to the 14×14 medium head
    — were completely invisible to the evaluator.
    """
    all_boxes, all_scores, all_cls = [], [], []
    device = None

    heads = {}
    if isinstance(model_output, dict):
        if 'large'  in model_output: heads['large']  = model_output['large']
        if 'medium' in model_output: heads['medium'] = model_output['medium']
    elif isinstance(model_output, torch.Tensor):
        heads['single'] = model_output
    else:
        raise TypeError(f"Unexpected model output: {type(model_output)}")

    for name, tensor in heads.items():
        device = tensor.device
        # tensor is [C,H,W] (already sliced to single image before calling)
        b, s, c = decode_predictions_advanced(
            tensor,
            conf_thresh=conf_thresh,
            iou_thresh=args.iou_thresh,
            anchors=args.anchors,
            img_size=args.img_size,
            max_detections=500,
            use_class_thresholds=False,
            box_scale=args.box_scale,
            apply_nms=False,  # merge first, then single NMS pass
        )
        all_boxes.append(b); all_scores.append(s); all_cls.append(c)

    if not any(len(b) > 0 for b in all_boxes):
        return (torch.empty((0,4), device=device),
                torch.empty((0,), device=device),
                torch.empty((0,), dtype=torch.int64, device=device))

    boxes     = torch.cat(all_boxes)
    scores    = torch.cat(all_scores)
    class_ids = torch.cat(all_cls)

    abs_boxes = boxes * args.img_size
    fb, fs, fc = [], [], []
    for c in class_ids.unique():
        m  = class_ids == c
        ki = nms(abs_boxes[m], scores[m], args.iou_thresh)[:300]
        fb.append(abs_boxes[m][ki])
        fs.append(scores[m][ki])
        fc.append(torch.full((len(ki),), int(c.item()), dtype=torch.int64, device=device))
    if not fb:
        return (torch.empty((0,4), device=device),
                torch.empty((0,), device=device),
                torch.empty((0,), dtype=torch.int64, device=device))

    boxes     = torch.cat(fb) / float(args.img_size)
    scores    = torch.cat(fs)
    class_ids = torch.cat(fc)
    valid = (boxes[:,2] > boxes[:,0]) & (boxes[:,3] > boxes[:,1])
    return boxes[valid], scores[valid], class_ids[valid]


# ─────────────────────────────────────────────────────────────
#  Visualisation helpers
# ─────────────────────────────────────────────────────────────

def draw_dashed_rectangle(img, pt1, pt2, color, thickness, dash=10):
    x1, y1 = pt1; x2, y2 = pt2
    for x in range(x1, x2, dash*2):
        cv2.line(img,(x,y1),(min(x+dash,x2),y1),color,thickness)
        cv2.line(img,(x,y2),(min(x+dash,x2),y2),color,thickness)
    for y in range(y1, y2, dash*2):
        cv2.line(img,(x1,y),(x1,min(y+dash,y2)),color,thickness)
        cv2.line(img,(x2,y),(x2,min(y+dash,y2)),color,thickness)


def save_detection_image(img_tensor, target, preds, path, class_names, args):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        img  = ((img_tensor.cpu() * std + mean).clamp(0,1)
                .numpy().transpose(1,2,0) * 255).astype(np.uint8).copy()
        h, w = img.shape[:2]

        boxes, scores, cls_ids = preds
        if isinstance(boxes,   torch.Tensor): boxes   = boxes.numpy()
        if isinstance(scores,  torch.Tensor): scores  = scores.numpy()
        if isinstance(cls_ids, torch.Tensor): cls_ids = cls_ids.numpy()

        for i in range(len(boxes)):
            if float(scores[i]) < args.eval_conf_thresh: continue
            x1=int(max(0,boxes[i][0]*w)); y1=int(max(0,boxes[i][1]*h))
            x2=int(min(w-1,boxes[i][2]*w)); y2=int(min(h-1,boxes[i][3]*h))
            if x2<=x1 or y2<=y1: continue
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            lbl=f"{class_names.get(int(cls_ids[i]),'?')}: {float(scores[i]):.2f}"
            (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
            cv2.rectangle(img,(x1,y1-th-6),(x1+tw,y1),(0,255,0),-1)
            cv2.putText(img,lbl,(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)

        if target:
            gb = target['boxes']; gl = target['labels']
            if isinstance(gb,torch.Tensor): gb=gb.numpy()
            if isinstance(gl,torch.Tensor): gl=gl.numpy()
            for i in range(len(gb)):
                x1=int(max(0,gb[i][0]*w)); y1=int(max(0,gb[i][1]*h))
                x2=int(min(w-1,gb[i][2]*w)); y2=int(min(h-1,gb[i][3]*h))
                if x2<=x1 or y2<=y1: continue
                draw_dashed_rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                lbl=f"GT:{class_names.get(int(gl[i]),'?')}"
                (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
                cv2.rectangle(img,(x1,y2),(x1+tw,y2+th+6),(0,0,255),-1)
                cv2.putText(img,lbl,(x1,y2+th),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)

        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"save_detection_image error: {e}")


def plot_training_curves(train_loss, val_metrics, output_dir):
    eps = range(1, len(train_loss)+1)
    plt.figure(); plt.plot(eps, train_loss, 'b-')
    plt.title('Train Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.savefig(os.path.join(output_dir,'train_loss.png')); plt.close()
    if val_metrics:
        plt.figure()
        plt.plot(eps,[m.get('map',0)    for m in val_metrics],'r-',label='mAP')
        plt.plot(eps,[m.get('map_50',0) for m in val_metrics],'g-',label='mAP@50')
        plt.legend(); plt.ylim(0,1)
        plt.savefig(os.path.join(output_dir,'val_metrics.png')); plt.close()


def plot_confusion_matrix(cm, class_names, path):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.tight_layout(); plt.savefig(path); plt.close()


def log_model_info(model, img_size, device, output_dir):
    try:
        x = torch.randn(1,3,img_size,img_size).to(device)
        flops,_ = profile(model, inputs=(x,), verbose=False)
        total   = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        info = {'flops':flops,'total_params':total,'trainable_params':trainable}
        with open(os.path.join(output_dir,'model_info.json'),'w') as f:
            json.dump(info,f,indent=2)
        print(f"Model: {trainable:,} trainable params  {flops/1e9:.2f} GFLOPs")
        return info
    except Exception as e:
        print(f"log_model_info: {e}"); return {}


# ─────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────

def validate_model(model, dataloader, device, class_names, args, epoch, phase='val'):
    """
    FIX (Bug 5): uses decode_and_merge_heads() so both the 7×7 and 14×14
    detection heads contribute to every prediction.  Previously only
    the 7×7 head was decoded, making stem recall appear to be zero.
    """
    model.eval()
    val_metrics = {'map':0.,'map_50':0.,'map_75':0.,'mar_100':0.,
                   'overall_precision':0.,'overall_recall':0.,'overall_f1':0.}
    for cn in class_names.values():
        val_metrics.update({f'{cn}_precision':0.,f'{cn}_recall':0.,
                             f'{cn}_f1':0.,f'{cn}_tp':0,f'{cn}_fp':0,f'{cn}_fn':0})

    all_true, all_pred = [], []
    # FIX (Bug 8 eval): using default COCO IoU range so map_75 is real
    map_metric = MeanAveragePrecision(class_metrics=True)
    map_metric.warn_on_many_detections = False

    try:
        with torch.no_grad():
            for idx, (imgs, targets) in enumerate(
                    tqdm(dataloader, desc=f"[{phase}] epoch {epoch}", leave=False)):
                imgs = imgs.to(device)
                out  = model(imgs)

                for b in range(imgs.shape[0]):
                    # Slice each head to [C,H,W] for this image
                    if isinstance(out, dict):
                        per_img = {k: v[b] for k, v in out.items()
                                   if isinstance(v, torch.Tensor)}
                    else:
                        per_img = {'single': out[b]}

                    # FIX (Bug 5): decode BOTH heads
                    boxes, scores, cls_ids = decode_and_merge_heads(
                        per_img, args, conf_thresh=args.eval_conf_thresh)

                    boxes    = boxes.cpu()
                    scores   = scores.cpu()
                    cls_ids  = cls_ids.cpu()
                    gt_boxes  = targets[b]['boxes'].cpu()
                    gt_labels = targets[b]['labels'].cpu()

                    if len(boxes) == 0:
                        boxes    = torch.empty((0,4))
                        scores   = torch.empty((0,))
                        cls_ids  = torch.empty((0,),dtype=torch.int64)

                    map_metric.update(
                        [{'boxes':boxes,'scores':scores,'labels':cls_ids}],
                        [{'boxes':gt_boxes,'labels':gt_labels}])

                    # IoU-based TP/FP matching for confusion matrix
                    if len(boxes) > 0 and len(gt_boxes) > 0:
                        ious    = box_iou(boxes, gt_boxes)
                        matched = set()
                        for g in range(len(gt_boxes)):
                            best_iou, best_p = ious[:, g].max(0)
                            if best_iou > 0.5:
                                all_true.append(gt_labels[g].item())
                                all_pred.append(cls_ids[best_p].item())
                                matched.add(best_p.item())
                        for p in range(len(boxes)):
                            if p not in matched:
                                all_true.append(-1)
                                all_pred.append(cls_ids[p].item())
                    elif len(boxes) > 0:
                        for c in cls_ids:
                            all_true.append(-1); all_pred.append(c.item())

                    if idx == 0 and b == 0 and phase == 'val':
                        save_detection_image(
                            imgs[b].cpu(), targets[b],
                            (boxes, scores, cls_ids),
                            os.path.join(args.output_dir,'detections',phase,f'epoch_{epoch}.jpg'),
                            class_names, args)

        res = map_metric.compute()
        val_metrics.update({'map':res['map'].item(),'map_50':res['map_50'].item(),
                             'map_75':res['map_75'].item(),'mar_100':res['mar_100'].item()})
    except Exception as e:
        import traceback; traceback.print_exc()

    valid = [i for i,l in enumerate(all_true) if l != -1]
    if valid:
        ft = [all_true[i] for i in valid]
        fp_l = [all_pred[i] for i in valid]
        cm   = confusion_matrix(ft, fp_l, labels=list(class_names.keys()))
        val_metrics['confusion_matrix'] = cm.tolist()
        fp_counts = {}
        for i,l in enumerate(all_true):
            if l == -1:
                fp_counts[all_pred[i]] = fp_counts.get(all_pred[i],0) + 1
        total_tp = total_fp = total_fn = 0
        for idx_c, cn in class_names.items():
            if idx_c < cm.shape[0]:
                tp   = cm[idx_c,idx_c]
                fp_c = cm[:,idx_c].sum() - tp + fp_counts.get(idx_c,0)
                fn   = cm[idx_c,:].sum() - tp
                pr   = tp/(tp+fp_c) if (tp+fp_c)>0 else 0.
                rc   = tp/(tp+fn)   if (tp+fn)>0   else 0.
                f1   = 2*pr*rc/(pr+rc) if (pr+rc)>0 else 0.
                val_metrics.update({f'{cn}_precision':pr,f'{cn}_recall':rc,f'{cn}_f1':f1,
                                     f'{cn}_tp':int(tp),f'{cn}_fp':int(fp_c),f'{cn}_fn':int(fn)})
                total_tp+=tp; total_fp+=fp_c; total_fn+=fn
        op  = total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0.
        ore = total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0.
        of1 = 2*op*ore/(op+ore) if (op+ore)>0 else 0.
        val_metrics.update({'overall_precision':op,'overall_recall':ore,'overall_f1':of1})
        if phase == 'val':
            plot_confusion_matrix(cm, list(class_names.values()),
                                  os.path.join(args.output_dir,f'cm_epoch_{epoch}.png'))
    return val_metrics


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir',  default='data_t/train')
    parser.add_argument('--val_dir',    default='data_t/valid')
    parser.add_argument('--test_dir',   default='data_t/test')
    parser.add_argument('--epochs',     type=int,   default=300)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--img_size',   type=int,   default=224)
    parser.add_argument('--model',      default='base', choices=['base','strong'])
    parser.add_argument('--box_scale',  type=float, default=1.3)
    parser.add_argument('--conf_thresh',      type=float, default=0.35)
    # FIX (Bug 8): eval_conf_thresh lowered to 0.01 (standard for mAP eval)
    parser.add_argument('--eval_conf_thresh', type=float, default=0.01)
    parser.add_argument('--iou_thresh',       type=float, default=0.45)
    parser.add_argument('--output_dir', default='weights')
    parser.add_argument('--amp',        action='store_true')
    parser.add_argument('--patience',   type=int, default=30)
    parser.add_argument('--accumulate', type=int, default=2)
    parser.add_argument('--use_wandb',  action='store_true')
    args = parser.parse_args()   # FIX (Bug 7): called ONCE only

    args.anchors = [[11, 8], [17, 10], [23, 15], [29, 16], [35, 21],
                    [65, 24], [49, 60], [95, 50], [137, 71]]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  LR={args.lr}  batch={args.batch_size}  "
          f"eval_conf={args.eval_conf_thresh}")

    for d in [args.output_dir,
              f"{args.output_dir}/checkpoints",
              f"{args.output_dir}/detections/val",
              f"{args.output_dir}/detections/test"]:
        os.makedirs(d, exist_ok=True)

    class_names = {0: "stem", 1: "tomato"}
    if args.use_wandb:
        wandb.init(project='tomato-detection', config=vars(args))

    # ── datasets ──
    train_ds = BotanicalDataset(args.train_dir, img_size=args.img_size,
                                mode='train', transform=create_augmentations(args.img_size),
                                use_mosaic=False)
    val_ds   = BotanicalDataset(args.val_dir,  img_size=args.img_size, mode='val')
    test_ds  = BotanicalDataset(args.test_dir, img_size=args.img_size, mode='test')

    # Debug: class distribution
    label_counts = {0:0, 1:0}
    for i in range(min(200,len(train_ds))):
        _, t = train_ds[i]
        for l in t['labels']: label_counts[int(l)] += 1
    print(f"Class dist (first 200): stems={label_counts[0]}  tomatoes={label_counts[1]}")

    # Weighted sampler – stems strongly up-sampled
    sample_weights = []
    for img_id in train_ds.image_ids:
        anns = train_ds.image_annotations.get(img_id, [])
        n_stem = sum(1 for a in anns if a.get('category_id') == 1)
        n_tom  = sum(1 for a in anns if a.get('category_id') == 0)
        sample_weights.append(1.0 + 12.0 * n_stem + 1.0 * n_tom)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              collate_fn=collate_fn, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    # ── model ──
    model = (HighAccuracyPhytoSparseNetStrong(num_classes=2)
             if args.model == 'strong'
             else HighAccuracyPhytoSparseNet(num_classes=2)).to(device)
    log_model_info(model, args.img_size, device, args.output_dir)

    # Confirm both heads present
    with torch.no_grad():
        _test = model(torch.randn(1,3,args.img_size,args.img_size).to(device))
    if isinstance(_test, dict):
        print(f"Model heads: { {k:tuple(v.shape) for k,v in _test.items()} }")
    else:
        print(f"Model output shape: {_test.shape}")

    # ── loss ──
    # FIX (Bug 1): pass raw [12.0, 1.0] — DetectionLoss no longer re-normalises
    class_weights_tensor = torch.tensor([12.0, 1.0], dtype=torch.float32, device=device)
    loss_fn = DetectionLoss(
        alpha=0.25, gamma=2.0,
        lambda_box=8.0, lambda_obj=2.0, lambda_cls=6.0,
        class_weights=class_weights_tensor,
        num_classes=2, anchors=args.anchors,
        img_size=args.img_size, box_scale=args.box_scale
    )

    # ── optimiser ──
    base_lr   = max(1e-6, min(args.lr, 5e-3))
    optimizer = optim.AdamW(model.parameters(), lr=base_lr,
                            weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=base_lr * 0.2)

    amp_enabled = args.amp and torch.cuda.is_available()
    scaler      = GradScaler(enabled=amp_enabled, init_scale=4096)  # created ONCE

    ema_model = deepcopy(model).eval()
    ema_decay  = 0.9999
    for p in ema_model.parameters(): p.requires_grad_(False)

    best_map50, best_epoch, patience_counter = 0., 0, 0
    train_loss_history, val_metrics_history   = [], []
    start_epoch, WARMUP = 1, 3

    # ── resume ──
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    ckpts    = sorted([f for f in os.listdir(ckpt_dir)
                       if f.startswith('epoch_') and f.endswith('.pth')])
    if ckpts:
        try:
            ck = torch.load(os.path.join(ckpt_dir, ckpts[-1]),
                            map_location=device, weights_only=False)
            model.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            scheduler.load_state_dict(ck['scheduler_state_dict'])
            start_epoch = ck['epoch'] + 1
            best_map50  = ck.get('best_map50', 0.)
            print(f"Resumed from epoch {ck['epoch']}  best_mAP50={best_map50:.4f}")
        except Exception as e:
            print(f"Checkpoint load failed ({e}), starting fresh")

    # ─────────────────────────────────────────────────────────
    #  Training loop
    # ─────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            train_ds.current_epoch = epoch

            if epoch <= WARMUP:
                wf = 0.1 + 0.9 * epoch / WARMUP
                for pg in optimizer.param_groups: pg['lr'] = base_lr * wf
                print(f"Warmup LR = {base_lr * wf:.2e}")

            model.train()
            e_loss = e_obj = e_cls = e_box = 0.
            optimizer.zero_grad()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

            for bidx, (imgs, targets) in enumerate(pbar):
                imgs = imgs.to(device, non_blocking=True)
                dev_tgts = [{'boxes': t['boxes'].to(device),
                             'labels': t['labels'].to(device)} for t in targets]

                with autocast(enabled=amp_enabled):
                    # FIX (Bug 3): single forward pass — shape inferred from output
                    outputs = model(imgs)
                    pred_dict = prepare_predictions_for_loss(outputs)

                    # Build targets for BOTH heads
                    if isinstance(outputs, dict) and 'large' in outputs and 'medium' in outputs:
                        tgt_large  = prepare_targets_for_loss(
                            dev_tgts, tuple(outputs['large'].shape),
                            img_size=args.img_size, anchors=args.anchors, num_classes=2)
                        tgt_medium = prepare_targets_for_loss(
                            dev_tgts, tuple(outputs['medium'].shape),
                            img_size=args.img_size, anchors=args.anchors, num_classes=2)

                        loss_fn.img_size  = args.img_size
                        loss_fn.box_scale = args.box_scale

                        # FIX (Bug 4): unpack order (total, obj, cls, box)
                        loss_l, obj_l, cls_l, box_l = loss_fn(pred_dict['large'],  tgt_large)
                        loss_m, obj_m, cls_m, box_m = loss_fn(pred_dict['medium'], tgt_medium)
                        loss    = loss_l + 0.4 * loss_m   # medium head contributes 40%
                        obj_loss = obj_l + 0.4 * obj_m
                        cls_loss = cls_l + 0.4 * cls_m
                        box_loss = box_l + 0.4 * box_m
                    else:
                        shape = (tuple(outputs.shape) if isinstance(outputs, torch.Tensor)
                                 else tuple(next(iter(outputs.values())).shape))
                        tgt = prepare_targets_for_loss(
                            dev_tgts, shape, img_size=args.img_size,
                            anchors=args.anchors, num_classes=2)
                        loss_fn.img_size  = args.img_size
                        loss_fn.box_scale = args.box_scale
                        loss, obj_loss, cls_loss, box_loss = loss_fn(pred_dict, tgt)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Bad loss batch {bidx}, skip")
                    optimizer.zero_grad()
                    if amp_enabled: scaler.update()
                    continue

                scaled = loss / args.accumulate
                if amp_enabled:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                e_loss += loss.item(); e_obj += obj_loss.item()
                e_cls  += cls_loss.item(); e_box += box_loss.item()

                if (bidx + 1) % args.accumulate == 0:
                    if amp_enabled: scaler.unscale_(optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        optimizer.zero_grad()
                        if amp_enabled: scaler.update()
                        continue

                    if amp_enabled:
                        old_scale = scaler.get_scale()
                        scaler.step(optimizer); scaler.update()
                        did_step = scaler.get_scale() >= old_scale
                    else:
                        optimizer.step(); did_step = True

                    if did_step:
                        with torch.no_grad():
                            for ep, mp in zip(ema_model.parameters(), model.parameters()):
                                ep.data.mul_(ema_decay).add_(mp.data, alpha=1-ema_decay)
                    optimizer.zero_grad()

                n = bidx + 1
                pbar.set_postfix({'loss':f'{e_loss/n:.3f}',
                                  'obj': f'{e_obj/n:.3f}',
                                  'cls': f'{e_cls/n:.3f}',
                                  'box': f'{e_box/n:.3f}',
                                  'lr':  f'{optimizer.param_groups[0]["lr"]:.1e}'})

            if epoch > WARMUP: scheduler.step()

            nb = len(train_loader)
            avg = e_loss/nb
            train_loss_history.append(avg)
            print(f"\nEpoch {epoch} ({time.time()-t0:.0f}s) | "
                  f"loss={avg:.4f} obj={e_obj/nb:.4f} cls={e_cls/nb:.4f} box={e_box/nb:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

            val_m = validate_model(ema_model, val_loader, device,
                                   class_names, args, epoch, 'val')
            val_metrics_history.append(val_m)
            print(f"  mAP={val_m['map']:.4f}  mAP@50={val_m['map_50']:.4f}  "
                  f"stem_R={val_m.get('stem_recall',0.):.3f}  "
                  f"tom_R={val_m.get('tomato_recall',0.):.3f}  "
                  f"F1={val_m['overall_f1']:.4f}")

            if args.use_wandb:
                wandb.log({'epoch':epoch,'train/loss':avg,
                           'val/map':val_m['map'],'val/map_50':val_m['map_50'],
                           'val/stem_recall':val_m.get('stem_recall',0.),
                           'val/tomato_recall':val_m.get('tomato_recall',0.),
                           'lr':optimizer.param_groups[0]['lr']})

            if val_m['map_50'] > best_map50:
                best_map50, best_epoch, patience_counter = val_m['map_50'], epoch, 0
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir,'best_model.pth'))
                print(f"  ✅ New best mAP@50={best_map50:.4f}")
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == args.epochs:
                torch.save({'epoch':epoch,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'scheduler_state_dict':scheduler.state_dict(),
                            'best_map50':best_map50},
                           os.path.join(ckpt_dir, f'epoch_{epoch}.pth'))

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
            torch.cuda.empty_cache(); gc.collect()

    except Exception as e:
        import traceback; traceback.print_exc()
        if args.use_wandb: wandb.alert(title="Training Failed", text=str(e))

    # ── test ──
    bp = os.path.join(args.output_dir, 'best_model.pth')
    if os.path.exists(bp):
        model.load_state_dict(torch.load(bp, map_location=device, weights_only=False))
    test_m = validate_model(model, test_loader, device, class_names, args, best_epoch, 'test')
    print(f"\nTest | mAP={test_m['map']:.4f}  mAP@50={test_m['map_50']:.4f}  "
          f"P={test_m['overall_precision']:.4f}  R={test_m['overall_recall']:.4f}")

    plot_training_curves(train_loss_history, val_metrics_history, args.output_dir)
    with open(os.path.join(args.output_dir,'training_metrics.json'),'w') as f:
        json.dump({'train_loss':train_loss_history,'val_metrics':val_metrics_history,
                   'test_metrics':test_m,'best_epoch':best_epoch,'best_map50':best_map50},
                  f, indent=2, cls=NumpyEncoder)

    try:
        qm = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
        torch.save(qm.state_dict(), os.path.join(args.output_dir,'quantized_model.pth'))
        print("Quantized model saved")
    except Exception as e:
        print(f"Quantization failed: {e}")
        torch.save(model.state_dict(), os.path.join(args.output_dir,'final_model.pth'))

    if args.use_wandb: wandb.finish()


if __name__ == '__main__':
    main()
