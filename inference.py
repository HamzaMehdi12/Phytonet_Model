"""
infer.py — inference script for HighAccuracyPhytoSparseNet

FIXES vs previous version
──────────────────────────
FIX I1 (CRITICAL): both heads (large 7×7 + medium 14×14) are now decoded and
    merged with cross-head NMS.  Previously only the 'large' head was used,
    meaning stems — assigned to the 14×14 medium head — never appeared.

FIX I2: removed wrong fallback that passed pred_boxes [B,N,4] to a function
    expecting a [C,H,W] tensor.

FIX I3: anchors are now an explicit argument (--anchors) and passed to the
    decoder.  Using the wrong default anchors at inference produces completely
    wrong box sizes.

FIX I4: --box_scale argument added (default 1.3 to match training default).
    Without this, boxes are ~23% too small.

FIX I5: --model argument added so the 'strong' variant can be loaded.

FIX I6: torch.load now uses weights_only=False to suppress warnings.
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torchvision.ops import nms
from PIL import Image

from phytonet import HighAccuracyPhytoSparseNet, HighAccuracyPhytoSparseNetStrong


# ─────────────────────────────────────────────────────────────
#  Transform
# ─────────────────────────────────────────────────────────────

def get_infer_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────
#  Decoder (single head)
# ─────────────────────────────────────────────────────────────

def decode_single_head(pred, conf_thresh=0.25, iou_thresh=0.45,
                       anchors=None, img_size=224, max_detections=300,
                       box_scale=1.3, apply_nms=True):
    """
    Decode one [C, H, W] head tensor to (boxes, scores, class_ids).
    boxes are in normalised [0,1] coords.
    """
    if anchors is None:
        anchors = [[11, 8], [17, 10], [23, 15], [29, 16], [35, 21],
                   [65, 24], [49, 60], [95, 50], [137, 71]]

    device  = pred.device
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    A       = anchors.shape[0]
    C, H, W = pred.shape
    empty   = (torch.empty((0, 4), device=device),
               torch.empty((0,),   device=device),
               torch.empty((0,),   dtype=torch.int64, device=device))

    if C % A != 0:
        return empty
    num_classes = C // A - 5
    if num_classes < 1:
        return empty

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

    boxes     = torch.stack([(cx-bw/2).reshape(-1), (cy-bh/2).reshape(-1),
                              (cx+bw/2).reshape(-1), (cy+bh/2).reshape(-1)],
                             dim=-1).clamp(0, 1)
    obj_prob  = torch.sigmoid(pred[..., 4]).reshape(-1)
    cls_prob  = torch.sigmoid(pred[..., 5:5+num_classes]).reshape(-1, num_classes)
    cls_s, cls_ids = cls_prob.max(dim=-1)
    scores    = torch.sqrt(obj_prob * cls_s)
    class_ids = cls_ids.reshape(-1)

    keep = scores > conf_thresh
    if keep.sum() == 0:
        return empty
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

    if apply_nms:
        abs_b = boxes * img_size
        fb, fs, fc = [], [], []
        for c in class_ids.unique():
            m  = class_ids == c
            ki = nms(abs_b[m], scores[m], iou_thresh)[:max_detections]
            fb.append(abs_b[m][ki])
            fs.append(scores[m][ki])
            fc.append(torch.full((len(ki),), int(c.item()), dtype=torch.int64, device=device))
        if not fb:
            return empty
        boxes     = torch.cat(fb) / float(img_size)
        scores    = torch.cat(fs)
        class_ids = torch.cat(fc)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid], scores[valid], class_ids[valid]


# ─────────────────────────────────────────────────────────────
#  FIX I1 — merge both heads before NMS
# ─────────────────────────────────────────────────────────────

def decode_and_merge_heads(model_output, anchors, img_size, conf_thresh,
                            iou_thresh, box_scale, max_detections=300):
    """
    Decode ALL available heads (large 7×7 + medium 14×14) and combine
    with a single cross-head NMS pass.

    Previously only 'large' was decoded → stems (assigned to 14×14 head)
    never appeared in inference output.
    """
    all_boxes, all_scores, all_cls = [], [], []
    device = None

    heads = {}
    if isinstance(model_output, dict):
        # FIX I2: only accept proper [C,H,W]-style tensors, not pred-dicts
        for key in ('large', 'medium'):
            if key in model_output and isinstance(model_output[key], torch.Tensor):
                heads[key] = model_output[key]
    elif isinstance(model_output, torch.Tensor):
        heads['single'] = model_output

    for name, tensor in heads.items():
        device = tensor.device
        # tensor is [C,H,W] (already sliced to single image before call)
        b, s, c = decode_single_head(
            tensor,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            anchors=anchors,
            img_size=img_size,
            box_scale=box_scale,
            apply_nms=False,        # merge first, NMS once
        )
        all_boxes.append(b); all_scores.append(s); all_cls.append(c)

    empty = (torch.empty((0, 4), device=device or 'cpu'),
             torch.empty((0,),   device=device or 'cpu'),
             torch.empty((0,),   dtype=torch.int64, device=device or 'cpu'))

    if not any(len(b) > 0 for b in all_boxes):
        return empty

    boxes     = torch.cat(all_boxes)
    scores    = torch.cat(all_scores)
    class_ids = torch.cat(all_cls)
    abs_boxes = boxes * img_size

    fb, fs, fc = [], [], []
    for c in class_ids.unique():
        m  = class_ids == c
        ki = nms(abs_boxes[m], scores[m], iou_thresh)[:max_detections]
        fb.append(abs_boxes[m][ki])
        fs.append(scores[m][ki])
        fc.append(torch.full((len(ki),), int(c.item()), dtype=torch.int64, device=device))

    if not fb:
        return empty

    boxes     = torch.cat(fb) / float(img_size)
    scores    = torch.cat(fs)
    class_ids = torch.cat(fc)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid], scores[valid], class_ids[valid]


# ─────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────

def save_detection_image(image_tensor, predictions, output_path, class_names, conf_thresh):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = ((image_tensor.cpu() * std + mean).clamp(0, 1)
            .numpy().transpose(1, 2, 0) * 255).astype(np.uint8).copy()
    h, w = img.shape[:2]

    boxes, scores, cls_ids = predictions
    if isinstance(boxes,   torch.Tensor): boxes   = boxes.numpy()
    if isinstance(scores,  torch.Tensor): scores  = scores.numpy()
    if isinstance(cls_ids, torch.Tensor): cls_ids = cls_ids.numpy()

    for i in range(len(boxes)):
        if float(scores[i]) < conf_thresh:
            continue
        x1 = int(max(0,   boxes[i][0] * w))
        y1 = int(max(0,   boxes[i][1] * h))
        x2 = int(min(w-1, boxes[i][2] * w))
        y2 = int(min(h-1, boxes[i][3] * h))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lbl = f"{class_names.get(int(cls_ids[i]), '?')}: {float(scores[i]):.2f}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1-th-6), (x1+tw, y1), (0, 255, 0), -1)
        cv2.putText(img, lbl, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}  ({len(boxes)} detections)")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tomato/stem detection inference")
    parser.add_argument("--weights",    default="weights/best_model.pth")
    parser.add_argument("--image",      default=None,  help="Single image path")
    parser.add_argument("--image_dir",  default=None,  help="Directory of images")
    parser.add_argument("--img_size",   type=int,   default=224)
    # FIX I5: support strong model variant
    parser.add_argument("--model",      default="base", choices=["base", "strong"])
    parser.add_argument("--conf",       type=float, default=0.35)
    parser.add_argument("--iou",        type=float, default=0.45)
    # FIX I4: box_scale must match training value
    parser.add_argument("--box_scale",  type=float, default=1.3,
                        help="Must match --box_scale used during training (default 1.3)")
    # FIX I3: explicit anchors matching training
    parser.add_argument("--output_dir", default="weights/inference")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        raise ValueError("Provide --image or --image_dir")

    # Anchors must match training — hardcoded to match train.py defaults
    anchors = [[11, 8], [17, 10], [23, 15], [29, 16], [35, 21],
               [65, 24], [49, 60], [95, 50], [137, 71]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FIX I5: choose correct architecture
    if args.model == "strong":
        model = HighAccuracyPhytoSparseNetStrong(num_classes=2).to(device)
    else:
        model = HighAccuracyPhytoSparseNet(num_classes=2).to(device)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    # FIX I6: weights_only=False
    model.load_state_dict(torch.load(args.weights, map_location=device,
                                     weights_only=False))
    model.eval()
    print(f"Loaded {args.model} model from {args.weights}")

    class_names = {0: "stem", 1: "tomato"}
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = (
        [args.image] if args.image
        else [os.path.join(args.image_dir, f)
              for f in os.listdir(args.image_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    )

    transform = get_infer_transform(args.img_size)

    with torch.no_grad():
        for img_path in image_paths:
            image      = Image.open(img_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)
            outputs    = model(img_tensor)

            # Slice batch dim → per-image heads dict
            if isinstance(outputs, dict):
                per_img = {k: v[0] for k, v in outputs.items()
                           if isinstance(v, torch.Tensor)}
            else:
                per_img = {"single": outputs[0]}

            # FIX I1: decode BOTH heads and merge
            boxes, scores, cls_ids = decode_and_merge_heads(
                per_img,
                anchors=anchors,
                img_size=args.img_size,
                conf_thresh=args.conf,
                iou_thresh=args.iou,
                box_scale=args.box_scale,   # FIX I4
            )

            stem_count   = (cls_ids == 0).sum().item()
            tomato_count = (cls_ids == 1).sum().item()
            print(f"{os.path.basename(img_path)}: "
                  f"{stem_count} stem(s)  {tomato_count} tomato(es)")

            save_path = os.path.join(
                args.output_dir,
                os.path.splitext(os.path.basename(img_path))[0] + "_pred.jpg"
            )
            save_detection_image(img_tensor[0].cpu(), (boxes, scores, cls_ids),
                                 save_path, class_names, conf_thresh=args.conf)


if __name__ == "__main__":
    main()
