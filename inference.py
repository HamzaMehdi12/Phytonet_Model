import os
import argparse
from PIL import Image
import torch
import numpy as np
import cv2
import torchvision.transforms as T
from torchvision.ops import nms

from phytonet import HighAccuracyPhytoSparseNet


def get_infer_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def decode_predictions_advanced(pred, conf_thresh=0.45, iou_thresh=0.45,
                                anchors=None, img_size=224, max_detections=300):
    if anchors is None:
        anchors = [[10, 12], [16, 18], [24, 28], [32, 36], [48, 52],
                   [64, 68], [80, 84], [96, 100], [112, 116]]

    device = pred.device
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    A = anchors.shape[0]

    C, H, W = pred.shape

    if (C % A) == 0:
        num_classes = (C // A) - 5
    else:
        return (torch.empty((0, 4), device=device),
                torch.empty((0,), device=device),
                torch.empty((0,), dtype=torch.int64, device=device))

    if num_classes < 1:
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

    bw = torch.exp(tw_clamped) * aw * 0.5  # Optimized: 0.25 too small, 1.0 too large
    bh = torch.exp(th_clamped) * ah * 0.5

    x1 = (cx - bw / 2.0).reshape(-1)
    y1 = (cy - bh / 2.0).reshape(-1)
    x2 = (cx + bw / 2.0).reshape(-1)
    y2 = (cy + bh / 2.0).reshape(-1)

    boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, 1)

    obj_prob = torch.sigmoid(to).reshape(-1)
    cls_prob = torch.softmax(tcls, dim=-1).reshape(-1, num_classes)
    cls_scores, cls_ids = cls_prob.max(dim=-1)
    scores = torch.sqrt(obj_prob * cls_scores)

    keep_mask = scores > conf_thresh
    if keep_mask.sum() == 0:
        return (torch.empty((0, 4), device=device),
                torch.empty((0,), device=device),
                torch.empty((0,), dtype=torch.int64, device=device))

    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    class_ids = cls_ids[keep_mask]

    abs_boxes = boxes * img_size

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
        keep = nms(cls_boxes, cls_scores, iou_thresh)
        keep = keep[:max_detections]
        final_boxes.append(cls_boxes[keep])
        final_scores.append(cls_scores[keep])
        final_classes.append(torch.full((len(keep),), int(c.item()),
                                       dtype=torch.int64, device=device))

    if len(final_boxes) == 0:
        return (torch.empty((0, 4), device=device),
                torch.empty((0,), device=device),
                torch.empty((0,), dtype=torch.int64, device=device))

    final_boxes = torch.cat(final_boxes, dim=0) / float(img_size)
    final_scores = torch.cat(final_scores, dim=0)
    final_classes = torch.cat(final_classes, dim=0)

    valid_mask = (final_boxes[:, 2] > final_boxes[:, 0]) & (final_boxes[:, 3] > final_boxes[:, 1])
    return final_boxes[valid_mask], final_scores[valid_mask], final_classes[valid_mask]


def save_detection_image(image_tensor, predictions, output_path, class_names, conf_thresh=0.45):
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

            bx0, by0, bx1, by1 = pred_boxes[i]
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
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img_draw, (x1, y1 - text_height - 6), (x1 + text_width, y1), color, -1)
            cv2.putText(img_draw, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imwrite(output_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description='Inference for tomato detection')
    parser.add_argument('--weights', default='weights/best_model.pth', help='Path to model weights')
    parser.add_argument('--image', default=None, help='Single image path')
    parser.add_argument('--image_dir', default=None, help='Directory of images')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--conf', type=float, default=0.45, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--output_dir', default='weights/inference', help='Output directory')

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        raise ValueError('Provide --image or --image_dir')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HighAccuracyPhytoSparseNet(num_classes=2).to(device)

    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    class_names = {0: 'stem', 1: 'tomato'}
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = []
    if args.image:
        image_paths = [args.image]
    else:
        for fname in os.listdir(args.image_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(args.image_dir, fname))

    transform = get_infer_transform(args.img_size)

    model.eval()
    with torch.no_grad():
        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(device)

            outputs = model(img_tensor)
            if isinstance(outputs, dict):
                if 'large' in outputs:
                    output_tensor = outputs['large']
                elif 'pred_boxes' in outputs:
                    output_tensor = outputs['pred_boxes']
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
                conf_thresh=args.conf,
                iou_thresh=args.iou,
                img_size=args.img_size
            )

            save_path = os.path.join(
                args.output_dir,
                f"{os.path.splitext(os.path.basename(img_path))[0]}_pred.jpg"
            )
            save_detection_image(
                img_tensor[0].cpu(),
                (boxes, scores, class_ids),
                save_path,
                class_names,
                conf_thresh=args.conf
            )


if __name__ == '__main__':
    main()
