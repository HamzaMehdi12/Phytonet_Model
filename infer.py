import argparse
import cv2
import torch
import numpy as np
from torchvision.ops import nms
from phytonet import PhytoSparseNet

def load_image(path: str, img_size: int = 320):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {path}")
    h, w = img_bgr.shape[:2]
    img_resized = cv2.resize(img_bgr, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize
    img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_bgr, img_tensor.unsqueeze(0), (w, h)

def decode_predictions(outputs, conf_thresh=0.35, iou_thresh=0.45, num_classes=3, img_size=320):
    """Decode model predictions to bounding boxes"""
    pred = outputs[0]
    A = 3  # 3 anchors
    C, H, W = pred.shape
    
    # Reshape predictions
    pred = pred.view(A, 5 + num_classes, H, W).permute(0, 2, 3, 1)
    pred = torch.sigmoid(pred)
    
    boxes_list, scores_list, classes_list = [], [], []
    mask = pred[..., 4] > conf_thresh
    
    if not mask.any():
        return [], [], []
    
    anchors, ys, xs = mask.nonzero(as_tuple=True)
    p = pred[anchors, ys, xs]
    
    # Decode box coordinates
    grid_size_x = img_size / W
    grid_size_y = img_size / H
    px = p[:, 0]; py = p[:, 1]
    pconf = p[:, 4]
    pcls = p[:, 5:5+num_classes].argmax(dim=1)
    
    cx = (xs.float() + px) * grid_size_x
    cy = (ys.float() + py) * grid_size_y
    bw = torch.exp(p[:, 2]) * grid_size_x
    bh = torch.exp(p[:, 3]) * grid_size_y
    
    x1 = cx - bw/2
    y1 = cy - bh/2
    x2 = cx + bw/2
    y2 = cy + bh/2
    
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    scores = pconf

    # Apply NMS
    if len(boxes) > 0:
        keep = nms(boxes, scores, iou_thresh)
        boxes = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        classes = pcls[keep].cpu().numpy()
        
        boxes_list = boxes.tolist()
        scores_list = scores.tolist()
        classes_list = classes.tolist()

    return boxes_list, scores_list, classes_list

def draw_detections(image, boxes, scores, classes, class_names, orig_size, img_size=320):
    img = image.copy()
    orig_w, orig_h = orig_size
    scale_x, scale_y = orig_w / img_size, orig_h / img_size
    
    for box, score, cls in zip(boxes, scores, classes):
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)
        
        # Different colors for different classes
        color = (0, 255, 0)  # Green for tomato
        if cls == 1:  # Stem
            color = (0, 0, 255)  # Red
        elif cls == 2:  # Stem-tomato
            color = (255, 0, 0)  # Blue
            
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls]}: {score:.2f}"
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def main():
    parser = argparse.ArgumentParser(description='PhytoSparseNet Detection')
    parser.add_argument('--weights', required=True, help='Path to model weights')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    args = parser.parse_args()

    # Load model with 3 classes
    model = PhytoSparseNet(num_classes=3)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.eval()

    # Inference
    orig_img, inp, orig_size = load_image(args.image)
    with torch.no_grad():
        outputs = model(inp)

    # Decode & draw
    boxes, scores, classes = decode_predictions(outputs, conf_thresh=args.conf, iou_thresh=args.iou)
    class_names = ['tomato', 'stem', 'stem-tomato']  # 3 classes
    result = draw_detections(orig_img, boxes, scores, classes, class_names, orig_size)

    # Save and show results
    output_path = args.image.replace('.jpg', '_detection.jpg')
    cv2.imwrite(output_path, result)
    print(f"Saved detection results to {output_path}")
    
    # Resize for display if too large
    display_img = result
    if max(result.shape[:2]) > 1200:
        scale = 1200 / max(result.shape[:2])
        display_img = cv2.resize(result, (0,0), fx=scale, fy=scale)
        
    cv2.imshow('Botanical Detections', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()