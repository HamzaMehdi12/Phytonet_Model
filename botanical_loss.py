import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, lambda_box=1.0, lambda_cls=1.0, 
                 lambda_obj=1.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.class_weights = class_weights
    
    def focal_classification_loss(self, pred_logits, targets):
        """Applies focal loss to class logits with class weights."""
        pred_prob = torch.sigmoid(pred_logits)
        pt = pred_prob * targets + (1 - pred_prob) * (1 - targets)
        focal_weight = self.alpha * (1 - pt).pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="none")
        
        # Apply class weights - CRITICAL FOR IMBALANCED CLASSES
        if self.class_weights is not None:
            weights = self.class_weights.view(1, 1, -1).to(pred_logits.device)
            loss = loss * weights
        
        return (focal_weight * loss).mean()
    
    def forward(self, predictions, targets):
        pred_boxes = predictions["pred_boxes"]
        pred_cls = predictions["pred_cls"]
        pred_obj = predictions["pred_obj"]
        
        target_boxes = targets["boxes"]
        target_cls = targets["cls"]
        target_obj = targets["obj"]
        
        # Classification loss with class weights
        cls_loss = self.focal_classification_loss(pred_cls, target_cls)
        
        # Objectness loss
        obj_loss = F.binary_cross_entropy_with_logits(pred_obj, target_obj)
        
        # Bounding box regression loss
        if pred_boxes.numel() > 0:
            iou_loss = 1.0 - self._bbox_iou(pred_boxes, target_boxes).mean()
            l1_loss = F.smooth_l1_loss(pred_boxes, target_boxes)
            box_loss = 0.5 * iou_loss + 0.5 * l1_loss
        else:
            box_loss = torch.tensor(0.0, device=pred_boxes.device)
        
        total_loss = (
            self.lambda_cls * cls_loss +
            self.lambda_obj * obj_loss +
            self.lambda_box * box_loss
        )
        
        return total_loss, cls_loss, obj_loss, box_loss
    
    def _bbox_iou(self, boxes1, boxes2, eps=1e-7):
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.tensor(0.0, device=boxes1.device)
        
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - inter + eps
        return inter / union