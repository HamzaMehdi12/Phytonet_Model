import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, lambda_box=2.0, lambda_cls=4.0, 
                 lambda_obj=2.0, class_weights=None, num_classes=2,
                 anchors=None, img_size=224, box_scale=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.num_classes = num_classes
        self.img_size = img_size
        self.box_scale = box_scale
        # Default anchors if not provided (K-means optimized for Tomato_d)
        if anchors is None:
            anchors = [[11, 8], [17, 10], [23, 15], [29, 16], [35, 21],
                       [65, 24], [49, 60], [95, 50], [137, 71]]
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        
        # Normalize class weights if provided
        if class_weights is not None:
            class_weights = class_weights / class_weights.sum() * len(class_weights)
        self.class_weights = class_weights
        
    def focal_loss(self, pred_logits, targets, reduction='mean'):
        """Standard focal loss without weird weight multiplication"""
        # Ensure shapes match
        assert pred_logits.shape == targets.shape, f"Shape mismatch: {pred_logits.shape} vs {targets.shape}"
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        
        # Calculate pt for focal weight
        pred_prob = torch.sigmoid(pred_logits)
        targets = targets.float()
        pt = targets * pred_prob + (1 - targets) * (1 - pred_prob)
        
        # Focal weight
        focal_weight = (1 - pt).pow(self.gamma)
        
        # Alpha weighting
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Combine
        loss = alpha_weight * focal_weight * bce_loss
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def weighted_focal_loss(self, pred_logits, targets):
        """Focal loss with class weights applied correctly"""
        focal_losses = self.focal_loss(pred_logits, targets, reduction='none')
        
        if self.class_weights is not None:
            weights = self.class_weights.to(pred_logits.device)

            focal_losses = focal_losses * weights.view(1, -1)
        
        return focal_losses.mean()
    
    def giou_loss(self, pred_boxes, target_boxes, eps=1e-7):
        """Generalized IoU loss for bounding boxes"""
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        # Ensure valid boxes
        pred_boxes = pred_boxes.clamp(0, 1)
        target_boxes = target_boxes.clamp(0, 1)
        
        # Get coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]
        
        # Ensure x2 > x1 and y2 > y1
        b1_x2 = torch.max(b1_x1 + eps, b1_x2)
        b1_y2 = torch.max(b1_y1 + eps, b1_y2)
        b2_x2 = torch.max(b2_x1 + eps, b2_x2)
        b2_y2 = torch.max(b2_y1 + eps, b2_y2)
        
        # Intersection area
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        # Union area
        b1_area = (b1_x2 - b1_x1).clamp(min=eps) * (b1_y2 - b1_y1).clamp(min=eps)
        b2_area = (b2_x2 - b2_x1).clamp(min=eps) * (b2_y2 - b2_y1).clamp(min=eps)
        union_area = b1_area + b2_area - inter_area + eps
        
        # IoU
        iou = inter_area / union_area
        
        # Smallest enclosing box
        c_x1 = torch.min(b1_x1, b2_x1)
        c_y1 = torch.min(b1_y1, b2_y1)
        c_x2 = torch.max(b1_x2, b2_x2)
        c_y2 = torch.max(b1_y2, b2_y2)
        
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + eps
        
        # GIoU
        giou = iou - (c_area - union_area) / c_area
        
        # Return loss
        return (1 - giou).mean()
    
    def forward(self, predictions, targets):
        """
        Forward pass of detection loss
        
        Args:
            predictions: dict with 'pred_boxes', 'pred_cls', 'pred_obj'
                pred_boxes are RAW logits (tx, ty, tw, th)
            targets: dict with 'boxes', 'cls', 'obj'
                boxes are normalized [0,1] coordinates (x1, y1, x2, y2)
        """
        pred_boxes = predictions["pred_boxes"]  # [B, N, 4] - raw logits
        pred_cls = predictions["pred_cls"]      # [B, N, C]
        pred_obj = predictions["pred_obj"]      # [B, N]
        
        target_boxes = targets["boxes"]         # [B, N, 4] - normalized coords
        target_cls = targets["cls"]             # [B, N, C]
        target_obj = targets["obj"]             # [B, N]
        
        device = pred_boxes.device
        B, N, _ = pred_boxes.shape
        
        # Infer grid size from N and num anchors
        A = self.anchors.shape[0]
        H = W = int((N / A) ** 0.5)
        if A * H * W != N:
            raise ValueError(f"Invalid shape: N={N} not divisible by anchors={A} and grid={H}x{W}")
        
        # Create masks for positive samples
        pos_mask = target_obj > 0.5  # [B, N]
        num_pos = pos_mask.sum().clamp(min=1)
        
        # ==========================================
        # CONVERT PREDICTIONS TO NORMALIZED COORDS
        # ==========================================
        # Reshape pred_boxes to [B, A, H, W, 4]
        pred_boxes_grid = pred_boxes.view(B, A, H, W, 4)
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        grid_x = grid_x.view(1, 1, H, W, 1)
        grid_y = grid_y.view(1, 1, H, W, 1)
        
        # Decode boxes: tx, ty -> cx, cy (normalized to grid)
        tx = pred_boxes_grid[..., 0:1]
        ty = pred_boxes_grid[..., 1:2]
        tw = pred_boxes_grid[..., 2:3]
        th = pred_boxes_grid[..., 3:4]
        
        # Apply sigmoid to center offsets, normalize by grid size
        cx = (torch.sigmoid(tx) + grid_x) / W
        cy = (torch.sigmoid(ty) + grid_y) / H
        
        # Apply exp to width/height (they're log-scale)
        # Clamp to prevent explosion
        tw_clamped = tw.clamp(min=-10.0, max=10.0)
        th_clamped = th.clamp(min=-10.0, max=10.0)
        
        # Anchor-relative sizing (MUST match decode_predictions_advanced)
        anchors = self.anchors.to(device) / float(self.img_size)
        aw = anchors[:, 0].view(A, 1, 1, 1)
        ah = anchors[:, 1].view(A, 1, 1, 1)
        w = torch.exp(tw_clamped) * aw * self.box_scale
        h = torch.exp(th_clamped) * ah * self.box_scale
        
        # Convert to x1, y1, x2, y2 format
        x1 = (cx - w / 2.0).clamp(0, 1)
        y1 = (cy - h / 2.0).clamp(0, 1)
        x2 = (cx + w / 2.0).clamp(0, 1)
        y2 = (cy + h / 2.0).clamp(0, 1)
        
        # Reshape back to [B, N, 4]
        pred_boxes_decoded = torch.stack([x1, y1, x2, y2], dim=-1)
        pred_boxes_decoded = pred_boxes_decoded.view(B, N, 4)
        
        # ==========================================
        # 1. OBJECTNESS LOSS
        # ==========================================
        num_total = target_obj.numel()
        num_pos_obj = target_obj.sum().clamp(min=1)
        num_neg_obj = num_total - num_pos_obj
        pos_weight = (num_neg_obj / num_pos_obj).clamp(max=100.0)  # Cap at 100

        obj_loss = F.binary_cross_entropy_with_logits(
            pred_obj, 
            target_obj, 
            pos_weight=pos_weight,  # Up-weight positives dynamically
            reduction='mean'
        )
        
        # ==========================================
        # 2. CLASSIFICATION LOSS (positive samples only)
        # ==========================================
        if num_pos > 0:
            pos_pred_cls = pred_cls[pos_mask]      # [num_pos, C]
            pos_target_cls = target_cls[pos_mask]  # [num_pos, C]
            
            # Use weighted focal loss
            cls_loss = self.weighted_focal_loss(pos_pred_cls, pos_target_cls)
        else:
            cls_loss = torch.tensor(0.0, device=device)
        
        # ==========================================
        # 3. BOX LOSS (positive samples only)
        # ==========================================
        if num_pos > 0:
            pos_pred_boxes = pred_boxes_decoded[pos_mask]  # [num_pos, 4] - normalized
            pos_target_boxes = target_boxes[pos_mask]      # [num_pos, 4] - normalized
            
            # GIoU loss
            giou_loss = self.giou_loss(pos_pred_boxes, pos_target_boxes)
            
            # L1 loss for stability
            l1_loss = F.l1_loss(pos_pred_boxes, pos_target_boxes)
            
            # Combine with more weight on GIoU for better localization
            box_loss = 0.5 * giou_loss + 0.5 * l1_loss
        else:
            box_loss = torch.tensor(0.0, device=device)
        
        # ==========================================
        # COMBINE ALL LOSSES
        # ==========================================
        total_loss = (
            self.lambda_obj * obj_loss +
            self.lambda_cls * cls_loss +
            self.lambda_box * box_loss
        )
        
        # Sanity check
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss < 0:
            print("WARNING: Invalid loss detected!")
            print(f"  total_loss: {total_loss.item()}")
            print(f"  obj_loss: {obj_loss.item()}")
            print(f"  cls_loss: {cls_loss.item()}")
            print(f"  box_loss: {box_loss.item()}")
            print(f"  num_pos: {num_pos.item()}")
            
            # Emergency: return small positive loss
            return (torch.tensor(0.1, device=device, requires_grad=True),
                    torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device))
        
        return total_loss, cls_loss, obj_loss, box_loss
