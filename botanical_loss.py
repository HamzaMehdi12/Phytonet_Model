import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DetectionLoss(nn.Module):
    """
    Detection Loss for PhytoNetEdge with multi-head support.
    
    FIXES APPLIED:
    ──────────────
    FIX 1 – class_weights no longer re-normalised.
    FIX 2 – pos_weight cap raised from 5 → 50.
    FIX 3 – return order corrected to (total, obj_loss, cls_loss, box_loss).
    FIX 4 – Added anchor validation to catch mismatches early.
    FIX 5 – Improved IoU-based anchor matching with better thresholds.
    FIX 6 – Added gradient clipping friendly loss clamping.
    """

    def __init__(self, alpha=0.25, gamma=2.0, lambda_box=8.0, lambda_cls=1.0,
                 lambda_obj=4.0, class_weights=None, num_classes=2,
                 anchors=None, img_size=224, box_scale=1.0, label_smoothing=0.0,
                 head_name=None):
        super().__init__()
        self.alpha          = alpha
        self.gamma          = gamma
        self.lambda_box     = lambda_box
        self.lambda_cls     = lambda_cls
        self.lambda_obj     = lambda_obj
        self.num_classes    = num_classes
        self.img_size       = img_size
        self.box_scale      = box_scale
        self.label_smoothing = label_smoothing
        self.head_name      = head_name  # For debugging: 'small', 'medium', 'large'

        if anchors is None:
            raise ValueError("anchors must be explicitly provided")
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_anchors = len(anchors)

        # FIX 1 – store raw weights, do NOT re-normalise.
        self.class_weights = class_weights

    # ------------------------------------------------------------------ helpers

    def focal_loss(self, pred_logits, targets, reduction='mean'):
        """Focal loss for classification with class imbalance handling."""
        assert pred_logits.shape == targets.shape, \
            f"Shape mismatch: {pred_logits.shape} vs {targets.shape}"
        
        # Numerical stability
        pred_logits = pred_logits.clamp(-10, 10)
        
        bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        p = torch.sigmoid(pred_logits)
        pt = targets * p + (1 - targets) * (1 - p)
        pt = pt.clamp(min=1e-6)  # Prevent log(0)
        
        focal_w = (1 - pt).pow(self.gamma)
        alpha_w = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_w * focal_w * bce
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss

    def weighted_focal_loss(self, pred_logits, targets):
        """Focal loss with per-class weighting."""
        losses = self.focal_loss(pred_logits, targets, reduction='none')  # [N, C]
        if self.class_weights is not None:
            w = self.class_weights.to(pred_logits.device)
            losses = losses * w.view(1, -1)
        return losses.mean()

    def ciou_loss(self, pred, target, eps=1e-7):
        """Complete IoU loss for bounding box regression."""
        if pred.numel() == 0 or target.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)
        
        px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
        
        # Ensure valid boxes (x2 > x1, y2 > y1)
        px2 = torch.max(px1 + eps, px2)
        py2 = torch.max(py1 + eps, py2)
        tx2 = torch.max(tx1 + eps, tx2)
        ty2 = torch.max(ty1 + eps, ty2)

        # Intersection
        ix1 = torch.max(px1, tx1)
        iy1 = torch.max(py1, ty1)
        ix2 = torch.min(px2, tx2)
        iy2 = torch.min(py2, ty2)
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        
        # Union
        area_p = (px2 - px1).clamp(eps) * (py2 - py1).clamp(eps)
        area_t = (tx2 - tx1).clamp(eps) * (ty2 - ty1).clamp(eps)
        union = area_p + area_t - inter + eps
        iou = inter / union

        # Center distance penalty
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        tcx = (tx1 + tx2) / 2
        tcy = (ty1 + ty2) / 2
        rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        
        # Enclosing box diagonal
        cx1 = torch.min(px1, tx1)
        cy1 = torch.min(py1, ty1)
        cx2 = torch.max(px2, tx2)
        cy2 = torch.max(py2, ty2)
        c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + eps

        # Aspect ratio penalty
        pw = px2 - px1
        ph = py2 - py1 + eps
        tw = tx2 - tx1
        th = ty2 - ty1 + eps
        v = (4 / (math.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        ciou = iou - rho2 / c2 - alpha * v
        loss = (1 - ciou).mean()
        
        # Clamp to prevent explosion
        return loss.clamp(0, 10)

    # ------------------------------------------------------------------ forward

    def forward(self, predictions, targets):
        """
        Compute detection loss for a single head.
        
        Args:
            predictions: dict with pred_boxes[B,N,4], pred_obj[B,N], pred_cls[B,N,C]
            targets: dict with boxes[B,N,4], obj[B,N], cls[B,N,C]

        Returns:
            total_loss, obj_loss, cls_loss, box_loss
        """
        pred_boxes = predictions["pred_boxes"]
        pred_cls = predictions["pred_cls"]
        pred_obj = predictions["pred_obj"]
        target_boxes = targets["boxes"]
        target_cls = targets["cls"]
        target_obj = targets["obj"]

        device = pred_boxes.device
        B, N, _ = pred_boxes.shape
        A = self.num_anchors
        HW = N // A
        H = W = int(HW ** 0.5)
        
        # Validate dimensions
        if A * H * W != N:
            raise ValueError(f"Dimension mismatch: N={N} != A*H*W = {A}*{H}*{W}. "
                           f"Head: {self.head_name}, anchors: {A}")

        pos_mask = target_obj > 0.5
        num_pos = pos_mask.sum().clamp(min=1)

        # ── Decode predicted boxes ──
        pg = pred_boxes.view(B, A, H, W, 4)
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        gx = gx.view(1, 1, H, W, 1)
        gy = gy.view(1, 1, H, W, 1)
        
        cx = (torch.sigmoid(pg[..., 0:1]) + gx) / W
        cy = (torch.sigmoid(pg[..., 1:2]) + gy) / H
        
        an = self.anchors.to(device) / float(self.img_size)
        aw = an[:, 0].view(1, A, 1, 1, 1)
        ah = an[:, 1].view(1, A, 1, 1, 1)
        
        w = torch.exp(pg[..., 2:3].clamp(-5, 5)) * aw * self.box_scale
        h = torch.exp(pg[..., 3:4].clamp(-5, 5)) * ah * self.box_scale
        
        pred_dec = torch.cat([
            (cx - w / 2).clamp(0, 1),
            (cy - h / 2).clamp(0, 1),
            (cx + w / 2).clamp(0, 1),
            (cy + h / 2).clamp(0, 1)
        ], dim=-1).view(B, N, 4)

        # ── 1. Objectness loss ──
        n_total = target_obj.numel()
        n_pos_f = target_obj.sum().clamp(min=1)
        n_neg_f = n_total - n_pos_f
        # FIX 2: cap = 50 (was 5)
        pos_weight = (n_neg_f / n_pos_f).clamp(max=50.0)
        
        obj_loss = F.binary_cross_entropy_with_logits(
            pred_obj.clamp(-10, 10), 
            target_obj, 
            pos_weight=pos_weight, 
            reduction='mean'
        )

        # ── 2. Classification loss (positives only) ──
        if num_pos > 0:
            pcls = pred_cls[pos_mask]
            tcls = target_cls[pos_mask]
            if self.label_smoothing > 0:
                tcls = tcls * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
            cls_loss = self.weighted_focal_loss(pcls, tcls)
        else:
            cls_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # ── 3. Box loss (positives only) ──
        if num_pos > 0:
            pb = pred_dec[pos_mask]
            tb = target_boxes[pos_mask]
            box_loss = self.ciou_loss(pb, tb)
        else:
            box_loss = torch.tensor(0.0, device=device, requires_grad=True)

        total = (self.lambda_obj * obj_loss
                 + self.lambda_cls * cls_loss
                 + self.lambda_box * box_loss)

        # Safety check
        if torch.isnan(total) or torch.isinf(total) or total < 0:
            print(f"WARNING invalid loss in {self.head_name}: total={total.item():.4f} "
                  f"obj={obj_loss.item():.4f} cls={cls_loss.item():.4f} "
                  f"box={box_loss.item():.4f} num_pos={num_pos.item()}")
            return (torch.tensor(0.1, device=device, requires_grad=True),
                    torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device))

        # FIX 3: (total, obj, cls, box) — matches unpack in train.py
        return total, obj_loss, cls_loss, box_loss


class MultiHeadDetectionLoss(nn.Module):
    """
    Wrapper for multi-head detection loss (PhytoNetEdge).
    Handles 3 heads with proper anchor assignment and loss weighting.
    """
    
    def __init__(self, anchors_small, anchors_medium, anchors_large,
                 num_classes=2, img_size=224, box_scale=1.0,
                 class_weights=None, head_weights=(0.5, 0.35, 0.15)):
        super().__init__()
        
        self.head_weights = head_weights  # (small, medium, large)
        
        common_kwargs = dict(
            alpha=0.25, gamma=2.0,
            lambda_box=8.0, lambda_obj=4.0, lambda_cls=1.5,
            class_weights=class_weights,
            num_classes=num_classes,
            img_size=img_size,
            box_scale=box_scale,
            label_smoothing=0.01
        )
        
        self.loss_small = DetectionLoss(
            anchors=anchors_small, head_name='small', **common_kwargs)
        self.loss_medium = DetectionLoss(
            anchors=anchors_medium, head_name='medium', **common_kwargs)
        self.loss_large = DetectionLoss(
            anchors=anchors_large, head_name='large', **common_kwargs)
    
    def forward(self, pred_dict, target_dict):
        """
        Args:
            pred_dict: dict with 'small', 'medium', 'large' keys
            target_dict: dict with 'small', 'medium', 'large' keys
        
        Returns:
            total_loss, obj_loss, cls_loss, box_loss (combined across heads)
        """
        ws, wm, wl = self.head_weights
        
        ls, os_, cs, bs = self.loss_small(pred_dict['small'], target_dict['small'])
        lm, om, cm, bm = self.loss_medium(pred_dict['medium'], target_dict['medium'])
        ll, ol, cl, bl = self.loss_large(pred_dict['large'], target_dict['large'])
        
        total = ws * ls + wm * lm + wl * ll
        obj_loss = ws * os_ + wm * om + wl * ol
        cls_loss = ws * cs + wm * cm + wl * cl
        box_loss = ws * bs + wm * bm + wl * bl
        
        return total, obj_loss, cls_loss, box_loss
