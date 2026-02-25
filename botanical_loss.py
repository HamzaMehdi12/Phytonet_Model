import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """
    FIXES vs submitted version
    ──────────────────────────
    FIX 1 – class_weights no longer re-normalised.
        Old code:  class_weights = class_weights / class_weights.sum() * len(class_weights)
        [12.0, 1.0]  →  [1.846, 0.154]  — a 12x stem boost became 1.85x.
        Now the tensor is stored as-is so stems get the full 12× focal weight.

    FIX 2 – pos_weight cap raised from 5 → 50.
        With ~3 positive cells out of 441 (9×7×7) the true neg/pos ratio is ~146.
        Clamping at 5 discards 97% of the objectness signal and teaches the model
        that predicting "no object everywhere" is the safe, low-loss strategy.
        50 is still a cap (prevents gradient explosion) but lets real signal through.

    FIX 3 – return order corrected to (total, obj_loss, cls_loss, box_loss).
        botanical_loss previously returned (total, cls, obj, box).
        train.py unpacks as:  loss, obj_loss, cls_loss, box_loss = loss_fn(...)
        The swap made obj and cls display values look identical but were wrong,
        masking whether objectness was actually being trained.
    """

    def __init__(self, alpha=0.25, gamma=2.0, lambda_box=10.0, lambda_cls=4.0,
                 lambda_obj=2.0, class_weights=None, num_classes=2,
                 anchors=None, img_size=224, box_scale=1.0, label_smoothing=0.01):
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

        if anchors is None:
            anchors = [[11, 8], [17, 10], [23, 15], [29, 16], [35, 21],
                       [65, 24], [49, 60], [95, 50], [137, 71]]
        self.anchors = torch.tensor(anchors, dtype=torch.float32)

        # FIX 1 – store raw weights, do NOT re-normalise.
        self.class_weights = class_weights

    # ------------------------------------------------------------------ helpers

    def focal_loss(self, pred_logits, targets, reduction='mean'):
        assert pred_logits.shape == targets.shape, \
            f"Shape mismatch: {pred_logits.shape} vs {targets.shape}"
        bce     = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        p       = torch.sigmoid(pred_logits)
        pt      = targets * p + (1 - targets) * (1 - p)
        focal_w = (1 - pt).pow(self.gamma)
        alpha_w = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss    = alpha_w * focal_w * bce
        if   reduction == 'mean': return loss.mean()
        elif reduction == 'sum':  return loss.sum()
        return loss

    def weighted_focal_loss(self, pred_logits, targets):
        losses = self.focal_loss(pred_logits, targets, reduction='none')  # [N, C]
        if self.class_weights is not None:
            w = self.class_weights.to(pred_logits.device)
            losses = losses * w.view(1, -1)
        return losses.mean()

    def giou_loss(self, pred, target, eps=1e-7):
        if pred.numel() == 0 or target.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred   = pred.clamp(0, 1)
        target = target.clamp(0, 1)
        b1x1, b1y1, b1x2, b1y2 = pred[:,0],   pred[:,1],   pred[:,2],   pred[:,3]
        b2x1, b2y1, b2x2, b2y2 = target[:,0], target[:,1], target[:,2], target[:,3]
        b1x2 = torch.max(b1x1 + eps, b1x2);  b1y2 = torch.max(b1y1 + eps, b1y2)
        b2x2 = torch.max(b2x1 + eps, b2x2);  b2y2 = torch.max(b2y1 + eps, b2y2)
        ix1  = torch.max(b1x1, b2x1);  iy1  = torch.max(b1y1, b2y1)
        ix2  = torch.min(b1x2, b2x2);  iy2  = torch.min(b1y2, b2y2)
        inter   = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        area1   = (b1x2 - b1x1).clamp(min=eps) * (b1y2 - b1y1).clamp(min=eps)
        area2   = (b2x2 - b2x1).clamp(min=eps) * (b2y2 - b2y1).clamp(min=eps)
        union   = area1 + area2 - inter + eps
        iou     = inter / union
        cx1 = torch.min(b1x1, b2x1); cy1 = torch.min(b1y1, b2y1)
        cx2 = torch.max(b1x2, b2x2); cy2 = torch.max(b1y2, b2y2)
        c_area  = (cx2 - cx1) * (cy2 - cy1) + eps
        giou    = iou - (c_area - union) / c_area
        return (1 - giou).mean()

    # ------------------------------------------------------------------ forward

    def forward(self, predictions, targets):
        """
        predictions : dict  pred_boxes[B,N,4]  pred_obj[B,N]  pred_cls[B,N,C]
        targets     : dict  boxes[B,N,4]        obj[B,N]       cls[B,N,C]

        Returns
        -------
        total_loss, obj_loss, cls_loss, box_loss
        (FIX 3: order matches train.py unpack: loss, obj_loss, cls_loss, box_loss)
        """
        pred_boxes   = predictions["pred_boxes"]
        pred_cls     = predictions["pred_cls"]
        pred_obj     = predictions["pred_obj"]
        target_boxes = targets["boxes"]
        target_cls   = targets["cls"]
        target_obj   = targets["obj"]

        device = pred_boxes.device
        B, N, _ = pred_boxes.shape
        A = self.anchors.shape[0]
        HW = N // A
        H = W = int(HW ** 0.5)
        if A * H * W != N:
            raise ValueError(f"N={N} != A*H*W = {A}*{H}*{W}")

        pos_mask = target_obj > 0.5
        num_pos  = pos_mask.sum().clamp(min=1)

        # ── decode predicted boxes ──
        pg   = pred_boxes.view(B, A, H, W, 4)
        gy, gx = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                 torch.arange(W, device=device, dtype=torch.float32),
                                 indexing='ij')
        gx   = gx.view(1, 1, H, W, 1)
        gy   = gy.view(1, 1, H, W, 1)
        cx   = (torch.sigmoid(pg[..., 0:1]) + gx) / W
        cy   = (torch.sigmoid(pg[..., 1:2]) + gy) / H
        an   = self.anchors.to(device) / float(self.img_size)
        aw   = an[:, 0].view(A, 1, 1, 1)
        ah   = an[:, 1].view(A, 1, 1, 1)
        w    = torch.exp(pg[..., 2:3].clamp(-10, 10)) * aw * self.box_scale
        h    = torch.exp(pg[..., 3:4].clamp(-10, 10)) * ah * self.box_scale
        pred_dec = torch.stack([(cx - w/2).clamp(0,1), (cy - h/2).clamp(0,1),
                                 (cx + w/2).clamp(0,1), (cy + h/2).clamp(0,1)],
                                dim=-1).view(B, N, 4)

        # ── 1. Objectness loss ──
        n_total = target_obj.numel()
        n_pos_f = target_obj.sum().clamp(min=1)
        n_neg_f = n_total - n_pos_f
        # FIX 2: cap = 50 (was 5).  Ratio ~146 → cap 5 = 97% signal lost.
        pos_weight = (n_neg_f / n_pos_f).clamp(max=50.0)
        obj_loss = F.binary_cross_entropy_with_logits(
            pred_obj, target_obj, pos_weight=pos_weight, reduction='mean')

        # ── 2. Classification loss (positives only) ──
        if num_pos > 0:
            pcls = pred_cls[pos_mask]
            tcls = target_cls[pos_mask]
            if self.label_smoothing > 0:
                tcls = tcls * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
            cls_loss = self.weighted_focal_loss(pcls, tcls)
        else:
            cls_loss = torch.tensor(0.0, device=device)

        # ── 3. Box loss (positives only) ──
        if num_pos > 0:
            pb = pred_dec[pos_mask]
            tb = target_boxes[pos_mask]
            box_loss = 0.7 * self.giou_loss(pb, tb) + 0.3 * F.l1_loss(pb, tb)
        else:
            box_loss = torch.tensor(0.0, device=device)

        total = (self.lambda_obj * obj_loss
                 + self.lambda_cls * cls_loss
                 + self.lambda_box * box_loss)

        if torch.isnan(total) or torch.isinf(total) or total < 0:
            print(f"WARNING invalid loss: total={total.item():.4f}  "
                  f"obj={obj_loss.item():.4f}  cls={cls_loss.item():.4f}  "
                  f"box={box_loss.item():.4f}  num_pos={num_pos.item()}")
            return (torch.tensor(0.1, device=device, requires_grad=True),
                    torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device),
                    torch.tensor(0.0, device=device))

        # FIX 3: (total, obj, cls, box) — matches unpack in train.py
        return total, obj_loss, cls_loss, box_loss
