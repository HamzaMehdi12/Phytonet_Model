import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HighAccuracyPhytoSparseNet(nn.Module):
    """
    Detection model that outputs in YOLO-style format.
    Output shape: [B, A*(5+C), H, W] where:
        - B = batch size
        - A = number of anchors (9)
        - C = number of classes (2)
        - H, W = grid dimensions
        - 5 = (tx, ty, tw, th, objectness)
    
    For 9 anchors and 2 classes: 9 * (5 + 2) = 63 output channels
    """
    def __init__(self, num_classes=2, num_anchors=9):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Lightweight backbone - downsamples to 7x7
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, 3, 2, 1),      # 224 -> 112
            ConvBlock(32, 64, 3, 2, 1),     # 112 -> 56
            ConvBlock(64, 128, 3, 2, 1),    # 56 -> 28
            ConvBlock(128, 256, 3, 2, 1),   # 28 -> 14
            ConvBlock(256, 256, 3, 2, 1)    # 14 -> 7
        )
        
        # Feature refinement
        self.neck = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256),
        )
        
        # Detection head - outputs per-anchor predictions
        # Each anchor predicts: (tx, ty, tw, th, objectness, class_logits...)
        output_channels = self.num_anchors * (5 + self.num_classes)  # 9 * 7 = 63
        
        self.head = nn.Sequential(
            ConvBlock(256, 256),
            nn.Dropout2d(p=0.3),  # Add dropout to reduce overfitting
            ConvBlock(256, 256),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(256, output_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            Output tensor [B, A*(5+C), H', W'] where:
            - A = num_anchors (9)
            - C = num_classes (2)
            - H', W' = output grid size (e.g., 7x7)
            
            The output is organized as:
            [B, A*(5+C), H, W] where for each anchor:
            - channels 0-3: bbox offsets (tx, ty, tw, th)
            - channel 4: objectness logit
            - channels 5-(5+C-1): class logits
        """
        # Backbone feature extraction
        feat = self.backbone(x)  # [B, 256, 7, 7]
        
        # Neck refinement
        feat = self.neck(feat)   # [B, 256, 7, 7]
        
        # Detection head
        output = self.head(feat)  # [B, 63, 7, 7] for 9 anchors, 2 classes
        
        return output


# Alternative: Model that returns dict format
class HighAccuracyPhytoSparseNetDict(nn.Module):
    """
    Same model but returns predictions in dict format.
    This version directly outputs the dict expected by your original loss function.
    """
    def __init__(self, num_classes=2, num_anchors=9):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Backbone
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, 3, 2, 1),
            ConvBlock(32, 64, 3, 2, 1),
            ConvBlock(64, 128, 3, 2, 1),
            ConvBlock(128, 256, 3, 2, 1),
            ConvBlock(256, 256, 3, 2, 1)
        )
        
        # Neck
        self.neck = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256),
        )
        
        # Separate heads for each component
        self.head_box = nn.Conv2d(256, num_anchors * 4, kernel_size=1)
        self.head_obj = nn.Conv2d(256, num_anchors * 1, kernel_size=1)
        self.head_cls = nn.Conv2d(256, num_anchors * num_classes, kernel_size=1)

    def forward(self, x):
        """
        Returns dict with:
        - pred_boxes: [B, A*H*W, 4]
        - pred_obj: [B, A*H*W]
        - pred_cls: [B, A*H*W, num_classes]
        """
        B = x.size(0)
        
        # Feature extraction
        feat = self.backbone(x)  # [B, 256, H, W]
        feat = self.neck(feat)
        
        H, W = feat.shape[2:]
        
        # Predictions
        pred_box = self.head_box(feat)  # [B, A*4, H, W]
        pred_obj = self.head_obj(feat)  # [B, A*1, H, W]
        pred_cls = self.head_cls(feat)  # [B, A*C, H, W]
        
        # Reshape to [B, A, H, W, ...] then flatten to [B, A*H*W, ...]
        A = self.num_anchors
        
        # Reshape boxes: [B, A*4, H, W] -> [B, A, 4, H, W] -> [B, A, H, W, 4] -> [B, A*H*W, 4]
        pred_box = pred_box.view(B, A, 4, H, W).permute(0, 1, 3, 4, 2).contiguous()
        pred_box = pred_box.view(B, A * H * W, 4)
        
        # Reshape objectness: [B, A*1, H, W] -> [B, A, H, W] -> [B, A*H*W]
        pred_obj = pred_obj.view(B, A, H, W)
        pred_obj = pred_obj.view(B, A * H * W)
        
        # Reshape classes: [B, A*C, H, W] -> [B, A, C, H, W] -> [B, A, H, W, C] -> [B, A*H*W, C]
        pred_cls = pred_cls.view(B, A, self.num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
        pred_cls = pred_cls.view(B, A * H * W, self.num_classes)
        
        return {
            "pred_boxes": pred_box,
            "pred_obj": pred_obj,
            "pred_cls": pred_cls
        }


# For testing
if __name__ == "__main__":
    # Test tensor output version
    print("Testing HighAccuracyPhytoSparseNet (tensor output)...")
    model = HighAccuracyPhytoSparseNet(num_classes=2, num_anchors=9)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify output shape
    B, C, H, W = output.shape
    expected_channels = 9 * (5 + 2)  # 9 anchors * 7 values = 63
    assert C == expected_channels, f"Expected {expected_channels} channels, got {C}"
    print(f"✓ Tensor output correct: {C} channels = 9 anchors * 7 values")
    
    print("\n" + "="*60 + "\n")
    
    # Test dict output version
    print("Testing HighAccuracyPhytoSparseNetDict (dict output)...")
    model_dict = HighAccuracyPhytoSparseNetDict(num_classes=2, num_anchors=9)
    output_dict = model_dict(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output keys: {output_dict.keys()}")
    print(f"  pred_boxes shape: {output_dict['pred_boxes'].shape}")
    print(f"  pred_obj shape: {output_dict['pred_obj'].shape}")
    print(f"  pred_cls shape: {output_dict['pred_cls'].shape}")
    
    # Verify shapes
    assert output_dict['pred_boxes'].shape == (2, 9*7*7, 4), "Box shape incorrect"
    assert output_dict['pred_obj'].shape == (2, 9*7*7), "Obj shape incorrect"
    assert output_dict['pred_cls'].shape == (2, 9*7*7, 2), "Cls shape incorrect"
    print("✓ Dict output correct!")
    
    print("\n" + "="*60)
    print("Both models working correctly!")
    print("\nUse HighAccuracyPhytoSparseNet for your training pipeline.")