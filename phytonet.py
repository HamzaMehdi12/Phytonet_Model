import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ConvBlock(nn.Module):
    """Basic Conv + BN + SiLU block"""
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block with residual connection"""
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBlock(in_channels, hidden_channels, k=1, s=1, p=0)
        self.conv2 = ConvBlock(hidden_channels, out_channels, k=3, s=1, p=1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C2f(nn.Module):
    """C2f block - CSP Bottleneck with 2 convolutions (YOLOv8/v11 style)"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBlock(in_channels, 2 * hidden_channels, k=1, s=1, p=0)
        self.conv2 = ConvBlock((2 + n) * hidden_channels, out_channels, k=1, s=1, p=0)
        self.bottlenecks = nn.ModuleList(
            Bottleneck(hidden_channels, hidden_channels, shortcut, expansion=1.0) 
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.bottlenecks)
        return self.conv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (YOLOv5/v8/v11)"""
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBlock(in_channels, hidden_channels, k=1, s=1, p=0)
        self.conv2 = ConvBlock(hidden_channels * 4, out_channels, k=1, s=1, p=0)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], 1))


class ChannelAttention(nn.Module):
    """Channel attention mechanism (Squeeze-and-Excitation)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class HighAccuracyPhytoSparseNet(nn.Module):
    """
    Custom YOLOv11-inspired detection model with unique PhytoNet architecture.
    """
    def __init__(self, num_classes=2, num_anchors=9, width_mult=0.75, depth_mult=0.67):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        def make_divisible(x, divisor=8):
            return int((x + divisor / 2) // divisor * divisor)
        
        c1 = make_divisible(64 * width_mult)
        c2 = make_divisible(128 * width_mult)
        c3 = make_divisible(256 * width_mult)
        c4 = make_divisible(512 * width_mult)
        c5 = make_divisible(512 * width_mult)
        
        n1 = max(round(3 * depth_mult), 1)
        n2 = max(round(6 * depth_mult), 1)
        n3 = max(round(9 * depth_mult), 1)
        n4 = max(round(3 * depth_mult), 1)
        
        # Backbone
        self.stem = ConvBlock(3, c1, k=3, s=2, p=1)
        
        self.stage1 = nn.Sequential(
            ConvBlock(c1, c2, k=3, s=2, p=1),
            C2f(c2, c2, n=n1, shortcut=True),
            CBAM(c2)
        )
        
        self.stage2 = nn.Sequential(
            ConvBlock(c2, c3, k=3, s=2, p=1),
            C2f(c3, c3, n=n2, shortcut=True),
            CBAM(c3)
        )
        
        self.stage3 = nn.Sequential(
            ConvBlock(c3, c4, k=3, s=2, p=1),
            C2f(c4, c4, n=n3, shortcut=True),
            CBAM(c4)
        )
        
        self.stage4 = nn.Sequential(
            ConvBlock(c4, c5, k=3, s=2, p=1),
            C2f(c5, c5, n=n1, shortcut=True),
            SPPF(c5, c5, k=5),
            CBAM(c5)
        )
        
        # Neck (FPN-style)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.lateral1 = ConvBlock(c4, c5, k=1, s=1, p=0)
        self.c2f_up1 = C2f(c5 * 2, c4, n=n4, shortcut=False)
        
        self.down1 = ConvBlock(c4, c4, k=3, s=2, p=1)
        self.c2f_down1 = C2f(c4 + c5, c5, n=n4, shortcut=False)
        
        # Detection Heads
        output_channels = self.num_anchors * (5 + self.num_classes)
        
        self.head_medium = nn.Sequential(
            C2f(c4, c4, n=n4, shortcut=False),
            ConvBlock(c4, c4, k=3, s=1, p=1),
            ConvBlock(c4, c4, k=1, s=1, p=0),
            nn.Dropout2d(p=0.10),
            nn.Conv2d(c4, output_channels, kernel_size=1, stride=1, padding=0)
        )

        self.head_large = nn.Sequential(
            C2f(c5, c5, n=n4, shortcut=False),
            ConvBlock(c5, c5, k=3, s=1, p=1),
            ConvBlock(c5, c5, k=1, s=1, p=0),
            nn.Dropout2d(p=0.15),
            nn.Conv2d(c5, output_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.stem(x)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        
        p4 = self.up1(s4)
        p4 = torch.cat([p4, self.lateral1(s3)], dim=1)
        p4 = self.c2f_up1(p4)
        
        p5 = self.down1(p4)
        p5 = torch.cat([p5, s4], dim=1)
        p5 = self.c2f_down1(p5)
        
        output_medium = self.head_medium(p4)
        output_large = self.head_large(p5)
        
        return {'medium': output_medium, 'large': output_large}


class HighAccuracyPhytoSparseNetStrong(HighAccuracyPhytoSparseNet):
    """Stronger variant with increased width/depth."""
    def __init__(self, num_classes=2, num_anchors=9):
        super().__init__(
            num_classes=num_classes,
            num_anchors=num_anchors,
            width_mult=1.0,
            depth_mult=1.0
        )


# ─────────────────────────────────────────────────────────────────────────────
#  PhytoNetEdge  –  edge-deployable detector with pretrained MobileNetV3-Small
# ─────────────────────────────────────────────────────────────────────────────

class PhytoNetEdge(nn.Module):
    """
    Edge-deployable botanical detector.
    Uses a pretrained MobileNetV3-Small backbone with a 3-scale FPN + PANet head.
    
    FIXES APPLIED:
    - Proper channel dimensions for each head
    - Improved neck with better feature fusion
    - Consistent 3 anchors per head (21 output channels each)
    
    Architecture:
        Backbone: MobileNetV3-Small (ImageNet pretrained, ~2.5M params)
        Neck: FPN + PANet style bidirectional feature fusion
        Heads: 3 detection heads at 28×28, 14×14, 7×7
    
    Recommended anchors (tuned for stems + tomatoes):
        anchors_small  = [[10,6],  [15,9],  [22,14]]   # stride-8  (28×28)
        anchors_medium = [[28,18], [38,25], [55,35]]   # stride-16 (14×14)
        anchors_large  = [[70,45], [95,60], [130,80]]  # stride-32 (7×7)
    """

    # MobileNetV3-Small feature channel sizes
    _C_P3 = 24    # features[:4]  → 28×28 (stride 8)
    _C_P4 = 48    # features[4:9] → 14×14 (stride 16)
    _C_P5 = 576   # features[9:]  →  7×7  (stride 32)

    def __init__(self, num_classes: int = 2, num_anchors: int = 3,
                 neck_channels: int = 96):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.neck_channels = neck_channels

        # ── Backbone ─────────────────────────────────────────────────────────
        try:
            from torchvision.models import (mobilenet_v3_small,
                                            MobileNet_V3_Small_Weights)
            backbone = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            from torchvision.models import mobilenet_v3_small
            backbone = mobilenet_v3_small(pretrained=True)

        feats = backbone.features

        # Split into three stages for multi-scale features
        self.stage_p3 = feats[:4]    # out: 28×28, 24 ch  (stride 8)
        self.stage_p4 = feats[4:9]   # out: 14×14, 48 ch  (stride 16)
        self.stage_p5 = feats[9:]    # out:  7×7, 576 ch  (stride 32)

        nc = neck_channels

        # ── FPN Neck (Top-down path) ──────────────────────────────────────────
        self.lat5 = ConvBlock(self._C_P5, nc, k=1, s=1, p=0)
        self.lat4 = ConvBlock(self._C_P4, nc, k=1, s=1, p=0)
        self.lat3 = ConvBlock(self._C_P3, nc, k=1, s=1, p=0)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        # Top-down fusion convs
        self.fuse_p4 = ConvBlock(nc * 2, nc, k=3, s=1, p=1)
        self.fuse_p3 = ConvBlock(nc * 2, nc, k=3, s=1, p=1)

        # ── PANet (Bottom-up path) ────────────────────────────────────────────
        self.bu_conv_p3 = ConvBlock(nc, nc, k=3, s=2, p=1)  # 28→14
        self.bu_fuse_p4 = ConvBlock(nc * 2, nc, k=3, s=1, p=1)
        
        self.bu_conv_p4 = ConvBlock(nc, nc, k=3, s=2, p=1)  # 14→7
        self.bu_fuse_p5 = ConvBlock(nc * 2, nc, k=3, s=1, p=1)

        # ── Detection Heads ───────────────────────────────────────────────────
        out_ch = num_anchors * (5 + num_classes)  # 3 × 7 = 21

        # Small objects head (28×28) - uses only top-down features
        self.head_small = nn.Sequential(
            ConvBlock(nc, nc, k=3, s=1, p=1),
            ConvBlock(nc, nc, k=3, s=1, p=1),
            nn.Conv2d(nc, out_ch, kernel_size=1),
        )
        
        # Medium objects head (14×14) - uses PANet augmented features
        self.head_medium = nn.Sequential(
            ConvBlock(nc, nc, k=3, s=1, p=1),
            ConvBlock(nc, nc, k=3, s=1, p=1),
            nn.Conv2d(nc, out_ch, kernel_size=1),
        )
        
        # Large objects head (7×7) - uses PANet augmented features
        self.head_large = nn.Sequential(
            ConvBlock(nc, nc, k=3, s=1, p=1),
            ConvBlock(nc, nc, k=3, s=1, p=1),
            nn.Conv2d(nc, out_ch, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize detection head weights for better convergence."""
        for m in [self.head_small, self.head_medium, self.head_large]:
            for layer in m:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        # Initialize objectness bias for ~1% positive rate
                        # bias = -log((1-p)/p) where p=0.01
                        nn.init.constant_(layer.bias, -4.6)

    def forward(self, x: torch.Tensor):
        # ── Backbone feature extraction ───────────────────────────────────────
        p3 = self.stage_p3(x)   # [B, 24, 28, 28]
        p4 = self.stage_p4(p3)  # [B, 48, 14, 14]
        p5 = self.stage_p5(p4)  # [B, 576, 7, 7]

        # ── FPN Top-down path ─────────────────────────────────────────────────
        f5 = self.lat5(p5)  # [B, nc, 7, 7]
        f4 = self.fuse_p4(torch.cat([self.up(f5), self.lat4(p4)], dim=1))  # [B, nc, 14, 14]
        f3 = self.fuse_p3(torch.cat([self.up(f4), self.lat3(p3)], dim=1))  # [B, nc, 28, 28]

        # ── PANet Bottom-up path ──────────────────────────────────────────────
        f4_bu = self.bu_fuse_p4(torch.cat([self.bu_conv_p3(f3), f4], dim=1))  # [B, nc, 14, 14]
        f5_bu = self.bu_fuse_p5(torch.cat([self.bu_conv_p4(f4_bu), f5], dim=1))  # [B, nc, 7, 7]

        # ── Detection heads ───────────────────────────────────────────────────
        return {
            'small':  self.head_small(f3),      # [B, 21, 28, 28] - small objects
            'medium': self.head_medium(f4_bu),  # [B, 21, 14, 14] - medium objects
            'large':  self.head_large(f5_bu),   # [B, 21, 7, 7]   - large objects
        }

    @staticmethod
    def backbone_params(model):
        """Return backbone parameter groups (for differential LR)."""
        return (list(model.stage_p3.parameters()) +
                list(model.stage_p4.parameters()) +
                list(model.stage_p5.parameters()))

    @staticmethod
    def head_params(model):
        """Return neck + head parameters."""
        backbone_ids = {id(p) for p in PhytoNetEdge.backbone_params(model)}
        return [p for p in model.parameters() if id(p) not in backbone_ids]


# ─────────────────────────────────────────────────────────────────────────────
#  Testing
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 80)
    print("Testing PhytoNetEdge")
    print("=" * 80)
    
    model = PhytoNetEdge(num_classes=2, num_anchors=3)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output heads:")
    for name, tensor in output.items():
        print(f"  {name}: {tensor.shape}")
        expected_ch = 3 * (5 + 2)  # 3 anchors × 7 values = 21
        assert tensor.shape[1] == expected_ch, f"Expected {expected_ch} channels"
    
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in PhytoNetEdge.backbone_params(model))
    head_params = sum(p.numel() for p in PhytoNetEdge.head_params(model))
    
    print(f"\nModel Statistics:")
    print(f"  Total params:    {total_params:,}")
    print(f"  Backbone params: {backbone_params:,}")
    print(f"  Neck+Head params: {head_params:,}")
    print(f"  Size (FP32):     {total_params * 4 / 1024**2:.2f} MB")
    
    print("\n✓ All tests passed!") 
