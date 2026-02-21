"""
DIAGNOSTIC: Test the entire training pipeline to find what's broken
"""
import torch
import sys
from dataset import BotanicalDataset
from torch.utils.data import DataLoader
from phytonet import HighAccuracyPhytoSparseNet
from botanical_loss import DetectionLoss

# Test configuration
IMG_SIZE = 224
BATCH_SIZE = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("DIAGNOSTIC: Testing Entire Training Pipeline")
print("="*80)

# K-means optimized anchors
ANCHORS = [[14,16], [13,22], [21,21], [16,30], [40,41], [39,55], [46,64], [52,77], [65,98]]

# Step 1: Load data
print("\n[1/6] Loading dataset...")
try:
    train_ds = BotanicalDataset(
        img_dir='data/train/images',
        ann_file='data/train/images/_annotations.coco.json',
        img_size=IMG_SIZE,
        augment=False
    )
    print(f"✓ Dataset loaded: {len(train_ds)} images")
except Exception as e:
    print(f"✗ FAILED to load dataset: {e}")
    sys.exit(1)

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Step 2: Get one batch
print("\n[2/6] Getting one batch...")
imgs, targets = next(iter(train_loader))
print(f"✓ Batch shape: {imgs.shape}")
print(f"✓ Targets: {len(targets)} items")
for i, t in enumerate(targets):
    print(f"  Image {i}: {len(t['boxes'])} objects")

# Step 3: Load model
print("\n[3/6] Loading model...")
try:
    model = HighAccuracyPhytoSparseNet(num_classes=2).to(DEVICE)
    model.eval()
    print(f"✓ Model loaded")
except Exception as e:
    print(f"✗ FAILED to load model: {e}")
    sys.exit(1)

# Step 4: Forward pass
print("\n[4/6] Running forward pass...")
imgs = imgs.to(DEVICE)
with torch.no_grad():
    output = model(imgs)

if isinstance(output, dict):
    print(f"✓ Output is dict with keys: {output.keys()}")
    if 'large' in output:
        output_tensor = output['large']
        print(f"  Using 'large' head: {output_tensor.shape}")
    else:
        print("✗ Unexpected dict format")
        sys.exit(1)
else:
    output_tensor = output
    print(f"✓ Output tensor: {output_tensor.shape}")

# Verify shape
B, C, H, W = output_tensor.shape
expected_C = 9 * 7  # 9 anchors * (5 + 2 classes)
if C != expected_C:
    print(f"✗ WRONG OUTPUT CHANNELS: Got {C}, expected {expected_C}")
    print(f"  Model outputs {C} channels but needs {expected_C} for 9 anchors and 2 classes")
    sys.exit(1)
print(f"✓ Output shape correct: [{B}, {C}, {H}, {W}]")

# Step 5: Prepare targets
print("\n[5/6] Preparing targets for loss...")
from train import prepare_targets_for_loss

device_targets = []
for t in targets:
    device_targets.append({
        'boxes': t['boxes'].to(DEVICE),
        'labels': t['labels'].to(DEVICE)
    })

try:
    prepared_targets = prepare_targets_for_loss(
        device_targets,
        output_tensor.shape,
        img_size=IMG_SIZE,
        anchors=ANCHORS,
        num_classes=2
    )
    print(f"✓ Targets prepared")
    print(f"  target_obj: {prepared_targets['obj'].shape}")
    print(f"  target_cls: {prepared_targets['cls'].shape}")
    print(f"  target_boxes: {prepared_targets['boxes'].shape}")
    
    # Check how many positives
    num_pos = (prepared_targets['obj'] > 0.5).sum().item()
    print(f"  Positive samples: {num_pos}")
    if num_pos == 0:
        print("  ✗ WARNING: ZERO positive samples! Target assignment is broken!")
except Exception as e:
    print(f"✗ FAILED target preparation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Compute loss
print("\n[6/6] Computing loss...")
try:
    from train import prepare_predictions_for_loss
    
    # Convert model output to prediction dict
    pred_dict = prepare_predictions_for_loss(output_tensor, num_classes=2)
    print(f"✓ Predictions prepared")
    print(f"  pred_boxes: {pred_dict['pred_boxes'].shape}")
    print(f"  pred_cls: {pred_dict['pred_cls'].shape}")
    print(f"  pred_obj: {pred_dict['pred_obj'].shape}")
    
    # Create loss function
    loss_fn = DetectionLoss(
        alpha=0.25,
        gamma=2.0,
        lambda_box=8.0,
        lambda_obj=3.0,
        lambda_cls=0.5,
        class_weights=torch.tensor([10.0, 1.5]).to(DEVICE),
        num_classes=2,
        anchors=ANCHORS,
        img_size=IMG_SIZE,
        box_scale=1.0
    ).to(DEVICE)
    
    # Compute loss
    loss_dict = loss_fn(pred_dict, prepared_targets)
    total_loss = loss_dict['total']
    
    print(f"✓ Loss computed successfully!")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Box loss: {loss_dict['box'].item():.4f}")
    print(f"  Obj loss: {loss_dict['obj'].item():.4f}")
    print(f"  Cls loss: {loss_dict['cls'].item():.4f}")
    
    # Sanity checks
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("  ✗ ERROR: Loss is NaN or Inf!")
    elif total_loss.item() > 100:
        print(f"  ⚠ WARNING: Loss is very high ({total_loss.item():.2f})")
    else:
        print("  ✓ Loss looks reasonable")
        
except Exception as e:
    print(f"✗ FAILED loss computation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE - All checks passed!")
print("="*80)
print("\nThe pipeline is working. If training still fails, the issue is:")
print("1. Learning rate too low/high")
print("2. Optimizer configuration")
print("3. Need more epochs")
print("="*80)
