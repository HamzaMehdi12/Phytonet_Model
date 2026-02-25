#!/bin/bash
# Train PhytoNetEdge – pretrained MobileNetV3-Small backbone + 3-head FPN
# Target: 80%+ mAP@50 on the Tomato_d dataset (stems + tomatoes)
#
# FIXES APPLIED:
# - Proper per-head anchor assignment (3 anchors per head, not 9)
# - Balanced head loss weights (0.5/0.35/0.15 for small/medium/large)
# - Reduced class weights (4x for stems, was 8x)
# - Higher eval_conf_thresh (0.10 vs 0.05) to reduce false positives
# - Fixed target building to validate anchor-channel alignment

cd "$(dirname "$0")"

echo "========================================="
echo "PhytoNetEdge Training (Fixed)"
echo "========================================="
echo ""
echo "Architecture:"
echo "  Backbone : MobileNetV3-Small (ImageNet pretrained)"
echo "  Heads    : 28×28 / 14×14 / 7×7 (3 anchors each)"
echo "  Neck     : FPN + PANet bidirectional fusion"
echo ""
echo "Anchors (tuned for stems + tomatoes at 224px):"
echo "  Small  (28×28): [[10,6], [15,9], [22,14]]"
echo "  Medium (14×14): [[28,18], [38,25], [55,35]]"
echo "  Large  (7×7):   [[70,45], [95,60], [130,80]]"
echo ""
echo "Loss Configuration:"
echo "  Head weights    : small=0.50, medium=0.35, large=0.15"
echo "  Class weights   : stem=4.0, tomato=1.0"
echo "  Lambda          : box=8.0, obj=4.0, cls=1.5"
echo ""
echo "Learning Rate:"
echo "  Backbone : 5e-5 (pretrained, slow update)"
echo "  Neck+Head: 5e-4"
echo ""
echo "Expected Results:"
echo "  Epoch  20: mAP@50 ~ 35-50%"
echo "  Epoch  80: mAP@50 ~ 70-80%"
echo "  Epoch 150: mAP@50 ~ 80-88%"
echo "========================================="
echo ""

# Check GPU
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" 2>/dev/null

echo ""
echo "Starting training..."
echo ""

python train.py \
    --train_dir Tomato_d/train \
    --val_dir   Tomato_d/valid \
    --test_dir  Tomato_d/test \
    --model     edge \
    --epochs    200 \
    --lr        1e-3 \
    --img_size  224 \
    --conf_thresh      0.25 \
    --eval_conf_thresh 0.10 \
    --iou_thresh       0.45 \
    --box_scale 1.0 \
    --batch_size 16 \
    --accumulate 2 \
    --patience  40 \
    --amp \
    --output_dir weights_edge

echo ""
echo "========================================="
echo "Training complete. Check weights_edge/"
echo "========================================="