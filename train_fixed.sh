#!/bin/bash
# Quick Start Training Script
# Run this to start training with CORRECT settings

cd "$(dirname "$0")"

echo "========================================="
echo "Starting Training with FIXED Configuration"
echo "========================================="
echo ""
echo "Key Fixes Applied:"
echo "  ✓ Learning Rate: 2e-4 (was 1e-5 - 20x increase)"
echo "  ✓ Box Decoding: Unified scale factors (0.15)"
echo "  ✓ Loss Weights: Balanced (box=5, obj=2, cls=1)"
echo "  ✓ Conf Threshold: 0.25 (was 0.35)"
echo "  ✓ Training Schedule: Simplified (3 phases)"
echo ""
echo "Expected Results:"
echo "  • Epoch 20: mAP50 ~15-25%"
echo "  • Epoch 100: mAP50 ~60-70%"
echo "  • Epoch 150: mAP50 ~70-75%"
echo ""
echo "========================================="
echo ""

# Check if GPU is available
python -c "import torch; print('GPU Available:', torch.cuda.is_available())" 2>/dev/null

echo ""
echo "Starting training..."
echo ""

# Run training with CORRECT settings
python train.py \
    --epochs 200 \
    --lr 2e-4 \
    --img_size 224 \
    --conf_thresh 0.25 \
    --amp \
    --batch_size 16 \
    --patience 30

echo ""
echo "========================================="
echo "Training Complete!"
echo "Check the 'weights' directory for results"
echo "========================================="
