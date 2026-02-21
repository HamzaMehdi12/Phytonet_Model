import json
import os

# Load COCO annotations from train split
anno_path = 'data/train/images/_annotations.coco.json'
if os.path.exists(anno_path):
    with open(anno_path) as f:
        data = json.load(f)
    
    # Collect all box dimensions
    widths_stem = []
    heights_stem = []
    widths_tom = []
    heights_tom = []
    
    # Get image dimensions
    img_dims = {img['id']: (img['width'], img['height']) for img in data['images']}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_dims:
            continue
        img_w, img_h = img_dims[img_id]
        
        # Get box dimensions in pixels
        x, y, w_px, h_px = ann['bbox']
        
        # Normalize to 224x224 scale (model input size)
        w_norm = (w_px / img_w) * 224
        h_norm = (h_px / img_h) * 224
        
        cat_id = ann['category_id']
        if cat_id == 1:  # stem
            widths_stem.append(w_norm)
            heights_stem.append(h_norm)
        elif cat_id == 2:  # tomato
            widths_tom.append(w_norm)
            heights_tom.append(h_norm)
    
    def median(lst):
        s = sorted(lst)
        n = len(s)
        if n % 2 == 0:
            return (s[n//2-1] + s[n//2]) / 2.0
        return s[n//2]
    
    print('='*70)
    print('=== Object Size Analysis (normalized to 224x224) ===')
    print('='*70)
    print(f'\nStems: {len(widths_stem)} annotations')
    print(f'  Width:  min={min(widths_stem):.1f} median={median(widths_stem):.1f} max={max(widths_stem):.1f}')
    print(f'  Height: min={min(heights_stem):.1f} median={median(heights_stem):.1f} max={max(heights_stem):.1f}')
    print(f'\nTomatoes: {len(widths_tom)} annotations')
    print(f'  Width:  min={min(widths_tom):.1f} median={median(widths_tom):.1f} max={max(widths_tom):.1f}')
    print(f'  Height: min={min(heights_tom):.1f} median={median(heights_tom):.1f} max={max(heights_tom):.1f}')
    
    print(f'\n' + '='*70)
    print('Current Anchors (raw): [[10,12], [16,18], [24,28], [32,36], [48,52], [64,68], [80,84], [96,100], [112,116]]')
    print('\nAnchor Coverage with scale=0.25:')
    print('  Effective sizes: [2.5x3, 4x4.5, 6x7, 8x9, 12x13, 16x17, 20x21, 24x25, 28x29]')
    print('='*70)
    print('\n⚠️  PROBLEM IDENTIFIED:')
    print('   - Stem median: ~16-20 pixels')
    print('   - Tomato median: ~24-30 pixels')  
    print('   - Current anchors with scale=0.25: 2.5-29 pixels')
    print('   - Anchors are TOO SMALL for the objects!')
    print('\n   SOLUTION: Increase box_scale from 0.25 to 1.0-1.5')
    print('='*70)
else:
    print('Annotations not found')
