"""Generate optimal anchors using K-means clustering on actual dataset object sizes"""
import json
import math
import random

def kmeans(data, k, max_iters=100):
    """Simple K-means implementation"""
    # Initialize centroids randomly
    centroids = random.sample(data, k)
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [math.sqrt((point[0]-c[0])**2 + (point[1]-c[1])**2) for c in centroids]
            closest = distances.index(min(distances))
            clusters[closest].append(point)
        
        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                avg_w = sum(p[0] for p in cluster) / len(cluster)
                avg_h = sum(p[1] for p in cluster) / len(cluster)
                new_centroids.append([avg_w, avg_h])
            else:
                new_centroids.append(random.choice(data))
        
        if new_centroids == centroids:
            break
        centroids = new_centroids
    
    return centroids

# Load annotations
with open('data/train/images/_annotations.coco.json') as f:
    data = json.load(f)

# Get image dimensions
img_dims = {img['id']: (img['width'], img['height']) for img in data['images']}

# Collect all box dimensions normalized to 224x224
boxes = []
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in img_dims:
        continue
    img_w, img_h = img_dims[img_id]
    
    # Get box dimensions
    x, y, w_px, h_px = ann['bbox']
    
    # Normalize to 224x224 scale
    w_norm = (w_px / img_w) * 224
    h_norm = (h_px / img_h) * 224
    
    boxes.append([w_norm, h_norm])

print(f"Total boxes: {len(boxes)}")
print(f"Width range: {min(b[0] for b in boxes):.1f} - {max(b[0] for b in boxes):.1f}")
print(f"Height range: {min(b[1] for b in boxes):.1f} - {max(b[1] for b in boxes):.1f}")

# Run K-means with 9 clusters (9 anchors)
anchors = kmeans(boxes, k=9, max_iters=200)

# Sort by area
anchors.sort(key=lambda x: x[0] * x[1])

print("\n" + "="*70)
print("OPTIMIZED ANCHORS (based on K-means clustering):")
print("="*70)
print("\nPython format:")
print("anchors = [")
for i, (w, h) in enumerate(anchors):
    print(f"    [{int(round(w))}, {int(round(h))}],  # Anchor {i+1}: {int(round(w))}x{int(round(h))}")
print("]")

print("\nCommand-line format:")
anchor_str = " ".join([f"{int(round(w))},{int(round(h))}" for w, h in anchors])
print(f"--anchors {anchor_str}")

# Calculate coverage
print("\n" + "="*70)
print("ANCHOR COVERAGE ANALYSIS:")
print("="*70)

for i, (aw, ah) in enumerate(anchors):
    # Count boxes well-matched by this anchor (IoU > 0.3)
    matches = 0
    for bw, bh in boxes:
        inter_w = min(bw, aw)
        inter_h = min(bh, ah)
        inter = inter_w * inter_h
        union = bw * bh + aw * ah - inter
        iou = inter / (union + 1e-6)
        if iou > 0.3:
            matches += 1
    coverage = (matches / len(boxes)) * 100
    print(f"Anchor {i+1} [{int(aw):3d}x{int(ah):3d}]: {matches:4d} boxes ({coverage:5.1f}% coverage)")

print("\n" + "="*70)
print("RECOMMENDED: Use these anchors with box_scale=1.0")
print("="*70)
