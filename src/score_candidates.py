import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np, torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from app.demo import model, DEFAULT_THRESHOLD

VAL = "data/PhenoBench/val"          # CHANGE to your real path
CANDIDATES = [
    # dense (overlap / occlusion candidates)
    "06-05_00109_P0037822.png",
    "06-05_00046_P0037822.png",
    "06-05_00088_P0037822.png",
    "05-26_00201_P0034016.png",
    "05-26_00243_P0034016.png",
    "06-05_00066_P0037822.png",
    "06-05_00199_P0038051.png",
    "06-05_00129_P0037822.png",
    # low visibility (occlusion candidates)
    "06-05_00040_P0037991.png",
    "05-15_00215_P0030792.png",
    "06-05_00103_P0037991.png",
    "05-15_00174_P0030792.png",
    "05-26_00139_P0034282.png",
    "05-15_00232_P0030792.png",
    "06-05_00091_P0038051.png",
    "05-15_00080_P0030859.png",
    # small leaves
    "05-15_00248_P0030686.png",
    "05-15_00143_P0030953.png",
    "05-15_00219_P0030792.png",
    "05-15_00150_P0030686.png",
    "05-15_00183_P0030953.png",
    "05-15_00282_P0030792.png",
    "05-15_00239_P0030792.png",
    # darkest (shadow candidates)
    "06-05_00195_P0038051.png",
    "06-05_00216_P0038051.png",
    "06-05_00058_P0037991.png",
    "06-05_00132_P0038051.png",
    "06-05_00025_P0037822.png",
    # brightest (bright light)
    "05-26_00101_P0034282.png",
    "05-26_00080_P0034282.png",
    "05-26_00076_P0034282.png",
    "05-26_00097_P0034282.png",
    "05-26_00121_P0034282.png",
    "05-26_00059_P0034282.png",
    "05-26_00036_P0034282.png",
    "05-26_00221_P0034121.png",
]

def mean_best_iou(gt, pred_masks):
    ious = []
    for leaf_id in np.unique(gt):
        if leaf_id == 0:
            continue
        g = gt == leaf_id
        best = 0.0
        for p in pred_masks:
            union = np.logical_or(g, p).sum()
            if union:
                best = max(best, np.logical_and(g, p).sum() / union)
        ious.append(best)
    return float(np.mean(ious)) if ious else 0.0

results = []
for name in CANDIDATES:
    image = np.array(Image.open(os.path.join(VAL, "images", name)).convert("RGB"))
    gt = np.array(Image.open(os.path.join(VAL, "leaf_instances", name)))
    with torch.no_grad():
        pred = model([to_tensor(image)])[0]
    keep = pred["scores"] > DEFAULT_THRESHOLD
    masks = [np.squeeze(m) > 0.5 for m in pred["masks"][keep].cpu().numpy()]
    n_gt = len(np.unique(gt[gt > 0]))
    s = mean_best_iou(gt, masks)
    results.append((s, name))
    print(f"{name}: score {s:.3f}  predicted {len(masks)}  ground truth {n_gt}")

print("\nWorst first:")
for s, name in sorted(results):
    print(f"  {s:.3f}  {name}")
    