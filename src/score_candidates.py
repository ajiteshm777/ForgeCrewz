import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np, torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from app.demo import model, DEFAULT_THRESHOLD

VAL = "data/PhenoBench/val"          # CHANGE to your real path
CANDIDATES = [                       # paste your shortlist here
    "FILE_A.png", "FILE_B.png",
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
    