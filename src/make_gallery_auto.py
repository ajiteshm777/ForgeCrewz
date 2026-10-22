import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np, cv2, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor

from app.demo import model, DEFAULT_THRESHOLD
from src.visualize import overlay

VAL = "data/PhenoBench/val"
OUT = "results/gallery.png"

# One image per condition. Swap a name if it doesn't show the condition.
CONDITIONS = [
    ("Overlap",      "06-05_00109_P0037822.png"),
    ("Occlusion",    "06-05_00103_P0037991.png"),
    ("Shadow",       "06-05_00195_P0038051.png"),
    ("Small leaves", "05-15_00248_P0030686.png"),
    ("Bright light", "05-26_00101_P0034282.png"),
]

# The 2 lowest-scoring images in this pool become Failure 1 and Failure 2.
FAILURE_POOL = [
    "05-15_00215_P0030792.png", "06-05_00199_P0038051.png",
    "05-15_00174_P0030792.png", "06-05_00046_P0037822.png",
    "05-26_00139_P0034282.png", "05-15_00080_P0030859.png",
    "06-05_00040_P0037991.png", "05-15_00232_P0030792.png",
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


cache = {}

def process(name):
    """Run the model once per image, then reuse the result."""
    if name in cache:
        return cache[name]
    image = np.array(Image.open(os.path.join(VAL, "images", name)).convert("RGB"))
    gt = np.array(Image.open(os.path.join(VAL, "leaf_instances", name)))
    with torch.no_grad():
        pred = model([to_tensor(image)])[0]
    keep = pred["scores"] > DEFAULT_THRESHOLD
    masks = pred["masks"][keep].cpu().numpy()
    boxes = pred["boxes"][keep].cpu().numpy()
    scores = pred["scores"][keep].cpu().numpy()
    score = mean_best_iou(gt, [np.squeeze(m) > 0.5 for m in masks])

    rng = np.random.default_rng(0)
    colored = image.copy()
    for leaf_id in np.unique(gt):
        if leaf_id:
            colored[gt == leaf_id] = rng.integers(0, 256, 3)
    gt_vis = cv2.addWeighted(colored, 0.5, image, 0.5, 0)

    cache[name] = (score, [image, gt_vis, overlay(image, masks, boxes, scores)])
    print(f"{name}: score {score:.3f}  predicted {len(scores)}  ground truth {len(np.unique(gt[gt > 0]))}")
    return cache[name]


rows = []
for label, name in CONDITIONS:
    score, panels = process(name)
    rows.append((f"{label}\n{name[:-4]}", panels))

pool = sorted(((process(n)[0], n) for n in FAILURE_POOL
               if n not in [c[1] for c in CONDITIONS]))
for i, (score, name) in enumerate(pool[:2], start=1):
    rows.append((f"Failure {i}\n{name[:-4]}\nscore {score:.2f}", cache[name][1]))

fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
for r, (label, panels) in enumerate(rows):
    for c, panel in enumerate(panels):
        ax = axes[r][c]
        ax.imshow(panel)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(["Input", "Ground truth", "Prediction"][c])
    axes[r][0].set_ylabel(label, fontsize=8)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig(OUT, dpi=130)
print("Saved", OUT) 