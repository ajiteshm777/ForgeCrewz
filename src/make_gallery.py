import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np, cv2, torch
import matplotlib
matplotlib.use("Agg")                      # draw to a file, no window
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor

from app.demo import model, DEFAULT_THRESHOLD
from src.visualize import overlay

IMG_DIR = "data/PhenoBench/val/images"
GT_DIR = "data/PhenoBench/val/leaf_instances"
OUT = "results/gallery.png"

ROWS = [   # (row label, filename). Swap any that don't show their condition.
    ("Overlap",       "06-05_00109_P0037822.png"),
    ("Occlusion",     "06-05_00103_P0037991.png"),
    ("Shadow",        "06-05_00195_P0038051.png"),
    ("Small leaves",  "05-15_00248_P0030686.png"),
    ("Bright light",  "05-26_00101_P0034282.png"),
    ("Failure 1",     "05-15_00215_P0030792.png"),
    ("Failure 2",     "06-05_00199_P0038051.png"),
]


def color_gt(image, inst):
    """Tint each ground-truth leaf with a random color, 50% blend."""
    rng = np.random.default_rng(0)
    colored = image.copy()
    for leaf_id in np.unique(inst):
        if leaf_id == 0:
            continue
        colored[inst == leaf_id] = rng.integers(0, 256, 3)
    return cv2.addWeighted(colored, 0.5, image, 0.5, 0)


def predict(image):
    with torch.no_grad():
        pred = model([to_tensor(image)])[0]
    keep = pred["scores"] > DEFAULT_THRESHOLD
    return (pred["masks"][keep].cpu().numpy(),
            pred["boxes"][keep].cpu().numpy(),
            pred["scores"][keep].cpu().numpy())


fig, axes = plt.subplots(len(ROWS), 3, figsize=(9, 3 * len(ROWS)))
for r, (label, name) in enumerate(ROWS):
    image = np.array(Image.open(os.path.join(IMG_DIR, name)).convert("RGB"))
    gt = np.array(Image.open(os.path.join(GT_DIR, name)))
    masks, boxes, scores = predict(image)
    print(f"{name}: predicted {len(scores)}, ground truth {len(np.unique(gt[gt > 0]))}")
    panels = [image, color_gt(image, gt), overlay(image, masks, boxes, scores)]
    for c, panel in enumerate(panels):
        ax = axes[r][c]
        ax.imshow(panel)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(["Input", "Ground truth", "Prediction"][c])
    axes[r][0].set_ylabel(f"{label}\n{name[:-4]}", fontsize=7)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig(OUT, dpi=130)
print("Saved", OUT)