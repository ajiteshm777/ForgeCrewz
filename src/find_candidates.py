import os, glob
import numpy as np
from PIL import Image

VAL = "data/PhenoBench/val"          # CHANGE to your real path
STEP = 2                             # look at every 2nd image, to be quicker

rows = []
paths = sorted(glob.glob(os.path.join(VAL, "images", "*.png")))[::STEP]
for path in paths:
    name = os.path.basename(path)
    img = np.array(Image.open(path).convert("RGB"))
    inst = np.array(Image.open(os.path.join(VAL, "leaf_instances", name)))
    ids, areas = np.unique(inst[inst > 0], return_counts=True)
    n_leaves = len(ids)
    med_area = float(np.median(areas)) if n_leaves else 0.0
    vis_file = os.path.join(VAL, "leaf_visibility", name)
    if os.path.exists(vis_file):
        vis = np.array(Image.open(vis_file)).astype(float)
        mean_vis = vis[inst > 0].mean() / 255 if n_leaves else 1.0
    else:
        mean_vis = float("nan")
    rows.append((name, n_leaves, med_area, img.mean(), mean_vis))

def show(title, key, reverse):
    print(f"\n{title}")
    good = [r for r in rows if r[1] > 0 and r[4] == r[4]]   # drop NaN
    for r in sorted(good, key=key, reverse=reverse)[:8]:
        print(f"  {r[0]}  leaves={r[1]}  median_area={r[2]:.0f}  "
              f"brightness={r[3]:.0f}  visibility={r[4]:.2f}")

show("Dense (overlap candidates)", lambda r: r[1], True)
show("Low visibility (occlusion candidates)", lambda r: r[4], False)
show("Smallest median leaf (small leaves)", lambda r: r[2], False)
show("Darkest (shadow candidates)", lambda r: r[3], False)
show("Brightest (bright light)", lambda r: r[3], True)
