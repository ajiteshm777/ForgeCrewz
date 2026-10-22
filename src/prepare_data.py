"""Build <out>/{train,val}/{images,leaf_instances}/ from the HF mirror Voxel51/PhenoBench_Raw.

    pip install -U fiftyone huggingface_hub
    python src/prepare_data.py --out /content/PhenoBench
"""
import argparse
import os
import shutil

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_samples", type=int, default=None, help="for quick tests")
    args = ap.parse_args()

    from fiftyone.utils.huggingface import load_from_hub
    ds = load_from_hub("Voxel51/PhenoBench_Raw", max_samples=args.max_samples)

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, split, "leaf_instances"), exist_ok=True)

    n = 0
    for s in ds:
        if s.split not in ("train", "val") or s.ground_truth_leaves is None:
            continue
        name = os.path.basename(s.filepath)
        H, W = 1024, 1024
        inst = np.zeros((H, W), dtype=np.uint16)
        for k, det in enumerate(s.ground_truth_leaves.detections, start=1):
            if det.mask is None:
                continue
            bx, by, bw, bh = det.bounding_box  # relative [x, y, w, h]
            x0, y0 = int(round(bx * W)), int(round(by * H))
            m = det.mask
            hh, ww = min(m.shape[0], H - y0), min(m.shape[1], W - x0)
            if hh <= 0 or ww <= 0:
                continue
            region = inst[y0:y0 + hh, x0:x0 + ww]
            region[m[:hh, :ww]] = k  # later leaves overwrite on overlap
        shutil.copy(s.filepath, os.path.join(args.out, s.split, "images", name))
        Image.fromarray(inst).save(os.path.join(args.out, s.split, "leaf_instances", name))
        n += 1
        if n % 200 == 0:
            print(f"{n} samples written")
    print(f"Done: {n} samples -> {args.out}")


if __name__ == "__main__":
    main()
