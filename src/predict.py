"""Write PhenoBench-format leaf instance predictions (owner: B).

Output layout expected by `phenobench-eval --task leaf_instances`:
    <out>/leaf_instances/<same file name as the image>.png
Each PNG is a 16-bit image, same height/width as the photo:
    pixel value 0 = background, 1..N = leaf instance IDs.

Usage (Colab):
    python src/predict.py --ckpt checkpoints/best.pth \
        --images /content/drive/MyDrive/PhenoBench/val/images \
        --out predictions/run1 --thr 0.5
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model import load_checkpoint_for_inference

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def load_model(ckpt_path, device):
    # pretrained=False inside: no weight download, accepts {"model": ...} or a plain state dict
    return load_checkpoint_for_inference(ckpt_path, device)


@torch.no_grad()
def predict_instance_map(model, img_rgb, device, score_thr):
    """Return a uint16 map [H,W]: 0 = background, 1..N = leaf IDs."""
    img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    out = model([img.to(device)])[0]
    keep = out["scores"] >= score_thr
    masks = (out["masks"][keep, 0] > 0.5).cpu().numpy()
    scores = out["scores"][keep].cpu().numpy()

    inst = np.zeros(img_rgb.shape[:2], dtype=np.uint16)
    # Paint low-confidence leaves first, so high-confidence leaves win overlaps.
    for new_id, i in enumerate(np.argsort(scores), start=1):
        inst[masks[i]] = new_id
    return inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", required=True, help="folder with val/images")
    ap.add_argument("--out", required=True, help="prediction folder (leaf_instances is created inside)")
    ap.add_argument("--thr", type=float, default=0.5, help="score threshold")
    ap.add_argument("--limit", type=int, default=None, help="only first N images (for quick tests)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)

    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        paths = paths[: args.limit]
    out_dir = Path(args.out) / "leaf_instances"
    out_dir.mkdir(parents=True, exist_ok=True)

    times = []
    for p in tqdm(paths):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        inst = predict_instance_map(model, rgb, device, args.thr)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        # The file name must match the ground-truth file name, so always save as .png
        cv2.imwrite(str(out_dir / f"{p.stem}.png"), inst)

    print(f"Saved {len(paths)} predictions to {out_dir}")
    print(f"Mean inference time: {np.mean(times):.3f} s/image on {device} "
          f"(first image included; it is usually slower)")


if __name__ == "__main__":
    main()