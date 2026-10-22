"""Quick COCO-style mask AP on the PhenoBench val split (sanity check; B's evaluate.py is the official report).

Run from the repo root, AFTER training finishes (it would share the GPU and slow training otherwise):
    !pip install -q pycocotools
    !cd src && python quick_eval.py --data /content/PhenoBench \
        --ckpt /content/drive/MyDrive/ckpt_main/epoch05.pth --limit 100
"""
import argparse
import json

import numpy as np
import torch
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import PhenoBenchLeaves
from model import load_checkpoint_for_inference


def to_rle(binary_mask):
    rle = mu.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode()
    return rle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=100, help="number of val images (0 = all 772)")
    ap.add_argument("--mask_thr", type=float, default=0.5)
    ap.add_argument("--out", default="quick_eval_metrics.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PhenoBenchLeaves(args.data, "val", train=False, limit=args.limit or None)
    model = load_checkpoint_for_inference(args.ckpt, device)

    images, anns, dets, ann_id = [], [], [], 1
    for i in range(len(ds)):
        img, tgt = ds[i]
        h, w = img.shape[1:]
        images.append({"id": i, "height": h, "width": w})
        for m in tgt["masks"].numpy():
            rle = to_rle(m)
            anns.append({"id": ann_id, "image_id": i, "category_id": 1, "segmentation": rle,
                         "area": float(mu.area(rle)), "bbox": mu.toBbox(rle).tolist(), "iscrowd": 0})
            ann_id += 1

        with torch.no_grad():
            out = model([img.to(device)])[0]
        masks = (out["masks"][:, 0] > args.mask_thr).cpu().numpy()
        for m, s in zip(masks, out["scores"].cpu().numpy()):
            if m.sum() == 0:
                continue
            dets.append({"image_id": i, "category_id": 1, "segmentation": to_rle(m), "score": float(s)})
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(ds)} images done")

    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "leaf"}]}
    gt.createIndex()
    if not dets:
        print("Model produced no detections.")
        return
    ev = COCOeval(gt, gt.loadRes(dets), "segm")
    ev.evaluate(); ev.accumulate(); ev.summarize()

    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    metrics = {n: float(v) for n, v in zip(names, ev.stats)}
    json.dump(metrics, open(args.out, "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
