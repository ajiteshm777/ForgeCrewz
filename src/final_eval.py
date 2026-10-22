import sys, os, json, time, argparse, random, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model import load_checkpoint_for_inference

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--root", default="/content/PhenoBench/val")
p.add_argument("--n", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--thrs", default="0.3,0.5,0.7,0.9")
p.add_argument("--areas", default="1.0,0.1")
p.add_argument("--save_thr", type=float, default=0.5)
p.add_argument("--out", default="results/run")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
o_ = load_checkpoint_for_inference(a.ckpt, device=dev)
model = (o_[0] if isinstance(o_, tuple) else o_).to(dev).eval()

def infer(img):
    with torch.no_grad():
        o = model([TF.to_tensor(img).to(dev)])[0]
    s = o["scores"].cpu().numpy()
    m = (o["masks"][:, 0] > 0.5).cpu().numpy()
    return s, m

def build_map(s, m, thr, frac):
    H, W = m.shape[1:] if len(m) else (1, 1)
    out = np.zeros((m.shape[1], m.shape[2]), np.uint16) if len(m) else None
    if out is None:
        return None
    areas = m.reshape(len(m), -1).sum(1)
    idx = [i for i in np.argsort(s) if s[i] >= thr and areas[i] <= frac * m.shape[1] * m.shape[2]]
    for k, i in enumerate(idx):
        out[m[i]] = k + 1
    return out

def pq_image(pred, gt, vis):
    if pred is None:
        pred = np.zeros_like(gt)
    gu, ginv = np.unique(gt.ravel(), return_inverse=True)
    pu, pinv = np.unique(pred.ravel(), return_inverse=True)
    joint = np.bincount(pinv * len(gu) + ginv, minlength=len(pu) * len(gu)).reshape(len(pu), len(gu)).astype(np.float64)
    areaP_all = joint.sum(1); areaG_all = joint.sum(0)
    rows = pu > 0; cols = gu > 0
    inter = joint[rows][:, cols]; aP = areaP_all[rows]; aG = areaG_all[cols]
    if cols.sum() == 0:
        return None
    if vis is not None:
        visv = np.zeros(len(gu)); visv[ginv] = vis.ravel() / 255.0
        visible = visv[cols] > 0.5
    else:
        visible = np.ones(cols.sum(), bool)
    if visible.sum() == 0:
        return None
    partial = ~visible
    if partial.any() and len(aP):
        drop = ((inter[:, partial] / np.maximum(aP[:, None], 1e-12)) > 0.5).any(1)
    else:
        drop = np.zeros(len(aP), bool)
    keep = ~drop
    I = inter[keep][:, visible]
    iou = I / np.maximum(aP[keep][:, None] + aG[visible][None, :] - I, 1e-12) if I.size else np.zeros((keep.sum(), visible.sum()))
    matched = np.zeros(visible.sum(), bool); tp = 0; s = 0.0
    for i in range(iou.shape[0]):
        row = np.where(matched, 0, iou[i]); j = row.argmax()
        if row[j] > 0.5:
            matched[j] = True; tp += 1; s += row[j]
    fp = int(keep.sum()) - tp; fn = int(visible.sum()) - tp
    return dict(tp=tp, fp=fp, fn=fn, n_gt=int(visible.sum()), n_pred=int(keep.sum()), iou_sum=s,
                pq=s / (tp + 0.5 * fp + 0.5 * fn), rq=tp / (tp + 0.5 * fp + 0.5 * fn))

thrs = [float(x) for x in a.thrs.split(",")]
areas_ = [float(x) for x in a.areas.split(",")]
cfgs = [(t, f) for t in thrs for f in areas_]
files = sorted(os.listdir(a.root + "/images"))
random.Random(a.seed).shuffle(files)
files = files[:a.n]
has_vis = os.path.isdir(a.root + "/leaf_visibility")
print("device:", dev, "| images:", len(files), "| visibility filter:", has_vis)

stats = {c: [] for c in cfgs}
metric = MeanAveragePrecision(iou_type="segm")
times = []
save_dir = a.out + "/pred/leaf_instances"; os.makedirs(save_dir, exist_ok=True)
for f in tqdm(files):
    img = Image.open(a.root + "/images/" + f).convert("RGB")
    gt = np.array(Image.open(a.root + "/leaf_instances/" + f)).astype(np.int64)
    vis = np.array(Image.open(a.root + "/leaf_visibility/" + f)).astype(np.float64) if has_vis else None
    if dev == "cuda": torch.cuda.synchronize()
    t = time.time(); s, m = infer(img)
    if dev == "cuda": torch.cuda.synchronize()
    times.append(time.time() - t)
    # AP (supplementary, all GT leaves)
    keep = s >= 0.05
    ids = [int(i) for i in np.unique(gt) if i > 0]
    gm = torch.from_numpy(np.stack([gt == i for i in ids])) if ids else torch.zeros((0,) + gt.shape, dtype=torch.bool)
    metric.update([{"masks": torch.from_numpy(m[keep]), "scores": torch.from_numpy(s[keep]), "labels": torch.ones(int(keep.sum()), dtype=torch.int64)}],
                  [{"masks": gm, "labels": torch.ones(len(ids), dtype=torch.int64)}])
    for (t_, f_) in cfgs:
        pm = build_map(s, m, t_, f_)
        if pm is None: pm = np.zeros_like(gt, dtype=np.uint16)
        if t_ == a.save_thr and f_ == 1.0:
            Image.fromarray(pm).save(save_dir + "/" + f)
        r = pq_image(pm.astype(np.int64), gt, vis)
        if r is not None:
            r["image"] = f; stats[(t_, f_)].append(r)

r = metric.compute()
ap = {k: round(float(r[k]), 4) for k in ["map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_100"]}
table = []
for c in cfgs:
    L = stats[c]
    tp = sum(x["tp"] for x in L)
    table.append(dict(thr=c[0], max_area_frac=c[1], n_scored_images=len(L),
        PQ=round(float(np.mean([x["pq"] for x in L])), 4), RQ=round(float(np.mean([x["rq"] for x in L])), 4),
        SQ=round(sum(x["iou_sum"] for x in L) / max(tp, 1), 4),
        TP=tp, FP=sum(x["fp"] for x in L), FN=sum(x["fn"] for x in L)))
best = max(table, key=lambda x: x["PQ"])
res = dict(ckpt=a.ckpt, device=dev, n_images=len(files), seed=a.seed, visibility_filter=has_vis,
           sec_per_image=round(float(np.mean(times)), 3), best_config=best, sweep=table, ap=ap)
json.dump(res, open(a.out + "/metrics.json", "w"), indent=2)
bc = (best["thr"], best["max_area_frac"])
with open(a.out + "/per_image_best.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["image", "n_gt", "n_pred", "tp", "fp", "fn", "pq"])
    for x in sorted(stats[bc], key=lambda x: x["pq"]):
        w.writerow([x["image"], x["n_gt"], x["n_pred"], x["tp"], x["fp"], x["fn"], round(x["pq"], 4)])
print(json.dumps(res, indent=1))

# worst-6 figure at best config
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    worst = sorted(stats[bc], key=lambda x: x["pq"])[:6]
    fig, ax = plt.subplots(len(worst), 3, figsize=(9, 3 * len(worst)))
    ax = np.atleast_2d(ax)
    def col(mp):
        z = np.ma.masked_where(mp == 0, mp % 20); return z
    for i, x in enumerate(worst):
        img = Image.open(a.root + "/images/" + x["image"]).convert("RGB")
        gt = np.array(Image.open(a.root + "/leaf_instances/" + x["image"])).astype(np.int64)
        s, m = infer(img); pm = build_map(s, m, bc[0], bc[1])
        if pm is None: pm = np.zeros_like(gt)
        ax[i, 0].imshow(img); ax[i, 0].set_title(x["image"][:22] + f" PQ={x['pq']:.2f}", fontsize=7)
        ax[i, 1].imshow(col(pm), cmap="tab20"); ax[i, 1].set_title(f"pred ({x['n_pred']})", fontsize=7)
        ax[i, 2].imshow(col(gt), cmap="tab20"); ax[i, 2].set_title(f"GT ({x['n_gt']} scored)", fontsize=7)
        for z in ax[i]: z.axis("off")
    plt.tight_layout(); plt.savefig(a.out + "/worst_failures.png", dpi=90)
except Exception as e:
    print("figure skipped:", e)
