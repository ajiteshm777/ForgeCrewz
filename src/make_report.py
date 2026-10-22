import json, re, csv, sys, os
R = sys.argv[1] if len(sys.argv) > 1 else "results"
D = sys.argv[2] if len(sys.argv) > 2 else "docs"
os.makedirs(D, exist_ok=True)

def load(p):
    try: return json.load(open(p))
    except Exception: return None

def official(p):
    try:
        v = float(re.search(r"leaves_pq\D*([0-9.eE+-]+)", open(p).read()).group(1))
        return v / 100 if v > 1.5 else v
    except Exception: return None

runs = {}
for n in ["e14", "e10"]:
    m = load(f"{R}/{n}/metrics.json")
    if m: runs[n] = {"m": m, "off": official(f"{R}/{n}/official/all/eval_leaf_pq.yaml")}
if "e14" not in runs: sys.exit("no results/e14/metrics.json found")

def row(m, thr=0.5, fr=1.0):
    for r in m["sweep"]:
        if r["thr"] == thr and r["max_area_frac"] == fr: return r

tags = []
p = f"{R}/e14/per_image_best.csv"
if os.path.exists(p):
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k in ["n_gt", "n_pred", "tp", "fp", "fn", "pq"]: r[k] = float(r[k])
    med = sorted(r["n_gt"] for r in rows)[len(rows) // 2]
    for r in rows[:30]:
        t = []
        if r["fn"] >= 2 * max(r["fp"], 1): t.append("missed_leaves")
        if r["fp"] >= 2 * max(r["fn"], 1): t.append("false_positives")
        if r["n_gt"] >= 1.5 * med: t.append("crowded_likely_overlap")
        tags.append((r["image"], round(r["pq"], 3), int(r["n_gt"]), int(r["fp"]), int(r["fn"]), ";".join(t) or "mixed"))
    with open(f"{R}/failure_tags.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "pq", "n_gt", "fp", "fn", "tags"]); w.writerows(tags)

allm = {k: v["m"] for k, v in runs.items()}
for k, v in runs.items(): allm[k]["official_leaves_pq"] = v["off"]
json.dump(allm, open(f"{R}/metrics_all.json", "w"), indent=2)

m = runs["e14"]["m"]; b = m["best_config"]; ap = m["ap"]
chk = row(m)
off = runs["e14"]["off"]
L = []
L.append("# Validation report (B)\n")
L.append(f"Checkpoint: epoch 14 (`{m['ckpt']}`). Split: PhenoBench val, {m['n_images']} images. Device: {m['device']}. Visibility filter: {m['visibility_filter']}.\n")
L.append("Note: `leaf_visibility` was regenerated from the Hugging Face mirror (not the official download). Numbers are official-style, not guaranteed identical to the official benchmark.\n")
L.append("## Headline (my scorer, official PQ logic: mean per-image PQ, IoU>0.5, visible leaves only)\n")
L.append(f"- Best config: score threshold {b['thr']}, max area fraction {b['max_area_frac']}")
L.append(f"- PQ {b['PQ']} | SQ {b['SQ']} | RQ {b['RQ']} | TP {b['TP']} FP {b['FP']} FN {b['FN']}")
if off is not None and chk:
    L.append(f"- Official `phenobench-eval` leaves_pq at thr 0.5: {off:.4f} vs my PQ for the same predictions: {chk['PQ']} (difference {abs(off-chk['PQ']):.4f})")
else:
    L.append("- Official phenobench-eval result: not available in this run")
L.append(f"- Inference time: {m['sec_per_image']} s/image on {m['device']} (model forward pass only)\n")
L.append("## Threshold sweep (epoch 14)\n")
L.append("| thr | max area frac | PQ | SQ | RQ | TP | FP | FN |\n|---|---|---|---|---|---|---|---|")
for r in m["sweep"]:
    L.append(f"| {r['thr']} | {r['max_area_frac']} | {r['PQ']} | {r['SQ']} | {r['RQ']} | {r['TP']} | {r['FP']} | {r['FN']} |")
L.append("\n## Supplementary mask AP (all GT leaves, not visibility-filtered)\n")
L.append("| AP | AP50 | AP75 | AP-small | AP-medium | AP-large |\n|---|---|---|---|---|---|")
L.append(f"| {ap['map']} | {ap['map_50']} | {ap['map_75']} | {ap['map_small']} | {ap['map_medium']} | {ap['map_large']} |")
L.append("(-1 means no GT leaves in that size bucket.)\n")
if "e10" in runs:
    b10 = runs["e10"]["m"]["best_config"]
    L.append("## Checkpoint comparison\n\n| checkpoint | best thr | PQ | SQ | RQ |\n|---|---|---|---|---|")
    L.append(f"| epoch 10 | {b10['thr']} | {b10['PQ']} | {b10['SQ']} | {b10['RQ']} |")
    L.append(f"| epoch 14 | {b['thr']} | {b['PQ']} | {b['SQ']} | {b['RQ']} |\n")
L.append("## Failure analysis\n")
L.append("Worst 30 images by PQ are in `results/failure_tags.csv`; figure: `results/e14/worst_failures.png`.")
L.append("Tags are count-based proxies (missed leaves: FN >= 2xFP; false positives: FP >= 2xFN; crowded: GT count >= 1.5x median, likely overlap). Visual causes (shadow, soil, small leaves) need manual inspection of the figure.\n")
if tags:
    from collections import Counter
    c = Counter(x for t in tags for x in t[5].split(";"))
    L.append("Tag counts among the worst 30: " + ", ".join(f"{k}: {v}" for k, v in c.items()) + "\n")
L.append("## Reproduce\n\n```\nbash run_all.sh   # needs T4 GPU, PhenoBench_val.zip, checkpoint\n```\n")
L.append("## License\n\nDataset license: quote the README of the original PhenoBench download (CC-BY-SA 4.0 vs CC BY-NC-SA 4.0 conflict not yet resolved) and cite the PhenoBench paper.\n")
open(f"{D}/validation_report.md", "w").write("\n".join(L))
print("wrote", f"{D}/validation_report.md")
