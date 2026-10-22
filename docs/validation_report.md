# Validation report (B)

Checkpoint: epoch 14 (`/content/epoch14.pth`). Split: PhenoBench val, 100 images. Device: cuda. Visibility filter: True.

Note: `leaf_visibility` was regenerated from the Hugging Face mirror (not the official download). Numbers are official-style, not guaranteed identical to the official benchmark.

## Headline (my scorer, official PQ logic: mean per-image PQ, IoU>0.5, visible leaves only)

- Best config: score threshold 0.7, max area fraction 1.0
- PQ 0.7153 | SQ 0.8437 | RQ 0.8597 | TP 3432 FP 222 FN 832
- Official phenobench-eval result: not available in this run
- Inference time: 0.324 s/image on cuda (model forward pass only)

## Threshold sweep (epoch 14)

| thr | max area frac | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| 0.3 | 1.0 | 0.7001 | 0.8368 | 0.8481 | 3601 | 556 | 663 |
| 0.3 | 0.1 | 0.7002 | 0.8368 | 0.8482 | 3601 | 555 | 663 |
| 0.5 | 1.0 | 0.7126 | 0.8399 | 0.8603 | 3529 | 345 | 735 |
| 0.5 | 0.1 | 0.7126 | 0.8399 | 0.8603 | 3529 | 345 | 735 |
| 0.7 | 1.0 | 0.7153 | 0.8437 | 0.8597 | 3432 | 222 | 832 |
| 0.7 | 0.1 | 0.7153 | 0.8437 | 0.8597 | 3432 | 222 | 832 |
| 0.9 | 1.0 | 0.7016 | 0.8527 | 0.8342 | 3174 | 99 | 1090 |
| 0.9 | 0.1 | 0.7016 | 0.8527 | 0.8342 | 3174 | 99 | 1090 |

## Supplementary mask AP (all GT leaves, not visibility-filtered)

| AP | AP50 | AP75 | AP-small | AP-medium | AP-large |
|---|---|---|---|---|---|
| 0.5675 | 0.8282 | 0.6365 | 0.3669 | 0.7279 | 0.7956 |
(-1 means no GT leaves in that size bucket.)

## Failure analysis

Worst 30 images by PQ are in `results/failure_tags.csv`; figure: `results/e14/worst_failures.png`.
Tags are count-based proxies (missed leaves: FN >= 2xFP; false positives: FP >= 2xFN; crowded: GT count >= 1.5x median, likely overlap). Visual causes (shadow, soil, small leaves) need manual inspection of the figure.

Tag counts among the worst 30: missed_leaves: 25, mixed: 4, false_positives: 1, crowded_likely_overlap: 5

## Reproduce

```
bash run_all.sh   # needs T4 GPU, PhenoBench_val.zip, checkpoint
```

## License

Dataset license: quote the README of the original PhenoBench download (CC-BY-SA 4.0 vs CC BY-NC-SA 4.0 conflict not yet resolved) and cite the PhenoBench paper.
