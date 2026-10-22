# Limitations

Status: DRAFT. Items marked TODO are waiting for B's full-validation results.
Model: Mask R-CNN (ResNet-50 FPN v2), epoch-14 checkpoint. Data: PhenoBench, sugar beet only.

## 1. Small leaves are the weakest case
- **Observed (A, first 100 val images, epoch 14, mask AP):** AP small 0.402 vs AP medium 0.803.
- **Caveat:** those 100 images are the first 100 val files alphabetically, probably mostly young plants, so they are not representative. COCO evaluation also caps detections at 100 per image while the model can output 300, so recall on dense images may be understated.
- **Gallery:** row "Small leaves" (`TODO: filename`).
- TODO (B): full-val result and the count of worst images tagged "small leaves".

## 2. Outlines are less precise than detection
- **Observed (A, same 100 images):** AP50 0.865 vs AP75 0.629.
- **Hypothesis (untested):** the model finds most leaves but boundaries are less exact.

## 3. Possible false positives on bare soil
- **Preliminary:** B saw one large false-positive "leaf" covering about 60% of one image on bare soil, using an early weak test checkpoint (not the final one).
- **Not confirmed** with epoch 14 or on the full val set.
- TODO (B): recheck with epoch14 and give counts.
- **Possible mitigation, not implemented:** drop detections above a set fraction of image area.

## 4. Overlap, occlusion, shadow and bright light
- TODO: after viewing `results/gallery.png`, write what each row actually shows (merged leaves, missed leaves, etc.).
- TODO (B): counts of worst images by cause.

## 5. Threshold trade-off
Score and NMS thresholds trade merging leaves against over-splitting them. The demo default is TODO (B's PQ-based value).

## 6. One field, one UAV setup
Trained and tested on a single sugar beet field and camera setup. Other crops, cameras and altitudes are untested.

## 7. Short training, reduced resolution
Only 14 epochs, and 1024 px images are resized to about 800 px internally, because of the 8-hour / T4 budget. A 1024 px run was considered and not done.

## 8. Dataset copy is not the official zip
The data comes from the Hugging Face mirror Voxel51/PhenoBench_Raw.
- Leaf instance ids are renumbered per image.
- Where leaves overlap, the later leaf overwrites the earlier one.
- `leaf_visibility` was regenerated and is unverified against the official files.
- Any PQ number depends on that regenerated file.

## 9. Speed
CPU about 12.6 to 13.3 s/image (measured by B). GPU about 2.3 s/image (A's figure, not measured by B). The demo on a CPU is slow.

## General caveats
- Single training run, so no confidence intervals.
- Any subset result must be quoted with its sample size and seed.
