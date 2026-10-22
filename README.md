   ---
   title: ForgeCrewz
   emoji: 🌿
   sdk: gradio
   app_file: app/demo.py
   ---
# Leaf Instance Segmentation with Mask R-CNN

Detects each visible leaf in a field image and gives it its own pixel-level mask.
Built for WEB FORGE 2026, PS 07.

![Sample result](results/sample.jpg)
![Demo](results/demo.gif)

## Overview
- **Problem:** separating individual leaves is hard because of overlap, occlusion, shadows and clutter.
- **Approach:** Mask R-CNN (ResNet-50 FPN, Torchvision) fine-tuned on the PhenoBench dataset.
- **Result:** mask AP = <X>, AP50 = <X>, AP75 = <X> on the validation set.

## Installation
```bash
git clone https://github.com/ajiteshm777/ForgeCrewz.git
cd ForgeCrewz
pip install -r requirements.txt
```
Download PhenoBench from https://www.phenobench.org and place it in `data/PhenoBench/`.

## Usage
Train:
```bash
python src/train.py
```
Run inference on one image:
```bash
python src/infer.py --image x.jpg
```
Launch the demo:
```bash
python app/demo.py
```

## Model checkpoint
Download: <link>. Place the file in `checkpoints/`.

## Results
| Metric | Value |
|---|---|
| mask AP | <X> |
| AP50 | <X> |
| AP75 | <X> |

Setup: Colab T4 GPU, <N> epochs, batch size <N>, SGD lr <X>, augmentations: flips and color jitter.
Full report: [docs/validation_report.md](docs/validation_report.md)

## Limitations
- Small or heavily occluded leaves can be missed or merged.
- Trained on sugar beet fields, so other crops or cameras may perform worse.
- Limited epochs and resolution because of the T4 and the 8-hour limit.

More in [docs/limitations.md](docs/limitations.md).

## Team
Team Leader : Ishan Shrivastava
Member A : Ajitesh Mohan, Member B : Ishan Shrivastava, Member C : Pranjal Mojad

## Credits
PhenoBench dataset, PyTorch and Torchvision.
