import os
import sys

# Let this file import from ../src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import gradio as gr
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor

from src.visualize import overlay

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

CHECKPOINT = "models/last.pth"   # change if your file has a different name
NUM_CLASSES = 2   # background + leaf. Confirm with A!


def load_model():
    # Build the empty architecture (no COCO weights)
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)

    # Swap the box head to output NUM_CLASSES
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)

    # Swap the mask head to output NUM_CLASSES
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, NUM_CLASSES)

    # Load A's trained numbers
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    model.eval()
    return model


model = load_model()


def run(image):
    if image is None:
        return None, "Please upload an image."

    with torch.no_grad():
        pred = model([to_tensor(image)])[0]

    keep = pred["scores"] > 0.5                       # drop low-confidence detections
    masks = pred["masks"][keep].cpu().numpy()
    boxes = pred["boxes"][keep].cpu().numpy()
    scores = pred["scores"][keep].cpu().numpy()

    result = overlay(image, masks, boxes, scores)
    return result, f"Leaf count: {len(scores)}"


demo = gr.Interface(fn=run, inputs=gr.Image(), outputs=[gr.Image(), gr.Textbox()])

if __name__ == "__main__":
    demo.launch()