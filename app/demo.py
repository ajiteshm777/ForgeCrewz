import os
import sys

# Let this file import from ../src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import gradio as gr
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor

from src.visualize import overlay

# PLACEHOLDER: COCO-pretrained model, used only to test the UI.
# Replace with A's trained checkpoint once it exists.
model = maskrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()


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