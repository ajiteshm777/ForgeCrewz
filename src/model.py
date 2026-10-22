"""Mask R-CNN (ResNet50-FPN v2) with heads replaced for background + leaf."""
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

NUM_CLASSES = 2  # background + leaf


def build_model(pretrained=True, detections_per_img=300, min_size=800, max_size=1333):
    """Leaves are dense, so raise detections_per_img above the default 100.

    PhenoBench images are 1024x1024. min_size=800 is cheaper on a T4;
    try min_size=1024, max_size=1024 later if memory allows (small leaves benefit).
    """
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=weights,
        box_detections_per_img=detections_per_img,
        min_size=min_size,
        max_size=max_size,
    )
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)

    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, NUM_CLASSES)
    return model


def load_checkpoint_for_inference(path, device="cpu", **kwargs):
    """Interface for Member B/C: returns an eval-mode model."""
    model = build_model(pretrained=False, **kwargs)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model.to(device).eval()
