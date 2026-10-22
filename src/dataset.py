"""PhenoBench leaf-instance dataset for torchvision Mask R-CNN.

Expected layout (verify against your download in Hour 0!):
    <root>/{train,val}/images/*.png
    <root>/{train,val}/leaf_instances/*.png   # 0 = background, >0 = leaf id
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import to_tensor

MIN_MASK_PIXELS = 20  # drop tiny fragments that produce degenerate boxes


class PhenoBenchLeaves(torch.utils.data.Dataset):
    def __init__(self, root, split="train", train=True, limit=None):
        self.img_dir = os.path.join(root, split, "images")
        self.inst_dir = os.path.join(root, split, "leaf_instances")
        self.files = sorted(os.listdir(self.img_dir))
        if limit:
            self.files = self.files[:limit]
        self.train = train
        self.jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)

    def __len__(self):
        return len(self.files)

    def _augment(self, img, inst):
        # img: PIL, inst: np.ndarray (H, W). Geometric ops applied to both.
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            inst = np.ascontiguousarray(inst[:, ::-1])
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            inst = np.ascontiguousarray(inst[::-1, :])
        if random.random() < 0.5:  # 90 degree rotation: leaves have no canonical orientation
            img = img.transpose(Image.ROTATE_90)
            inst = np.ascontiguousarray(np.rot90(inst))
        img = self.jitter(img)  # photometric only
        return img, inst

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        inst = np.array(Image.open(os.path.join(self.inst_dir, name)))
        inst = inst.astype(np.int32)

        if self.train:
            img, inst = self._augment(img, inst)

        ids = [i for i in np.unique(inst) if i != 0]
        masks, boxes = [], []
        for i in ids:
            m = inst == i
            if m.sum() < MIN_MASK_PIXELS:
                continue
            ys, xs = np.where(m)
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            masks.append(m)
            boxes.append([x0, y0, x1, y1])

        n = len(masks)
        h, w = inst.shape
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(n, 4),
            "labels": torch.ones((n,), dtype=torch.int64),  # 1 = leaf
            "masks": torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            if n else torch.zeros((0, h, w), dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor([m.sum() for m in masks], dtype=torch.float32),
            "iscrowd": torch.zeros((n,), dtype=torch.int64),
        }
        return to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))
