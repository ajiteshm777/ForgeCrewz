import numpy as np
import cv2


def overlay(image, masks, boxes, scores):
    """Color each leaf randomly, blend 50% over the image, draw score on each box.

    image  : numpy array (H, W, 3), uint8
    masks  : array of N masks, each (H, W) or (1, H, W); probabilities or booleans
    boxes  : array of N boxes, each (x1, y1, x2, y2)
    scores : array of N confidence values
    """
    image = np.ascontiguousarray(image.astype(np.uint8))
    colored = image.copy()
    rng = np.random.default_rng()
    colors = []

    # 1. Paint each leaf's pixels with its own random color
    for mask in masks:
        mask = np.squeeze(mask) > 0.5          # turn probabilities into True/False
        color = rng.integers(0, 256, size=3).tolist()
        colored[mask] = color
        colors.append(color)

    # 2. Blend: 50% painted version + 50% original
    blended = cv2.addWeighted(colored, 0.5, image, 0.5, 0)

    # 3. Draw each box and its score
    for box, score, color in zip(boxes, scores, colors):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)
        cv2.putText(blended, f"{float(score):.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return blended
    