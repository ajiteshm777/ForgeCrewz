import numpy as np, cv2
from src.visualize import overlay

img = np.full((200, 200, 3), 255, dtype=np.uint8)       # white image
mask = np.zeros((200, 200)); mask[50:120, 60:140] = 1   # one fake leaf
out = overlay(img, [mask], [[60, 50, 140, 120]], [0.93])
cv2.imwrite("test_out.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))