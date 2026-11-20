"""FHAPD region, contour and distance targets for the existing HBGNet interface."""
from __future__ import annotations

import cv2
import numpy as np

TARGET_VERSION = 'fhapd-bsinet-canny100-200-l2-mask3-minmax-u8-v1'
REFERENCE_URL = 'https://github.com/long123524/BsiNet-torch/blob/main/preprocess.py'


def build_targets(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, (0, 1)).all():
        raise ValueError('FHAPD source mask must be a uint8 [H,W] array encoded as 0/1')
    binary = np.ascontiguousarray(mask * np.uint8(255))
    contour = cv2.Canny(binary, 100, 200)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    low, high = float(distance.min()), float(distance.max())
    # The reference has a zero denominator for constant maps, including empty GT.
    if high > low:
        distance_u8 = ((distance - low) / (high - low) * 255).astype(np.uint8)
    else:
        distance_u8 = np.zeros_like(mask)
    return np.stack((mask.astype(np.float32), (contour > 0).astype(np.float32),
                     distance_u8.astype(np.float32) / 255.0))
