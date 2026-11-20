from __future__ import annotations
import numpy as np
from skimage import measure, morphology, segmentation


def align_dims(np_input: np.ndarray, expected_dims: int = 2) -> np.ndarray:
    dim_input = len(np_input.shape)
    np_output = np_input
    if dim_input > expected_dims:
        np_output = np_input.squeeze()
    elif dim_input < expected_dims:
        np_output = np.expand_dims(np_input, 0)
    assert len(np_output.shape) == expected_dims
    return np_output


def binary_confusion(pred: np.ndarray, label: np.ndarray) -> tuple[int, int, int, int]:
    pred = align_dims(pred, 2) >= 0.5
    label = align_dims(label, 2) >= 0.5
    tp = int((pred & label).sum())
    fp = int((pred & ~label).sum())
    fn = int((~pred & label).sum())
    tn = int((~pred & ~label).sum())
    return tp, fp, fn, tn


def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> tuple[float, float, float, float, float]:
    if tp + fp + fn == 0:
        precision = recall = f1 = iou = 1.0
    else:
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        iou = tp / (tp + fp + fn + 1e-10)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-10)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return accuracy, precision, recall, f1, iou


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract a one-pixel inner boundary from a binary parcel mask."""
    return segmentation.find_boundaries(align_dims(mask, 2) >= 0.5, mode="inner")


def boundary_statistics(
    pred: np.ndarray,
    label: np.ndarray,
    tolerance_pixels: int,
) -> tuple[int, int, int, int]:
    """Return matched truth, truth total, matched prediction and prediction total."""
    pred_boundary = mask_boundary(pred)
    label_boundary = mask_boundary(label)
    pred_count = int(pred_boundary.sum())
    label_count = int(label_boundary.sum())
    if pred_count == 0 or label_count == 0:
        return 0, label_count, 0, pred_count

    footprint = morphology.disk(max(int(tolerance_pixels), 0))
    buffered_pred = morphology.binary_dilation(pred_boundary, footprint=footprint)
    buffered_label = morphology.binary_dilation(label_boundary, footprint=footprint)
    matched_label = int((label_boundary & buffered_pred).sum())
    matched_pred = int((pred_boundary & buffered_label).sum())
    return matched_label, label_count, matched_pred, pred_count


def boundary_metrics_from_statistics(
    matched_label: int,
    label_count: int,
    matched_pred: int,
    pred_count: int,
) -> tuple[float, float, float]:
    if pred_count == 0 and label_count == 0:
        return 1.0, 1.0, 1.0
    if pred_count == 0 or label_count == 0:
        return 0.0, 0.0, 0.0
    completeness = matched_label / label_count
    correctness = matched_pred / pred_count
    f_boundary = 2.0 * completeness * correctness / (completeness + correctness + 1e-10)
    return completeness, correctness, f_boundary


def object_error_sums(pred: np.ndarray, label: np.ndarray) -> tuple[float, float, float, int]:
    """Return area-weighted error numerators and total predicted parcel area.

    Each connected predicted parcel is paired with the reference parcel with
    which it has the largest overlap. Eight-connectivity is used for both maps.
    """
    pred_objects = measure.label(align_dims(pred, 2) >= 0.5, connectivity=2)
    label_objects = measure.label(align_dims(label, 2) >= 0.5, connectivity=2)
    pred_ids, pred_areas = np.unique(pred_objects[pred_objects > 0], return_counts=True)
    if pred_ids.size == 0:
        return 0.0, 0.0, 0.0, 0

    label_areas = np.bincount(label_objects.ravel())
    total_pred_area = int(pred_areas.sum())
    goc_sum = guc_sum = gtc_sum = 0.0
    for pred_id, pred_area_value in zip(pred_ids, pred_areas):
        parcel = pred_objects == pred_id
        overlapping_labels, intersections = np.unique(label_objects[parcel], return_counts=True)
        foreground = overlapping_labels > 0
        if not np.any(foreground):
            oc = uc = tc = 1.0
        else:
            candidate_labels = overlapping_labels[foreground]
            candidate_intersections = intersections[foreground]
            best_index = int(np.argmax(candidate_intersections))
            label_id = int(candidate_labels[best_index])
            intersection = float(candidate_intersections[best_index])
            pred_area = float(pred_area_value)
            label_area = float(label_areas[label_id])
            oc = 1.0 - intersection / label_area
            uc = 1.0 - intersection / pred_area
            tc = float(np.sqrt((oc * oc + uc * uc) / 2.0))
        area = float(pred_area_value)
        goc_sum += oc * area
        guc_sum += uc * area
        gtc_sum += tc * area
    return goc_sum, guc_sum, gtc_sum, total_pred_area
