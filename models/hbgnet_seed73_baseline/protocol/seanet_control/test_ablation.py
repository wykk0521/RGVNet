from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from skimage import measure, morphology, segmentation
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai4boundaries.dataset import AI4BoundariesSEANetDataset, load_config
from tools.seanet_ablation_model import SEANetAblation as SEANet


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


def binary_accuracy(pred: np.ndarray, label: np.ndarray) -> tuple[float, float, float, float, float]:
    return metrics_from_confusion(*binary_confusion(pred, label))


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


def boundary_accuracy(
    pred: np.ndarray,
    label: np.ndarray,
    tolerance_pixels: int,
) -> tuple[float, float, float]:
    """Return HBGNet-style boundary completeness, correctness and F_bdy."""
    return boundary_metrics_from_statistics(*boundary_statistics(pred, label, tolerance_pixels))


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


def object_errors(pred: np.ndarray, label: np.ndarray) -> tuple[float, float, float]:
    """Calculate per-image GOC, GUC and GTC for diagnostic logging."""
    goc_sum, guc_sum, gtc_sum, total_pred_area = object_error_sums(pred, label)
    if total_pred_area == 0:
        return (0.0, 0.0, 0.0) if not np.any(label) else (1.0, 1.0, 1.0)
    return goc_sum / total_pred_area, guc_sum / total_pred_area, gtc_sum / total_pred_area


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        self.sum += float(value)
        self.count += 1

    @property
    def average(self) -> float:
        return self.sum / max(self.count, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official SEANet evaluation adapted to AI4Boundaries_1m")
    parser.add_argument("--config", default="ai4boundaries/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--cuda-no", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def region_probability(
    model: SEANet,
    image: torch.Tensor,
) -> torch.Tensor:
    return torch.sigmoid(model(image)[0])


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Official SEANet evaluation requires CUDA")
    config = load_config(Path(args.config).resolve())
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    output_root = Path(config["paths"]["results_root"]) / args.test_name
    prediction_root = output_root / "predictions"
    metric_root = Path(config["paths"]["test_log_root"]) / args.test_name
    metric_path = metric_root / "metrics.json"
    text_path = metric_root / "accuracy.txt"
    if not args.overwrite and (metric_path.exists() or (prediction_root.exists() and any(prediction_root.iterdir()))):
        raise FileExistsError(f"Test output already exists; choose another --test-name or pass --overwrite: {args.test_name}")
    prediction_root.mkdir(parents=True, exist_ok=True)
    metric_root.mkdir(parents=True, exist_ok=True)

    dataset = AI4BoundariesSEANetDataset(config, args.split, augment=False)
    batch_size = int(args.batch_size or config["dataset"]["val_batch_size"])
    num_workers = int(config["dataset"]["num_workers"] if args.num_workers is None else args.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    device = torch.device(f"cuda:{args.cuda_no}")
    torch.cuda.set_device(device)
    model = SEANet(
        num_classes=1,
        candidate_config=config["model"]["semantic_wavelet_candidate"]["architecture"],
        relation_config=config["model"]["relation_validation"]["architecture"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    metric_names = (
        "oa",
        "precision",
        "recall",
        "f1",
        "iou",
        "boundary_completeness",
        "boundary_correctness",
        "f_boundary",
        "goc",
        "guc",
        "gtc",
    )
    meters = {name: AverageMeter() for name in metric_names}
    global_confusion = np.zeros(4, dtype=np.int64)
    global_boundary = np.zeros(4, dtype=np.int64)
    global_object_numerators = np.zeros(3, dtype=np.float64)
    global_predicted_object_area = 0
    empty_counts = {
        "target_empty": 0,
        "prediction_empty": 0,
        "both_empty": 0,
        "target_empty_prediction_nonempty": 0,
        "target_nonempty_prediction_empty": 0,
        "nonempty_without_overlap": 0,
    }
    threshold = float(config["evaluation"]["region_threshold"])
    spatial_resolution_m = float(config["evaluation"]["spatial_resolution_m"])
    boundary_tolerance_m = float(config["evaluation"]["boundary_tolerance_m"])
    boundary_tolerance_pixels = int(round(boundary_tolerance_m / spatial_resolution_m))
    evaluated = 0

    with text_path.open("w", encoding="utf-8") as text_stream:
        for batch_index, batch in enumerate(tqdm(loader, desc=f"SEANet {args.split}"), start=1):
            if args.max_batches is not None and batch_index > args.max_batches:
                break
            image = batch["image"].to(device, non_blocking=True)
            probability = region_probability(model, image).cpu().numpy()
            targets = batch["region"].numpy()
            for sample_index, image_id in enumerate(batch["image_id"]):
                prediction = probability[sample_index, 0] > threshold
                target = targets[sample_index, 0]
                confusion = binary_confusion(prediction, target)
                boundary_stats = boundary_statistics(prediction, target, boundary_tolerance_pixels)
                object_stats = object_error_sums(prediction, target)
                pixel_values = metrics_from_confusion(*confusion)
                boundary_values = boundary_metrics_from_statistics(*boundary_stats)
                if object_stats[3] > 0:
                    object_values = tuple(value / object_stats[3] for value in object_stats[:3])
                else:
                    object_values = (0.0, 0.0, 0.0) if not np.any(target) else (1.0, 1.0, 1.0)
                values = pixel_values + boundary_values + object_values
                for name, value in zip(metric_names, values):
                    meters[name].update(value)
                global_confusion += np.asarray(confusion, dtype=np.int64)
                global_boundary += np.asarray(boundary_stats, dtype=np.int64)
                global_object_numerators += np.asarray(object_stats[:3], dtype=np.float64)
                global_predicted_object_area += int(object_stats[3])
                target_empty = not np.any(target)
                prediction_empty = not np.any(prediction)
                empty_counts["target_empty"] += int(target_empty)
                empty_counts["prediction_empty"] += int(prediction_empty)
                empty_counts["both_empty"] += int(target_empty and prediction_empty)
                empty_counts["target_empty_prediction_nonempty"] += int(target_empty and not prediction_empty)
                empty_counts["target_nonempty_prediction_empty"] += int(not target_empty and prediction_empty)
                empty_counts["nonempty_without_overlap"] += int(
                    not target_empty and not prediction_empty and confusion[0] == 0
                )
                cv2.imwrite(str(prediction_root / f"{image_id}.png"), prediction.astype(np.uint8) * 255)
                evaluated += 1
                line = (
                    f"Eval {evaluated}/{len(dataset)}, OA {values[0] * 100:.2f}, "
                    f"P {values[1] * 100:.2f}, R {values[2] * 100:.2f}, "
                    f"F1 {values[3] * 100:.2f}, IoU {values[4] * 100:.2f}, "
                    f"F_bdy {values[7]:.4f}, GOC {values[8]:.4f}, "
                    f"GUC {values[9]:.4f}, GTC {values[10]:.4f}"
                )
                print(line)
                text_stream.write(line + "\n")

        averages = {name: meter.average for name, meter in meters.items()}
        region_global = metrics_from_confusion(*(int(v) for v in global_confusion))
        boundary_global = boundary_metrics_from_statistics(*(int(v) for v in global_boundary))
        if global_predicted_object_area > 0:
            object_global = tuple(
                float(value / global_predicted_object_area) for value in global_object_numerators
            )
        else:
            object_global = (0.0, 0.0, 0.0) if global_confusion[2] == 0 else (1.0, 1.0, 1.0)
        summary = (
            f"PRIMARY GLOBAL: F1 {region_global[3] * 100:.2f}, IoU {region_global[4] * 100:.2f}, "
            f"F_bdy {boundary_global[2]:.4f}, GOC {object_global[0]:.4f}, "
            f"GUC {object_global[1]:.4f}, GTC {object_global[2]:.4f}\n"
            f"DIAGNOSTIC GLOBAL: OA {region_global[0] * 100:.2f}, "
            f"Precision {region_global[1] * 100:.2f}, Recall {region_global[2] * 100:.2f}, "
            f"Com {boundary_global[0]:.4f}, Corr {boundary_global[1]:.4f}\n"
            f"DIAGNOSTIC PER-IMAGE MACRO: F1 {averages['f1'] * 100:.2f}, "
            f"IoU {averages['iou'] * 100:.2f}, F_bdy {averages['f_boundary']:.4f}"
        )
        print(summary)
        text_stream.write(summary + "\n")

    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "split": args.split,
        "test_name": args.test_name,
        "complete_split": args.max_batches is None and evaluated == len(dataset),
        "samples": evaluated,
        "protocol": str(config["evaluation"]["protocol"]),
        "aggregation": "region_micro_boundary_micro_object_global_area_weighted",
        "tta": False,
        "region_threshold": threshold,
        "small_object_removal": False,
        "spatial_resolution_m": spatial_resolution_m,
        "boundary_tolerance_m": boundary_tolerance_m,
        "boundary_tolerance_pixels": boundary_tolerance_pixels,
        "primary_metrics": {
            "f1": region_global[3],
            "iou": region_global[4],
            "f_boundary": boundary_global[2],
            "goc": object_global[0],
            "guc": object_global[1],
            "gtc": object_global[2],
        },
        "diagnostic_metrics": {
            "oa": region_global[0],
            "precision": region_global[1],
            "recall": region_global[2],
            "boundary_completeness": boundary_global[0],
            "boundary_correctness": boundary_global[1],
            "per_image_macro_f1": averages["f1"],
            "per_image_macro_iou": averages["iou"],
            "per_image_macro_f_boundary": averages["f_boundary"],
        },
        "global_counts": {
            "tp": int(global_confusion[0]),
            "fp": int(global_confusion[1]),
            "fn": int(global_confusion[2]),
            "tn": int(global_confusion[3]),
            "matched_truth_boundary_pixels": int(global_boundary[0]),
            "truth_boundary_pixels": int(global_boundary[1]),
            "matched_prediction_boundary_pixels": int(global_boundary[2]),
            "prediction_boundary_pixels": int(global_boundary[3]),
            "predicted_object_area_pixels": int(global_predicted_object_area),
        },
        "sample_diagnostics": empty_counts,
        "prediction_root": str(prediction_root),
    }
    metric_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
