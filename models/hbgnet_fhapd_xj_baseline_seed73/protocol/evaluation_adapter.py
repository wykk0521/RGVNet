"""Streaming, model-free FHAPD evaluation adapter for HBGNet Protocol B.

Scope
-----
This module turns per-image region predictions into the single split-level
report used by the existing HBGNet parcel protocol.  It contains no model, no
checkpoint handling, no dataset reader and no file/CLI side effects: the
caller streams ``(image_id, probability_or_mask, ground_truth)`` triples and
receives one aggregated report in memory.  It never performs a model test.

The metric functions are the *unchanged* shared implementations in
``metrics.py`` (byte-for-byte the HBGNet ``protocol/metrics.py``):

* pixel micro-aggregation      -> ``binary_confusion`` / ``metrics_from_confusion``
* inner-boundary micro         -> ``boundary_statistics`` / ``boundary_metrics_from_statistics``
* global object area weighting -> ``object_error_sums``

Aggregation contract (``aggregation`` setting)
---------------------------------------------
``region_micro_boundary_micro_object_global_area_weighted``:

* region metrics sum the four pixel counts over the split (micro);
* boundary metrics sum matched/total counts with the configured pixel
  tolerance (micro);
* object errors are recombined as ``sum(error * object_area) / sum(object_area)``
  over the whole split (global area weighted, *not* a per-image mean) and are
  ``None`` when the split predicts no foreground at all.

Public API
----------
``validate_evaluation_settings(cfg) -> dict``
    Validates ``cfg['evaluation']`` (strict threshold / comparison / TTA /
    post-processing drift checks) and returns the resolved settings including
    ``input_size`` and the genuine integer ``boundary_tolerance_pixels``.

``resolve_boundary_tolerance(cfg) -> int``
    Returns the genuine integer pixel tolerance.  Refuses booleans, negatives
    and non-integral values.  While ``physical_tolerance_verified`` is false
    both ``spatial_resolution_m`` and ``boundary_tolerance_m`` must be None
    (currently the genuine tolerance is 3 px, not a verified 3 m); when it is
    true the metre value must convert to the pixel value through the ground
    sampling distance.

``resolve_input_size(cfg) -> tuple[int, int]``
    Reads ``dataset.input_size`` or resolves it from ``dataset.variants``
    with the same variant-selection semantics as ``dataset.py``.

``accumulate_prediction_masks(cfg, ids, predictions, split, include_per_image=True) -> dict``
    Streams ``(id, probability H,W float in [0,1] or bool mask, gt binary
    H,W)`` items, enforcing the configured split size, exact id order,
    configured spatial size, finiteness, encodings and the strict ``> 0.5``
    decision rule, and returns the split report.

Input validation
----------------
* ``ids`` must be a unique, clean list whose length equals
  ``cfg['dataset']['expected_counts'][split]``.
* The stream must yield exactly those ids in exactly that order: missing,
  duplicated or extra items are rejected.
* Probability maps must be floating point (or boolean decisions), finite and
  inside ``[0, 1]``; logits or other out-of-range floats are rejected instead
  of being sigmoided.
* Ground truth must be boolean or binary 0/1 values; 255-encoded masks are
  rejected.
* Both maps are shape-checked against the configured ``input_size`` *before*
  any metric is computed.

Report
------
``accumulate_prediction_masks`` returns::

    {
      'protocol': str,
      'split': str,
      'counts': {'expected': int, 'evaluated': int},
      'samples': int,
      'complete_split': True,
      'aggregation': str,
      'boundary_tolerance_pixels': int,
      'metrics': {
          'f1': float, 'iou': float, 'f_boundary': float,
          'goc': float | None, 'guc': float | None, 'gtc': float | None,
          'oa': float, 'precision': float, 'recall': float,
          'boundary_completeness': float, 'boundary_correctness': float,
      },
      'pixel_counts': [tp, fp, fn, tn],
      'boundary_counts': [matched_label, label_count, matched_pred, pred_count],
      'predicted_object_area': int,
      'per_image': [  # omitted when include_per_image=False
          {'index': int, 'id': str, 'pixel_counts': [...],
           'boundary_counts': [...], 'object_numerators': [goc, guc, gtc],
           'predicted_object_area': int},
          ...
      ],
    }

``pixel_counts`` and ``boundary_counts`` keep the argument order of the
shared metric helpers, so the summed per-image counts can be re-fed to
``metrics_from_confusion`` and ``boundary_metrics_from_statistics``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "EvaluationAdapterError",
    "EvaluationConfigError",
    "PROTOCOL",
    "AGGREGATION",
    "SPLITS",
    "validate_evaluation_settings",
    "resolve_boundary_tolerance",
    "resolve_input_size",
    "accumulate_prediction_masks",
]

PROTOCOL = "HBGNet-metrics-FHAPD-256-3px-spatial-v1"
AGGREGATION = "region_micro_boundary_micro_object_global_area_weighted"
REGION_THRESHOLD = 0.5
THRESHOLD_COMPARISON = ">"
SPLITS = ("train", "val", "test")

_METRIC_CACHE = None
_END = object()


class EvaluationAdapterError(ValueError):
    """Raised for adapter configuration or prediction-stream violations."""


class EvaluationConfigError(EvaluationAdapterError):
    """Raised when the evaluation/dataset configuration drifts from the protocol."""


def _is_real_number(value):
    """True for int/float/numpy scalars but never for booleans."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _is_false(value):
    return isinstance(value, (bool, np.bool_)) and not bool(value)


def _require_evaluation(cfg):
    if not isinstance(cfg, dict):
        raise EvaluationConfigError("config must be a mapping")
    evaluation = cfg.get("evaluation")
    if not isinstance(evaluation, dict):
        raise EvaluationConfigError("config must contain an evaluation mapping")
    return evaluation


def _validate_size(size, origin):
    if isinstance(size, (str, bytes)) or not isinstance(size, (list, tuple)):
        raise EvaluationConfigError(f"{origin} must be a [height, width] sequence")
    if len(size) != 2:
        raise EvaluationConfigError(f"{origin} must contain exactly two entries, got {size!r}")
    values = []
    for entry in size:
        if (
            not _is_real_number(entry)
            or not math.isfinite(float(entry))
            or float(entry) != math.floor(float(entry))
            or float(entry) <= 0
        ):
            raise EvaluationConfigError(f"{origin} entries must be positive integers, got {size!r}")
        values.append(int(entry))
    return tuple(values)


def resolve_input_size(cfg):
    """Resolve the configured spatial size, including ``dataset.variants``."""
    if not isinstance(cfg, dict):
        raise EvaluationConfigError("config must be a mapping")
    dataset = cfg.get("dataset")
    if not isinstance(dataset, dict):
        raise EvaluationConfigError("config must contain a dataset mapping")
    if dataset.get("input_size") is not None:
        return _validate_size(dataset["input_size"], "dataset.input_size")

    variants = dataset.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise EvaluationConfigError("dataset needs input_size or a non-empty variants mapping")
    variant = dataset.get("variant")
    if not isinstance(variant, str) or not variant:
        raise EvaluationConfigError("dataset.variant must name one configured variant")
    if variant not in variants:
        raise EvaluationConfigError(
            f"unknown dataset.variant {variant!r}; expected one of {sorted(variants)}"
        )
    selected = variants[variant]
    if not isinstance(selected, dict) or selected.get("input_size") is None:
        raise EvaluationConfigError(f"dataset.variants.{variant} must define input_size")
    return _validate_size(selected["input_size"], f"dataset.variants.{variant}.input_size")


def resolve_boundary_tolerance(cfg):
    """Return the genuine integer boundary tolerance, refusing physical drift."""
    evaluation = _require_evaluation(cfg)

    verified = evaluation.get("physical_tolerance_verified")
    if not isinstance(verified, (bool, np.bool_)):
        raise EvaluationConfigError("evaluation.physical_tolerance_verified must be a boolean")

    raw_pixels = evaluation.get("boundary_tolerance_pixels")
    if not _is_real_number(raw_pixels):
        raise EvaluationConfigError(
            "evaluation.boundary_tolerance_pixels must be an integer number of pixels "
            "(booleans are not accepted)"
        )
    if float(raw_pixels) < 0:
        raise EvaluationConfigError("evaluation.boundary_tolerance_pixels must be non-negative")
    if not math.isfinite(float(raw_pixels)) or float(raw_pixels) != math.floor(float(raw_pixels)):
        raise EvaluationConfigError(
            "evaluation.boundary_tolerance_pixels must be an integral number of pixels"
        )
    pixels = int(raw_pixels)

    requested = evaluation.get("requested_boundary_tolerance_m")
    if not _is_real_number(requested) or not math.isfinite(float(requested)) or float(requested) <= 0:
        raise EvaluationConfigError(
            "evaluation.requested_boundary_tolerance_m must be a positive finite number"
        )

    resolution = evaluation.get("spatial_resolution_m")
    tolerance_m = evaluation.get("boundary_tolerance_m")
    if bool(verified):
        if (
            not _is_real_number(resolution)
            or not math.isfinite(float(resolution))
            or float(resolution) <= 0
        ):
            raise EvaluationConfigError(
                "verified physical tolerance requires a positive finite spatial_resolution_m"
            )
        if (
            not _is_real_number(tolerance_m)
            or not math.isfinite(float(tolerance_m))
            or float(tolerance_m) <= 0
        ):
            raise EvaluationConfigError(
                "verified physical tolerance requires a positive finite boundary_tolerance_m"
            )
        ratio = float(tolerance_m) / float(resolution)
        nearest = int(round(ratio))
        if not math.isclose(ratio, nearest, rel_tol=1e-9, abs_tol=1e-9):
            raise EvaluationConfigError(
                "boundary_tolerance_m must convert to a whole number of pixels through "
                f"spatial_resolution_m (got {ratio!r} px)"
            )
        if nearest != pixels:
            raise EvaluationConfigError(
                "boundary_tolerance_m / spatial_resolution_m "
                f"({nearest} px) does not match boundary_tolerance_pixels ({pixels} px)"
            )
        if not math.isclose(
            float(tolerance_m), pixels * float(resolution), rel_tol=1e-9, abs_tol=1e-12
        ):
            raise EvaluationConfigError(
                "boundary_tolerance_m must equal boundary_tolerance_pixels * spatial_resolution_m"
            )
    else:
        if resolution is not None:
            raise EvaluationConfigError(
                "spatial_resolution_m must stay None until physical_tolerance_verified is true"
            )
        if tolerance_m is not None:
            raise EvaluationConfigError(
                "boundary_tolerance_m must stay None: the genuine tolerance is 3 px, "
                "not a verified 3 m"
            )
    return pixels


def validate_evaluation_settings(cfg):
    """Validate ``cfg['evaluation']`` strictly and return resolved settings."""
    evaluation = _require_evaluation(cfg)

    protocol = evaluation.get("protocol")
    if protocol != PROTOCOL:
        raise EvaluationConfigError(f"evaluation.protocol must be {PROTOCOL!r}")

    threshold = evaluation.get("region_threshold")
    if not _is_real_number(threshold) or float(threshold) != REGION_THRESHOLD:
        raise EvaluationConfigError("evaluation.region_threshold must be exactly 0.5")
    comparison = evaluation.get("threshold_comparison")
    if comparison != THRESHOLD_COMPARISON:
        raise EvaluationConfigError("evaluation.threshold_comparison must be '>'")
    if not _is_false(evaluation.get("tta")):
        raise EvaluationConfigError("evaluation.tta must be exactly false")
    if not _is_false(evaluation.get("small_object_removal")):
        raise EvaluationConfigError("evaluation.small_object_removal must be exactly false")
    aggregation = evaluation.get("aggregation")
    if aggregation != AGGREGATION:
        raise EvaluationConfigError(f"evaluation.aggregation must be {AGGREGATION!r}")

    return {
        "protocol": protocol,
        "region_threshold": REGION_THRESHOLD,
        "threshold_comparison": THRESHOLD_COMPARISON,
        "tta": False,
        "small_object_removal": False,
        "aggregation": aggregation,
        "spatial_resolution_m": evaluation.get("spatial_resolution_m"),
        "boundary_tolerance_m": evaluation.get("boundary_tolerance_m"),
        "requested_boundary_tolerance_m": evaluation.get("requested_boundary_tolerance_m"),
        "boundary_tolerance_pixels": resolve_boundary_tolerance(cfg),
        "physical_tolerance_verified": bool(evaluation.get("physical_tolerance_verified")),
        "input_size": resolve_input_size(cfg),
    }


def _expected_count(cfg, split):
    if split not in SPLITS:
        raise EvaluationConfigError(f"split must be one of {SPLITS}, got {split!r}")
    dataset = cfg.get("dataset") if isinstance(cfg, dict) else None
    if not isinstance(dataset, dict):
        raise EvaluationConfigError("config must contain a dataset mapping")
    counts = dataset.get("expected_counts")
    if not isinstance(counts, dict):
        raise EvaluationConfigError("dataset.expected_counts must map train/val/test to counts")
    value = counts.get(split)
    if (
        not _is_real_number(value)
        or not math.isfinite(float(value))
        or float(value) != math.floor(float(value))
        or float(value) < 0
    ):
        raise EvaluationConfigError(f"dataset.expected_counts.{split} must be a non-negative integer")
    return int(value)


def _validated_ids(ids, expected):
    if isinstance(ids, (str, bytes)) or not isinstance(ids, (list, tuple)):
        raise EvaluationAdapterError("ids must be a list or tuple of unique sample ids")
    if len(ids) != expected:
        raise EvaluationAdapterError(
            f"sample count mismatch: got {len(ids)} ids, dataset.expected_counts requires {expected}"
        )
    seen = set()
    for ident in ids:
        if not isinstance(ident, str) or not ident:
            raise EvaluationAdapterError(f"sample id must be a non-empty string, got {ident!r}")
        if ident != ident.strip():
            raise EvaluationAdapterError(f"sample id has surrounding whitespace: {ident!r}")
        if ident in (".", ".."):
            raise EvaluationAdapterError(f"unsafe sample id: {ident!r}")
        if any(ch in ident for ch in ("/", "\\", "\x00")):
            raise EvaluationAdapterError(f"sample id must not contain path separators: {ident!r}")
        if ident in seen:
            raise EvaluationAdapterError(f"duplicate sample id: {ident!r}")
        seen.add(ident)
    return list(ids)


def _metric_functions():
    """Import the shared metric helpers from ``metrics.py`` (cached)."""
    global _METRIC_CACHE
    if _METRIC_CACHE is None:
        try:
            from metrics import (
                binary_confusion,
                boundary_metrics_from_statistics,
                boundary_statistics,
                metrics_from_confusion,
                object_error_sums,
            )
        except ImportError:
            try:
                from protocol.metrics import (
                    binary_confusion,
                    boundary_metrics_from_statistics,
                    boundary_statistics,
                    metrics_from_confusion,
                    object_error_sums,
                )
            except ImportError as exc:
                raise EvaluationAdapterError(
                    f"shared metric functions (metrics.py) are unavailable: {exc}"
                ) from exc
        _METRIC_CACHE = {
            "binary_confusion": binary_confusion,
            "boundary_metrics_from_statistics": boundary_metrics_from_statistics,
            "boundary_statistics": boundary_statistics,
            "metrics_from_confusion": metrics_from_confusion,
            "object_error_sums": object_error_sums,
        }
    return _METRIC_CACHE


def _as_2d_array(field, value, input_size):
    if isinstance(value, (str, bytes)):
        raise EvaluationAdapterError(f"{field} must be a 2-D array, got {type(value).__name__}")
    try:
        array = np.asarray(value)
    except Exception as exc:  # pragma: no cover - numpy conversion diagnostics
        raise EvaluationAdapterError(f"{field} is not convertible to a numpy array: {exc}") from exc
    if array.ndim != 2:
        raise EvaluationAdapterError(f"{field} must be 2-D [H,W], got shape {array.shape}")
    if tuple(array.shape) != tuple(input_size):
        raise EvaluationAdapterError(
            f"{field} shape {tuple(array.shape)} does not match configured "
            f"input_size {tuple(input_size)}"
        )
    return array


def _decision_from_probability(probability, ident, settings):
    if np.issubdtype(probability.dtype, np.bool_):
        return probability
    if not np.issubdtype(probability.dtype, np.floating):
        raise EvaluationAdapterError(
            f"sample {ident!r}: probability map must be floating point or boolean, "
            f"got dtype {probability.dtype}"
        )
    if not np.isfinite(probability).all():
        raise EvaluationAdapterError(f"sample {ident!r}: probability map contains non-finite values")
    low = float(probability.min())
    high = float(probability.max())
    if low < 0.0 or high > 1.0:
        raise EvaluationAdapterError(
            f"sample {ident!r}: probability values must be inside [0,1] "
            f"(got [{low}, {high}]); pass probabilities, not logits"
        )
    return probability > settings["region_threshold"]


def _validated_truth(truth, ident):
    if np.issubdtype(truth.dtype, np.bool_):
        return truth
    if np.issubdtype(truth.dtype, np.floating):
        if not np.isfinite(truth).all():
            raise EvaluationAdapterError(
                f"sample {ident!r}: ground truth contains non-finite values"
            )
    elif not np.issubdtype(truth.dtype, np.integer):
        raise EvaluationAdapterError(
            f"sample {ident!r}: ground truth must be boolean or binary numeric, "
            f"got dtype {truth.dtype}"
        )
    if not np.isin(truth, (0, 1)).all():
        values = np.unique(np.asarray(truth))
        raise EvaluationAdapterError(
            f"sample {ident!r}: ground truth must be binary 0/1 "
            f"(255 encoding is rejected), got values {values[:8].tolist()!r}"
        )
    return truth


def accumulate_prediction_masks(cfg, ids, predictions, split, *, include_per_image=True):
    """Stream a full split and return the aggregated HBGNet metric report.

    ``predictions`` is any iterable of ``(id, probability, ground_truth)``
    items.  ``probability`` is a 2-D float map with values in ``[0, 1]`` (the
    decision rule is the strict ``> 0.5``) or an already-decided 2-D boolean
    mask.  ``ground_truth`` is a binary 2-D map (bool or 0/1 values).

    The function enforces the configured split size and exact id order and
    raises :class:`EvaluationAdapterError` (a ``ValueError``) on any missing,
    duplicated or extra sample, shape mismatch, non-finite value, invalid
    encoding or invalid settings.  No file, model or CLI interaction happens.
    """
    settings = validate_evaluation_settings(cfg)
    expected = _expected_count(cfg, split)
    id_list = _validated_ids(ids, expected)
    try:
        iterator = iter(predictions)
    except TypeError as exc:
        raise EvaluationAdapterError(
            "predictions must be an iterable of (id, probability, ground_truth) items"
        ) from exc

    tolerance = settings["boundary_tolerance_pixels"]
    input_size = settings["input_size"]
    pixel_counts = [0, 0, 0, 0]
    boundary_counts = [0, 0, 0, 0]
    object_sums = [0.0, 0.0, 0.0]
    predicted_area = 0
    per_image = []
    metrics = None

    for index, ident in enumerate(id_list):
        try:
            item = next(iterator)
        except StopIteration:
            raise EvaluationAdapterError(
                f"prediction stream ended early: {len(id_list) - index} sample(s) missing"
            ) from None
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise EvaluationAdapterError(
                f"item {index} must be a (id, probability, ground_truth) tuple"
            )
        stream_id, probability, truth = item
        if not isinstance(stream_id, str) or stream_id != ident:
            raise EvaluationAdapterError(
                f"prediction id/order mismatch at index {index}: "
                f"expected {ident!r}, got {stream_id!r}"
            )
        probability = _as_2d_array("probability", probability, input_size)
        truth = _as_2d_array("ground truth", truth, input_size)
        decision = _decision_from_probability(probability, ident, settings)
        truth = _validated_truth(truth, ident)
        if metrics is None:
            metrics = _metric_functions()

        tp, fp, fn, tn = metrics["binary_confusion"](decision, truth)
        matched_label, label_count, matched_pred, pred_count = metrics["boundary_statistics"](
            decision, truth, tolerance
        )
        goc, guc, gtc, area = metrics["object_error_sums"](decision, truth)

        pixel_counts[0] += tp
        pixel_counts[1] += fp
        pixel_counts[2] += fn
        pixel_counts[3] += tn
        boundary_counts[0] += matched_label
        boundary_counts[1] += label_count
        boundary_counts[2] += matched_pred
        boundary_counts[3] += pred_count
        object_sums[0] += float(goc)
        object_sums[1] += float(guc)
        object_sums[2] += float(gtc)
        predicted_area += int(area)

        if include_per_image:
            per_image.append(
                {
                    "index": index,
                    "id": ident,
                    "pixel_counts": [int(tp), int(fp), int(fn), int(tn)],
                    "boundary_counts": [
                        int(matched_label),
                        int(label_count),
                        int(matched_pred),
                        int(pred_count),
                    ],
                    "object_numerators": [float(goc), float(guc), float(gtc)],
                    "predicted_object_area": int(area),
                }
            )

    extra = next(iterator, _END)
    if extra is not _END:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
        raise EvaluationAdapterError(
            f"prediction stream yielded more than the expected {expected} sample(s)"
        )

    if metrics is None:
        metrics = _metric_functions()
    oa, precision, recall, f1, iou = metrics["metrics_from_confusion"](*pixel_counts)
    completeness, correctness, f_boundary = metrics["boundary_metrics_from_statistics"](
        *boundary_counts
    )
    if predicted_area > 0:
        goc = object_sums[0] / predicted_area
        guc = object_sums[1] / predicted_area
        gtc = object_sums[2] / predicted_area
    else:
        # Empty object aggregation: None instead of a fabricated 0.0.
        goc = guc = gtc = None

    report = {
        "protocol": settings["protocol"],
        "split": split,
        "counts": {"expected": expected, "evaluated": len(id_list)},
        "samples": len(id_list),
        "complete_split": True,
        "aggregation": settings["aggregation"],
        "boundary_tolerance_pixels": tolerance,
        "metrics": {
            "f1": float(f1),
            "iou": float(iou),
            "f_boundary": float(f_boundary),
            "goc": None if goc is None else float(goc),
            "guc": None if guc is None else float(guc),
            "gtc": None if gtc is None else float(gtc),
            "oa": float(oa),
            "precision": float(precision),
            "recall": float(recall),
            "boundary_completeness": float(completeness),
            "boundary_correctness": float(correctness),
        },
        "pixel_counts": [int(value) for value in pixel_counts],
        "boundary_counts": [int(value) for value in boundary_counts],
        "predicted_object_area": int(predicted_area),
    }
    if include_per_image:
        report["per_image"] = per_image
    return report
