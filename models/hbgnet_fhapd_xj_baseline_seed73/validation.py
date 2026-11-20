"""Independent validation for the FHAPD seed73 HBGNet training packages.

This module is the single validation entry point for both runtime identities.
``common.evaluate`` delegates here, and the runtime identity comes from the
package's ``runtime_identity.MODEL_KIND`` (re-exported by ``common``), with an
explicit ``cfg['runtime_identity']['MODEL_KIND']`` override used by tests:

``baseline``
    The official HBGNet Field forward only.  One ``final`` route is measured
    from the first forward output.  ``last_auxiliary`` is never read and no
    module diagnostics are required or reported.

``combined``
    SR v2 + rd_v9_context candidate verifier.  ``raw``/``sr``/``final`` route
    metrics come from ``model.last_auxiliary`` together with the frozen
    ``protocol.candidate_diagnostics`` semantics: strict ``logit > 0``
    direction routing, the canonical cut/fill contingency table and its
    retained v6 aliases, the JSON-null rate view (``undefined=None``), net
    accepted-benefit pixels, the thin background/foreground proxy and the far
    pixel (distance > 10) diagnostics.  ``delta``/``valid_count`` statistics
    (``valid_pixels``, ``delta_abs_mean``, ``delta_saturation_fraction``) are
    preserved from the rd_v9 validation.

Model inputs are always ``batch['image']`` only; the Region ground truth is
never a forward input, only compared.  The split identity, sample count and
spatial size are read from the frozen configuration through
``common.config(ROOT / 'config_protocol_b.yaml')`` and
``protocol.evaluation_adapter``; no split count such as the former
AI4Boundaries sizes is ever hardcoded.  The boundary tolerance is the genuine
integer from ``resolve_boundary_tolerance``.  The region decision is the FHAPD
data-adapter rule, the strict ``sigmoid > 0.5`` (``REGION_THRESHOLD`` /
``THRESHOLD_COMPARISON``); direction labels keep the frozen strict logit-zero
routing convention.

Split identity
--------------
Full validation (``max_batches is None``) must cover the frozen validation
manifest exactly: the batch ``image_id`` sequence must equal the manifest
lines in order with no missing, duplicated or extra sample, and the manifest
length must equal ``dataset.expected_counts.val``.  ``max_batches=N`` is the
smoke path: at most N batches are consumed, the consumed ids must be the
manifest prefix, and ``complete_split`` is true only when the whole split was
in fact covered.

Returned mapping
----------------
Every value is a flat JSON-safe scalar (int/float/None/bool) so the training
record and the TensorBoard writer can consume it directly.  Pixel and boundary
raw counts are plain numeric fields, ``tp/fp/fn/tn`` plus
``matched_gt_boundary/gt_boundary/matched_pred_boundary/pred_boundary`` (the
``raw_``/``sr_`` prefixed routes carry the same fields).  A split without any
predicted object reports ``goc/guc/gtc = None`` next to
``predicted_object_area = 0`` instead of a fabricated ``0.0``, and an
undefined rate is ``None`` next to its always-present denominator count.

No test split, no dataset loader, no training and no input/output side effect
belongs to this module: it consumes prepared batches and returns one mapping.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from protocol.evaluation_adapter import resolve_boundary_tolerance, resolve_input_size

REGION_THRESHOLD = 0.5
THRESHOLD_COMPARISON = '>'
MODEL_KINDS = ('baseline', 'combined')
ROUTES = ('raw', 'sr', 'final')
FAR_PIXEL_THRESHOLD = 10
AUXILIARY_SCALAR_KEYS = (
    'raw_region_logits', 'sr_region_logits', 'final_region_logits', 'delta',
    'edit', 'cut_edit', 'fill_edit', 'keep', 'cut_keep', 'fill_keep', 'valid_count',
)
AUXILIARY_KEYS = AUXILIARY_SCALAR_KEYS + ('verifier_logits',)
DIRECTION_MASK_KEYS = (('edit', 'edit'), ('cut_edit', 'cut_edit'), ('fill_edit', 'fill_edit'))
DIAGNOSTIC_KEYS = (
    'thin_bg_candidates', 'thin_bg_raw_fp', 'thin_bg_sr_fp', 'thin_bg_final_fp',
    'thin_fg_candidates', 'thin_fg_raw_fn', 'thin_fg_sr_fn', 'thin_fg_final_fn',
    'far_pixels', 'raw_far_fp', 'sr_far_fp', 'final_far_fp',
    'raw_far_fn', 'sr_far_fn', 'final_far_fn',
)
CANDIDATE_SMOKE_KEYS = (
    'edit_pixels', 'cut_edit_pixels', 'fill_edit_pixels',
    'accepted_pixels', 'beneficial_pixels', 'harmful_pixels',
)

_END = object()

__all__ = [
    'CANDIDATE_SMOKE_KEYS', 'FAR_PIXEL_THRESHOLD', 'MODEL_KINDS', 'REGION_THRESHOLD',
    'ROUTES', 'THRESHOLD_COMPARISON', 'canonical_config', 'evaluate_validation',
]


def canonical_config(path=None):
    """Return the frozen package configuration through ``common.config``."""
    import common
    return common.config(ROOT / 'config_protocol_b.yaml' if path is None else Path(path))


def _module_model_kind():
    try:
        import runtime_identity
    except ImportError:
        runtime_kind = None
    else:
        runtime_kind = getattr(runtime_identity, 'MODEL_KIND', None)
    if runtime_kind is not None:
        return runtime_kind
    try:
        import common
    except ImportError:
        return None
    return getattr(common, 'MODEL_KIND', None)


def _model_kind(cfg):
    declared = None
    identity = cfg.get('runtime_identity') if isinstance(cfg, dict) else None
    if isinstance(identity, dict) and 'MODEL_KIND' in identity:
        declared = identity['MODEL_KIND']
    if declared is None:
        declared = _module_model_kind()
    if declared not in MODEL_KINDS:
        raise ValueError("MODEL_KIND must be 'baseline' or 'combined'")
    return declared


def _val_identity(cfg):
    """Frozen validation manifest ids (order and uniqueness) and its count."""
    dataset = cfg.get('dataset')
    if not isinstance(dataset, dict):
        raise ValueError('Validation requires a dataset mapping')
    manifests = dataset.get('manifests')
    if not isinstance(manifests, dict) or not isinstance(manifests.get('val'), str):
        raise ValueError('dataset.manifests.val must name the frozen validation manifest')
    lines = Path(manifests['val']).read_text(encoding='utf-8').splitlines()
    ids = [line.strip() for line in lines if line.strip()]
    if not ids:
        raise ValueError('Frozen validation manifest is empty')
    for ident in ids:
        if (not ident or ident != ident.strip() or ident in ('.', '..')
                or any(character in ident for character in ('/', '\\', '\x00'))):
            raise ValueError(f'Unsafe frozen validation sample id: {ident!r}')
    if len(set(ids)) != len(ids):
        raise ValueError('Frozen validation manifest contains duplicate sample ids')
    counts = dataset.get('expected_counts')
    if not isinstance(counts, dict) or type(counts.get('val')) is not int:
        raise ValueError('dataset.expected_counts.val must be an integer sample count')
    expected = counts['val']
    if expected < 1:
        raise ValueError('dataset.expected_counts.val must be positive')
    if expected != len(ids):
        raise ValueError(
            f'dataset.expected_counts.val ({expected}) does not match the frozen '
            f'validation manifest ({len(ids)} ids)')
    return tuple(ids), expected


def _max_delta(cfg):
    model = cfg.get('model')
    refinement = model.get('refinement') if isinstance(model, dict) else None
    value = refinement.get('max_delta') if isinstance(refinement, dict) else None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0):
        raise ValueError('Combined validation requires model.refinement.max_delta')
    return float(value)


def _thin_background_proxy(truth):
    """Zero-filled two-sided foreground geometry; not instance labels or a loss."""
    height, width = truth.shape[-2:]

    def sample(dy, dx):
        result = np.zeros_like(truth, dtype=bool)
        if abs(dy) >= height or abs(dx) >= width:
            return result
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        result[..., y0:y1, x0:x1] = truth[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        return result

    candidate = np.zeros_like(truth, dtype=bool)
    for radius in (2, 4):
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            candidate |= sample(radius * dy, radius * dx) & sample(-radius * dy, -radius * dx)
    return candidate & ~truth


def _route_block(pixel, boundary, objects):
    """Flat per-route metrics; micro pixel/boundary counts, area-weighted objects."""
    from protocol.metrics import boundary_metrics_from_statistics, metrics_from_confusion
    oa, precision, recall, f1, iou = metrics_from_confusion(*map(int, pixel))
    completeness, correctness, bf1 = boundary_metrics_from_statistics(*map(int, boundary))
    area = int(objects[3])
    if area > 0:
        goc, guc, gtc = (float(objects[0] / area), float(objects[1] / area),
                         float(objects[2] / area))
    else:
        # No predicted object: the error is not defined and must not be filled with 0.
        goc = guc = gtc = None
    return {
        'iou': float(iou), 'f1': float(f1), 'oa': float(oa),
        'precision': float(precision), 'recall': float(recall), 'bf1': float(bf1),
        'boundary_completeness': float(completeness),
        'boundary_correctness': float(correctness),
        'goc': goc, 'guc': guc, 'gtc': gtc, 'predicted_object_area': area,
        'tp': int(pixel[0]), 'fp': int(pixel[1]), 'fn': int(pixel[2]), 'tn': int(pixel[3]),
        'matched_gt_boundary': int(boundary[0]), 'gt_boundary': int(boundary[1]),
        'matched_pred_boundary': int(boundary[2]), 'pred_boundary': int(boundary[3]),
    }


def _check_flat_result(result):
    """Refuse anything the training record / TensorBoard writer cannot consume."""
    for name, value in result.items():
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                raise FloatingPointError(f'Non-finite validation diagnostic: {name}')
            continue
        raise TypeError(
            f'Validation diagnostics must be flat int/float/None/bool: {name}={value!r}')


@torch.no_grad()
def evaluate_validation(model, batches, device, max_batches=None, *, cfg=None):
    """Validate ``model`` on prepared ``batches`` and return one flat mapping.

    ``batches`` is any iterable of mappings with ``image`` (forward input only),
    ``region`` (compared, never forwarded) and ``image_id``.  ``cfg`` is the
    frozen configuration; when omitted it is loaded through
    ``common.config(ROOT / 'config_protocol_b.yaml')`` so the package identity,
    split count, spatial size and boundary tolerance all come from the frozen
    sources.
    """
    if cfg is None:
        cfg = canonical_config()
    if not isinstance(cfg, dict):
        raise ValueError('Validation requires the frozen FHAPD configuration mapping')
    kind = _model_kind(cfg)
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError('max_batches must be a positive integer or None')
    evaluation = cfg.get('evaluation')
    if isinstance(evaluation, dict):
        if 'region_threshold' in evaluation:
            threshold = evaluation['region_threshold']
            if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                    or float(threshold) != REGION_THRESHOLD):
                raise ValueError('evaluation.region_threshold must be exactly 0.5')
        if evaluation.get('threshold_comparison', THRESHOLD_COMPARISON) != THRESHOLD_COMPARISON:
            raise ValueError("evaluation.threshold_comparison must be '>'")
    tolerance = resolve_boundary_tolerance(cfg)
    input_size = tuple(resolve_input_size(cfg))
    frozen_ids, expected_samples = _val_identity(cfg)
    max_delta = _max_delta(cfg) if kind == 'combined' else None
    routes = ('final',) if kind == 'baseline' else ROUTES

    from protocol.metrics import (binary_confusion, boundary_statistics, mask_boundary,
                                  object_error_sums)
    if kind == 'combined':
        from protocol.candidate_diagnostics import (accumulate, contingency, legacy_aliases,
                                                    net_pixels, new_counts, rates, strict_labels)
        from scipy.ndimage import distance_transform_edt

    model.eval()
    pixel = {name: np.zeros(4, dtype=np.int64) for name in routes}
    boundary = {name: np.zeros(4, dtype=np.int64) for name in routes}
    objects = {name: np.zeros(4, dtype=np.float64) for name in routes}
    candidate_counts = new_counts() if kind == 'combined' else None
    diagnostics = dict.fromkeys(DIAGNOSTIC_KEYS, 0) if kind == 'combined' else None
    count = total_pixels = valid_pixels = saturated = batches_consumed = 0
    delta_abs_sum = 0.0
    seen = []

    try:
        iterator = iter(batches)
    except TypeError as exc:
        raise ValueError('batches must be an iterable of validation batch mappings') from exc
    stopped = False
    while True:
        if max_batches is not None and batches_consumed >= max_batches:
            stopped = True
            break
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if not isinstance(batch, dict):
            raise ValueError('Validation batches must be mappings')
        batch_ids = batch.get('image_id')
        if isinstance(batch_ids, (str, bytes)) or not isinstance(batch_ids, (list, tuple)):
            raise ValueError("Each validation batch must carry an 'image_id' list")
        if not batch_ids or any(not isinstance(ident, str) or not ident for ident in batch_ids):
            raise ValueError('Validation image ids must be non-empty strings')
        expected_slice = frozen_ids[count:count + len(batch_ids)]
        if tuple(batch_ids) != expected_slice:
            raise ValueError(
                f'Validation ids/order mismatch at batch {batches_consumed}: '
                f'expected {list(expected_slice)!r}, got {list(batch_ids)!r}')
        seen.extend(batch_ids)
        # Ground truth is never a forward input; only the image is.
        out = model(batch['image'].to(device))[0]
        if (not isinstance(out, torch.Tensor) or out.ndim != 4 or out.shape[1] != 1
                or out.shape[0] != len(batch_ids)):
            raise ValueError('Region output must be a [B,1,H,W] tensor matching the batch ids')
        if tuple(out.shape[2:]) != input_size:
            raise ValueError('Region output does not match the resolved dataset.input_size')
        if not torch.isfinite(out).all().item():
            raise FloatingPointError('Non-finite validation output')
        truth_tensor = batch.get('region')
        if not isinstance(truth_tensor, torch.Tensor) or truth_tensor.shape != out.shape:
            raise ValueError('Region target must match the Region output [B,1,H,W]')
        truth = truth_tensor.detach().cpu().numpy() >= .5
        if truth.ndim != 4 or truth.shape[1] != 1:
            raise ValueError('Region target must be [B,1,H,W]')

        if kind == 'combined':
            auxiliary = getattr(model, 'last_auxiliary', None)
            if not isinstance(auxiliary, dict) or any(key not in auxiliary for key in AUXILIARY_KEYS):
                raise ValueError('Model must expose raw/SR/final and directional verifier diagnostics')
            tensors = {key: auxiliary[key] for key in AUXILIARY_SCALAR_KEYS}
            for key, tensor in tensors.items():
                if not isinstance(tensor, torch.Tensor) or tensor.shape != out.shape:
                    raise ValueError(f'Auxiliary must match region logits [B,1,H,W]: {key}')
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise FloatingPointError(f'Non-finite validation auxiliary: {key}')
            for key in ('raw_region_logits', 'sr_region_logits', 'final_region_logits', 'delta'):
                if not tensors[key].is_floating_point():
                    raise ValueError(f'Combined validation requires floating point {key}')
            verifier_logits = auxiliary['verifier_logits']
            expected_verifier_shape = (out.shape[0], 2, out.shape[2], out.shape[3])
            if (not isinstance(verifier_logits, torch.Tensor)
                    or verifier_logits.shape != expected_verifier_shape
                    or not verifier_logits.is_floating_point()):
                raise ValueError('Verifier diagnostics must be floating [B,2,H,W] logits')
            if not torch.isfinite(verifier_logits).all():
                raise FloatingPointError('Non-finite validation verifier logits')
            if not torch.equal((tensors['cut_edit'].bool() | tensors['fill_edit'].bool()),
                               tensors['edit'].bool()):
                raise ValueError('Directional edit diagnostics do not match edit union')
            if not torch.equal((tensors['cut_keep'].bool() | tensors['fill_keep'].bool()),
                               tensors['keep'].bool()):
                raise ValueError('Directional keep diagnostics do not match keep union')
            if not torch.equal(out.detach(), tensors['final_region_logits'].detach()):
                raise ValueError('Final auxiliary does not match the primary region output')
            raw = tensors['raw_region_logits'].detach()
            sr = tensors['sr_region_logits'].detach()
            final = tensors['final_region_logits'].detach()
            delta = tensors['delta'].detach()
            valid = tensors['valid_count'].detach()
            predictions = {
                'raw': raw.sigmoid().cpu().numpy() > REGION_THRESHOLD,
                'sr': sr.sigmoid().cpu().numpy() > REGION_THRESHOLD,
                'final': final.sigmoid().cpu().numpy() > REGION_THRESHOLD,
            }
            cpu = lambda tensor: tensor.detach().cpu()
            # Route labels are recomputed from the strict routing convention and
            # refused if they disagree with the model's own candidate masks.
            labels = strict_labels(cpu(raw), cpu(sr), cpu(delta),
                                   torch.from_numpy(truth.astype(np.float32)))
            for key, name in DIRECTION_MASK_KEYS:
                if not torch.equal(tensors[key].detach().cpu().bool(), labels[name]):
                    raise ValueError(f'Model routing mask disagrees with strict direction labels: {key}')
            accumulate(candidate_counts, contingency(
                labels, cpu(tensors['cut_keep']).bool(), cpu(tensors['fill_keep']).bool()))
            n_valid = int((valid > 0).sum().item())
            valid_pixels += n_valid
            if n_valid:
                absolute = delta[valid > 0].abs()
                delta_abs_sum += float(absolute.double().sum().item())
                saturated += int((absolute >= .95 * max_delta).sum().item())
            candidate = _thin_background_proxy(truth)
            foreground_gap = _thin_background_proxy(~truth)
            diagnostics['thin_bg_candidates'] += int(candidate.sum())
            diagnostics['thin_fg_candidates'] += int(foreground_gap.sum())
            for name, prediction in predictions.items():
                diagnostics['thin_bg_' + name + '_fp'] += int((candidate & prediction).sum())
                diagnostics['thin_fg_' + name + '_fn'] += int((foreground_gap & ~prediction).sum())
        else:
            predictions = {'final': out.sigmoid().cpu().numpy() > REGION_THRESHOLD}
        if predictions['final'].shape != truth.shape:
            raise ValueError('Region prediction/target shape mismatch')

        for index in range(truth.shape[0]):
            target = truth[index, 0]
            for name in routes:
                prediction = predictions[name][index, 0]
                pixel[name] += binary_confusion(prediction, target)
                boundary[name] += boundary_statistics(prediction, target, tolerance)
                goc, guc, gtc, area = object_error_sums(prediction, target)
                objects[name] += (float(goc), float(guc), float(gtc), float(area))
            if kind == 'combined':
                gt_boundary = mask_boundary(target)
                far = (distance_transform_edt(~gt_boundary) > FAR_PIXEL_THRESHOLD
                       if gt_boundary.any() else np.ones_like(target, dtype=bool))
                diagnostics['far_pixels'] += int(far.sum())
                for name in routes:
                    prediction = predictions[name][index, 0]
                    diagnostics[name + '_far_fp'] += int((far & prediction & ~target).sum())
                    diagnostics[name + '_far_fn'] += int((far & ~prediction & target).sum())
        total_pixels += int(predictions['final'].size)
        count += len(batch_ids)
        batches_consumed += 1

    if stopped:
        # One probe decides whether the smoke subset stopped short of the loader end.
        truncated = next(iterator, _END) is not _END
        if truncated:
            close = getattr(iterator, 'close', None)
            if callable(close):
                close()
    else:
        truncated = False
    if count == 0:
        raise ValueError('Validation must contain at least one sample')
    if max_batches is None:
        if count != expected_samples or tuple(seen) != frozen_ids:
            raise ValueError(
                f'Full validation must cover the frozen validation split exactly: '
                f'{count} of {expected_samples} samples')
        complete_split = True
    else:
        complete_split = (not truncated) and count == expected_samples

    result = {}
    for name in routes:
        block = _route_block(pixel[name], boundary[name], objects[name])
        prefix = '' if name == 'final' else name + '_'
        for key, value in block.items():
            result[prefix + key] = value
    result['samples'] = int(count)
    result['expected_samples'] = int(expected_samples)
    result['batches_consumed'] = int(batches_consumed)
    result['complete_split'] = bool(complete_split)
    result['total_pixels'] = int(total_pixels)
    result['checkpoint_score'] = .7 * result['iou'] + .3 * result['bf1']
    if kind == 'combined':
        result['valid_pixels'] = int(valid_pixels)
        result['delta_abs_mean'] = float(delta_abs_sum / valid_pixels) if valid_pixels else 0.0
        result['delta_saturation_fraction'] = (float(saturated / valid_pixels)
                                               if valid_pixels else 0.0)
        result.update(candidate_counts)
        result.update(legacy_aliases(candidate_counts))
        result.update(rates(candidate_counts, undefined=None))
        result.update(net_pixels(candidate_counts))
        result.update(diagnostics)
    _check_flat_result(result)
    return result
