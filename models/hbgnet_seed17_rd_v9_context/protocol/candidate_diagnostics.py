"""Direction-level cut/fill candidate diagnostics for rd_v8_grok.

Single source of truth for the directional contingency table shared by
``common.evaluate`` (training validation) and ``evaluate.py`` (formal full-split
report).

Label convention (STRICT, identical to the verifier routing in
``tools.region_discrimination_v8_grok_fullloss``)::

    raw_positive   = raw_region_logit > 0
    sr_positive    = sr_region_logit  > 0
    changed        = delta != 0
    truth_positive = region_truth > 0.5
    cut_edit       = raw_positive & ~sr_positive & changed
    fill_edit      = ~raw_positive & sr_positive & changed
    edit           = cut_edit | fill_edit
    beneficial     = edit & (sr_positive == truth_positive)
    harmful        = edit & (sr_positive != truth_positive)

The frozen validation ``sigmoid(x) >= 0.5`` and formal test ``sigmoid(x) > 0.5``
booleans are deliberately NOT used for these labels. They can differ from the
strict ``logit > 0`` convention only at the ``logit == 0`` float boundary, where
``sigmoid`` rounds to exactly ``0.5``; keeping the labels on the routing
convention guarantees the accepted/rejected masks and the benefit labels cannot
disagree with each other.

On every actual edit pixel ``raw_positive != sr_positive``, so exactly one of
``beneficial``/``harmful`` holds. The four cells ``accepted_beneficial``,
``accepted_harmful``, ``rejected_beneficial`` and ``rejected_harmful`` therefore
partition the edit set, and the direction split (cut/fill) is likewise exact.

Zero-denominator rates
----------------------
A rate is undefined when its denominator count is zero. Two explicit policies
are provided and the caller must choose one:

* ``rates(counts, undefined=None)`` -> JSON ``null`` together with the
  always-present denominator count (used by the formal ``evaluate.py`` report).
* ``rates(counts, undefined=0.0)`` -> finite ``0.0`` together with the
  denominator count (used by ``common.evaluate``, whose values feed
  ``SummaryWriter.add_scalar`` in ``train.py`` and therefore must be numeric).

Denominator counts are never dropped: every rate's denominator is one of the
``counts`` keys, so an undefined rate is always accompanied by the count that
explains it.
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch


STRICT_THRESHOLD = 0.0
TRUTH_THRESHOLD = 0.5

# Canonical per-scope pixel counts. "cut" is foreground->background, "fill" is
# background->foreground; totals are the exact sums of the two directions.
EDIT_COUNT_KEYS = ('edit_pixels', 'cut_edit_pixels', 'fill_edit_pixels')
BENEFIT_COUNT_KEYS = (
    'beneficial_pixels', 'harmful_pixels',
    'cut_beneficial_pixels', 'cut_harmful_pixels',
    'fill_beneficial_pixels', 'fill_harmful_pixels',
)
ACCEPT_COUNT_KEYS = ('accepted_pixels', 'cut_accepted_pixels', 'fill_accepted_pixels')
REJECT_COUNT_KEYS = ('rejected_pixels', 'cut_rejected_pixels', 'fill_rejected_pixels')
ACCEPT_LABEL_COUNT_KEYS = (
    'accepted_beneficial_pixels', 'accepted_harmful_pixels',
    'cut_accepted_beneficial_pixels', 'cut_accepted_harmful_pixels',
    'fill_accepted_beneficial_pixels', 'fill_accepted_harmful_pixels',
)
REJECT_LABEL_COUNT_KEYS = (
    'rejected_beneficial_pixels', 'rejected_harmful_pixels',
    'cut_rejected_beneficial_pixels', 'cut_rejected_harmful_pixels',
    'fill_rejected_beneficial_pixels', 'fill_rejected_harmful_pixels',
)
COUNT_KEYS = (
    EDIT_COUNT_KEYS + BENEFIT_COUNT_KEYS + ACCEPT_COUNT_KEYS + REJECT_COUNT_KEYS +
    ACCEPT_LABEL_COUNT_KEYS + REJECT_LABEL_COUNT_KEYS
)

# v6 ``common.evaluate`` diagnostic names retained verbatim for compatibility.
LEGACY_ALIASES = {
    'beneficial_accepted_pixels': 'accepted_beneficial_pixels',
    'harmful_accepted_pixels': 'accepted_harmful_pixels',
    'cut_beneficial_accepted_pixels': 'cut_accepted_beneficial_pixels',
    'cut_harmful_accepted_pixels': 'cut_accepted_harmful_pixels',
    'fill_beneficial_accepted_pixels': 'fill_accepted_beneficial_pixels',
    'fill_harmful_accepted_pixels': 'fill_accepted_harmful_pixels',
    'beneficial_rejected_pixels': 'rejected_beneficial_pixels',
    'harmful_rejected_pixels': 'rejected_harmful_pixels',
    'cut_beneficial_rejected_pixels': 'cut_rejected_beneficial_pixels',
    'cut_harmful_rejected_pixels': 'cut_rejected_harmful_pixels',
    'fill_beneficial_rejected_pixels': 'fill_rejected_beneficial_pixels',
    'fill_harmful_rejected_pixels': 'fill_rejected_harmful_pixels',
}

# rate name -> (numerator count key, denominator count key). The denominator is
# always a COUNT_KEYS entry, so it is reported even when the rate is undefined.
RATE_DEFINITIONS = (
    ('acceptance_precision', 'accepted_beneficial_pixels', 'accepted_pixels'),
    ('accepted_beneficial_rate', 'accepted_beneficial_pixels', 'accepted_pixels'),
    ('accepted_harmful_rate', 'accepted_harmful_pixels', 'accepted_pixels'),
    ('rejected_beneficial_rate', 'rejected_beneficial_pixels', 'rejected_pixels'),
    ('rejected_harmful_rate', 'rejected_harmful_pixels', 'rejected_pixels'),
    ('beneficial_acceptance_rate', 'accepted_beneficial_pixels', 'beneficial_pixels'),
    ('harmful_rejection_rate', 'rejected_harmful_pixels', 'harmful_pixels'),
    ('cut_acceptance_precision', 'cut_accepted_beneficial_pixels', 'cut_accepted_pixels'),
    ('cut_beneficial_acceptance_rate', 'cut_accepted_beneficial_pixels', 'cut_beneficial_pixels'),
    ('cut_harmful_rejection_rate', 'cut_rejected_harmful_pixels', 'cut_harmful_pixels'),
    ('fill_acceptance_precision', 'fill_accepted_beneficial_pixels', 'fill_accepted_pixels'),
    ('fill_beneficial_acceptance_rate', 'fill_accepted_beneficial_pixels', 'fill_beneficial_pixels'),
    ('fill_harmful_rejection_rate', 'fill_rejected_harmful_pixels', 'fill_harmful_pixels'),
)
RATE_KEYS = tuple(name for name, _, _ in RATE_DEFINITIONS)

# net accepted benefit minus accepted harm, per scope.
NET_DEFINITIONS = (
    ('net_pixels_vs_raw', 'accepted_beneficial_pixels', 'accepted_harmful_pixels'),
    ('cut_net_pixels_vs_raw', 'cut_accepted_beneficial_pixels', 'cut_accepted_harmful_pixels'),
    ('fill_net_pixels_vs_raw', 'fill_accepted_beneficial_pixels', 'fill_accepted_harmful_pixels'),
)
NET_KEYS = tuple(name for name, _, _ in NET_DEFINITIONS)

LABEL_KEYS = ('raw_positive', 'sr_positive', 'changed', 'truth_positive',
              'cut_edit', 'fill_edit', 'edit', 'beneficial', 'harmful')


def _positive(tensor: torch.Tensor) -> torch.Tensor:
    """Strict ``> 0`` boolean; a boolean input is already a strict decision."""
    return tensor if tensor.dtype == torch.bool else tensor > STRICT_THRESHOLD


def strict_labels(raw, sr, delta, truth) -> Dict[str, torch.Tensor]:
    """Detached boolean direction labels under the strict routing convention.

    ``raw``/``sr``/``delta`` are Region logits/deltas; ``truth`` is the Region
    ground truth. All must share one rank-4 shape. Ground truth is only ever
    compared, never used as a model input.
    """
    tensors = {'raw': raw, 'sr': sr, 'delta': delta, 'truth': truth}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise ValueError(f'Direction labels require rank-4 tensors: {name}')
    if not (raw.shape == sr.shape == delta.shape == truth.shape):
        raise ValueError('Direction label tensors must share one shape')
    raw_positive = _positive(raw)
    sr_positive = _positive(sr)
    changed = delta if delta.dtype == torch.bool else delta != 0
    truth_positive = truth if truth.dtype == torch.bool else truth > TRUTH_THRESHOLD
    cut_edit = raw_positive & ~sr_positive & changed
    fill_edit = ~raw_positive & sr_positive & changed
    edit = cut_edit | fill_edit
    agree = sr_positive == truth_positive
    return {
        'raw_positive': raw_positive,
        'sr_positive': sr_positive,
        'changed': changed,
        'truth_positive': truth_positive,
        'cut_edit': cut_edit,
        'fill_edit': fill_edit,
        'edit': edit,
        'beneficial': edit & agree,
        'harmful': edit & ~agree,
    }


def _count(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_identities(counts: Mapping[str, int]) -> None:
    """Refuse a contingency table that is not complete, disjoint and consistent."""
    missing = [key for key in COUNT_KEYS if key not in counts]
    _require(not missing, f'Contingency table omits keys: {missing}')
    for key in COUNT_KEYS:
        value = counts[key]
        _require(type(value) is int and value >= 0,
                 f'Contingency counts must be non-negative ints: {key}={value!r}')

    def total(left: str, right: str) -> int:
        return counts[left] + counts[right]

    for parent, left, right in (
        ('edit_pixels', 'cut_edit_pixels', 'fill_edit_pixels'),
        ('beneficial_pixels', 'cut_beneficial_pixels', 'fill_beneficial_pixels'),
        ('harmful_pixels', 'cut_harmful_pixels', 'fill_harmful_pixels'),
        ('accepted_pixels', 'cut_accepted_pixels', 'fill_accepted_pixels'),
        ('rejected_pixels', 'cut_rejected_pixels', 'fill_rejected_pixels'),
        ('accepted_beneficial_pixels', 'cut_accepted_beneficial_pixels',
         'fill_accepted_beneficial_pixels'),
        ('accepted_harmful_pixels', 'cut_accepted_harmful_pixels',
         'fill_accepted_harmful_pixels'),
        ('rejected_beneficial_pixels', 'cut_rejected_beneficial_pixels',
         'fill_rejected_beneficial_pixels'),
        ('rejected_harmful_pixels', 'cut_rejected_harmful_pixels',
         'fill_rejected_harmful_pixels'),
    ):
        _require(total(left, right) == counts[parent],
                 f'Direction split is not complete: {left}+{right} != {parent}')

    for parent, left, right in (
        ('edit_pixels', 'beneficial_pixels', 'harmful_pixels'),
        ('edit_pixels', 'accepted_pixels', 'rejected_pixels'),
        ('cut_edit_pixels', 'cut_beneficial_pixels', 'cut_harmful_pixels'),
        ('fill_edit_pixels', 'fill_beneficial_pixels', 'fill_harmful_pixels'),
        ('cut_edit_pixels', 'cut_accepted_pixels', 'cut_rejected_pixels'),
        ('fill_edit_pixels', 'fill_accepted_pixels', 'fill_rejected_pixels'),
        ('beneficial_pixels', 'accepted_beneficial_pixels', 'rejected_beneficial_pixels'),
        ('harmful_pixels', 'accepted_harmful_pixels', 'rejected_harmful_pixels'),
        ('accepted_pixels', 'accepted_beneficial_pixels', 'accepted_harmful_pixels'),
        ('rejected_pixels', 'rejected_beneficial_pixels', 'rejected_harmful_pixels'),
        ('cut_accepted_pixels', 'cut_accepted_beneficial_pixels',
         'cut_accepted_harmful_pixels'),
        ('cut_rejected_pixels', 'cut_rejected_beneficial_pixels',
         'cut_rejected_harmful_pixels'),
        ('fill_accepted_pixels', 'fill_accepted_beneficial_pixels',
         'fill_accepted_harmful_pixels'),
        ('fill_rejected_pixels', 'fill_rejected_beneficial_pixels',
         'fill_rejected_harmful_pixels'),
    ):
        _require(total(left, right) == counts[parent],
                 f'Benefit/decision split is not complete: {left}+{right} != {parent}')

    _require(counts['accepted_pixels'] <= counts['edit_pixels'],
             'Accepted pixels must be a subset of edit pixels')


def new_counts() -> Dict[str, int]:
    """A zeroed accumulator over the canonical count keys."""
    return dict.fromkeys(COUNT_KEYS, 0)


def accumulate(total: Dict[str, int], counts: Mapping[str, int]) -> Dict[str, int]:
    """Add one validated contingency table into an accumulator, in place."""
    check_identities(counts)
    for key in COUNT_KEYS:
        total[key] = int(total.get(key, 0)) + int(counts[key])
    return total


def contingency(labels: Mapping[str, torch.Tensor], keep_cut, keep_fill) -> Dict[str, int]:
    """Full cut/fill contingency table for one batch or one image.

    ``keep_cut``/``keep_fill`` are the verifier's direction-specific accepted
    masks. They must be subsets of their candidate direction; this is checked
    rather than assumed so the "verifier cannot create edits" contract is
    enforced at diagnostic time too.
    """
    missing = [key for key in LABEL_KEYS if key not in labels]
    _require(not missing, f'Direction labels omit keys: {missing}')
    cut = labels['cut_edit']
    fill = labels['fill_edit']
    edit = labels['edit']
    beneficial = labels['beneficial']
    harmful = labels['harmful']
    for name, mask in (('keep_cut', keep_cut), ('keep_fill', keep_fill)):
        if not isinstance(mask, torch.Tensor) or mask.shape != cut.shape:
            raise ValueError(f'{name} must be a tensor matching the candidate shape')
    keep_cut = keep_cut.to(dtype=torch.bool)
    keep_fill = keep_fill.to(dtype=torch.bool)
    if bool((keep_cut & ~cut).any()) or bool((keep_fill & ~fill).any()):
        raise ValueError('Accepted masks must be subsets of their candidate direction')
    cut_accepted = keep_cut & cut
    fill_accepted = keep_fill & fill
    accepted = cut_accepted | fill_accepted
    rejected = edit & ~accepted
    cut_rejected = cut & ~cut_accepted
    fill_rejected = fill & ~fill_accepted
    counts = {
        'edit_pixels': _count(edit),
        'cut_edit_pixels': _count(cut),
        'fill_edit_pixels': _count(fill),
        'beneficial_pixels': _count(beneficial),
        'harmful_pixels': _count(harmful),
        'cut_beneficial_pixels': _count(cut & beneficial),
        'cut_harmful_pixels': _count(cut & harmful),
        'fill_beneficial_pixels': _count(fill & beneficial),
        'fill_harmful_pixels': _count(fill & harmful),
        'accepted_pixels': _count(accepted),
        'cut_accepted_pixels': _count(cut_accepted),
        'fill_accepted_pixels': _count(fill_accepted),
        'rejected_pixels': _count(rejected),
        'cut_rejected_pixels': _count(cut_rejected),
        'fill_rejected_pixels': _count(fill_rejected),
        'accepted_beneficial_pixels': _count(accepted & beneficial),
        'accepted_harmful_pixels': _count(accepted & harmful),
        'cut_accepted_beneficial_pixels': _count(cut_accepted & beneficial),
        'cut_accepted_harmful_pixels': _count(cut_accepted & harmful),
        'fill_accepted_beneficial_pixels': _count(fill_accepted & beneficial),
        'fill_accepted_harmful_pixels': _count(fill_accepted & harmful),
        'rejected_beneficial_pixels': _count(rejected & beneficial),
        'rejected_harmful_pixels': _count(rejected & harmful),
        'cut_rejected_beneficial_pixels': _count(cut_rejected & beneficial),
        'cut_rejected_harmful_pixels': _count(cut_rejected & harmful),
        'fill_rejected_beneficial_pixels': _count(fill_rejected & beneficial),
        'fill_rejected_harmful_pixels': _count(fill_rejected & harmful),
    }
    check_identities(counts)
    return counts


def batch_contingency(raw, sr, delta, truth, keep_cut, keep_fill) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Convenience wrapper: strict labels plus the validated contingency table."""
    labels = strict_labels(raw, sr, delta, truth)
    return labels, contingency(labels, keep_cut, keep_fill)


def legacy_aliases(counts: Mapping[str, int]) -> Dict[str, int]:
    """Map the canonical counts back onto the retained v6 diagnostic names."""
    return {legacy: int(counts[canonical]) for legacy, canonical in LEGACY_ALIASES.items()}


def rates(counts: Mapping[str, int], *, undefined=None) -> Dict[str, object]:
    """Named rates with the caller-selected zero-denominator policy.

    ``undefined=None`` yields JSON ``null``; ``undefined=0.0`` yields a finite
    ``0.0`` for ``add_scalar``-style numeric consumers. The denominator count
    stays in ``counts`` either way.
    """
    values: Dict[str, object] = {}
    for name, numerator, denominator in RATE_DEFINITIONS:
        total = int(counts[denominator])
        values[name] = undefined if total == 0 else float(counts[numerator]) / total
    return values


def net_pixels(counts: Mapping[str, int]) -> Dict[str, int]:
    """Accepted benefit minus accepted harm, per scope (always defined)."""
    return {name: int(counts[benefit]) - int(counts[harm])
            for name, benefit, harm in NET_DEFINITIONS}


__all__ = [
    'STRICT_THRESHOLD', 'TRUTH_THRESHOLD', 'COUNT_KEYS', 'RATE_KEYS', 'NET_KEYS',
    'LEGACY_ALIASES', 'RATE_DEFINITIONS', 'NET_DEFINITIONS', 'LABEL_KEYS',
    'strict_labels', 'contingency', 'batch_contingency', 'check_identities',
    'new_counts', 'accumulate', 'legacy_aliases', 'rates', 'net_pixels',
]
