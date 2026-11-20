"""Deterministic CPU acceptance tests for the rd_v9_context candidate diagnostics.

Synthetic only: no dataset, no CUDA, no training and no test split. The tests
pin the full cut/fill directional contingency table, the complete/disjoint/subset
identities, the retained v6 diagnostic aliases, both zero-denominator rate
policies, and the "ground truth is never a forward input" call sites.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent if ROOT.name == 'package' else ROOT.parents[1]
for path in (ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from protocol.candidate_diagnostics import (COUNT_KEYS, LEGACY_ALIASES, RATE_DEFINITIONS,
                                            accumulate, check_identities, contingency,
                                            legacy_aliases, net_pixels, new_counts, rates,
                                            strict_labels)
from tools.region_discrimination_v9 import (br_ha_curve, calibration_logit_threshold,
                                            candidate_masks as v9_candidate_masks,
                                            directional_labels as v9_directional_labels,
                                            monotone_affine_calibration)


def check(name, condition):
    if not condition:
        raise AssertionError(name)


# One 8-pixel row that exercises every cut/fill x beneficial/harmful cell plus
# the strict logit == 0 boundary and two no-edit cases.
RAW = [2.0, 2.0, -2.0, -2.0, 2.0, -2.0, 2.0, 0.0]
SR = [-1.0, -1.0, 1.0, 1.0, 2.0, -2.0, 2.0, 1.0]
DELTA = [-3.0, -3.0, 3.0, 3.0, 0.0, 0.0, 1.0, 1.0]
TRUTH = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
KEEP_CUT = [1, 1, 0, 0, 0, 0, 0, 0]
KEEP_FILL = [0, 0, 0, 1, 0, 0, 0, 0]

EXPECTED = {
    'edit_pixels': 5, 'cut_edit_pixels': 2, 'fill_edit_pixels': 3,
    'beneficial_pixels': 3, 'harmful_pixels': 2,
    'cut_beneficial_pixels': 1, 'cut_harmful_pixels': 1,
    'fill_beneficial_pixels': 2, 'fill_harmful_pixels': 1,
    'accepted_pixels': 3, 'cut_accepted_pixels': 2, 'fill_accepted_pixels': 1,
    'rejected_pixels': 2, 'cut_rejected_pixels': 0, 'fill_rejected_pixels': 2,
    'accepted_beneficial_pixels': 1, 'accepted_harmful_pixels': 2,
    'cut_accepted_beneficial_pixels': 1, 'cut_accepted_harmful_pixels': 1,
    'fill_accepted_beneficial_pixels': 0, 'fill_accepted_harmful_pixels': 1,
    'rejected_beneficial_pixels': 2, 'rejected_harmful_pixels': 0,
    'cut_rejected_beneficial_pixels': 0, 'cut_rejected_harmful_pixels': 0,
    'fill_rejected_beneficial_pixels': 2, 'fill_rejected_harmful_pixels': 0,
}
EXPECTED_RATES = {
    'acceptance_precision': 1 / 3, 'accepted_beneficial_rate': 1 / 3,
    'accepted_harmful_rate': 2 / 3, 'rejected_beneficial_rate': 1.0,
    'rejected_harmful_rate': 0.0, 'beneficial_acceptance_rate': 1 / 3,
    'harmful_rejection_rate': 0.0, 'cut_acceptance_precision': 1 / 2,
    'cut_beneficial_acceptance_rate': 1.0, 'cut_harmful_rejection_rate': 0.0,
    'fill_acceptance_precision': 0.0, 'fill_beneficial_acceptance_rate': 0.0,
    'fill_harmful_rejection_rate': 0.0,
}


def tensor(values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1)


def fixture():
    raw, sr, delta, truth = (tensor(RAW), tensor(SR), tensor(DELTA), tensor(TRUTH))
    keep_cut = torch.tensor(KEEP_CUT, dtype=torch.bool).reshape(1, 1, 1, -1)
    keep_fill = torch.tensor(KEEP_FILL, dtype=torch.bool).reshape(1, 1, 1, -1)
    return raw, sr, delta, truth, keep_cut, keep_fill


def brute_counts(raw, sr, delta, truth, keep_cut, keep_fill):
    """Independent NumPy reference for the same contingency table."""
    import numpy as np
    raw = raw.numpy(); sr = sr.numpy(); delta = delta.numpy(); truth = truth.numpy()
    keep_cut = keep_cut.numpy(); keep_fill = keep_fill.numpy()
    raw_pos = raw > 0; sr_pos = sr > 0; changed = delta != 0; truth_pos = truth > 0.5
    cut = raw_pos & ~sr_pos & changed
    fill = ~raw_pos & sr_pos & changed
    edit = cut | fill
    agree = sr_pos == truth_pos
    beneficial = edit & agree
    harmful = edit & ~agree
    cut_acc = keep_cut & cut
    fill_acc = keep_fill & fill
    accepted = cut_acc | fill_acc
    rejected = edit & ~accepted
    cut_rej = cut & ~cut_acc
    fill_rej = fill & ~fill_acc

    def n(mask):
        return int(np.count_nonzero(mask))

    return {
        'edit_pixels': n(edit), 'cut_edit_pixels': n(cut), 'fill_edit_pixels': n(fill),
        'beneficial_pixels': n(beneficial), 'harmful_pixels': n(harmful),
        'cut_beneficial_pixels': n(cut & beneficial), 'cut_harmful_pixels': n(cut & harmful),
        'fill_beneficial_pixels': n(fill & beneficial), 'fill_harmful_pixels': n(fill & harmful),
        'accepted_pixels': n(accepted), 'cut_accepted_pixels': n(cut_acc),
        'fill_accepted_pixels': n(fill_acc),
        'rejected_pixels': n(rejected), 'cut_rejected_pixels': n(cut_rej),
        'fill_rejected_pixels': n(fill_rej),
        'accepted_beneficial_pixels': n(accepted & beneficial),
        'accepted_harmful_pixels': n(accepted & harmful),
        'cut_accepted_beneficial_pixels': n(cut_acc & beneficial),
        'cut_accepted_harmful_pixels': n(cut_acc & harmful),
        'fill_accepted_beneficial_pixels': n(fill_acc & beneficial),
        'fill_accepted_harmful_pixels': n(fill_acc & harmful),
        'rejected_beneficial_pixels': n(rejected & beneficial),
        'rejected_harmful_pixels': n(rejected & harmful),
        'cut_rejected_beneficial_pixels': n(cut_rej & beneficial),
        'cut_rejected_harmful_pixels': n(cut_rej & harmful),
        'fill_rejected_beneficial_pixels': n(fill_rej & beneficial),
        'fill_rejected_harmful_pixels': n(fill_rej & harmful),
    }


def check_strict_labels():
    raw, sr, delta, truth, _, _ = fixture()
    labels = strict_labels(raw, sr, delta, truth)
    expected = {
        'cut_edit': [True, True, False, False, False, False, False, False],
        'fill_edit': [False, False, True, True, False, False, False, True],
        'beneficial': [False, True, True, False, False, False, False, True],
        'harmful': [True, False, False, True, False, False, False, False],
        'edit': [True, True, True, True, False, False, False, True],
    }
    for key, values in expected.items():
        want = torch.tensor(values, dtype=torch.bool).reshape(1, 1, 1, -1)
        check(f'strict label {key}', torch.equal(labels[key], want))
    check('beneficial/harmful partition edit',
          bool((labels['beneficial'] | labels['harmful']).equal(labels['edit']))
          and not bool((labels['beneficial'] & labels['harmful']).any()))
    check('cut/fill partition edit',
          bool((labels['cut_edit'] | labels['fill_edit']).equal(labels['edit']))
          and not bool((labels['cut_edit'] & labels['fill_edit']).any()))
    # Strict logit == 0 is NOT positive, while the frozen sigmoid >= 0.5 rule is.
    check('sigmoid(0) == 0.5 exactly', float(torch.sigmoid(torch.tensor(0.0))) == 0.5)
    check('strict zero logit is negative', not bool(labels['raw_positive'][0, 0, 0, 7]))
    check('zero logit pixel is still a fill candidate', bool(labels['fill_edit'][0, 0, 0, 7]))
    return {'labels': {key: int(value.sum()) for key, value in labels.items()}}


def check_contingency_table():
    raw, sr, delta, truth, keep_cut, keep_fill = fixture()
    labels, counts = strict_labels(raw, sr, delta, truth), None
    counts = contingency(labels, keep_cut, keep_fill)
    check('canonical key set', set(counts) == set(COUNT_KEYS))
    for key, value in EXPECTED.items():
        check(f'contingency {key}={value}', counts[key] == value)
    check('brute force agrees', counts == brute_counts(raw, sr, delta, truth, keep_cut, keep_fill))
    check_identities(counts)
    return counts


def check_identity_rejection():
    counts = dict(EXPECTED)
    for key in ('edit_pixels', 'beneficial_pixels', 'accepted_pixels',
                'cut_accepted_beneficial_pixels', 'fill_rejected_harmful_pixels'):
        broken = dict(counts)
        broken[key] += 1
        try:
            check_identities(broken)
        except ValueError:
            continue
        raise AssertionError(f'tampered contingency accepted: {key}')
    broken = dict(counts)
    broken['edit_pixels'] = -1
    try:
        check_identities(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('negative count accepted')
    return {'rejected_tampered_keys': 5}


def check_keep_subset():
    raw, sr, delta, truth, keep_cut, keep_fill = fixture()
    labels = strict_labels(raw, sr, delta, truth)
    # keep_cut may only cover cut pixels (index 2 is a fill pixel) and keep_fill
    # may only cover fill pixels (index 0 is a cut pixel).
    bad_cut = keep_cut.clone()
    bad_cut[0, 0, 0, 2] = True
    bad_fill = keep_fill.clone()
    bad_fill[0, 0, 0, 0] = True
    for name, cut_mask, fill_mask in (('cut', bad_cut, keep_fill), ('fill', keep_cut, bad_fill)):
        try:
            contingency(labels, cut_mask, fill_mask)
        except ValueError:
            continue
        raise AssertionError(f'keep outside candidates accepted: {name}')
    return {'subset_enforced': True}


def check_legacy_aliases():
    aliases = legacy_aliases(EXPECTED)
    check('alias key set', set(aliases) == set(LEGACY_ALIASES))
    for legacy, canonical in LEGACY_ALIASES.items():
        check(f'alias {legacy}', aliases[legacy] == EXPECTED[canonical])
    check('retained v6 names present', 'beneficial_accepted_pixels' in aliases
          and 'harmful_accepted_pixels' in aliases and 'beneficial_rejected_pixels' in aliases
          and 'harmful_rejected_pixels' in aliases
          and 'cut_beneficial_rejected_pixels' in aliases
          and 'fill_harmful_rejected_pixels' in aliases)
    return {'aliases': aliases}


def check_rates():
    for name, value in EXPECTED_RATES.items():
        check(f'rate {name}', math.isclose(rates(EXPECTED, undefined=None)[name], value,
                                           rel_tol=0, abs_tol=1e-12))
    zero = new_counts()
    nulls = rates(zero, undefined=None)
    finites = rates(zero, undefined=0.0)
    check('all rates undefined on empty table', set(nulls) == set(finites))
    check('null policy', all(value is None for value in nulls.values()))
    check('numeric policy', all(value == 0.0 for value in finites.values()))
    check('denominators always reported',
          all(denominator in zero for _, _, denominator in RATE_DEFINITIONS))
    # A cut-only fixture leaves every fill denominator at zero.
    raw = tensor([1.0, 1.0]); sr = tensor([-1.0, -1.0]); delta = tensor([-2.0, -2.0])
    truth = tensor([1.0, 0.0])
    labels = strict_labels(raw, sr, delta, truth)
    # Pixel 0 is a harmful cut, pixel 1 a beneficial cut; accept the beneficial.
    keep_cut = torch.tensor([[[[False, True]]]])
    keep_fill = torch.tensor([[[[False, False]]]])
    counts = contingency(labels, keep_cut, keep_fill)
    check('cut-only fill denominators zero',
          counts['fill_beneficial_pixels'] == 0 and counts['fill_harmful_pixels'] == 0
          and counts['fill_accepted_pixels'] == 0)
    cut_only = rates(counts, undefined=None)
    check('defined cut rate', math.isclose(cut_only['cut_acceptance_precision'], 1.0))
    check('defined cut rejection rate', math.isclose(cut_only['cut_harmful_rejection_rate'], 1.0))
    check('undefined fill rate is null', cut_only['fill_acceptance_precision'] is None)
    check('undefined fill rate is 0.0 for validation',
          rates(counts, undefined=0.0)['fill_acceptance_precision'] == 0.0)
    return {'rates': {key: rates(EXPECTED, undefined=None)[key] for key in EXPECTED_RATES},
            'undefined_policy': 'null or 0.0 plus denominator count'}


def check_net_pixels():
    net = net_pixels(EXPECTED)
    check('net total', net['net_pixels_vs_raw'] == -1)
    check('net cut', net['cut_net_pixels_vs_raw'] == 0)
    check('net fill', net['fill_net_pixels_vs_raw'] == -1)
    check('net is always defined', all(isinstance(value, int) for value in net.values()))
    return net


def check_accumulate():
    raw, sr, delta, truth, keep_cut, keep_fill = fixture()
    counts = contingency(strict_labels(raw, sr, delta, truth), keep_cut, keep_fill)
    total = new_counts()
    accumulate(total, counts)
    accumulate(total, counts)
    check('accumulator sums', all(total[key] == 2 * counts[key] for key in COUNT_KEYS))
    check_identities(total)
    empty = new_counts()
    check('empty accumulator is zero', set(empty) == set(COUNT_KEYS)
          and all(value == 0 for value in empty.values()))
    return {'accumulated_edit_pixels': total['edit_pixels']}


def check_random_exhaustive():
    generator = torch.Generator().manual_seed(20260924)
    checks = 0
    for index in range(6):
        shape = (2 + index % 2, 1, 7 + index, 5 + index)
        raw = torch.randn(shape, generator=generator)
        sr = torch.randn(shape, generator=generator)
        delta = torch.randn(shape, generator=generator)
        truth = (torch.rand(shape, generator=generator) > 0.5).to(torch.float32)
        labels = strict_labels(raw, sr, delta, truth)
        keep_cut = labels['cut_edit'] & (torch.rand(shape, generator=generator) > 0.5)
        keep_fill = labels['fill_edit'] & (torch.rand(shape, generator=generator) > 0.5)
        counts = contingency(labels, keep_cut, keep_fill)
        check('random agrees with brute force',
              counts == brute_counts(raw, sr, delta, truth, keep_cut, keep_fill))
        check_identities(counts)
        # The formal evaluator accumulates one image at a time; that slicing must
        # reproduce the whole-batch table exactly.
        per_image = new_counts()
        for index in range(shape[0]):
            accumulate(per_image, contingency(
                {key: value[index:index + 1] for key, value in labels.items()},
                keep_cut[index:index + 1], keep_fill[index:index + 1]))
        check('per-image slicing sums to the batch table', per_image == counts)
        checks += 1
    return {'random_fixtures': checks}


def check_tool_consistency():
    """The verifier tool's own labels must agree with this module's table.

    The verifier is owned elsewhere, so this is the drift guard: if its strict
    convention ever changes, the counting layer must change with it.
    """
    try:
        from tools.region_discrimination_v8_grok_fullloss import (candidate_masks,
                                                                  directional_labels)
    except Exception as exc:  # pragma: no cover - tool is optional to this test
        return {'skipped': f'{type(exc).__name__}: {exc}'}
    check('v9 reuses the v8 candidate_masks object', v9_candidate_masks is candidate_masks)
    check('v9 reuses the v8 directional_labels object', v9_directional_labels is directional_labels)
    generator = torch.Generator().manual_seed(7)
    shape = (2, 1, 6, 5)
    raw = torch.randn(shape, generator=generator)
    sr = torch.randn(shape, generator=generator)
    delta = torch.randn(shape, generator=generator)
    truth = (torch.rand(shape, generator=generator) > 0.5).to(torch.float32)
    # Force the strict logit == 0 boundary into the fixture.
    raw[0, 0, 0, 0] = 0.0; sr[0, 0, 0, 0] = 1.0; delta[0, 0, 0, 0] = 1.0
    mine = strict_labels(raw, sr, delta, truth)
    theirs = directional_labels(raw, sr, delta, truth)
    cut, fill = candidate_masks(raw, sr, delta)
    check('tool cut matches', torch.equal(mine['cut_edit'], theirs['cut_edit'])
          and torch.equal(mine['cut_edit'], cut))
    check('tool fill matches', torch.equal(mine['fill_edit'], theirs['fill_edit'])
          and torch.equal(mine['fill_edit'], fill))
    check('tool benefit matches',
          torch.equal(mine['beneficial'], theirs['beneficial_cut'] | theirs['beneficial_fill'])
          and torch.equal(mine['harmful'], theirs['harmful_cut'] | theirs['harmful_fill']))
    check('tool boundary pixel is fill', bool(fill[0, 0, 0, 0]))
    return {'cut_pixels': int(cut.sum()), 'fill_pixels': int(fill.sum())}


def check_forward_inputs():
    """Ground truth must never be a forward input in either evaluation path."""
    for filename in ('evaluate.py', 'common.py'):
        text = (ROOT / filename).read_text(encoding='utf-8')
        check(f'{filename} calls model with the image only',
              "model(batch['image'].to(device))" in text)
        check(f'{filename} never forwards the region truth',
              re.search(r"model\(\s*batch\[['\"]region", text) is None)
        check(f'{filename} uses the shared strict labels', 'strict_labels(' in text)
    return {'files': ['evaluate.py', 'common.py']}


def check_launcher_audit_receipts():
    """An early crash must never be recorded as a successful startup audit.

    The launcher is executed in a temporary directory with stubbed ``common``
    and ``train`` modules, so the audit wiring (cleanup vs success marking and
    the exit receipt) is exercised without touching CUDA or the dataset.
    """
    import importlib.util
    import tempfile
    import types

    source = (ROOT / 'launch_rd_v9_context_audited.py').read_text(encoding='utf-8')
    original_modules = {name: sys.modules.get(name) for name in ('common', 'train')}
    try:
        with tempfile.TemporaryDirectory(prefix='v9_audit_crash_') as folder:
            root = Path(folder)
            (root / 'launch_rd_v9_context_audited.py').write_bytes(source.encode('utf-8'))
            fake_common = types.ModuleType('common')
            fake_common.torch = torch
            fake_common.MAIN_RUN = 'hbgnet_protocol_b_seed73_rd_v9_context'
            fake_common.VERIFIER_LOSS_MODE = 'rd_v9_context_fullloss_sr_band'
            fake_common.config = lambda path: {'experiment': {'run_name': fake_common.MAIN_RUN}}
            fake_common.source_fingerprint = lambda: {'files_sha256': {'stub': '0' * 64}}
            fake_common.require_engineering = lambda cfg, fingerprint: None
            fake_common.atomic_json = lambda path, payload: Path(path).write_bytes(
                json.dumps(payload, allow_nan=False).encode('utf-8'))
            fake_train = types.ModuleType('train')
            fake_train.build = lambda *args, **kwargs: None
            fake_train._finite_step = lambda *args, **kwargs: None
            fake_train.loss_fn = lambda *args, **kwargs: None

            class _Lock:
                def __init__(self, *args, **kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            def _crash(*args, **kwargs):
                raise RuntimeError('simulated early crash before any observation')

            fake_train.training_lock = _Lock
            fake_train._output_paths = lambda cfg: (root / 'log', root / 'weights')
            fake_train._validate_outputs = lambda *args, **kwargs: None
            fake_train._train_locked = _crash
            fake_train._status = lambda *args, **kwargs: None
            sys.modules['common'] = fake_common
            sys.modules['train'] = fake_train

            spec = importlib.util.spec_from_file_location(
                'v9_audit_crash_launcher', root / 'launch_rd_v9_context_audited.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            crashed = False
            try:
                module.main()
            except RuntimeError:
                crashed = True
            check('early crash propagates', crashed)
            evidence = root / 'formal_launch_v9_context_01'
            check('launch receipt written', (evidence / 'launch.json').is_file())
            exit_receipt = json.loads((evidence / 'exit.json').read_text(encoding='utf-8'))
            check('crashed audit is not marked complete',
                  exit_receipt['audit_complete'] is False
                  and exit_receipt['exit_code'] == 1
                  and exit_receipt['loss_batches'] == 0
                  and exit_receipt['optimizer_updates'] == 0)
            records = [json.loads(line) for line
                       in (evidence / 'activity.jsonl').read_text(encoding='utf-8').splitlines()]
            check('no success record on the crash path',
                  all(record['kind'] != 'audit_complete' for record in records))
            check('cleanup restored the patched functions',
                  fake_train.build is not module.audited_build
                  and fake_train._finite_step is not module.audited_step
                  and fake_train.loss_fn is not module.audited_loss)
            # Cleanup alone must never mark success, and a satisfied maybe_finish
            # must: this is the separation the audit receipt depends on.
            module.cleanup_audit_hooks()
            check('cleanup never marks success', module.audit_complete is False)
            module.loss_count = 5
            module.step_count = 2
            module.forward_counts['verifier'] = 5
            module.forward_counts['refiner'] = 5
            module.maybe_finish_audit()
            check('satisfied observations mark success', module.audit_complete is True)
            records = [json.loads(line) for line
                       in (evidence / 'activity.jsonl').read_text(encoding='utf-8').splitlines()]
            check('success record carries the observation counts',
                  any(record['kind'] == 'audit_complete'
                      and record['loss_batches'] == 5
                      and record['optimizer_updates'] == 2
                      and record['training_continues'] is True for record in records))
    finally:
        for name, module_object in original_modules.items():
            if module_object is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module_object
    return {'crash_not_marked_complete': True, 'cleanup_separated_from_success': True,
            'observations_required_for_success': True}


def check_calibration_and_curve():
    """A monotone calibration may only move the operating point."""
    generator = torch.Generator().manual_seed(20261003)
    scores = torch.randn(2, 2, 4, 5, generator=generator)
    beneficial = torch.rand(2, 2, 4, 5, generator=generator) > 0.6
    harmful = (torch.rand(2, 2, 4, 5, generator=generator) > 0.7) & ~beneficial
    thresholds = (-1.0, 0.0, 1.0)
    a, b = (2.0, 0.5), (1.0, -3.0)
    calibrated = monotone_affine_calibration(scores, a, b)
    logit_threshold = calibration_logit_threshold(a, b)
    check('calibrated zero threshold is the raw threshold', logit_threshold == (-0.5, 6.0))
    for index in range(2):
        check('monotone keeps every decision',
              torch.equal(calibrated[:, index] > 0, scores[:, index] > logit_threshold[index]))
    # Point i of the calibrated curve at threshold t corresponds to the raw
    # curve at threshold (t - b_d) / a_d for the cut direction.
    mapped_raw = [(threshold - b[0]) / a[0] for threshold in thresholds]
    raw_curve = br_ha_curve(scores, beneficial, harmful, mapped_raw)
    calibrated_curve = br_ha_curve(calibrated, beneficial, harmful, thresholds)
    for raw_point, calibrated_point in zip(raw_curve['points'], calibrated_curve['points']):
        check('the curve is only re-parameterised',
              raw_point['directions']['cut']['accepted_harmful']
              == calibrated_point['directions']['cut']['accepted_harmful']
              and raw_point['directions']['cut']['rejected_beneficial']
              == calibrated_point['directions']['cut']['rejected_beneficial'])
    # Brute-force checks on a tiny table with a zero denominator.
    tiny_scores = torch.tensor([[[[0.0]], [[2.0]]]])
    tiny_good = torch.tensor([[[[True]], [[False]]]])
    tiny_bad = torch.tensor([[[[False]], [[False]]]])
    point = br_ha_curve(tiny_scores, tiny_good, tiny_bad, [0.0])['points'][0]
    check('strict threshold accepted counts',
          point['directions']['cut']['accepted_harmful'] == 0
          and point['directions']['cut']['rejected_beneficial'] == 1
          and point['directions']['cut']['harmful_total'] == 0
          and point['directions']['fill']['beneficial_total'] == 0)
    check('zero denominator is null',
          point['directions']['fill']['beneficial_rejection_rate'] is None
          and point['directions']['cut']['harmful_acceptance_rate'] is None
          and point['directions']['cut']['beneficial_rejection_rate'] == 1.0)
    for bad_scores, bad_good, bad_bad in (
            (scores[:, :1], beneficial, harmful),
            (scores, beneficial.long(), harmful),
            (scores, beneficial, beneficial | harmful),
            (torch.full_like(scores, float('nan')), beneficial, harmful)):
        try:
            br_ha_curve(bad_scores, bad_good, bad_bad, [0.0])
        except (ValueError, FloatingPointError):
            continue
        raise AssertionError('invalid BR/HA input accepted')
    return {'logit_threshold': list(logit_threshold), 'reparameterised_only': True,
            'disjoint_and_finite_enforced': True}


def check_import_package():
    """Optional wiring check: common imports the v8 verifier and new diagnostics."""
    import common
    from protocol import candidate_diagnostics
    check('v9 run identity', common.MAIN_RUN.startswith('hbgnet_protocol_b_seed')
          and common.MAIN_RUN.endswith('_rd_v9_context'))
    check('v9 stage identity', common.STAGE.startswith('HBGNet_seed')
          and common.STAGE.endswith('_rd_v9_context'))
    check('v9 external tool', 'tools/region_discrimination_v9.py' in common.EXTERNAL_FILES)
    check('v8 dependency hashed',
          'tools/region_discrimination_v8_grok_fullloss.py' in common.EXTERNAL_FILES)
    check('diagnostics required', 'protocol/candidate_diagnostics.py' in common.REQUIRED_PACKAGE_FILES)
    check('diagnostics test required', 'verify_v9_diagnostics.py' in common.REQUIRED_PACKAGE_FILES)
    check('v9 launcher required', 'launch_rd_v9_context_audited.py' in common.REQUIRED_PACKAGE_FILES)
    check('v9 verify script required', 'verify_rd_v9_context.py' in common.REQUIRED_PACKAGE_FILES)
    check('shared module reused', common.contingency is candidate_diagnostics.contingency)
    check('loss name', common._expected_loss(common.MAIN_RUN)['name'] == 'rd_v9_context_fullloss_sr_band')
    check('verifier loss mode', common.VERIFIER_LOSS_MODE == 'ordinary_bce_present_direction_average')
    check('verifier aggregation', common.VERIFIER_SPECS[common.MAIN_RUN] ==
          {'enabled': True, 'mode': 'enabled', 'aggregation': 'multi_scale_support'})
    check('center-only control is reserved',
          'hbgnet_protocol_b_rd_v9_context_sr_v2_center_only_control' in common.RESERVED_RUNS)
    return {'run_name': common.MAIN_RUN, 'stage': common.STAGE,
            'external_files': sorted(common.EXTERNAL_FILES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-package', action='store_true',
                        help='also import common to check v8 wiring (no dataset access)')
    args = parser.parse_args()
    report = {
        'status': 'pass',
        'strict_labels': check_strict_labels(),
        'contingency': check_contingency_table(),
        'identity_rejection': check_identity_rejection(),
        'keep_subset': check_keep_subset(),
        'legacy_aliases': check_legacy_aliases(),
        'rates': check_rates(),
        'net_pixels': check_net_pixels(),
        'accumulate': check_accumulate(),
        'random_exhaustive': check_random_exhaustive(),
        'tool_consistency': check_tool_consistency(),
        'calibration_and_curve': check_calibration_and_curve(),
        'forward_inputs': check_forward_inputs(),
        'launcher_audit_receipts': check_launcher_audit_receipts(),
    }
    if args.import_package:
        report['package_wiring'] = check_import_package()
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == '__main__':
    main()
