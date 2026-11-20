"""Deterministic, label-blind spatial splitting of FHAPD tile ids (HBGNet protocol B).

Scope
-----
Pure-stdlib helper for the FHAPD_XJ / FHAPD_JS adaptation.  It reads only
the geometry encoded in a tile id following the protocol-B naming

    <parent>_<x>_<y>_<ox>_<oy>_<w>_<h>          e.g. XJ1_000383_005887_0_0_256_256

* ``parent`` -- region block, e.g. ``XJ1``, ``JS1`` (letter + alphanumerics).
* first coordinate is x (horizontal / image columns), second is y
  (vertical / image rows); a tile covers the half-open pixel box
  ``[x, x + width) x [y, y + height)``.
* current schema requires crop offsets ``ox == oy == 0`` and
  ``width == height == 256`` (anything else is rejected as malformed).

No imagery, masks, labels, foreground statistics, training seeds or other
dataset content are consulted, and no RNG is used: the result is a pure
function of the (validated, unique) tile ids.

Pipeline (per parent)
---------------------
1. Validate every unique id with :func:`parse_tile_id`.
2. Select the 256-stride lattice anchored at ``min(x)`` / ``min(y)`` of the
   parent.  Tiles off that lattice are excluded with reason
   ``overlap_lattice``: the source product is cut with a 128-stride tiling,
   so off-lattice neighbours overlap a selected tile by 128 px.
3. Project the retained tiles on the parent's **longest-span axis**
   (footprint span ``max(coord + size) - min(coord)``; x wins ties).  The
   sorted lattice columns along that axis are cut into three contiguous
   non-empty stripes train / val / test, with exactly one whole lattice
   column dropped between train and val and one between val and test.
   Because retained columns differ by at least the 256 px lattice stride,
   dropping one column leaves a clear pixel gap of at least 256 px between
   the neighbouring stripes.
4. Choose the cut pair minimising the documented objective

       sum over k in (train, val, test) of (share_k - target_k) ** 2

   where ``share_k`` is the split's share of the *retained* tiles, i.e. of
   the tiles assigned to a split (all tiles minus the two dropped columns),
   and ``target_k`` defaults to ``(0.70, 0.15, 0.15)``.  Ties are broken
   deterministically: more retained tiles first, then the smaller train/val
   cut index, then the smaller val/test cut index.
   Tiles inside the two dropped columns get reasons ``gap_train_val`` and
   ``gap_val_test``.

Public API
----------
``parse_tile_id(tile_id) -> dict``
    Returns ``{"parent": str, "x": int, "y": int, "width": int,
    "height": int}``.  Raises ``TypeError`` for non-str input and
    ``SpatialSplitError`` (a ``ValueError`` subclass) for malformed or
    unsafe ids, non-zero crop offsets, negative coordinates or sizes != 256.

``normalize_tiles(tiles_or_ids) -> list[dict]``
    Accepts an iterable of id strings and/or mappings with keys
    ``{"id", "parent", "x", "y", "width", "height"}``; returns validated
    records ``{"id", "parent", "x", "y", "width", "height"}`` in input
    order.  Mapping fields are cross-checked against the id.  Rejects
    duplicates, non-integer fields (bool included) and inconsistent ids.

``build_split(tiles_or_ids, *, targets=(0.70, 0.15, 0.15)) -> dict``
    Primary entry point for the prepare step; pass all unique tile ids of
    the region (one call handles several parents, and parents are processed
    independently).  Returns exactly

        {"assignments":      {id: "train"|"val"|"test"|"excluded"},
         "excluded_reasons": {id: "overlap_lattice"|"gap_train_val"|"gap_val_test"},
         "summary":          {...}}

    ``assignments`` covers every input id; use it to classify tile
    dirs/files, ``excluded_reasons`` for the audit log.  Raises
    ``SpatialSplitError`` when any parent cannot host the 3 stripe cuts
    (fewer than 5 lattice columns on its longest axis, or no lattice
    survivor at all).

``spatial_split(tiles_or_ids, *, targets=(0.70, 0.15, 0.15)) -> dict``
    Same computation with full detail: ``{"assignment": ...`` (singular),
    ``"excluded_reasons": ..., "summary": ..., "audit": {...}}``.
    ``summary`` holds schema constants, global counts, per-parent lattice /
    axis / columns / stripe extents / gap boundaries / counts / fractions /
    objective details, and ``all_ids_accounted``.  ``audit`` is the result
    of the built-in self-check (see below).  Raises ``SpatialSplitError``
    when any parent cannot host the 3 stripe cuts (fewer than 5 lattice
    columns on its longest axis, or no lattice survivor at all).

``spatial_audit(tiles_or_ids, assignment, *, min_gap_px=256) -> dict``
    Independent verification of an assignment (it never trusts the split
    summary).  Rejects unaccounted / unknown ids and unknown split labels.
    Top level: ``ok``, ``failures``, ``n_tiles``/``n_selected``/``n_excluded``,
    ``overlap_conflicts`` (positive-area footprint overlaps among selected
    tiles; must be 0), ``cross_split_pairs`` (exact count of same-parent tile
    pairs in different splits), ``gap_violations`` (exact count of those pairs
    whose clear axis gap is < ``min_gap_px``), ``min_cross_split_gap_px`` and
    ``min_required_gap_px``; per-parent detail under ``parents``
    (``n_by_split``, counts, examples).  The axis gap of a pair is
    ``max(sep_x, sep_y)`` where ``sep`` is the edge-to-edge footprint distance
    along one axis (0 when the intervals overlap or touch).  ``ok`` is True
    only when there are no overlaps and no cross-split pair closer than
    ``min_gap_px``.  ``min_cross_split_gap_px`` is exact when some violation
    exists (it is then < ``min_gap_px``); otherwise it is a lower bound equal
    to ``min_gap_px`` with ``min_cross_split_gap_exact`` False, because the
    sweep prunes pairs that are provably >= ``min_gap_px`` apart.
    Complexity: x-sorted sweep, near-linear for lattice data, O(n^2) worst
    case; ~1.5k tiles audit in milliseconds.

Reproducibility / blind spots
-----------------------------
No seeds and no randomness: shuffling the input or changing
``random.seed`` does not change any output.  The split is spatial only
within the coordinates present in the ids; it makes **no claim about
geographic independence between different parents** (XJ1 vs JS1 layouts are
unknown to this module) and cannot see tiles missing from the manifest.
"""

import re
from collections.abc import Mapping as _MappingABC

__all__ = [
    "SpatialSplitError",
    "TRAIN",
    "VAL",
    "TEST",
    "EXCLUDED",
    "SPLITS",
    "ASSIGN_VALUES",
    "STRIDE",
    "TILE_SIZE",
    "MIN_LATTICE_COLUMNS",
    "DEFAULT_TARGETS",
    "OBJECTIVE_METRIC",
    "REASON_OVERLAP_LATTICE",
    "REASON_GAP_TRAIN_VAL",
    "REASON_GAP_VAL_TEST",
    "parse_tile_id",
    "normalize_tiles",
    "build_split",
    "spatial_split",
    "spatial_audit",
]

# Hard geometry constants of the current FHAPD / protocol-B schema.
TILE_SIZE = 256
STRIDE = 256
# 3 non-empty stripes + 2 single-column gaps on the longest-span axis.
MIN_LATTICE_COLUMNS = 5

TRAIN = "train"
VAL = "val"
TEST = "test"
EXCLUDED = "excluded"
SPLITS = (TRAIN, VAL, TEST)
ASSIGN_VALUES = (TRAIN, VAL, TEST, EXCLUDED)

REASON_OVERLAP_LATTICE = "overlap_lattice"
REASON_GAP_TRAIN_VAL = "gap_train_val"
REASON_GAP_VAL_TEST = "gap_val_test"

DEFAULT_TARGETS = (0.70, 0.15, 0.15)
OBJECTIVE_METRIC = "sum_squared_fraction_error"

_TILE_ID_RE = re.compile(
    r"^(?P<parent>[A-Za-z][A-Za-z0-9]*)"
    r"_(?P<x>[0-9]+)_(?P<y>[0-9]+)"
    r"_(?P<ox>[0-9]+)_(?P<oy>[0-9]+)"
    r"_(?P<w>[0-9]+)_(?P<h>[0-9]+)$"
)


class SpatialSplitError(ValueError):
    """Malformed / unsafe tile id, unusable input or unviable split geometry."""


def parse_tile_id(tile_id):
    """Parse and validate one FHAPD protocol-B tile id.

    Returns ``{"parent", "x", "y", "width", "height"}``.  The parent must be
    a letter followed by alphanumerics (``XJ1``, ``JS1``, ...); x, y, ox, oy,
    w, h must be plain ASCII decimal integers.  Current schema: ``ox == oy
    == 0`` and ``width == height == 256``.

    Raises ``TypeError`` when ``tile_id`` is not a str, and
    ``SpatialSplitError`` (ValueError) for malformed or unsafe ids.
    """
    if not isinstance(tile_id, str):
        raise TypeError("tile id must be a str, got %s" % type(tile_id).__name__)
    match = _TILE_ID_RE.match(tile_id)
    if match is None:
        raise SpatialSplitError(
            "malformed tile id %r: expected <parent>_<x>_<y>_<ox>_<oy>_<w>_<h> "
            "with parent like 'XJ1' and decimal integer fields" % (tile_id,)
        )
    parent = match.group("parent")
    x = int(match.group("x"))
    y = int(match.group("y"))
    ox = int(match.group("ox"))
    oy = int(match.group("oy"))
    width = int(match.group("w"))
    height = int(match.group("h"))
    if x < 0 or y < 0:  # unreachable with the regex, kept as a hard guard
        raise SpatialSplitError("tile id %r has a negative coordinate" % (tile_id,))
    if ox != 0 or oy != 0:
        raise SpatialSplitError(
            "tile id %r has crop offsets ox=%d oy=%d; the current schema "
            "requires 0/0" % (tile_id, ox, oy)
        )
    if width != TILE_SIZE or height != TILE_SIZE:
        raise SpatialSplitError(
            "tile id %r has size %dx%d; the current schema requires %dx%d"
            % (tile_id, width, height, TILE_SIZE, TILE_SIZE)
        )
    return {"parent": parent, "x": x, "y": y, "width": width, "height": height}


def _require_int(value, field_name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpatialSplitError(
            "field %r must be an int, got %r" % (field_name, value)
        )
    return value


def _normalize_one(item):
    if isinstance(item, str):
        parsed = parse_tile_id(item)
        return {
            "id": item,
            "parent": parsed["parent"],
            "x": parsed["x"],
            "y": parsed["y"],
            "width": parsed["width"],
            "height": parsed["height"],
        }
    if isinstance(item, _MappingABC):
        for key in ("id", "parent", "x", "y", "width", "height"):
            if key not in item:
                raise SpatialSplitError(
                    "tile mapping is missing required key %r" % (key,)
                )
        tile_id = item["id"]
        if not isinstance(tile_id, str):
            raise TypeError(
                "tile mapping 'id' must be a str, got %s"
                % type(tile_id).__name__
            )
        parsed = parse_tile_id(tile_id)
        if item["parent"] != parsed["parent"]:
            raise SpatialSplitError(
                "tile mapping %r parent %r != parent in id %r"
                % (tile_id, item["parent"], parsed["parent"])
            )
        for key in ("x", "y", "width", "height"):
            value = _require_int(item[key], key)
            if value != parsed[key]:
                raise SpatialSplitError(
                    "tile mapping %r field %s=%r != %s=%r in id"
                    % (tile_id, key, value, key, parsed[key])
                )
        return {
            "id": tile_id,
            "parent": parsed["parent"],
            "x": parsed["x"],
            "y": parsed["y"],
            "width": parsed["width"],
            "height": parsed["height"],
        }
    raise SpatialSplitError(
        "tile entry must be an id string or a mapping, got %s"
        % type(item).__name__
    )


def normalize_tiles(tiles_or_ids):
    """Validate ids/mappings; return records in input order; reject duplicates."""
    if isinstance(tiles_or_ids, (str, bytes)) or isinstance(
        tiles_or_ids, _MappingABC
    ):
        raise SpatialSplitError(
            "expected an iterable of tile ids, not a single %s"
            % type(tiles_or_ids).__name__
        )
    try:
        items = list(tiles_or_ids)
    except TypeError:
        raise SpatialSplitError(
            "expected an iterable of tile ids, got %s"
            % type(tiles_or_ids).__name__
        )
    records = []
    seen = set()
    for item in items:
        record = _normalize_one(item)
        if record["id"] in seen:
            raise SpatialSplitError("duplicate tile id %r" % (record["id"],))
        seen.add(record["id"])
        records.append(record)
    return records


def _validate_targets(targets):
    if not isinstance(targets, (tuple, list)) or len(targets) != len(SPLITS):
        raise SpatialSplitError(
            "targets must be a 3-item sequence (train, val, test), got %r"
            % (targets,)
        )
    values = []
    for index, raw in enumerate(targets):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise SpatialSplitError(
                "targets[%d] must be a number, got %r" % (index, raw)
            )
        value = float(raw)
        if not value > 0.0:
            raise SpatialSplitError(
                "targets[%d] must be > 0, got %r" % (index, raw)
            )
        values.append(value)
    if abs(sum(values) - 1.0) > 1e-6:
        raise SpatialSplitError("targets must sum to 1.0, got %r" % (targets,))
    return tuple(values)


def _axis_sep(start_a, end_a, start_b, end_b):
    """Edge-to-edge distance of two half-open 1-D intervals; 0 if they touch."""
    return max(0, max(start_a, start_b) - min(end_a, end_b))


def spatial_split(tiles_or_ids, *, targets=DEFAULT_TARGETS):
    """Split unique FHAPD tile ids into spatial train/val/test stripes.

    See the module docstring for the exact algorithm and the returned
    structure.  ``targets`` is the (train, val, test) fraction triple used by
    the documented squared-error objective; it must be positive and sum to 1.
    """
    tiles = normalize_tiles(tiles_or_ids)
    target_values = _validate_targets(targets)

    assignment = {}
    excluded_reasons = {}
    parents_summary = {}

    by_parent = {}
    for tile in tiles:
        by_parent.setdefault(tile["parent"], []).append(tile)

    for parent in sorted(by_parent):
        ptiles = sorted(
            by_parent[parent], key=lambda t: (t["x"], t["y"], t["id"])
        )
        min_x = min(t["x"] for t in ptiles)
        min_y = min(t["y"] for t in ptiles)

        lattice_tiles = []
        for tile in ptiles:
            if (tile["x"] - min_x) % STRIDE == 0 and (
                tile["y"] - min_y
            ) % STRIDE == 0:
                lattice_tiles.append(tile)
            else:
                assignment[tile["id"]] = EXCLUDED
                excluded_reasons[tile["id"]] = REASON_OVERLAP_LATTICE

        if not lattice_tiles:
            raise SpatialSplitError(
                "parent %r is not split-viable: no tile lies on the %d-stride "
                "lattice anchored at (min_x=%d, min_y=%d)"
                % (parent, STRIDE, min_x, min_y)
            )

        x_min = min(t["x"] for t in lattice_tiles)
        x_max = max(t["x"] for t in lattice_tiles)
        y_min = min(t["y"] for t in lattice_tiles)
        y_max = max(t["y"] for t in lattice_tiles)
        span_x = max(t["x"] + t["width"] for t in lattice_tiles) - x_min
        span_y = max(t["y"] + t["height"] for t in lattice_tiles) - y_min
        axis = "x" if span_x >= span_y else "y"

        columns = {}
        for tile in lattice_tiles:
            columns.setdefault(tile[axis], []).append(tile)
        col_coords = sorted(columns)
        if len(col_coords) < MIN_LATTICE_COLUMNS:
            raise SpatialSplitError(
                "parent %r is not split-viable on its longest-span axis %r: "
                "%d lattice columns < %d required (3 non-empty stripes + "
                "2 single-column gaps)"
                % (parent, axis, len(col_coords), MIN_LATTICE_COLUMNS)
            )

        weights = [len(columns[coord]) for coord in col_coords]
        prefix = [0]
        for weight in weights:
            prefix.append(prefix[-1] + weight)

        n_cols = len(col_coords)
        best_key = None
        best = None
        for i in range(0, n_cols - 4):  # i <= n - 5
            train_n = prefix[i + 1]
            for j in range(i + 2, n_cols - 2):  # j <= n - 3
                val_n = prefix[j + 1] - prefix[i + 2]
                test_n = prefix[n_cols] - prefix[j + 2]
                retained_n = train_n + val_n + test_n
                error = (
                    (train_n / retained_n - target_values[0]) ** 2
                    + (val_n / retained_n - target_values[1]) ** 2
                    + (test_n / retained_n - target_values[2]) ** 2
                )
                key = (error, -retained_n, i, j)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (i, j, train_n, val_n, test_n, retained_n, error)

        i, j, train_n, val_n, test_n, retained_n, error = best
        gap_train_val = col_coords[i + 1]
        gap_val_test = col_coords[j + 1]

        for index, coord in enumerate(col_coords):
            if index <= i:
                label = TRAIN
                reason = None
            elif index == i + 1:
                label = EXCLUDED
                reason = REASON_GAP_TRAIN_VAL
            elif index <= j:
                label = VAL
                reason = None
            elif index == j + 1:
                label = EXCLUDED
                reason = REASON_GAP_VAL_TEST
            else:
                label = TEST
                reason = None
            for tile in columns[coord]:
                assignment[tile["id"]] = label
                if reason is not None:
                    excluded_reasons[tile["id"]] = reason

        dropped_overlap = len(ptiles) - len(lattice_tiles)
        clear_gap_train_val = col_coords[i + 2] - col_coords[i] - TILE_SIZE
        clear_gap_val_test = col_coords[j + 2] - col_coords[j] - TILE_SIZE

        parents_summary[parent] = {
            "total_ids": len(ptiles),
            "axis": axis,
            "spans": {"x": span_x, "y": span_y},
            "extent": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
            "lattice": {
                "stride": STRIDE,
                "anchor_x": min_x,
                "anchor_y": min_y,
                "selected": len(lattice_tiles),
                "dropped_overlap_lattice": dropped_overlap,
            },
            "n_columns": n_cols,
            "columns": list(col_coords),
            "column_tile_counts": list(weights),
            "cuts": {
                "train_val": {
                    "gap_axis_coord": gap_train_val,
                    "train_last_axis_coord": col_coords[i],
                    "val_first_axis_coord": col_coords[i + 2],
                    "clear_gap_px": clear_gap_train_val,
                },
                "val_test": {
                    "gap_axis_coord": gap_val_test,
                    "val_last_axis_coord": col_coords[j],
                    "test_first_axis_coord": col_coords[j + 2],
                    "clear_gap_px": clear_gap_val_test,
                },
            },
            "stripes": {
                TRAIN: {
                    "axis_min": col_coords[0],
                    "axis_max": col_coords[i],
                    "count": train_n,
                },
                VAL: {
                    "axis_min": col_coords[i + 2],
                    "axis_max": col_coords[j],
                    "count": val_n,
                },
                TEST: {
                    "axis_min": col_coords[j + 2],
                    "axis_max": col_coords[n_cols - 1],
                    "count": test_n,
                },
            },
            "counts": {
                TRAIN: train_n,
                VAL: val_n,
                TEST: test_n,
                "excluded_overlap_lattice": dropped_overlap,
                "excluded_gap_train_val": weights[i + 1],
                "excluded_gap_val_test": weights[j + 1],
            },
            "fractions": {
                TRAIN: train_n / retained_n,
                VAL: val_n / retained_n,
                TEST: test_n / retained_n,
            },
            "objective": {
                "metric": OBJECTIVE_METRIC,
                "targets": list(target_values),
                "error": error,
                "retained": retained_n,
                "cut_column_indices": [i, j],
                "cut_columns": [gap_train_val, gap_val_test],
            },
        }

    counts_total = {TRAIN: 0, VAL: 0, TEST: 0, EXCLUDED: 0}
    for label in assignment.values():
        counts_total[label] += 1
    reason_counts = {}
    for reason in excluded_reasons.values():
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    audit = _audit_records(tiles, assignment, min_gap_px=STRIDE)
    if not audit["ok"]:
        raise SpatialSplitError(
            "internal spatial invariant check failed: %r" % (audit["failures"],)
        )

    summary = {
        "schema": {
            "tile_size_px": TILE_SIZE,
            "lattice_stride_px": STRIDE,
            "min_clear_gap_px": STRIDE,
            "targets": list(target_values),
            "objective_metric": OBJECTIVE_METRIC,
            "min_lattice_columns": MIN_LATTICE_COLUMNS,
            "tie_breaks": [
                "larger_number_of_retained_tiles",
                "smaller_train_val_cut_index",
                "smaller_val_test_cut_index",
            ],
            "label_blind": True,
            "rng": None,
        },
        "n_ids": len(tiles),
        "n_parents": len(by_parent),
        "counts": counts_total,
        "retained": counts_total[TRAIN] + counts_total[VAL] + counts_total[TEST],
        "excluded": counts_total[EXCLUDED],
        "excluded_reason_counts": reason_counts,
        "all_ids_accounted": len(assignment) == len(tiles)
        and set(assignment) == set(t["id"] for t in tiles),
        "parents": parents_summary,
    }
    return {
        "assignment": assignment,
        "excluded_reasons": excluded_reasons,
        "summary": summary,
        "audit": audit,
    }


def build_split(tiles_or_ids, *, targets=DEFAULT_TARGETS):
    """Primary entry point for the prepare step; see the module docstring.

    Returns exactly ``{"assignments", "excluded_reasons", "summary"}`` where
    ``assignments`` maps every valid unique input id to ``"train"``, ``"val"``,
    ``"test"`` or ``"excluded"``.  The full computation (including the
    built-in spatial self-check, which raises ``SpatialSplitError`` on any
    internal invariant failure) is performed by :func:`spatial_split`.
    """
    result = spatial_split(tiles_or_ids, targets=targets)
    return {
        "assignments": result["assignment"],
        "excluded_reasons": result["excluded_reasons"],
        "summary": result["summary"],
    }


def _audit_records(tiles, assignment, min_gap_px):
    id_set = set(t["id"] for t in tiles)
    if len(id_set) != len(tiles):
        raise SpatialSplitError("duplicate tile ids in input")
    assigned_ids = set(assignment)
    missing = sorted(id_set - assigned_ids)
    unknown = sorted(assigned_ids - id_set)
    if missing or unknown:
        raise SpatialSplitError(
            "assignment must cover exactly the input ids "
            "(missing=%r, unknown=%r)" % (missing[:5], unknown[:5])
        )
    for tile_id in sorted(assignment):
        if assignment[tile_id] not in ASSIGN_VALUES:
            raise SpatialSplitError(
                "assignment[%r]=%r is invalid; expected one of %r"
                % (tile_id, assignment[tile_id], ASSIGN_VALUES)
            )

    by_parent = {}
    for tile in tiles:
        by_parent.setdefault(tile["parent"], []).append(tile)

    parents = {}
    total_overlaps = 0
    total_gap_violations = 0
    total_cross_pairs = 0
    total_selected = 0
    global_min_gap = None

    for parent in sorted(by_parent):
        selected = [
            t for t in by_parent[parent] if assignment[t["id"]] in SPLITS
        ]
        selected.sort(key=lambda t: (t["x"], t["y"], t["id"]))
        n_by_split = {split: 0 for split in SPLITS}
        for tile in selected:
            n_by_split[assignment[tile["id"]]] += 1
        n_selected = len(selected)
        cross_pairs = (
            n_selected * n_selected - sum(c * c for c in n_by_split.values())
        ) // 2

        overlaps = []
        gap_violations = []
        checked_min_gap = None
        for left_index, left in enumerate(selected):
            left_x_end = left["x"] + left["width"]
            left_y_end = left["y"] + left["height"]
            left_split = assignment[left["id"]]
            for right_index in range(left_index + 1, n_selected):
                right = selected[right_index]
                if right["x"] >= left_x_end + min_gap_px:
                    break  # sorted by x: every later pair is farther apart
                sep_x = (
                    0 if right["x"] < left_x_end else right["x"] - left_x_end
                )
                if (
                    right["x"] < left_x_end
                    and right["y"] < left_y_end
                    and left["y"] < right["y"] + right["height"]
                ):
                    overlaps.append((left["id"], right["id"]))
                if left_split != assignment[right["id"]]:
                    sep_y = _axis_sep(
                        left["y"],
                        left_y_end,
                        right["y"],
                        right["y"] + right["height"],
                    )
                    axis_gap = sep_x if sep_x >= sep_y else sep_y
                    if checked_min_gap is None or axis_gap < checked_min_gap:
                        checked_min_gap = axis_gap
                    if axis_gap < min_gap_px:
                        gap_violations.append(
                            (left["id"], right["id"], axis_gap)
                        )

        if cross_pairs == 0:
            min_gap = None
            min_gap_exact = True
        elif checked_min_gap is not None and checked_min_gap < min_gap_px:
            min_gap = checked_min_gap
            min_gap_exact = True
        else:
            # Sweep pruned every cross pair, so all of them are known to be
            # at least min_gap_px apart (bound, not the exact minimum).
            min_gap = min_gap_px
            min_gap_exact = False

        parents[parent] = {
            "n_selected": n_selected,
            "n_by_split": n_by_split,
            "cross_split_pairs": cross_pairs,
            "overlap_conflicts": len(overlaps),
            "overlap_examples": overlaps[:3],
            "gap_violations": len(gap_violations),
            "gap_violation_examples": gap_violations[:3],
            "min_cross_split_gap_px": min_gap,
            "min_cross_split_gap_exact": min_gap_exact,
        }
        total_overlaps += len(overlaps)
        total_gap_violations += len(gap_violations)
        total_cross_pairs += cross_pairs
        total_selected += n_selected
        if min_gap is not None:
            if global_min_gap is None or min_gap < global_min_gap:
                global_min_gap = min_gap

    failures = []
    if total_overlaps:
        failures.append(
            "%d positive-area overlap(s) among selected tiles" % total_overlaps
        )
    if total_gap_violations:
        failures.append(
            "%d cross-split pair(s) closer than %d px on both axes"
            % (total_gap_violations, min_gap_px)
        )
    return {
        "ok": not failures,
        "failures": failures,
        "n_tiles": len(tiles),
        "n_selected": total_selected,
        "n_excluded": len(tiles) - total_selected,
        "overlap_conflicts": total_overlaps,
        "overlap_examples": [
            example for info in parents.values() for example in info["overlap_examples"]
        ][:3],
        "cross_split_pairs": total_cross_pairs,
        "gap_violations": total_gap_violations,
        "gap_violation_examples": [
            example for info in parents.values() for example in info["gap_violation_examples"]
        ][:3],
        "min_cross_split_gap_px": global_min_gap,
        "min_cross_split_gap_exact": all(
            info["min_cross_split_gap_exact"] for info in parents.values()
        ),
        "min_required_gap_px": min_gap_px,
        "parents": parents,
    }


def spatial_audit(tiles_or_ids, assignment, *, min_gap_px=STRIDE):
    """Verify an (id -> train/val/test/excluded) assignment; see module docstring.

    ``min_gap_px`` is the required clear separation between tiles of different
    splits of the same parent along at least one axis (default 256).
    """
    if (
        isinstance(min_gap_px, bool)
        or not isinstance(min_gap_px, int)
        or min_gap_px <= 0
    ):
        raise SpatialSplitError(
            "min_gap_px must be a positive int, got %r" % (min_gap_px,)
        )
    if not isinstance(assignment, _MappingABC):
        raise SpatialSplitError(
            "assignment must be a mapping of tile id -> split label, got %s"
            % type(assignment).__name__
        )
    tiles = normalize_tiles(tiles_or_ids)
    return _audit_records(tiles, assignment, min_gap_px)
