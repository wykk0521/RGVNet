"""Unit tests for spatial_split.py (FHAPD spatial splitting).

Run from anywhere:

    py -3 -B -m unittest discover -s <this directory> -t <this directory>

Standard library only.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spatial_split as ss  # noqa: E402


def tid(parent, x, y, ox=0, oy=0, w=256, h=256):
    return "%s_%06d_%06d_%d_%d_%d_%d" % (parent, x, y, ox, oy, w, h)


def grid_ids(parent, xs, ys):
    return [tid(parent, x, y) for x in xs for y in ys]


def rows(parent, xs, row_counts):
    """One tile per lattice row: column xs[k] gets rows y = 0, 256, ..."""
    ids = []
    for x, count in zip(xs, row_counts):
        for j in range(count):
            ids.append(tid(parent, x, 256 * j))
    return ids


def boxes_overlap(id_a, id_b):
    """Positive-area footprint overlap of two tile ids."""
    a = ss.parse_tile_id(id_a)
    b = ss.parse_tile_id(id_b)
    return (
        a["x"] < b["x"] + b["width"]
        and b["x"] < a["x"] + a["width"]
        and a["y"] < b["y"] + b["height"]
        and b["y"] < a["y"] + a["height"]
    )


def brute_force_cuts(column_weights, targets=(0.70, 0.15, 0.15)):
    """Independent re-implementation of the documented objective.

    Returns (key, counts, retained, error) with key = (error, -retained, i, j).
    """
    n = len(column_weights)
    best = None
    for i in range(0, n - 4):
        for j in range(i + 2, n - 2):
            counts = (
                sum(column_weights[: i + 1]),
                sum(column_weights[i + 2 : j + 1]),
                sum(column_weights[j + 2 :]),
            )
            retained = sum(counts)
            error = 0.0
            for count, target in zip(counts, targets):
                error += (count / retained - target) ** 2
            key = (error, -retained, i, j)
            if best is None or key < best[0]:
                best = (key, counts, retained, error)
    return best


SIMPLE_XS = [256 * k for k in range(10)]
SIMPLE_IDS = grid_ids("XJ1", SIMPLE_XS, [0])
WEIGHTED_WEIGHTS = [2, 1, 4, 1, 1, 2, 1, 3, 1, 1]


class ParseTileIdTests(unittest.TestCase):
    def test_valid_ids_and_coordinate_order(self):
        record = ss.parse_tile_id("XJ1_000383_005887_0_0_256_256")
        self.assertEqual(
            record,
            {"parent": "XJ1", "x": 383, "y": 5887, "width": 256, "height": 256},
        )
        self.assertEqual(set(record), {"parent", "x", "y", "width", "height"})
        js = ss.parse_tile_id("JS1_000000_000256_0_0_256_256")
        self.assertEqual((js["parent"], js["x"], js["y"]), ("JS1", 0, 256))
        xj = ss.parse_tile_id("XJ1_001536_000000_0_0_256_256")
        self.assertEqual((xj["x"], xj["y"]), (1536, 0))

    def test_malformed_strings_rejected(self):
        bad = [
            "",
            "XJ1",
            "XJ1_1_2_3_4_5",
            "XJ1_000383_005887_0_0_256",
            "XJ1_000383_005887_0_0_256_256_0",
            "XJ1_abc_005887_0_0_256_256",
            "XJ1_000383_005887_x_0_256_256",
            "1XJ_000383_005887_0_0_256_256",
            "XJ1_000383_005887_0_0_256_256.bak",
            "XJ1_000383_005887_0_0_256_256/../../etc",
            "../XJ1_000383_005887_0_0_256_256",
            "XJ1_000383_005887_0_0_256_256 ",
            "XJ1_00 0383_005887_0_0_256_256",
            "XJ1_000\u0663383_005887_0_0_256_256",  # non-ASCII digit
        ]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ss.parse_tile_id(value)

    def test_negative_and_non_integer_strings_rejected(self):
        for value in [
            "XJ1_-1_0_0_0_256_256",
            "XJ1_0_-1_0_0_256_256",
            "XJ1_1.5_0_0_0_256_256",
            "XJ1_+1_0_0_0_256_256",
        ]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ss.parse_tile_id(value)

    def test_crop_offsets_must_be_zero(self):
        for value in [
            "XJ1_000383_005887_1_0_256_256",
            "XJ1_000383_005887_0_128_256_256",
        ]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ss.parse_tile_id(value)

    def test_size_must_be_256(self):
        for value in [
            "XJ1_000383_005887_0_0_256_128",
            "XJ1_000383_005887_0_0_128_256",
            "XJ1_000383_005887_0_0_512_512",
            "XJ1_000383_005887_0_0_256_255",
        ]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ss.parse_tile_id(value)

    def test_non_string_input_rejected(self):
        for value in [None, 123, 1.0, b"XJ1_000000_000000_0_0_256_256", []]:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    ss.parse_tile_id(value)


class NormalizeTests(unittest.TestCase):
    def test_duplicate_ids_rejected(self):
        duplicate = [SIMPLE_IDS[0], SIMPLE_IDS[1], SIMPLE_IDS[0]]
        with self.assertRaises(ValueError):
            ss.normalize_tiles(duplicate)
        with self.assertRaises(ValueError):
            ss.spatial_split(duplicate)

    def test_mapping_records_accepted_and_cross_checked(self):
        records = [{"id": i, **ss.parse_tile_id(i)} for i in SIMPLE_IDS[:3]]
        normalized = ss.normalize_tiles(records)
        self.assertEqual([r["id"] for r in normalized], SIMPLE_IDS[:3])

        for mutate in (
            lambda r: r.__setitem__("x", 128),
            lambda r: r.__setitem__("y", 256),
            lambda r: r.__setitem__("width", 128),
            lambda r: r.__setitem__("parent", "JS1"),
            lambda r: r.__setitem__("x", True),
        ):
            broken = dict(records[0])
            mutate(broken)
            with self.subTest(broken=broken):
                with self.assertRaises(ValueError):
                    ss.normalize_tiles([broken])

    def test_mapping_missing_keys_and_bad_ids_rejected(self):
        with self.assertRaises(ValueError):
            ss.normalize_tiles([{"id": SIMPLE_IDS[0]}])
        with self.assertRaises(ValueError):
            ss.normalize_tiles([{"id": "not-a-tile-id"}])
        with self.assertRaises(TypeError):
            ss.normalize_tiles(
                [{"id": 7, "parent": "XJ1", "x": 0, "y": 0, "width": 256, "height": 256}]
            )
        with self.assertRaises(ValueError):
            ss.normalize_tiles(["XJ1_000000_000000_0_0_256_256", 42])

    def test_single_value_and_none_rejected(self):
        with self.assertRaises(ValueError):
            ss.normalize_tiles(SIMPLE_IDS[0])
        with self.assertRaises(ValueError):
            ss.normalize_tiles(None)


class SpatialSplitTests(unittest.TestCase):
    def assert_gap_boundaries_consistent(self, parent, tile_size=256):
        """Reported cut boundaries must match the first/last stripe columns."""
        cuts = parent["cuts"]
        stripes = parent["stripes"]
        self.assertEqual(
            cuts["train_val"]["train_last_axis_coord"],
            stripes["train"]["axis_max"],
        )
        self.assertEqual(
            cuts["train_val"]["val_first_axis_coord"],
            stripes["val"]["axis_min"],
        )
        self.assertEqual(
            cuts["train_val"]["clear_gap_px"],
            cuts["train_val"]["val_first_axis_coord"]
            - cuts["train_val"]["train_last_axis_coord"]
            - tile_size,
        )
        self.assertEqual(
            cuts["val_test"]["val_last_axis_coord"],
            stripes["val"]["axis_max"],
        )
        self.assertEqual(
            cuts["val_test"]["test_first_axis_coord"],
            stripes["test"]["axis_min"],
        )
        self.assertEqual(
            cuts["val_test"]["clear_gap_px"],
            cuts["val_test"]["test_first_axis_coord"]
            - cuts["val_test"]["val_last_axis_coord"]
            - tile_size,
        )
        for name in ("train_val", "val_test"):
            self.assertGreaterEqual(cuts[name]["clear_gap_px"], 256)

    def test_simple_expected_cuts(self):
        result = ss.spatial_split(SIMPLE_IDS)
        assignment = result["assignment"]
        expected = {}
        for k in range(10):
            x = 256 * k
            if k <= 5:
                expected[tid("XJ1", x, 0)] = ss.TRAIN
            elif k == 7:
                expected[tid("XJ1", x, 0)] = ss.VAL
            elif k == 9:
                expected[tid("XJ1", x, 0)] = ss.TEST
            else:
                expected[tid("XJ1", x, 0)] = ss.EXCLUDED
        self.assertEqual(assignment, expected)
        self.assertEqual(
            result["excluded_reasons"][tid("XJ1", 1536, 0)], "gap_train_val"
        )
        self.assertEqual(
            result["excluded_reasons"][tid("XJ1", 2048, 0)], "gap_val_test"
        )

        parent = result["summary"]["parents"]["XJ1"]
        self.assertEqual(parent["axis"], "x")
        self.assertEqual(parent["objective"]["cut_columns"], [1536, 2048])
        self.assertEqual(parent["objective"]["cut_column_indices"], [5, 7])
        self.assertEqual(
            (parent["counts"]["train"], parent["counts"]["val"], parent["counts"]["test"]),
            (6, 1, 1),
        )
        self.assertTrue(result["audit"]["ok"])

    def test_minimal_five_columns(self):
        ids = grid_ids("XJ1", [256 * k for k in range(5)], [0])
        result = ss.spatial_split(ids)
        parent = result["summary"]["parents"]["XJ1"]
        self.assertEqual(parent["objective"]["cut_columns"], [256, 768])
        self.assertEqual(
            (parent["counts"]["train"], parent["counts"]["val"], parent["counts"]["test"]),
            (1, 1, 1),
        )
        self.assertEqual(parent["cuts"]["train_val"]["clear_gap_px"], 256)
        self.assertEqual(parent["cuts"]["val_test"]["clear_gap_px"], 256)
        self.assertEqual(parent["stripes"]["train"]["axis_min"], 0)
        self.assertEqual(parent["stripes"]["test"]["axis_max"], 1024)
        self.assertTrue(result["audit"]["ok"])

    def test_weighted_cuts_match_independent_bruteforce(self):
        ids = rows("JS1", [256 * k for k in range(len(WEIGHTED_WEIGHTS))], WEIGHTED_WEIGHTS)
        result = ss.spatial_split(ids)
        parent = result["summary"]["parents"]["JS1"]
        key, counts, retained, error = brute_force_cuts(WEIGHTED_WEIGHTS)
        self.assertEqual(parent["objective"]["cut_column_indices"], [key[2], key[3]])
        self.assertEqual(
            (parent["counts"]["train"], parent["counts"]["val"], parent["counts"]["test"]),
            counts,
        )
        self.assertEqual(parent["objective"]["retained"], retained)
        self.assertAlmostEqual(parent["objective"]["error"], error, places=12)
        self.assertAlmostEqual(sum(parent["fractions"].values()), 1.0, places=12)

    def test_custom_targets_follow_bruteforce(self):
        weights = [1] * 10
        ids = rows("JS1", [256 * k for k in range(10)], weights)
        targets = (0.5, 0.25, 0.25)
        result = ss.spatial_split(ids, targets=targets)
        parent = result["summary"]["parents"]["JS1"]
        key, counts, retained, error = brute_force_cuts(weights, targets)
        self.assertEqual(tuple(parent["objective"]["targets"]), targets)
        self.assertEqual(parent["objective"]["cut_column_indices"], [key[2], key[3]])
        self.assertAlmostEqual(parent["objective"]["error"], error, places=12)

    def test_bad_targets_rejected(self):
        for bad in [
            (0.7, 0.15),
            (0.7, 0.15, 0.2),
            (0.7, 0.15, 0.14),
            (1.0, 0.0, 0.0),
            ("a", 0.5, 0.5),
            (True, 0.5, 0.5),
        ]:
            with self.subTest(targets=bad):
                with self.assertRaises(ValueError):
                    ss.spatial_split(SIMPLE_IDS, targets=bad)

    def test_lattice_drops_128_stride_overlaps(self):
        xs = [128 * k for k in range(9)]  # 0 .. 1024, half-stride grid
        ids = grid_ids("XJ1", xs, [0, 256])
        result = ss.spatial_split(ids)
        dropped = sorted(
            i
            for i, label in result["assignment"].items()
            if label == ss.EXCLUDED
            and result["excluded_reasons"][i] == ss.REASON_OVERLAP_LATTICE
        )
        expected_dropped = sorted(
            grid_ids("XJ1", [128, 384, 640, 896], [0, 256])
        )
        self.assertEqual(dropped, expected_dropped)

        parent = result["summary"]["parents"]["XJ1"]
        self.assertEqual(parent["lattice"]["anchor_x"], 0)
        self.assertEqual(parent["lattice"]["anchor_y"], 0)
        self.assertEqual(parent["lattice"]["selected"], 10)
        self.assertEqual(parent["lattice"]["dropped_overlap_lattice"], 8)
        self.assertEqual(parent["counts"]["excluded_overlap_lattice"], 8)

        # The raw 128-stride tiling really does overlap pixel area.
        raw = ss.spatial_audit(ids, {i: ss.TRAIN for i in ids})
        self.assertGreater(raw["overlap_conflicts"], 0)
        self.assertFalse(raw["ok"])
        # A dropped tile overlapped a kept neighbour.
        self.assertTrue(boxes_overlap(tid("XJ1", 128, 0), tid("XJ1", 0, 0)))
        # After splitting, selected tiles never overlap.
        self.assertTrue(result["audit"]["ok"])
        self.assertEqual(result["audit"]["overlap_conflicts"], 0)
        self.assertGreaterEqual(result["audit"]["min_cross_split_gap_px"], 256)

    def test_deterministic_shuffle_and_seed_invariant(self):
        ids = rows("XJ1", [256 * k for k in range(10)], WEIGHTED_WEIGHTS)
        ids += grid_ids("JS1", [256 * k for k in range(6)], [0])
        random.seed(1)
        baseline = ss.spatial_split(ids)
        shuffled = list(ids)
        random.Random(999).shuffle(shuffled)
        random.seed(4242)
        again = ss.spatial_split(shuffled)
        self.assertEqual(baseline, again)

    def test_accounting_and_no_content_based_removal(self):
        ids = rows("XJ1", [256 * k for k in range(10)], WEIGHTED_WEIGHTS)
        result = ss.spatial_split(ids)
        assignment = result["assignment"]
        reasons = result["excluded_reasons"]

        self.assertEqual(set(assignment), set(ids))
        self.assertEqual(set(reasons), {i for i, v in assignment.items() if v == ss.EXCLUDED})
        for label in assignment.values():
            self.assertIn(label, ss.ASSIGN_VALUES)
        for reason in reasons.values():
            self.assertIn(
                reason,
                (
                    ss.REASON_OVERLAP_LATTICE,
                    ss.REASON_GAP_TRAIN_VAL,
                    ss.REASON_GAP_VAL_TEST,
                ),
            )

        counts = result["summary"]["counts"]
        for label in ("train", "val", "test", "excluded"):
            self.assertEqual(
                counts[label], sum(1 for v in assignment.values() if v == label)
            )
        self.assertEqual(
            result["summary"]["retained"],
            counts["train"] + counts["val"] + counts["test"],
        )
        self.assertEqual(result["summary"]["n_ids"], len(ids))
        self.assertTrue(result["summary"]["all_ids_accounted"])

        parent = result["summary"]["parents"]["XJ1"]
        self.assertEqual(parent["counts"]["excluded_overlap_lattice"], 0)
        self.assertEqual(
            parent["counts"]["excluded_gap_train_val"]
            + parent["counts"]["excluded_gap_val_test"],
            counts["excluded"],
        )
        # Exactly two whole columns are removed for the gaps; nothing else
        # is filtered (no label / foreground based removal).
        self.assertEqual(
            counts["train"] + counts["val"] + counts["test"]
            + parent["counts"]["excluded_gap_train_val"]
            + parent["counts"]["excluded_gap_val_test"],
            parent["lattice"]["selected"],
        )
        i, j = parent["objective"]["cut_column_indices"]
        self.assertEqual(
            (parent["counts"]["excluded_gap_train_val"], parent["counts"]["excluded_gap_val_test"]),
            (WEIGHTED_WEIGHTS[i + 1], WEIGHTED_WEIGHTS[j + 1]),
        )

    def test_gap_boundaries_and_clear_gap(self):
        result = ss.spatial_split(SIMPLE_IDS)
        parent = result["summary"]["parents"]["XJ1"]
        for name in ("train_val", "val_test"):
            cut = parent["cuts"][name]
            self.assertGreaterEqual(cut["clear_gap_px"], 256)
        self.assertEqual(parent["cuts"]["train_val"]["gap_axis_coord"], 1536)
        self.assertEqual(parent["cuts"]["val_test"]["gap_axis_coord"], 2048)
        self.assert_gap_boundaries_consistent(parent)
        audit = result["audit"]
        self.assertEqual(audit["gap_violations"], 0)
        self.assertGreater(audit["cross_split_pairs"], 0)
        self.assertEqual(audit["min_cross_split_gap_px"], 256)

    def test_multi_column_test_reports_first_test_column(self):
        # 12 columns: best cut is train=7, val=1, test=2 (err 0.005), so the
        # test stripe has two columns and its first column (2560) must be the
        # reported boundary -- not the last test column (2816).
        ids = grid_ids("XJ1", [256 * k for k in range(12)], [0])
        result = ss.spatial_split(ids)
        parent = result["summary"]["parents"]["XJ1"]
        self.assertEqual(parent["objective"]["cut_column_indices"], [6, 8])
        self.assertEqual(parent["stripes"]["test"]["axis_min"], 2560)
        self.assertEqual(parent["stripes"]["test"]["axis_max"], 2816)
        self.assertEqual(parent["counts"]["test"], 2)
        self.assertEqual(parent["cuts"]["val_test"]["test_first_axis_coord"], 2560)
        self.assertEqual(parent["cuts"]["val_test"]["val_last_axis_coord"], 2048)
        self.assertEqual(parent["cuts"]["val_test"]["clear_gap_px"], 256)
        self.assertNotEqual(
            parent["cuts"]["val_test"]["test_first_axis_coord"],
            parent["stripes"]["test"]["axis_max"],
        )
        self.assert_gap_boundaries_consistent(parent)
        self.assertTrue(result["audit"]["ok"])

    def test_longest_axis_selection(self):
        x_dominant = grid_ids("XJ1", [256 * k for k in range(10)], [0, 256])
        parent = ss.spatial_split(x_dominant)["summary"]["parents"]["XJ1"]
        self.assertEqual(parent["axis"], "x")
        self.assertGreaterEqual(parent["spans"]["x"], parent["spans"]["y"])
        for split in ("train", "val", "test"):
            self.assertGreater(parent["counts"][split], 0)
            self.assertIn(split, parent["stripes"])

        y_dominant = grid_ids("JS1", [0, 256, 512], [256 * k for k in range(8)])
        parent_y = ss.spatial_split(y_dominant)["summary"]["parents"]["JS1"]
        self.assertEqual(parent_y["axis"], "y")
        self.assertGreaterEqual(parent_y["spans"]["y"], parent_y["spans"]["x"])
        for cut in parent_y["objective"]["cut_columns"]:
            self.assertEqual(cut % 256, 0)
            self.assertIn(cut, parent_y["columns"])
        for split in ("train", "val", "test"):
            self.assertGreater(parent_y["counts"][split], 0)

    def test_too_few_lattice_columns_rejected(self):
        cases = {
            "four_columns": grid_ids("XJ1", [256 * k for k in range(4)], [0]),
            "lattice_reduces_to_three": grid_ids("JS1", [128 * k for k in range(6)], [0]),
            "no_lattice_survivor": [tid("XK1", 0, 128), tid("XK1", 128, 0)],
            "two_scattered_columns": grid_ids("XK1", [0, 512], [0]),
        }
        for name, ids in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    ss.spatial_split(ids)

    def test_multi_parent_isolation(self):
        ids = grid_ids("XJ1", [256 * k for k in range(10)], [0])
        ids += grid_ids("JS1", [256 * k for k in range(6)], [0])
        result = ss.spatial_split(ids)
        self.assertEqual(set(result["summary"]["parents"]), {"XJ1", "JS1"})
        for parent_name, info in result["summary"]["parents"].items():
            for split in ("train", "val", "test"):
                self.assertGreater(
                    info["counts"][split], 0, msg="%s/%s" % (parent_name, split)
                )
        self.assertEqual(
            result["summary"]["parents"]["XJ1"]["objective"]["cut_columns"],
            [1536, 2048],
        )
        self.assertEqual(
            result["summary"]["parents"]["JS1"]["objective"]["cut_columns"],
            [512, 1024],
        )
        self.assertTrue(result["audit"]["ok"])
        # Per-parent counts sum to the global counts.
        global_counts = result["summary"]["counts"]
        for split in ("train", "val", "test"):
            self.assertEqual(
                global_counts[split],
                sum(
                    p["counts"][split]
                    for p in result["summary"]["parents"].values()
                ),
            )

    def test_large_grid_audit_and_fractions(self):
        ids = grid_ids("XJ1", [256 * k for k in range(20)], [256 * k for k in range(7)])
        result = ss.spatial_split(ids)
        self.assertEqual(len(result["assignment"]), 140)
        self.assertTrue(result["audit"]["ok"])
        self.assertEqual(result["audit"]["overlap_conflicts"], 0)
        self.assertGreaterEqual(result["audit"]["min_cross_split_gap_px"], 256)
        parent = result["summary"]["parents"]["XJ1"]
        self.assertEqual(parent["lattice"]["selected"], 140)
        self.assertGreaterEqual(
            parent["stripes"]["test"]["axis_max"] - parent["stripes"]["test"]["axis_min"], 256
        )
        self.assert_gap_boundaries_consistent(parent)
        for split, target in zip(("train", "val", "test"), (0.70, 0.15, 0.15)):
            self.assertLess(abs(parent["fractions"][split] - target), 0.1)

    def test_empty_input(self):
        result = ss.spatial_split([])
        self.assertEqual(result["assignment"], {})
        self.assertEqual(result["excluded_reasons"], {})
        self.assertEqual(result["summary"]["n_ids"], 0)
        self.assertEqual(result["summary"]["parents"], {})
        self.assertTrue(result["summary"]["all_ids_accounted"])
        self.assertTrue(result["audit"]["ok"])
        audit = ss.spatial_audit([], {})
        self.assertTrue(audit["ok"])
        self.assertIsNone(audit["min_cross_split_gap_px"])


class BuildSplitTests(unittest.TestCase):
    def test_build_split_exact_contract(self):
        ids = rows("XJ1", [256 * k for k in range(10)], WEIGHTED_WEIGHTS)
        ids += grid_ids("JS1", [256 * k for k in range(6)], [0])
        built = ss.build_split(ids)
        self.assertEqual(set(built), {"assignments", "excluded_reasons", "summary"})

        full = ss.spatial_split(ids)
        self.assertEqual(built["assignments"], full["assignment"])
        self.assertEqual(built["excluded_reasons"], full["excluded_reasons"])
        self.assertEqual(built["summary"], full["summary"])

        self.assertEqual(set(built["assignments"]), set(ids))
        self.assertEqual(
            set(built["excluded_reasons"]),
            {i for i, v in built["assignments"].items() if v == ss.EXCLUDED},
        )
        for label in built["assignments"].values():
            self.assertIn(label, ss.ASSIGN_VALUES)
        self.assertTrue(ss.spatial_audit(ids, built["assignments"])["ok"])
        for parent_name, info in built["summary"]["parents"].items():
            for split in ("train", "val", "test"):
                self.assertGreater(
                    info["counts"][split], 0, msg="%s/%s" % (parent_name, split)
                )

    def test_build_split_error_surface(self):
        with self.assertRaises(ValueError):
            ss.build_split(grid_ids("XJ1", [256 * k for k in range(4)], [0]))
        with self.assertRaises(ValueError):
            ss.build_split([SIMPLE_IDS[0], SIMPLE_IDS[0]])
        with self.assertRaises(ValueError):
            ss.build_split(["bogus-id"])

    def test_build_split_shuffle_invariant(self):
        ids = grid_ids("XJ1", [128 * k for k in range(9)], [0, 256])
        shuffled = list(ids)
        random.Random(5).shuffle(shuffled)
        self.assertEqual(ss.build_split(ids), ss.build_split(shuffled))

    def test_build_split_real_shape_xj1_like(self):
        # Spans mirror the verified XJ1 manifest (x 383..3967, y 255..7295);
        # 1580 of the 1624 full 128-stride grid tiles are present.  Extremes
        # are kept so the verified min/max anchor is preserved.
        corners = [(383, 255), (383, 7295), (3967, 255), (3967, 7295)]
        pool = [
            tid("XJ1", 383 + 128 * i, 255 + 128 * j)
            for i in range(29)
            for j in range(56)
            if (383 + 128 * i, 255 + 128 * j) not in corners
        ]
        ids = random.Random(20261005).sample(pool, 1576)
        ids += [tid("XJ1", x, y) for x, y in corners]
        self.assertEqual(len(ids), 1580)

        built = ss.build_split(ids)
        parent = built["summary"]["parents"]["XJ1"]
        self.assertEqual(parent["axis"], "y")
        # Spans are measured over lattice-retained tiles: the top row
        # y = 7295 lies off the anchored 256 lattice and is dropped, so the
        # retained y extent ends at 7167 + 256.
        self.assertEqual(parent["spans"], {"x": 3840, "y": 7168})
        self.assertEqual(parent["lattice"]["anchor_x"], 383)
        self.assertEqual(parent["lattice"]["anchor_y"], 255)
        self.assertGreaterEqual(parent["n_columns"], 5)
        self.assertTrue(ss.spatial_audit(ids, built["assignments"])["ok"])
        counts = built["summary"]["counts"]
        self.assertEqual(
            counts["train"] + counts["val"] + counts["test"] + counts["excluded"],
            1580,
        )
        for split in ("train", "val", "test"):
            self.assertGreater(parent["counts"][split], 0)


class SpatialAuditTests(unittest.TestCase):
    def test_audit_detects_raw_128_overlaps(self):
        ids = grid_ids("XJ1", [128 * k for k in range(8)], [0])
        audit = ss.spatial_audit(ids, {i: ss.TRAIN for i in ids})
        self.assertGreater(audit["overlap_conflicts"], 0)
        self.assertFalse(audit["ok"])
        self.assertTrue(audit["failures"])

    def test_audit_ok_on_split_result(self):
        result = ss.spatial_split(SIMPLE_IDS)
        audit = ss.spatial_audit(SIMPLE_IDS, result["assignment"])
        self.assertEqual(audit, result["audit"])
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["overlap_conflicts"], 0)
        self.assertGreaterEqual(audit["min_cross_split_gap_px"], 256)
        self.assertEqual(
            audit["parents"]["XJ1"]["n_by_split"],
            {"train": 6, "val": 1, "test": 1},
        )

    def test_audit_detects_gap_violation(self):
        result = ss.spatial_split(SIMPLE_IDS)
        assignment = dict(result["assignment"])
        assignment[tid("XJ1", 1536, 0)] = ss.VAL  # pull a gap tile into val
        audit = ss.spatial_audit(SIMPLE_IDS, assignment)
        self.assertEqual(audit["overlap_conflicts"], 0)
        self.assertGreaterEqual(audit["gap_violations"], 1)
        self.assertFalse(audit["ok"])
        self.assertLess(audit["min_cross_split_gap_px"], 256)
        self.assertTrue(audit["gap_violation_examples"])

    def test_audit_detects_injected_overlap(self):
        ids = grid_ids("XJ1", [128 * k for k in range(9)], [0, 256])
        result = ss.spatial_split(ids)
        assignment = dict(result["assignment"])
        assignment[tid("XJ1", 128, 0)] = ss.TRAIN  # off-lattice tile back in
        audit = ss.spatial_audit(ids, assignment)
        self.assertGreaterEqual(audit["overlap_conflicts"], 1)
        self.assertFalse(audit["ok"])

    def test_audit_rejects_bad_coverage_and_labels(self):
        with self.assertRaises(ValueError):
            ss.spatial_audit(SIMPLE_IDS, {})
        partial = {i: ss.TRAIN for i in SIMPLE_IDS[:-1]}
        with self.assertRaises(ValueError):
            ss.spatial_audit(SIMPLE_IDS, partial)
        extra = {i: ss.TRAIN for i in SIMPLE_IDS}
        extra["XJ1_999999_999999_0_0_256_256"] = ss.TRAIN
        with self.assertRaises(ValueError):
            ss.spatial_audit(SIMPLE_IDS, extra)
        bad_label = {i: "holdout" for i in SIMPLE_IDS}
        with self.assertRaises(ValueError):
            ss.spatial_audit(SIMPLE_IDS, bad_label)
        with self.assertRaises(ValueError):
            ss.spatial_audit(SIMPLE_IDS, {i: ss.TRAIN for i in SIMPLE_IDS}, min_gap_px=0)
        with self.assertRaises(ValueError):
            ss.spatial_audit(SIMPLE_IDS, None)

    def test_audit_accepts_mapping_tiles(self):
        result = ss.spatial_split(SIMPLE_IDS)
        records = [{"id": i, **ss.parse_tile_id(i)} for i in SIMPLE_IDS]
        from_strings = ss.spatial_audit(SIMPLE_IDS, result["assignment"])
        from_records = ss.spatial_audit(records, result["assignment"])
        self.assertEqual(from_strings, from_records)

    def test_audit_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            ss.spatial_audit(
                SIMPLE_IDS + SIMPLE_IDS[:1],
                {i: ss.TRAIN for i in SIMPLE_IDS},
            )


if __name__ == "__main__":
    unittest.main()
