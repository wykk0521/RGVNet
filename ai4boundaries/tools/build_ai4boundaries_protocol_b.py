#!/usr/bin/env python3
"""Build and verify the AI4Boundaries parcel-focused 256x256 protocol."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import yaml
from rasterio.windows import Window
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai4boundaries.dataset import (  # noqa: E402
    AI4BoundariesSEANetDataset,
    resolve_dataset_variant,
)


SPLITS = ("train", "val", "test")
EXPECTED_PARENT_COUNTS = {"train": 5319, "val": 1140, "test": 1139}
PATCH_SIZE = 256
PATCHES = (
    (0, 0, 0),
    (1, 256, 0),
    (2, 0, 256),
    (3, 256, 256),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Protocol B from Protocol A without changing parent splits."
    )
    parser.add_argument("--config", default="ai4boundaries/config.yaml")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly delete and rebuild only the configured Protocol B root.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not build; rerun all checks against an existing Protocol B.",
    )
    return parser.parse_args()


def read_raw_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Invalid YAML mapping: {path}")
    return config


def protocol_configs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = config.get("dataset", {})
    variants = dataset.get("variants", {})
    if "official_complete" not in variants or "parcel_focused" not in variants:
        raise KeyError("Config must define official_complete and parcel_focused variants")
    protocol_a = variants["official_complete"]
    protocol_b = variants["parcel_focused"]
    source_root = Path(protocol_a["root"]).resolve()
    output_root = Path(protocol_b["root"]).resolve()
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError(
            f"Unsafe Protocol B output {output_root}; it must be separate from {source_root}"
        )
    if tuple(protocol_a["input_size"]) != (512, 512):
        raise ValueError("Protocol A input_size must remain [512, 512]")
    if tuple(protocol_b["input_size"]) != (256, 256):
        raise ValueError("Protocol B input_size must be [256, 256]")
    return protocol_a, protocol_b


def tree_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def read_split_ids(protocol: dict[str, Any]) -> dict[str, list[str]]:
    split_ids: dict[str, list[str]] = {}
    for split in SPLITS:
        manifest = Path(protocol["manifests"][split])
        ids = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(ids) != EXPECTED_PARENT_COUNTS[split]:
            raise RuntimeError(
                f"Protocol A {split} count is {len(ids)}, expected {EXPECTED_PARENT_COUNTS[split]}"
            )
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate parent IDs inside Protocol A {split}")
        split_ids[split] = ids

    parent_sets = {split: set(ids) for split, ids in split_ids.items()}
    assert_disjoint(parent_sets, "Protocol A parent")
    return split_ids


def assert_disjoint(sets: dict[str, set[str]], label: str) -> None:
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = sets[left].intersection(sets[right])
            if overlap:
                raise RuntimeError(
                    f"{label} leakage between {left} and {right}: {sorted(overlap)[:10]}"
                )


def occupancy_bin(value: float) -> str:
    if value == 0.0:
        return "0%"
    if value <= 0.05:
        return "0-5%"
    if value <= 0.20:
        return "5-20%"
    if value <= 0.50:
        return "20-50%"
    return ">50%"


def empty_occupancy_counts() -> dict[str, int]:
    return {key: 0 for key in ("0%", "0-5%", "5-20%", "20-50%", ">50%")}


def output_profile(source: rasterio.io.DatasetReader, transform: Any) -> dict[str, Any]:
    profile = source.profile.copy()
    profile.update(width=PATCH_SIZE, height=PATCH_SIZE, transform=transform)
    if profile.get("tiled"):
        profile["blockxsize"] = min(PATCH_SIZE, int(profile.get("blockxsize", PATCH_SIZE)))
        profile["blockysize"] = min(PATCH_SIZE, int(profile.get("blockysize", PATCH_SIZE)))
    return profile


def validate_parent_pair(
    image_source: rasterio.io.DatasetReader,
    mask_source: rasterio.io.DatasetReader,
    parent_id: str,
    config: dict[str, Any],
) -> None:
    for source, label in ((image_source, "image"), (mask_source, "mask")):
        if (source.height, source.width) != (512, 512):
            raise RuntimeError(
                f"{parent_id} {label} shape is {(source.height, source.width)}, expected (512, 512)"
            )
    if image_source.count < max(int(v) for v in config["dataset"]["image_bands"]):
        raise RuntimeError(f"Insufficient image bands for {parent_id}")
    if mask_source.count < max(
        int(config["dataset"][key])
        for key in ("region_band", "boundary_band", "distance_band")
    ):
        raise RuntimeError(f"Insufficient mask bands for {parent_id}")
    if image_source.crs != mask_source.crs or not image_source.transform.almost_equals(
        mask_source.transform
    ):
        raise RuntimeError(f"Image/mask georeferencing mismatch for {parent_id}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_protocol(
    config: dict[str, Any],
    protocol_a: dict[str, Any],
    protocol_b: dict[str, Any],
    split_ids: dict[str, list[str]],
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    image_bands = tuple(int(v) for v in config["dataset"]["image_bands"])
    region_band = int(config["dataset"]["region_band"])
    records: list[dict[str, Any]] = []
    protocol_a_stats: dict[str, Any] = {}
    protocol_b_stats: dict[str, Any] = {}

    for split in SPLITS:
        image_out_dir = Path(protocol_b["image_dir"]) / split
        mask_out_dir = Path(protocol_b["mask_dir"]) / split
        image_out_dir.mkdir(parents=True, exist_ok=False)
        mask_out_dir.mkdir(parents=True, exist_ok=False)
        kept_ids: list[str] = []
        a_bins = empty_occupancy_counts()
        b_bins = empty_occupancy_counts()
        zero_count = 0

        for parent_id in split_ids[split]:
            image_path = Path(protocol_a["image_dir"]) / split / f"{parent_id}{protocol_a['image_suffix']}"
            mask_path = Path(protocol_a["mask_dir"]) / split / f"{parent_id}{protocol_a['mask_suffix']}"
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Missing Protocol A pair: {image_path}, {mask_path}")

            with rasterio.open(image_path) as image_source, rasterio.open(mask_path) as mask_source:
                validate_parent_pair(image_source, mask_source, parent_id, config)
                image_array = image_source.read(image_bands)
                mask_array = mask_source.read()
                full_region = mask_array[region_band - 1] > 0.5
                a_bins[occupancy_bin(float(full_region.mean()))] += 1

                for patch_index, x, y in PATCHES:
                    patch_id = f"{parent_id}_{patch_index}"
                    image_name = f"{patch_id}{protocol_b['image_suffix']}"
                    mask_name = f"{patch_id}{protocol_b['mask_suffix']}"
                    ys = slice(y, y + PATCH_SIZE)
                    xs = slice(x, x + PATCH_SIZE)
                    region_patch = full_region[ys, xs]
                    foreground_pixels = int(region_patch.sum())
                    occupancy = foreground_pixels / float(PATCH_SIZE * PATCH_SIZE)
                    kept = foreground_pixels > 0
                    record = {
                        "patch_name": image_name,
                        "mask_name": mask_name,
                        "patch_id": patch_id,
                        "parent_image": image_path.name,
                        "parent_id": parent_id,
                        "split": split,
                        "patch_index": patch_index,
                        "x": x,
                        "y": y,
                        "width": PATCH_SIZE,
                        "height": PATCH_SIZE,
                        "foreground_pixels": foreground_pixels,
                        "occupancy": occupancy,
                        "kept": kept,
                    }
                    records.append(record)
                    if not kept:
                        zero_count += 1
                        continue

                    kept_ids.append(patch_id)
                    b_bins[occupancy_bin(occupancy)] += 1
                    window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                    image_transform = image_source.window_transform(window)
                    mask_transform = mask_source.window_transform(window)
                    with rasterio.open(
                        image_out_dir / image_name,
                        "w",
                        **output_profile(image_source, image_transform),
                    ) as destination:
                        destination.write(image_array[:, ys, xs])
                    with rasterio.open(
                        mask_out_dir / mask_name,
                        "w",
                        **output_profile(mask_source, mask_transform),
                    ) as destination:
                        destination.write(mask_array[:, ys, xs])

        manifest_path = Path(protocol_b["manifests"][split])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("\n".join(kept_ids) + "\n", encoding="utf-8")
        generated = len(split_ids[split]) * len(PATCHES)
        retained = len(kept_ids)
        protocol_a_stats[split] = {
            "sample_count": len(split_ids[split]),
            "occupancy_distribution": a_bins,
        }
        protocol_b_stats[split] = {
            "parent_image_count": len(split_ids[split]),
            "generated_patch_count": generated,
            "zero_parcel_patch_count": zero_count,
            "retained_patch_count": retained,
            "removed_ratio": zero_count / generated,
            "occupancy_distribution_retained": b_bins,
        }

    manifest_payload = {
        "protocol": "B",
        "variant": "parcel_focused",
        "source_variant": "official_complete",
        "patch_size": PATCH_SIZE,
        "stride": PATCH_SIZE,
        "overlap": 0,
        "filter": "keep if and only if region foreground_pixels > 0",
        "label_strategy": "crop the existing full-parent four-band mask; do not regenerate boundary/distance",
        "patch_index_order": {
            "0": "top_left",
            "1": "top_right",
            "2": "bottom_left",
            "3": "bottom_right",
        },
        "records": records,
    }
    protocol_a_payload = {
        "protocol": "A",
        "variant": "official_complete",
        "modified": False,
        "splits": protocol_a_stats,
    }
    protocol_b_payload = {
        "protocol": "B",
        "variant": "parcel_focused",
        "occupancy_bin_definition": "0 < x <= 5%; 5% < x <= 20%; 20% < x <= 50%; x > 50%",
        "splits": protocol_b_stats,
    }
    write_json(output_root / "protocol_b_manifest.json", manifest_payload)
    write_json(output_root / "protocol_a_statistics.json", protocol_a_payload)
    write_json(output_root / "protocol_b_statistics.json", protocol_b_payload)
    return records, protocol_a_payload, protocol_b_payload


def load_existing_outputs(output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    required = (
        "protocol_b_manifest.json",
        "protocol_a_statistics.json",
        "protocol_b_statistics.json",
    )
    missing = [name for name in required if not (output_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Existing Protocol B is incomplete: {missing}")
    manifest = json.loads((output_root / required[0]).read_text(encoding="utf-8"))
    protocol_a_stats = json.loads((output_root / required[1]).read_text(encoding="utf-8"))
    protocol_b_stats = json.loads((output_root / required[2]).read_text(encoding="utf-8"))
    return manifest["records"], protocol_a_stats, protocol_b_stats


def validate_output_files(
    config: dict[str, Any],
    protocol_a: dict[str, Any],
    protocol_b: dict[str, Any],
    split_ids: dict[str, list[str]],
    records: list[dict[str, Any]],
    protocol_b_stats: dict[str, Any],
) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_parent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)
        by_parent[(record["split"], record["parent_id"])].append(record)

    parent_sets = {
        split: {record["parent_id"] for record in by_split[split]} for split in SPLITS
    }
    assert_disjoint(parent_sets, "Protocol B parent")
    for split in SPLITS:
        if parent_sets[split] != set(split_ids[split]):
            raise RuntimeError(f"Protocol B {split} parent set differs from Protocol A")

    patch_sets: dict[str, set[str]] = {}
    total_retained = 0
    for split in SPLITS:
        split_records = by_split[split]
        if len(split_records) != EXPECTED_PARENT_COUNTS[split] * len(PATCHES):
            raise RuntimeError(f"Unexpected generated record count for {split}")
        kept = [record for record in split_records if record["kept"]]
        if any(int(record["foreground_pixels"]) == 0 for record in kept):
            raise RuntimeError(f"Retained zero-parcel patch detected in {split}")
        patch_sets[split] = {record["patch_id"] for record in kept}
        if len(patch_sets[split]) != len(kept):
            raise RuntimeError(f"Duplicate retained patch IDs in {split}")
        manifest_ids = {
            line.strip()
            for line in Path(protocol_b["manifests"][split]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if manifest_ids != patch_sets[split]:
            raise RuntimeError(f"Protocol B manifest mismatch in {split}")

        expected_images = {record["patch_name"] for record in kept}
        expected_masks = {record["mask_name"] for record in kept}
        actual_images = {path.name for path in (Path(protocol_b["image_dir"]) / split).glob("*.tif")}
        actual_masks = {path.name for path in (Path(protocol_b["mask_dir"]) / split).glob("*.tif")}
        if actual_images != expected_images or actual_masks != expected_masks:
            raise RuntimeError(f"Protocol B image/mask file set mismatch in {split}")
        if len(kept) != int(protocol_b_stats["splits"][split]["retained_patch_count"]):
            raise RuntimeError(f"Protocol B statistics mismatch in {split}")
        total_retained += len(kept)

    assert_disjoint(patch_sets, "Protocol B patch")

    region_band = int(config["dataset"]["region_band"])
    checked = 0
    for (split, parent_id), parent_records in by_parent.items():
        image_source_path = Path(protocol_a["image_dir"]) / split / f"{parent_id}{protocol_a['image_suffix']}"
        mask_source_path = Path(protocol_a["mask_dir"]) / split / f"{parent_id}{protocol_a['mask_suffix']}"
        with rasterio.open(image_source_path) as image_source, rasterio.open(mask_source_path) as mask_source:
            for record in parent_records:
                if not record["kept"]:
                    continue
                x, y = int(record["x"]), int(record["y"])
                window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
                image_output_path = Path(protocol_b["image_dir"]) / split / record["patch_name"]
                mask_output_path = Path(protocol_b["mask_dir"]) / split / record["mask_name"]
                if image_output_path.stat().st_size == 0 or mask_output_path.stat().st_size == 0:
                    raise RuntimeError(f"Empty output file for {record['patch_id']}")
                with rasterio.open(image_output_path) as image_output, rasterio.open(mask_output_path) as mask_output:
                    if (image_output.height, image_output.width) != (PATCH_SIZE, PATCH_SIZE):
                        raise RuntimeError(f"Bad image shape for {record['patch_id']}")
                    if (mask_output.height, mask_output.width) != (PATCH_SIZE, PATCH_SIZE):
                        raise RuntimeError(f"Bad mask shape for {record['patch_id']}")
                    if image_output.count != image_source.count or mask_output.count != mask_source.count:
                        raise RuntimeError(f"Band count mismatch for {record['patch_id']}")
                    if not image_output.transform.almost_equals(image_source.window_transform(window)):
                        raise RuntimeError(f"Image crop transform mismatch for {record['patch_id']}")
                    if not mask_output.transform.almost_equals(mask_source.window_transform(window)):
                        raise RuntimeError(f"Mask crop transform mismatch for {record['patch_id']}")
                    if image_output.crs != image_source.crs or mask_output.crs != mask_source.crs:
                        raise RuntimeError(f"CRS mismatch for {record['patch_id']}")
                    if not np.array_equal(image_output.read(), image_source.read(window=window), equal_nan=True):
                        raise RuntimeError(f"Image crop content mismatch for {record['patch_id']}")
                    output_mask = mask_output.read()
                    if not np.array_equal(output_mask, mask_source.read(window=window), equal_nan=True):
                        raise RuntimeError(f"Mask/boundary/distance crop mismatch for {record['patch_id']}")
                    foreground = int(np.count_nonzero(output_mask[region_band - 1] > 0.5))
                    if foreground != int(record["foreground_pixels"]) or foreground == 0:
                        raise RuntimeError(f"Foreground audit mismatch for {record['patch_id']}")
                checked += 1

    if checked != total_retained:
        raise RuntimeError(f"Checked {checked} outputs, expected {total_retained}")
    return {
        "retained_files_checked": checked,
        "parent_leakage": False,
        "patch_leakage": False,
        "all_output_crops_match_source": True,
        "all_retained_patches_have_foreground": True,
    }


def validate_dataset_loaders(
    raw_config: dict[str, Any], protocol_b_stats: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant, spatial_size in (("official_complete", 512), ("parcel_focused", 256)):
        config = resolve_dataset_variant(raw_config, variant)
        result[variant] = {}
        for split in SPLITS:
            dataset = AI4BoundariesSEANetDataset(config, split, augment=False)
            expected = (
                EXPECTED_PARENT_COUNTS[split]
                if variant == "official_complete"
                else int(protocol_b_stats["splits"][split]["retained_patch_count"])
            )
            if len(dataset) != expected:
                raise RuntimeError(f"Dataset length mismatch for {variant}/{split}")
            loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
            batch = next(iter(loader))
            if tuple(batch["image"].shape[-2:]) != (spatial_size, spatial_size):
                raise RuntimeError(f"Image batch shape mismatch for {variant}/{split}")
            for key in ("region", "boundary", "distance"):
                tensor = batch[key]
                if tuple(tensor.shape[-2:]) != (spatial_size, spatial_size):
                    raise RuntimeError(f"{key} batch shape mismatch for {variant}/{split}")
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"Non-finite {key} batch for {variant}/{split}")
            result[variant][split] = {
                "dataset_length": len(dataset),
                "batch_image_shape": list(batch["image"].shape),
                "batch_target_shape": list(batch["region"].shape),
            }
    return result


def print_summary(protocol_b_stats: dict[str, Any]) -> None:
    print("Protocol B build/verification summary")
    for split in SPLITS:
        stats = protocol_b_stats["splits"][split]
        print(
            f"{split}: parents={stats['parent_image_count']} "
            f"generated={stats['generated_patch_count']} "
            f"zero_removed={stats['zero_parcel_patch_count']} "
            f"retained={stats['retained_patch_count']} "
            f"removed_ratio={stats['removed_ratio']:.6f} "
            f"occupancy={stats['occupancy_distribution_retained']}"
        )


def main() -> None:
    args = parse_args()
    if args.overwrite and args.verify_only:
        raise ValueError("--overwrite and --verify-only cannot be used together")
    config_path = Path(args.config).resolve()
    raw_config = read_raw_config(config_path)
    protocol_a, protocol_b = protocol_configs(raw_config)
    source_root = Path(protocol_a["root"]).resolve()
    output_root = Path(protocol_b["root"]).resolve()
    split_ids = read_split_ids(protocol_a)
    source_before = tree_snapshot(source_root)

    if args.verify_only:
        if not output_root.is_dir():
            raise FileNotFoundError(output_root)
        records, protocol_a_stats, protocol_b_stats = load_existing_outputs(output_root)
    else:
        if output_root.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Protocol B output already exists: {output_root}. "
                    "Exit without changes; pass --overwrite only for an intentional rebuild."
                )
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=False)
        records, protocol_a_stats, protocol_b_stats = build_protocol(
            raw_config, protocol_a, protocol_b, split_ids, output_root
        )

    file_checks = validate_output_files(
        raw_config, protocol_a, protocol_b, split_ids, records, protocol_b_stats
    )
    loader_checks = validate_dataset_loaders(raw_config, protocol_b_stats)
    source_after = tree_snapshot(source_root)
    if source_before != source_after:
        raise RuntimeError("Protocol A changed during Protocol B processing")

    report = {
        "status": "PASS",
        "config": str(config_path),
        "protocol_a_root": str(source_root),
        "protocol_a_file_count_before": len(source_before),
        "protocol_a_file_count_after": len(source_after),
        "protocol_a_unchanged": True,
        "protocol_b_root": str(output_root),
        "label_strategy": "full-parent boundary/distance labels cropped with the same window as image/region",
        "file_checks": file_checks,
        "dataset_loader_checks": loader_checks,
        "protocol_a_statistics": protocol_a_stats,
        "protocol_b_statistics": protocol_b_stats,
    }
    write_json(output_root / "protocol_b_sanity_report.json", report)
    print_summary(protocol_b_stats)
    print(json.dumps({"status": "PASS", "sanity_report": str(output_root / 'protocol_b_sanity_report.json')}, indent=2))


if __name__ == "__main__":
    main()
