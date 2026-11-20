from __future__ import annotations

from pathlib import Path
from typing import Any
from copy import deepcopy

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return resolve_dataset_variant(yaml.safe_load(stream))


def resolve_dataset_variant(
    config: dict[str, Any], variant: str | None = None
) -> dict[str, Any]:
    """Resolve one configured dataset protocol without changing loader logic."""
    resolved = deepcopy(config)
    dataset = resolved.get("dataset")
    if not isinstance(dataset, dict):
        raise KeyError("Config must contain a dataset mapping")

    variants = dataset.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise KeyError("dataset.variants must define protocol variants")
    selected_name = str(variant or dataset.get("variant", ""))
    if selected_name not in variants:
        raise ValueError(
            f"Unknown dataset.variant {selected_name!r}; expected one of {sorted(variants)}"
        )
    selected = variants[selected_name]
    if not isinstance(selected, dict):
        raise TypeError(f"dataset.variants.{selected_name} must be a mapping")

    required = {
        "protocol",
        "purpose",
        "root",
        "image_dir",
        "mask_dir",
        "manifests",
        "image_suffix",
        "mask_suffix",
        "input_size",
    }
    missing = sorted(required.difference(selected))
    if missing:
        raise KeyError(f"dataset.variants.{selected_name} missing fields: {missing}")

    # Keep common fields (bands, batch sizes, workers) and overlay only paths,
    # suffixes and spatial size from the selected protocol.
    dataset.update(deepcopy(selected))
    dataset["variant"] = selected_name
    resolved["dataset"] = dataset
    return resolved


class AI4BoundariesSEANetDataset(Dataset):
    """Read the three native AI4Boundaries targets required by official SEANet."""

    def __init__(self, config: dict[str, Any], split: str, augment: bool | None = None) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")

        dataset = config["dataset"]
        augmentation = config["augmentation"]
        normalization = config["normalization"]
        self.split = split
        self.image_dir = Path(dataset["image_dir"]) / split
        self.mask_dir = Path(dataset["mask_dir"]) / split
        self.image_suffix = str(dataset["image_suffix"])
        self.mask_suffix = str(dataset["mask_suffix"])
        self.image_bands = tuple(int(v) for v in dataset["image_bands"])
        self.region_band = int(dataset["region_band"])
        self.boundary_band = int(dataset["boundary_band"])
        self.distance_band = int(dataset["distance_band"])
        self.size = tuple(int(v) for v in dataset["input_size"])
        self.mean = torch.tensor(normalization["mean"], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(normalization["std"], dtype=torch.float32).view(3, 1, 1)
        self.augment = split == "train" and augmentation["enabled"] if augment is None else augment
        self.hflip = float(augmentation["horizontal_flip_probability"])
        self.vflip = float(augmentation["vertical_flip_probability"])
        self.rotate90 = float(augmentation["rotate_90_probability"])

        manifest = Path(dataset["manifests"][split])
        self.image_ids = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.image_ids:
            raise RuntimeError(f"Empty split manifest: {manifest}")

        missing: list[str] = []
        for image_id in self.image_ids:
            if not self._image_path(image_id).is_file() or not self._mask_path(image_id).is_file():
                missing.append(image_id)
        if missing:
            raise FileNotFoundError(f"Missing AI4Boundaries pairs in {split}: {missing[:10]}")

    def _image_path(self, image_id: str) -> Path:
        return self.image_dir / f"{image_id}{self.image_suffix}"

    def _mask_path(self, image_id: str) -> Path:
        return self.mask_dir / f"{image_id}{self.mask_suffix}"

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_id = self.image_ids[index]
        with rasterio.open(self._image_path(image_id)) as source:
            image_array = source.read(self.image_bands, out_dtype="float32")
        with rasterio.open(self._mask_path(image_id)) as source:
            region_array = source.read(self.region_band, out_dtype="float32")
            boundary_array = source.read(self.boundary_band, out_dtype="float32")
            distance_array = source.read(self.distance_band, out_dtype="float32")

        image = torch.from_numpy(np.ascontiguousarray(image_array)) / 255.0
        region = torch.from_numpy(np.ascontiguousarray(region_array > 0.5)).float().unsqueeze(0)
        boundary = torch.from_numpy(np.ascontiguousarray(boundary_array > 0.5)).float().unsqueeze(0)
        distance = torch.from_numpy(np.ascontiguousarray(np.clip(distance_array, 0.0, 1.0))).float().unsqueeze(0)

        if tuple(image.shape[-2:]) != self.size:
            image = F.interpolate(image.unsqueeze(0), self.size, mode="bilinear", align_corners=False).squeeze(0)
            region = F.interpolate(region.unsqueeze(0), self.size, mode="nearest").squeeze(0)
            boundary = F.interpolate(boundary.unsqueeze(0), self.size, mode="nearest").squeeze(0)
            distance = F.interpolate(distance.unsqueeze(0), self.size, mode="bilinear", align_corners=False).squeeze(0)

        if self.augment:
            if torch.rand(()) < self.hflip:
                image, region, boundary, distance = [torch.flip(v, (-1,)) for v in (image, region, boundary, distance)]
            if torch.rand(()) < self.vflip:
                image, region, boundary, distance = [torch.flip(v, (-2,)) for v in (image, region, boundary, distance)]
            if torch.rand(()) < self.rotate90:
                k = int(torch.randint(1, 4, ()).item())
                image, region, boundary, distance = [torch.rot90(v, k, (-2, -1)) for v in (image, region, boundary, distance)]

        image = (image - self.mean) / self.std
        return {
            "image_id": image_id,
            "image": image.contiguous(),
            "region": region.contiguous(),
            "boundary": boundary.contiguous(),
            "distance": distance.contiguous(),
        }
