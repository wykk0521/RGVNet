from __future__ import annotations

import math
from functools import partial
from numbers import Real
from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils import checkpoint as checkpoint_utils


class LocalSeparationRefinement(nn.Module):
    """Predict a bounded residual without differentiating through coarse predictions."""

    DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

    def __init__(
        self,
        hidden: int = 16,
        path_channels: int = 4,
        radii: Tuple[int, int] = (2, 4),
        max_delta: float = 2.0,
        checkpoint_groups: bool = True,
    ) -> None:
        super().__init__()
        if type(hidden) is not int or hidden != 16:
            raise ValueError("hidden must be exactly 16")
        if type(path_channels) is not int or path_channels != 4:
            raise ValueError("path_channels must be exactly 4")
        if not isinstance(radii, (tuple, list)) or tuple(radii) != (2, 4):
            raise ValueError("radii must be exactly (2, 4)")
        if any(type(radius) is not int for radius in radii):
            raise ValueError("radii must contain integers")
        if isinstance(max_delta, bool) or not isinstance(max_delta, Real):
            raise ValueError("max_delta must be positive and finite")
        if not math.isfinite(max_delta) or max_delta <= 0:
            raise ValueError("max_delta must be positive and finite")
        if type(checkpoint_groups) is not bool:
            raise TypeError("checkpoint_groups must be a bool")

        self.hidden = hidden
        self.path_channels = path_channels
        self.radii = tuple(radii)
        self.max_delta = float(max_delta)
        self.checkpoint_groups = checkpoint_groups

        self.projection = nn.Sequential(
            nn.Conv2d(64, 16, 1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
        )
        self.path_projection = nn.Conv2d(16, 4, 1, bias=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(94, 16, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(16, 16, 1, bias=True),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(50, 16, 1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1, bias=True),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def _sample(value: Tensor, row: int, col: int) -> Tensor:
        """Read value[..., y + row, x + col], zero-filling outside the image."""
        height, width = value.shape[-2:]
        y0, y1 = max(0, -row), min(height, height - row)
        x0, x1 = max(0, -col), min(width, width - col)
        if y0 >= y1 or x0 >= x1:
            return torch.zeros_like(value)
        crop = value[..., y0 + row : y1 + row, x0 + col : x1 + col]
        return F.pad(crop, (x0, width - x1, y0, height - y1))

    @staticmethod
    def _valid_mask(
        height: int,
        width: int,
        radius: int,
        direction: Tuple[int, int],
        device: Union[torch.device, str, None] = None,
    ) -> Tensor:
        """Return a [1, 1, H, W] boolean mask for one undirected group."""
        row, col = direction
        yy = torch.arange(height, device=device).view(height, 1)
        xx = torch.arange(width, device=device).view(1, width)
        valid = torch.ones((height, width), dtype=torch.bool, device=device)
        for distance in (-radius, -radius // 2, 0, radius // 2, radius):
            y, x = yy + distance * row, xx + distance * col
            valid = valid & (y >= 0) & (y < height) & (x >= 0) & (x < width)
        for distance in (-radius, radius):
            y, x = yy + distance * row, xx + distance * col
            valid = valid & (y >= 1) & (y < height - 1)
            valid = valid & (x >= 1) & (x < width - 1)
        return valid.unsqueeze(0).unsqueeze(0)

    def _path_statistics(
        self, value: Tensor, radius: int, direction: Tuple[int, int]
    ) -> Tensor:
        row, col = direction
        inner_minus = self._sample(value, -(radius // 2) * row, -(radius // 2) * col)
        inner_plus = self._sample(value, (radius // 2) * row, (radius // 2) * col)
        outer_minus = self._sample(value, -radius * row, -radius * col)
        outer_plus = self._sample(value, radius * row, radius * col)
        return torch.cat(
            (
                value,
                (inner_minus + inner_plus) * 0.5,
                (inner_minus - inner_plus).abs(),
                (outer_minus + outer_plus) * 0.5,
                (outer_minus - outer_plus).abs(),
            ),
            dim=1,
        )

    def _descriptor(
        self,
        s: Tensor,
        c: Tensor,
        u: Tensor,
        p: Tensor,
        b: Tensor,
        radius: int,
        direction: Tuple[int, int],
        local_mean: Union[Tensor, None] = None,
    ) -> Tensor:
        row, col = direction
        # Incomplete endpoint windows never contribute to group aggregation.
        if local_mean is None:
            local_mean = F.avg_pool2d(s, 3, stride=1, padding=1, count_include_pad=False)
        mu_minus = self._sample(local_mean, -radius * row, -radius * col)
        mu_plus = self._sample(local_mean, radius * row, radius * col)
        return torch.cat(
            (
                (mu_minus + mu_plus) * 0.5,
                (mu_minus - mu_plus).abs(),
                s,
                c,
                self._path_statistics(u, radius, direction),
                self._path_statistics(p, radius, direction),
                self._path_statistics(b, radius, direction),
            ),
            dim=1,
        )

    def _group_features(
        self,
        s: Tensor,
        c: Tensor,
        u: Tensor,
        p: Tensor,
        b: Tensor,
        local_mean: Union[Tensor, None] = None,
        *,
        radius: int,
        direction: Tuple[int, int],
    ) -> Tensor:
        return self.encoder(self._descriptor(s, c, u, p, b, radius, direction, local_mean))

    @staticmethod
    def _check_inputs(features: Tensor, z0: Tensor, edge_logp: Tensor) -> None:
        if features.ndim != 4 or features.shape[1] != 64:
            raise ValueError("features must have shape [B, 64, H, W]")
        batch, _, height, width = features.shape
        if min(batch, height, width) < 1:
            raise ValueError("batch and spatial dimensions must be positive")
        if z0.shape != (batch, 1, height, width):
            raise ValueError("z0 must have shape [B, 1, H, W]")
        if edge_logp.shape != (batch, 2, height, width):
            raise ValueError("edge_logp must have shape [B, 2, H, W]")
        if features.device != z0.device or features.device != edge_logp.device:
            raise ValueError("all inputs must be on the same device")
        if not all(value.is_floating_point() for value in (features, z0, edge_logp)):
            raise TypeError("all inputs must be floating-point tensors")

    def forward(
        self,
        features: Tensor,
        z0: Tensor,
        edge_logp: Tensor,
        return_aux: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        self._check_inputs(features, z0, edge_logp)
        s = self.projection(features)
        c = F.avg_pool2d(F.pad(s, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
        u = self.path_projection(s)
        p = torch.sigmoid(z0.detach())
        b = torch.exp(edge_logp.detach()[:, 1:2])
        local_mean = F.avg_pool2d(s, 3, stride=1, padding=1, count_include_pad=False)
        inputs = (s, c, u, p, b, local_mean)
        use_checkpoint = (
            self.checkpoint_groups
            and self.training
            and torch.is_grad_enabled()
            and any(value.requires_grad for value in inputs)
        )
        batch, _, height, width = s.shape
        total = torch.zeros_like(s)
        valid_count = torch.zeros((1, 1, height, width), dtype=torch.int64, device=s.device)
        for radius in self.radii:
            for direction in self.DIRECTIONS:
                mask = self._valid_mask(height, width, radius, direction, s.device)
                # Bind offsets now; recomputation must not observe later loop values.
                group = partial(self._group_features, radius=radius, direction=direction)
                if use_checkpoint:
                    encoded = checkpoint_utils.checkpoint(group, *inputs, use_reentrant=False)
                else:
                    encoded = group(*inputs)
                total = total + encoded * mask.to(encoded.dtype)
                valid_count = valid_count + mask.to(valid_count.dtype)

        h = total / valid_count.clamp_min(1).to(total.dtype)
        raw = self.head(torch.cat((s, c, h, p, b), dim=1))
        delta = (valid_count > 0).to(raw.dtype) * self.max_delta * torch.tanh(raw)
        if return_aux:
            # No training graph or per-forward diagnostics are retained on the module.
            return delta, {"valid_count": valid_count.expand(batch, -1, -1, -1).detach()}
        return delta
