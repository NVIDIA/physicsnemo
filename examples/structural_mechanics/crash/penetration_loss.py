# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Optional physical-coordinate endpoint penalty for crash autoregression."""

import math

import torch

from physicsnemo.nn.functional.neighbors.surface_penetration import (
    surface_penetration_loss,
)


def penetration_weight(
    epoch: int, *, max_weight: float, warmup_epochs: int, ramp_epochs: int
) -> float:
    """Zero-based epoch: warmup off, then a linear ramp, then fixed weight."""
    if not math.isfinite(max_weight) or max_weight < 0:
        raise ValueError("max_weight must be finite and nonnegative")
    if epoch < 0 or warmup_epochs < 0 or ramp_epochs < 0:
        raise ValueError("epoch counts must be nonnegative")
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return max_weight
    return max_weight * min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)


def rollout_penetration_loss(trajectory, graph, data_stats, cfg):
    """Average over every predicted step; no detach within the BPTT window."""
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3 or trajectory.shape[1] < 1:
        raise ValueError("penetration loss requires a position trajectory [N,T,3]")
    required = ("contact_faces", "shell_thickness", "penetration_reference_positions")
    if any(getattr(graph, name, None) is None for name in required):
        raise ValueError(
            "penetration loss requires surface topology, thickness and physical initial positions"
        )
    # The datapipe normalization is per coordinate; reverse it explicitly.
    with torch.autocast(device_type=trajectory.device.type, enabled=False):
        physical = trajectory.float() * data_stats["node"]["pos_std"].reshape(1, 1, 3)
        physical = physical + data_stats["node"]["pos_mean"].reshape(1, 1, 3)
        outputs = [
            surface_penetration_loss(
                physical[:, step],
                graph.contact_faces,
                graph.shell_thickness,
                reference_positions=graph.penetration_reference_positions,
                tolerance=float(cfg["tolerance_mm"]),
                length_scale=float(cfg["length_scale_mm"]),
                excluded_pairs=getattr(graph, "contact_surface_exclusions", None),
                batch=getattr(graph, "batch", None),
                implementation=str(cfg.get("implementation", "warp")),
                pair_chunk_size=int(cfg.get("pair_chunk_size", 65_536)),
            )
            for step in range(trajectory.shape[1])
        ]
    return torch.stack([o.loss for o in outputs]).mean(), {
        "penetration_active_nodes": torch.stack(
            [o.active_nodes.float() for o in outputs]
        ).mean(),
        "penetration_max_excess_mm": torch.stack([o.max_excess for o in outputs]).max(),
    }
