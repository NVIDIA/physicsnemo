# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable PyTorch QoI evaluators for the RTE benchmarks.

These match KiT-RT's SNSolverHPC::IterPostprocessing() and are shared by the
training-time physics loss (losses.py) and the evaluation-time QoI metrics
(inference.py).

https://github.com/KiT-RT/kitrt_code/blob/d257b1a3c6fb3fa13d8a346adca5360a95101932/src/solvers/snsolver_hpc.cpp#L594

The evaluators are differentiable and final-time (T=1); ``sim_times`` is
accepted only for callsite uniformity with the time-dependent variants.
"""

from __future__ import annotations

import torch

__all__ = [
    "evaluate_lattice_qoi_torch",
    "evaluate_hohlraum_qoi_torch",
    "extract_geometry_params",
]


_HOHLRAUM_GEOMETRY_KEYS = ("ulr", "llr", "urr", "lrr", "hlr", "hrr", "cx", "cy")


def extract_geometry_params(sample) -> dict:
    """Extract hohlraum geometry parameters from a sample TensorDict.

    Reads the eight 0-D float32 tensors that the curator writes into
    ``mesh.global_data`` for hohlraum stores (``ulr, llr, urr, lrr, hlr,
    hrr, cx, cy``) and that :meth:`MeshDataReader.load` promotes to the
    TensorDict top level. Returns ``{}`` if any key is missing (e.g. on a
    lattice sample, which has no geometry parameters).
    """
    if sample is None:
        return {}
    try:
        if not all(k in sample for k in _HOHLRAUM_GEOMETRY_KEYS):
            return {}
    except TypeError:
        return {}

    out: dict = {}
    for k in _HOHLRAUM_GEOMETRY_KEYS:
        v = sample[k]
        if hasattr(v, "ndim") and v.ndim > 0:
            # Batched value (e.g. shape ``(B,)``): collapse to a single
            # scalar by picking the first entry. Geometry parameters are
            # static per simulation, so every batch element matches.
            v = v.reshape(-1)[0]
        out[k] = float(v.item() if hasattr(v, "item") else v)
    return out


def evaluate_lattice_qoi_torch(
    cell_centers: torch.Tensor,
    cell_areas: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    scalar_flux: torch.Tensor,
    sim_times: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Lattice absorption QoI, differentiable.

    When the leading dim of ``cell_centers`` is a (size-1) batch dim, the
    call recurses on the squeezed slot and re-adds the dim on the way out.

    Args:
        cell_centers: (N, 2) or (1, N, 2)
        cell_areas: (N,) or (1, N)
        sigma_t: (N,) or (1, N)
        sigma_s: (N,) or (1, N)
        scalar_flux: (T, N) or (1, T, N) — only T=1 is exercised
        sim_times: (T,) or (1, T) — accepted for callsite uniformity, unused

    Returns:
        ``{"cur_absorption": (T,) or (1, T)}``
    """
    if cell_centers.ndim == 3:
        if cell_centers.shape[0] != 1:
            raise NotImplementedError(
                "evaluate_lattice_qoi_torch only supports batch_size=1; "
                f"got batch={cell_centers.shape[0]}."
            )
        result = evaluate_lattice_qoi_torch(
            cell_centers[0],
            cell_areas[0],
            sigma_t[0],
            sigma_s[0],
            scalar_flux[0],
            sim_times[0] if sim_times.ndim == 2 else sim_times,
        )
        return {k: v.unsqueeze(0) for k, v in result.items()}

    if scalar_flux.ndim != 2:
        raise ValueError(f"Expected scalar_flux shape (T, N), got {scalar_flux.shape}")

    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    sigma_a = sigma_t - sigma_s

    xy_corrector = -3.5
    lbounds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]) + xy_corrector
    ubounds = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]) + xy_corrector

    in_absorption = torch.zeros_like(x, dtype=torch.bool)
    for k in range(5):
        for l in range(5):  # noqa: E741
            if (l + k) % 2 == 1:
                continue
            if (k == 2 and l == 2) or (k == 2 and l == 4):
                continue
            in_square = (
                (x >= lbounds[k])
                & (x <= ubounds[k])
                & (y >= lbounds[l])
                & (y <= ubounds[l])
            )
            in_absorption = in_absorption | in_square

    absorption_density = scalar_flux * sigma_a.unsqueeze(0) * cell_areas.unsqueeze(0)
    cur_absorption = torch.sum(
        absorption_density * in_absorption.unsqueeze(0).to(dtype=torch.float32),
        dim=1,
    )
    return {"cur_absorption": cur_absorption}


def evaluate_hohlraum_qoi_torch(
    cell_centers: torch.Tensor,
    cell_areas: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    scalar_flux: torch.Tensor,
    sim_times: torch.Tensor,
    geometry_params: dict[str, float],
) -> dict[str, torch.Tensor]:
    """Hohlraum per-region absorption QoI, differentiable.

    Three regions are returned: ``center`` (the capsule volume), ``vertical``
    (red wall strips on either x boundary), and ``horizontal`` (the top + bottom
    strips). The vertical-wall predicate uses ``pos_red_left_bottom`` for both
    sides — see the inline ``NOTE`` for why.

    When the leading dim of ``cell_centers`` is a (size-1) batch dim, the
    call recurses on the squeezed slot and re-adds the dim on the way out.

    Args:
        cell_centers: (N, 2) or (1, N, 2)
        cell_areas: (N,) or (1, N)
        sigma_t: (N,) or (1, N)
        sigma_s: (N,) or (1, N)
        scalar_flux: (T, N) or (1, T, N) — only T=1 is exercised
        sim_times: (T,) or (1, T) — accepted for callsite uniformity, unused
        geometry_params: dict with cx, cy, hlr, hrr, llr, ulr, lrr, urr

    Returns:
        Dict with ``cur_absorption_{center,vertical,horizontal}``.
    """
    if cell_centers.ndim == 3:
        if cell_centers.shape[0] != 1:
            raise NotImplementedError(
                "evaluate_hohlraum_qoi_torch only supports batch_size=1; "
                f"got batch={cell_centers.shape[0]}."
            )
        result = evaluate_hohlraum_qoi_torch(
            cell_centers[0],
            cell_areas[0],
            sigma_t[0],
            sigma_s[0],
            scalar_flux[0],
            sim_times[0] if sim_times.ndim == 2 else sim_times,
            geometry_params,
        )
        return {k: v.unsqueeze(0) for k, v in result.items()}

    if scalar_flux.ndim != 2:
        raise ValueError(f"Expected scalar_flux shape (T, N), got {scalar_flux.shape}")

    x = cell_centers[:, 0]
    y = cell_centers[:, 1]

    cx = geometry_params["cx"]
    cy = geometry_params["cy"]
    pos_red_left_border = geometry_params["hlr"]
    pos_red_right_border = geometry_params["hrr"]
    pos_red_left_bottom = geometry_params["llr"]
    pos_red_left_top = geometry_params["ulr"]
    pos_red_right_top = geometry_params["urr"]

    sigma_a = sigma_t - sigma_s

    in_center = (x > -0.2 + cx) & (x < 0.2 + cx) & (y > -0.4 + cy) & (y < 0.4 + cy)
    # NOTE: matches KiT-RT's behavior of using pos_red_left_bottom for both sides
    in_vertical = (
        (x < pos_red_left_border) & (y > pos_red_left_bottom) & (y < pos_red_left_top)
    ) | (
        (x > pos_red_right_border) & (y > pos_red_left_bottom) & (y < pos_red_right_top)
    )
    in_horizontal = (y > 0.6) | (y < -0.6)

    absorption_density = scalar_flux * sigma_a.unsqueeze(0) * cell_areas.unsqueeze(0)

    def _region_sum(mask: torch.Tensor) -> torch.Tensor:
        return torch.sum(
            absorption_density * mask.unsqueeze(0).to(dtype=torch.float32), dim=1
        )

    return {
        "cur_absorption_center": _region_sum(in_center),
        "cur_absorption_vertical": _region_sum(in_vertical),
        "cur_absorption_horizontal": _region_sum(in_horizontal),
    }
