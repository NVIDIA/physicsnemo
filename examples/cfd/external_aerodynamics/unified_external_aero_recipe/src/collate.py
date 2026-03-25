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

"""Collate functions for converting datapipe output to model batch format.

The datapipe produces ``(TensorDict, metadata_dict)`` tuples.  The collate
function stacks these into batched tensors matching the GeoTransolver
forward signature::

    {
        "geometry":         (B, N, 3),   # point positions
        "local_embedding":  (B, N, 6),   # positions + surface normals
        "global_embedding": (B, 1, 3),   # freestream velocity (U_inf)
        "fields":           (B, N, C),   # concatenated target fields
    }
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


def surface_collate(samples: list[tuple[TensorDict, dict]]) -> dict[str, torch.Tensor]:
    """Stack datapipe samples into a model-ready batch.

    Each sample is a ``(data, metadata)`` tuple where ``data`` is a TensorDict
    with ``input/`` and ``output/`` groups produced by
    :class:`~physicsnemo.datapipes.transforms.mesh.RestructureTensorDict`.

    The ``input`` group must contain ``points``, ``normals``, and ``U_inf``.
    The ``output`` group must contain ``pressure`` (scalar) and ``wss`` (3-vector).

    Parameters
    ----------
    samples : list[tuple[TensorDict, dict]]
        List of ``(data, metadata)`` pairs from the datapipe.

    Returns
    -------
    dict[str, torch.Tensor]
        Batch dictionary with keys ``geometry``, ``global_embedding``,
        ``local_embedding``, and ``fields``.
    """
    points_list = []
    embedding_list = []
    velocity_list = []
    fields_list = []

    for data, _meta in samples:
        inp = data["input"]
        out = data["output"]

        pts = inp["points"]  # (N, 3)
        normals = inp["normals"]  # (N, 3)
        vel = inp["U_inf"]  # (1, 3) or (3,)

        pressure = out["pressure"]  # (N,) or (N, 1)
        wss = out["wss"]  # (N, 3)

        if pressure.ndim == 1:
            pressure = pressure.unsqueeze(-1)  # (N, 1)
        while vel.ndim < 2:
            vel = vel.unsqueeze(0)  # ensure (1, 3)

        target = torch.cat([pressure, wss], dim=-1)  # (N, 4)

        points_list.append(pts)
        embedding_list.append(torch.cat([pts, normals], dim=-1))  # (N, 6)
        velocity_list.append(vel)
        fields_list.append(target)

    stacked_points = torch.stack(points_list)
    return {
        "geometry": stacked_points,  # (B, N, 3)
        "local_embedding": torch.stack(embedding_list),  # (B, N, 6)
        "local_positions": stacked_points,  # (B, N, 3) for local feature builder
        "global_embedding": torch.stack(velocity_list),  # (B, 1, 3)
        "fields": torch.stack(fields_list),  # (B, N, 4)
    }
