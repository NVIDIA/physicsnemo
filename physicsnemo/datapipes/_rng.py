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

"""
Internal RNG utilities for deterministic generator forking.

Used by :class:`DataLoader` and :class:`MeshDataset` to derive
independent per-component generators from a single master seed.
"""

from __future__ import annotations

import torch


def fork_generator(
    parent: torch.Generator,
    n: int,
) -> list[torch.Generator]:
    """Deterministically derive *n* child generators from *parent*.

    Each child is seeded with ``parent.initial_seed() + i + 1``, so
    children are independent of each other and stable across runs.

    Parameters
    ----------
    parent : torch.Generator
        Master generator whose ``initial_seed()`` is used as the base.
    n : int
        Number of child generators to create.

    Returns
    -------
    list[torch.Generator]
        *n* independent generators on the same device as *parent*.
    """

    # I miss JAX ...
    # https://docs.jax.dev/en/latest/jax.random.html

    base_seed = parent.initial_seed()
    children: list[torch.Generator] = []
    for i in range(n):
        g = torch.Generator(device=parent.device)
        g.manual_seed(base_seed + i + 1)
        children.append(g)
    return children
