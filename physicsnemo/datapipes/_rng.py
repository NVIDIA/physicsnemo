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

import numpy as np
import torch


def derive_seed(base_seed: int, *coords: int) -> int:
    """Deterministically mix a base seed with integer coordinates.

    Combines ``base_seed`` with arbitrary integer ``coords`` (typically
    ``epoch`` and sample ``index``) into a single well-mixed 64-bit seed
    using :class:`numpy.random.SeedSequence`.  The result depends only on
    the inputs, not on call order or thread, so per-sample RNG derived
    from it is reproducible and safe to compute concurrently.

    Parameters
    ----------
    base_seed : int
        Base seed (e.g. a generator's ``initial_seed()``).
    *coords : int
        Additional non-negative integer coordinates to fold in, such as
        ``(epoch, index)``.

    Returns
    -------
    int
        A deterministic 64-bit seed.
    """
    seq = np.random.SeedSequence([int(base_seed), *(int(c) for c in coords)])
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def spawn_generator(
    base_seed: int,
    *coords: int,
    device: torch.device | str = "cpu",
) -> torch.Generator:
    """Create a fresh :class:`torch.Generator` seeded from mixed coordinates.

    Returns an independent generator whose seed is
    :func:`derive_seed(base_seed, *coords) <derive_seed>`.  Because each
    call returns a new generator seeded purely from its inputs, draws are
    reproducible regardless of execution order and can be made
    concurrently from multiple threads without sharing mutable state.

    Parameters
    ----------
    base_seed : int
        Base seed (e.g. a generator's ``initial_seed()``).
    *coords : int
        Additional integer coordinates to fold in, such as ``(epoch, index)``.
    device : torch.device or str, default="cpu"
        Device the generator is created on.

    Returns
    -------
    torch.Generator
        A new generator seeded deterministically from the inputs.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(derive_seed(base_seed, *coords))
    return generator


def fork_generator(
    parent: torch.Generator,
    n: int,
) -> list[torch.Generator]:
    """Deterministically derive *n* child generators from *parent*.

    Child *i* is seeded with :func:`derive_seed(parent.initial_seed(), i)
    <derive_seed>`, so children are well-mixed and stable across runs.
    Unlike sequential ``base_seed + i`` seeding, nearby parent seeds (or
    forks at different depths of the pipeline tree) do not produce
    overlapping child streams.

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
    base_seed = parent.initial_seed()
    return [spawn_generator(base_seed, i, device=parent.device) for i in range(n)]
