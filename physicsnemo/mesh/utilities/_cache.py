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

"""Cache utilities for TensorDict-based data storage.

Provides clean interface for storing and retrieving cached computed values
in nested TensorDict structures under the "_cache" key.
"""

import torch
from tensordict import TensorDict

CACHE_KEY = "_cache"


def get_cached(data: TensorDict, key: str) -> torch.Tensor | None:
    """Get a cached value from a TensorDict.

    Parameters
    ----------
    data : TensorDict
        TensorDict containing potentially cached data.
    key : str
        Name of the cached value (without "_cache" prefix).

    Returns
    -------
    torch.Tensor or None
        The cached tensor if it exists, None otherwise.

    Examples
    --------
    >>> cached_areas = get_cached(mesh.cell_data, "areas")  # doctest: +SKIP
    >>> if cached_areas is None:  # doctest: +SKIP
    ...     # Compute areas
    ...     pass  # doctest: +SKIP
    """
    return data.get((CACHE_KEY, key), None)


def set_cached(data: TensorDict, key: str, value: torch.Tensor) -> None:
    """Set a cached value in a TensorDict.

    Creates the "_cache" sub-TensorDict if it doesn't exist, then stores
    the value under ("_cache", key).

    Parameters
    ----------
    data : TensorDict
        TensorDict to store cached value in.
    key : str
        Name of the cached value (without "_cache" prefix).
    value : torch.Tensor
        Tensor to cache.

    Examples
    --------
    >>> set_cached(mesh.cell_data, "areas", computed_areas)  # doctest: +SKIP
    """
    if CACHE_KEY not in data:
        data[CACHE_KEY] = TensorDict({}, batch_size=data.batch_size, device=data.device)
    data[(CACHE_KEY, key)] = value
