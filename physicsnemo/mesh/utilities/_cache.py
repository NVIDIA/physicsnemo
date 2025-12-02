"""Cache utilities for TensorDict-based data storage.

Provides clean interface for storing and retrieving cached computed values
in nested TensorDict structures under the "_cache" key.
"""

import torch
from tensordict import TensorDict


def get_cached(data: TensorDict, key: str) -> torch.Tensor | None:
    """Get a cached value from a TensorDict.

    Args:
        data: TensorDict containing potentially cached data
        key: Name of the cached value (without "_cache" prefix)

    Returns:
        The cached tensor if it exists, None otherwise

    Example:
        >>> cached_areas = get_cached(mesh.cell_data, "areas")
        >>> if cached_areas is None:
        >>>     # Compute areas
        >>>     ...
    """
    return data.get(("_cache", key), None)


def set_cached(data: TensorDict, key: str, value: torch.Tensor) -> None:
    """Set a cached value in a TensorDict.

    Creates the "_cache" sub-TensorDict if it doesn't exist, then stores
    the value under ("_cache", key).

    Args:
        data: TensorDict to store cached value in
        key: Name of the cached value (without "_cache" prefix)
        value: Tensor to cache

    Example:
        >>> set_cached(mesh.cell_data, "areas", computed_areas)
    """
    if "_cache" not in data:
        data["_cache"] = TensorDict({}, batch_size=data.batch_size, device=data.device)
    data[("_cache", key)] = value
