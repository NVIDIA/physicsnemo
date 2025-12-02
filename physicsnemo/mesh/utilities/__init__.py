"""Utility functions for physicsnemo.mesh."""

from physicsnemo.mesh.utilities._cache import get_cached, set_cached
from physicsnemo.mesh.utilities._scatter_ops import scatter_aggregate

__all__ = [
    "get_cached",
    "set_cached",
    "scatter_aggregate",
]
