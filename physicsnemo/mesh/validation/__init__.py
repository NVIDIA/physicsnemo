"""Mesh validation, quality metrics, and statistics.

This module provides tools for validating mesh integrity, computing quality
metrics, and generating mesh statistics.
"""

from physicsnemo.mesh.validation.validate import validate_mesh
from physicsnemo.mesh.validation.quality import compute_quality_metrics
from physicsnemo.mesh.validation.statistics import compute_mesh_statistics

__all__ = [
    "validate_mesh",
    "compute_quality_metrics",
    "compute_mesh_statistics",
]
