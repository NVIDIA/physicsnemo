"""``torch.compile``-wrapped PhysicsNeMo-Mesh operations for benchmarking.

Each compiled function is a thin wrapper around the corresponding function in
:mod:`raw_ops`, with ``torch.compile`` applied.  See ``raw_ops`` for the
actual implementations and docstrings.
"""

import torch

from . import raw_ops

cell_normals = torch.compile(raw_ops.cell_normals)
gaussian_curvature = torch.compile(raw_ops.gaussian_curvature)
gradient = torch.compile(raw_ops.gradient)
subdivide = torch.compile(raw_ops.subdivide)
p2p_neighbors = torch.compile(raw_ops.p2p_neighbors)
c2c_neighbors = torch.compile(raw_ops.c2c_neighbors)
sample_points = torch.compile(raw_ops.sample_points)
sample_points_area_weighted = torch.compile(raw_ops.sample_points_area_weighted)
smooth = torch.compile(raw_ops.smooth)
transforms = torch.compile(raw_ops.transforms)
