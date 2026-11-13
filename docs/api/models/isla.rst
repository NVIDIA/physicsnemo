ISLA (Experimental)
===================

ISLA (Invariant Slice Attention) is a measure-aware slice-attention surrogate
for steady boundary-value problems. It is :math:`E(d)`-equivariant (covariant
under rotations, reflections, and translations) in any spatial dimension
:math:`d \geq 2`.

Each boundary comes in as a named :class:`~physicsnemo.mesh.mesh.Mesh`, one per
boundary condition. The problem data are *named fields of declared tensor
rank*, specified with the mesh library's
:class:`~physicsnemo.mesh.fields.FieldSchema` and
:class:`~physicsnemo.mesh.fields.RankSpec`:

- Global inputs, such as a direction or a partial differential equation (PDE)
  parameter.
- Per-cell inputs of a boundary condition, such as an inlet's prescribed
  velocity or a loaded face's traction. A no-slip wall declares none.

A surface task predicts at the cells of named boundaries. An interior task
predicts at points in the domain. Its only input for each point is the point's
position: the model itself computes the point's distance and direction to the
boundary. Both tasks output named fields of declared rank: scalars, vectors,
and tensors up to rank 4.

Each output field is covariant according to its rank. The attention uses only
invariants of each cell's and each point's vectors, such as its position, its
normal, and the global directions. The output heads rebuild each field from
those vectors, so the network is exactly covariant by construction.

The interior task samples its boundary tokens (a subset of the boundary's
cells) and builds its spatial trees in the boundary's canonical frame. That
frame's axes are the principal axes of the cells' measure-weighted covariance,
each signed by its third moment. A tie is a boundary whose canonical frame is
not unique. It has either a symmetry that reverses a principal axis (a mirror
plane, a twofold axis, or a center of symmetry) or equal spread along two
principal axes.

Away from a tie, the full model is exactly covariant, up to roundoff. Near a
tie, roundoff can change the cells that an interior task samples, so its
outputs agree only approximately. At a tie, a surface task stays exact. An
interior task is exact at a tie only when all of the following hold:

- ``n_boundary_tokens`` is at least the total cell count of the boundaries.
- ``tree_read_theta`` is ``0``.
- The boundary is closed.

Otherwise, the interior task is approximate at a tie.

ISLA is in ``physicsnemo.experimental.models.isla`` and can change between
releases. The class docstring below documents its inputs and outputs, and the
module docstring of ``physicsnemo.experimental.models.isla.model`` documents
its design.

Citation
--------

The architecture and its guarantees are described in Peter Sharpe, *ISLA: A
Coordinate-Free, Measure-Weighted Transformer for Boundary-Driven PDEs*, arXiv
preprint `arXiv:2611.17141 <https://arxiv.org/abs/2611.17141>`__, 2026. If you
use or extend ISLA, please cite:

.. code-block:: bibtex

    @misc{sharpe2026isla,
      author        = {Sharpe, Peter},
      title         = {{ISLA}: A Coordinate-Free, Measure-Weighted Transformer
                       for Boundary-Driven {PDE}s},
      year          = {2026},
      howpublished  = {arXiv preprint arXiv:2611.17141},
      eprint        = {2611.17141},
      archivePrefix = {arXiv},
      primaryClass  = {cs.LG},
      url           = {https://arxiv.org/abs/2611.17141}
    }

Quick Start
-----------

The first example is a surface task. It predicts at the body's cells and uses
the floor only as a boundary condition. The class defaults are ``hidden=512``,
``n_layers=12``, and ``n_slices=128``. The example uses a smaller model, a
sphere as the body, and a plane below it as the floor:

>>> import torch
>>> from physicsnemo.experimental.models.isla import ISLA
>>> from physicsnemo.mesh.primitives.surfaces import plane, sphere_icosahedral
>>> body = sphere_icosahedral.load(subdivisions=1)  # Mesh[2, 3], 80 triangles
>>> floor = plane.load(size=4.0, subdivisions=4).translate([0.0, 0.0, -1.5])
>>> model = ISLA(
...     output_schema={"pressure": {"rank": 0}, "wss": {"rank": 1}},  # a scalar and a vector field
...     boundary_schemas={"body": {}, "floor": {}},  # conditions without per-cell fields
...     global_schema={"U_inf_dir": {"rank": 1}},  # one global direction
...     hidden=32,
...     n_layers=2,
...     n_slices=8,
... )
>>> out = model(
...     boundary_meshes={"body": body, "floor": floor},
...     prediction_boundaries=["body"],  # the outputs live on the body
...     global_data={"U_inf_dir": torch.tensor([1.0, 0.0, 0.0])},
... )
>>> out.point_data["pressure"].shape, out.point_data["wss"].shape  # one row per body cell
(torch.Size([80]), torch.Size([80, 3]))

For an interior task, set ``interior=True`` and pass ``prediction_points``, a
point-cloud ``Mesh`` or an ``(N, n_spatial_dims)`` tensor, instead of
``prediction_boundaries``. Pass the boundaries at full resolution: the forward
pass samples up to ``n_boundary_tokens`` of their cells as boundary tokens. Each
prediction depends only on its own point and the boundary, so ``forward``
runs a large request in chunks of points (``query_chunk_pairs``), and
:meth:`~physicsnemo.experimental.models.isla.ISLA.query_chunks` splits a
training step the same way. This example predicts a pressure and a velocity at
points around the same body:

>>> model = ISLA(
...     output_schema={"pressure": {"rank": 0}, "velocity": {"rank": 1}},
...     boundary_schemas={"body": {}, "floor": {}},
...     global_schema={"U_inf_dir": {"rank": 1}},
...     interior=True,
...     hidden=32,
...     n_layers=2,
...     n_slices=8,
... )
>>> points = 1.25 * torch.nn.functional.normalize(torch.randn(100, 3), dim=-1)  # (N, 3), in the fluid around the body
>>> out = model(
...     boundary_meshes={"body": body, "floor": floor},
...     prediction_points=points,
...     global_data={"U_inf_dir": torch.tensor([1.0, 0.0, 0.0])},
... )
>>> out.point_data["pressure"].shape, out.point_data["velocity"].shape
(torch.Size([100]), torch.Size([100, 3]))

The `unified external aerodynamics recipe
<https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/external_aerodynamics/unified_external_aero_recipe#isla>`_
trains and evaluates ISLA on DrivAerML with three configurations:

- ``isla_surface``: a surface task on the vehicle.
- ``isla_volume``: an interior task.
- ``isla_volume_joint``: an interior task that also predicts the wall
  pressure and wall shear stress at points just off the vehicle.

Its README documents how the dataset maps to the model's inputs.

.. autoclass:: physicsnemo.experimental.models.isla.ISLA
    :show-inheritance:
    :members:
    :exclude-members: forward

.. automodule:: physicsnemo.experimental.models.isla.model
    :no-members:

.. autoclass:: physicsnemo.experimental.models.isla.QueryChunks
    :members:
    :special-members: __call__

.. autofunction:: physicsnemo.experimental.models.isla.case_scales
