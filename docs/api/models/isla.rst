ISLA
====

ISLA (Invariant Slice Attention) is a measure-aware slice-attention surrogate
for steady boundary-value problems. It is :math:`E(d)`-equivariant for any
spatial dimension :math:`d \geq 2`. Provide each boundary as a named
:class:`~physicsnemo.mesh.mesh.Mesh` object, one for each boundary condition.
Provide the problem data as *named fields of declared tensor rank*:

- Global inputs include a direction or a partial differential equation (PDE)
  parameter.
- Inputs for each cell and boundary condition include an inlet's prescribed
  velocity or a loaded face's traction. A no-slip wall declares no inputs.

For an interior task, the model reads prediction points using only their
positions. The model internally computes each point's distance and direction
relative to the boundary. The model outputs named fields of declared rank,
including scalars, vectors, and tensors up to rank 4. Each field is covariant
according to its rank. The attention uses only invariants of the vector set
for each point. The heads reattach the frame, so the network is exactly
covariant under rotations, reflections, and translations by construction. The
full model is exactly covariant, up to roundoff, when the boundary's canonical
frame is unique and not close to a tie. Near a tie, roundoff can change the
cells drawn for an interior task. In that case, the outputs agree only
approximately. At a tie, a surface task remains exact.
A tie occurs for a symmetry that reverses a principal axis, such as a mirror
plane, a twofold axis, or a center of symmetry. Equal spread along two
principal axes also creates a tie. An interior task is exact at a tie only
when all the following conditions hold:

- ``n_boundary_tokens`` is at least the cell count.
- ``tree_read_theta`` is set to ``0``.
- The boundary is closed.

These conditions are necessary because the model lays out the interior task's
draw, read, and winding-number tree in the canonical frame. Without all three
conditions, the interior task is approximate.

ISLA is available in ``physicsnemo.experimental.models.isla`` and can change
between releases. The class docstring below documents its inputs and outputs.
The module docstring for ``physicsnemo.experimental.models.isla.model``
documents its design.
The field declarations use :class:`~physicsnemo.mesh.fields.FieldSchema` and
:class:`~physicsnemo.mesh.fields.RankSpec` from the mesh library.

Citation
--------

Peter Sharpe describes the architecture and its guarantees in *ISLA: A
Coordinate-Free, Measure-Weighted Transformer for Boundary-Driven PDEs*, an
arXiv preprint from 2026. If you use or extend ISLA, cite:

.. code-block:: bibtex

    @misc{sharpe2026isla,
      author = {Sharpe, Peter},
      title  = {{ISLA}: A Coordinate-Free, Measure-Weighted Transformer
                for Boundary-Driven {PDE}s},
      year   = {2026},
      note   = {arXiv preprint}
    }

Quick Start
-----------

A surface task predicts at the cells of named boundaries. The model reads
other boundaries only as conditions. The class defaults are ``hidden=512``,
``n_layers=12``, and ``n_slices=128``:

.. code-block:: python

    model = ISLA(
        output_schema={"pressure": {"rank": 0}, "wss": {"rank": 1}},  # a scalar and a vector field
        boundary_schemas={"vehicle": {}, "floor": {}, "inlet": {}},  # conditions without per-cell fields
        global_schema={"U_inf_dir": {"rank": 1}},  # one global direction
    )
    out = model(
        boundary_meshes={"vehicle": surface, "floor": floor, "inlet": inlet},  # Mesh[2, 3] each
        prediction_boundaries=["vehicle"],          # the outputs live on the vehicle
        global_data={"U_inf_dir": direction},       # (3,)
    )
    out.point_data["pressure"]                      # (N,), one per vehicle cell
    out.point_data["wss"]                           # (N, 3)

To build an interior task, set ``interior=True`` and provide
``prediction_points`` instead of ``prediction_boundaries``. Set
``prediction_points`` to a point-cloud ``Mesh`` or an
``(N, n_spatial_dims)`` tensor. Provide ``boundary_meshes`` at full resolution.
During the forward pass, the model draws ``n_boundary_tokens`` boundary tokens
from their cells. Each prediction depends only on its point and the boundary,
so you can process a large request in chunks.

The unified external-aerodynamics recipe
(``examples/cfd/external_aerodynamics/unified_external_aero_recipe``) trains
and evaluates ISLA on DrivAerML with the configurations ``isla_surface`` (a
surface task) and ``isla_volume`` (an interior task). Its README documents
the mapping from the dataset to the model's inputs.

.. autoclass:: physicsnemo.experimental.models.isla.model.ISLA
    :show-inheritance:
    :members:
    :exclude-members: forward

.. automodule:: physicsnemo.experimental.models.isla.model
    :no-members:

.. autoclass:: physicsnemo.experimental.models.isla.model.QueryChunks
    :members:
    :special-members: __call__

.. autofunction:: physicsnemo.experimental.models.isla.case_scales
