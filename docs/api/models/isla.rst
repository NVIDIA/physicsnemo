ISLA
====

ISLA (Invariant Slice Attention) is an E(d)-equivariant (any spatial
dimension d >= 2), measure-aware slice-attention surrogate for steady
boundary-value problems. The boundary goes in as named
:class:`~physicsnemo.mesh.mesh.Mesh` objects, one per boundary condition, together
with the problem's data as *named fields of declared tensor rank*: global inputs
(a direction, a PDE parameter) and per-cell inputs declared per boundary
condition (an inlet's prescribed velocity, a loaded face's traction; a no-slip
wall declares none). An interior task's prediction points are read by their
positions alone: their distance to the boundary and its direction are computed
inside the model. Named fields of declared rank come out: scalars, vectors and
tensors up to rank 4, each covariant by its rank. The attention operates only
on invariants of the per-point vector set, and the frame is re-attached at the
heads, so the network is exactly covariant under rotations, reflections and
translations by construction. The full model is exactly covariant, up to
roundoff, when the boundary's canonical frame is unique and not close to a tie
(close to one, roundoff can change the interior task's drawn cells, and the
outputs then agree only approximately). At a tie (a symmetry that reverses a
principal axis: a mirror plane, a two-fold axis, a centre of symmetry; or equal
spread along two principal axes), a surface task stays exact, and an interior
task is exact only with ``n_boundary_tokens`` at least the cell count,
``tree_read_theta=0`` and a closed boundary, since its draw, read and
winding-number tree are laid out in that frame; otherwise it is approximate.

ISLA lives in ``physicsnemo.experimental.models.isla`` and may change between
releases. The class docstring below documents its inputs and outputs, and the
module docstring of ``physicsnemo.experimental.models.isla.model`` its design.
The field declarations follow :class:`~physicsnemo.mesh.fields.FieldSchema` and
:class:`~physicsnemo.mesh.fields.RankSpec` of the mesh library.

Citation
--------

The architecture and its guarantees are described in
Peter Sharpe, *ISLA: A Coordinate-Free, Measure-Weighted Transformer for
Boundary-Driven PDEs*, arXiv preprint, 2026. If you use ISLA, please cite:

.. code-block:: bibtex

    @misc{sharpe2026isla,
      author = {Sharpe, Peter},
      title  = {{ISLA}: A Coordinate-Free, Measure-Weighted Transformer
                for Boundary-Driven {PDE}s},
      year   = {2026},
      note   = {arXiv preprint}
    }

Quick start
-----------

A surface task predicts at the cells of named boundaries; the other boundaries
are read as conditions only. The class defaults are ``hidden=512``,
``n_layers=12`` and ``n_slices=128``:

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

An interior task is built with ``interior=True`` and called with
``prediction_points=`` (a point-cloud ``Mesh`` or an ``(N, n_spatial_dims)``
tensor) in place of ``prediction_boundaries``. Its ``boundary_meshes`` are the
boundaries at full resolution: the model draws ``n_boundary_tokens`` boundary
tokens from their cells inside the forward pass, and each prediction depends on
its own point and the boundary alone, so a large request can be answered in
chunks.

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
