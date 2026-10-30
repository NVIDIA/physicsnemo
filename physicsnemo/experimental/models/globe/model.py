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

from dataclasses import dataclass
from math import prod
from typing import Any, Literal, Sequence

import torch
import torch.nn as nn
from jaxtyping import Float
from tensordict import PackLayout, TensorDict
from torch.profiler import record_function

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.globe.field_kernel import (
    MultiscaleKernel,
    _output_layout,
    _require_supported,
    _target_node_centroids,
)
from physicsnemo.mesh import FieldSchema, FieldSchemaLike, Mesh
from physicsnemo.mesh.calculus.measure import cell_measures
from physicsnemo.mesh.spatial.cluster_tree import (
    ClusterTree,
    DualInteractionPlan,
)
from physicsnemo.utils.logging import PythonLogger

logger = PythonLogger("globe.model")


def _column_runs(layout: PackLayout) -> tuple[tuple[int, int, int], ...]:
    r"""Maximal column ranges of a packed layout whose fields share one shape.

    An operation defined per field then runs once per range.

    Parameters
    ----------
    layout : PackLayout
        Layout of scalar and vector fields.

    Returns
    -------
    tuple[tuple[int, int, int], ...]
        ``(start, stop, n_dims)`` column ranges, ``n_dims`` being 0 for scalar
        fields and the vector length otherwise.
    """
    runs: list[tuple[int, int, int]] = []
    for shape, offset in zip(layout.shapes, layout.offsets):
        n_dims = shape[0] if shape else 0
        stop = offset + prod(shape)
        if runs and runs[-1][2] == n_dims and runs[-1][1] == offset:
            runs[-1] = (runs[-1][0], stop, n_dims)
        else:
            runs.append((offset, stop, n_dims))
    return tuple(runs)

@dataclass
class MetaData(ModelMetaData):
    jit: bool = False  # Refers to torch.compile compatibility - this is compatible.
    cuda_graphs: bool = False  # Computational graph changes depending on inputs due to tree traversals
    amp: bool = True
    torch_fx: bool = False
    onnx: bool = False


class GLOBE(Module):
    r"""Green's-function-Like Operator for Boundary Element PDEs.

    GLOBE is a neural surrogate architecture for boundary-driven elliptic PDEs
    that combines learnable Green's-function-like kernels with equivariant ML.
    The model represents solutions as superpositions of kernel evaluations from
    boundary faces to target points, with communication hyperlayers enabling
    boundary-to-boundary information propagation before final interior
    evaluation.

    The architecture is designed to satisfy fundamental physical requirements:

    - Translation-, rotation-, and parity-equivariant through relative positions
      and local basis reprojection
    - Discretization-invariant via area-weighted boundary integrals
    - Units-invariant through rigorous nondimensionalization
    - Global receptive field through all-to-all boundary-to-target evaluation

    Architecture overview (see paper Section 3):

    1. Communication hyperlayers propagate latent information between boundary
       condition partitions (Section 3.4)
    2. Each hyperlayer uses multiscale kernels operating at different reference
       length scales (Section 3.3)
    3. Final hyperlayer evaluates fields at user-specified query points
    4. Learnable per-field calibration transforms applied to outputs

    For more details, see the paper: https://arxiv.org/abs/2511.15856

    Parameters
    ----------
    n_spatial_dims : int
        Number of spatial dimensions (2 or 3).
    output_schema : FieldSchemaLike
        Field schema declaring the output fields, e.g.
        ``{"pressure": {"rank": 0}, "velocity": {"rank": 1}}`` (see
        :class:`~physicsnemo.mesh.FieldSchema`). GLOBE implements true
        scalars (rank 0) and vectors (rank 1); any other declaration, in this
        or the other schemas, raises :class:`NotImplementedError`. Derive from
        data via :meth:`FieldSchema.from_tensordict`.
    boundary_source_schemas : dict[str, FieldSchemaLike]
        Mapping of boundary condition type names to field schemas
        describing the per-face source features for each BC type. The keys
        implicitly define the set of boundary condition names. The face normal
        vector is automatically added, so don't include it.
    reference_length_names : Sequence[str]
        Sequence of identifiers for reference length scales (e.g.,
        ``["viscous_length", "chord_length"]``). Each creates a separate kernel
        branch in the multiscale composition.
    reference_area : float
        Scalar used to nondimensionalize face areas. Typically a characteristic
        area of the problem (e.g., chord^2 for airfoils).
    global_schema : FieldSchemaLike or None, optional
        Field schema for global conditioning features. Defaults to empty
        (no global conditioning).
    n_communication_hyperlayers : int, optional, default=2
        Number of boundary-to-boundary communication layers before final
        evaluation.
    n_latent_scalars : int, optional, default=12
        Number of scalar latent channels propagated between hyperlayers.
    n_latent_vectors : int, optional, default=6
        Number of vector latent channels propagated between hyperlayers.
    smoothing_radius : float, optional, default=1e-8
        Small value for numerical stability in magnitude computations.
    hidden_layer_sizes : Sequence[int] | None, optional, default=None
        Hidden layer sizes for kernel neural networks. If ``None``, defaults to
        ``[64, 64, 64]``.
    n_spherical_harmonics : int, optional, default=4
        Number of Legendre polynomial terms used for angle-dependent features in
        kernel functions.
    theta : float, optional, default=1.5
        Barnes-Hut opening angle controlling the near/far-field split in the
        dual-tree traversal. The criterion is :math:`(D_T + D_S) / r < \theta`,
        where :math:`D_T` and :math:`D_S` are AABB diagonals and :math:`r` is
        the minimum inter-AABB distance. Larger values approximate more
        aggressively; ``0`` forces all interactions to be exact (no far-field
        approximation). A trained model adapts to its ``theta``, so evaluate it
        with the value it was trained with.
    leaf_size : int, optional, default=1
        Maximum number of source points per leaf node in the cluster tree.
        Larger values produce shallower trees (fewer traversal iterations) at
        the cost of more exact near-field interactions per leaf hit.
    expand_far_targets : bool, optional, default=True
        If ``True``, far-field target nodes are expanded to individual points,
        converting ``(far, far)`` pairs into ``(near, far)`` pairs. This
        eliminates the target-side approximation, which makes the output
        piecewise constant over each far target node, at the cost of more
        kernel evaluations.
    use_gradient_checkpointing : bool, optional, default=True
        If ``True``, applies ``torch.utils.checkpoint.checkpoint`` to each
        kernel evaluation during training, trading compute for memory. See
        :class:`~physicsnemo.experimental.models.globe.field_kernel.Kernel` for
        details.
    tree_build_device : torch.device or str or None, optional, default=None
        Device on which to build cluster trees and run the dual-tree Barnes-Hut
        traversal in :meth:`_build_trees_and_plans` and
        :meth:`_build_prediction_plans`.  Trees and plans are transferred back
        to the input's device at function exit (a no-op when the devices
        match).  ``None`` (default) uses the input's device.  ``"cpu"`` can be
        faster for small problems (~a few thousand boundary cells) where CUDA
        launch latency and ``cudaStreamSynchronize`` round-trips dominate the
        real work in tree traversal; for large problems (hundreds of thousands
        of cells) the GPU's parallelism wins and the input's device is the
        right choice.

    Forward
    -------
    prediction_points : Float[torch.Tensor, "n_points n_dims"]
        Target points for field evaluation of shape :math:`(N_{points}, D)`.
    boundary_meshes : dict[str, Mesh["n-1", "n"]]
        Dictionary mapping boundary condition type names to
        :class:`~physicsnemo.mesh.Mesh` objects. Keys must be a subset of the
        model's boundary condition names (from ``boundary_source_schemas``).
    reference_lengths : dict[str, torch.Tensor]
        Dictionary mapping reference length names to scalar tensors.
    global_data : TensorDict or None, optional, default=None
        Nondimensional conditioning features. Leaf keys and ranks must match
        ``global_schema``. Passed through to the output Mesh.

    Outputs
    -------
    Mesh[0, "n"]
        A point-cloud :class:`~physicsnemo.mesh.Mesh` (0-dimensional manifold)
        whose ``.points`` attribute equals the input ``prediction_points``. The
        predicted fields are in ``.point_data``, keyed by the names from
        ``output_schema``. Scalar fields have shape :math:`(N_{points},)`,
        vector fields have shape :math:`(N_{points}, D)`. Cells are empty (shape
        ``(0, 1)``). ``global_data`` is passed through from the input.

    Notes
    -----
    - ``kernel_layers`` is a :class:`~torch.nn.ModuleList` of communication
      hyperlayers, each containing a :class:`~torch.nn.ModuleDict` mapping BC
      type names to
      :class:`~physicsnemo.experimental.models.globe.field_kernel.MultiscaleKernel`
      instances.
    - ``final_field_transforms`` is a :class:`~torch.nn.ModuleList` of per-field
      linear calibration layers, ordered alphabetically by field name.
    - Cell areas are automatically normalized by ``reference_area`` to preserve
      discretization-invariance.  If a boundary mesh carries per-cell
      measure weights (see :mod:`physicsnemo.mesh.calculus.measure`), the
      effective cell measures ``cell_areas * measure_weights`` are used --
      so e.g. boundary integrals over a subsampled surface are unbiased
      estimates of the full-surface ones.
    - The cell normal vector is automatically added to source data for each
      mesh.
    - The ``Mesh["n-1", "n"]`` type annotations assume the PDE domain fills the
      full ambient space (domain manifold dim = spatial dim), so boundary meshes
      are codimension-1 in the ambient space. For a PDE on a ``d``-dimensional
      manifold embedded in ``n``-dimensional space (``d < n``), the boundary
      type would be ``Mesh[d-1, n]`` instead.

    Examples
    --------
    >>> model = GLOBE(
    ...     n_spatial_dims=3,
    ...     output_schema={"pressure": {"rank": 0}, "velocity": {"rank": 1}},
    ...     boundary_source_schemas={"no_slip": {}, "freestream": {}},
    ...     reference_length_names=["delta_FS", "chord"],
    ...     reference_area=1.0,
    ... )
    >>> result = model(
    ...     prediction_points=torch.randn(100, 3),
    ...     boundary_meshes={"no_slip": wing_mesh, "freestream": freestream_mesh},
    ...     reference_lengths={"delta_FS": torch.tensor(0.01), "chord": torch.tensor(1.0)},
    ... )
    """

    reference_area: torch.Tensor

    def __init__(
        self,
        n_spatial_dims: int,
        output_schema: FieldSchemaLike,
        boundary_source_schemas: dict[str, FieldSchemaLike],
        reference_length_names: Sequence[str],
        reference_area: float,
        global_schema: FieldSchemaLike | None = None,
        n_communication_hyperlayers: int = 2,
        n_latent_scalars: int = 12,
        n_latent_vectors: int = 6,
        smoothing_radius: float = 1e-8,
        hidden_layer_sizes: Sequence[int] | None = None,
        n_spherical_harmonics: int = 4,
        theta: float = 1.5,
        leaf_size: int = 1,
        network_type: Literal["pade", "mlp"] = "pade",
        self_regularization_beta: float | None = None,
        latent_compression_scale: float | None = None,
        expand_far_targets: bool = True,
        use_gradient_checkpointing: bool = True,
        tree_build_device: torch.device | str | None = None,
    ):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64, 64, 64]

        ### Parse the field declarations (``Module`` has already recorded the
        ### arguments as given; a FieldSchema serializes to JSON as it stands).
        ### The kernels refuse what they do not implement as well; checking
        ### here names the declaration as the caller wrote it.
        output_schema = FieldSchema.parse(output_schema, label="output_schema")
        boundary_source_schemas = {
            bc_name: FieldSchema.parse(
                spec, label=f"boundary_source_schemas[{bc_name!r}]"
            )
            for bc_name, spec in boundary_source_schemas.items()
        }
        global_schema = FieldSchema.parse(
            {} if global_schema is None else global_schema, label="global_schema"
        )
        _require_supported(output_schema, label="output_schema")
        for bc_name, schema in boundary_source_schemas.items():
            _require_supported(schema, label=f"boundary_source_schemas[{bc_name!r}]")
        _require_supported(global_schema, label="global_schema")

        boundary_condition_names = list(boundary_source_schemas.keys())

        ### Input validation (eager mode only).  Only validate parameters whose
        ### use sites are inside `GLOBE` itself; parameters plumbed through to a
        ### deeper owner (e.g. `self_regularization_beta` -> `Pade`) are validated
        ### at that owner's constructor to avoid drift between the two checks.
        for bc_name in boundary_condition_names:
            if "." in bc_name:
                raise ValueError(
                    f"In `boundary_source_schemas`, got {bc_name=!r};\n"
                    "BC names must not contain `.` for TensorDict compatibility."
                )
        if smoothing_radius <= 0:
            raise ValueError(
                f"smoothing_radius must be positive, got {smoothing_radius=!r}"
            )
        if theta < 0:
            raise ValueError(
                f"theta must be non-negative (0 means exact summation), "
                f"got {theta=!r}"
            )
        if leaf_size < 1:
            raise ValueError(
                f"leaf_size must be at least 1, got {leaf_size=!r}"
            )
        if reference_area <= 0:
            raise ValueError(
                f"reference_area must be positive, got {reference_area=!r}"
            )
        if latent_compression_scale is not None and latent_compression_scale <= 0:
            raise ValueError(
                f"latent_compression_scale must be positive (use None to disable "
                f"compression), got {latent_compression_scale=!r}"
            )

        super().__init__(meta=MetaData())

        self.n_spatial_dims = n_spatial_dims
        self.output_schema = output_schema
        self.boundary_condition_names = boundary_condition_names
        self.boundary_source_schemas = boundary_source_schemas
        self.reference_length_names = reference_length_names
        self.register_buffer("reference_area", torch.tensor(reference_area))
        self.global_schema = global_schema
        self.n_communication_hyperlayers = n_communication_hyperlayers
        self.n_latent_scalars = n_latent_scalars
        self.n_latent_vectors = n_latent_vectors
        self.smoothing_radius = smoothing_radius
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_spherical_harmonics = n_spherical_harmonics
        self.theta = theta
        self.leaf_size = leaf_size
        self.latent_compression_scale = latent_compression_scale
        self.network_type = network_type
        self.self_regularization_beta = self_regularization_beta
        self.expand_far_targets = expand_far_targets
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.tree_build_device: torch.device | None = (
            torch.device(tree_build_device) if tree_build_device is not None else None
        )

        ### Build the intermediate output schema for communication
        # hyperlayers. Only the final hyperlayer emits output_schema.
        intermediate_schema: FieldSchemaLike = {
            **{f"strengths.{name}": {"rank": 0} for name in reference_length_names},
            **{f"latent.scalars.{i}": {"rank": 0} for i in range(n_latent_scalars)},
            **{f"latent.vectors.{i}": {"rank": 1} for i in range(n_latent_vectors)},
        }

        kernel_layers = []

        for layer_idx in range(self.n_communication_hyperlayers + 1):
            is_first_hyperlayer = layer_idx == 0
            is_last_hyperlayer = layer_idx == self.n_communication_hyperlayers

            layer = nn.ModuleDict(
                {
                    bc_type: MultiscaleKernel(
                        n_spatial_dims=n_spatial_dims,
                        output_schema=(
                            output_schema
                            if is_last_hyperlayer
                            else intermediate_schema
                        ),
                        reference_length_names=reference_length_names,
                        source_schema=self._build_source_schema(
                            bc_source_schema=boundary_source_schemas[bc_type],
                            include_latents=not is_first_hyperlayer,
                        ),
                        global_schema=global_schema,
                        smoothing_radius=smoothing_radius,
                        hidden_layer_sizes=hidden_layer_sizes,
                        n_spherical_harmonics=n_spherical_harmonics,
                        leaf_size=leaf_size,
                        network_type=network_type,
                        self_regularization_beta=self_regularization_beta,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                    )
                    for bc_type in boundary_condition_names
                }
            )
            kernel_layers.append(layer)

        self.kernel_layers = nn.ModuleList(kernel_layers)

        ### Packed hyperlayer outputs.  Each kernel returns its fields in one
        ### tensor (see ``MultiscaleKernel.forward_packed``); these layouts
        ### name its columns with nested keys, so that unpacking a block of
        ### rows gives a boundary mesh's new ``cell_data`` (or the output
        ### mesh's ``point_data``) as views, and compression runs once per
        ### range of like columns instead of once per field.
        self._intermediate_layout = _output_layout(
            FieldSchema.parse(intermediate_schema, label="intermediate_schema"),
            n_spatial_dims,
            nested=True,
        )
        self._intermediate_runs = _column_runs(self._intermediate_layout)
        self._output_layout = _output_layout(output_schema, n_spatial_dims, nested=True)

        ### Per-field learnable affine calibration (y = a*x + b). Bias is only
        # applied to scalar fields; adding bias to vector fields would break
        # rotational equivariance. Uses ModuleList (not ModuleDict) to support
        # output field names containing dots from nested schemas.
        flat_output_ranks = output_schema.ranks
        self._output_field_order = sorted(flat_output_ranks.keys())
        self.final_field_transforms = nn.ModuleList(
            [
                nn.Linear(
                    in_features=1,
                    out_features=1,
                    bias=(flat_output_ranks[name] == 0),
                )
                for name in self._output_field_order
            ]
        )

    def _build_source_schema(
        self,
        bc_source_schema: FieldSchemaLike,
        include_latents: bool,
    ) -> FieldSchemaLike:
        """Build the full source schema for a specific (layer, bc_type) kernel.

        Combines the BC's physical features (under ``"physical"``), cell
        normals, and optionally latent features into a single schema
        declaration that mirrors the ``source_data`` structure produced by
        :meth:`_evaluate_hyperlayer`.

        Parameters
        ----------
        bc_source_schema : FieldSchemaLike
            Schema of the per-face physical source features for one BC type.
        include_latents : bool
            Whether to add the latent scalars and vectors, which exist only
            after the first hyperlayer.

        Returns
        -------
        FieldSchemaLike
            Nested declaration with keys ``"physical"``, ``"normals"``, and
            (if ``include_latents``) ``"latent"``.
        """
        result: dict[str, Any] = {
            "physical": bc_source_schema,
            "normals": {"rank": 1},
        }
        if include_latents:
            result["latent"] = {
                "scalars": {str(i): {"rank": 0} for i in range(self.n_latent_scalars)},
                "vectors": {str(i): {"rank": 1} for i in range(self.n_latent_vectors)},
            }
        return result

    @torch.compiler.disable
    def _build_trees_and_plans(
        self,
        boundary_meshes: dict[str, Mesh["n-1", "n"]],  # ty: ignore[unresolved-reference]
    ) -> tuple[
        dict[str, ClusterTree],
        dict[str, torch.Tensor],
        dict[str, DualInteractionPlan],
        dict[str, torch.Tensor | None],
    ]:
        r"""Build per-BC-type cluster trees and cross-BC dual interaction plans.

        Builds one :class:`ClusterTree` per BC type (O(B) trees), then plans
        the interactions of every (source BC, destination BC) pair: one dual
        traversal per source BC against a forest of all B trees
        (:meth:`ClusterTree.merge`), whose plan is that source's B plans
        joined, indexing the cell centroids of all BCs laid end to end (in
        ``boundary_meshes`` order).  A communication layer then evaluates
        each source kernel once for all boundary cells.  Plans are reused
        across all communication layers since the geometry is fixed.

        Parameters
        ----------
        boundary_meshes : dict[str, Mesh]
            Raw (un-enriched) per-BC-type boundary meshes, so that
            measure weights are still visible at the top level
            of ``cell_data`` (see :mod:`physicsnemo.mesh.calculus.measure`).

        Returns
        -------
        cluster_trees : dict[str, ClusterTree]
            Per-BC-type cluster trees built from cell centroids.
        bc_areas : dict[str, torch.Tensor]
            Per-BC-type ``reference_area``-normalized effective cell-measure
            tensors, consumed both by Barnes-Hut source aggregation
            and by the source-strength integration in
            :meth:`_evaluate_hyperlayer`.
        comm_plans : dict[str, DualInteractionPlan]
            Communication plan of each source BC to the cell centroids of all
            BCs, concatenated in ``boundary_meshes`` order.
        comm_target_centroids : dict[str, torch.Tensor or None]
            Target-node centroids that each communication plan refers to,
            concatenated the same way, or ``None`` if the plan has no
            far-target pairs (always so with ``expand_far_targets``). They
            are area-weighted for the source's own tree and plain node means
            for the others, as in the kernels.
        """
        from physicsnemo.mesh.spatial.cluster_tree import ClusterTree

        ### ``no_grad`` is safe: tree inputs (centroids, areas) carry no grad
        ### and the outputs are consumed downstream as integer indices and as
        ### a non-grad-tracked area divisor in ``compute_source_aggregates``.
        ### Without it we'd pay autograd bookkeeping on dozens of tensor ops
        ### inside the ``@torch.compiler.disable`` body for nothing.  See the
        ### ``tree_build_device`` docstring on :class:`GLOBE` for why building
        ### on CPU can be faster than on CUDA at small ``N``.
        ### ``boundary_meshes`` may be a TensorDict, which has no truth value.
        bc_types = list(boundary_meshes.keys())
        meshes = [boundary_meshes[bc] for bc in bc_types]
        original_device = (
            meshes[0].points.device if meshes else self.reference_area.device
        )
        build_device = self.tree_build_device or original_device

        with torch.no_grad():
            ### Pass 1: build one tree per BC.  Must complete before pass 2
            ### so every plan can see every tree.
            cluster_trees_built: dict[str, ClusterTree] = {}
            bc_areas_built: dict[str, torch.Tensor] = {}
            centroids_built: dict[str, torch.Tensor] = {}
            for bc_type, mesh in zip(bc_types, meshes):
                centroids = mesh.cell_centroids.to(build_device)
                areas = (cell_measures(mesh) / self.reference_area).to(
                    build_device
                )
                centroids_built[bc_type] = centroids
                bc_areas_built[bc_type] = areas
                cluster_trees_built[bc_type] = ClusterTree.from_points(
                    centroids, leaf_size=self.leaf_size, areas=areas,
                )

            ### Pass 2: one dual traversal per source BC, against a forest
            ### of every BC's tree: the B destination plans of that source,
            ### already joined (B traversals instead of B^2).
            n_targets = sum(mesh.n_cells for mesh in meshes)
            if len(bc_types) == 1:  # the tree itself: self-interaction
                forest = cluster_trees_built[bc_types[0]]
            else:
                forest = ClusterTree.merge(
                    [cluster_trees_built[bc] for bc in bc_types]
                )
            comm_plans_built: dict[str, DualInteractionPlan] = {}
            comm_centroids_built: dict[str, torch.Tensor | None] = {}
            for src_bc, src_mesh in zip(bc_types, meshes):
                plan = cluster_trees_built[src_bc].find_dual_interaction_pairs(
                    target_tree=forest, theta=self.theta,
                    expand_far_targets=self.expand_far_targets, validate=False,
                )
                comm_plans_built[src_bc] = plan
                logger.logger.debug(
                    "comm plan [%s -> all]: %d near + %d nf + %d fn + %d far_node "
                    "(%.2f%% near-field, %d src x %d dst faces, "
                    "theta=%.2f, leaf_size=%d)",
                    src_bc,
                    plan.n_near, plan.n_nf, plan.n_fn, plan.n_far_nodes,
                    100.0 * plan.n_near / max(src_mesh.n_cells * n_targets, 1),
                    src_mesh.n_cells, n_targets, self.theta, self.leaf_size,
                )
                ### Target-node centroids, only for plans that use them, in
                ### the forest's node order, one block per destination tree
                ### by the kernels' rule (``_target_node_centroids``).
                if plan.n_far_nodes + plan.n_fn > 0:
                    src_tree = cluster_trees_built[src_bc]
                    src_node_centroids = src_tree.compute_source_aggregates(
                        source_points=centroids_built[src_bc],
                        areas=bc_areas_built[src_bc],
                    ).node_centroid
                    comm_centroids_built[src_bc] = torch.cat(
                        [
                            _target_node_centroids(
                                src_tree,
                                cluster_trees_built[dst_bc],
                                centroids_built[dst_bc],
                                src_node_centroids,
                            )
                            for dst_bc in bc_types
                        ]
                    )
                else:
                    comm_centroids_built[src_bc] = None

        ### Transfer to the original device.  ``ClusterTree`` and
        ### ``DualInteractionPlan`` are both tensorclasses, so ``.to`` moves
        ### all member tensors at once.  No-op when devices already match.
        cluster_trees = {
            bc: t.to(original_device)  # ty: ignore[unresolved-attribute]
            for bc, t in cluster_trees_built.items()
        }
        bc_areas = {bc: a.to(original_device) for bc, a in bc_areas_built.items()}
        comm_plans = {
            bc: p.to(original_device)  # ty: ignore[unresolved-attribute]
            for bc, p in comm_plans_built.items()
        }
        comm_target_centroids = {
            bc: None if c is None else c.to(original_device)
            for bc, c in comm_centroids_built.items()
        }
        return cluster_trees, bc_areas, comm_plans, comm_target_centroids

    @torch.compiler.disable
    def _build_prediction_plans(
        self,
        cluster_trees: dict[str, ClusterTree],
        prediction_points: torch.Tensor,
    ) -> tuple[ClusterTree, dict[str, DualInteractionPlan]]:
        """Build target tree and dual plans for prediction-point evaluation.

        Builds a single target tree from ``prediction_points`` and computes
        one :class:`DualInteractionPlan` per source BC type against it.

        Returns
        -------
        pred_target_tree : ClusterTree
            Target tree built from prediction points.
        pred_plans : dict[str, DualInteractionPlan]
            Plans indexed by source BC type, each computed from that source
            BC's tree to ``pred_target_tree``.
        """
        from physicsnemo.mesh.spatial.cluster_tree import ClusterTree

        ### See ``_build_trees_and_plans`` for the ``no_grad`` + build-device
        ### rationale.  ``cluster_trees`` arrive on the caller's device from
        ### that earlier call, so we move them to ``build_device`` for the
        ### dual traversal and transfer both the new prediction-point tree
        ### and the resulting plans back at function exit.
        original_device = prediction_points.device
        build_device = self.tree_build_device or original_device

        n_pred = prediction_points.shape[0]
        with torch.no_grad():
            pred_target_tree_built = ClusterTree.from_points(
                prediction_points.to(build_device), leaf_size=self.leaf_size,
            )
            ### Single pass: each iteration is independent - transfer the
            ### source tree to ``build_device``, run the dual traversal
            ### against the shared ``pred_target_tree_built``, log inline.
            pred_plans_built: dict[str, DualInteractionPlan] = {}
            for bc_type, tree in cluster_trees.items():
                plan = tree.to(build_device).find_dual_interaction_pairs(  # ty: ignore[unresolved-attribute]
                    target_tree=pred_target_tree_built, theta=self.theta,
                    expand_far_targets=self.expand_far_targets, validate=False,
                )
                pred_plans_built[bc_type] = plan
                logger.logger.debug(
                    "pred plan [%s]: %d near + %d nf + %d fn + %d far_node "
                    "(%d sources x %d targets, theta=%.2f)",
                    bc_type, plan.n_near, plan.n_nf, plan.n_fn, plan.n_far_nodes,
                    tree.n_sources, n_pred, self.theta,
                )

        pred_target_tree = pred_target_tree_built.to(original_device)  # ty: ignore[unresolved-attribute]
        pred_plans = {
            bc: p.to(original_device)  # ty: ignore[unresolved-attribute]
            for bc, p in pred_plans_built.items()
        }
        return pred_target_tree, pred_plans

    def _evaluate_hyperlayer(
        self,
        layer_idx: int,
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        source_meshes: dict[str, Mesh["n-1", "n"]],  # ty: ignore[unresolved-reference]
        reference_lengths: dict[str, Float[torch.Tensor, ""]],
        global_data: TensorDict[str, Float[torch.Tensor, "..."]] | None,
        cluster_trees: dict[str, ClusterTree],
        dual_plans: dict[str, DualInteractionPlan],
        source_areas: dict[str, torch.Tensor],
        target_tree: ClusterTree | None = None,
        target_centroids: dict[str, torch.Tensor | None] | None = None,
    ) -> Float[torch.Tensor, "n_targets n_features"]:
        r"""Evaluate one hyperlayer by summing kernel contributions from all BC types.

        Each call evaluates all source BC types against a single set of target
        points.  The per-source-BC ``dual_plans`` (and ``target_tree`` or
        ``target_centroids``, if given) must correspond to those target
        points.

        Parameters
        ----------
        layer_idx : int
            Index into ``self.kernel_layers``.
        target_points : Float[torch.Tensor, "n_targets n_dims"]
            Target points of shape :math:`(N_{targets}, D)`.
        source_meshes : dict[str, Mesh["n-1", "n"]]
            Enriched boundary meshes with cell_data containing physical
            features, strengths, and (after layer 0) latent state.
        reference_lengths : dict[str, Float[torch.Tensor, ""]]
            Reference length names to scalar tensors.
        global_data : TensorDict or None
            Problem-level features.
        cluster_trees : dict[str, ClusterTree]
            Per-BC-type precomputed source trees.
        dual_plans : dict[str, DualInteractionPlan]
            Per-source-BC-type precomputed dual interaction plans to the
            targets.
        source_areas : dict[str, torch.Tensor]
            Per-BC-type source area tensors.
        target_tree : ClusterTree or None, optional
            Tree over ``target_points`` that the plans were built against,
            used for target-node centroids when a plan has far-target pairs
            and ``target_centroids`` does not give them.
        target_centroids : dict[str, torch.Tensor or None] or None, optional
            Per-source-BC target-node centroids of the plans (see
            :meth:`_build_trees_and_plans`).

        Returns
        -------
        Float[torch.Tensor, "n_targets n_features"]
            Kernel outputs summed over all boundary condition types, of shape
            :math:`(N_{targets}, F)`, packed as the layer's kernels'
            ``output_layout`` describes.
        """
        result: torch.Tensor | None = None

        for bc_type, mesh in source_meshes.items():
            ### `source_areas` holds the reference_area-normalized effective
            ### cell measures (cell_areas * measure weights),
            ### computed once from the raw input meshes in
            ### `_build_trees_and_plans` -- the mesh geometry is fixed across
            ### layers, so reusing them here keeps the strength integration
            ### and the Barnes-Hut aggregation on the same measure.
            strengths: TensorDict[str, Float[torch.Tensor, " n_cells"]] = (
                mesh.cell_data["strengths"].apply(  # ty: ignore[unresolved-attribute]
                    lambda x: x * source_areas[bc_type]
                )
            )

            kernel: MultiscaleKernel = self.kernel_layers[layer_idx][bc_type]  # ty: ignore[not-subscriptable]
            ### Pull only the leaves the kernel was built to consume.
            ### `kernel.source_schema` covers the per-bc declared
            ### `physical.<X>` plus, in layers >= 1, `latent.<...>` added
            ### by the previous communication step.  `"normals"` is
            ### excluded here because we auto-inject it from
            ### `mesh.cell_normals` on the next line; pulling it from
            ### `cell_data` too would double-count.  Extras in cell_data
            ### are silently dropped by ``select`` -- the contract is
            ### enforced by ``FieldSchema.check`` at ``GLOBE.forward``
            ### entry.
            kernel_source_keys = kernel.source_schema.keys() - {"normals"}
            source_data = mesh.cell_data.select(
                *(FieldSchema.key(name) for name in kernel_source_keys)
            )
            source_data["normals"] = mesh.cell_normals

            piece = kernel.forward_packed(
                source_points=mesh.cell_centroids,
                source_data=source_data,
                source_strengths=strengths,
                target_points=target_points,
                reference_lengths=reference_lengths,
                global_data=global_data,
                theta=self.theta,
                cluster_tree=cluster_trees[bc_type],
                target_tree=target_tree,
                dual_plan=dual_plans[bc_type],
                source_areas=source_areas[bc_type],
                target_centroids=(
                    None if target_centroids is None else target_centroids[bc_type]
                ),
            )
            result = piece if result is None else result + piece

        return result

    def _compress_latents(
        self, packed: Float[torch.Tensor, "n_cells n_features"]
    ) -> Float[torch.Tensor, "n_cells n_features"]:
        r"""Compress packed communication-layer outputs, field by field.

        Prevents the communication-layer amplification loop from producing
        O(10^4) intermediate values.  Uses arcsinh for C-infinity smooth
        logarithmic compression: of each scalar, and of each vector's
        magnitude (keeping its direction).

        Parameters
        ----------
        packed : Float[torch.Tensor, "n_cells n_features"]
            Communication-layer outputs, of shape :math:`(N_{cells}, F)`,
            packed as ``self._intermediate_layout`` describes.

        Returns
        -------
        Float[torch.Tensor, "n_cells n_features"]
            The compressed outputs, of shape :math:`(N_{cells}, F)`.
        """
        C = self.latent_compression_scale
        eps_sq = self.smoothing_radius**2
        pieces: list[torch.Tensor] = []
        for start, stop, n_dims in self._intermediate_runs:
            block = packed[:, start:stop]
            if n_dims == 0:
                pieces.append(C * torch.arcsinh(block / C))
                continue
            t = block.unflatten(-1, (-1, n_dims))
            r = (t.pow(2).sum(dim=-1, keepdim=True) + eps_sq).sqrt()
            pieces.append((t * (C * torch.arcsinh(r / C) / r)).flatten(-2))
        return torch.cat(pieces, dim=-1)

    def _evaluate_communication_hyperlayer(
        self,
        layer_idx: int,
        boundary_meshes: dict[str, Mesh["n-1", "n"]],  # ty: ignore[unresolved-reference]
        reference_lengths: dict[str, Float[torch.Tensor, ""]],
        global_data: TensorDict[str, Float[torch.Tensor, "..."]] | None,
        cluster_trees: dict[str, ClusterTree],
        comm_plans: dict[str, DualInteractionPlan],
        comm_target_centroids: dict[str, torch.Tensor | None],
        source_areas: dict[str, torch.Tensor],
    ) -> dict[str, Mesh["n-1", "n"]]:  # ty: ignore[unresolved-reference]
        r"""Run one boundary-to-boundary communication step.

        Evaluates :meth:`_evaluate_hyperlayer` once, at the cell centroids
        of every BC type laid end to end, summing contributions from all
        source BC types: each source kernel runs once for all destinations,
        with the joined plans of :meth:`_build_trees_and_plans`.  The packed
        result is split back into one block of rows per BC type, and each
        block's fields become that boundary mesh's new ``cell_data``.

        Parameters
        ----------
        layer_idx : int
            Index into ``self.kernel_layers``.
        boundary_meshes : dict[str, Mesh["n-1", "n"]]
            Boundary meshes whose ``cell_data`` holds the physical features,
            strengths, and (after layer 0) latent state.
        reference_lengths : dict[str, Float[torch.Tensor, ""]]
            Reference length names to 0-d tensors.
        global_data : TensorDict or None
            Problem-level features.
        cluster_trees : dict[str, ClusterTree]
            Per-BC-type source trees.
        comm_plans : dict[str, DualInteractionPlan]
            Per-source-BC plans to the cell centroids of all BC types (see
            :meth:`_build_trees_and_plans`).
        comm_target_centroids : dict[str, torch.Tensor or None]
            Per-source-BC target-node centroids of ``comm_plans``, each of
            shape :math:`(M, D)` or ``None``.
        source_areas : dict[str, torch.Tensor]
            Per-BC-type source areas, each of shape :math:`(N_{cells},)`.

        Returns
        -------
        dict[str, Mesh["n-1", "n"]]
            Updated boundary meshes with evaluation results merged into
            each mesh's ``cell_data``.
        """
        bc_types = list(boundary_meshes.keys())
        meshes = [boundary_meshes[bc] for bc in bc_types]
        packed = self._evaluate_hyperlayer(
            layer_idx=layer_idx,
            target_points=(
                meshes[0].cell_centroids
                if len(meshes) == 1
                else torch.cat([mesh.cell_centroids for mesh in meshes])
            ),
            source_meshes=boundary_meshes,
            reference_lengths=reference_lengths,
            global_data=global_data,
            cluster_trees=cluster_trees,
            dual_plans=comm_plans,
            source_areas=source_areas,
            target_centroids=comm_target_centroids,
        )
        if self.latent_compression_scale is not None:
            packed = self._compress_latents(packed)

        new_meshes: dict[str, Mesh["n-1", "n"]] = {}  # ty: ignore[unresolved-reference]
        rows = torch.split(packed, [mesh.n_cells for mesh in meshes])
        for bc_type, mesh, mesh_rows in zip(bc_types, meshes, rows):
            new_cell_data = TensorDict(
                {"physical": mesh.cell_data["physical"]},
                batch_size=torch.Size([mesh.n_cells]),
                device=mesh.points.device,
            )
            ### The new strengths and latents, as views of the packed rows.
            new_cell_data.update(self._intermediate_layout.unpack(mesh_rows))
            new_meshes[bc_type] = mesh.with_data(
                point_data={},
                cell_data=new_cell_data,
                global_data={},
            )
        return new_meshes

    def forward(
        self,
        prediction_points: Float[torch.Tensor, "n_points n_dims"],
        boundary_meshes: dict[str, Mesh["n-1", "n"]],  # ty: ignore[unresolved-reference]
        reference_lengths: dict[str, torch.Tensor],
        global_data: TensorDict[str, Float[torch.Tensor, "..."]] | None = None,
        prediction_chunk_size: int | Literal["auto"] | None = "auto",
    ) -> Mesh[0, "n"]:  # ty: ignore[unresolved-reference]
        r"""Evaluate GLOBE model to predict fields at target points.

        Runs the full GLOBE forward pass in three phases:

        1. **Init**: Enrich boundary meshes with initial (all-ones) strengths,
           wrapping original ``cell_data`` under a ``"physical"`` namespace.
        2. **Communication**: Run ``n_communication_hyperlayers`` boundary-to-
           boundary communication steps via
           :meth:`_evaluate_communication_hyperlayer`.  Each boundary mesh
           carries its state from layer to layer in its ``cell_data``:
           the physical features, plus the strengths and latents that the
           previous layer computed.  Each source BC's kernel runs once per
           layer for the cells of all BCs.
        3. **Final evaluation**: Evaluate the last hyperlayer at
           ``prediction_points`` and apply per-field calibration transforms.
           When the number of prediction points exceeds
           ``prediction_chunk_size``, Phase 3 is executed in chunks to
           bound memory usage.  The communication layers (Phase 2) run
           once and their results are reused across all chunks.

        Parameters
        ----------
        prediction_points : Float[torch.Tensor, "n_points n_dims"]
            Target points of shape :math:`(N_{points}, D)`.
        boundary_meshes : dict[str, Mesh["n-1", "n"]]
            Dictionary mapping BC type names to pre-merged
            :class:`~physicsnemo.mesh.Mesh` objects.
        reference_lengths : dict[str, torch.Tensor]
            Mapping of reference length names to scalar tensors.
        global_data : TensorDict or None, optional, default=None
            Nondimensional conditioning features. Leaf keys and ranks must
            match ``global_schema``. Passed through to the output Mesh.
        prediction_chunk_size : int or "auto" or None, optional, default="auto"
            Maximum number of prediction points to evaluate in a single
            pass through the final hyperlayer.  ``"auto"`` (the default)
            uses the total number of boundary faces across all BC types,
            which gives the final evaluation roughly the same memory
            footprint as the communication layers.  ``None`` disables
            chunking, evaluating all prediction points in one pass.

        Returns
        -------
        Mesh[0, "n"]
            A point-cloud Mesh (0-dimensional manifold) with predicted fields.
        """
        device = prediction_points.device

        if global_data is None:
            global_data = TensorDict({}, device=device)

        ### Filter `global_data` down to declared leaves before any
        ### downstream code sees it. Mirrors the per-bc cell_data filter
        ### in `_evaluate_hyperlayer`: extras are silently dropped, so
        ### users can pass the recipe's full mesh `global_data` (which
        ### may carry e.g. metadata fields) without polluting the kernel
        ### feature stream. A user-supplied leaf that's missing from the
        ### declaration is caught by ``FieldSchema.check`` below.  Schema
        ### names are "."-joined, which TensorDict does not parse, so
        ### convert them to nested keys.
        global_data = global_data.select(
            *(FieldSchema.key(name) for name in self.global_schema)
        )

        ### Input validation
        if not torch.compiler.is_compiling():
            if prediction_points.ndim != 2:
                raise ValueError(
                    f"Expected 2D prediction_points (N, D), got {prediction_points.ndim}D "
                    f"tensor with shape {tuple(prediction_points.shape)}"
                )
            if prediction_points.shape[-1] != self.n_spatial_dims:
                raise ValueError(
                    f"Expected prediction_points with {self.n_spatial_dims} spatial dims, "
                    f"got {prediction_points.shape[-1]}"
                )
            if set(reference_lengths.keys()) != set(self.reference_length_names):
                raise ValueError(
                    f"This model was instantiated to expect reference lengths "
                    f"{set(self.reference_length_names)!r},\n"
                    f"but the forward-method input gives {set(reference_lengths.keys())!r}."
                )
            ### Check the bc-types subset BEFORE the per-bc loop so a typo'd
            ### bc name fails with a clear message instead of crashing on
            ### `self.boundary_source_schemas[bc_type]` lookup below.
            bc_types_from_input = set(boundary_meshes.keys())
            if not bc_types_from_input.issubset(self.boundary_condition_names):
                raise ValueError(
                    f"The input gives boundary meshes with these boundary condition types:\n"
                    f"{bc_types_from_input!r}\n"
                    f"but the model was instantiated to expect only these boundary condition types:\n"
                    f"{self.boundary_condition_names!r}\n"
                    f"Please ensure that the input boundary meshes are a subset of the model's boundary condition types."
                )
            for bc_type, mesh in boundary_meshes.items():
                if mesh.n_spatial_dims != self.n_spatial_dims:
                    raise ValueError(
                        f"Boundary mesh for BC type {bc_type!r} has "
                        f"{mesh.n_spatial_dims} spatial dims, but the model expects "
                        f"{self.n_spatial_dims}"
                    )
                ### Subset cell_data contract: every leaf declared in
                ### `boundary_source_schemas[bc_type]` must be present
                ### in `mesh.cell_data` with the declared rank. Extras
                ### are silently dropped by the `select` in
                ### `_evaluate_hyperlayer`. Catches typo'd declarations
                ### and shape regressions with a friendlier error than
                ### the bare KeyError that `select(strict=True)` would
                ### otherwise raise on a missing leaf.
                self.boundary_source_schemas[bc_type].check(
                    mesh.cell_data,
                    label=f"`boundary_meshes[{bc_type!r}].cell_data`",
                )
            ### Same subset contract for `global_data`. The pre-filter
            ### above already dropped extras, so this only fires when a
            ### declared leaf was missing or had a rank mismatch.
            self.global_schema.check(global_data, label="`global_data`")

        ### Build per-BC-type trees and areas (reused across all layers).
        ### Tree construction and traversal involve irregular control flow
        ### (morton codes, variable-depth loops) that cannot be traced by
        ### torch.compile, so we skip compilation for this block.
        ### Built from the raw input meshes, BEFORE enrichment: tree and
        ### area construction consume only geometry (identical either way),
        ### and the measure weights must be read before
        ### enrichment nests each mesh's cell_data under the "physical"
        ### namespace.
        with record_function("globe::build_trees_and_plans"):
            cluster_trees, bc_areas, comm_plans, comm_target_centroids = (
                self._build_trees_and_plans(boundary_meshes)
            )

        ### Phase 1: Enrich boundary meshes with initial (all-ones) strengths.
        ### (The reserved sampling-weights key rides along inside
        ### "physical"; it is inert there because kernels `select` only
        ### their declared feature keys.)
        with record_function("globe::enrich_meshes"):
            boundary_meshes = {
                bc_type: mesh.with_data(
                    point_data={},
                    cell_data=TensorDict(
                        {
                            "physical": mesh.cell_data,
                            "strengths": TensorDict(
                                {
                                    name: torch.ones(mesh.n_cells, device=device)
                                    for name in self.reference_length_names
                                },
                                batch_size=torch.Size([mesh.n_cells]),
                                device=device,
                            ),
                        },
                        batch_size=torch.Size([mesh.n_cells]),
                        device=device,
                    ),
                    global_data={},
                )
                for bc_type, mesh in boundary_meshes.items()
            }

        ### Phase 2: Communication hyperlayers (boundary-to-boundary).
        # Trees and comm_plans are reused across all layers because cell
        # centroids (the source/target points) are fixed - only the
        # cell_data (latent features, strengths) changes between layers.
        for i in range(self.n_communication_hyperlayers):
            with record_function(f"globe::communication_layer/{i}"):
                boundary_meshes = self._evaluate_communication_hyperlayer(
                    layer_idx=i,
                    boundary_meshes=boundary_meshes,
                    reference_lengths=reference_lengths,
                    global_data=global_data,
                    cluster_trees=cluster_trees,
                    comm_plans=comm_plans,
                    comm_target_centroids=comm_target_centroids,
                    source_areas=bc_areas,
                )

        ### Free comm plans - no longer needed after communication layers.
        # At 800k faces, near-pair indices can be ~3 GB of int64.
        del comm_plans, comm_target_centroids

        ### Phase 3: Final evaluation at prediction points (chunked).
        # When n_prediction_points > chunk_size, prediction points are
        # split into chunks to bound memory.  Each chunk builds its own
        # target tree and dual plans (cheap), then evaluates the final
        # hyperlayer.  Results are concatenated at the end.
        n_points = prediction_points.shape[0]
        if prediction_chunk_size == "auto":
            prediction_chunk_size = sum(
                m.n_cells for m in boundary_meshes.values()
            )
        elif prediction_chunk_size is None:
            prediction_chunk_size = n_points
        else:
            if not (isinstance(prediction_chunk_size, int) and prediction_chunk_size > 0):
                raise ValueError(
                    f"Expected prediction_chunk_size to be a positive integer or 'auto' or None, "
                    f"got {prediction_chunk_size=!r}."
                )

        chunk_results: list[torch.Tensor] = []
        for start in range(0, n_points, prediction_chunk_size):
            chunk_pts = prediction_points[start : start + prediction_chunk_size]

            with record_function("globe::build_prediction_plans"):
                pred_target_tree, pred_plans = self._build_prediction_plans(
                    cluster_trees, chunk_pts
                )

            with record_function("globe::final_evaluation"):
                chunk_result = self._evaluate_hyperlayer(
                    layer_idx=self.n_communication_hyperlayers,
                    target_points=chunk_pts,
                    source_meshes=boundary_meshes,
                    reference_lengths=reference_lengths,
                    global_data=global_data,
                    cluster_trees=cluster_trees,
                    target_tree=pred_target_tree,
                    dual_plans=pred_plans,
                    source_areas=bc_areas,
                )

            del pred_plans, pred_target_tree
            chunk_results.append(chunk_result)

        packed = (
            torch.cat(chunk_results, dim=0)
            if len(chunk_results) > 1
            else chunk_results[0]
        )

        ### Wrap as point-cloud Mesh and apply per-field calibration.
        with record_function("globe::calibration"):
            output_mesh = Mesh(
                points=prediction_points,
                point_data=self._output_layout.unpack(packed),
                global_data=global_data,
            )
            for idx, name in enumerate(self._output_field_order):
                key = FieldSchema.key(name)
                t = output_mesh.point_data[key]
                output_mesh.point_data[key] = self.final_field_transforms[idx](
                    t.reshape(-1, 1)
                ).reshape(t.shape)
        return output_mesh
