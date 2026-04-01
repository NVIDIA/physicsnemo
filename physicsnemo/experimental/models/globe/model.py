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

import operator
from dataclasses import dataclass
from functools import reduce
from typing import Sequence

import torch
import torch.nn as nn
from jaxtyping import Float
from tensordict import TensorDict
from torch.profiler import record_function

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.globe.cluster_tree import (
    ClusterTree,
    DualInteractionPlan,
)
from physicsnemo.experimental.models.globe.field_kernel import MultiscaleKernel
from physicsnemo.experimental.models.globe.utilities.rank_spec import (
    RankSpecDict,
    flatten_rank_spec,
)
from physicsnemo.mesh import Mesh
from physicsnemo.utils.logging import PythonLogger

# allow_in_graph wraps these TensorDict methods as opaque graph nodes so that
# torch.compile doesn't trace into them (their internals cause graph breaks).
# This is safe because flatten_keys/unflatten_keys are pure structural
# key-renaming operations with no tensor-data side effects — the set of tensor
# storages in equals the set coming out.  Do NOT generalise this pattern to
# functions with tensor-value-dependent control flow or side effects.
# If a future tensordict version makes these natively Dynamo-traceable, remove
# these wrappers.
_flatten_keys = torch.compiler.allow_in_graph(TensorDict.flatten_keys)
_unflatten_keys = torch.compiler.allow_in_graph(TensorDict.unflatten_keys)

logger = PythonLogger("globe.model")

@dataclass
class MetaData(ModelMetaData):
    jit: bool = False  # Refers to torch.compile compatibility - this is compatible.
    cuda_graphs: bool = False  # Computational graph changes depending on inputs due to tree traversals
    amp: bool = True
    torch_fx: bool = False
    onnx: bool = False


class GLOBE(Module):
    r"""Green's-function-Like Operator for Boundary Element PDEs.

    GLOBE is a neural surrogate architecture for boundary-driven elliptic PDEs that
    combines learnable Green's-function-like kernels with equivariant ML. The model
    represents solutions as superpositions of kernel evaluations from boundary faces
    to target points, with communication hyperlayers enabling boundary-to-boundary
    information propagation before final interior evaluation.

    The architecture is designed to satisfy fundamental physical requirements:

    - Translation-, rotation-, and parity-equivariant through relative positions and
      local basis reprojection
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
    output_field_ranks : TensorDict
        Rank-spec TensorDict with integer leaves (0 = scalar, 1 = vector)
        describing the output fields. Derive from data via
        :func:`ranks_from_tensordict`.
    boundary_source_data_ranks : dict[str, TensorDict]
        Mapping of boundary condition type names to rank-spec TensorDicts
        describing the per-face source features for each BC type. The keys
        implicitly define the set of boundary condition names. The face
        normal vector is automatically added, so don't include it.
    reference_length_names : Sequence[str]
        Sequence of identifiers for reference length scales
        (e.g., ``["viscous_length", "chord_length"]``). Each creates a separate
        kernel branch in the multiscale composition.
    reference_area : float
        Scalar used to nondimensionalize face areas. Typically a characteristic
        area of the problem (e.g., chord^2 for airfoils).
    global_data_ranks : TensorDict or None, optional
        Rank-spec TensorDict for global conditioning features. Defaults to
        empty (no global conditioning).
    n_communication_hyperlayers : int, optional, default=2
        Number of boundary-to-boundary communication layers before final evaluation.
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
    theta : float, optional, default=1.0
        Barnes-Hut opening angle controlling the near/far-field split in the
        dual-tree traversal. The criterion is
        :math:`(D_T + D_S) / r < \theta`, where :math:`D_T` and :math:`D_S`
        are AABB diagonals and :math:`r` is the minimum inter-AABB distance.
        Larger values approximate more aggressively; ``0`` forces all
        interactions to be exact (no far-field approximation).
    leaf_size : int, optional, default=1
        Maximum number of source points per leaf node in the cluster tree.
        Larger values produce shallower trees (fewer traversal iterations) at
        the cost of more exact near-field interactions per leaf hit.
    expand_far_targets : bool, optional, default=False
        If ``True``, far-field target nodes are expanded to individual points,
        converting ``(far, far)`` pairs into ``(near, far)`` pairs. This
        eliminates the target-side approximation at the cost of more kernel
        evaluations.

    Forward
    -------
    prediction_points : Float[torch.Tensor, "n_points n_dims"]
        Target points for field evaluation of shape :math:`(N_{points}, D)`.
    boundary_meshes : dict[str, Mesh["n-1", "n"]]
        Dictionary mapping boundary condition type names to
        :class:`~physicsnemo.mesh.Mesh` objects. Keys must be a subset of the
        model's boundary condition names (from ``boundary_source_data_ranks``).
    reference_lengths : dict[str, torch.Tensor]
        Dictionary mapping reference length names to scalar tensors.
    global_data : TensorDict or None, optional, default=None
        Nondimensional conditioning features. Leaf keys and ranks must match
        ``global_data_ranks``. Passed through to the output Mesh.

    Outputs
    -------
    Mesh[0, "n"]
        A point-cloud :class:`~physicsnemo.mesh.Mesh` (0-dimensional manifold)
        whose ``.points`` attribute equals the input ``prediction_points``. The
        predicted fields are in ``.point_data``, keyed by the names from
        ``output_field_ranks``.
        Scalar fields have shape :math:`(N_{points},)`, vector fields have shape
        :math:`(N_{points}, D)`. Cells are empty (shape ``(0, 1)``).
        ``global_data`` is passed through from the input.

    Notes
    -----
    - ``kernel_layers`` is a :class:`~torch.nn.ModuleList` of communication
      hyperlayers, each containing a :class:`~torch.nn.ModuleDict` mapping BC type
      names to :class:`~physicsnemo.experimental.models.globe.field_kernel.MultiscaleKernel`
      instances.
    - ``final_field_transforms`` is a :class:`~torch.nn.ModuleList` of per-field
      linear calibration layers, ordered alphabetically by field name.
    - Cell areas are automatically normalized by ``reference_area`` to preserve
      discretization-invariance.
    - The cell normal vector is automatically added to source data for each mesh.
    - The ``Mesh["n-1", "n"]`` type annotations assume the PDE domain fills the
      full ambient space (domain manifold dim = spatial dim), so boundary meshes
      are codimension-1 in the ambient space. For a PDE on a ``d``-dimensional
      manifold embedded in ``n``-dimensional space (``d < n``), the boundary
      type would be ``Mesh[d-1, n]`` instead.

    Examples
    --------
    >>> model = GLOBE(
    ...     n_spatial_dims=3,
    ...     output_field_ranks=TensorDict({"pressure": 0, "velocity": 1}),
    ...     boundary_source_data_ranks={
    ...         "no_slip": TensorDict({}),
    ...         "freestream": TensorDict({}),
    ...     },
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
        output_field_ranks: RankSpecDict,
        boundary_source_data_ranks: dict[str, RankSpecDict],
        reference_length_names: Sequence[str],
        reference_area: float,
        global_data_ranks: RankSpecDict | None = None,
        n_communication_hyperlayers: int = 2,
        n_latent_scalars: int = 12,
        n_latent_vectors: int = 6,
        smoothing_radius: float = 1e-8,
        hidden_layer_sizes: Sequence[int] | None = None,
        n_spherical_harmonics: int = 4,
        theta: float = 1.0,
        leaf_size: int = 1,
        expand_far_targets: bool = False,
    ):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64, 64, 64]
        if global_data_ranks is None:
            global_data_ranks = {}

        boundary_condition_names = list(boundary_source_data_ranks.keys())

        ### Input validation (eager mode only)
        for rank in flatten_rank_spec(output_field_ranks).values():
            if rank not in (0, 1):
                raise ValueError(
                    f"All leaves of output_field_ranks must be 0 (scalar) or 1 (vector), "
                    f"got {rank!r}"
                )
        for bc_name in boundary_condition_names:
            if "." in bc_name:
                raise ValueError(
                    f"In `boundary_source_data_ranks`, got {bc_name=!r};\n"
                    "BC names must not contain `.` for TensorDict compatibility."
                )

        super().__init__(meta=MetaData())

        self.n_spatial_dims = n_spatial_dims
        self.output_field_ranks = output_field_ranks
        self.boundary_condition_names = boundary_condition_names
        self.boundary_source_data_ranks = boundary_source_data_ranks
        self.reference_length_names = reference_length_names
        self.register_buffer("reference_area", torch.tensor(reference_area))
        self.global_data_ranks = global_data_ranks
        self.n_communication_hyperlayers = n_communication_hyperlayers
        self.n_latent_scalars = n_latent_scalars
        self.n_latent_vectors = n_latent_vectors
        self.smoothing_radius = smoothing_radius
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_spherical_harmonics = n_spherical_harmonics
        self.theta = theta
        self.leaf_size = leaf_size
        self.expand_far_targets = expand_far_targets

        ### Build the intermediate output-field rank spec for communication
        # hyperlayers. Only the final hyperlayer emits output_field_ranks.
        intermediate_field_ranks: RankSpecDict = {
            **{f"strengths.{name}": 0 for name in reference_length_names},
            **{f"latent.scalars.{i}": 0 for i in range(n_latent_scalars)},
            **{f"latent.vectors.{i}": 1 for i in range(n_latent_vectors)},
        }

        kernel_layers = []

        for layer_idx in range(self.n_communication_hyperlayers + 1):
            is_first_hyperlayer = layer_idx == 0
            is_last_hyperlayer = layer_idx == self.n_communication_hyperlayers

            layer = nn.ModuleDict(
                {
                    bc_type: MultiscaleKernel(
                        n_spatial_dims=n_spatial_dims,
                        output_field_ranks=(
                            output_field_ranks
                            if is_last_hyperlayer
                            else intermediate_field_ranks
                        ),
                        reference_length_names=reference_length_names,
                        source_data_ranks=self._build_source_data_ranks(
                            bc_source_ranks=boundary_source_data_ranks[bc_type],
                            include_latents=not is_first_hyperlayer,
                        ),
                        global_data_ranks=global_data_ranks,
                        smoothing_radius=smoothing_radius,
                        hidden_layer_sizes=hidden_layer_sizes,
                        n_spherical_harmonics=n_spherical_harmonics,
                        leaf_size=leaf_size,
                    )
                    for bc_type in boundary_condition_names
                }
            )
            kernel_layers.append(layer)

        self.kernel_layers = nn.ModuleList(kernel_layers)

        ### Per-field learnable affine calibration (y = a*x + b). Bias is only
        # applied to scalar fields; adding bias to vector fields would break
        # rotational equivariance. Uses ModuleList (not ModuleDict) to support
        # output field names containing dots from nested rank specs.
        flat_output_ranks = flatten_rank_spec(output_field_ranks)
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

    def _build_source_data_ranks(
        self,
        bc_source_ranks: RankSpecDict,
        include_latents: bool,
    ) -> RankSpecDict:
        """Build the full source_data_ranks for a specific (layer, bc_type) kernel.

        Combines the BC's physical features (under ``"physical"``), cell
        normals, and optionally latent features into a single rank spec
        that mirrors the ``source_data`` structure produced by
        :meth:`_evaluate_hyperlayer`.
        """
        result: RankSpecDict = {"physical": bc_source_ranks, "normals": 1}
        if include_latents:
            result["latent"] = {
                "scalars": {str(i): 0 for i in range(self.n_latent_scalars)},
                "vectors": {str(i): 1 for i in range(self.n_latent_vectors)},
            }
        return result

    @torch.compiler.disable
    def _build_trees_and_plans(
        self,
        boundary_meshes: dict[str, Mesh["n-1", "n"]],  # ty: ignore[unresolved-reference]
    ) -> tuple[
        dict[str, ClusterTree],
        dict[str, torch.Tensor],
        dict[str, dict[str, DualInteractionPlan]],
    ]:
        """Build per-BC-type cluster trees and cross-BC dual interaction plans.

        Builds one :class:`ClusterTree` per BC type (O(B) trees), then computes
        a :class:`DualInteractionPlan` for every (source BC, destination BC)
        pair (B^2 plans total).  For self-interaction (source == destination),
        the target tree is the same object as the source tree.  Plans are
        reused across all communication layers since the geometry is fixed.

        Returns
        -------
        cluster_trees : dict[str, ClusterTree]
            Per-BC-type cluster trees built from cell centroids.
        bc_areas : dict[str, torch.Tensor]
            Per-BC-type normalized cell area tensors.
        comm_plans : dict[str, dict[str, DualInteractionPlan]]
            Communication plans indexed as ``comm_plans[dst_bc][src_bc]``.
        """
        from physicsnemo.experimental.models.globe.cluster_tree import ClusterTree

        cluster_trees: dict[str, ClusterTree] = {}
        bc_areas: dict[str, torch.Tensor] = {}
        for bc_type, mesh in boundary_meshes.items():
            areas = mesh.cell_areas / self.reference_area
            bc_areas[bc_type] = areas
            cluster_trees[bc_type] = ClusterTree.from_points(
                mesh.cell_centroids, leaf_size=self.leaf_size, areas=areas
            )

        ### Build interaction plans for all (source, destination) BC pairs.
        comm_plans: dict[str, dict[str, DualInteractionPlan]] = {}
        for dst_bc in boundary_meshes:
            comm_plans[dst_bc] = {
                src_bc: cluster_trees[src_bc].find_dual_interaction_pairs(
                    target_tree=cluster_trees[dst_bc], theta=self.theta,
                    expand_far_targets=self.expand_far_targets,
                )
                for src_bc in boundary_meshes
            }

        for dst_bc, plans_for_dst in comm_plans.items():
            n_dst = boundary_meshes[dst_bc].n_cells
            for src_bc, plan in plans_for_dst.items():
                n_src = boundary_meshes[src_bc].n_cells
                logger.logger.debug(
                    "comm plan [%s -> %s]: %d near + %d nf + %d fn + %d far_node "
                    "(%.2f%% near-field, %d src x %d dst faces, "
                    "theta=%.2f, leaf_size=%d)",
                    src_bc, dst_bc,
                    plan.n_near, plan.n_nf, plan.n_fn, plan.n_far_nodes,
                    100.0 * plan.n_near / max(n_src * n_dst, 1),
                    n_src, n_dst, self.theta, self.leaf_size,
                )

        return cluster_trees, bc_areas, comm_plans

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
        from physicsnemo.experimental.models.globe.cluster_tree import ClusterTree

        pred_target_tree = ClusterTree.from_points(
            prediction_points, leaf_size=self.leaf_size,
        )
        pred_plans = {
            bc_type: tree.find_dual_interaction_pairs(
                target_tree=pred_target_tree, theta=self.theta,
                expand_far_targets=self.expand_far_targets,
            )
            for bc_type, tree in cluster_trees.items()
        }

        n_pred = prediction_points.shape[0]
        for bc_type, plan in pred_plans.items():
            n_src = cluster_trees[bc_type].n_sources
            logger.logger.debug(
                "pred plan [%s]: %d near + %d nf + %d fn + %d far_node "
                "(%d sources x %d targets, theta=%.2f)",
                bc_type, plan.n_near, plan.n_nf, plan.n_fn, plan.n_far_nodes,
                n_src, n_pred, self.theta,
            )

        return pred_target_tree, pred_plans

    def _evaluate_hyperlayer(
        self,
        layer_idx: int,
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        source_meshes: dict[str, Mesh["n-1", "n"]],  # ty: ignore[unresolved-reference]
        reference_lengths: dict[str, Float[torch.Tensor, ""]],
        global_data: TensorDict[str, Float[torch.Tensor, "..."]] | None,
        cluster_trees: dict[str, ClusterTree],
        target_tree: ClusterTree | None,
        dual_plans: dict[str, DualInteractionPlan],
        source_areas: dict[str, torch.Tensor],
    ) -> TensorDict[str, Float[torch.Tensor, "n_targets ..."]]:
        r"""Evaluate one hyperlayer by summing kernel contributions from all BC types.

        Each call evaluates all source BC types against a single set of target
        points.  The ``target_tree`` and per-source-BC ``dual_plans`` must
        correspond to those target points.

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
        target_tree : ClusterTree or None
            Precomputed target tree shared by all source BCs in this call.
            For communication self-interaction, this is the destination BC's
            own cluster tree.  If ``None``, each kernel branch builds a tree
            from ``target_points`` on the fly.
        dual_plans : dict[str, DualInteractionPlan]
            Per-source-BC-type precomputed dual interaction plans, each
            computed from that source BC's tree to ``target_tree``.
        source_areas : dict[str, torch.Tensor]
            Per-BC-type source area tensors.

        Returns
        -------
        TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
            Summed kernel outputs across all boundary condition types.
        """
        result_pieces: list[TensorDict[str, Float[torch.Tensor, "n_targets ..."]]] = []

        for bc_type, mesh in source_meshes.items():
            strengths: TensorDict[str, Float[torch.Tensor, " n_cells"]] = (
                mesh.cell_data["strengths"].apply(  # ty: ignore[unresolved-attribute]
                    lambda x: x * (mesh.cell_areas / self.reference_area)
                )
            )

            source_data = _flatten_keys(mesh.cell_data.exclude("strengths"))
            source_data["normals"] = mesh.cell_normals

            kernel: MultiscaleKernel = self.kernel_layers[layer_idx][bc_type]  # ty: ignore[not-subscriptable]
            kernel_result: TensorDict[str, Float[torch.Tensor, "n_targets ..."]] = kernel(
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
            )
            result_pieces.append(_unflatten_keys(kernel_result))

        return reduce(operator.add, result_pieces)

    def _evaluate_communication_hyperlayer(
        self,
        layer_idx: int,
        boundary_meshes: dict[str, Mesh["n-1", "n"]],  # ty: ignore[unresolved-reference]
        reference_lengths: dict[str, Float[torch.Tensor, ""]],
        global_data: TensorDict[str, Float[torch.Tensor, "..."]] | None,
        cluster_trees: dict[str, ClusterTree],
        comm_plans: dict[str, dict[str, DualInteractionPlan]],
        source_areas: dict[str, torch.Tensor],
    ) -> dict[str, Mesh["n-1", "n"]]:  # ty: ignore[unresolved-reference]
        r"""Run one boundary-to-boundary communication step.

        For each destination BC type, evaluates :meth:`_evaluate_hyperlayer`
        at that BC's cell centroids, summing contributions from all source
        BC types.  The target tree for each destination is that BC's own
        cluster tree; for self-interaction (source == destination), this is
        the same object as the source tree.

        Returns
        -------
        dict[str, Mesh["n-1", "n"]]
            Updated boundary meshes with evaluation results merged into
            each mesh's ``cell_data``.
        """
        new_meshes: dict[str, Mesh["n-1", "n"]] = {}  # ty: ignore[unresolved-reference]
        for bc_type, mesh in boundary_meshes.items():
            result_td = self._evaluate_hyperlayer(
                layer_idx=layer_idx,
                target_points=mesh.cell_centroids,
                source_meshes=boundary_meshes,
                reference_lengths=reference_lengths,
                global_data=global_data,
                cluster_trees=cluster_trees,
                target_tree=cluster_trees[bc_type],
                dual_plans=comm_plans[bc_type],
                source_areas=source_areas,
            )
            new_cell_data = TensorDict(
                {"physical": mesh.cell_data["physical"]},
                batch_size=torch.Size([mesh.n_cells]),
                device=mesh.points.device,
            )
            new_cell_data.update(result_td)
            new_meshes[bc_type] = Mesh(
                points=mesh.points,
                cells=mesh.cells,
                cell_data=new_cell_data,
                _cache=mesh._cache,
            )
        return new_meshes

    def forward(
        self,
        prediction_points: Float[torch.Tensor, "n_points n_dims"],
        boundary_meshes: dict[str, Mesh["n-1", "n"]],  # ty: ignore[unresolved-reference]
        reference_lengths: dict[str, torch.Tensor],
        global_data: TensorDict[str, Float[torch.Tensor, "..."]] | None = None,
    ) -> Mesh[0, "n"]:  # ty: ignore[unresolved-reference]
        r"""Evaluate GLOBE model to predict fields at target points.

        Runs the full GLOBE forward pass in three phases:

        1. **Init**: Enrich boundary meshes with initial (all-ones) strengths,
           wrapping original ``cell_data`` under a ``"physical"`` namespace.
        2. **Communication**: Run ``n_communication_hyperlayers`` boundary-to-
           boundary communication steps via
           :meth:`_evaluate_communication_hyperlayer`.
        3. **Final evaluation**: Evaluate the last hyperlayer at
           ``prediction_points`` and apply per-field calibration transforms.

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
            match ``global_data_ranks``. Passed through to the output Mesh.

        Returns
        -------
        Mesh[0, "n"]
            A point-cloud Mesh (0-dimensional manifold) with predicted fields.
        """
        device = prediction_points.device

        if global_data is None:
            global_data = TensorDict({}, device=device)

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
            for bc_type, mesh in boundary_meshes.items():
                if mesh.n_spatial_dims != self.n_spatial_dims:
                    raise ValueError(
                        f"Boundary mesh for BC type {bc_type!r} has "
                        f"{mesh.n_spatial_dims} spatial dims, but the model expects "
                        f"{self.n_spatial_dims}"
                    )
            bc_types_from_input = set(boundary_meshes.keys())
            if not bc_types_from_input.issubset(self.boundary_condition_names):
                raise ValueError(
                    f"The input gives boundary meshes with these boundary condition types:\n"
                    f"{bc_types_from_input!r}\n"
                    f"but the model was instantiated to expect only these boundary condition types:\n"
                    f"{self.boundary_condition_names!r}\n"
                    f"Please ensure that the input boundary meshes are a subset of the model's boundary condition types."
                )

        ### Phase 1: Enrich boundary meshes with initial (all-ones) strengths.
        with record_function("globe::enrich_meshes"):
            boundary_meshes = {
                bc_type: Mesh(
                    points=mesh.points,
                    cells=mesh.cells,
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
                    _cache=mesh._cache,
                )
                for bc_type, mesh in boundary_meshes.items()
            }

        ### Build per-BC-type trees and areas (reused across all layers).
        ### Tree construction and traversal involve irregular control flow
        ### (morton codes, variable-depth loops) that cannot be traced by
        ### torch.compile, so we skip compilation for this block.
        with record_function("globe::build_trees_and_plans"):
            cluster_trees, bc_areas, comm_plans = self._build_trees_and_plans(
                boundary_meshes
            )

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
                    source_areas=bc_areas,
                )

        ### Free comm plans - no longer needed after communication layers.
        # At 800k faces, near-pair indices can be ~3 GB of int64.
        del comm_plans

        ### Phase 3: Final evaluation at prediction points.
        with record_function("globe::build_prediction_plans"):
            pred_target_tree, pred_plans = self._build_prediction_plans(
                cluster_trees, prediction_points
            )

        with record_function("globe::final_evaluation"):
            result: TensorDict[str, Float[torch.Tensor, "n_points ..."]] = self._evaluate_hyperlayer(
                layer_idx=self.n_communication_hyperlayers,
                target_points=prediction_points,
                source_meshes=boundary_meshes,
                reference_lengths=reference_lengths,
                global_data=global_data,
                cluster_trees=cluster_trees,
                target_tree=pred_target_tree,
                dual_plans=pred_plans,
                source_areas=bc_areas,
            )

        del pred_plans, pred_target_tree

        ### Wrap as point-cloud Mesh and apply per-field calibration.
        with record_function("globe::calibration"):
            output_mesh = Mesh(
                points=prediction_points,
                point_data=result,
                global_data=global_data,
            )
            for idx, name in enumerate(self._output_field_order):
                key = tuple(name.split("."))
                t = output_mesh.point_data[key]
                output_mesh.point_data[key] = self.final_field_transforms[idx](
                    t.view(-1, 1)
                ).view(t.shape)
        return output_mesh
