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
from typing import Literal, Sequence

import torch
import torch.nn as nn
from jaxtyping import Float
from tensordict import TensorDict

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.mesh import Mesh
from physicsnemo.utils.logging import PythonLogger

from physicsnemo.experimental.models.globe.field_kernel import MultiscaleKernel
from physicsnemo.experimental.models.globe.utilities.tensordict_utils import (
    concatenated_length,
    split_by_leaf_rank,
)

logger = PythonLogger("globe.model")


@dataclass
class MetaData(ModelMetaData):
    jit: bool = True
    cuda_graphs: bool = True
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

    Parameters
    ----------
    n_spatial_dims : int
        Number of spatial dimensions (2 or 3).
    output_fields : dict[str, Literal["scalar", "vector"]]
        Dictionary mapping field names to types (``"scalar"`` or ``"vector"``).
        These are the physical fields predicted at query points in the final
        hyperlayer.
    boundary_condition_names : Sequence[str]
        Sequence of boundary condition type identifiers
        (e.g., ``["no_slip", "freestream", "slip"]``). Each BC type gets its own
        set of kernels. Names must not contain ``"."`` for TensorDict compatibility.
    boundary_condition_n_source_scalars : dict[str, int]
        Dictionary mapping each BC type name to the number of scalar features per
        boundary face for that BC type.
    boundary_condition_n_source_vectors : dict[str, int]
        Dictionary mapping each BC type name to the number of vector features per
        boundary face for that BC type. The face normal vector is automatically
        added, so don't include it in the count.
    reference_length_names : Sequence[str]
        Sequence of identifiers for reference length scales
        (e.g., ``["viscous_length", "chord_length"]``). Each creates a separate
        kernel branch in the multiscale composition.
    reference_area : float
        Scalar used to nondimensionalize face areas. Typically a characteristic
        area of the problem (e.g., chord^2 for airfoils).
    n_global_scalars : int
        Number of global scalar features (e.g., Reynolds number, Mach number). These
        are shared across all source faces.
    n_global_vectors : int
        Number of global vector features (e.g., freestream velocity direction). These
        are shared across all source faces.
    n_communication_hyperlayers : int, optional, default=2
        Number of boundary-to-boundary communication layers before final evaluation.
        Higher values enable more information exchange between boundary partitions.
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

    Forward
    -------
    prediction_points : Float[torch.Tensor, "n_points n_dims"]
        Target points for field evaluation of shape :math:`(N_{points}, D)`. These
        are typically interior domain points where the solution is predicted.
    boundary_meshes : dict[str, Mesh]
        Dictionary mapping boundary condition type names to
        :class:`~physicsnemo.mesh.Mesh` objects. Each key must match one of the
        model's ``boundary_condition_names``. Each Mesh should be pre-merged
        (one Mesh per BC type); use :meth:`~physicsnemo.mesh.Mesh.merge` to
        combine multiple meshes of the same BC type before passing them here.
        Cell data (``mesh.cell_data``) should contain only the source features
        expected by the model.
    reference_lengths : dict[str, torch.Tensor]
        Dictionary mapping reference length names to scalar tensors. Keys must match
        the model's ``reference_length_names``.
    global_scalars : TensorDict | None, optional, default=None
        TensorDict with ``batch_size=()`` containing problem-level scalar features.
        The total concatenated length must match ``n_global_scalars``.
    global_vectors : TensorDict | None, optional, default=None
        TensorDict with ``batch_size=(n_spatial_dims,)`` containing problem-level
        vector features. The total concatenated length must match ``n_global_vectors``.
    chunk_size : None | int | Literal["auto"], optional, default=None
        Controls memory usage during kernel evaluation. ``None`` evaluates all target
        points at once, an ``int`` processes in chunks of that size, and ``"auto"``
        automatically determines chunk size targeting ~1GB per chunk.

    Outputs
    -------
    Mesh
        A point-cloud :class:`~physicsnemo.mesh.Mesh` (0-dimensional manifold)
        with ``points`` equal to the input ``prediction_points``. The predicted
        fields are in ``point_data``, keyed by the names from ``output_fields``.
        Scalar fields have shape :math:`(N_{points},)`, vector fields have shape
        :math:`(N_{points}, D)`. Cells are empty (shape ``(0, 1)``).

    Notes
    -----
    - ``kernel_layers`` is a :class:`~torch.nn.ModuleList` of communication
      hyperlayers, each containing a :class:`~torch.nn.ModuleDict` mapping BC type
      names to :class:`~physicsnemo.experimental.models.globe.field_kernel.MultiscaleKernel`
      instances.
    - ``final_field_transforms`` is a :class:`~torch.nn.ModuleDict` of per-field
      linear calibration layers.
    - Cell areas are automatically normalized by ``reference_area`` to preserve
      discretization-invariance.
    - The cell normal vector is automatically added to ``source_vectors`` for each
      mesh.
    - Hyperlayer communication enables long-range coupling between boundary
      partitions, analogous to the influence coefficient matrix solve in traditional
      boundary-element methods (but learned rather than explicitly computed).

    Examples
    --------
    >>> model = GLOBE(
    ...     n_spatial_dims=3,
    ...     output_fields={"pressure": "scalar", "velocity": "vector"},
    ...     boundary_condition_names=["no_slip", "freestream"],
    ...     boundary_condition_n_source_scalars={"no_slip": 0, "freestream": 0},
    ...     boundary_condition_n_source_vectors={"no_slip": 0, "freestream": 0},
    ...     reference_length_names=["delta_FS", "chord"],
    ...     reference_area=1.0,
    ...     n_global_scalars=0,
    ...     n_global_vectors=0,
    ... )
    >>> result = model(
    ...     prediction_points=torch.randn(100, 3),
    ...     boundary_meshes={"no_slip": wing_mesh, "freestream": freestream_mesh},
    ...     reference_lengths={"delta_FS": torch.tensor(0.01), "chord": torch.tensor(1.0)},
    ... )
    """

    def __init__(
        self,
        n_spatial_dims: int,
        output_fields: dict[str, Literal["scalar", "vector"]],
        boundary_condition_names: Sequence[str],
        boundary_condition_n_source_scalars: dict[str, int],
        boundary_condition_n_source_vectors: dict[str, int],
        reference_length_names: Sequence[str],
        reference_area: float,
        n_global_scalars: int,
        n_global_vectors: int,
        n_communication_hyperlayers: int = 2,
        n_latent_scalars: int = 12,
        n_latent_vectors: int = 6,
        smoothing_radius: float = 1e-8,
        hidden_layer_sizes: Sequence[int] | None = None,
        n_spherical_harmonics: int = 4,
    ):
        ### Sets defaults
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64, 64, 64]

        # Validate inputs
        if not torch.compiler.is_compiling():
            for field_name, field_type in output_fields.items():
                if field_type not in ["scalar", "vector"]:
                    raise ValueError(
                        f"In `output_fields`, for {field_name=!r}, got {field_type=!r};\n"
                        "This must be one of ['scalar', 'vector']"
                    )
            for bc_name in boundary_condition_names:
                if "." in bc_name:
                    raise ValueError(
                        f"In `boundary_condition_names`, got {bc_name=!r};\n"
                        "This must not contain any `.` characters, for nested-TensorDict compatibility."
                    )

        super().__init__(meta=MetaData())

        # Store arguments
        self.n_spatial_dims = n_spatial_dims
        self.output_fields = output_fields
        self.boundary_condition_names = boundary_condition_names
        self.boundary_condition_n_source_scalars = boundary_condition_n_source_scalars
        self.boundary_condition_n_source_vectors = boundary_condition_n_source_vectors
        self.reference_length_names = reference_length_names
        self.register_buffer("reference_area", torch.tensor(reference_area))
        self.n_global_scalars = n_global_scalars
        self.n_global_vectors = n_global_vectors
        self.n_communication_hyperlayers = n_communication_hyperlayers
        self.n_latent_scalars = n_latent_scalars
        self.n_latent_vectors = n_latent_vectors
        self.smoothing_radius = smoothing_radius
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_spherical_harmonics = n_spherical_harmonics

        ### Build the intermediate output-field spec for communication hyperlayers.
        # These latent fields carry information between hyperlayers; only the
        # final hyperlayer emits the user-requested output_fields.
        intermediate_fields: dict[str, Literal["scalar", "vector"]] = {
            f"strengths.{name}": "scalar" for name in reference_length_names
        } | {
            f"latent.scalars.{i}": "scalar" for i in range(n_latent_scalars)
        } | {
            f"latent.vectors.{i}": "vector" for i in range(n_latent_vectors)
        }

        kernel_layers = []

        for layer_idx in range(self.n_communication_hyperlayers + 1):
            is_first_hyperlayer = layer_idx == 0
            is_last_hyperlayer = layer_idx == self.n_communication_hyperlayers

            layer = nn.ModuleDict({
                bc_type: MultiscaleKernel(
                    n_spatial_dims=n_spatial_dims,
                    output_fields=(
                        output_fields if is_last_hyperlayer else intermediate_fields
                    ),
                    reference_length_names=reference_length_names,
                    n_source_scalars=(
                        boundary_condition_n_source_scalars[bc_type]
                        + (0 if is_first_hyperlayer else n_latent_scalars)
                    ),
                    n_source_vectors=(
                        boundary_condition_n_source_vectors[bc_type]
                        + (0 if is_first_hyperlayer else n_latent_vectors)
                        + 1  # +1 for the normal vector
                    ),
                    n_global_scalars=n_global_scalars,
                    n_global_vectors=n_global_vectors,
                    smoothing_radius=smoothing_radius,
                    hidden_layer_sizes=hidden_layer_sizes,
                    n_spherical_harmonics=n_spherical_harmonics,
                )
                for bc_type in boundary_condition_names
            })
            kernel_layers.append(layer)

        self.kernel_layers = nn.ModuleList(kernel_layers)

        # Per-field learnable affine calibration (y = a*x + b). Bias is only
        # applied to scalar fields; adding bias to vector fields would break
        # rotational equivariance.
        self.final_field_transforms = nn.ModuleDict({
            field_name: nn.Linear(
                in_features=1,
                out_features=1,
                bias=(output_fields[field_name] == "scalar"),
            )
            for field_name in output_fields
        })

    def _evaluate_hyperlayer(
        self,
        layer_idx: int,
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        source_meshes: dict[str, Mesh],
        reference_lengths: dict[str, Float[torch.Tensor, ""]],
        global_scalars: TensorDict | None,
        global_vectors: TensorDict | None,
        chunk_size: None | int | Literal["auto"],
    ) -> TensorDict:
        r"""Evaluate one hyperlayer by summing kernel contributions from all BC types.

        For each boundary condition type, extracts source data from the mesh's
        enriched ``cell_data``, evaluates the corresponding
        :class:`MultiscaleKernel`, and sums the results.

        Each mesh's ``cell_data`` carries a namespaced structure:

        - ``"physical"``: original boundary condition features
        - ``"strengths"``: per-reference-length kernel strengths
        - ``"latent"``: (after first layer) learned scalar and vector features

        Strengths are extracted and area-normalized separately. All remaining
        features are flattened and split by tensor rank into scalars and vectors
        for the kernel.

        Parameters
        ----------
        layer_idx : int
            Index into ``self.kernel_layers`` selecting which hyperlayer to evaluate.
        target_points : Float[torch.Tensor, "n_targets n_dims"]
            Target points of shape :math:`(N_{targets}, D)`.
        source_meshes : dict[str, Mesh]
            Mapping of BC type names to enriched :class:`~physicsnemo.mesh.Mesh`
            objects whose ``cell_data`` carries both physical features and latent
            state.
        reference_lengths : dict[str, Float[torch.Tensor, ""]]
            Mapping of reference length names to scalar tensors.
        global_scalars : TensorDict or None
            Problem-level scalar features.
        global_vectors : TensorDict or None
            Problem-level vector features.
        chunk_size : None or int or {"auto"}
            Controls memory usage during kernel evaluation.

        Returns
        -------
        TensorDict
            Summed kernel outputs across all boundary condition types.
        """
        result_pieces: list[TensorDict] = []

        for bc_type, mesh in source_meshes.items():
            strengths: TensorDict = mesh.cell_data["strengths"]  # ty: ignore[invalid-assignment]
            strengths = strengths.apply(
                lambda x: x * (mesh.cell_areas / self.reference_area)
            )

            ### Split non-strength features by tensor rank (scalars vs vectors).
            # flatten_keys ensures split_by_leaf_rank produces flat TensorDicts.
            features = mesh.cell_data.exclude("strengths").flatten_keys(".")
            data_by_rank = split_by_leaf_rank(features)
            scalars = data_by_rank[0]
            vectors = data_by_rank[1]
            vectors["normals"] = mesh.cell_normals
            vectors.batch_size = torch.Size([mesh.n_cells, self.n_spatial_dims])

            kernel: MultiscaleKernel = self.kernel_layers[layer_idx][bc_type]
            kernel_result: TensorDict = kernel(
                source_points=mesh.cell_centroids,
                source_scalars=scalars,
                source_vectors=vectors,
                source_strengths=strengths,
                target_points=target_points,
                reference_lengths=reference_lengths,
                global_scalars=global_scalars,
                global_vectors=global_vectors,
                chunk_size=chunk_size,
            )
            result_pieces.append(kernel_result.unflatten_keys())

        return reduce(operator.add, result_pieces)

    def _evaluate_communication_hyperlayer(
        self,
        layer_idx: int,
        boundary_meshes: dict[str, Mesh],
        reference_lengths: dict[str, Float[torch.Tensor, ""]],
        global_scalars: TensorDict | None,
        global_vectors: TensorDict | None,
        chunk_size: None | int | Literal["auto"],
    ) -> dict[str, Mesh]:
        r"""Run one boundary-to-boundary communication step.

        For each BC type, evaluates :meth:`_evaluate_hyperlayer` at the mesh's
        cell centroids and wraps the result into an enriched Mesh that carries
        both the original physical ``cell_data`` (under ``"physical"``) and the
        new latent state (``"strengths"``, ``"latent"``).

        Geometry tensors and cached properties (centroids, areas, normals) are
        shared by reference across layers - no copies are made.

        Parameters
        ----------
        layer_idx : int
            Index into ``self.kernel_layers`` for this communication layer.
        boundary_meshes : dict[str, Mesh]
            Current enriched boundary meshes (from the previous layer or init).
        reference_lengths : dict[str, Float[torch.Tensor, ""]]
            Mapping of reference length names to scalar tensors.
        global_scalars : TensorDict or None
            Problem-level scalar features.
        global_vectors : TensorDict or None
            Problem-level vector features.
        chunk_size : None or int or {"auto"}
            Controls memory usage during kernel evaluation.

        Returns
        -------
        dict[str, Mesh]
            New enriched boundary meshes for the next layer.
        """
        new_meshes: dict[str, Mesh] = {}
        for bc_type, mesh in boundary_meshes.items():
            result_td = self._evaluate_hyperlayer(
                layer_idx=layer_idx,
                target_points=mesh.cell_centroids,
                source_meshes=boundary_meshes,
                reference_lengths=reference_lengths,
                global_scalars=global_scalars,
                global_vectors=global_vectors,
                chunk_size=chunk_size,
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
        boundary_meshes: dict[str, Mesh],
        reference_lengths: dict[str, torch.Tensor],
        global_scalars: TensorDict | None = None,
        global_vectors: TensorDict | None = None,
        chunk_size: None | int | Literal["auto"] = None,
    ) -> Mesh:
        r"""Evaluate GLOBE model to predict fields at target points.

        Runs the full GLOBE forward pass in three phases:

        1. **Init**: Enrich boundary meshes with initial (all-ones) strengths,
           wrapping original ``cell_data`` under a ``"physical"`` namespace.
        2. **Communication**: Run ``n_communication_hyperlayers`` boundary-to-
           boundary communication steps via
           :meth:`_evaluate_communication_layer`.
        3. **Final evaluation**: Evaluate the last hyperlayer at
           ``prediction_points`` and apply per-field calibration transforms.

        Parameters
        ----------
        prediction_points : Float[torch.Tensor, "n_points n_dims"]
            Target points of shape :math:`(N_{points}, D)`.
        boundary_meshes : dict[str, Mesh]
            Dictionary mapping BC type names to pre-merged
            :class:`~physicsnemo.mesh.Mesh` objects.
        reference_lengths : dict[str, torch.Tensor]
            Mapping of reference length names to scalar tensors.
        global_scalars : TensorDict | None, optional, default=None
            Problem-level scalar features.
        global_vectors : TensorDict | None, optional, default=None
            Problem-level vector features.
        chunk_size : None | int | Literal["auto"], optional, default=None
            Controls memory usage during kernel evaluation.

        Returns
        -------
        Mesh
            A point-cloud Mesh (0-dimensional manifold) with:

            - ``points``: the input ``prediction_points``
            - ``point_data``: calibrated output fields (keys from
              ``output_fields``)
            - ``cells``: empty (shape ``(0, 1)``)
            - ``cell_data``: empty
        """
        device = prediction_points.device

        ### Set defaults
        if global_scalars is None:
            global_scalars = TensorDict({}, batch_size=torch.Size([]), device=device)
        if global_vectors is None:
            global_vectors = TensorDict(
                {}, batch_size=torch.Size([self.n_spatial_dims]), device=device
            )

        ### Input validation
        # Skip validation when running under torch.compile for performance
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
            for name, (actual, expected) in {
                "reference lengths": (
                    set(reference_lengths.keys()),
                    set(self.reference_length_names),
                ),
                "global scalars": (
                    concatenated_length(global_scalars),
                    self.n_global_scalars,
                ),
                "global vectors": (
                    concatenated_length(global_vectors),
                    self.n_global_vectors,
                ),
            }.items():
                if actual != expected:
                    raise ValueError(
                        f"This model was instantiated to expect {expected} {name},\n"
                        f"but the forward-method input gives {actual} {name}."
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
        # Wraps original cell_data under "physical" and adds "strengths".
        # Geometry tensors are shared by reference - no copies.
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

        ### Phase 2: Communication hyperlayers (boundary-to-boundary).
        for i in range(self.n_communication_hyperlayers):
            boundary_meshes = self._evaluate_communication_hyperlayer(
                layer_idx=i,
                boundary_meshes=boundary_meshes,
                reference_lengths=reference_lengths,
                global_scalars=global_scalars,
                global_vectors=global_vectors,
                chunk_size=chunk_size,
            )

        ### Phase 3: Final evaluation at prediction points.
        result: TensorDict = self._evaluate_hyperlayer(
            layer_idx=self.n_communication_hyperlayers,
            target_points=prediction_points,
            source_meshes=boundary_meshes,
            reference_lengths=reference_lengths,
            global_scalars=global_scalars,
            global_vectors=global_vectors,
            chunk_size=chunk_size,
        )

        ### Per-field calibration transforms
        for field_name, field_tensor in result.items():
            original_shape = field_tensor.shape
            result[field_name] = self.final_field_transforms[field_name](
                field_tensor.view(-1, 1)
            ).view(original_shape)

        return Mesh(
            points=prediction_points,
            point_data=result,
        )
