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
from physicsnemo.models.globe.field_kernel import MultiscaleKernel
from physicsnemo.models.globe.utilities.tensordict_utils import (
    combine_tensordicts,
    concatenated_length,
    split_by_leaf_rank,
)


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
    reference_area : torch.Tensor
        Scalar tensor used to nondimensionalize face areas. Typically a characteristic
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
    verbose : bool, optional, default=False
        If ``True``, prints progress information during evaluation.

    Outputs
    -------
    TensorDict
        TensorDict with ``batch_size=(n_points,)`` containing the predicted fields.
        Keys are the field names from ``output_fields``. Scalar fields have shape
        :math:`(N_{points},)`, vector fields have shape
        :math:`(N_{points}, D)`.

    Notes
    -----
    - ``kernel_layers`` is a :class:`~torch.nn.ModuleList` of communication
      hyperlayers, each containing a :class:`~torch.nn.ModuleDict` mapping BC type
      names to :class:`~physicsnemo.models.globe.field_kernel.MultiscaleKernel`
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
    ...     reference_area=torch.tensor(1.0),
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
        reference_area: torch.Tensor,
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
        self.register_buffer("reference_area", reference_area)
        self.n_global_scalars = n_global_scalars
        self.n_global_vectors = n_global_vectors
        self.n_communication_hyperlayers = n_communication_hyperlayers
        self.n_latent_scalars = n_latent_scalars
        self.n_latent_vectors = n_latent_vectors
        self.smoothing_radius = smoothing_radius
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_spherical_harmonics = n_spherical_harmonics

        kernel_layers = []

        for i in range(self.n_communication_hyperlayers + 1):
            is_first_hyperlayer = i == 0
            is_last_hyperlayer = i == self.n_communication_hyperlayers

            kernel_layers.append(
                nn.ModuleDict(
                    {
                        source_bc_type: MultiscaleKernel(
                            n_spatial_dims=n_spatial_dims,
                            output_fields=output_fields
                            if is_last_hyperlayer
                            else {
                                **{
                                    f"strengths.{name}": "scalar"
                                    for name in reference_length_names
                                },
                                **{
                                    f"latent_scalars.{i}": "scalar"
                                    for i in range(n_latent_scalars)
                                },
                                **{
                                    f"latent_vectors.{i}": "vector"
                                    for i in range(n_latent_vectors)
                                },
                            },
                            reference_length_names=reference_length_names,
                            n_source_scalars=(
                                boundary_condition_n_source_scalars[source_bc_type]
                                + (0 if is_first_hyperlayer else n_latent_scalars)
                            ),
                            n_source_vectors=(
                                boundary_condition_n_source_vectors[source_bc_type]
                                + (0 if is_first_hyperlayer else n_latent_vectors)
                                + 1  # +1 for the normal vector
                            ),
                            n_global_scalars=n_global_scalars,
                            n_global_vectors=n_global_vectors,
                            smoothing_radius=smoothing_radius,
                            hidden_layer_sizes=hidden_layer_sizes,
                            n_spherical_harmonics=n_spherical_harmonics,
                        )
                        for source_bc_type in boundary_condition_names
                    }
                )
            )

        self.kernel_layers = nn.ModuleList(kernel_layers)
        self.final_field_transforms = nn.ModuleDict(
            {
                field_name: nn.Linear(
                    in_features=1,
                    out_features=1,
                    bias=(output_fields[field_name] == "scalar"),
                )
                for field_name in output_fields.keys()
            }
        )

    def _evaluate_hyperlayer(
        self,
        layer_idx: int,
        target_points: torch.Tensor,
        latent_data: dict[str, TensorDict],
        boundary_meshes: dict[str, "Mesh"],
        reference_lengths: dict[str, torch.Tensor],
        global_scalars: TensorDict | None,
        global_vectors: TensorDict | None,
        verbose: bool,
        chunk_size: None | int | Literal["auto"],
    ) -> TensorDict:
        """Evaluate one hyperlayer: sum kernel contributions from all BC types."""
        result_pieces: list[TensorDict] = []

        for source_bc_type, source_bc_mesh in boundary_meshes.items():
            source_latent_data: TensorDict = latent_data[source_bc_type]
            source_strengths: TensorDict = source_latent_data["strengths"]  # ty: ignore[invalid-assignment]
            source_strengths = source_strengths.apply(
                lambda x: x * (source_bc_mesh.cell_areas / self.reference_area)
            )

            source_data_by_rank: dict[int, TensorDict] = split_by_leaf_rank(
                source_bc_mesh.cell_data
            )
            source_scalars = combine_tensordicts(
                source_data_by_rank[0],
                source_latent_data["latent_scalars"],  # ty: ignore[invalid-argument-type]
            )
            source_vectors = combine_tensordicts(
                source_data_by_rank[1],
                source_latent_data["latent_vectors"],  # ty: ignore[invalid-argument-type]
            )
            source_vectors["normals"] = source_bc_mesh.cell_normals
            source_vectors.batch_size = torch.Size(
                [source_bc_mesh.n_cells, self.n_spatial_dims]
            )

            kernel: MultiscaleKernel = self.kernel_layers[layer_idx][source_bc_type]
            result_from_kernel: TensorDict = kernel(
                source_points=source_bc_mesh.cell_centroids,
                source_scalars=source_scalars,
                source_vectors=source_vectors,
                source_strengths=source_strengths,
                target_points=target_points,
                reference_lengths=reference_lengths,
                global_scalars=global_scalars,
                global_vectors=global_vectors,
                verbose=verbose,
                chunk_size=chunk_size,
            )
            result_pieces.append(result_from_kernel.unflatten_keys())

        return reduce(operator.add, result_pieces)

    def forward(
        self,
        prediction_points: Float[torch.Tensor, "n_points n_dims"],
        boundary_meshes: dict[str, Mesh],
        reference_lengths: dict[str, torch.Tensor],
        global_scalars: TensorDict | None = None,
        global_vectors: TensorDict | None = None,
        chunk_size: None | int | Literal["auto"] = None,
        verbose: bool = False,
    ) -> TensorDict:
        r"""Evaluate GLOBE model to predict fields at target points.

        Runs the full GLOBE forward pass: communication hyperlayers, final
        evaluation, and per-field calibration. See the class docstring for full
        input/output documentation.

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
        verbose : bool, optional, default=False
            If ``True``, prints progress information.

        Returns
        -------
        TensorDict
            Predicted fields at target points. See class docstring for details.
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
            # Check that input lengths are consistent with the model
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

            # Check that dimensionality of all boundary meshes is consistent
            for bc_type, mesh in boundary_meshes.items():
                if mesh.n_spatial_dims != self.n_spatial_dims:
                    raise ValueError(
                        f"Boundary mesh for BC type {bc_type!r} has "
                        f"{mesh.n_spatial_dims} spatial dims, but the model expects "
                        f"{self.n_spatial_dims}"
                    )

            # Check that all boundary condition types are in the model
            bc_types_from_input = set(boundary_meshes.keys())
            if not bc_types_from_input.issubset(self.boundary_condition_names):
                raise ValueError(
                    f"The input gives boundary meshes with these boundary condition types:\n"
                    f"{bc_types_from_input!r}\n"
                    f"but the model was instantiated to expect only these boundary condition types:\n"
                    f"{self.boundary_condition_names!r}\n"
                    f"Please ensure that the input boundary meshes are a subset of the model's boundary condition types."
                )

        ### Initialize the latent data that's passed between hyperlayers
        latent_data: dict[str, TensorDict] = {
            source_bc_type: TensorDict(
                {
                    "strengths": {
                        name: torch.ones(source_bc_mesh.n_cells, device=device)
                        for name in self.reference_length_names
                    },
                    "latent_scalars": {},
                    "latent_vectors": {},
                },
                batch_size=torch.Size([source_bc_mesh.n_cells]),
                device=device,
            )
            for source_bc_type, source_bc_mesh in boundary_meshes.items()
        }

        ### Kernel evaluations
        for i in range(self.n_communication_hyperlayers + 1):
            if verbose:
                print(f"Evaluating hypernetwork layer {i}...")

            is_last_hyperlayer = i == self.n_communication_hyperlayers

            if not is_last_hyperlayer:
                latent_data: TensorDict = {
                    target_bc_type: self._evaluate_hyperlayer(
                        i,
                        target_bc_mesh.cell_centroids,
                        latent_data,
                        boundary_meshes,
                        reference_lengths,
                        global_scalars,
                        global_vectors,
                        verbose,
                        chunk_size,
                    )
                    for target_bc_type, target_bc_mesh in boundary_meshes.items()
                }
            else:
                result: TensorDict = self._evaluate_hyperlayer(
                    i,
                    prediction_points,
                    latent_data,
                    boundary_meshes,
                    reference_lengths,
                    global_scalars,
                    global_vectors,
                    verbose,
                    chunk_size,
                )

        for field_name, field_tensor in result.items():
            original_shape = field_tensor.shape
            result[field_name] = self.final_field_transforms[field_name](
                field_tensor.view(-1, 1)
            ).view(original_shape)

        return result
