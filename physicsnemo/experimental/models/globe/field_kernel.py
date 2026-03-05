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

import itertools
import logging
import operator
from functools import cached_property, reduce
from math import ceil, comb, prod
from typing import Literal, Sequence

import torch
import torch.nn as nn
import tqdm
from jaxtyping import Float
from tensordict import TensorDict
from torch.utils.checkpoint import checkpoint

from physicsnemo.core.module import Module
from physicsnemo.experimental.models.globe.utilities.rank_spec import (
    RankSpecDict,
    flatten_rank_spec,
    rank_counts,
)
from physicsnemo.experimental.models.globe.utilities.tensordict_utils import (
    concatenate_leaves,
    concatenated_length,
    split_by_leaf_rank,
)
from physicsnemo.nn import Mlp, Pade
from physicsnemo.nn.functional.equivariant_ops import (
    legendre_polynomials,
    polar_and_dipole_basis,
    smooth_log,
    spherical_basis,
)
from physicsnemo.utils.logging import PythonLogger

logger = PythonLogger("globe.field_kernel")


class Kernel(Module):
    r"""A kernel function for evaluating scalar and vector fields from source points.

    This class implements a learnable neural-network-based kernel function that
    computes scalar and vector fields at target points based on the influence of
    source points with associated normals and strengths. The kernel uses a Pade
    rational neural network to model the field interactions while preserving
    physical properties such as proper far-field decay rates, translational
    invariance, rotational invariance, parity invariance, and scale invariance.

    The kernel takes as input the relative positions, orientations, and magnitudes
    of source points, then outputs field values that are consistent with physical
    conservation laws. For vector fields, the output is automatically reprojected
    onto a local coordinate system to maintain rotational invariance.

    Parameters
    ----------
    n_spatial_dims : int
        Number of spatial dimensions (2 or 3).
    output_field_ranks : TensorDict
        Rank-spec TensorDict with integer leaves (0 = scalar, 1 = vector)
        describing the output fields. Nesting is supported and mirrors the
        desired output structure. Derive from data via
        :func:`ranks_from_tensordict`.
    source_data_ranks : TensorDict
        Rank-spec TensorDict describing per-source features. The number of rank-0 leaves determines scalar input
        width; rank-1 leaves determine vector input width.
    global_data_ranks : TensorDict
        Rank-spec TensorDict describing global conditioning features.
    smoothing_radius : float, optional, default=1e-8
        Small value used to smooth power functions near zero to avoid numerical
        instabilities.
    hidden_layer_sizes : Sequence[int] or None, optional, default=None
        Sequence of hidden layer sizes for the neural network. When ``None``,
        defaults to ``[64]``.
    n_spherical_harmonics : int, optional, default=4
        Number of spherical harmonic terms to use as features.
    network_type : {"pade", "mlp"}, optional, default="pade"
        Type of neural network to use for the kernel function.
    spectral_norm : bool, optional, default=False
        Whether to apply spectral normalization to network weights.
    use_gradient_checkpointing : bool, optional, default=True
        If ``True``, applies ``torch.utils.checkpoint.checkpoint`` during
        training to trade compute for memory. Disable for small models or
        when profiling.

    Forward
    -------
    reference_length : Float[torch.Tensor, ""]
        Scalar reference length scale used to convert position-based features
        into dimensionless quantities.
    source_points : Float[torch.Tensor, "n_sources n_dims"]
        Physical coordinates of the source points, which are the centers of
        the influence fields. Shape :math:`(N_{sources}, D)`.
    target_points : Float[torch.Tensor, "n_targets n_dims"]
        Physical coordinates of the target points where the field is evaluated.
        Shape :math:`(N_{targets}, D)`.
    source_strengths : Float[torch.Tensor, "n_sources"] or None, optional, default=None
        Scalar strength values associated with each source point. Shape
        :math:`(N_{sources},)`. Defaults to all ones if ``None``.
    source_data : TensorDict or None, optional, default=None
        Per-source features with ``batch_size=(N_sources,)``. Contains a mix
        of scalar (rank-0) and vector (rank-1) tensors; the kernel splits
        them internally via :func:`split_by_leaf_rank`. Leaf keys and ranks
        must match ``source_data_ranks``. All values must be dimensionless.
    global_data : TensorDict or None, optional, default=None
        Problem-level features with ``batch_size=()``. Contains a mix of
        scalar (rank-0) and vector (rank-1) tensors; split internally.
        Leaf keys and ranks must match ``global_data_ranks``. All values
        must be dimensionless.

    Outputs
    -------
    TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
        TensorDict with batch_size :math:`(N_{targets},)` containing the computed
        fields. Each scalar field has shape :math:`(N_{targets},)` and each vector
        field has shape :math:`(N_{targets}, D)`.
    """

    def __init__(
        self,
        *,
        n_spatial_dims: int,
        output_field_ranks: RankSpecDict,
        source_data_ranks: RankSpecDict | None = None,
        global_data_ranks: RankSpecDict | None = None,
        smoothing_radius: float = 1e-8,
        hidden_layer_sizes: Sequence[int] | None = None,
        n_spherical_harmonics: int = 4,
        network_type: Literal["pade", "mlp"] = "pade",
        spectral_norm: bool = False,
        use_gradient_checkpointing: bool = True,
    ):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64]
        if source_data_ranks is None:
            source_data_ranks = {}
        if global_data_ranks is None:
            global_data_ranks = {}

        super().__init__()

        self.n_spatial_dims = n_spatial_dims
        self.output_field_ranks = output_field_ranks
        self.source_data_ranks = source_data_ranks
        self.global_data_ranks = global_data_ranks
        self.smoothing_radius = smoothing_radius
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_spherical_harmonics = n_spherical_harmonics
        self.use_gradient_checkpointing = use_gradient_checkpointing

        in_features = self.network_in_features
        hidden_features = list(self.hidden_layer_sizes)
        out_features = self.network_out_features

        if network_type == "pade":
            self.network = Pade(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                spectral_norm=spectral_norm,
                numerator_order=2,
                denominator_order=2,
                use_separate_mlps=False,
                share_denominator_across_channels=False,
            )
        elif network_type == "mlp":
            self.network = nn.Sequential(
                Mlp(
                    in_features=in_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    spectral_norm=spectral_norm,
                    act_layer=nn.SiLU(),
                    final_dropout=False,
                ),
                nn.Tanh(),
            )
        else:
            raise ValueError(
                f"Invalid network type: {network_type=!r}; must be one of ['pade', 'mlp']"
            )

    @cached_property
    def _floats_per_interaction(self) -> int:
        """Identifiable float allocations per (target, source) interaction.

        Counts tensor elements from feature engineering, MLP evaluation,
        and post-processing that coexist at peak during ``Kernel.forward``.
        Used by :class:`ChunkedKernel` to estimate chunk memory budgets.

        This is a lower bound - the actual peak is higher due to autograd
        saving input tensors for backward through each element-wise
        operation.  The caller applies a runtime multiplier to account for
        this (see ``ChunkedKernel.forward``).
        """
        source_rc = rank_counts(self.source_data_ranks)
        global_rc = rank_counts(self.global_data_ranks)
        n_vec = 1 + source_rc[1] + global_rc[1]
        n_pairs = comb(n_vec, 2)

        return (
            ### Feature engineering: spatial vectors (n_targets, n_sources, 3, ...)
            3                                                  # r = target - source
            + 3 * n_vec * 2                                    # vectors + unit vectors
            ### Feature engineering: scalars (n_targets, n_sources, ...)
            + n_vec * 3                                        # magnitudes: squared, raw, log
            + n_pairs * (1 + 2 * self.n_spherical_harmonics)   # cos_theta + harmonics + products
            + self.network_in_features                         # concatenated MLP input
            ### MLP layers (sequential; peak is largest layer plus I/O)
            + self.network_in_features
            + sum(self.hidden_layer_sizes)
            + self.network_out_features
            ### Post-processing
            + self.network_out_features                        # reshaped output
            + 1                                                # far-field r_mag_sq
            + self.n_spatial_dims * max(1, 2 * n_vec - 1)      # basis vectors
        )

    @cached_property
    def network_in_features(self) -> int:
        r"""Number of input features for the kernel's internal network.

        Derived from the invariant feature engineering pipeline (Section 3.2.2):

        1. Raw source and global scalars
        2. Smoothed log-magnitudes of all input vectors (relative position ``r``,
           source vectors, global vectors)
        3. Pairwise spherical harmonic features for all :math:`\binom{n}{2}` vector
           pairs, each producing ``n_spherical_harmonics`` Legendre polynomial terms
        """
        source_rank_counts = rank_counts(self.source_data_ranks)
        global_rank_counts = rank_counts(self.global_data_ranks)

        n_vectors_in: int = (
            1 + source_rank_counts[1] + global_rank_counts[1]
        )  # +1 for r
        n_scalars_in: int = source_rank_counts[0] + global_rank_counts[0]
        n_vector_pairs_in: int = comb(n_vectors_in, 2)

        return (
            n_scalars_in + n_vectors_in + n_vector_pairs_in * self.n_spherical_harmonics
        )

    @cached_property
    def network_out_features(self) -> int:
        r"""Number of output features for the kernel's internal network.

        One channel per scalar output field, plus vector reprojection coefficients
        for each vector output field (1 radial + 2 per non-radial input vector).
        """
        source_rank_counts = rank_counts(self.source_data_ranks)
        global_rank_counts = rank_counts(self.global_data_ranks)
        output_rank_counts = rank_counts(self.output_field_ranks)
        n_vectors_in: int = (
            1 + source_rank_counts[1] + global_rank_counts[1]
        )  # +1 for r

        return output_rank_counts[0] + output_rank_counts[1] * (
            1  # r_hat
            + 2 * (n_vectors_in - 1)  # All non-r vectors
        )

    def add_semantics(
        self,
        tensor: Float[torch.Tensor, "... total_dims"],
        shape_for_scalars: torch.Size | None = None,
        shape_for_vectors: torch.Size | None = None,
    ) -> TensorDict[str, Float[torch.Tensor, "..."]]:
        r"""Adds semantics to a tensor by splitting it into named fields.

        The input tensor is assumed to have its last dimension of size equal to the sum
        of the flattened dimensions of all output fields. This function separates the
        tensor into its constituent fields according to the model's output field
        definitions, maintaining the proper shapes for scalar and vector fields.

        Parameters
        ----------
        tensor : Float[torch.Tensor, "... total_dims"]
            Tensor with shape :math:`(\ldots, D_{total})` where :math:`D_{total}` is
            the sum of ``prod(shape)`` for all output fields.
        shape_for_scalars : torch.Size or None, optional
            Shape to use for scalar fields. If ``None``, defaults to ``()``.
        shape_for_vectors : torch.Size or None, optional
            Shape to use for vector fields. If ``None``, defaults to
            :math:`(D,)`.

        Returns
        -------
        TensorDict[str, Float[torch.Tensor, "..."]]
            TensorDict with ``batch_size`` matching ``tensor.shape[:-1]``,
            containing the separated fields with proper shapes. Each scalar field
            has shape :math:`(\ldots, S)` and each vector field has shape
            :math:`(\ldots, V)` where :math:`S` and :math:`V` are determined by
            ``shape_for_scalars`` and ``shape_for_vectors`` respectively.

        Raises
        ------
        ValueError
            If the size of the last dimension does not match the expected total
            number of flattened output dimensions.
        """
        if shape_for_scalars is None:
            shape_for_scalars = torch.Size([])
        if shape_for_vectors is None:
            shape_for_vectors = torch.Size([self.n_spatial_dims])

        shapes_by_rank: dict[int, torch.Size] = {
            0: shape_for_scalars,
            1: shape_for_vectors,
        }

        ranks_dict = flatten_rank_spec(self.output_field_ranks)
        output_field_shapes: dict[str, torch.Size] = {
            field_name: shapes_by_rank[rank]
            for field_name, rank in ranks_dict.items()
        }

        if not torch.compiler.is_compiling():
            if not tensor.shape[-1] == sum(
                prod(shape) for shape in output_field_shapes.values()
            ):
                raise ValueError(
                    f"Expected an array with length {sum(prod(shape) for shape in output_field_shapes.values())} along dimension -1;\n"
                    f"got {tensor.shape=!r}."
                )

        batch_size = tensor.shape[:-1]

        ### Split the flat tensor into per-field views
        fields: dict[str, torch.Tensor] = {}
        i: int = 0
        for field_name, shape in sorted(output_field_shapes.items()):
            field_width = prod(shape)
            slc = [slice(None)] * (tensor.ndim - 1) + [slice(i, i + field_width)]
            fields[field_name] = tensor[tuple(slc)].reshape(batch_size + shape)
            i += field_width
        return TensorDict(fields, batch_size=batch_size)

    def forward(
        self,
        *,
        reference_length: Float[torch.Tensor, ""],
        source_points: Float[torch.Tensor, "n_sources n_dims"],
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        source_strengths: Float[torch.Tensor, " n_sources"] | None = None,
        source_data: TensorDict | None = None,
        global_data: TensorDict | None = None,
    ) -> TensorDict[str, Float[torch.Tensor, "n_targets ..."]]:
        r"""Evaluates a field kernel at target points based on source point influences.

        Parameters
        ----------
        reference_length : Float[torch.Tensor, ""]
            Scalar tensor, shape :math:`()`. The reference length scale used
            to convert position-based features into dimensionless quantities.
        source_points : Float[torch.Tensor, "n_sources n_dims"]
            Tensor of shape :math:`(N_{sources}, D)`. The physical coordinates
            of the source points, which are the centers of the influence fields.
        target_points : Float[torch.Tensor, "n_targets n_dims"]
            Tensor of shape :math:`(N_{targets}, D)`. The physical coordinates
            of the target points where the field is evaluated.
        source_strengths : Float[torch.Tensor, "n_sources"] or None, optional
            Tensor of shape :math:`(N_{sources},)`. Scalar strength values
            associated with each source point. Defaults to all ones if ``None``.
        source_data : TensorDict or None, optional
            Per-source features with ``batch_size=(N_sources,)``. Contains a
            mix of scalar (rank-0) and vector (rank-1) tensors, split
            internally via :func:`split_by_leaf_rank`. Scalar count must
            match ``n_source_scalars``; vector count must match
            ``n_source_vectors``. All values must be dimensionless.
            ``None`` (the default) indicates no per-source features; an empty
            TensorDict is used internally.
        global_data : TensorDict or None, optional
            Problem-level features with ``batch_size=()``. Contains a mix of
            scalar (rank-0) and vector (rank-1) tensors, split internally.
            Scalar count must match ``n_global_scalars``; vector count must
            match ``n_global_vectors``. All values must be dimensionless.
            ``None`` (the default) indicates no global conditioning; an empty
            TensorDict is used internally.

        Returns
        -------
        TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
            TensorDict with batch_size :math:`(N_{targets},)` containing the computed
            fields. Each scalar field has shape :math:`(N_{targets},)` and each vector
            field has shape :math:`(N_{targets}, D)`.
        """
        n_sources: int = len(source_points)
        n_targets: int = len(target_points)
        device = source_points.device

        ### Set defaults
        if source_strengths is None:
            source_strengths = torch.ones(n_sources, device=device)
        if source_data is None:
            source_data = TensorDict({}, batch_size=[n_sources], device=device)
        if global_data is None:
            global_data = TensorDict({}, device=device)

        ### Split by tensor rank for equivariant feature engineering
        source_by_rank = split_by_leaf_rank(source_data)
        source_scalars = source_by_rank[0]
        source_vectors = source_by_rank[1]
        source_vectors.batch_size = torch.Size([n_sources, self.n_spatial_dims])

        global_by_rank = split_by_leaf_rank(global_data)
        global_scalars = global_by_rank[0]
        global_vectors = global_by_rank[1]
        global_vectors.batch_size = torch.Size([self.n_spatial_dims])

        ### Input validation
        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            if source_points.ndim != 2:
                raise ValueError(
                    f"Expected source_points to be 2-dimensional, "
                    f"got {source_points.ndim}D tensor with shape {source_points.shape}"
                )
            if target_points.ndim != 2:
                raise ValueError(
                    f"Expected target_points to be 2-dimensional, "
                    f"got {target_points.ndim}D tensor with shape {target_points.shape}"
                )
            if source_points.shape[-1] != self.n_spatial_dims:
                raise ValueError(
                    f"Expected source_points last dimension to be {self.n_spatial_dims}, "
                    f"got {source_points.shape[-1]}"
                )
            if target_points.shape[-1] != self.n_spatial_dims:
                raise ValueError(
                    f"Expected target_points last dimension to be {self.n_spatial_dims}, "
                    f"got {target_points.shape[-1]}"
                )
            source_rank_counts = rank_counts(self.source_data_ranks)
            global_rank_counts = rank_counts(self.global_data_ranks)
            for name, (actual, expected) in {
                "source scalars": (
                    concatenated_length(source_scalars),
                    source_rank_counts[0],
                ),
                "source vectors": (
                    concatenated_length(source_vectors),
                    source_rank_counts[1],
                ),
                "global scalars": (
                    concatenated_length(global_scalars),
                    global_rank_counts[0],
                ),
                "global vectors": (
                    concatenated_length(global_vectors),
                    global_rank_counts[1],
                ),
            }.items():
                if actual != expected:
                    raise ValueError(
                        f"This kernel was instantiated to expect {expected} {name},\n"
                        f"but the forward-method input gives {actual} {name}."
                    )

        ### Assemble inputs to the neural network
        interaction_dims = torch.Size([n_targets, n_sources])
        scalars = TensorDict(
            {
                "source_scalars": source_scalars.expand(
                    n_targets, *source_scalars.batch_size
                ),
                "global_scalars": global_scalars.expand(
                    n_targets, n_sources, *global_scalars.batch_size
                ),
            },
            batch_size=interaction_dims,
            device=device,
        )

        # `vectors` is a list of tensors, each of shape (n_targets, n_sources, n_dims)
        # EVERY TENSOR IN THIS LIST SHOULD BE PHYSICALLY UNITLESS to preserve units-invariance.
        vectors = TensorDict(
            {
                "source_vectors": source_vectors.expand(
                    torch.Size([n_targets]) + source_vectors.batch_size
                ),
                "global_vectors": global_vectors.expand(
                    torch.Size([n_targets, n_sources]) + global_vectors.batch_size
                ),
            },
            batch_size=interaction_dims + torch.Size([self.n_spatial_dims]),
            device=device,
        )
        vectors["r"] = (
            target_points[:, None, :]  # (n_targets, 1, n_dims)
            - source_points[None, :, :]  # (1, n_sources, n_dims)
        ) / reference_length  # (n_targets, n_sources, n_dims)

        ### Core feature engineering, network evaluation, and post-processing
        result = self._evaluate_interactions(
            scalars=scalars,
            vectors=vectors,
            device=device,
        )

        ### Aggregate over sources, weighted by source strengths
        final_result = TensorDict(
            {
                k: torch.einsum(
                    "ts...,s->t...",
                    v,
                    source_strengths,
                )
                for k, v in result.items()
            },
            batch_size=torch.Size([n_targets]),
            device=device,
        )

        return final_result

    def _evaluate_interactions(
        self,
        *,
        scalars: TensorDict[str, Float[torch.Tensor, "*interaction_dims"]],
        vectors: TensorDict[str, Float[torch.Tensor, "*interaction_dims n_spatial_dims"]],
        device: torch.device,
    ) -> TensorDict[str, Float[torch.Tensor, "*interaction_dims"]]:
        r"""Core kernel computation: feature engineering, network, and post-processing.

        Operates on pre-assembled interaction feature tensors with arbitrary
        leading batch dimensions. Both ``Kernel.forward()`` (with dense
        ``(N_{tgt}, N_{src})`` interactions) and ``BarnesHutKernel`` (with
        sparse ``(N_{pairs},)`` interactions) call this method.

        Parameters
        ----------
        scalars : TensorDict
            Scalar features with ``batch_size=(*interaction_dims,)``.
            Must contain ``"source_scalars"`` and ``"global_scalars"`` sub-dicts.
        vectors : TensorDict
            Vector features with ``batch_size=(*interaction_dims, D)``.
            Must contain ``"r"`` (displacement), ``"source_vectors"``, and
            ``"global_vectors"`` sub-dicts. All values must be dimensionless.
        device : torch.device
            Device for tensor allocation.

        Returns
        -------
        TensorDict[str, Float[torch.Tensor, "..."]]
            Per-interaction output fields with ``batch_size=(*interaction_dims,)``.
            NOT aggregated over sources. Scalar fields have shape
            ``(*interaction_dims,)``, vector fields ``(*interaction_dims, D)``.
        """
        # Cast to autocast dtype after the fp32-critical r computation
        if torch.is_autocast_enabled(device.type):
            dtype = torch.get_autocast_dtype(device.type)
            scalars = scalars.to(dtype=dtype)
            vectors = vectors.to(dtype=dtype)
        else:
            dtype = None

        smoothing_radius = torch.tensor(
            self.smoothing_radius, device=device, dtype=dtype
        )

        ### Vector magnitude, direction, and log-magnitude features
        vectors_mag_squared: TensorDict = (  # ty: ignore[invalid-assignment]
            (vectors * vectors).sum(dim=-1).apply(lambda x: x + smoothing_radius**2)
        )
        vectors_mag = vectors_mag_squared.sqrt()
        vectors_hat = vectors / vectors_mag.unsqueeze(-1)
        vectors_log_mag = smooth_log(vectors_mag)

        # Each of the vectors' magnitudes become an input feature
        scalars["vectors_log_mag"] = vectors_log_mag

        # TODO in 3D, add cross products of pairs of vectors as input features
        
        ### Pairwise spherical harmonic features from vector pairs
        keypairs = list(itertools.combinations(range(concatenated_length(vectors)), 2))
        k1, k2 = zip(*keypairs) if keypairs else ([], [])
        vectors_hat_concatenated: torch.Tensor = concatenate_leaves(vectors_hat)
        # shape: (*interaction_dims, n_spatial_dims, n_vectors_in)

        v1_hat = vectors_hat_concatenated[..., :, k1]
        v2_hat = vectors_hat_concatenated[..., :, k2]
        cos_theta_pairs = torch.sum(v1_hat * v2_hat, dim=-2)
        # shape: (*interaction_dims, len(keypairs))

        # [1:] skips P_0(x) = 1 (constant), which carries no angular information
        spherical_harmonics: list[torch.Tensor] = legendre_polynomials(
            x=cos_theta_pairs, n=self.n_spherical_harmonics + 1
        )[1:]

        vectors_mag_concatenated: torch.Tensor = concatenate_leaves(vectors_mag)
        v1_mag = vectors_mag_concatenated[..., k1]
        v2_mag = vectors_mag_concatenated[..., k2]

        for i, harmonics in enumerate(spherical_harmonics):
            scalars[f"pairwise_spherical_harmonics_{i}"] = (
                smooth_log(v1_mag * v2_mag) * harmonics
            )

        cat_input_tensors: torch.Tensor = concatenate_leaves(scalars)
        del scalars
        # shape: (*interaction_dims, self.network_in_features)

        ### Validate and evaluate the neural network
        if not torch.compiler.is_compiling():
            if not cat_input_tensors.shape[-1] == self.network_in_features:
                raise RuntimeError(
                    f"The input tensor has {cat_input_tensors.shape[-1]=!r} features, but the network expects {self.network_in_features=!r} input features.\n"
                    f"This is due to a shape inconsistency between the `network_in_features` and `forward` methods of the {self.__class__.__name__!r} class."
                )

        interaction_dims = cat_input_tensors.shape[:-1]
        flattened_input = cat_input_tensors.reshape(prod(interaction_dims), self.network_in_features)

        ### Lazy-compile the MLP on first call. This fuses linear+activation
        ### layers, giving ~12% speedup on memory-bound H100 workloads.
        ### Deferred from __init__ so that torchinfo and other introspection
        ### tools can inspect the uncompiled module tree. Skipped when an
        ### outer torch.compile is already tracing (it handles fusion itself).
        ### Uses dynamic=True so that varying chunk sizes (batch dimension)
        ### share one compiled graph per kernel, avoiding repeated recompilation.
        ### Stored via object.__setattr__ to bypass nn.Module submodule
        ### registration, keeping self.network (and thus state_dict) unmodified.
        if torch.compiler.is_compiling():
            network = self.network
        else:
            if not hasattr(self, "_compiled_network"):
                object.__setattr__(
                    self,
                    "_compiled_network",
                    torch.compile(self.network, dynamic=True, mode="default"),
                )
            network = self._compiled_network
        flattened_output = network(flattened_input)

        output = flattened_output.reshape(*interaction_dims, self.network_out_features)

        ### Far-field decay envelope
        r_mag_sq: torch.Tensor = vectors_mag_squared["r"]  # ty: ignore[invalid-assignment]
        output = output * (
            -torch.expm1(-r_mag_sq[..., None])
        )  # Lamb-Oseen vortex kernel, numerically stable via expm1
        if self.n_spatial_dims == 2:
            output = output / (r_mag_sq[..., None] + 1).sqrt()
        elif self.n_spatial_dims == 3:
            output = output / (r_mag_sq[..., None] + 1)
        else:
            output = output / (r_mag_sq[..., None] + 1) ** (
                (self.n_spatial_dims - 1) / 2
            )

        ### Add field-name semantics to the flat output channels
        n_vectors_in = len(vectors.keys(include_nested=True, leaves_only=True))
        result: TensorDict[str, Float[torch.Tensor, "..."]] = self.add_semantics(
            output,
            shape_for_scalars=torch.Size([]),
            shape_for_vectors=torch.Size(
                [
                    1  # r_hat
                    + 2 * (n_vectors_in - 1),  # All non-r vectors
                ]
            ),
        )

        ### Vector reprojection onto local rotationally-equivariant basis
        ranks_dict = flatten_rank_spec(self.output_field_ranks)
        vector_reprojection_needed = any(
            rank == 1 for rank in ranks_dict.values()
        )

        if vector_reprojection_needed:
            # Helmholtz-like decomposition: each vector field is expressed in a
            # local basis derived from the input vectors (r_hat, source vectors,
            # and their derived dipole/polar/spherical directions).
            basis_vector_components: list[torch.Tensor] = []

            basis_vector_components.append(vectors_hat["r"])

            for k in vectors.keys(include_nested=True, leaves_only=True):
                if k == "r":
                    continue

                scale: torch.Tensor = vectors_log_mag[k][..., None]  # ty: ignore[invalid-assignment]

                basis_vector_components.append(scale * vectors_hat[k])

                if self.n_spatial_dims == 2:
                    _, e_theta, e_kappa = polar_and_dipole_basis(
                        r_hat=vectors_hat["r"],
                        n_hat=vectors_hat[k],
                        normalize_basis_vectors=False,
                    )
                    basis_vector_components.append(scale * e_kappa)

                elif self.n_spatial_dims == 3:
                    _, e_theta, e_phi = spherical_basis(
                        r_hat=vectors_hat["r"],
                        n_hat=vectors_hat[k],
                        normalize_basis_vectors=False,
                    )
                    basis_vector_components.append(scale * e_theta)

                else:
                    raise NotImplementedError(
                        f"The {self.__class__.__name__!r} class does not support {self.n_spatial_dims=!r}-dimensional problems."
                    )

            basis_vectors = torch.stack(basis_vector_components, dim=-1)

            for field_name, rank in ranks_dict.items():
                if rank == 1:
                    result[field_name] = torch.sum(
                        basis_vectors
                        * result[field_name].unsqueeze(-2),
                        dim=-1,
                    )

        return result


class BarnesHutKernel(Kernel):
    r"""Tree-accelerated kernel evaluation via Barnes-Hut monopole approximation.

    Reduces the :math:`O(N_{src} \cdot N_{tgt})` cost of the all-to-all kernel
    evaluation to :math:`O((N_{src} + N_{tgt}) \log N_{src})` by building a
    spatial cluster tree over source points and using aggregate (monopole)
    representations for distant clusters.

    For each target point, sources are classified as either:

    - **Near-field**: within the opening-angle threshold, evaluated exactly
      using the underlying :class:`Kernel`'s neural network.
    - **Far-field**: beyond the threshold, approximated by evaluating the
      same network with the cluster's area-weighted centroid, average normal,
      and average features as a "virtual source."

    Both near- and far-field interactions are accumulated into a single batch
    and evaluated in one call to :meth:`Kernel._evaluate_interactions`,
    minimizing kernel launch overhead ("accumulate pairs, evaluate once").

    The ``ClusterTree`` spatial structure can be precomputed per mesh geometry
    and reused across kernel branches and hyperlayers. The ``InteractionPlan``
    (which target-source pairs are near vs. far) can be cached when targets
    equal sources (communication hyperlayers).

    Parameters
    ----------
    Inherits all parameters from :class:`Kernel`.

    leaf_size : int, optional, default=32
        Maximum sources per tree leaf node. Larger values produce shallower
        trees (fewer traversal iterations) at the cost of more exact
        interactions per leaf.

    Forward
    -------
    Same parameters as :class:`Kernel`, with additions:

    theta : float, optional, default=0.5
        Barnes-Hut opening angle parameter. Smaller values are more
        conservative (more exact interactions, higher accuracy, slower).
    cluster_tree : ClusterTree or None, optional, default=None
        Precomputed spatial tree over source points. If ``None``, built
        from ``source_points`` on each call.
    interaction_plan : InteractionPlan or None, optional, default=None
        Precomputed traversal result. If ``None``, computed from the tree
        and target points on each call.
    source_areas : Float[torch.Tensor, "n_sources"] or None, optional, default=None
        Per-source areas for aggregate weighting. Defaults to ones.

    Outputs
    -------
    TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
        Approximate kernel output, converging to the exact result as
        ``theta`` increases.
    """

    def __init__(
        self,
        *,
        n_spatial_dims: int,
        output_field_ranks: RankSpecDict,
        source_data_ranks: RankSpecDict | None = None,
        global_data_ranks: RankSpecDict | None = None,
        smoothing_radius: float = 1e-8,
        hidden_layer_sizes: Sequence[int] | None = None,
        n_spherical_harmonics: int = 4,
        network_type: Literal["pade", "mlp"] = "pade",
        spectral_norm: bool = False,
        use_gradient_checkpointing: bool = True,
        leaf_size: int = 32,
    ):
        super().__init__(
            n_spatial_dims=n_spatial_dims,
            output_field_ranks=output_field_ranks,
            source_data_ranks=source_data_ranks,
            global_data_ranks=global_data_ranks,
            smoothing_radius=smoothing_radius,
            hidden_layer_sizes=hidden_layer_sizes,
            n_spherical_harmonics=n_spherical_harmonics,
            network_type=network_type,
            spectral_norm=spectral_norm,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.leaf_size = leaf_size

    def forward(
        self,
        *,
        reference_length: Float[torch.Tensor, ""],
        source_points: Float[torch.Tensor, "n_sources n_dims"],
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        source_strengths: Float[torch.Tensor, " n_sources"] | None = None,
        source_data: TensorDict | None = None,
        global_data: TensorDict | None = None,
        theta: float = 0.5,
        cluster_tree: "ClusterTree | None" = None,
        interaction_plan: "InteractionPlan | None" = None,
        source_areas: Float[torch.Tensor, " n_sources"] | None = None,
    ) -> TensorDict[str, Float[torch.Tensor, "n_targets ..."]]:
        r"""Evaluate the kernel with Barnes-Hut tree acceleration.

        Parameters
        ----------
        reference_length : Float[torch.Tensor, ""]
            Reference length scale for nondimensionalization.
        source_points : Float[torch.Tensor, "n_sources n_dims"]
            Source point coordinates.
        target_points : Float[torch.Tensor, "n_targets n_dims"]
            Target point coordinates.
        source_strengths : Float[torch.Tensor, "n_sources"] or None
            Per-source strength weights. Defaults to ones.
        source_data : TensorDict or None
            Per-source features (normals, latents).
        global_data : TensorDict or None
            Problem-level conditioning features.
        theta : float
            Opening angle parameter controlling accuracy/speed tradeoff.
        cluster_tree : ClusterTree or None
            Precomputed tree. Built on-the-fly if ``None``.
        interaction_plan : InteractionPlan or None
            Precomputed traversal plan. Computed on-the-fly if ``None``.
        source_areas : Float[torch.Tensor, "n_sources"] or None
            Per-source areas for aggregate weighting. Defaults to ones.

        Returns
        -------
        TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
            Kernel output fields at target points.
        """
        from physicsnemo.experimental.models.globe.cluster_tree import (
            ClusterTree,
            InteractionPlan,
        )

        n_sources = source_points.shape[0]
        n_targets = target_points.shape[0]
        device = source_points.device

        ### Set defaults
        if source_strengths is None:
            source_strengths = torch.ones(n_sources, device=device)
        if source_data is None:
            source_data = TensorDict({}, batch_size=[n_sources], device=device)
        if global_data is None:
            global_data = TensorDict({}, device=device)
        if source_areas is None:
            source_areas = torch.ones(n_sources, device=device)

        ### Build tree if not precomputed
        if cluster_tree is None:
            cluster_tree = ClusterTree.from_points(
                source_points, leaf_size=self.leaf_size, areas=source_areas
            )

        ### Find interaction pairs if not precomputed
        if interaction_plan is None:
            interaction_plan = cluster_tree.find_interaction_pairs(
                target_points, theta=theta
            )

        ### Compute source aggregates for far-field clusters
        aggregates = cluster_tree.compute_source_aggregates(
            source_points=source_points,
            areas=source_areas,
            source_data=source_data,
        )

        ### Compute per-node total strengths for far-field weighting
        node_total_strength = self._compute_node_strengths(
            cluster_tree, source_strengths
        )

        ### Prepare rank-split source/global data (shared setup)
        source_by_rank = split_by_leaf_rank(source_data)
        source_scalars = source_by_rank[0]
        source_vectors = source_by_rank[1]
        source_vectors.batch_size = torch.Size([n_sources, self.n_spatial_dims])

        global_by_rank = split_by_leaf_rank(global_data)
        global_scalars = global_by_rank[0]
        global_vectors = global_by_rank[1]
        global_vectors.batch_size = torch.Size([self.n_spatial_dims])

        n_near = interaction_plan.n_near
        n_far = interaction_plan.n_far
        n_total = n_near + n_far

        ### Handle degenerate case: no interactions at all
        if n_total == 0:
            return self._empty_result(n_targets, device)

        ### Gather near-field interaction inputs
        if n_near > 0:
            near_tgt_pts = target_points[interaction_plan.near_target_ids]
            near_src_pts = source_points[interaction_plan.near_source_ids]
            near_r = (near_tgt_pts - near_src_pts) / reference_length
            near_src_scalars = source_scalars[interaction_plan.near_source_ids]
            near_src_vectors = source_vectors[interaction_plan.near_source_ids]
            near_strengths = source_strengths[interaction_plan.near_source_ids]
        else:
            near_r = torch.empty(0, self.n_spatial_dims, device=device)
            near_src_scalars = source_scalars[:0]
            near_src_vectors = source_vectors[:0]
            near_strengths = torch.empty(0, device=device)

        ### Gather far-field interaction inputs from cluster aggregates
        if n_far > 0:
            far_tgt_pts = target_points[interaction_plan.far_target_ids]
            far_src_pts = aggregates.node_centroid[interaction_plan.far_node_ids]
            far_r = (far_tgt_pts - far_src_pts) / reference_length

            if aggregates.node_source_data is not None:
                agg_data_by_rank = split_by_leaf_rank(aggregates.node_source_data)
            else:
                agg_data_by_rank = split_by_leaf_rank(
                    TensorDict(
                        {}, batch_size=[cluster_tree.n_nodes], device=device
                    )
                )
            far_src_scalars = agg_data_by_rank[0][interaction_plan.far_node_ids]
            far_src_vectors = agg_data_by_rank[1]
            far_src_vectors.batch_size = torch.Size(
                [cluster_tree.n_nodes, self.n_spatial_dims]
            )
            far_src_vectors = far_src_vectors[interaction_plan.far_node_ids]
            far_strengths = node_total_strength[interaction_plan.far_node_ids]
        else:
            far_r = torch.empty(0, self.n_spatial_dims, device=device)
            far_src_scalars = source_scalars[:0]
            far_src_vectors = source_vectors[:0]
            far_strengths = torch.empty(0, device=device)

        ### Concatenate near + far into one batch for a single _evaluate_interactions call
        all_r = torch.cat([near_r, far_r], dim=0)  # (n_total, D)
        all_src_scalars = TensorDict.cat(
            [near_src_scalars, far_src_scalars], dim=0
        )
        all_src_vectors = TensorDict.cat(
            [near_src_vectors, far_src_vectors], dim=0
        )
        all_strengths = torch.cat([near_strengths, far_strengths], dim=0)

        ### Assemble scalars/vectors TensorDicts for _evaluate_interactions
        scalars = TensorDict(
            {
                "source_scalars": all_src_scalars,
                "global_scalars": global_scalars.expand(n_total, *global_scalars.batch_size),
            },
            batch_size=torch.Size([n_total]),
            device=device,
        )
        vectors = TensorDict(
            {
                "source_vectors": all_src_vectors,
                "global_vectors": global_vectors.expand(
                    torch.Size([n_total]) + global_vectors.batch_size
                ),
            },
            batch_size=torch.Size([n_total, self.n_spatial_dims]),
            device=device,
        )
        vectors["r"] = all_r

        ### Evaluate the kernel network on all pairs at once
        fn = self._evaluate_interactions
        kwargs = dict(scalars=scalars, vectors=vectors, device=device)
        if self.training and self.use_gradient_checkpointing:
            per_pair_result = checkpoint(fn, use_reentrant=False, **kwargs)
        else:
            per_pair_result = fn(**kwargs)

        ### Weight by strengths and scatter-add back to target indices
        all_target_ids = torch.cat(
            [interaction_plan.near_target_ids, interaction_plan.far_target_ids],
            dim=0,
        )

        final_result = TensorDict(
            {},
            batch_size=torch.Size([n_targets]),
            device=device,
        )
        for k, v in per_pair_result.items():
            weighted = v * all_strengths.view(-1, *([1] * (v.ndim - 1)))
            out = torch.zeros(
                (n_targets,) + v.shape[1:], dtype=weighted.dtype, device=device
            )
            idx = all_target_ids.view(-1, *([1] * (v.ndim - 1))).expand_as(weighted)
            out.scatter_add_(0, idx, weighted)
            final_result[k] = out

        return final_result

    def _compute_node_strengths(
        self,
        tree: "ClusterTree",
        source_strengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total source strength per tree node via bottom-up summation.

        Parameters
        ----------
        tree : ClusterTree
            The spatial cluster tree.
        source_strengths : torch.Tensor
            Per-source strength values, shape ``(n_sources,)``.

        Returns
        -------
        torch.Tensor
            Total strength per node, shape ``(n_nodes,)``.
        """
        device = source_strengths.device
        n_nodes = tree.n_nodes
        node_strengths = torch.zeros(n_nodes, dtype=source_strengths.dtype, device=device)

        is_leaf = tree.leaf_count > 0
        leaf_ids = torch.where(is_leaf)[0]

        if leaf_ids.numel() == 0:
            return node_strengths

        ### Sum strengths within each leaf
        leaf_starts = tree.leaf_start[leaf_ids]
        leaf_counts = tree.leaf_count[leaf_ids]
        n_leaves = leaf_ids.shape[0]
        total = int(leaf_counts.sum())

        if total > 0:
            seg_ids = torch.repeat_interleave(
                torch.arange(n_leaves, dtype=torch.long, device=device),
                leaf_counts,
            )
            cum = leaf_counts.cumsum(0)
            offsets = torch.arange(total, dtype=torch.long, device=device)
            offsets = offsets - torch.repeat_interleave(cum - leaf_counts, leaf_counts)
            positions = torch.repeat_interleave(leaf_starts, leaf_counts) + offsets

            sorted_strengths = source_strengths[tree.sorted_source_order[positions]]
            leaf_sums = torch.zeros(n_leaves, dtype=source_strengths.dtype, device=device)
            leaf_sums.scatter_add_(0, seg_ids, sorted_strengths)
            node_strengths[leaf_ids] = leaf_sums

        ### Bottom-up propagation
        for depth in range(int(tree.max_depth.item()), 0, -1):
            # Find internal nodes by checking left_child >= 0
            internal = tree.node_left_child >= 0
            left = tree.node_left_child
            right = tree.node_right_child

            has_children = internal & (left < n_nodes) & (right >= 0) & (right < n_nodes)
            int_ids = torch.where(has_children)[0]
            if int_ids.numel() > 0:
                node_strengths[int_ids] = (
                    node_strengths[tree.node_left_child[int_ids]]
                    + node_strengths[tree.node_right_child[int_ids]]
                )

        return node_strengths

    def _empty_result(
        self,
        n_targets: int,
        device: torch.device,
    ) -> TensorDict[str, Float[torch.Tensor, "n_targets ..."]]:
        """Produce a zero-valued result TensorDict for the degenerate case."""
        ranks_dict = flatten_rank_spec(self.output_field_ranks)
        fields: dict[str, torch.Tensor] = {}
        for name, rank in sorted(ranks_dict.items()):
            if rank == 0:
                fields[name] = torch.zeros(n_targets, device=device)
            else:
                fields[name] = torch.zeros(
                    n_targets, self.n_spatial_dims, device=device
                )
        return TensorDict(fields, batch_size=torch.Size([n_targets]), device=device)


class MultiscaleKernel(Module):
    r"""Multiscale kernel composition that linearly combines kernels at different length scales.

    This class implements the multiscale kernel architecture described in paper Section 3.3.
    Physical systems often exhibit phenomena at multiple characteristic length scales
    (e.g., viscous boundary layer thickness, geometric features, wakes).
    :class:`MultiscaleKernel` creates independent kernel branches for each reference
    length, allowing each to specialize at different spatial scales while sharing the
    same functional form.

    Each kernel branch:

    - Operates at a user-specified reference length (e.g., ``viscous_length``,
      ``chord_length``)
    - Has its own learnable parameters (separate neural network weights)
    - Has a learnable scale adjustment factor (``log_scalefactor``) that fine-tunes its
      effective reference length during training
    - Receives the same inputs but normalizes relative positions by its effective length
    - Has separate per-source, per-branch strength values

    The outputs from all branches are linearly summed, forming a multiscale superposition.
    This enables efficient representation of fields with disparate spatial scales without
    requiring a single network to span the entire range.

    Additionally, log-ratios of all reference length pairs are automatically added as
    global scalar features. This provides scale relationship information and enables the
    model to behave equivariantly under uniform scaling when all nondimensional parameters
    (e.g., Reynolds number) are held constant.

    Parameters
    ----------
    n_spatial_dims : int
        Number of spatial dimensions (2 or 3).
    output_field_ranks : TensorDict
        Rank-spec TensorDict (see :class:`Kernel`).
    reference_length_names : Sequence[str]
        Sequence of identifiers for reference length scales. Each creates an
        independent kernel branch. Examples: ``["viscous", "geometric"]``.
    source_data_ranks : TensorDict or None, optional
        Rank-spec TensorDict for per-source features (see :class:`Kernel`).
    global_data_ranks : TensorDict or None, optional
        Rank-spec TensorDict for global features (see :class:`Kernel`).
        Log-ratios of reference lengths are automatically added as scalar
        entries before passing to each kernel branch.
    smoothing_radius : float, optional, default=1e-8
        Small value for numerical stability in magnitude computations.
    hidden_layer_sizes : Sequence[int] or None, optional, default=None
        Hidden layer sizes for kernel networks.
    n_spherical_harmonics : int, optional, default=4
        Number of Legendre polynomial terms for angle features.
    network_type : {"pade", "mlp"}, optional, default="pade"
        Type of network to use.
    spectral_norm : bool, optional, default=False
        Whether to apply spectral normalization to network weights.
    use_gradient_checkpointing : bool, optional, default=True
        Forwarded to each :class:`Kernel` branch. See
        :class:`Kernel` for details.

    Forward
    -------
    reference_lengths : dict[str, torch.Tensor]
        Mapping of reference length names to scalar tensors.
    source_points : Float[torch.Tensor, "n_sources n_dims"]
        Physical coordinates of the source points. Shape :math:`(N_{sources}, D)`.
    target_points : Float[torch.Tensor, "n_targets n_dims"]
        Physical coordinates of the target points. Shape :math:`(N_{targets}, D)`.
    source_strengths : TensorDict[str, Float[torch.Tensor, " n_sources"]] or None, optional, default=None
        Per-source, per-branch strength values. TensorDict keyed by
        ``reference_length_names``. Defaults to all ones.
    source_data : TensorDict or None, optional, default=None
        Per-source features with ``batch_size=(N_sources,)``. Mixed-rank
        TensorDict passed through to each :class:`ChunkedKernel` branch.
    global_data : TensorDict or None, optional, default=None
        Problem-level features with ``batch_size=()``. Automatically
        augmented with log-ratios of reference lengths before being passed
        to each kernel branch.
    chunk_size : None or int or {"auto"}, optional, default="auto"
        Chunking behavior.

    Outputs
    -------
    TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
        TensorDict with the summed results from all kernel branches. Each scalar
        field has shape :math:`(N_{targets},)` and each vector field has shape
        :math:`(N_{targets}, D)`.

    Examples
    --------
    >>> kernel = MultiscaleKernel(
    ...     n_spatial_dims=2,
    ...     output_field_ranks=TensorDict({"phi": 0, "u": 1}),
    ...     reference_length_names=["viscous_length", "chord_length"],
    ...     source_data_ranks=TensorDict({"normal": 1}),
    ...     hidden_layer_sizes=[64, 64],
    ... )
    >>> result = kernel(
    ...     source_points=boundary_face_centers,
    ...     target_points=query_points,
    ...     reference_lengths={"viscous_length": torch.tensor(0.001),
    ...                        "chord_length": torch.tensor(1.0)},
    ...     source_data=TensorDict({"normal": normals}, batch_size=[n_sources]),
    ...     source_strengths=TensorDict({"viscous_length": strengths_v,
    ...                                  "chord_length": strengths_c}, ...),
    ... )
    """

    def __init__(
        self,
        *,
        n_spatial_dims: int,
        output_field_ranks: RankSpecDict,
        reference_length_names: Sequence[str],
        source_data_ranks: RankSpecDict | None = None,
        global_data_ranks: RankSpecDict | None = None,
        smoothing_radius: float = 1e-8,
        hidden_layer_sizes: Sequence[int] | None = None,
        n_spherical_harmonics: int = 4,
        network_type: Literal["pade", "mlp"] = "pade",
        spectral_norm: bool = False,
        use_gradient_checkpointing: bool = True,
        leaf_size: int = 32,
    ):
        super().__init__()

        if source_data_ranks is None:
            source_data_ranks = {}
        if global_data_ranks is None:
            global_data_ranks = {}

        self.n_spatial_dims = n_spatial_dims
        self.output_field_ranks = output_field_ranks
        self.reference_length_names = reference_length_names
        self.source_data_ranks = source_data_ranks
        self.global_data_ranks = global_data_ranks
        self.smoothing_radius = smoothing_radius
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_spherical_harmonics = n_spherical_harmonics
        self.network_type = network_type
        self.spectral_norm = spectral_norm
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.leaf_size = leaf_size

        ### Augment global_data_ranks with log-ratio entries for each
        # pair of reference lengths. These are rank-0 (scalar) features.
        augmented_global = {
            **global_data_ranks,
            "log_reference_length_ratios": {
                f"{k1}_{k2}": 0
                for k1, k2 in itertools.combinations(reference_length_names, 2)
            },
        }

        self.kernels = nn.ModuleDict(
            {
                name: BarnesHutKernel(
                    n_spatial_dims=n_spatial_dims,
                    output_field_ranks=output_field_ranks,
                    source_data_ranks=source_data_ranks,
                    global_data_ranks=augmented_global,
                    smoothing_radius=smoothing_radius,
                    hidden_layer_sizes=hidden_layer_sizes,
                    n_spherical_harmonics=n_spherical_harmonics,
                    network_type=network_type,
                    spectral_norm=spectral_norm,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    leaf_size=leaf_size,
                )
                for name in reference_length_names
            }
        )

        self.log_scalefactors = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in reference_length_names}
        )

    def forward(
        self,
        *,
        reference_lengths: dict[str, torch.Tensor],
        source_points: Float[torch.Tensor, "n_sources n_dims"],
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        source_strengths: TensorDict[str, Float[torch.Tensor, " n_sources"]]
        | None = None,
        source_data: TensorDict[str, Float[torch.Tensor, "n_sources ..."]]
        | None = None,
        global_data: TensorDict[str, Float[torch.Tensor, "..."]] | None = None,
        theta: float = 0.5,
        cluster_tree: "ClusterTree | None" = None,
        interaction_plan: "InteractionPlan | None" = None,
        source_areas: Float[torch.Tensor, " n_sources"] | None = None,
    ) -> TensorDict[str, Float[torch.Tensor, "n_targets ..."]]:
        r"""Evaluates the multiscale kernel by combining results from multiple scales.

        Builds a shared :class:`ClusterTree` and :class:`InteractionPlan` once,
        then evaluates each :class:`BarnesHutKernel` branch at its respective
        reference length. Log-ratios of reference lengths are automatically
        added to ``global_data`` as scalar features.

        Parameters
        ----------
        reference_lengths : dict[str, torch.Tensor]
            Mapping of reference length names to scalar tensors.
        source_points : Float[torch.Tensor, "n_sources n_dims"]
            Source point coordinates, shape :math:`(N_{sources}, D)`.
        target_points : Float[torch.Tensor, "n_targets n_dims"]
            Target point coordinates, shape :math:`(N_{targets}, D)`.
        source_strengths : TensorDict or None, optional
            Per-source, per-branch strength values. Defaults to all ones.
        source_data : TensorDict or None, optional
            Per-source features with ``batch_size=(N_sources,)``.
        global_data : TensorDict or None, optional
            Problem-level features with ``batch_size=()``.
        theta : float
            Barnes-Hut opening angle parameter.
        cluster_tree : ClusterTree or None, optional
            Precomputed tree. Built from ``source_points`` if ``None``.
        interaction_plan : InteractionPlan or None, optional
            Precomputed traversal plan. Computed if ``None``.
        source_areas : Float[torch.Tensor, "n_sources"] or None, optional
            Per-source areas for aggregate weighting. Defaults to ones.

        Returns
        -------
        TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
            Summed results from all kernel branches.
        """
        from physicsnemo.experimental.models.globe.cluster_tree import ClusterTree

        n_sources: int = len(source_points)
        device = source_points.device

        ### Set defaults
        if source_strengths is None:
            source_strengths = TensorDict(
                {
                    name: torch.ones(n_sources, device=device)
                    for name in self.reference_length_names
                },
                batch_size=torch.Size([n_sources]),
                device=device,
            )
        if source_data is None:
            source_data = TensorDict({}, batch_size=[n_sources], device=device)
        if global_data is None:
            global_data = TensorDict({}, device=device)
        if source_areas is None:
            source_areas = torch.ones(n_sources, device=device)

        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            for name, (actual, expected) in {
                "reference_lengths": (
                    set(reference_lengths.keys()),
                    set(self.reference_length_names),
                ),
                "source_strengths": (
                    set(source_strengths.keys()),
                    set(self.reference_length_names),
                ),
            }.items():
                if actual != expected:
                    raise ValueError(
                        f"This kernel was instantiated to expect {expected} {name},\n"
                        f"but the forward-method input gives {actual} {name}."
                    )

        ### Build shared tree and interaction plan (reused across branches)
        if cluster_tree is None:
            cluster_tree = ClusterTree.from_points(
                source_points, areas=source_areas,
            )
        if interaction_plan is None:
            interaction_plan = cluster_tree.find_interaction_pairs(
                target_points, theta=theta
            )

        ### Augment global_data with log-ratios of reference lengths.
        log_ratios = TensorDict(
            {
                f"{k1}_{k2}": (
                    reference_lengths[k1] / reference_lengths[k2]
                ).log()
                for k1, k2 in itertools.combinations(
                    self.reference_length_names, 2
                )
            },
            device=device,
        )
        global_data["log_reference_length_ratios"] = log_ratios

        ### Evaluate each branch with the shared tree and plan
        results_pieces: list[TensorDict[str, Float[torch.Tensor, "n_targets ..."]]] = [
            self.kernels[name](
                reference_length=reference_lengths[name]
                * torch.exp(self.log_scalefactors[name]),
                source_points=source_points,
                target_points=target_points,
                source_strengths=source_strengths[name],
                source_data=source_data,
                global_data=global_data,
                theta=theta,
                cluster_tree=cluster_tree,
                interaction_plan=interaction_plan,
                source_areas=source_areas,
            )
            for name in self.reference_length_names
        ]

        result: TensorDict[str, Float[torch.Tensor, "n_targets ..."]] = reduce(
            operator.add, results_pieces
        )

        return result
