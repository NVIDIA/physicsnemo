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
from typing import TYPE_CHECKING, Literal, Sequence

import torch
import torch.nn as nn
from jaxtyping import Float
from tensordict import TensorDict
from torch.profiler import record_function
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

logger = logging.getLogger("globe.field_kernel")

if TYPE_CHECKING:
    from physicsnemo.experimental.models.globe.cluster_tree import (
        ClusterTree,
        DualInteractionPlan,
        SourceAggregates,
    )


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
        self_regularization_beta: float | None = None,
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
                self_regularization_beta=self_regularization_beta,
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
        Used by :class:`BarnesHutKernel` to estimate chunk memory budgets.

        This is a lower bound - the actual peak is higher due to autograd
        saving input tensors for backward through each element-wise
        operation.  The caller applies a runtime multiplier to account for
        this (see ``BarnesHutKernel._auto_chunk_size``).
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
        with record_function("kernel::feature_engineering"):
            vectors_mag_squared: TensorDict = (
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

        with record_function("kernel::network"):
            flattened_output = self.network(flattened_input)

        output = flattened_output.reshape(*interaction_dims, self.network_out_features)

        ### Far-field decay envelope and vector reprojection
        with record_function("kernel::postprocess"):
            r_mag_sq: torch.Tensor = vectors_mag_squared["r"]
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

            ### Local rotationally-equivariant basis (built only when needed)
            ranks_dict = flatten_rank_spec(self.output_field_ranks)
            needs_basis = any(rank == 1 for rank in ranks_dict.values())

            if needs_basis:
                # Helmholtz-like decomposition: each vector field is expressed in a
                # local basis derived from the input vectors (r_hat, source vectors,
                # and their derived dipole/polar/spherical directions).
                basis_vector_components: list[torch.Tensor] = [vectors_hat["r"]]

                for k in sorted(
                    vectors.keys(include_nested=True, leaves_only=True),
                    key=str,
                ):
                    if k == "r":
                        continue

                    scale: torch.Tensor = vectors_log_mag[k][..., None]
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

            ### Build per-field outputs in a single pass over the flat tensor.
            # One immutable dict + one TensorDict construction (no setitem on
            # the result), sidestepping pytorch/tensordict#1680 under
            # torch.compile + torch.utils.checkpoint.
            n_vectors_in = len(vectors.keys(include_nested=True, leaves_only=True))
            coeffs_per_vector = 1 + 2 * (n_vectors_in - 1)  # r_hat + (theta, kappa) per non-r vector

            final_fields: dict[str, torch.Tensor] = {}
            offset = 0
            for name in sorted(ranks_dict):
                if ranks_dict[name] == 0:
                    final_fields[name] = output[..., offset]
                    offset += 1
                else:
                    coeffs = output[..., offset : offset + coeffs_per_vector]
                    final_fields[name] = torch.sum(
                        basis_vectors * coeffs.unsqueeze(-2),
                        dim=-1,
                    )
                    offset += coeffs_per_vector

            result: TensorDict[str, Float[torch.Tensor, "..."]] = TensorDict(
                final_fields,
                batch_size=output.shape[:-1],
                device=device,
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
    and reused across kernel branches and hyperlayers. The
    ``DualInteractionPlan`` can be cached when targets equal sources
    (communication hyperlayers).

    Parameters
    ----------
    Inherits all parameters from :class:`Kernel`.

    leaf_size : int, optional, default=1
        Maximum sources per tree leaf node. Larger values produce shallower
        trees (fewer traversal iterations) at the cost of more exact
        interactions per leaf.

    Forward
    -------
    Same parameters as :class:`Kernel`, with additions:

    theta : float, optional, default=1.0
        Barnes-Hut opening angle.  A node is approximated when
        ``D/r < theta``.  Larger values are more aggressive (more
        approximation, faster).  At ``theta = 0``, all interactions
        are exact.
    cluster_tree : ClusterTree or None, optional, default=None
        Precomputed spatial tree over source points. If ``None``, built
        from ``source_points`` on each call.
    dual_plan : DualInteractionPlan or None, optional, default=None
        Precomputed dual traversal plan. If ``None``, computed from the
        trees and target points on each call.
    source_areas : Float[torch.Tensor, "n_sources"] or None, optional, default=None
        Per-source areas for aggregate weighting. Defaults to ones.
    source_aggregates : SourceAggregates or None, optional, default=None
        Precomputed per-node aggregates. If ``None``, computed on each
        call.  Pass this to avoid redundant computation across branches.

    Outputs
    -------
    TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
        Approximate kernel output, converging to the exact result as
        ``theta`` approaches zero.
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
        leaf_size: int = 1,
        self_regularization_beta: float | None = None,
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
            self_regularization_beta=self_regularization_beta,
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
        theta: float = 1.0,
        cluster_tree: "ClusterTree | None" = None,
        target_tree: "ClusterTree | None" = None,
        dual_plan: "DualInteractionPlan | None" = None,
        source_areas: Float[torch.Tensor, " n_sources"] | None = None,
        source_aggregates: "SourceAggregates | None" = None,
        target_centroids: Float[torch.Tensor, "n_target_nodes n_dims"] | None = None,
        near_chunk_size: int | None = None,
        expand_far_targets: bool = False,
    ) -> TensorDict[str, Float[torch.Tensor, "n_targets ..."]]:
        r"""Evaluate the kernel with dual-tree Barnes-Hut acceleration.

        Uses two separate evaluation phases:

        - **Phase A (near-field)**: individual target-source pairs from
          nearby leaf nodes, evaluated exactly with chunked processing.
        - **Phase B (far-field node pairs)**: the kernel is evaluated ONCE
          at ``(centroid_T, centroid_S, avg_data_S)`` per well-separated
          node pair, then broadcast to all individual targets in the
          target node via scatter_add.

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
            Barnes-Hut opening angle (larger = more aggressive).
        cluster_tree : ClusterTree or None
            Precomputed source tree. Built on-the-fly if ``None``.
        target_tree : ClusterTree or None
            Precomputed target tree. Built on-the-fly if ``None``.
            For self-interaction (comm layers), pass the same tree as
            ``cluster_tree``.
        dual_plan : DualInteractionPlan or None
            Precomputed dual traversal plan. Computed on-the-fly if ``None``.
        source_areas : Float[torch.Tensor, "n_sources"] or None
            Per-source areas for aggregate weighting. Defaults to ones.
        source_aggregates : SourceAggregates or None
            Precomputed per-node source aggregates.
        target_centroids : Float[torch.Tensor, "n_target_nodes n_dims"] or None
            Per-node centroids for the target tree. If ``None`` and
            ``target_tree is cluster_tree`` (self-interaction), source
            aggregates' centroids are reused. Otherwise computed from
            the target tree.
        near_chunk_size : int or None
            Fixed chunk size for near-field pair processing. When provided,
            overrides :meth:`_auto_chunk_size`. Pass this from an outer scope
            to ensure deterministic chunking inside ``torch.utils.checkpoint``
            replay (free GPU memory changes between forward and backward,
            so ``_auto_chunk_size`` would return different values).
        expand_far_targets : bool, optional, default=False
            If ``True``, far-field node pairs are expanded to individual
            target points during plan construction, eliminating the
            target-side centroid broadcast.  Passed through to
            :meth:`ClusterTree.find_dual_interaction_pairs`.

        Returns
        -------
        TensorDict[str, Float[torch.Tensor, "n_targets ..."]]
            Kernel output fields at target points.
        """
        from physicsnemo.experimental.models.globe.cluster_tree import (
            ClusterTree,
            DualInteractionPlan,
            SourceAggregates,
        )
        from physicsnemo.mesh.spatial._ragged import _ragged_arange

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

        ### Build trees if not precomputed
        if cluster_tree is None:
            cluster_tree = ClusterTree.from_points(
                source_points, leaf_size=self.leaf_size, areas=source_areas
            )
        if target_tree is None:
            target_tree = ClusterTree.from_points(
                target_points, leaf_size=self.leaf_size,
            )

        ### Find dual interaction pairs if not precomputed
        if dual_plan is None:
            dual_plan = cluster_tree.find_dual_interaction_pairs(
                target_tree=target_tree, theta=theta,
                expand_far_targets=expand_far_targets,
            )

        ### Compute source aggregates for far-field clusters.
        if source_aggregates is not None:
            aggregates = source_aggregates
        else:
            aggregates = cluster_tree.compute_source_aggregates(
                source_points=source_points,
                areas=source_areas,
                source_data=source_data,
            )

        ### Resolve target centroids for far-field node pairs.
        # For self-interaction (target_tree is cluster_tree), reuse source
        # centroids. For separate targets, compute from the target tree.
        if target_centroids is None:
            if target_tree is cluster_tree:
                target_centroids = aggregates.node_centroid
            else:
                tgt_agg = target_tree.compute_source_aggregates(
                    source_points=target_points,
                    areas=torch.ones(n_targets, device=device, dtype=target_points.dtype),
                    source_data=None,
                )
                target_centroids = tgt_agg.node_centroid

        with record_function("bh_kernel::compute_strengths"):
            node_total_strength = self._compute_node_strengths(
                cluster_tree, source_strengths
            )

        ### Prepare rank-split source/global data (shared setup)
        with record_function("bh_kernel::prepare_data"):
            source_by_rank = split_by_leaf_rank(source_data)
            source_scalars = source_by_rank[0]
            source_vectors = source_by_rank[1]
            source_vectors.batch_size = torch.Size([n_sources, self.n_spatial_dims])

            global_by_rank = split_by_leaf_rank(global_data)
            global_scalars = global_by_rank[0]
            global_vectors = global_by_rank[1]
            global_vectors.batch_size = torch.Size([self.n_spatial_dims])

            n_near = dual_plan.n_near
            n_nf = dual_plan.n_nf
            n_fn = dual_plan.n_fn
            n_far_nodes = dual_plan.n_far_nodes

            if not torch.compiler.is_compiling():
                n_dense = n_sources * n_targets
                logger.debug(
                    "BarnesHutKernel: %d near + %d nf + %d fn + %d far_node "
                    "(%d sources x %d targets = %d dense, %.2f%% near-field)",
                    n_near, n_nf, n_fn, n_far_nodes,
                    n_sources, n_targets, n_dense,
                    100.0 * n_near / max(n_dense, 1),
                )

            if n_near == 0 and n_nf == 0 and n_fn == 0 and n_far_nodes == 0:
                return self._empty_result(n_targets, device)

            ### Prepare aggregate data for far-field and (near,far) phases
            if n_far_nodes > 0 or n_nf > 0:
                if aggregates.node_source_data is not None:
                    agg_by_rank = split_by_leaf_rank(aggregates.node_source_data)
                else:
                    agg_by_rank = split_by_leaf_rank(
                        TensorDict(
                            {}, batch_size=[cluster_tree.n_nodes], device=device
                        )
                    )
                agg_scalars = agg_by_rank[0]
                agg_vectors = agg_by_rank[1]
                agg_vectors.batch_size = torch.Size(
                    [cluster_tree.n_nodes, self.n_spatial_dims]
                )

        ### Initialize output buffers
        output_bufs: dict[str, torch.Tensor] = {}

        # ==================================================================
        # Phase A: Near-field (individual target-source pairs, chunked)
        # ==================================================================
        if n_near > 0:
            near_tgt_ids = dual_plan.near_target_ids
            near_src_ids = dual_plan.near_source_ids
            chunk_size = (
                near_chunk_size
                if near_chunk_size is not None
                else self._auto_chunk_size(n_near, device)
            )

            for start in range(0, n_near, chunk_size):
                end = min(start + chunk_size, n_near)

                chunk_tgt_ids = near_tgt_ids[start:end]
                chunk_src_ids = near_src_ids[start:end]

                ### Gather + evaluate inside one checkpoint boundary.
                # By checkpointing a function that takes INDICES (int64,
                # ~8 bytes/pair) and references to the shared source data
                # (O(1)), the autograd graph saves only the indices - not
                # the gathered float data (~300 bytes/pair).  This is a
                # ~37x reduction in checkpoint-saved memory per branch.
                with record_function("bh_kernel::near_chunk"):
                    if self.training and self.use_gradient_checkpointing:
                        chunk_result = checkpoint(
                            self._gather_and_evaluate,
                            chunk_tgt_ids, chunk_src_ids,
                            target_points, source_points,
                            source_scalars, source_vectors,
                            global_scalars, global_vectors,
                            reference_length, device,
                            use_reentrant=False,
                        )
                    else:
                        chunk_result = self._gather_and_evaluate(
                            chunk_tgt_ids, chunk_src_ids,
                            target_points, source_points,
                            source_scalars, source_vectors,
                            global_scalars, global_vectors,
                            reference_length, device,
                        )

                with record_function("bh_kernel::near_scatter"):
                    chunk_strengths = source_strengths[chunk_src_ids]
                    for k, v in chunk_result.items():
                        weighted = v * chunk_strengths.view(-1, *([1] * (v.ndim - 1)))
                        if k not in output_bufs:
                            output_bufs[k] = torch.zeros(
                                (n_targets,) + v.shape[1:],
                                dtype=weighted.dtype,
                                device=device,
                            )
                        idx = chunk_tgt_ids.view(
                            -1, *([1] * (v.ndim - 1))
                        ).expand_as(weighted)
                        output_bufs[k].scatter_add_(0, idx, weighted)

        # ==================================================================
        # Phase B: Far-field node pairs (evaluate once, broadcast to targets)
        # ==================================================================
        if n_far_nodes > 0:
            far_tgt_nids = dual_plan.far_target_node_ids
            far_src_nids = dual_plan.far_source_node_ids

            ### Evaluate kernel at (centroid_T, centroid_S, avg_data_S).
            # Same gather-inside-checkpoint pattern: the checkpoint saves
            # only the node ID indices, not the gathered aggregate data.
            with record_function("bh_kernel::far_node_evaluate"):
                if self.training and self.use_gradient_checkpointing:
                    far_result = checkpoint(
                        self._gather_and_evaluate,
                        far_tgt_nids, far_src_nids,
                        target_centroids, aggregates.node_centroid,
                        agg_scalars, agg_vectors,
                        global_scalars, global_vectors,
                        reference_length, device,
                        use_reentrant=False,
                    )
                else:
                    far_result = self._gather_and_evaluate(
                        far_tgt_nids, far_src_nids,
                        target_centroids, aggregates.node_centroid,
                        agg_scalars, agg_vectors,
                        global_scalars, global_vectors,
                        reference_length, device,
                    )

            ### Broadcast node-level results to individual targets.
            with record_function("bh_kernel::far_node_broadcast"):
                far_strengths = node_total_strength[far_src_nids]

                node_starts = target_tree.node_range_start[far_tgt_nids]
                node_counts = target_tree.node_range_count[far_tgt_nids]
                positions, pair_ids = _ragged_arange(node_starts, node_counts)
                expanded_tgt_ids = target_tree.sorted_source_order[positions]

                for k, v in far_result.items():
                    weighted = v * far_strengths.view(-1, *([1] * (v.ndim - 1)))
                    expanded = weighted[pair_ids]
                    if k not in output_bufs:
                        output_bufs[k] = torch.zeros(
                            (n_targets,) + v.shape[1:],
                            dtype=expanded.dtype,
                            device=device,
                        )
                    idx = expanded_tgt_ids.view(
                        -1, *([1] * (v.ndim - 1))
                    ).expand_as(expanded)
                    output_bufs[k].scatter_add_(0, idx, expanded)

        # ==================================================================
        # Phase C: (near,far) - individual targets × source node centroids
        # ==================================================================
        if n_nf > 0:
            nf_tgt_ids = dual_plan.nf_target_ids
            nf_src_nids = dual_plan.nf_source_node_ids

            ### Same evaluation as Phase B (source centroids + aggregates),
            # but same scatter as Phase A (per-target, no broadcast).
            with record_function("bh_kernel::nf_evaluate"):
                if self.training and self.use_gradient_checkpointing:
                    nf_result = checkpoint(
                        self._gather_and_evaluate,
                        nf_tgt_ids, nf_src_nids,
                        target_points, aggregates.node_centroid,
                        agg_scalars, agg_vectors,
                        global_scalars, global_vectors,
                        reference_length, device,
                        use_reentrant=False,
                    )
                else:
                    nf_result = self._gather_and_evaluate(
                        nf_tgt_ids, nf_src_nids,
                        target_points, aggregates.node_centroid,
                        agg_scalars, agg_vectors,
                        global_scalars, global_vectors,
                        reference_length, device,
                    )

            with record_function("bh_kernel::nf_scatter"):
                nf_strengths = node_total_strength[nf_src_nids]
                for k, v in nf_result.items():
                    weighted = v * nf_strengths.view(-1, *([1] * (v.ndim - 1)))
                    if k not in output_bufs:
                        output_bufs[k] = torch.zeros(
                            (n_targets,) + v.shape[1:],
                            dtype=weighted.dtype,
                            device=device,
                        )
                    idx = nf_tgt_ids.view(
                        -1, *([1] * (v.ndim - 1))
                    ).expand_as(weighted)
                    output_bufs[k].scatter_add_(0, idx, weighted)

        # ==================================================================
        # Phase D: (far,near) - target node centroid × individual sources,
        #          broadcast to stage-1 survivors
        # ==================================================================
        if n_fn > 0:
            fn_tgt_nids = dual_plan.fn_target_node_ids
            fn_src_ids = dual_plan.fn_source_ids

            ### Evaluate K(target_centroid, source_point, source_data).
            # Uses target centroids (like Phase B) but individual source
            # points and data (like Phase A).
            with record_function("bh_kernel::fn_evaluate"):
                if self.training and self.use_gradient_checkpointing:
                    fn_result = checkpoint(
                        self._gather_and_evaluate,
                        fn_tgt_nids, fn_src_ids,
                        target_centroids, source_points,
                        source_scalars, source_vectors,
                        global_scalars, global_vectors,
                        reference_length, device,
                        use_reentrant=False,
                    )
                else:
                    fn_result = self._gather_and_evaluate(
                        fn_tgt_nids, fn_src_ids,
                        target_centroids, source_points,
                        source_scalars, source_vectors,
                        global_scalars, global_vectors,
                        reference_length, device,
                    )

            ### Broadcast to stage-1 survivors via the ragged mapping.
            with record_function("bh_kernel::fn_broadcast"):
                fn_strengths = source_strengths[fn_src_ids]

                positions, pair_ids = _ragged_arange(
                    dual_plan.fn_broadcast_starts,
                    dual_plan.fn_broadcast_counts,
                )
                expanded_tgt_ids = dual_plan.fn_broadcast_targets[positions]

                for k, v in fn_result.items():
                    weighted = v * fn_strengths.view(-1, *([1] * (v.ndim - 1)))
                    expanded = weighted[pair_ids]
                    if k not in output_bufs:
                        output_bufs[k] = torch.zeros(
                            (n_targets,) + v.shape[1:],
                            dtype=expanded.dtype,
                            device=device,
                        )
                    idx = expanded_tgt_ids.view(
                        -1, *([1] * (v.ndim - 1))
                    ).expand_as(expanded)
                    output_bufs[k].scatter_add_(0, idx, expanded)

        if not output_bufs:
            return self._empty_result(n_targets, device)

        return TensorDict(
            output_bufs,
            batch_size=torch.Size([n_targets]),
            device=device,
        )

    def _compute_node_strengths(
        self,
        tree: "ClusterTree",
        source_strengths: Float[torch.Tensor, " n_sources"],
    ) -> Float[torch.Tensor, " n_nodes"]:
        """Compute total source strength per tree node via bottom-up summation.

        Parameters
        ----------
        tree : ClusterTree
            The spatial cluster tree.
        source_strengths : Float[torch.Tensor, "n_sources"]
            Per-source strength values.

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

        if int(leaf_counts.sum()) > 0:
            from physicsnemo.mesh.spatial._ragged import _ragged_arange

            positions, seg_ids = _ragged_arange(
                leaf_starts, leaf_counts, total=tree.n_sources,
            )
            sorted_strengths = source_strengths[tree.sorted_source_order[positions]]
            leaf_sums = torch.zeros(n_leaves, dtype=source_strengths.dtype, device=device)
            leaf_sums.scatter_add_(0, seg_ids, sorted_strengths)
            node_strengths[leaf_ids] = leaf_sums

        ### Bottom-up propagation using cached level ordering
        for level_ids in reversed(tree.internal_nodes_per_level):
            node_strengths[level_ids] = (
                node_strengths[tree.node_left_child[level_ids]]
                + node_strengths[tree.node_right_child[level_ids]]
            )

        return node_strengths

    def _empty_result(
        self,
        n_targets: int,
        device: torch.device,
    ) -> TensorDict[str, Float[torch.Tensor, "n_targets ..."]]:
        """Produce a zero-valued result TensorDict for the degenerate case."""
        # Match the dtype that AMP autocast would produce for real activations,
        # so downstream ops don't hit a float32-vs-half mismatch.
        dtype = (
            torch.get_autocast_dtype(device.type)
            if torch.is_autocast_enabled(device.type)
            else torch.float32
        )
        ranks_dict = flatten_rank_spec(self.output_field_ranks)
        fields: dict[str, torch.Tensor] = {}
        for name, rank in sorted(ranks_dict.items()):
            if rank == 0:
                fields[name] = torch.zeros(n_targets, device=device, dtype=dtype)
            else:
                fields[name] = torch.zeros(
                    n_targets, self.n_spatial_dims, device=device, dtype=dtype
                )
        return TensorDict(fields, batch_size=torch.Size([n_targets]), device=device)

    def _gather_and_evaluate(
        self,
        tgt_ids: torch.Tensor,
        src_ids: torch.Tensor,
        target_positions: torch.Tensor,
        source_positions: torch.Tensor,
        source_scalars: TensorDict,
        source_vectors: TensorDict,
        global_scalars: TensorDict,
        global_vectors: TensorDict,
        reference_length: torch.Tensor,
        device: torch.device,
    ) -> TensorDict:
        """Gather source/target data by index and evaluate interactions.

        This function is the checkpoint boundary for memory-efficient
        training.  By wrapping both the gather (indexing into shared
        source data) and the evaluate (feature engineering + MLP) in one
        checkpointed call, the autograd graph saves only the int64 index
        tensors (~8 bytes/pair) and references to the shared source data
        (O(1)), instead of the gathered float features (~300 bytes/pair).

        Source scalars and vectors are pre-flattened via
        ``concatenate_leaves`` before indexing, reducing K per-leaf index
        ops to 1 cat + 1 index each.  Vectors are split back into
        individual named leaves afterward because the feature engineering
        pipeline in ``_evaluate_interactions`` processes each vector
        separately (magnitudes, dot products, basis construction).
        """
        n_pairs = tgt_ids.shape[0]
        chunk_r = (
            target_positions[tgt_ids] - source_positions[src_ids]
        ) / reference_length

        ### Flatten source scalars into one tensor, gather once, split back.
        # concatenate_leaves: 1 GPU kernel (torch.cat)
        # [src_ids]: 1 GPU kernel (aten::index)
        # Total: 2 kernels instead of K (one per TensorDict leaf).
        # The split-back uses sorted keys matching concatenate_leaves's
        # canonical column ordering so position i maps to the correct leaf.
        src_scalar_keys = sorted(
            source_scalars.keys(include_nested=True, leaves_only=True),
            key=str,
        )
        gathered_src_scalars = concatenate_leaves(source_scalars)[src_ids]
        scalars = TensorDict(
            {
                "source_scalars": TensorDict(
                    {k: gathered_src_scalars[..., i] for i, k in enumerate(src_scalar_keys)},
                    batch_size=torch.Size([n_pairs]),
                    device=device,
                ),
                "global_scalars": global_scalars.expand(
                    n_pairs, *global_scalars.batch_size
                ),
            },
            batch_size=torch.Size([n_pairs]),
            device=device,
        )

        ### Flatten source vectors, gather once, split back into named leaves.
        # The split-back is required because _evaluate_interactions processes
        # each vector leaf separately for magnitude/direction extraction and
        # rotationally-equivariant basis construction.  Integer indexing
        # along the last dimension creates non-contiguous views (zero copies).
        # Sorted keys match concatenate_leaves's canonical column ordering.
        src_vector_keys = sorted(
            source_vectors.keys(include_nested=True, leaves_only=True),
            key=str,
        )
        gathered_src_vectors = concatenate_leaves(source_vectors)[src_ids]
        vectors = TensorDict(
            {
                "source_vectors": TensorDict(
                    {k: gathered_src_vectors[..., i] for i, k in enumerate(src_vector_keys)},
                    batch_size=torch.Size([n_pairs, self.n_spatial_dims]),
                    device=device,
                ),
                "global_vectors": global_vectors.expand(
                    torch.Size([n_pairs]) + global_vectors.batch_size
                ),
            },
            batch_size=torch.Size([n_pairs, self.n_spatial_dims]),
            device=device,
        )
        vectors["r"] = chunk_r

        return self._evaluate_interactions(scalars=scalars, vectors=vectors, device=device)

    def _auto_chunk_size(self, n_total_pairs: int, device: torch.device) -> int:
        """Determine chunk size for pair-batched kernel evaluation.

        Estimates peak memory per pair from the kernel's feature engineering
        pipeline and sizes chunks to fit within ~50% of GPU memory. During
        inference (no grad), the autograd overhead multiplier is dropped,
        allowing larger chunks.

        Returns ``n_total_pairs`` (i.e., no chunking) when the estimated
        peak fits comfortably, or when running on CPU.
        """
        if device.type != "cuda":
            return n_total_pairs

        if torch.is_autocast_enabled(device.type):
            element_bytes = torch.tensor(
                [], dtype=torch.get_autocast_dtype(device.type)
            ).element_size()
        else:
            element_bytes = 4  # fp32

        autograd_overhead = 5 if torch.is_grad_enabled() else 1
        approx_peak_bytes = (
            n_total_pairs
            * self._floats_per_interaction
            * element_bytes
            * autograd_overhead
        )
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        target_bytes = free_bytes // 2

        n_chunks = max(1, ceil(approx_peak_bytes / target_bytes))
        chunk_size = max(1, ceil(n_total_pairs / n_chunks))

        if not torch.compiler.is_compiling():
            logger.debug(
                "auto_chunk_size: %d pairs -> %d chunks of %d "
                "(%.1f MB est. peak, %.1f MB free / %.1f MB total GPU)",
                n_total_pairs, n_chunks, chunk_size,
                approx_peak_bytes / 1e6, free_bytes / 1e6, total_bytes / 1e6,
            )

        return chunk_size


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
        TensorDict passed through to each :class:`BarnesHutKernel` branch.
    global_data : TensorDict or None, optional, default=None
        Problem-level features with ``batch_size=()``. Automatically
        augmented with log-ratios of reference lengths before being passed
        to each kernel branch.
    theta : float, optional, default=1.0
        Barnes-Hut opening angle (larger = more aggressive).
    cluster_tree : ClusterTree or None, optional, default=None
        Pre-built cluster tree for source points.  If ``None``, one is
        built from ``source_points`` using the kernel's ``leaf_size``.
    target_tree : ClusterTree or None, optional, default=None
        Pre-built target tree.  For self-interaction, pass the same tree
        as ``cluster_tree``.
    dual_plan : DualInteractionPlan or None, optional, default=None
        Pre-computed dual traversal plan.  If ``None``, computed from trees.
    source_areas : Float[torch.Tensor, " n_sources"] or None, optional, default=None
        Area weight per source, used for cluster aggregation.

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
        leaf_size: int = 1,
        self_regularization_beta: float | None = None,
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
                    self_regularization_beta=self_regularization_beta,
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
        theta: float = 1.0,
        cluster_tree: "ClusterTree | None" = None,
        target_tree: "ClusterTree | None" = None,
        dual_plan: "DualInteractionPlan | None" = None,
        source_areas: Float[torch.Tensor, " n_sources"] | None = None,
        expand_far_targets: bool = False,
    ) -> TensorDict[str, Float[torch.Tensor, "n_targets ..."]]:
        r"""Evaluates the multiscale kernel by combining results from multiple scales.

        Builds a shared :class:`ClusterTree` and :class:`DualInteractionPlan`
        once, then evaluates each :class:`BarnesHutKernel` branch at its
        respective reference length.

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
            Barnes-Hut opening angle (larger = more aggressive).
        cluster_tree : ClusterTree or None, optional
            Precomputed source tree. Built from ``source_points`` if ``None``.
        target_tree : ClusterTree or None, optional
            Precomputed target tree. Built from ``target_points`` if ``None``.
        dual_plan : DualInteractionPlan or None, optional
            Precomputed dual traversal plan. Computed if ``None``.
        source_areas : Float[torch.Tensor, "n_sources"] or None, optional
            Per-source areas for aggregate weighting. Defaults to ones.
        expand_far_targets : bool, optional, default=False
            If ``True``, eliminates target-side centroid broadcast by
            expanding far-field node pairs to individual target points.
            Passed through to
            :meth:`ClusterTree.find_dual_interaction_pairs`.

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

        ### Build shared trees, dual plan, and aggregates (reused across branches)
        with record_function("multiscale_kernel::build_tree"):
            if cluster_tree is None:
                cluster_tree = ClusterTree.from_points(
                    source_points, leaf_size=self.leaf_size, areas=source_areas,
                )
            if target_tree is None:
                target_tree = ClusterTree.from_points(
                    target_points, leaf_size=self.leaf_size,
                )
            if dual_plan is None:
                dual_plan = cluster_tree.find_dual_interaction_pairs(
                    target_tree=target_tree, theta=theta,
                    expand_far_targets=expand_far_targets,
                )
        with record_function("multiscale_kernel::compute_aggregates"):
            source_aggregates = cluster_tree.compute_source_aggregates(
                source_points=source_points,
                areas=source_areas,
                source_data=source_data,
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
        global_data = global_data.copy()
        global_data["log_reference_length_ratios"] = log_ratios

        ### Precompute near-field chunk sizes outside the checkpoint boundary.
        # _auto_chunk_size queries free GPU memory, which differs between
        # forward and checkpoint replay (backward).  Computing here ensures
        # each branch's chunk size is a fixed checkpoint input.
        near_chunk_sizes: dict[str, int] = {
            name: self.kernels[name]._auto_chunk_size(
                dual_plan.n_near, source_points.device
            )
            for name in self.reference_length_names
        }

        ### Decide whether branch-level checkpointing is worthwhile.
        # Each branch accumulates ~34 bytes/near-pair of autograd state
        # (int64 checkpoint-saved indices + multiply/scatter graph nodes).
        # Branch checkpointing avoids holding all branches' graphs
        # simultaneously, which is essential at large N (800k+ faces)
        # but a pure compute overhead at small N (20k faces).
        _AUTOGRAD_BYTES_PER_PAIR = 34
        n_branches = len(self.reference_length_names)
        use_branch_ckpt = False
        if self.training and self.use_gradient_checkpointing and n_branches > 1:
            n_total_pairs = dual_plan.n_near + dual_plan.n_nf + dual_plan.n_fn
            per_branch_bytes = n_total_pairs * _AUTOGRAD_BYTES_PER_PAIR
            all_branches_bytes = per_branch_bytes * n_branches
            if device.type == "cuda":
                free_bytes = torch.cuda.mem_get_info(device)[0]
                use_branch_ckpt = all_branches_bytes > free_bytes * 0.1
            else:
                use_branch_ckpt = False

            if not torch.compiler.is_compiling():
                logger.debug(
                    "branch checkpoint: %s (est. %.1f MB/branch, "
                    "%.1f MB all branches, %.1f MB free, %d branches)",
                    "ENABLED" if use_branch_ckpt else "DISABLED",
                    per_branch_bytes / 1e6,
                    all_branches_bytes / 1e6,
                    free_bytes / 1e6 if device.type == "cuda" else 0,
                    n_branches,
                )

        ### Evaluate each branch with the shared tree, plan, and aggregates.
        # When enabled, branch-level checkpointing ensures only ONE branch's
        # autograd graph exists at a time during backward, preventing
        # autograd memory from accumulating across all branches.
        results_pieces: list[TensorDict[str, Float[torch.Tensor, "n_targets ..."]]] = []
        for name in self.reference_length_names:
            with record_function(f"multiscale_kernel::branch/{name}"):
                ref_length = (
                    reference_lengths[name]
                    * torch.exp(self.log_scalefactors[name])
                )
                strengths = source_strengths[name]
                chunk_size = near_chunk_sizes[name]
                kernel = self.kernels[name]
                if use_branch_ckpt:
                    results_pieces.append(
                        checkpoint(
                            kernel,
                            use_reentrant=False,
                            reference_length=ref_length,
                            source_points=source_points,
                            target_points=target_points,
                            source_strengths=strengths,
                            source_data=source_data,
                            global_data=global_data,
                            theta=theta,
                            cluster_tree=cluster_tree,
                            target_tree=target_tree,
                            dual_plan=dual_plan,
                            source_areas=source_areas,
                            source_aggregates=source_aggregates,
                            near_chunk_size=chunk_size,
                        )
                    )
                else:
                    results_pieces.append(
                        kernel(
                            reference_length=ref_length,
                            source_points=source_points,
                            target_points=target_points,
                            source_strengths=strengths,
                            source_data=source_data,
                            global_data=global_data,
                            theta=theta,
                            cluster_tree=cluster_tree,
                            target_tree=target_tree,
                            dual_plan=dual_plan,
                            source_areas=source_areas,
                            source_aggregates=source_aggregates,
                            near_chunk_size=chunk_size,
                        )
                    )

        result: TensorDict[str, Float[torch.Tensor, "n_targets ..."]] = reduce(
            operator.add, results_pieces
        )

        return result
