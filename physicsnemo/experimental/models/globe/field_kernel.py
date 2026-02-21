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
from physicsnemo.experimental.models.globe.mlp import MLP
from physicsnemo.experimental.models.globe.pade import Pade
from physicsnemo.nn.functional.equivariant_ops import (
    legendre_polynomials,
    polar_and_dipole_basis,
    smooth_log,
    spherical_basis,
)
from physicsnemo.experimental.models.globe.utilities.tensordict_utils import (
    combine_tensordicts,
    concatenate_leaves,
    concatenated_length,
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
    output_fields : dict[str, Literal["scalar", "vector"]]
        Dictionary mapping field names to their types. Each field type must be
        either ``"scalar"`` or ``"vector"``. Vector fields have dimension equal
        to ``n_spatial_dims``.
    n_source_scalars : int, optional, default=0
        Number of scalar properties expected per source point.
    n_source_vectors : int, optional, default=0
        Number of vector properties expected per source point.
    n_global_scalars : int, optional, default=0
        Number of global scalar properties.
    n_global_vectors : int, optional, default=0
        Number of global vector properties.
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
    source_scalars : TensorDict or None, optional, default=None
        Physically-meaningful scalars associated with each source point with
        batch_size :math:`(N_{sources},)`. All scalars should be dimensionless.
    source_vectors : TensorDict or None, optional, default=None
        Physically-meaningful vectors associated with each source point with
        batch_size :math:`(N_{sources}, D)`. All vectors should be dimensionless.
    global_scalars : TensorDict or None, optional, default=None
        Global physically-meaningful scalars with batch_size :math:`()`. All
        scalars should be dimensionless.
    global_vectors : TensorDict or None, optional, default=None
        Global physically-meaningful vectors with batch_size :math:`(D,)`. All
        vectors should be dimensionless.
    verbose : bool, optional, default=False
        Whether to print progress information during evaluation.

    Outputs
    -------
    TensorDict
        TensorDict with batch_size :math:`(N_{targets},)` containing the computed
        fields. Each scalar field has shape :math:`(N_{targets},)` and each vector
        field has shape :math:`(N_{targets}, D)`.
    """

    def __init__(
        self,
        *,
        n_spatial_dims: int,
        output_fields: dict[str, Literal["scalar", "vector"]],
        n_source_scalars: int = 0,
        n_source_vectors: int = 0,
        n_global_scalars: int = 0,
        n_global_vectors: int = 0,
        smoothing_radius: float = 1e-8,
        hidden_layer_sizes: Sequence[int] | None = None,
        n_spherical_harmonics: int = 4,
        network_type: Literal["pade", "mlp"] = "pade",
        spectral_norm: bool = False,
        use_gradient_checkpointing: bool = True,
    ):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64]

        super().__init__()

        self.n_spatial_dims = n_spatial_dims
        self.output_fields = output_fields
        self.n_source_scalars = n_source_scalars
        self.n_source_vectors = n_source_vectors
        self.n_global_scalars = n_global_scalars
        self.n_global_vectors = n_global_vectors
        self.smoothing_radius = smoothing_radius
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_spherical_harmonics = n_spherical_harmonics
        self.use_gradient_checkpointing = use_gradient_checkpointing

        if network_type == "pade":
            self.network = Pade(
                layer_sizes=self.network_layer_sizes,
                numerator_order=2,
                denominator_order=2,
                use_separate_mlps=False,
                share_denominator_across_channels=False,
                spectral_norm=spectral_norm,
            )
        elif network_type == "mlp":
            self.network = nn.Sequential(
                MLP(
                    layer_sizes=self.network_layer_sizes,
                    activation_function=nn.SiLU(),
                    spectral_norm=spectral_norm,
                ),
                nn.Tanh(),
            )
        else:
            raise ValueError(
                f"Invalid network type: {network_type=!r}; must be one of ['pade', 'mlp']"
            )

    @cached_property
    def network_layer_sizes(self) -> tuple[int, ...]:
        r"""Computes the neural network layer sizes for the kernel function.

        This cached property determines the input and output dimensions of the kernel's
        internal neural network based on the number of scalar/vector inputs and output
        fields. The calculation follows the invariant feature engineering pipeline
        described in paper Section 3.2.2.

        Input features (first layer):

        1. Raw source and global scalars (``n_source_scalars`` + ``n_global_scalars``)
        2. Smoothed log-magnitudes of all vectors (``n_vectors_in``), where vectors include:

           - The relative position vector ``r = (x_target - x_source) / reference_length``
           - Source-associated vectors (e.g., face normals, boundary condition values)
           - Global vectors (e.g., freestream velocity direction)

        3. Pairwise spherical harmonic features for all vector pairs: for each of the
           ``C(n_vectors_in, 2)`` pairs, we compute ``n_spherical_harmonics`` Legendre
           polynomial terms weighted by the product of the pair's smoothed log-magnitudes

        Output features (last layer):

        1. Direct scalar outputs (one channel per scalar field)
        2. Vector reprojection coefficients for each vector field: for each vector output,
           we generate coefficients for projection onto a local basis consisting of:

           - The radial direction ``r_hat`` (1 basis vector)
           - For each non-r input vector: axis direction + dipole/polar direction
             (2 basis vectors per non-r vector)

        Returns
        -------
        tuple[int, ...]
            Tuple of layer sizes:
            ``(first_layer_size, *hidden_layer_sizes, last_layer_size)``.
            The hidden layer sizes are those provided during initialization.

        Notes
        -----
        This property is cached, so it's only computed once per kernel instance.
        The calculation ensures rotation-invariance by working with invariant scalar
        features in the MLP and restoring directionality through basis reprojection.
        """
        ### Inputs
        n_vectors_in: int = (
            1  # r
            + self.n_source_vectors
            + self.n_global_vectors
        )
        n_scalars_in: int = self.n_source_scalars + self.n_global_scalars
        n_vector_pairs_in: int = comb(n_vectors_in, 2)

        first_layer_size = (
            n_scalars_in  # Scalars
            + n_vectors_in  # Smoothed magnitudes of all vectors
            + n_vector_pairs_in * self.n_spherical_harmonics
        )

        ### Outputs
        n_scalars_out = sum(v == "scalar" for v in self.output_fields.values())
        n_vectors_out = sum(v == "vector" for v in self.output_fields.values())
        last_layer_size = n_scalars_out + n_vectors_out * (
            1  # r_hat
            + 2 * (n_vectors_in - 1)  # All non-r vectors
        )

        return (
            first_layer_size,
            *self.hidden_layer_sizes,
            last_layer_size,
        )

    def add_semantics(
        self,
        tensor: Float[torch.Tensor, "... total_dims"],
        shape_for_scalars: torch.Size | None = None,
        shape_for_vectors: torch.Size | None = None,
    ) -> TensorDict:
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
        TensorDict
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

        def get_shape(field_type: Literal["scalar", "vector"]) -> torch.Size:
            if field_type == "scalar":
                return shape_for_scalars  # type: ignore[return-value]
            elif field_type == "vector":
                return shape_for_vectors  # type: ignore[return-value]
            else:
                raise ValueError(
                    f"Got {field_type=!r}; this must be one of ['scalar', 'vector']"
                )

        output_field_shapes: dict[str, torch.Size] = {
            field_name: get_shape(field_type)
            for field_name, field_type in self.output_fields.items()
        }

        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            if not tensor.shape[-1] == sum(
                prod(shape) for shape in output_field_shapes.values()
            ):
                raise ValueError(
                    f"Expected an array with length {sum(prod(shape) for shape in output_field_shapes.values())} along dimension -1;\n"
                    f"got {tensor.shape=!r}."
                )

        batch_size = tensor.shape[:-1]
        result = TensorDict(
            {},
            batch_size=batch_size,
        )

        i: int = 0
        for field_name, shape in sorted(output_field_shapes.items()):
            field_width = prod(shape)
            # Use slice to get a view, not a copy
            slc = [slice(None)] * (tensor.ndim - 1) + [slice(i, i + field_width)]
            result[field_name] = tensor[tuple(slc)].reshape(batch_size + shape)
            i += field_width
        return result

    def forward(
        self,
        *,
        reference_length: Float[torch.Tensor, ""],
        source_points: Float[torch.Tensor, "n_sources n_dims"],
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        source_strengths: Float[torch.Tensor, " n_sources"] | None = None,
        source_scalars: TensorDict | None = None,
        source_vectors: TensorDict | None = None,
        global_scalars: TensorDict | None = None,
        global_vectors: TensorDict | None = None,
        verbose: bool = False,
    ) -> TensorDict:
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
        source_scalars : TensorDict or None, optional
            TensorDict with batch_size :math:`(N_{sources},)`.
            Physically-meaningful scalars associated with each source point. All
            scalars should be dimensionless. The total concatenated length must
            match ``n_source_scalars``. Defaults to empty TensorDict if ``None``.
        source_vectors : TensorDict or None, optional
            TensorDict with batch_size :math:`(N_{sources}, D)`.
            Physically-meaningful vectors associated with each source point. All
            vectors should be dimensionless. Common examples include normal vectors
            and direction vectors. The total concatenated length must match
            ``n_source_vectors``. Defaults to empty TensorDict if ``None``.
        global_scalars : TensorDict or None, optional
            TensorDict with batch_size :math:`()`. Global
            physically-meaningful scalars. All scalars should be dimensionless.
            The total concatenated length must match ``n_global_scalars``. Defaults
            to empty TensorDict if ``None``.
        global_vectors : TensorDict or None, optional
            TensorDict with batch_size :math:`(D,)`.
            Global physically-meaningful vectors. All vectors should be
            dimensionless. The total concatenated length must match ``n_global_vectors``.
            Defaults to empty TensorDict if ``None``.
        verbose : bool, optional
            Whether to print progress information during evaluation.
            Default is False.

        Returns
        -------
        TensorDict
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
        if source_scalars is None:
            source_scalars = TensorDict(
                {}, batch_size=torch.Size([n_sources]), device=device
            )
        if source_vectors is None:
            source_vectors = TensorDict(
                {},
                batch_size=torch.Size([n_sources, self.n_spatial_dims]),
                device=device,
            )
        if global_scalars is None:
            global_scalars = TensorDict({}, batch_size=torch.Size([]), device=device)
        if global_vectors is None:
            global_vectors = TensorDict(
                {}, batch_size=torch.Size([self.n_spatial_dims]), device=device
            )

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
            for name, (actual, expected) in {
                "source_scalars": (
                    concatenated_length(source_scalars),
                    self.n_source_scalars,
                ),
                "source_vectors": (
                    concatenated_length(source_vectors),
                    self.n_source_vectors,
                ),
                "global_scalars": (
                    concatenated_length(global_scalars),
                    self.n_global_scalars,
                ),
                "global_vectors": (
                    concatenated_length(global_vectors),
                    self.n_global_vectors,
                ),
            }.items():
                if actual != expected:
                    raise ValueError(
                        f"This kernel was instantiated to expect {expected} {name},\n"
                        f"but the forward-method input gives {actual} {name}."
                    )

        ### Assemble inputs to the neural network
        scalars = TensorDict(
            {
                "source_scalars": source_scalars.expand(
                    n_targets, *source_scalars.batch_size
                ),
                "global_scalars": global_scalars.expand(
                    n_targets, n_sources, *global_scalars.batch_size
                ),
            },
            batch_size=torch.Size([n_targets, n_sources]),
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
            batch_size=torch.Size([n_targets, n_sources, self.n_spatial_dims]),
            device=device,
        )
        vectors["r"] = (
            target_points[:, None, :]  # shape (n_targets, 1, n_dims)
            - source_points[None, :, :]  # shape (1, n_sources, n_dims)
        ) / reference_length  # shape (n_targets, n_sources, n_dims)

        # At this point, cast to the autocast dtype if possible and we're
        # currently in an autocast context. This saves tons of memory, and
        # really the only reason we needed to keep fp32 up to this point was to
        # prevent catastrophic cancellation on the `r` vector computation.
        if torch.is_autocast_enabled(device.type):
            dtype = torch.get_autocast_dtype(device.type)
            scalars = scalars.to(dtype=dtype)
            vectors = vectors.to(dtype=dtype)
        else:
            dtype = None

        smoothing_radius = torch.tensor(
            self.smoothing_radius, device=device, dtype=dtype
        )
        vectors_mag_squared = (
            (vectors * vectors).sum(dim=-1).apply(lambda x: x + smoothing_radius**2)
        )
        vectors_mag = vectors_mag_squared.sqrt()
        vectors_hat = vectors / vectors_mag.unsqueeze(-1)
        vectors_log_mag = smooth_log(vectors_mag)

        # Each of the vectors' magnitudes become an input feature
        scalars["vectors_log_mag"] = vectors_log_mag

        # TODO in 3D, add cross products of pairs of vectors as input features

        ### Now, engineer some features from pairs of vectors
        keypairs = list(itertools.combinations(range(concatenated_length(vectors)), 2))
        k1, k2 = zip(*keypairs) if keypairs else ([], [])
        vectors_hat_concatenated: torch.Tensor = concatenate_leaves(vectors_hat)
        # shape: (n_targets, n_sources, n_spatial_dims, n_vectors_in)

        v1_hat = vectors_hat_concatenated[:, :, :, k1]
        v2_hat = vectors_hat_concatenated[:, :, :, k2]
        cos_theta_pairs = torch.sum(v1_hat * v2_hat, dim=-2)
        # shape: (n_targets, n_sources, len(keypairs))

        spherical_harmonics: list[torch.Tensor] = legendre_polynomials(
            x=cos_theta_pairs, n=self.n_spherical_harmonics + 1
        )[1:]

        vectors_mag_concatenated: torch.Tensor = concatenate_leaves(vectors_mag)
        v1_mag = vectors_mag_concatenated[:, :, k1]
        v2_mag = vectors_mag_concatenated[:, :, k2]

        for i, harmonics in enumerate(spherical_harmonics):
            scalars[f"pairwise_spherical_harmonics_{i}"] = (
                smooth_log(v1_mag * v2_mag) * harmonics
            )

        cat_input_tensors: torch.Tensor = concatenate_leaves(scalars)
        # shape (n_targets, n_sources, self.network_layer_sizes[0])

        ### Evaluate the neural-network-based field kernel function
        if not torch.compiler.is_compiling():
            if not cat_input_tensors.shape[-1] == self.network_layer_sizes[0]:
                raise RuntimeError(
                    f"The input tensor has {cat_input_tensors.shape[-1]=!r} features, but the network expects {self.network_layer_sizes[0]=!r} input features.\n"
                    f"This is due to a shape inconsistency between the `network_layer_sizes` and `forward` methods of the {self.__class__.__name__!r} class."
                )

        flattened_input = cat_input_tensors.view(
            n_targets * n_sources, self.network_layer_sizes[0]
        )

        if self.training and self.use_gradient_checkpointing:
            flattened_output = checkpoint(
                self.network, flattened_input, use_reentrant=False
            )  # shape (n_targets * n_sources, last_layer_size)
        else:
            flattened_output = self.network(flattened_input)

        output = flattened_output.view(
            n_targets, n_sources, self.network_layer_sizes[-1]
        )

        ### Enforces correct far-field decay rate
        output = output * (
            -torch.expm1(-vectors_mag_squared["r"][..., None])
        )  # Lamb-Oseen vortex kernel, numerically stable using expm1
        if self.n_spatial_dims == 2:
            output = output / (vectors_mag_squared["r"][..., None] + 1).sqrt()
        elif self.n_spatial_dims == 3:
            output = output / (vectors_mag_squared["r"][..., None] + 1)
        else:
            output = output / (vectors_mag_squared["r"][..., None] + 1) ** (
                (self.n_spatial_dims - 1) / 2
            )

        ### Add semantics to the output
        n_vectors_in = len(vectors.keys(include_nested=True, leaves_only=True))
        result: TensorDict = self.add_semantics(
            output,
            shape_for_scalars=(),
            shape_for_vectors=(
                1  # r_hat
                + 2 * (n_vectors_in - 1),  # All non-r vectors
            ),
        )
        # Values are tensors of shape (n_targets, n_sources, field_dim), where
        # field_dim is taken from the `size_for_` arguments above.

        ### Vector Reprojection
        # If there are any vector fields, we want to interpret them as a vector
        # field on a local basis defined by the vectors we already have - this
        # preserves rotational invariance.

        vector_reprojection_needed = any(
            field_type == "vector" for field_type in self.output_fields.values()
        )

        if vector_reprojection_needed:
            ### Compute the local basis vectors
            # Note that each combination of source and target points yields its
            # own basis. In both 2D and 3D, we take the axis of the coordinate
            # system used to generate the basis vectors to be `source_vectors` -
            # a convenient source of non-arbitrary direction. We then repeat
            # this for each source vector, and stack them together.

            # This is effectively an expanded version of a Helmholtz
            # decomposition for vector fields: each field is the sum of a
            # uniform field, a source field, a solenoidal field, and a
            # dipole-like field.

            basis_vector_components: list[torch.Tensor] = []
            # Eventually, this is a list of length 3 * n_source_vectors (in both
            # 2D and 3D) with tensors of shape (n_targets, n_sources,
            # n_spatial_dims)

            basis_vector_components.append(vectors_hat["r"])

            for k in vectors.keys(include_nested=True, leaves_only=True):  # ty: ignore[not-iterable]
                if k == "r":
                    continue

                scale = vectors_log_mag[k][..., None]

                basis_vector_components.append(scale * vectors_hat[k])

                if self.n_spatial_dims == 2:
                    # In 2D, we use a polar/dipole basis: e_r is radial, e_theta
                    # is tangential (orthogonal to e_r), and e_kappa is a
                    # dipole-like direction (orthogonal to e_r, parallel to
                    # e_theta). This basis is not a true vector basis (it has 3
                    # vectors, not 2), but this third basis vector increases
                    # expressivity.
                    _, e_theta, e_kappa = polar_and_dipole_basis(
                        r_hat=vectors_hat["r"],
                        n_hat=vectors_hat[k],
                        normalize_basis_vectors=False,
                    )  # shape (n_targets, n_sources, 2)

                    basis_vector_components.extend(
                        [
                            # scale * e_theta,  # Vortex-like direction
                            scale * e_kappa,  # Dipole-like direction
                        ]
                    )

                elif self.n_spatial_dims == 3:
                    # In 3D, we use a modified spherical coordinate basis: e_r
                    # is radial, e_theta is the polar / dipole-like / "latitude"
                    # direction, and e_phi is the azimuthal / vortex-like /
                    # "longitude" direction.
                    _, e_theta, e_phi = spherical_basis(
                        r_hat=vectors_hat["r"],
                        n_hat=vectors_hat[k],
                        normalize_basis_vectors=False,
                    )  # Shape of each: (n_targets, n_sources, 3)

                    basis_vector_components.extend(
                        [
                            scale * e_theta,  # Polar / meridional direction
                            # scale * e_phi,  # Vortex-like / azimuthal direction
                        ]
                    )

                else:
                    raise NotImplementedError(
                        f"The {self.__class__.__name__!r} class does not support {self.n_spatial_dims=!r}-dimensional problems."
                    )

            basis_vectors = torch.stack(basis_vector_components, dim=-1)
            # shape (n_targets, n_sources, n_spatial_dims, 4 * n_vectors)

            ### Now, reproject each vector field onto the basis vectors
            for field_name in self.output_fields:
                if self.output_fields[field_name] == "vector":
                    # # ORIGINAL (SLOW) - keeping for reference, as this is the most readable version
                    # # Axes: t = target, s = source, d = dim, b = basis vector id
                    # result[field_name] = torch.einsum(
                    #     "tsb,tsdb->tsd",
                    #     result[field_name],
                    #     basis_vectors,
                    # )

                    # OPTIMIZED VERSION: Manual broadcast matrix-vector multiplication
                    # This is ~16x faster than the original einsum and uses less memory
                    result[field_name] = torch.sum(
                        basis_vectors
                        * result[field_name].unsqueeze(-2),  # Broadcasting
                        dim=-1,
                    )

        # Incorporate the source strengths and sum over all source points
        # Axes: t = target, s = source, ... = all remaining dimensions (i.e., n_spatial_dims for vectors, nothing for scalars)
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


class ChunkedKernel(Kernel):
    r"""Memory-efficient kernel evaluation through automatic target point chunking.

    :class:`ChunkedKernel` extends the base :class:`Kernel` class with chunking
    capabilities that enable memory-efficient evaluation on large target point sets.
    The kernel evaluation has ``O(n_sources * n_targets)`` memory complexity due to
    the all-to-all pairwise computation, which can exhaust GPU memory for large
    problems. Chunking processes target points in smaller batches, trading modest
    computational overhead for dramatic memory reduction.

    Chunking is particularly useful in three scenarios:

    1. **Training**: When using downsampled query points (e.g., 4096 points) but many
       source faces, chunking can reduce memory during the backward pass.
    2. **Inference on dense grids**: When evaluating on complete high-resolution volume
       meshes (e.g., 100k+ points), chunking prevents out-of-memory errors.
    3. **Limited GPU memory**: When running on GPUs with constrained memory (e.g., during
       development or deployment on smaller hardware).

    The chunking is implemented at the target point dimension, so each chunk independently
    computes its output from all source points, then results are concatenated. This is
    numerically identical to non-chunked evaluation - there are no approximations.

    Chunk size selection:

    - ``chunk_size=None``: No chunking, fastest but highest memory (default for small
      problems)
    - ``chunk_size="auto"``: Automatically determines size targeting ~1GB per chunk
    - ``chunk_size=int``: Manual specification for fine control

    The ``"auto"`` mode estimates memory based on network layer sizes and interaction
    count, providing a good balance for most use cases. The implementation uses recursive
    calls to handle the chunking logic, and the overhead is minimal for reasonable chunk
    sizes.

    Inherits all other functionality from :class:`Kernel`, including invariant feature
    engineering, Pade-approximant networks, far-field decay, and equivariant vector
    reprojection.

    Parameters
    ----------
    Inherits all parameters from :class:`Kernel`.

    Forward
    -------
    Same parameters as :class:`Kernel`, with the addition of:

    chunk_size : None or int or {"auto"}, optional, default="auto"
        Controls chunking behavior. ``"auto"`` determines chunk size targeting
        ~1GB per chunk. An integer processes in exact chunk sizes. ``None``
        evaluates all at once.

    Outputs
    -------
    TensorDict
        TensorDict with batch_size :math:`(N_{targets},)` containing the computed
        fields. Numerically identical to non-chunked :class:`Kernel` evaluation.

    Examples
    --------
    >>> # For a large problem with 1M query points:
    >>> kernel = ChunkedKernel(
    ...     n_spatial_dims=3,
    ...     output_fields={"pressure": "scalar"},
    ...     n_source_vectors=1,
    ...     hidden_layer_sizes=[64, 64],
    ... )
    >>> # Evaluate with automatic chunking to prevent OOM
    >>> result = kernel(
    ...     source_points=boundary_centers,  # e.g., 10k faces
    ...     target_points=volume_points,     # e.g., 1M points
    ...     reference_length=torch.tensor(1.0),
    ...     source_vectors=TensorDict({"normal": normals}, ...),
    ...     chunk_size="auto",  # Will process in chunks of ~10-20k points
    ... )

    Notes
    -----
    During training, chunking has limited benefit because PyTorch's autograd must
    store all intermediate activations regardless. Memory reduction is most effective
    during inference (with ``torch.no_grad()``) where chunking can reduce peak usage
    by orders of magnitude.
    """

    def forward(
        self,
        *,
        reference_length: Float[torch.Tensor, ""],
        source_points: Float[torch.Tensor, "n_sources n_dims"],
        target_points: Float[torch.Tensor, "n_targets n_dims"],
        source_strengths: Float[torch.Tensor, " n_sources"] | None = None,
        source_scalars: TensorDict | None = None,
        source_vectors: TensorDict | None = None,
        global_scalars: TensorDict | None = None,
        global_vectors: TensorDict | None = None,
        verbose: bool = False,
        chunk_size: None | int | Literal["auto"] = "auto",
    ) -> TensorDict:
        r"""Evaluates the kernel with optional chunking for memory efficiency.

        Parameters
        ----------
        chunk_size : None or int or {"auto"}, optional
            Controls chunking behavior:

            - ``"auto"``: Automatically determine chunk size based on estimated memory
              usage, targeting approximately 1GB per chunk.
            - ``int``: Process target points in chunks of exactly this size.
            - ``None``: No chunking, evaluate all target points at once.

        **kernel_kwargs
            All arguments accepted by :meth:`Kernel.forward`, including:
            ``reference_length``, ``source_points``, ``target_points``,
            ``source_strengths``, ``source_scalars``, ``source_vectors``,
            ``global_scalars``, ``global_vectors``, ``verbose``.

        Returns
        -------
        TensorDict
            TensorDict mapping field names to computed tensors.
            Each scalar field has shape :math:`(N_{targets},)` and each vector field
            has shape :math:`(N_{targets}, D)`.
        """
        n_sources: int = len(source_points)
        n_targets: int = len(target_points)
        n_interactions: int = n_targets * n_sources

        if chunk_size == "auto":
            # Automatically determine chunk size based on memory constraints
            # Assumes the network is dominating the memory cost
            approx_n_floats = n_interactions * sum(self.network_layer_sizes)
            approx_n_bytes = (
                approx_n_floats * 4
            )  # float32; conservative enough for bfloat16 too
            approx_memory_gb = approx_n_bytes / (1024**3)
            target_memory_gb = 1.0

            n_chunks_needed = max(1, ceil(approx_memory_gb / target_memory_gb))
            chunk_size: int = max(1, ceil(n_targets / n_chunks_needed))

            if verbose:
                print(f"Auto-chunking: {chunk_size=!r}, {n_chunks_needed=!r}")

            return self.forward(
                reference_length=reference_length,
                source_points=source_points,
                target_points=target_points,
                source_strengths=source_strengths,
                source_scalars=source_scalars,
                source_vectors=source_vectors,
                global_scalars=global_scalars,
                global_vectors=global_vectors,
                verbose=verbose,
                chunk_size=chunk_size,
            )

        elif isinstance(chunk_size, int):
            result_pieces: list[TensorDict] = []

            start_indices = range(0, n_targets, chunk_size)

            if not torch.compiler.is_compiling() and verbose:
                start_indices = tqdm.tqdm(
                    start_indices,
                    desc="Evaluating kernel in chunks",
                    unit=" chunks",
                )

            for start_idx in start_indices:
                end_idx = min(start_idx + chunk_size, n_targets)
                target_points_chunk = target_points[start_idx:end_idx]

                # Recursive call with chunk_size=None to avoid further chunking
                chunk_result = self.forward(
                    reference_length=reference_length,
                    source_points=source_points,
                    target_points=target_points_chunk,
                    source_strengths=source_strengths,
                    source_scalars=source_scalars,
                    source_vectors=source_vectors,
                    global_scalars=global_scalars,
                    global_vectors=global_vectors,
                    verbose=verbose,
                    chunk_size=None,
                )

                result_pieces.append(chunk_result)

            result = TensorDict.cat(result_pieces, dim=0)

            return result

        elif chunk_size is None:
            return super().forward(
                reference_length=reference_length,
                source_points=source_points,
                target_points=target_points,
                source_strengths=source_strengths,
                source_scalars=source_scalars,
                source_vectors=source_vectors,
                global_scalars=global_scalars,
                global_vectors=global_vectors,
                verbose=verbose,
            )

        else:
            raise ValueError(
                f"Got {chunk_size=!r}; this must be one of ['auto', int, None]"
            )


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
    output_fields : dict[str, Literal["scalar", "vector"]]
        Dictionary mapping field names to types (``"scalar"`` or ``"vector"``).
    reference_length_names : Sequence[str]
        Sequence of identifiers for reference length scales. Each creates an
        independent kernel branch. Examples: ``["viscous", "geometric"]``.
    n_source_scalars : int, optional, default=0
        Number of scalar features per source face.
    n_source_vectors : int, optional, default=0
        Number of vector features per source face.
    n_global_scalars : int, optional, default=0
        Number of global scalar features (excluding auto-added length ratios).
    n_global_vectors : int, optional, default=0
        Number of global vector features.
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
    source_strengths : TensorDict or None, optional, default=None
        Per-source, per-branch strength values. TensorDict keyed by
        ``reference_length_names``. Defaults to all ones.
    source_scalars : TensorDict or None, optional, default=None
        Scalars per source point with batch_size :math:`(N_{sources},)`.
    source_vectors : TensorDict or None, optional, default=None
        Vectors per source point with batch_size :math:`(N_{sources}, D)`.
    global_scalars : TensorDict or None, optional, default=None
        Global scalar features (auto-augmented with log length ratios).
    global_vectors : TensorDict or None, optional, default=None
        Global vector features with batch_size :math:`(D,)`.
    verbose : bool, optional, default=False
        Whether to print progress.
    chunk_size : None or int or {"auto"}, optional, default="auto"
        Chunking behavior.

    Outputs
    -------
    TensorDict
        TensorDict with the summed results from all kernel branches. Each scalar
        field has shape :math:`(N_{targets},)` and each vector field has shape
        :math:`(N_{targets}, D)`.

    Examples
    --------
    >>> kernel = MultiscaleKernel(
    ...     n_spatial_dims=2,
    ...     output_fields={"phi": "scalar", "u": "vector"},
    ...     reference_length_names=["viscous_length", "chord_length"],
    ...     n_source_vectors=1,  # e.g., normal vector
    ...     hidden_layer_sizes=[64, 64],
    ... )
    >>> # At runtime, provide reference lengths as a dict:
    >>> result = kernel(
    ...     source_points=boundary_face_centers,
    ...     target_points=query_points,
    ...     reference_lengths={"viscous_length": torch.tensor(0.001),
    ...                        "chord_length": torch.tensor(1.0)},
    ...     source_vectors=TensorDict({"normal": normals}, ...),
    ...     source_strengths=TensorDict({"viscous_length": strengths_v,
    ...                                  "chord_length": strengths_c}, ...),
    ... )
    """

    def __init__(
        self,
        *,
        n_spatial_dims: int,
        output_fields: dict[str, Literal["scalar", "vector"]],
        reference_length_names: Sequence[str],
        n_source_scalars: int = 0,
        n_source_vectors: int = 0,
        n_global_scalars: int = 0,
        n_global_vectors: int = 0,
        smoothing_radius: float = 1e-8,
        hidden_layer_sizes: Sequence[int] | None = None,
        n_spherical_harmonics: int = 4,
        network_type: Literal["pade", "mlp"] = "pade",
        spectral_norm: bool = False,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.n_spatial_dims = n_spatial_dims
        self.output_fields = output_fields
        self.reference_length_names = reference_length_names
        self.n_source_scalars = n_source_scalars
        self.n_source_vectors = n_source_vectors
        self.n_global_scalars = n_global_scalars
        self.n_global_vectors = n_global_vectors
        self.smoothing_radius = smoothing_radius
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_spherical_harmonics = n_spherical_harmonics
        self.network_type = network_type
        self.spectral_norm = spectral_norm
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.kernels = nn.ModuleDict(
            {
                name: ChunkedKernel(
                    n_spatial_dims=n_spatial_dims,
                    output_fields=output_fields,
                    n_source_scalars=n_source_scalars,
                    n_source_vectors=n_source_vectors,
                    n_global_scalars=n_global_scalars
                    + comb(len(reference_length_names), 2),
                    n_global_vectors=n_global_vectors,
                    smoothing_radius=smoothing_radius,
                    hidden_layer_sizes=hidden_layer_sizes,
                    n_spherical_harmonics=n_spherical_harmonics,
                    network_type=network_type,
                    spectral_norm=spectral_norm,
                    use_gradient_checkpointing=use_gradient_checkpointing,
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
        source_strengths: TensorDict | None = None,
        source_scalars: TensorDict | None = None,
        source_vectors: TensorDict | None = None,
        global_scalars: TensorDict | None = None,
        global_vectors: TensorDict | None = None,
        verbose: bool = False,
        chunk_size: None | int | Literal["auto"] = "auto",
    ) -> TensorDict:
        r"""Evaluates the multiscale kernel by combining results from multiple scales.

        This method evaluates each constituent kernel at its respective reference length
        (scaled by a learnable factor), automatically adds log-ratios of reference lengths
        as global scalars, and sums the results across all scales.

        Parameters
        ----------
        reference_lengths : dict[str, torch.Tensor]
            Mapping of reference length names to scalar tensors.
        source_points : Float[torch.Tensor, "n_sources n_dims"]
            Tensor of shape :math:`(N_{sources}, D)`. Physical coordinates of
            the source points.
        target_points : Float[torch.Tensor, "n_targets n_dims"]
            Tensor of shape :math:`(N_{targets}, D)`. Physical coordinates of
            the target points.
        source_strengths : TensorDict or None, optional
            Per-source, per-branch strength values, keyed by
            ``reference_length_names``. Defaults to all ones.
        source_scalars : TensorDict or None, optional
            TensorDict with batch_size :math:`(N_{sources},)`.
        source_vectors : TensorDict or None, optional
            TensorDict with batch_size :math:`(N_{sources}, D)`.
        global_scalars : TensorDict or None, optional
            TensorDict with batch_size :math:`()`. Will be augmented with log
            reference length ratios.
        global_vectors : TensorDict or None, optional
            TensorDict with batch_size :math:`(D,)`.
        verbose : bool, optional
            Whether to print progress information. Default is False.
        chunk_size : None or int or {"auto"}, optional
            Chunking behavior passed to :meth:`ChunkedKernel.forward`.
            Default is ``"auto"``.

        Returns
        -------
        TensorDict
            Dictionary mapping field names to the summed results from all kernels.
            Each scalar field has shape :math:`(N_{targets},)` and each vector field
            has shape :math:`(N_{targets}, D)`.
        """
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
        if source_scalars is None:
            source_scalars = TensorDict(
                {}, batch_size=torch.Size([n_sources]), device=device
            )
        if source_vectors is None:
            source_vectors = TensorDict(
                {},
                batch_size=torch.Size([n_sources, self.n_spatial_dims]),
                device=device,
            )
        if global_scalars is None:
            global_scalars = TensorDict({}, batch_size=torch.Size([]), device=device)
        if global_vectors is None:
            global_vectors = TensorDict(
                {}, batch_size=torch.Size([self.n_spatial_dims]), device=device
            )

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

        name_pairs = list(itertools.combinations(self.reference_length_names, 2))
        global_scalars = combine_tensordicts(
            global_scalars,
            TensorDict(
                {
                    f"log_reference_length_ratio_{k1}_{k2}": (
                        reference_lengths[k1] / reference_lengths[k2]
                    ).log()
                    for k1, k2 in name_pairs
                },
                batch_size=torch.Size([]),
                device=device,
            ),
        )

        results_pieces: list[TensorDict] = [
            self.kernels[name](
                reference_length=reference_lengths[name]
                * torch.exp(self.log_scalefactors[name]),
                source_points=source_points,
                target_points=target_points,
                source_strengths=source_strengths[name],
                source_scalars=source_scalars,
                source_vectors=source_vectors,
                global_scalars=global_scalars,
                global_vectors=global_vectors,
                verbose=verbose,
                chunk_size=chunk_size,
            )
            for name in self.reference_length_names
        ]

        result: TensorDict = reduce(operator.add, results_pieces)

        return result
