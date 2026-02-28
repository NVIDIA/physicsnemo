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

from typing import overload

import torch
from jaxtyping import Float
from tensordict import TensorDict


@overload
def smooth_log(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]: ...


@overload
def smooth_log(x: TensorDict) -> TensorDict: ...


def smooth_log(
    x: Float[torch.Tensor, "..."] | TensorDict,
) -> Float[torch.Tensor, "..."] | TensorDict:
    r"""Performs an elementwise operation on ``x`` with the following properties:

    - ``f(x) -> 0`` as ``x -> 0``
    - ``f(x) -> ln(x)`` as ``x -> infinity``
    - ``f(x)`` is smooth (``C_infty`` continuous) for all ``x >= 0``
    - ``f(x)`` is monotonically increasing for ``x > 0``
    - Has "nicely-behaved" higher-order derivatives for all ``x >= 0``

    Function is "intended" to be used with the domain ``x`` in ``[0, inf)``; technically it
    remains well-defined for ``(-1, inf)``.

    Parameters
    ----------
    x : Float[torch.Tensor, "..."] or TensorDict
        Input tensor or TensorDict with non-negative values.

    Returns
    -------
    Float[torch.Tensor, "..."] or TensorDict
        Result of the smooth log operation, same type and shape as ``x``.
    """
    return (-x).expm1().neg() * x.log1p()


@overload
def legendre_polynomials(
    x: Float[torch.Tensor, "..."], n: int
) -> list[Float[torch.Tensor, "..."]]: ...


@overload
def legendre_polynomials(x: TensorDict, n: int) -> list[TensorDict]: ...


def legendre_polynomials(
    x: Float[torch.Tensor, "..."] | TensorDict, n: int
) -> list[Float[torch.Tensor, "..."] | TensorDict]:
    r"""Computes the first ``n`` Legendre polynomials evaluated at ``x``.

    Acts elementwise on all entries of ``x``.

    Uses the recurrence relation for efficiency::

        P_0(x) = 1
        P_1(x) = x
        (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)

    Parameters
    ----------
    x : Float[torch.Tensor, "..."] or TensorDict
        Input tensor of any shape.
    n : int
        Number of Legendre polynomials to compute (must be >= 0).
        Returns ``P_0`` through ``P_{n-1}``.

    Returns
    -------
    list[Float[torch.Tensor, "..."] or TensorDict]
        List of ``n`` tensors, where the i-th tensor is ``P_i(x)`` with the same
        shape as ``x``. Returns an empty list when ``n = 0``.

    Raises
    ------
    ValueError
        If ``n`` is negative.

    Examples
    --------
    >>> x = torch.tensor([0.0, 0.5, 1.0])
    >>> polys = legendre_polynomials(x, 4)
    >>> # polys[0] is P_0(x) = 1
    >>> # polys[1] is P_1(x) = x
    >>> # polys[2] is P_2(x) = (3x^2 - 1)/2
    >>> # polys[3] is P_3(x) = (5x^3 - 3x)/2
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n=}")
    if n == 0:
        return []

    ### Seed with the two base cases; slice to handle n=1
    polynomials: list[Float[torch.Tensor, "..."] | TensorDict] = [
        torch.ones_like(x),  # P_0(x) = 1  # type: ignore[invalid-argument-type]
        x,  # P_1(x) = x
    ][:n]

    ### Recurrence relation for P_2 and beyond
    for i in range(2, n):
        # (i)*P_i(x) = (2i-1)*x*P_{i-1}(x) - (i-1)*P_{i-2}(x)
        p_i = ((2 * i - 1) * x * polynomials[i - 1] - (i - 1) * polynomials[i - 2]) / i  # type: ignore[operator]
        polynomials.append(p_i)

    return polynomials


def vector_project(
    v: Float[torch.Tensor, "... n_dims"],
    n_hat: Float[torch.Tensor, "... n_dims"],
) -> Float[torch.Tensor, "... n_dims"]:
    r"""Projects vector ``v`` onto the plane orthogonal to unit vector ``n_hat``.

    Uses the Gram-Schmidt orthogonalization:

    .. math::

        v_{\perp} = v - (v \cdot \hat{n}) \hat{n}

    Parameters
    ----------
    v : Float[torch.Tensor, "... n_dims"]
        Input vectors to project, with shape :math:`(*, D)`.
    n_hat : Float[torch.Tensor, "... n_dims"]
        Unit normal vectors defining the projection plane, with shape :math:`(*, D)`.

    Returns
    -------
    Float[torch.Tensor, "... n_dims"]
        Projected vectors with shape :math:`(*, D)`.
    """
    # Below are two equivalent implementations; on my machine the second is faster, but
    # in general einsums can be optimized more due to limiting intermediate allocations.
    # return v - torch.einsum("...i,...i->...", v, n_hat)[..., None] * n_hat
    return v - (v * n_hat).sum(dim=-1, keepdim=True) * n_hat


def polar_and_dipole_basis(
    r_hat: Float[torch.Tensor, "... 2"],
    n_hat: Float[torch.Tensor, "... 2"],
    normalize_basis_vectors: bool = True,
) -> tuple[
    Float[torch.Tensor, "... 2"],
    Float[torch.Tensor, "... 2"],
    Float[torch.Tensor, "... 2"],
]:
    r"""Computes a local vector basis for 2D vectors that is rotation-invariant
    w.r.t. ``n_hat``.

    Notably, this isn't a true vector basis, as it has 3 vectors, not the
    required 2. The basis is essentially a combination of a polar basis (r,
    theta) and an additional dipole-like direction (kappa) for the third vector.
    The axis for the dipole direction is set by ``n_hat``.

    Parameters
    ----------
    r_hat : Float[torch.Tensor, "... 2"]
        Unit direction vectors with shape :math:`(*, 2)`, assumed to be
        normalized.
    n_hat : Float[torch.Tensor, "... 2"]
        Axis vectors with shape :math:`(*, 2)`, assumed to be unit vectors.
    normalize_basis_vectors : bool, optional, default=True
        Whether to normalize ``e_kappa`` to be unit length
        (``e_r`` and ``e_theta`` are always unit). If ``False``, ``e_kappa`` is essentially
        multiplied by ``sin(theta)``. This gives the sometimes-useful property that
        ``e_kappa`` smoothly changes on the surface of a unit circle.

    Returns
    -------
    tuple[Float[torch.Tensor, "... 2"], Float[torch.Tensor, "... 2"], Float[torch.Tensor, "... 2"]]
        A tuple of 3 vectors, each of shape :math:`(*, 2)`:

        - ``e_r``: The radial direction, aligned with ``r_hat``. This corresponds to the
          influence field direction associated with a point source (i.e.,
          outwards from the origin).

        - ``e_theta``: The tangential direction, orthogonal to ``e_r``. This corresponds
          to the vortex field direction associated with a point vortex (i.e.,
          the direction of circulation around the source).

        - ``e_kappa``: A dipole-like direction. Notably, this is orthogonal to ``e_r``,
          but exactly parallel to ``e_theta`` - if you need to construct a full-rank
          basis, this is the one to drop.

    Note
    ----
    Edge Cases (even if ``normalize_basis_vectors`` is ``True``):

    - If ``r_hat`` is a zero vector, all basis vectors will be zero vectors.
    - If ``r_hat`` is aligned with ``n_hat``, ``e_kappa`` will be a zero vector.
    """
    # Validate input shapes
    if not torch.compiler.is_compiling():
        shape_validations = {
            "n_hat": (n_hat.shape[-1], 2),
            "r_hat": (r_hat.shape[-1], 2),
        }
        for name, (actual, expected) in shape_validations.items():
            if actual != expected:
                raise ValueError(
                    f"Expected {name} to have shape (..., {expected}), got shape {actual}."
                )

    # e_r is simply the input unit vector
    e_r = r_hat

    # Compute e_theta, the basis vector in the tangential direction
    e_theta = torch.stack([-r_hat[..., 1], r_hat[..., 0]], dim=-1)

    # Compute e_kappa, the basis vector in the dipole direction
    e_kappa = vector_project(-n_hat, r_hat)
    r_hat_is_zero = torch.all(r_hat == 0.0, dim=-1)
    e_kappa[r_hat_is_zero] = 0.0
    if normalize_basis_vectors:
        norm = torch.linalg.norm(e_kappa, dim=-1)
        e_kappa = e_kappa / norm[..., None]
        e_kappa[norm == 0] = 0.0  # Overwrites any NaNs with zero vectors

    return e_r, e_theta, e_kappa


def spherical_basis(
    r_hat: Float[torch.Tensor, "... 3"],
    n_hat: Float[torch.Tensor, "... 3"],
    normalize_basis_vectors: bool = True,
) -> tuple[
    Float[torch.Tensor, "... 3"],
    Float[torch.Tensor, "... 3"],
    Float[torch.Tensor, "... 3"],
]:
    r"""Computes a local vector basis for 3D vectors that is rotation-invariant
    w.r.t. ``n_hat``.

    The basis is essentially a spherical coordinate system, with the axis set by
    ``n_hat``.

    Parameters
    ----------
    r_hat : Float[torch.Tensor, "... 3"]
        Unit direction vectors with shape :math:`(*, 3)`, assumed to be
        normalized.
    n_hat : Float[torch.Tensor, "... 3"]
        Axis vectors with shape :math:`(*, 3)`, assumed to be unit vectors.
    normalize_basis_vectors : bool, optional, default=True
        Whether to normalize ``e_theta`` and ``e_phi`` to unit
        length (``e_r`` is always unit). If ``False``, ``e_theta`` and ``e_phi`` are
        essentially multiplied by ``sin(theta)``. This gives the
        sometimes-useful property that the basis vectors smoothly change on
        the surface of a unit sphere. (If ``e_theta`` and ``e_phi`` are normalized,
        then there is provably no possible way for these to smoothly vary on
        the surface of a sphere, as shown by the Hairy Ball theorem.)

    Returns
    -------
    tuple[Float[torch.Tensor, "... 3"], Float[torch.Tensor, "... 3"], Float[torch.Tensor, "... 3"]]
        A tuple of 3 vectors, each of shape :math:`(*, 3)`:

        - ``e_r``: The radial direction, pointing outward from the origin. This
          corresponds to the influence field direction associated with a point
          source.

        - ``e_theta``: The polar direction, orthogonal to both ``e_r`` and ``n_hat``. This
          corresponds to the meridional direction in spherical coordinates.

        - ``e_phi``: The azimuthal direction, orthogonal to both ``e_r`` and ``e_theta``.
          This corresponds to the circumferential direction in spherical
          coordinates.

    Note
    ----
    Edge Cases (even if ``normalize_basis_vectors`` is ``True``):

    - If ``r_hat`` is a zero vector, all basis vectors will be zero vectors.
    - If ``r_hat`` is aligned with ``n_hat``, ``e_theta`` and ``e_phi`` will be zero vectors.
    """
    # Validate input shapes
    if not torch.compiler.is_compiling():
        shape_validations = {
            "n_hat": (n_hat.shape[-1], 3),
            "r_hat": (r_hat.shape[-1], 3),
        }
        for name, (actual, expected) in shape_validations.items():
            if actual != expected:
                raise ValueError(
                    f"Expected {name} to have shape (..., {expected}), got shape {actual}."
                )

    # e_r is simply the input unit vector
    e_r = r_hat

    # Compute e_theta, the basis vector in the polar direction
    e_theta = vector_project(-n_hat, r_hat)
    r_hat_is_zero = torch.all(r_hat == 0.0, dim=-1)
    e_theta[r_hat_is_zero] = 0.0
    if normalize_basis_vectors:
        norm = torch.linalg.norm(e_theta, dim=-1)
        e_theta = e_theta / norm[..., None]
        e_theta[norm == 0] = 0.0  # Overwrites any NaNs with zero vectors

    # Compute e_phi, the basis vector in the azimuthal direction
    e_phi = torch.cross(e_r, e_theta, dim=-1)

    return e_r, e_theta, e_phi
