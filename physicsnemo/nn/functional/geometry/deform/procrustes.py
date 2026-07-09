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

"""Backend-dispatched Procrustes point registration."""

from __future__ import annotations

from typing import Literal

import torch
from jaxtyping import Float

from physicsnemo.core.function_spec import FunctionSpec

from ._procrustes_torch import procrustes_torch
from ._utils import normalize_procrustes_inputs, restore_procrustes_rank
from ._warp_impl import procrustes_warp


class Procrustes(FunctionSpec):
    r"""Register corresponding point sets with a similarity transformation.

    This functional solves the ordinary least-squares, orientation-preserving
    Procrustes problem. Given corresponding source and target points, it returns
    proper orthogonal rotation :math:`R`, translation :math:`t`, and
    nonnegative isotropic scale :math:`s` such that

    .. math::

       y_i \approx s R x_i + t.

    PhysicsNeMo stores points as row vectors. Apply ``rotation`` as
    ``source @ rotation.transpose(-1, -2)``. Then multiply by
    ``scale[..., None, None]`` and add ``translation[..., None, :]``. The
    operation accepts unbatched ``(N, D)`` or aligned batched ``(B, N, D)``
    inputs for one-, two-, and three-dimensional coordinates. Source and target
    batches are aligned and are not broadcast.

    Parameters
    ----------
    source : torch.Tensor
        Source points with shape ``(N, D)`` or ``(B, N, D)``, where
        ``D`` is 1, 2, or 3.
    target : torch.Tensor
        Corresponding target points with exactly the same shape, dtype, and
        device as ``source``.
    scale : bool, optional
        Whether to estimate an isotropic scale. If ``False``, the returned
        scale tensor contains ones. Default is ``True``.
    implementation : {"warp", "torch"} or None, optional
        Explicit backend. ``None`` selects Torch on CPU and Warp on CUDA when
        Warp is available, otherwise Torch with a one-time
        :class:`RuntimeWarning`.

    Returns
    -------
    rotation : torch.Tensor
        Proper orthogonal rotation matrices with shape ``(D, D)`` or
        ``(B, D, D)``.
    translation : torch.Tensor
        Translation vectors with shape ``(D,)`` or ``(B, D)``.
    scale_factor : torch.Tensor
        Nonnegative isotropic scale factors with shape ``(B,)``. An unbatched
        input returns a scalar tensor.

    Raises
    ------
    TypeError
        If inputs have unsupported types or dtypes, or ``scale`` is not bool.
    ValueError
        If input shapes or devices differ, the coordinate dimension is not 1,
        2, or 3, or there are too few points.

    Notes
    -----
    Float32 and float64 inputs are supported. Both backends support first-order
    reverse-mode autograd and :func:`torch.compile`; second-order and
    forward-mode derivatives are not supported. Configurations whose optimal
    proper rotation is not unique, such as sufficiently rank-deficient point
    sets, do not have a well-defined rotation gradient and may produce
    non-finite gradients.
    """

    _FORWARD_BENCHMARK_CASES = (
        ("small-n256-d2-rigid", 1, 256, 2, False),
        ("medium-b4-n4096-d3-similarity", 4, 4096, 3, True),
        ("large-b8-n16384-d3-similarity", 8, 16384, 3, True),
    )
    _BACKWARD_BENCHMARK_CASES = (
        ("medium-n4096-d3-rigid", 1, 4096, 3, False),
        ("medium-b4-n4096-d3-similarity", 4, 4096, 3, True),
    )
    _COMPARE_ATOL = 2.0e-5
    _COMPARE_RTOL = 2.0e-5
    _COMPARE_BACKWARD_ATOL = 2.0e-4
    _COMPARE_BACKWARD_RTOL = 2.0e-4

    @FunctionSpec.register(name="warp", required_imports=("warp>=1.14.0",), rank=0)
    def warp_forward(
        source: Float[torch.Tensor, "*batch num_points num_dims"],
        target: Float[torch.Tensor, "*batch num_points num_dims"],
        *,
        scale: bool = True,
    ) -> tuple[
        Float[torch.Tensor, "*batch num_dims num_dims"],
        Float[torch.Tensor, "*batch num_dims"],
        Float[torch.Tensor, "*batch"],
    ]:
        """Register corresponding point sets with the Warp backend."""

        source_b3, target_b3, was_unbatched = normalize_procrustes_inputs(
            source, target, scale
        )
        output = procrustes_warp(source_b3, target_b3, scale)
        return restore_procrustes_rank(*output, was_unbatched=was_unbatched)

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        source: Float[torch.Tensor, "*batch num_points num_dims"],
        target: Float[torch.Tensor, "*batch num_points num_dims"],
        *,
        scale: bool = True,
    ) -> tuple[
        Float[torch.Tensor, "*batch num_dims num_dims"],
        Float[torch.Tensor, "*batch num_dims"],
        Float[torch.Tensor, "*batch"],
    ]:
        """Register corresponding point sets with the Torch backend."""

        source_b3, target_b3, was_unbatched = normalize_procrustes_inputs(
            source, target, scale
        )
        output = procrustes_torch(source_b3, target_b3, scale)
        return restore_procrustes_rank(*output, was_unbatched=was_unbatched)

    @classmethod
    def dispatch(
        cls,
        source: Float[torch.Tensor, "*batch num_points num_dims"],
        target: Float[torch.Tensor, "*batch num_points num_dims"],
        *,
        scale: bool = True,
        implementation: Literal["torch", "warp"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "*batch num_dims num_dims"],
        Float[torch.Tensor, "*batch num_dims"],
        Float[torch.Tensor, "*batch"],
    ]:
        """Select Warp for CUDA inputs and Torch for CPU inputs by default.

        Falling back to Torch on CUDA inputs because Warp is unavailable emits
        the standard one-time :class:`RuntimeWarning`.
        """

        if implementation is None:
            impls = cls._get_impls()
            warp_impl = impls.get("warp")
            if isinstance(source, torch.Tensor) and source.is_cuda:
                if warp_impl is not None and warp_impl.available:
                    implementation = "warp"
                else:
                    cls._warn_fallback(warp_impl, impls["torch"])
                    implementation = "torch"
            else:
                implementation = "torch"
        return super().dispatch(
            source,
            target,
            scale=scale,
            implementation=implementation,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark cases."""
        device = torch.device(device)
        for seed, (
            label,
            batch_size,
            n_points,
            n_spatial_dims,
            estimate_scale,
        ) in enumerate(cls._FORWARD_BENCHMARK_CASES):
            generator = torch.Generator(device=device).manual_seed(2401 + seed)
            shape = (
                (n_points, n_spatial_dims)
                if batch_size == 1
                else (batch_size, n_points, n_spatial_dims)
            )
            source = torch.randn(shape, generator=generator, device=device)
            target = torch.randn(shape, generator=generator, device=device)
            yield label, (source, target), {"scale": estimate_scale}

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark cases."""
        device = torch.device(device)
        for seed, (
            label,
            batch_size,
            n_points,
            n_spatial_dims,
            estimate_scale,
        ) in enumerate(cls._BACKWARD_BENCHMARK_CASES):
            generator = torch.Generator(device=device).manual_seed(2501 + seed)
            shape = (
                (n_points, n_spatial_dims)
                if batch_size == 1
                else (batch_size, n_points, n_spatial_dims)
            )
            source = torch.randn(
                shape, generator=generator, device=device, requires_grad=True
            )
            target = torch.randn(
                shape, generator=generator, device=device, requires_grad=True
            )
            yield label, (source, target), {"scale": estimate_scale}

    @classmethod
    def compare_forward(
        cls,
        output: tuple[
            Float[torch.Tensor, "*batch num_dims num_dims"],
            Float[torch.Tensor, "*batch num_dims"],
            Float[torch.Tensor, "*batch"],
        ],
        reference: tuple[
            Float[torch.Tensor, "*batch num_dims num_dims"],
            Float[torch.Tensor, "*batch num_dims"],
            Float[torch.Tensor, "*batch"],
        ],
    ) -> None:
        """Compare registration outputs component-wise."""
        for actual, expected in zip(output, reference, strict=True):
            torch.testing.assert_close(
                actual,
                expected,
                atol=cls._COMPARE_ATOL,
                rtol=cls._COMPARE_RTOL,
            )

    @classmethod
    def compare_backward(
        cls,
        output: Float[torch.Tensor, "..."],
        reference: Float[torch.Tensor, "..."],
    ) -> None:
        """Compare registration gradients across backends."""
        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_BACKWARD_ATOL,
            rtol=cls._COMPARE_BACKWARD_RTOL,
        )


procrustes = Procrustes.make_function("procrustes")


__all__ = ["Procrustes", "procrustes"]
