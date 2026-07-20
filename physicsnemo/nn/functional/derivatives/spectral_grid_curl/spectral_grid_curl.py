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

from __future__ import annotations

from collections.abc import Sequence

import torch

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import spectral_grid_curl_torch


class SpectralGridCurl(FunctionSpec):
    r"""Compute curl on a periodic grid with Fourier differentiation.

    The input is a channel-first 2D or 3D vector field. For 2D inputs this
    functional returns scalar vorticity. For 3D inputs it returns the
    channel-first vector curl.

    Parameters
    ----------
    vector_field : torch.Tensor
        Vector field with shape ``(2, n0, n1)`` or ``(3, n0, n1, n2)``.
    lengths : float | Sequence[float], optional
        Physical domain lengths per spatial axis. A scalar applies the same
        length to every axis.
    implementation : {"torch"} or None
        Implementation to use. When ``None``, dispatch selects the available
        implementation.

    Returns
    -------
    torch.Tensor
        Scalar curl with shape ``grid_shape`` for 2D inputs, or vector curl
        with shape ``(3, *grid_shape)`` for 3D inputs.
    """

    _BENCHMARK_CASES = (
        ("2d-512x512", (512, 512), (2.0, 1.5)),
        ("3d-128x128x128", (128, 128, 128), (2.0, 1.5, 1.25)),
    )
    _BACKWARD_CASES = (
        ("2d-grad-256x256", (256, 256), (2.0, 1.5)),
        ("3d-grad-64x64x64", (64, 64, 64), (2.0, 1.5, 1.25)),
    )

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(
        vector_field: torch.Tensor,
        lengths: float | Sequence[float] = 1.0,
    ) -> torch.Tensor:
        """Dispatch spectral curl to the PyTorch backend."""
        return spectral_grid_curl_torch(
            vector_field=vector_field,
            lengths=lengths,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark input cases."""
        device = torch.device(device)
        for label, shape, lengths in cls._BENCHMARK_CASES:
            yield (
                label,
                (_make_periodic_vector_field(shape, device=device),),
                {"lengths": lengths},
            )

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark input cases."""
        device = torch.device(device)
        for label, shape, lengths in cls._BACKWARD_CASES:
            vector_field = (
                _make_periodic_vector_field(shape, device=device)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            yield label, (vector_field,), {"lengths": lengths}


def _make_periodic_vector_field(
    shape: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Construct a smooth periodic vector field for benchmark cases."""
    axes = tuple(
        torch.arange(size, device=device, dtype=torch.float32) / float(size)
        for size in shape
    )
    mesh = torch.meshgrid(*axes, indexing="ij")
    if len(shape) == 2:
        x0, x1 = mesh
        return torch.stack(
            (
                torch.sin(2.0 * torch.pi * x1),
                torch.cos(2.0 * torch.pi * x0),
            ),
            dim=0,
        )

    x0, x1, x2 = mesh
    return torch.stack(
        (
            torch.sin(2.0 * torch.pi * x1),
            torch.cos(2.0 * torch.pi * x2),
            torch.sin(2.0 * torch.pi * x0),
        ),
        dim=0,
    )


spectral_grid_curl = SpectralGridCurl.make_function("spectral_grid_curl")

__all__ = ["SpectralGridCurl", "spectral_grid_curl"]
