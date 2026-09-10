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

from typing import Tuple

import numpy as np
import pytest
import torch

from test.conftest import requires_module

from . import common

Tensor = torch.Tensor


def _bilinear_upsample_reference(coarse: np.ndarray, factor: int) -> np.ndarray:
    """Numpy reference for ``bilinear_upsample_batched_2d``.

    Coarse nodes sit at indices ``k * factor - 1``. Anything outside the array is
    a zero Dirichlet boundary, including the coarse boundary node at ``-1`` and
    the one past the last coarse node.
    """
    b, lx, ly = coarse.shape

    def value(bi, x, y):
        if 0 <= x < lx and 0 <= y < ly:
            return coarse[bi, x, y]
        return 0.0

    out = np.zeros_like(coarse)
    for bi in range(b):
        for x in range(lx):
            x0 = ((x + 1) // factor) * factor - 1
            x1 = x0 + factor
            rx = (x - x0) / factor
            for y in range(ly):
                y0 = ((y + 1) // factor) * factor - 1
                y1 = y0 + factor
                ry = (y - y0) / factor
                d_x0 = (1 - rx) * value(bi, x0, y0) + rx * value(bi, x1, y0)
                d_x1 = (1 - rx) * value(bi, x0, y1) + rx * value(bi, x1, y1)
                out[bi, x, y] = (1 - ry) * d_x0 + ry * d_x1
    return out


@requires_module("warp")
@pytest.mark.parametrize("factor", [2, 4, 8])
def test_bilinear_upsample_kernel(factor, device, pytestconfig):
    import warp as wp

    from physicsnemo.datapipes.benchmarks.kernels.utils import (
        bilinear_upsample_batched_2d,
    )

    wp.init()
    resolution = 16
    dim = (2, resolution + 1, resolution + 1)

    # Only coarse nodes carry data; sample a bilinear polynomial there so the
    # interior of the upsampled field must reproduce it exactly.
    rng = np.random.default_rng(0)
    a, b_, c, d = rng.uniform(-1.0, 1.0, size=4)
    xs = np.arange(resolution + 1, dtype=np.float64)
    poly = a + b_ * xs[:, None] + c * xs[None, :] + d * xs[:, None] * xs[None, :]
    coarse = np.zeros(dim, dtype=np.float32)
    coarse_idx = np.arange(factor - 1, resolution, factor)
    coarse[:, coarse_idx[:, None], coarse_idx[None, :]] = poly[
        coarse_idx[:, None], coarse_idx[None, :]
    ]
    coarse[1] *= 0.5

    src = wp.array(coarse, dtype=float, device=device)
    dst = wp.full(dim, 1.0e20, dtype=float, device=device)
    wp.launch(
        kernel=bilinear_upsample_batched_2d,
        dim=dim,
        inputs=[src, dst, dim[1], dim[2], factor],
        device=device,
    )
    out = dst.numpy()

    expected = _bilinear_upsample_reference(coarse, factor)
    assert np.allclose(out, expected, rtol=1e-5, atol=1e-5)

    # interior fine nodes bracketed by coarse nodes on all sides must match the
    # polynomial exactly (bilinear interpolation is exact for bilinear data)
    lo, hi = coarse_idx[0], coarse_idx[-1]
    interior = np.s_[lo : hi + 1, lo : hi + 1]
    assert np.allclose(out[0][interior], poly[interior], rtol=1e-4, atol=1e-4)
    assert np.allclose(out[1][interior], 0.5 * poly[interior], rtol=1e-4, atol=1e-4)


@requires_module("warp")
@pytest.mark.parametrize("nr_multigrids", [3, 4])
def test_darcy_2d_multigrid_stays_in_bounds(nr_multigrids, device, pytestconfig):
    """Regression test for NVIDIA/physicsnemo#1958.

    Alias the logical (1, 65, 65) solver buffers into padded allocations whose
    halo is filled with a huge canary. A bounds-correct solver can never observe
    the halo, so the returned field must stay finite and physically plausible.
    """
    import warp as wp

    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    resolution = 64
    logical = resolution + 1
    padded = logical + 8  # covers the factor-8 far neighbor at index 71
    canary = 1.0e20

    datapipe = Darcy2D(
        resolution=resolution,
        batch_size=1,
        nr_permeability_freq=5,
        max_permeability=2.0,
        min_permeability=0.5,
        max_iterations=1000,
        convergence_threshold=1e-4,
        iterations_per_convergence_check=50,
        nr_multigrids=nr_multigrids,
        normaliser={"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        device=device,
    )

    parents = []
    for name in ("darcy0", "darcy1"):
        parent = wp.full((1, padded, padded), canary, dtype=float, device=device)
        parents.append(parent)
        view = wp.array(
            ptr=parent.ptr,
            dtype=float,
            shape=(1, logical, logical),
            strides=parent.strides,
            capacity=parent.capacity,
            device=device,
        )
        setattr(datapipe, name, view)

    darcy = next(iter(datapipe))["darcy"]
    assert torch.isfinite(darcy).all()
    # unnormalised pressure for this problem is O(1e-2); anything near the
    # canary magnitude means the halo leaked into the solve
    assert darcy.abs().max().item() < 1.0


@requires_module("warp")
def test_darcy_2d_constructor(device, pytestconfig):
    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    # construct data pipe
    datapipe = Darcy2D(
        resolution=64,
        batch_size=1,
        nr_permeability_freq=5,
        max_permeability=2.0,
        min_permeability=0.5,
        max_iterations=300,
        convergence_threshold=1e-4,
        iterations_per_convergence_check=5,
        nr_multigrids=4,
        normaliser={"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        device=device,
    )

    # iterate datapipe is iterable
    assert common.check_datapipe_iterable(datapipe)


@requires_module("warp")
def test_darcy_2d_device(device, pytestconfig):
    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    # construct data pipe
    datapipe = Darcy2D(
        resolution=64,
        batch_size=1,
        nr_permeability_freq=5,
        max_permeability=2.0,
        min_permeability=0.5,
        max_iterations=300,
        convergence_threshold=1e-4,
        iterations_per_convergence_check=5,
        nr_multigrids=4,
        normaliser={"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        device=device,
    )

    # iterate datapipe is iterable
    for data in datapipe:
        assert common.check_datapipe_device(data["permeability"], device)
        assert common.check_datapipe_device(data["darcy"], device)
        break


@requires_module("warp")
@pytest.mark.parametrize("resolution", [128, 64])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_darcy_2d_shape(resolution, batch_size, device, pytestconfig):
    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    # construct data pipe
    datapipe = Darcy2D(
        resolution=resolution,
        batch_size=batch_size,
        nr_permeability_freq=5,
        max_permeability=2.0,
        min_permeability=0.5,
        max_iterations=300,
        convergence_threshold=1e-4,
        iterations_per_convergence_check=5,
        nr_multigrids=3,
        normaliser={"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        device=device,
    )

    # test single sample
    for data in datapipe:
        permeability = data["permeability"]
        darcy = data["darcy"]

        # check batch size
        assert common.check_batch_size([permeability, darcy], batch_size)

        # check channels
        assert common.check_channels([permeability, darcy], 1, axis=1)

        # check grid dims
        assert common.check_grid(
            [permeability, darcy], (resolution, resolution), axis=(2, 3)
        )
        break


@requires_module("warp")
def test_darcy_cudagraphs(device, pytestconfig):
    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    # CUDA only:
    if device == "cpu":
        pytest.skip("CUDA only")

    # Preprocess function to convert dataloader output into Tuple of tensors
    def input_fn(data) -> Tuple[Tensor, ...]:
        return (data["permeability"], data["darcy"])

    # construct data pipe
    datapipe = Darcy2D(
        resolution=64,
        batch_size=1,
        nr_permeability_freq=5,
        max_permeability=2.0,
        min_permeability=0.5,
        max_iterations=300,
        convergence_threshold=1e-4,
        iterations_per_convergence_check=5,
        nr_multigrids=4,
        normaliser={"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        device=device,
    )

    assert common.check_cuda_graphs(datapipe, input_fn)
