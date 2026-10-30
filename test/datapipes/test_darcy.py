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


def _darcy_residual_reference(
    darcy: np.ndarray, permeability: np.ndarray, source: float, dx: float
) -> np.ndarray:
    """Residual of ``k lap(u) + grad(k) . grad(u) + source`` on the solver grid.

    Matches the solver's boundary handling: ``u`` is zero outside the array and
    ``k`` is clamped at the edges. Both gradients are central differences, so
    their product is divided by ``(2 dx) ** 2``.
    """
    u = np.pad(darcy.astype(np.float64), ((0, 0), (1, 1), (1, 1)))
    k = np.pad(permeability.astype(np.float64), ((0, 0), (1, 1), (1, 1)), mode="edge")
    c = np.s_[:, 1:-1, 1:-1]
    lap = (
        u[:, :-2, 1:-1] + u[:, 2:, 1:-1] + u[:, 1:-1, :-2] + u[:, 1:-1, 2:] - 4 * u[c]
    ) / dx**2
    cross = (
        (k[:, 2:, 1:-1] - k[:, :-2, 1:-1]) * (u[:, 2:, 1:-1] - u[:, :-2, 1:-1])
        + (k[:, 1:-1, 2:] - k[:, 1:-1, :-2]) * (u[:, 1:-1, 2:] - u[:, 1:-1, :-2])
    ) / (4 * dx**2)
    return k[c] * lap + cross + source


def _darcy_point_update_reference(
    darcy: np.ndarray,
    permeability: np.ndarray,
    source: float,
    dx: float,
    factor: int,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy reference for ``darcy_point_update_batched_2d`` on one multi-grid level.

    Returns the center value that zeroes the local residual of
    ``k lap(u) + grad(k) . grad(u) + source`` at each of the level's ``n x n``
    points, and the full-grid index of those points along each axis.
    """
    # padding by `factor` reproduces zero edges for darcy and clamped edges for
    # permeability
    gdx = dx * factor
    pad = ((0, 0), (factor, factor), (factor, factor))
    u = np.pad(darcy.astype(np.float64), pad)
    k = np.pad(permeability.astype(np.float64), pad, mode="edge")
    idx = factor * np.arange(n) + (factor - 1)
    gx, gy = (idx + factor)[:, None], (idx + factor)[None, :]
    u_n = [u[:, gx - factor, gy], u[:, gx + factor, gy]]
    u_n += [u[:, gx, gy - factor], u[:, gx, gy + factor]]
    k_c = k[:, gx, gy]
    t_1 = k_c * sum(u_n) / gdx**2
    t_2 = (k[:, gx + factor, gy] - k[:, gx - factor, gy]) * (u_n[1] - u_n[0])
    t_3 = (k[:, gx, gy + factor] - k[:, gx, gy - factor]) * (u_n[3] - u_n[2])
    expected = (t_1 + (t_2 + t_3) / (4 * gdx**2) + source) / (4 * k_c / gdx**2)
    return expected, idx


def _random_darcy_level(factor: int):
    """Random solution and permeability on a 17 x 17 grid, plus level launch dims."""
    resolution = 16
    dim = (2, resolution + 1, resolution + 1)
    rng = np.random.default_rng(0)
    darcy = rng.uniform(0.0, 0.1, size=dim).astype(np.float32)
    permeability = rng.uniform(0.5, 2.0, size=dim).astype(np.float32)
    # same launch dims Darcy2D uses for each multi-grid level
    if factor == 1:
        mgrid_dim = dim
    else:
        mgrid_dim = (dim[0], resolution // factor, resolution // factor)
    return darcy, permeability, dim, mgrid_dim, 1.0 / (resolution + 1)


@requires_module("warp")
@pytest.mark.parametrize("factor", [1, 2, 4])
def test_darcy_jacobi_kernel(factor, device, pytestconfig):
    """Regression test for NVIDIA/physicsnemo#2036.

    One Jacobi step must solve ``k lap(u) + grad(k) . grad(u) + f = 0`` for the
    center value, with the cross term scaled by ``1 / (2 dx) ** 2``.
    """
    import warp as wp

    from physicsnemo.datapipes.benchmarks.kernels.finite_difference import (
        darcy_mgrid_jacobi_iterative_batched_2d,
    )

    wp.init()
    darcy, permeability, dim, mgrid_dim, dx = _random_darcy_level(factor)
    source = 1.0

    src = wp.array(darcy, dtype=float, device=device)
    dst = wp.zeros(dim, dtype=float, device=device)
    perm = wp.array(permeability, dtype=float, device=device)
    wp.launch(
        kernel=darcy_mgrid_jacobi_iterative_batched_2d,
        dim=mgrid_dim,
        inputs=[src, dst, perm, source, dim[1], dim[2], dx, factor],
        device=device,
    )
    out = dst.numpy()

    expected, idx = _darcy_point_update_reference(
        darcy, permeability, source, dx, factor, mgrid_dim[1]
    )
    written = out[:, idx[:, None], idx[None, :]]
    assert np.allclose(written, expected, rtol=1e-4, atol=1e-6)


@requires_module("warp")
@pytest.mark.parametrize("color", [0, 1])
@pytest.mark.parametrize("factor", [1, 2, 4])
def test_darcy_rbsor_kernel(factor, color, device, pytestconfig):
    """One red-black SOR half-sweep relaxes only its color's points.

    Points of ``color`` on the multi-grid level move to
    ``(1 - omega) * u + omega * u_star``, where ``u_star`` zeroes the local
    Darcy residual. Every other point of the full grid is left untouched.
    """
    import warp as wp

    from physicsnemo.datapipes.benchmarks.kernels.finite_difference import (
        darcy_mgrid_rbsor_batched_2d,
    )

    wp.init()
    darcy, permeability, dim, mgrid_dim, dx = _random_darcy_level(factor)
    source = 1.0
    omega = 1.5

    arr = wp.array(darcy, dtype=float, device=device)
    perm = wp.array(permeability, dtype=float, device=device)
    wp.launch(
        kernel=darcy_mgrid_rbsor_batched_2d,
        dim=mgrid_dim,
        inputs=[arr, perm, source, dim[1], dim[2], dx, factor, color, omega],
        device=device,
    )
    out = arr.numpy()

    # neighbors have the opposite color, so the half-sweep reads only values
    # it does not write and the reference can use the input field
    u_star, idx = _darcy_point_update_reference(
        darcy, permeability, source, dx, factor, mgrid_dim[1]
    )
    level = darcy[:, idx[:, None], idx[None, :]].astype(np.float64)
    relaxed = (1.0 - omega) * level + omega * u_star
    x, y = np.meshgrid(np.arange(mgrid_dim[1]), np.arange(mgrid_dim[2]), indexing="ij")
    on_color = (x + y) % 2 == color

    expected = darcy.astype(np.float64)
    expected[:, idx[:, None], idx[None, :]] = np.where(on_color, relaxed, level)
    assert np.allclose(out, expected, rtol=1e-4, atol=1e-6)


@requires_module("warp")
@pytest.mark.parametrize(
    "min_permeability, max_permeability",
    [(0.4, 2.0), (0.0, 1.0), (-0.5, 1.0), (2.0, 0.5)],
)
def test_darcy_2d_rejects_unsolvable_permeability(
    min_permeability, max_permeability, device, pytestconfig
):
    """Permeability bounds the solver diverges for must be rejected up front."""
    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    with pytest.raises(ValueError, match="permeability"):
        Darcy2D(
            resolution=32,
            batch_size=1,
            min_permeability=min_permeability,
            max_permeability=max_permeability,
            device=device,
        )


@requires_module("warp")
def test_darcy_2d_solves_darcy_equation(device, pytestconfig):
    """Regression test for NVIDIA/physicsnemo#2036.

    The converged solver output must satisfy the discretized Darcy equation
    ``k lap(u) + grad(k) . grad(u) + 1 = 0`` everywhere on the solver grid.
    """
    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    np.random.seed(0)
    datapipe = Darcy2D(
        resolution=32,
        batch_size=2,
        nr_permeability_freq=5,
        max_permeability=2.0,
        min_permeability=0.5,
        max_iterations=30000,
        convergence_threshold=1e-8,
        iterations_per_convergence_check=1000,
        nr_multigrids=3,
        device=device,
    )
    next(iter(datapipe))

    # use the full solver grid; the returned fields are cropped by one cell
    residual = _darcy_residual_reference(
        datapipe.darcy0.numpy(), datapipe.permeability.numpy(), 1.0, datapipe.dx
    )
    # with the cross term scaled by 1 / (2 dx) instead of 1 / (2 dx) ** 2 the
    # residual is O(1) wherever the permeability changes
    assert np.abs(residual).max() < 1e-2


@requires_module("warp")
def test_init_uniform_random_4d_kernel(device, pytestconfig):
    """Regression test for NVIDIA/physicsnemo#2035.

    Every element must get its own random draw, not one per leading index.
    """
    import warp as wp

    from physicsnemo.datapipes.benchmarks.kernels.initialization import (
        init_uniform_random_4d,
    )

    wp.init()
    shape = (4, 3, 5, 5)
    array = wp.zeros(shape, dtype=float, device=device)
    wp.launch(
        kernel=init_uniform_random_4d,
        dim=shape,
        inputs=[array, -1.0, 1.0, 1234],
        device=device,
    )
    values = array.numpy()

    assert np.unique(values).size == values.size
    assert values.min() >= -1.0 and values.max() <= 1.0
    assert values.min() < 0.0 < values.max()


@requires_module("warp")
def test_darcy_2d_permeability_varies_across_batch(device, pytestconfig):
    """Regression test for NVIDIA/physicsnemo#2035.

    Each sample in a batch must get its own random Fourier coefficients, so no
    two permeability fields in a batch may be identical.
    """
    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    np.random.seed(0)
    batch_size = 4
    nr_freq = 5
    datapipe = Darcy2D(
        resolution=32,
        batch_size=batch_size,
        nr_permeability_freq=nr_freq,
        device=device,
    )
    datapipe.initialize_batch()

    # coefficients are laid out as (component, batch, freq, freq)
    coefficients = datapipe.rand_fourier.numpy()
    assert np.unique(coefficients).size == coefficients.size

    permeability = datapipe.permeability.numpy()
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            assert not np.array_equal(permeability[i], permeability[j])


def _small_darcy(device, **kwargs):
    """Small Darcy2D that converges tightly in well under a second."""
    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    params = dict(
        resolution=32,
        batch_size=2,
        max_iterations=30000,
        convergence_threshold=1e-8,
        iterations_per_convergence_check=1000,
        nr_multigrids=3,
        device=device,
    )
    params.update(kwargs)
    return Darcy2D(**params)


@requires_module("warp")
def test_mgrid_inf_residual_reports_non_finite(device, pytestconfig):
    """A NaN update must not be swallowed by the solver's convergence check.

    ``atomic_max`` ignores NaN, so without explicit handling a diverged solve
    looks converged.
    """
    import warp as wp

    from physicsnemo.datapipes.benchmarks.kernels.finite_difference import (
        mgrid_inf_residual_batched_2d,
    )

    wp.init()
    phi0 = np.zeros((1, 5, 5), dtype=np.float32)
    phi1 = np.zeros_like(phi0)
    phi1[0, 2, 3] = np.nan
    inf_res = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        kernel=mgrid_inf_residual_batched_2d,
        dim=phi0.shape,
        inputs=[
            wp.array(phi0, dtype=float, device=device),
            wp.array(phi1, dtype=float, device=device),
            inf_res,
            1,
        ],
        device=device,
    )
    assert inf_res.numpy()[0] >= 1e30


@requires_module("warp")
def test_darcy_2d_falls_back_to_gauss_seidel(device, pytestconfig, monkeypatch):
    """A diverged SOR solve is re-run with Gauss-Seidel, which must still solve
    the Darcy equation."""
    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    caps = []
    real_solve = Darcy2D._solve

    def solve_diverging_once(self, max_omega):
        caps.append(max_omega)
        if len(caps) == 1:
            return False
        return real_solve(self, max_omega)

    monkeypatch.setattr(Darcy2D, "_solve", solve_diverging_once)
    np.random.seed(0)
    datapipe = _small_darcy(device)
    with pytest.warns(UserWarning, match="Gauss-Seidel"):
        datapipe.generate_batch()

    assert caps[1] == 1.0
    residual = _darcy_residual_reference(
        datapipe.darcy0.numpy(), datapipe.permeability.numpy(), 1.0, datapipe.dx
    )
    assert np.abs(residual).max() < 1e-2


@requires_module("warp")
def test_darcy_2d_raises_if_fallback_diverges(device, pytestconfig, monkeypatch):
    """Diverged data must never be returned silently."""
    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    monkeypatch.setattr(Darcy2D, "_solve", lambda self, max_omega: False)
    datapipe = _small_darcy(device)
    with pytest.warns(UserWarning, match="Gauss-Seidel"):
        with pytest.raises(RuntimeError, match="diverged"):
            datapipe.generate_batch()


@requires_module("warp")
def test_darcy_2d_is_reproducible(device, pytestconfig):
    """The same numpy seed must reproduce a batch exactly; a different seed must
    not."""

    def batch(seed):
        np.random.seed(seed)
        data = next(iter(_small_darcy(device)))
        return data["permeability"].cpu().numpy(), data["darcy"].cpu().numpy()

    first, again, other = batch(0), batch(0), batch(1)
    for a, b in zip(first, again):
        assert np.array_equal(a, b)
    assert not np.array_equal(first[0], other[0])


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
