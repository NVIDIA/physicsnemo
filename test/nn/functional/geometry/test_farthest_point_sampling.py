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

import pytest
import torch

from physicsnemo.nn.functional import farthest_point_sampling
from physicsnemo.nn.functional.geometry import FarthestPointSampling


def _well_separated_cloud(device: str, n: int = 64, d: int = 3):
    """Deterministic, well-separated cloud (tie-free, backend-stable)."""
    torch.manual_seed(0)
    return torch.rand(n, d, device=device, dtype=torch.float32)


def _assert_fps_output(indices, num_samples, num_points, batch_size=None):
    if batch_size is None:
        assert indices.shape == (num_samples,)
    else:
        assert indices.shape == (batch_size, num_samples)
    assert indices.dtype == torch.int64
    assert (indices >= 0).all()
    assert (indices < num_points).all()
    # FPS never reselects a point for tie-free, num_samples <= N inputs.
    flat = indices.reshape(-1, num_samples)
    for row in flat:
        assert row.unique().numel() == num_samples


def _assert_greedy_fps_optimal(points, selected, rtol=1e-4):
    """Verify the greedy FPS invariant directly (independent oracle).

    For every pick after the first, the chosen point must — within ``rtol`` —
    maximize the running minimum distance to the already-selected set, i.e. be
    a valid greedy farthest point. The check runs in float64 so the float32
    backends are validated against a higher-precision reference; ``rtol``
    absorbs the float32-vs-float64 gap (a genuinely wrong pick fails by far
    more than ``rtol``).
    """
    sel = [int(i) for i in selected.tolist()]
    pts = points.to(torch.float64)
    min_d = ((pts - pts[sel[0]]) ** 2).sum(-1)
    for step in range(1, len(sel)):
        max_val = float(min_d.max())
        chosen = float(min_d[sel[step]])
        assert chosen >= max_val * (1.0 - rtol) - 1e-12, (
            f"step {step}: picked idx {sel[step]} with running min-dist "
            f"{chosen}, but the max available was {max_val}"
        )
        min_d = torch.minimum(min_d, ((pts - pts[sel[step]]) ** 2).sum(-1))


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_fps_known_answer_collinear(device: str, implementation: str):
    # Equally spaced points on a line. Starting at index 0, greedy FPS must
    # pick the far endpoint, then the midpoint: [0, m-1, (m-1)//2].
    if implementation == "warp" and "cpu" in device:
        pytest.skip("warp FPS backend is CUDA-only (tile reductions)")
    m = 9
    xs = torch.arange(m, dtype=torch.float32, device=device).reshape(m, 1)
    points = torch.cat([xs, torch.zeros(m, 2, device=device)], dim=1)  # (m, 3)
    idx = farthest_point_sampling(points, 3, implementation=implementation)
    assert idx.tolist() == [0, m - 1, (m - 1) // 2]  # [0, 8, 4]


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_fps_known_answer_outlier(device: str, implementation: str):
    # A tight cluster near the origin plus one far outlier (last index).
    # Starting in the cluster, the second pick must be the outlier.
    if implementation == "warp" and "cpu" in device:
        pytest.skip("warp FPS backend is CUDA-only (tile reductions)")
    torch.manual_seed(0)
    cluster = 0.01 * torch.rand(16, 3, device=device, dtype=torch.float32)
    outlier = torch.tensor([[100.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    points = torch.cat([cluster, outlier], dim=0)  # outlier at index 16
    idx = farthest_point_sampling(points, 2, implementation=implementation)
    assert idx.tolist() == [0, 16]


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize(
    "n, d, k",
    [
        (5, 3, 5),  # k == n (select all); tiny block
        (37, 3, 11),  # non-power-of-2 block_size < 512
        (200, 2, 64),  # D=2, mid size
        (513, 4, 128),  # n > 512 -> multi-point-per-lane strided scan
        (1024, 3, 256),  # large, n = 2 * block_size
    ],
)
def test_fps_greedy_optimality(device, implementation, n, d, k):
    # Randomized correctness against an independent oracle, across sizes, dims,
    # k (incl. k == n), and block-size regimes (n < / == / > the 512 lanes).
    if implementation == "warp" and "cpu" in device:
        pytest.skip("warp FPS backend is CUDA-only (tile reductions)")
    torch.manual_seed(20240611)
    points = torch.rand(n, d, device=device, dtype=torch.float32)
    selected = farthest_point_sampling(points, k, implementation=implementation)
    assert selected.unique().numel() == k  # no point reselected
    _assert_greedy_fps_optimal(points, selected)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_fps_greedy_optimality_batched(device, implementation):
    # Each cloud in a batch must be sampled correctly and independently (guards
    # against cross-cloud contamination in the per-block warp path).
    if implementation == "warp" and "cpu" in device:
        pytest.skip("warp FPS backend is CUDA-only (tile reductions)")
    torch.manual_seed(7)
    points = torch.rand(3, 256, 3, device=device, dtype=torch.float32)
    selected = farthest_point_sampling(points, 50, implementation=implementation)
    for b in range(points.shape[0]):
        assert selected[b].unique().numel() == 50
        _assert_greedy_fps_optimal(points[b], selected[b])


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("num_samples", [1, 8, 32])
def test_fps_unbatched(device: str, implementation: str, num_samples: int):
    if implementation == "warp" and "cpu" in device:
        pytest.skip("warp FPS backend is CUDA-only (tile reductions)")
    points = _well_separated_cloud(device, n=64)
    idx = farthest_point_sampling(points, num_samples, implementation=implementation)
    _assert_fps_output(idx, num_samples, 64)
    # Deterministic start at index 0.
    assert int(idx[0]) == 0


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_fps_batched(device: str, implementation: str):
    if implementation == "warp" and "cpu" in device:
        pytest.skip("warp FPS backend is CUDA-only (tile reductions)")
    torch.manual_seed(1)
    points = torch.rand(4, 100, 3, device=device, dtype=torch.float32)
    idx = farthest_point_sampling(points, 16, implementation=implementation)
    _assert_fps_output(idx, 16, 100, batch_size=4)
    assert (idx[:, 0] == 0).all()


def test_fps_backend_parity(device: str):
    """Torch and Warp select the same set for tie-free inputs."""
    if "cpu" in device:
        pytest.skip("warp FPS backend is CUDA-only (tile reductions)")
    points = _well_separated_cloud(device, n=128)
    out_torch = farthest_point_sampling(points, 40, implementation="torch")
    out_warp = farthest_point_sampling(points, 40, implementation="warp")
    FarthestPointSampling.compare_forward(out_warp, out_torch)


def test_fps_higher_dim(device: str):
    """FPS works for D != 3 (feature-space sampling)."""
    torch.manual_seed(2)
    points = torch.rand(50, 7, device=device, dtype=torch.float32)
    idx = farthest_point_sampling(points, 10, implementation="torch")
    _assert_fps_output(idx, 10, 50)


def test_fps_determinism(device: str):
    if "cpu" in device:
        pytest.skip("warp FPS backend is CUDA-only (tile reductions)")
    points = _well_separated_cloud(device, n=64)
    a = farthest_point_sampling(points, 20, implementation="warp")
    b = farthest_point_sampling(points, 20, implementation="warp")
    torch.testing.assert_close(a, b)


def test_fps_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(FarthestPointSampling.make_inputs_forward(device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    out = FarthestPointSampling.dispatch(*args, implementation="torch", **kwargs)
    assert out.ndim in (1, 2)
    assert out.dtype == torch.int64


def test_fps_compare_forward_contract(device: str):
    _, args, kwargs = next(iter(FarthestPointSampling.make_inputs_forward(device)))
    out = FarthestPointSampling.dispatch(*args, implementation="torch", **kwargs)
    FarthestPointSampling.compare_forward(out, out.detach().clone())


def test_fps_error_handling(device: str):
    points = _well_separated_cloud(device, n=16)
    with pytest.raises(ValueError, match="cannot exceed"):
        farthest_point_sampling(points, 32, implementation="torch")
    with pytest.raises(ValueError, match="must be >= 1"):
        farthest_point_sampling(points, 0, implementation="torch")
    with pytest.raises(ValueError, match="must be 2D"):
        farthest_point_sampling(
            points.unsqueeze(0).unsqueeze(0), 4, implementation="torch"
        )
    with pytest.raises(ValueError, match="D >= 1"):
        farthest_point_sampling(
            torch.zeros(16, 0, device=device), 4, implementation="torch"
        )
    with pytest.raises(ValueError, match="must be an int"):
        farthest_point_sampling(points, 4.0, implementation="torch")


def test_fps_opcheck(device: str):
    if "cpu" in device:
        pytest.skip("warp FPS backend is CUDA-only (tile reductions)")
    # Import the raw registered op lazily: opcheck needs the underlying custom
    # op, and a function-scoped import keeps a Warp load failure from breaking
    # collection of the torch-only tests in this file.
    from physicsnemo.nn.functional.geometry.farthest_point_sampling._warp_impl import (
        farthest_point_sampling as fps_warp_op,
    )

    points = _well_separated_cloud(device, n=40)
    torch.library.opcheck(fps_warp_op, args=(points, 8), kwargs={"random_start": False})


def test_fps_torch_compile_no_graph_break(device: str):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")
    if "cpu" in device:
        pytest.skip("CUDA only")
    points = _well_separated_cloud(device, n=128)

    def sample_fn(p: torch.Tensor):
        return farthest_point_sampling(p, 16, implementation="warp")

    eager = sample_fn(points)
    compiled = torch.compile(sample_fn, fullgraph=True)(points)
    torch.testing.assert_close(eager, compiled)
