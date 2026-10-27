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

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.nn.functional import knn
from physicsnemo.nn.functional.neighbors import KNN
from physicsnemo.nn.functional.neighbors.knn._cuml_impl import knn_impl as knn_cuml
from physicsnemo.nn.functional.neighbors.knn._scipy_impl import knn_impl as knn_scipy
from physicsnemo.nn.functional.neighbors.knn._warp_impl import knn_impl as knn_warp


# Build a deterministic KNN problem with predictable neighborhood structure.
def _build_problem(device: str, dtype: torch.dtype):
    base = torch.linspace(0, 10, 11, device=device)
    x, y, z = torch.meshgrid(base, base, base, indexing="ij")
    queries = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

    offsets = torch.tensor(
        [
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.5],
        ],
        device=device,
    )
    points = queries[None, :, :] + offsets[:, None, :]
    points = points.reshape(-1, 3)
    return points.to(dtype), queries.to(dtype)


# Common KNN correctness checks shared by backend-specific tests.
def _assert_knn_outputs(
    points: torch.Tensor,
    queries: torch.Tensor,
    indices: torch.Tensor,
    distances: torch.Tensor,
    k: int,
) -> None:
    assert indices.shape == (queries.shape[0], k)
    assert distances.shape == (queries.shape[0], k)
    assert (indices >= 0).all()
    assert (indices < points.shape[0]).all()
    assert (distances >= 0).all()
    assert torch.all(distances[:, 1:] >= distances[:, :-1])


# Validate the torch implementation on representative dtypes and k values.
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_knn_torch(device: str, k: int, dtype: torch.dtype):
    points, queries = _build_problem(device, dtype)
    indices, distances = knn(points, queries, k=k, implementation="torch")
    _assert_knn_outputs(points, queries, indices, distances, k)


# Validate the cuML implementation when available on CUDA.
@pytest.mark.parametrize("k", [1, 5])
def test_knn_cuml(device: str, k: int):
    if "cuda" not in device:
        pytest.skip("cuml backend is CUDA-only")
    if not check_version_spec("cuml", "26.2.0", hard_fail=False):
        pytest.skip("cuml not available")

    points, queries = _build_problem(device, torch.float32)
    indices, distances = knn(points, queries, k=k, implementation="cuml")
    _assert_knn_outputs(points, queries, indices, distances, k)


# An empty query set gives empty outputs, like the torch backend.
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_knn_cuml_empty_queries(device: str, k: int, dtype: torch.dtype):
    if "cuda" not in device:
        pytest.skip("cuml backend is CUDA-only")
    if not check_version_spec("cuml", "26.2.0", hard_fail=False):
        pytest.skip("cuml not available")

    points, _ = _build_problem(device, dtype)
    queries = torch.empty(0, 3, device=device, dtype=dtype)
    indices, distances = knn(points, queries, k=k, implementation="cuml")
    ref_indices, ref_distances = knn(points, queries, k=k, implementation="torch")
    for actual, expected in ((indices, ref_indices), (distances, ref_distances)):
        assert actual.shape == expected.shape == (0, k)
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device


# Verify cuML kNN respects non-default CUDA stream synchronization.
def test_knn_cuml_non_default_cuda_stream(device: str):
    if "cuda" not in device:
        pytest.skip("cuml backend is CUDA-only")
    if not check_version_spec("cuml", "24.0.0", hard_fail=False):
        pytest.skip("cuml not available")

    k = 5
    cuda_device = torch.device(device)
    caller_stream = torch.cuda.current_stream(cuda_device)
    knn_stream = torch.cuda.Stream(device=cuda_device)

    # Baseline: run cuML on the default stream so we have a reference from
    # the same backend (any drift in the stream run is a real signal).
    points_baseline, queries_baseline = _build_problem(device, torch.float32)
    indices_baseline, distances_baseline = knn(
        points_baseline, queries_baseline, k=k, implementation="cuml"
    )
    torch.cuda.synchronize(cuda_device)

    # Stall knn_stream BEFORE creating the inputs so the input kernels are
    # still pending when cuML is enqueued. If cuML/cuPy do not honor DLPack
    # stream synchronization, cuML will read uninitialized memory and produce
    # visibly wrong output. ``torch.cuda._sleep`` only blocks the current
    # stream, so the CPU continues to enqueue normally.
    with torch.cuda.stream(knn_stream):
        torch.cuda._sleep(int(1e8))  # ~100 ms hazard window on knn_stream
        points, queries = _build_problem(device, torch.float32)
        indices, distances = knn(points, queries, k=k, implementation="cuml")
        stream_indices = indices.clone()
        stream_distances = distances.clone()

    caller_stream.wait_stream(knn_stream)

    _assert_knn_outputs(points, queries, stream_indices, stream_distances, k)
    KNN.compare_forward(
        (stream_indices, stream_distances),
        (indices_baseline, distances_baseline),
    )


# Validate the SciPy implementation when available on CPU.
@pytest.mark.parametrize("k", [1, 5])
def test_knn_scipy(device: str, k: int):
    if "cpu" not in device:
        pytest.skip("scipy backend is CPU-only")
    if not check_version_spec("scipy", "1.7.0", hard_fail=False):
        pytest.skip("scipy not available")

    points, queries = _build_problem(device, torch.float32)
    indices, distances = knn(points, queries, k=k, implementation="scipy")
    _assert_knn_outputs(points, queries, indices, distances, k)


# Validate the Warp implementation on the structured problem.
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_knn_warp(device: str, k: int, dtype: torch.dtype):
    points, queries = _build_problem(device, dtype)
    indices, distances = knn(points, queries, k=k, implementation="warp")
    assert indices.dtype == torch.int64
    assert distances.dtype == dtype
    _assert_knn_outputs(points, queries, indices, distances, k)


# The Warp search is exact: compare it with SciPy and torch on random clouds.
# k = 70 runs in the shared run-time-k kernel for up to 128 neighbors.
@pytest.mark.parametrize("k", [1, 11, 70])
@pytest.mark.parametrize("dim", [2, 3])
def test_knn_warp_matches_exact(device: str, k: int, dim: int):
    points = torch.randn(2000, dim, device=device)
    queries = torch.randn(300, dim, device=device)
    indices, distances = knn(points, queries, k=k, implementation="warp")
    _assert_knn_outputs(points, queries, indices, distances, k)

    # Distances recomputed from the returned indices match the reported ones.
    recomputed = (points[indices] - queries[:, None]).norm(dim=-1)
    torch.testing.assert_close(distances, recomputed, atol=1e-5, rtol=1e-5)

    reference = knn(points, queries, k=k, implementation="torch")
    KNN.compare_forward((indices, distances), reference)
    if check_version_spec("scipy", "1.7.0", hard_fail=False):
        expected = knn_scipy(points.cpu(), queries.cpu(), k)
        KNN.compare_forward((indices.cpu(), distances.cpu()), expected)


# Verify the Warp kNN runs on torch's current, non-default CUDA stream.
def test_knn_warp_non_default_cuda_stream(device: str):
    if "cuda" not in device:
        pytest.skip("CUDA only")
    k = 5
    cuda_device = torch.device(device)
    points_baseline, queries_baseline = _build_problem(device, torch.float32)
    baseline = knn(points_baseline, queries_baseline, k=k, implementation="warp")
    torch.cuda.synchronize(cuda_device)

    knn_stream = torch.cuda.Stream(device=cuda_device)
    with torch.cuda.stream(knn_stream):
        torch.cuda._sleep(int(1e8))  # ~100 ms hazard window on knn_stream
        points, queries = _build_problem(device, torch.float32)
        indices, distances = knn(points, queries, k=k, implementation="warp")
        stream_out = (indices.clone(), distances.clone())
    torch.cuda.current_stream(cuda_device).wait_stream(knn_stream)

    _assert_knn_outputs(points, queries, *stream_out, k)
    KNN.compare_forward(stream_out, baseline)


# Validate the Warp custom-op schema with torch opcheck.
def test_knn_warp_opcheck(device: str):
    points = torch.randn(100, 3, device=device)
    queries = torch.randn(10, 3, device=device)
    torch.library.opcheck(knn_warp, args=(points, queries, 5))


# Non-finite queries get index -1 and distance inf; the other rows are unaffected.
def test_knn_warp_non_finite_queries(device: str):
    points = torch.rand(1000, 3, device=device)
    queries = torch.rand(5, 3, device=device)
    queries[1] = float("nan")
    queries[2, 0] = float("inf")
    indices, distances = knn(points, queries, k=3, implementation="warp")
    bad = torch.tensor([1, 2], device=device)
    assert (indices[bad] == -1).all()
    assert torch.isinf(distances[bad]).all()
    good = torch.tensor([0, 3, 4], device=device)
    reference = knn(points, queries[good], k=3, implementation="torch")
    KNN.compare_forward((indices[good], distances[good]), reference)


# A neighbor whose squared distance is close to the float32 maximum is still found.
def test_knn_warp_large_finite_distance(device: str):
    points = torch.tensor([[1.8e19, 0.0, 0.0]], device=device)
    queries = torch.zeros(1, 3, device=device)
    indices, distances = knn(points, queries, k=1, implementation="warp")
    assert indices.tolist() == [[0]]
    torch.testing.assert_close(distances, points[:, :1])


# Validate the Warp implementation's input checks.
def test_knn_warp_error_handling(device: str):
    points = torch.randn(20, 4, device=device)
    with pytest.raises(ValueError, match="1 to 3 dimensions"):
        knn(points, points, k=3, implementation="warp")
    points = torch.randn(20, 3, device=device)
    with pytest.raises(ValueError, match="k <= number of points"):
        knn(points, points, k=21, implementation="warp")


# Auto-dispatch prefers Warp for CUDA points of up to three dimensions in
# float32 or lower precision, cuML for other CUDA points, and SciPy on CPU.
@pytest.mark.parametrize(
    "dim, dtype, expected_cuda",
    [
        (3, torch.float32, "warp"),
        (2, torch.bfloat16, "warp"),
        (3, torch.float64, "cuml"),
        (3, torch.int64, "cuml"),
        (4, torch.float32, "cuml"),
    ],
)
def test_knn_preferred_implementation(
    device: str, dim: int, dtype: torch.dtype, expected_cuda: str
):
    points = torch.zeros(8, dim, device=device, dtype=dtype)
    expected = expected_cuda if "cuda" in device else "scipy"
    assert KNN._preferred_name(points) == expected


# On CUDA the default path returns exactly what the Warp backend returns.
def test_knn_auto_dispatch_uses_warp_on_cuda(device: str):
    if "cuda" not in device:
        pytest.skip("CUDA only")
    points = torch.rand(5000, 3, device=device)
    queries = torch.rand(500, 3, device=device)
    auto_indices, auto_distances = knn(points, queries, k=7)
    warp_indices, warp_distances = knn(points, queries, k=7, implementation="warp")
    assert torch.equal(auto_indices, warp_indices)
    assert torch.equal(auto_distances, warp_distances)


# Validate benchmark input generation contract for KNN.
def test_knn_make_inputs_forward(device: str):
    label, args, kwargs = next(iter(KNN.make_inputs_forward(device=device)))
    assert isinstance(label, str)
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)

    indices, distances = KNN.dispatch(*args, implementation="torch", **kwargs)
    assert indices.ndim == 2
    assert distances.ndim == 2


# Compare torch and accelerated forward outputs where both implementations are available.
def test_knn_backend_forward_parity(device: str):
    points = torch.randn(53, 3, device=device)
    queries = torch.randn(21, 3, device=device)
    k = 5

    if "cuda" in device:
        if not check_version_spec("cuml", "26.2.0", hard_fail=False):
            pytest.skip("cuml not available")
        output_a = knn(points, queries, k, implementation="cuml")
    else:
        if not check_version_spec("scipy", "1.7.0", hard_fail=False):
            pytest.skip("scipy not available")
        output_a = knn(points, queries, k, implementation="scipy")

    output_b = knn(points, queries, k, implementation="torch")
    KNN.compare_forward(output_a, output_b)


# Validate compare-forward hook contract for KNN.
def test_knn_compare_forward_contract(device: str):
    _, args, kwargs = next(iter(KNN.make_inputs_forward(device=device)))
    output = KNN.dispatch(*args, implementation="torch", **kwargs)
    indices, distances = output
    reference = (indices.detach().clone(), distances.detach().clone())
    KNN.compare_forward(output, reference)


# Validate torch.compile support path for KNN.
def test_knn_torch_compile_no_graph_break(device: str):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available in this version of PyTorch")
    if "cpu" in device:
        pytest.skip("CUDA only")

    points = torch.randn(207, 3, device=device)
    queries = torch.randn(13, 3, device=device)
    k = 5

    implementation = (
        None if check_version_spec("cuml", "26.2.0", hard_fail=False) else "torch"
    )

    def search_fn(points: torch.Tensor, queries: torch.Tensor):
        return knn(points, queries, k=k, implementation=implementation)

    eager = search_fn(points, queries)
    compiled = torch.compile(search_fn, fullgraph=True)(points, queries)
    for eager_tensor, compiled_tensor in zip(eager, compiled):
        torch.testing.assert_close(eager_tensor, compiled_tensor, atol=1e-6, rtol=1e-6)


# Validate custom-op schemas with torch opcheck.
def test_knn_opcheck(device: str):
    points = torch.randn(100, 3, device=device)
    queries = torch.randn(10, 3, device=device)
    k = 5

    if "cuda" in device:
        if not check_version_spec("cuml", "26.2.0", hard_fail=False):
            pytest.skip("cuml not available")
        op = knn_cuml
    else:
        if not check_version_spec("scipy", "1.7.0", hard_fail=False):
            pytest.skip("scipy not available")
        op = knn_scipy

    torch.library.opcheck(op, args=(points, queries, k))


# Validate KNN error handling paths.
def test_knn_error_handling(device: str):
    points, queries = _build_problem(device, torch.float32)

    # Mismatched dtypes are rejected by all implementations.
    with pytest.raises(ValueError, match="must have the same dtype"):
        knn(
            points.to(torch.float32),
            queries.to(torch.float64),
            k=3,
            implementation="torch",
        )

    # Accelerated implementation/device mismatch checks.
    if "cpu" in device and check_version_spec("cuml", "26.2.0", hard_fail=False):
        with pytest.raises(ValueError, match="does not support CPU"):
            knn(points, queries, k=3, implementation="cuml")
    if "cuda" in device and check_version_spec("scipy", "1.7.0", hard_fail=False):
        with pytest.raises(ValueError, match="does not support CUDA"):
            knn(points, queries, k=3, implementation="scipy")
