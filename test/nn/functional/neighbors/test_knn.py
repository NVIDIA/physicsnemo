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
