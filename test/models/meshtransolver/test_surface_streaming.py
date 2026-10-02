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
"""Exhaustive bounded discovery must preserve contacts, ordering, and BPTT."""

import pytest
import torch

from physicsnemo.experimental.models.meshtransolver import SurfaceContactGraphBuilder
from physicsnemo.nn.functional.neighbors.surface_contact import (
    node_triangle_candidate_chunks,
    node_triangle_candidates,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def collect(chunks, faces):
    pages = list(chunks)
    pairs = torch.cat(pages, 1) if pages else faces.new_empty((2, 0))
    return pairs[:, torch.argsort(pairs[0] * len(faces) + pairs[1])], pages


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("page_size", [1, 7, 64])
def test_pages_split_single_dense_query_without_omissions(
    device, implementation, page_size
):
    tri = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], device=device)
    # Empty queries at BOTH ends exercise repeated prefix offsets. The interior
    # query overlaps 13 distinct facet IDs, more than the smallest page sizes.
    x = torch.cat(
        (
            tri.new_full((2, 3), 100),
            tri.repeat(13, 1),
            tri.new_tensor([[0.2, 0.2, 0.05], [100, 100, 100]]),
        )
    )
    faces = torch.arange(2, 41, device=device).reshape(-1, 3)
    args = (x, faces, x.new_full((len(x),), 0.1), x.new_zeros(len(faces)))
    expected = node_triangle_candidates(*args, implementation="torch")
    actual, pages = collect(
        node_triangle_candidate_chunks(
            *args, implementation=implementation, pair_chunk_size=page_size
        ),
        faces,
    )
    assert all(0 < p.shape[1] <= page_size for p in pages)
    assert int((actual[0] == 41).sum()) == 13
    assert actual.shape[1] == torch.unique(actual[0] * len(faces) + actual[1]).numel()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("quad", [False, True])
def test_swept_batched_excluded_pages_match_collector(device, implementation, quad):
    torch.manual_seed(45)
    x = torch.randn(40, 3, device=device, requires_grad=True)
    faces = torch.arange(32, device=device).reshape(-1, 4)
    if not quad:
        faces = faces[:, :3]
    batch = torch.arange(40, device=device) // 20
    pad = x.new_full((40,), 0.2, requires_grad=True)
    kwargs = dict(
        batch=batch,
        previous_positions=x + torch.randn_like(x) * 0.3,
        excluded_pairs=faces.new_tensor([[0, 5, 25, 25], [2, 1, 7, 7]]),
    )
    args = (x, faces, pad, pad[: len(faces)])
    expected = node_triangle_candidates(*args, implementation="torch", **kwargs)
    actual, pages = collect(
        node_triangle_candidate_chunks(
            *args, implementation=implementation, pair_chunk_size=11, **kwargs
        ),
        faces,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert all(p.shape[1] <= 11 and not p.requires_grad for p in pages)
    # Explicit total budget remains fail-closed, including before exclusions.
    with pytest.raises(RuntimeError, match="no pairs truncated"):
        list(
            node_triangle_candidate_chunks(
                *args,
                implementation=implementation,
                pair_chunk_size=11,
                max_pairs=1,
                **kwargs,
            )
        )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_false_positives_do_not_consume_active_graph_budget(device, implementation):
    tri = torch.tensor([[0.0, 0, 0], [10, 0, 0], [0, 10, 0]], device=device)
    x = torch.cat(
        (
            tri,
            tri.new_tensor([[9, 9, 0.01]]).repeat(51, 1),
            tri.new_tensor([[2, 2, 0.01]]),
        )
    )
    faces = torch.tensor([[0, 1, 2]], device=device)
    builder = SurfaceContactGraphBuilder(
        implementation=implementation,
        include_velocity=False,
        activation_distance=0.1,
        max_pairs=1,
        candidate_chunk_size=7,
    )
    graph = builder(x, faces=faces, shell_thickness=x.new_zeros(len(x)))
    assert graph.edge_index[1].tolist() == [54]
    assert builder.last_discovery_stats["candidates"] == 52
    assert builder.last_discovery_stats["active_pairs"] == 1
    assert builder.last_discovery_stats["largest_page"] <= 7
    # A second genuine contact must still trip the ACTIVE safety limit.
    x[3] = x[-1]
    with pytest.raises(RuntimeError, match="active contact budget.*no pairs truncated"):
        builder(x, faces=faces, shell_thickness=x.new_zeros(len(x)))


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("predictive", [False, True])
def test_chunk_sizes_preserve_live_values_and_multistep_gradients(
    device, implementation, predictive
):
    torch.manual_seed(7)
    reference = torch.tensor(
        [
            [0.0, 0, 0],
            [2, 0, 0.1],
            [2, 2, 0.2],
            [0, 2, 0],
            [0.4, 0.3, 0.2],
            [1.2, 0.7, 0.1],
            [1.3, 1.4, 0.3],
        ],
        device=device,
    )
    faces = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 2]], device=device)
    velocity = torch.randn_like(reference) * 0.1
    results = []
    for page_size in (1, 10000):
        x = reference.clone().requires_grad_()
        v = velocity.clone().requires_grad_()
        thickness = x.new_full((len(x),), 0.1, requires_grad=True)
        builder = SurfaceContactGraphBuilder(
            implementation=implementation,
            activation_distance=0.5,
            prediction_horizon=0.1 if predictive else 0,
            material_fan=True,
            candidate_chunk_size=page_size,
        )
        current = x
        graphs = []
        objective = x.new_zeros(())
        for _ in range(3):
            graph = builder(
                current,
                faces=faces,
                velocities=v,
                shell_thickness=thickness,
                reference_positions=reference,
            )
            graphs.append(graph)
            objective = (
                objective
                + graph.edge_features.square().sum()
                + graph.edge_weights.sum()
            )
            # Smooth causal feedback preserves the multi-step geometry graph.
            current = current + 0.01 * v + 0.0001 * graph.edge_weights.sum()
        gradients = torch.autograd.grad(objective, (x, v, thickness))
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
        results.append((graphs, gradients))
    for a, b in zip(results[0][0], results[1][0]):
        for field in (
            "edge_index",
            "source_nodes",
            "source_weights",
            "edge_features",
            "edge_weights",
        ):
            torch.testing.assert_close(
                getattr(a, field), getattr(b, field), atol=0, rtol=0
            )
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("page_size", [0, -1, True, 1.5])
def test_invalid_page_sizes_fail_eagerly(page_size):
    x = torch.zeros(0, 3)
    faces = torch.empty(0, 3, dtype=torch.long)
    with pytest.raises(ValueError, match="pair_chunk_size"):
        node_triangle_candidate_chunks(
            x, faces, x.new_zeros(0), x.new_zeros(0), pair_chunk_size=page_size
        )
    with pytest.raises(ValueError, match="candidate_chunk_size"):
        SurfaceContactGraphBuilder(candidate_chunk_size=page_size)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_empty_stream_and_empty_active_graph(device, implementation):
    x = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], device=device)
    for faces in (
        torch.tensor([[0, 1, 2]], device=device),
        torch.empty(0, 3, dtype=torch.long, device=device),
    ):
        assert (
            list(
                node_triangle_candidate_chunks(
                    x,
                    faces,
                    x.new_zeros(3),
                    x.new_zeros(len(faces)),
                    implementation=implementation,
                )
            )
            == []
        )
        graph = SurfaceContactGraphBuilder(
            implementation=implementation, include_velocity=False
        )(x, faces=faces, shell_thickness=x.new_zeros(3))
        assert graph.edge_index.shape == (2, 0)
        assert torch.isfinite(graph.edge_features).all()


def test_cuda_more_than_eight_million_candidates_are_exhaustive_and_bounded():
    if not torch.cuda.is_available():
        pytest.skip("CUDA stress test")
    tri = torch.tensor([[0.0, 0, 0], [10, 0, 0], [0, 10, 0]], device="cuda")
    x = torch.cat(
        (
            tri.repeat(100, 1),
            tri.new_tensor([[9, 9, 0.01]]).repeat(80_001, 1),
            tri.new_tensor([[2, 2, 0.01]]),
        )
    )
    faces = torch.arange(300, device=x.device).reshape(-1, 3)
    builder = SurfaceContactGraphBuilder(
        implementation="warp",
        include_velocity=False,
        activation_distance=0.1,
        max_pairs=30_000,
        candidate_chunk_size=65_536,
    )
    graph = builder(x, faces=faces, shell_thickness=x.new_zeros(len(x)))
    stats = builder.last_discovery_stats
    assert stats["candidates"] == 300 * 99 + 80_002 * 100 > 8_000_000
    assert stats["largest_page"] == 65_536
    assert graph.edge_index.shape[1] == stats["active_pairs"] == 29_800
    assert ((graph.edge_index[1] < 300) | (graph.edge_index[1] == len(x) - 1)).all()
    assert (graph.edge_index[1] == len(x) - 1).sum() == 100


def test_cuda_paged_discovery_on_nondefault_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.tensor(
            [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, 0.01], [0.3, 0.3, 0.01]],
            device="cuda",
        )
        faces = torch.tensor([[0, 1, 2]], device="cuda")
        args = (x, faces, x.new_full((5,), 0.1), x.new_zeros(1))
        actual, pages = collect(
            node_triangle_candidate_chunks(
                *args, implementation="warp", pair_chunk_size=1
            ),
            faces,
        )
        expected = node_triangle_candidates(*args, implementation="torch")
        assert len(pages) == 2
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    stream.synchronize()
