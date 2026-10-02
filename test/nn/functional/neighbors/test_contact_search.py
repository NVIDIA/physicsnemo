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

from physicsnemo.nn.functional import contact_search, radius_search
from physicsnemo.nn.functional.neighbors import ContactSearch
from physicsnemo.nn.functional.neighbors.contact_search._warp_impl import (
    contact_search_impl as contact_search_warp,
)
from test.conftest import requires_module


def _csr_from_rows(
    rows: list[list[int]], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct sorted int32 CSR tensors for tests."""
    flattened = [value for row in rows for value in sorted(row)]
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return (
        torch.tensor(flattened, dtype=torch.int32, device=device),
        torch.tensor(offsets, dtype=torch.int32, device=device),
    )


def _folded_chain_problem(device: str, dtype: torch.dtype = torch.float32):
    """Create a chain folded back on itself with nonlocal spatial neighbors."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.1, 0.0],
            [2.0, 0.1, 0.0],
            [1.0, 0.1, 0.0],
            [0.0, 0.1, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    # The graph-local ball contains self and direct chain neighbors.
    rows = [
        list(range(max(0, index - 1), min(len(points), index + 2)))
        for index in range(len(points))
    ]
    neighbors, offsets = _csr_from_rows(rows, device)
    return points, neighbors, offsets


def _dynamic_pair_map(indices: torch.Tensor, distances: torch.Tensor) -> dict:
    return {
        tuple(pair): float(distance)
        for pair, distance in zip(indices.t().cpu().tolist(), distances.cpu().tolist())
    }


def test_contact_search_filters_graph_local_neighbors(device: str):
    points, neighbors, offsets = _folded_chain_problem(device)

    radius_indices, radius_distances = radius_search(
        points,
        points,
        radius=0.15,
        return_dists=True,
        implementation="torch",
    )
    contact_indices, contact_distances = contact_search(
        points,
        points,
        radius=0.15,
        exclude_neighbors=neighbors,
        exclude_offsets=offsets,
        return_dists=True,
        implementation="torch",
    )

    # Radius search sees eight self-pairs plus four bidirectional fold pairs.
    assert radius_indices.shape[1] == 16
    assert contact_indices.shape[1] == 6
    expected = {(0, 7), (7, 0), (1, 6), (6, 1), (2, 5), (5, 2)}
    assert set(map(tuple, contact_indices.t().cpu().tolist())) == expected
    torch.testing.assert_close(
        contact_distances,
        torch.full_like(contact_distances, 0.1),
    )


@pytest.mark.parametrize("return_dists", [False, True])
@pytest.mark.parametrize("return_points", [False, True])
@pytest.mark.parametrize("max_points", [None, 2])
def test_contact_search_torch_return_contract(
    device: str,
    return_dists: bool,
    return_points: bool,
    max_points: int | None,
):
    points, neighbors, offsets = _folded_chain_problem(device)
    result = contact_search(
        points,
        points,
        0.15,
        neighbors,
        offsets,
        max_points=max_points,
        return_dists=return_dists,
        return_points=return_points,
        implementation="torch",
    )

    tensors = result if isinstance(result, tuple) else (result,)
    indices = tensors[0]
    if max_points is None:
        assert indices.shape == (2, 6)
    else:
        assert indices.shape == (8, max_points)

    if return_points:
        points_out = tensors[1]
        expected_shape = (6, 3) if max_points is None else (8, max_points, 3)
        assert points_out.shape == expected_shape
    if return_dists:
        distances = tensors[-1]
        expected_shape = (6,) if max_points is None else (8, max_points)
        assert distances.shape == expected_shape
        assert (distances >= 0).all()
        assert (distances <= 0.15).all()


@requires_module("warp")
@pytest.mark.parametrize("max_points", [None, 4])
def test_contact_search_backend_forward_parity(device: str, max_points: int | None):
    points, neighbors, offsets = _folded_chain_problem(device)
    outputs = {}
    for implementation in ("torch", "warp"):
        outputs[implementation] = contact_search(
            points,
            points,
            0.15,
            neighbors,
            offsets,
            max_points=max_points,
            return_dists=True,
            implementation=implementation,
        )

    if max_points is None:
        torch_pairs = _dynamic_pair_map(*outputs["torch"])
        warp_pairs = _dynamic_pair_map(*outputs["warp"])
        assert torch_pairs.keys() == warp_pairs.keys()
        for pair in torch_pairs:
            assert warp_pairs[pair] == pytest.approx(torch_pairs[pair], abs=1e-6)
    else:
        torch_indices, torch_distances = outputs["torch"]
        warp_indices, warp_distances = outputs["warp"]
        for query_index in range(len(points)):
            torch_valid = torch_indices[query_index] >= 0
            warp_valid = warp_indices[query_index] >= 0
            assert set(torch_indices[query_index, torch_valid].cpu().tolist()) == set(
                warp_indices[query_index, warp_valid].cpu().tolist()
            )


@requires_module("warp")
def test_contact_search_warp_return_points(device: str):
    points, neighbors, offsets = _folded_chain_problem(device)
    indices, points_out, distances = contact_search(
        points,
        points,
        0.15,
        neighbors,
        offsets,
        return_dists=True,
        return_points=True,
        implementation="warp",
    )
    torch.testing.assert_close(points_out, points[indices[1]])
    torch.testing.assert_close(
        distances,
        torch.linalg.vector_norm(points[indices[0]] - points[indices[1]], dim=1),
    )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_contact_search_shared_batched_csr(device: str, implementation: str):
    if implementation == "warp":
        pytest.importorskip("warp")
    points, neighbors, offsets = _folded_chain_problem(device)
    batched_points = torch.stack(
        [points, points + torch.tensor([0.0, 0.0, 1.0], device=device)]
    )
    indices, distances = contact_search(
        batched_points,
        batched_points,
        0.15,
        neighbors,
        offsets,
        return_dists=True,
        implementation=implementation,
    )
    assert indices.shape == (3, 12)
    assert torch.bincount(indices[0].to(torch.int64)).cpu().tolist() == [6, 6]
    torch.testing.assert_close(distances, torch.full_like(distances, 0.1))


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_contact_search_per_batch_csr(device: str, implementation: str):
    if implementation == "warp":
        pytest.importorskip("warp")
    points, shared_neighbors, shared_offsets = _folded_chain_problem(device)
    shared_rows = [
        shared_neighbors[shared_offsets[i] : shared_offsets[i + 1]].cpu().tolist()
        for i in range(len(points))
    ]
    second_rows = [list(row) for row in shared_rows]
    second_rows[0].append(7)
    second_rows[7].append(0)
    neighbors, offsets = _csr_from_rows(shared_rows + second_rows, device)
    batched_points = torch.stack([points, points])

    indices = contact_search(
        batched_points,
        batched_points,
        0.15,
        neighbors,
        offsets,
        implementation=implementation,
    )
    assert torch.bincount(indices[0].to(torch.int64)).cpu().tolist() == [6, 4]
    second_pairs = set(map(tuple, indices[1:, indices[0] == 1].t().cpu().tolist()))
    assert (0, 7) not in second_pairs
    assert (7, 0) not in second_pairs


def test_contact_search_exclusions_applied_before_max_points(device: str):
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        device=device,
    )
    queries = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    neighbors, offsets = _csr_from_rows([[0, 1]], device)
    indices, distances = contact_search(
        points,
        queries,
        0.25,
        neighbors,
        offsets,
        max_points=1,
        return_dists=True,
        implementation="torch",
    )
    assert indices.item() == 2
    assert distances.item() == pytest.approx(0.2)


def test_contact_search_make_inputs_forward(device: str):
    label, args, kwargs = next(ContactSearch.make_inputs_forward(device))
    assert isinstance(label, str)
    output = ContactSearch.dispatch(*args, **kwargs, implementation="torch")
    assert isinstance(output, tuple)


@requires_module("warp")
@pytest.mark.parametrize("max_points", [None, 4])
def test_contact_search_backend_backward_parity(device: str, max_points: int | None):
    source, neighbors, offsets = _folded_chain_problem(device)
    gradients = {}
    for implementation in ("torch", "warp"):
        points = source.clone().detach().requires_grad_(True)
        _, points_out = contact_search(
            points,
            points,
            0.15,
            neighbors,
            offsets,
            max_points=max_points,
            return_points=True,
            implementation=implementation,
        )
        points_out.square().sum().backward()
        gradients[implementation] = points.grad
    ContactSearch.compare_backward(gradients["warp"], gradients["torch"])


@requires_module("warp")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_contact_search_warp_reduced_precision_backward(
    device: str, dtype: torch.dtype
):
    if device == "cpu":
        pytest.skip("Reduced-precision backward is a CUDA contract")
    source, neighbors, offsets = _folded_chain_problem(device, dtype)
    points = source.clone().detach().requires_grad_(True)
    _, points_out = contact_search(
        points,
        points,
        0.15,
        neighbors,
        offsets,
        max_points=4,
        return_points=True,
        implementation="warp",
    )
    points_out.float().square().sum().backward()
    assert points.grad is not None
    assert points.grad.dtype == dtype
    assert torch.isfinite(points.grad).all()


@requires_module("warp")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_contact_search_warp_reduced_precision(device: str, dtype: torch.dtype):
    points, neighbors, offsets = _folded_chain_problem(device, dtype)
    _, points_out, distances = contact_search(
        points,
        points,
        0.15,
        neighbors,
        offsets,
        max_points=4,
        return_points=True,
        return_dists=True,
        implementation="warp",
    )
    assert points_out.dtype == dtype
    assert distances.dtype == dtype


@requires_module("warp")
def test_contact_search_torch_compile_no_graph_break(device: str):
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    if "cuda" in device:
        pytest.skip("Skipping contact search torch.compile on CUDA")
    points, neighbors, offsets = _folded_chain_problem(device)

    def search_fn(points, neighbors, offsets):
        return contact_search(
            points,
            points,
            0.15,
            neighbors,
            offsets,
            max_points=4,
            return_dists=True,
            return_points=True,
            implementation="warp",
        )

    eager = search_fn(points, neighbors, offsets)
    compiled = torch.compile(search_fn, fullgraph=True)(points, neighbors, offsets)
    for eager_tensor, compiled_tensor in zip(eager, compiled):
        torch.testing.assert_close(eager_tensor, compiled_tensor)


@requires_module("warp")
def test_contact_search_opcheck(device: str):
    if device == "cpu":
        pytest.skip("CUDA only")
    points, neighbors, offsets = _folded_chain_problem(device)
    torch.library.opcheck(
        contact_search_warp,
        args=(points, points, 0.15, neighbors, offsets, 4, True, True),
    )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda neighbors, offsets: (neighbors.to(torch.int64), offsets), "int32"),
        (lambda neighbors, offsets: (neighbors, offsets[:-1]), "CSR rows"),
    ],
)
def test_contact_search_error_handling(device: str, mutator, message: str):
    points, neighbors, offsets = _folded_chain_problem(device)
    neighbors, offsets = mutator(neighbors, offsets)
    with pytest.raises(ValueError, match=message):
        contact_search(
            points,
            points,
            0.15,
            neighbors,
            offsets,
            implementation="torch",
        )


@pytest.mark.parametrize("radius", [0.0, -1.0, float("inf"), float("nan")])
def test_contact_search_rejects_invalid_radius(device: str, radius: float):
    points, neighbors, offsets = _folded_chain_problem(device)
    with pytest.raises(ValueError, match="positive and finite"):
        contact_search(
            points,
            points,
            radius,
            neighbors,
            offsets,
            implementation="torch",
        )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("layout", ["unbatched", "shared", "per_batch"])
@pytest.mark.parametrize("max_points", [1, 5])
def test_contact_search_padding_preserves_coincident_node_zero(
    device: str, implementation: str, layout: str, max_points: int
):
    """Padding cannot alias a real contact at the origin, including node zero."""
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], device=device
    )
    queries = torch.tensor(
        [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.5, 0.0, 0.0]], device=device
    )
    # Retain node 0 for query 0; query 1 has no neighbors and query 2 excludes
    # its only neighbor. Per-batch topology also excludes node 0 in batch 1.
    rows = [[1], [], [2]]
    expected_counts = torch.tensor([1, 0, 0], device=device)
    if layout != "unbatched":
        points = torch.stack([points, points])
        queries = torch.stack([queries, queries])
        expected_counts = torch.stack([expected_counts, expected_counts])
        if layout == "per_batch":
            rows += [[0, 1], [], [2]]
            expected_counts[1, 0] = 0
    neighbors, offsets = _csr_from_rows(rows, device)
    indices, returned_points, distances = contact_search(
        points,
        queries,
        0.1,
        neighbors,
        offsets,
        max_points=max_points,
        return_points=True,
        return_dists=True,
        implementation=implementation,
    )

    valid = indices >= 0
    torch.testing.assert_close(valid.sum(dim=-1), expected_counts)
    assert torch.all(indices[valid] == 0)
    assert torch.all(indices[~valid] == -1)
    assert torch.all(returned_points == 0)
    assert torch.all(distances == 0)

    # The index-only API must convey the same validity information.
    indices_only = contact_search(
        points,
        queries,
        0.1,
        neighbors,
        offsets,
        max_points=max_points,
        implementation=implementation,
    )
    torch.testing.assert_close(indices_only, indices)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("exclude_contact", [False, True])
def test_contact_search_padding_does_not_scatter_gradients(
    device: str, implementation: str, batched: bool, exclude_contact: bool
):
    """Neither sentinel -1 nor zero coordinate padding may update real nodes."""
    points = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], device=device)
    queries = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], device=device)
    if batched:
        points = torch.stack([points, points])
        queries = torch.stack([queries, queries])
    points.requires_grad_(True)
    neighbors, offsets = _csr_from_rows([[0] if exclude_contact else [], []], device)
    _, returned_points = contact_search(
        points,
        queries,
        0.1,
        neighbors,
        offsets,
        max_points=4,
        return_points=True,
        implementation=implementation,
    )
    returned_points.sum().backward()
    expected = torch.zeros_like(points)
    if not exclude_contact:
        expected[..., 0, :] = 1.0
    torch.testing.assert_close(points.grad, expected)


@pytest.mark.parametrize("max_points", [None, 3])
def test_contact_search_torch_preserves_float64_boundary_and_gradients(
    device: str, max_points: int | None
):
    """Float32 rounding would incorrectly include point 0 and move point 1."""
    points = torch.tensor(
        [[1.00000004, 0.0, 0.0], [1.00000001, 0.0, 0.0]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    queries = torch.zeros((1, 3), dtype=torch.float64, device=device)
    neighbors, offsets = _csr_from_rows([[]], device)
    indices, returned_points, distances = contact_search(
        points,
        queries,
        1.00000002,
        neighbors,
        offsets,
        max_points=max_points,
        return_points=True,
        return_dists=True,
        implementation="torch",
    )
    if max_points is None:
        assert indices.tolist() == [[0], [1]]
        selected_points = returned_points
        selected_distances = distances
    else:
        assert indices.tolist() == [[1, -1, -1]]
        selected_points = returned_points[indices >= 0]
        selected_distances = distances[indices >= 0]
    torch.testing.assert_close(selected_points, points[1:], atol=0.0, rtol=0.0)
    torch.testing.assert_close(selected_distances, points[1:, 0], atol=0.0, rtol=0.0)
    returned_points.sum().backward()
    torch.testing.assert_close(points.grad, points.new_tensor([[0, 0, 0], [1, 1, 1]]))


@requires_module("warp")
@pytest.mark.parametrize("max_points", [None, 3])
@pytest.mark.parametrize("entry_point", ["default", "warp", "custom_op"])
def test_contact_search_warp_rejects_float64(
    device: str, max_points: int | None, entry_point: str
):
    points = torch.tensor([[1.00000004, 0.0, 0.0]], dtype=torch.float64, device=device)
    queries = torch.zeros((1, 3), dtype=torch.float64, device=device)
    neighbors, offsets = _csr_from_rows([[]], device)
    args = (points, queries, 1.00000002, neighbors, offsets)
    kwargs = dict(max_points=max_points, return_points=True, return_dists=True)
    with pytest.raises(ValueError, match="implementation='torch'.*float64"):
        if entry_point == "custom_op":
            contact_search_warp(*args, **kwargs)
        else:
            contact_search(
                *args,
                **kwargs,
                implementation=None if entry_point == "default" else "warp",
            )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("layout", ["unbatched", "shared", "per_batch"])
@pytest.mark.parametrize("max_points", [1, 7, 32, 96])
@pytest.mark.parametrize(
    ("return_points", "return_dists"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_contact_search_nearest_k_matches_brute_force(
    device: str,
    implementation: str,
    layout: str,
    max_points: int,
    return_points: bool,
    return_dists: bool,
):
    """Select globally nearest eligible points, not a traversal-order prefix."""
    generator = torch.Generator().manual_seed(512)
    source = torch.rand((73, 3), generator=generator) * 2 - 1
    queries = torch.rand((6, 3), generator=generator) - 0.5
    queries[0] = source[10]  # A coincident point must survive selection.
    queries[3] = 8.0  # No contact candidates.
    radius = 1.25
    rows = [[0, 1, 2], [3, 4], [], [], list(range(73)), [5, 6, 7]]
    if layout != "unbatched":
        shift = torch.tensor([0.23, -0.11, 0.37])
        source = torch.stack([source, source * 0.9 + shift])
        queries = torch.stack([queries, queries * 0.9 + shift])
        if layout == "per_batch":
            rows += [[10, 11, 12], [], [0, 2, 4], [], list(range(73)), [1, 8]]

    # Independent dense oracle in double precision; random distances avoid ties.
    reference_points = source.unsqueeze(0) if layout == "unbatched" else source
    reference_queries = queries.unsqueeze(0) if layout == "unbatched" else queries
    distances = torch.linalg.vector_norm(
        reference_queries.double().unsqueeze(2)
        - reference_points.double().unsqueeze(1),
        dim=-1,
    )
    distances[distances > radius] = torch.inf
    for batch in range(reference_points.shape[0]):
        for query in range(reference_queries.shape[1]):
            row = batch * 6 + query if layout == "per_batch" else query
            distances[batch, query, rows[row]] = torch.inf
    values, expected_indices = distances.sort(dim=-1)
    k = min(max_points, source.shape[-2])
    values, expected_indices = values[..., :k], expected_indices[..., :k]
    values = torch.nn.functional.pad(values, (0, max_points - k), value=torch.inf)
    expected_indices = torch.nn.functional.pad(
        expected_indices, (0, max_points - k), value=-1
    )
    valid = torch.isfinite(values)
    expected_indices = torch.where(valid, expected_indices, -1)
    expected_distances = torch.where(valid, values, 0).float()
    expected_points = torch.zeros((*expected_indices.shape, 3))
    batches, query_ids, slots = valid.nonzero(as_tuple=True)
    selected_ids = expected_indices[batches, query_ids, slots]
    expected_points[batches, query_ids, slots] = reference_points[batches, selected_ids]
    expected_grad = torch.zeros_like(reference_points)
    expected_grad.index_put_(
        (batches, selected_ids),
        (slots.float() + 1).unsqueeze(-1).expand(-1, 3),
        accumulate=True,
    )
    if layout == "unbatched":
        expected_indices = expected_indices[0]
        expected_distances = expected_distances[0]
        expected_points = expected_points[0]
        expected_grad = expected_grad[0]

    points = source.to(device).requires_grad_(return_points)
    neighbors, offsets = _csr_from_rows(rows, device)
    result = contact_search(
        points,
        queries.to(device),
        radius,
        neighbors,
        offsets,
        max_points=max_points,
        return_points=return_points,
        return_dists=return_dists,
        implementation=implementation,
    )
    tensors = result if isinstance(result, tuple) else (result,)
    torch.testing.assert_close(tensors[0].cpu().long(), expected_indices)
    if return_dists:
        torch.testing.assert_close(tensors[-1].cpu(), expected_distances)
    if return_points:
        torch.testing.assert_close(tensors[1].cpu(), expected_points)
        slot_weights = torch.arange(1, max_points + 1, device=device).unsqueeze(-1)
        (tensors[1] * slot_weights).sum().backward()
        torch.testing.assert_close(points.grad.cpu(), expected_grad)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("max_points", [1, 3, 8])
def test_contact_search_nearest_k_ties_and_radius_boundary(
    device: str, implementation: str, batched: bool, max_points: int
):
    """Equal-distance subsets may differ, but cannot displace nearer points."""
    points = torch.tensor(
        [[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0.25, 0, 0], [1.01, 0, 0]],
        device=device,
    )
    queries = torch.zeros((1, 3), device=device)
    if batched:
        points = torch.stack([points, points])
        queries = torch.stack([queries, queries])
    neighbors, offsets = _csr_from_rows([[0]], device)
    indices, returned_points, distances = contact_search(
        points,
        queries,
        1.0,
        neighbors,
        offsets,
        max_points=max_points,
        return_points=True,
        return_dists=True,
        implementation=implementation,
    )
    for row, coords, dists in zip(
        indices.reshape(-1, max_points),
        returned_points.reshape(-1, max_points, 3),
        distances.reshape(-1, max_points),
    ):
        valid = row >= 0
        ids = row[valid].tolist()
        assert ids[0] == 3
        assert len(ids) == min(max_points, 3)
        assert set(ids).issubset({1, 2, 3})
        assert len(set(ids)) == len(ids)
        assert torch.all(dists[valid][1:] >= dists[valid][:-1])
        torch.testing.assert_close(
            torch.linalg.vector_norm(coords[valid], dim=-1), dists[valid]
        )
        assert torch.all(row[~valid] == -1)
        assert torch.all(coords[~valid] == 0)
        assert torch.all(dists[~valid] == 0)


@requires_module("warp")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batched", [False, True])
def test_contact_search_warp_nearest_k_reduced_precision(
    device: str, dtype: torch.dtype, batched: bool
):
    """Rank quantized input coordinates in fp32, including index-only scratch."""
    generator = torch.Generator().manual_seed(901)
    source = torch.rand((37, 3), generator=generator).to(device=device, dtype=dtype)
    queries = torch.rand((4, 3), generator=generator).to(device=device, dtype=dtype)
    if batched:
        source = torch.stack([source, source + 1])
        queries = torch.stack([queries, queries + 1])
    neighbors, offsets = _csr_from_rows([[0, 2], [1, 3], [], [6]], device)
    expected = contact_search(
        source.float(),
        queries.float(),
        2.0,
        neighbors,
        offsets,
        max_points=5,
        return_points=True,
        return_dists=True,
        implementation="torch",
    )
    actual = contact_search(
        source,
        queries,
        2.0,
        neighbors,
        offsets,
        max_points=5,
        return_points=True,
        return_dists=True,
        implementation="warp",
    )
    torch.testing.assert_close(actual[0].long(), expected[0])
    torch.testing.assert_close(actual[1], expected[1].to(dtype))
    torch.testing.assert_close(actual[2], expected[2].to(dtype))
    indices_only = contact_search(
        source,
        queries,
        2.0,
        neighbors,
        offsets,
        max_points=5,
        implementation="warp",
    )
    torch.testing.assert_close(indices_only, actual[0])


@requires_module("warp")
@pytest.mark.parametrize("max_points", [1, 3])
def test_contact_search_nearest_k_bptt(device: str, max_points: int):
    """Five-step first-order BPTT matches the Torch reference and finite differences."""
    source = torch.tensor(
        [
            [0, 0.01, 0],
            [0.13, -0.02, 0.01],
            [0.31, 0.06, -0.04],
            [0.55, -0.01, 0.02],
            [0.78, 0.02, 0.04],
        ],
        device=device,
    )
    neighbors, offsets = _csr_from_rows([[i] for i in range(len(source))], device)

    def rollout(implementation, weight_value=0.4):
        initial = source.clone().requires_grad_()
        weight = torch.tensor(weight_value, device=device, requires_grad=True)
        state = initial
        for _ in range(5):
            indices, selected = contact_search(
                state,
                state,
                0.65,
                neighbors,
                offsets,
                max_points=max_points,
                return_points=True,
                implementation=implementation,
            )
            relative = selected - state.unsqueeze(1)
            message = torch.where(
                (indices >= 0).unsqueeze(-1), relative.tanh(), 0.0
            ).sum(1)
            state = state + 0.05 * (0.3 * state.sin() + weight * message)
        loss = state.square().sum()
        gradients = torch.autograd.grad(loss, (initial, weight))
        return loss.detach(), gradients[0], gradients[1]

    expected = rollout("torch")
    actual = rollout("warp")
    for reference, result in zip(expected, actual):
        torch.testing.assert_close(result, reference, atol=1e-6, rtol=1e-5)
        assert torch.isfinite(result).all()
    assert actual[1].abs().sum() > 0
    assert actual[2].abs() > 0
    epsilon = 0.002
    finite_difference = (
        rollout("warp", 0.4 + epsilon)[0] - rollout("warp", 0.4 - epsilon)[0]
    ) / (2 * epsilon)
    torch.testing.assert_close(actual[2], finite_difference, atol=2e-4, rtol=0.01)
