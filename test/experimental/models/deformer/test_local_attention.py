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
"""kNN attention processors, the fused kernel and their use in DeFormer."""

import contextlib

import pytest
import torch

from physicsnemo.experimental.models.deformer import (
    DeFormer,
    KNNAttentionProcessor,
    LocalNeighborhood,
    LocalNeighborhoodData,
)
from physicsnemo.experimental.nn.geodesic_knn import geodesic_knn

try:
    import torch_geometric as pyg
except ImportError:  # the kernel and processor tests need no PyG
    pyg = None
requires_pyg = pytest.mark.skipif(pyg is None, reason="requires torch_geometric")


def _neighborhood(num_nodes, k, seed=0, break_at=None):
    """Chain mesh; ``break_at`` removes one link so a small component gets padded."""
    torch.manual_seed(seed)
    pos = torch.cumsum(torch.rand(num_nodes, 3) + 0.1, dim=0)
    src = torch.arange(num_nodes - 1)
    if break_at is not None:
        src = src[src != break_at - 1]
    edge_index = torch.stack((torch.cat((src, src + 1)), torch.cat((src + 1, src))))
    knn = geodesic_knn(pos, edge_index, k)
    own = torch.arange(num_nodes)[:, None]
    index = torch.cat((own, knn.index), dim=1)
    mask = torch.cat((torch.ones_like(own, dtype=torch.bool), knn.mask), dim=1)
    displacement = pos[index] - pos[:, None]
    edge_attr = torch.cat((displacement, displacement.norm(dim=-1, keepdim=True)), -1)
    return pos, edge_index, index, mask, edge_attr


def _graph(num_nodes=9, k=4, metric="geodesic", node_dim=6):
    pos, edge_index, index, mask, edge_attr = _neighborhood(num_nodes, k)
    graph = LocalNeighborhoodData(
        x=torch.randn(num_nodes, node_dim),
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.shape[1], 4),
        num_nodes=num_nodes,
    )
    graph.knn_index, graph.knn_mask, graph.knn_edge_attr = index, mask, edge_attr
    graph.knn_metric = metric
    return graph


def _model(**overrides):
    kwargs = dict(
        functional_dim=6,
        out_dim=5,
        input_dim_edges=4,
        geometry_dim=3,
        global_dim=2,
        n_hidden=32,
        n_head=4,
        slice_num=8,
        n_layers=2,
        use_te=False,
        attention_type="GALE_FA",
        num_pre_processor_layers=1,
        num_post_processor_layers=2,
        mesh_context_fusion="pre_post",
        mesh_pre_residual_gate_init=0.3,
        mesh_post_residual_gate_init=0.2,
        local_knn_k=4,
        local_neighborhood="geodesic",
        local_edge_dim=8,
    )
    kwargs.update(overrides)
    return DeFormer(**kwargs)


def _forward(model, graph, global_embedding):
    return model(
        node_features=graph.x,
        graph=graph,
        geometry=graph.x[:, :3],
        global_embedding=global_embedding,
    )


def test_processor_is_neighbor_order_invariant_and_ignores_padding():
    torch.manual_seed(1)
    num_nodes, k = 11, 5
    _, _, index, mask, edge_attr = _neighborhood(num_nodes, k, break_at=8)
    assert not mask.all()  # the three-node component has only two neighbors
    processor = KNNAttentionProcessor(2, 16, 4, edge_input_dim=4, edge_dim=8)
    x = torch.randn(num_nodes, 16)
    reference = processor(x, LocalNeighborhood(index, mask, edge_attr))
    assert reference.shape == (num_nodes, 16) and torch.isfinite(reference).all()
    permutation = torch.stack([torch.randperm(k + 1) for _ in range(num_nodes)])
    shuffled = LocalNeighborhood(
        index.gather(1, permutation),
        mask.gather(1, permutation),
        edge_attr.gather(1, permutation[..., None].expand(-1, -1, 4)),
    )
    torch.testing.assert_close(processor(x, shuffled), reference, atol=1e-5, rtol=1e-5)
    corrupted = LocalNeighborhood(
        torch.where(mask, index, torch.randint(0, num_nodes, index.shape)),
        mask,
        torch.where(mask[..., None], edge_attr, torch.randn_like(edge_attr) * 50),
    )
    torch.testing.assert_close(processor(x, corrupted), reference, atol=0, rtol=0)
    reference.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in processor.parameters()
    )


def test_processor_checkpoint_segments_match_plain_execution():
    torch.manual_seed(2)
    _, _, index, mask, edge_attr = _neighborhood(8, 3)
    plain = KNNAttentionProcessor(3, 16, 4, edge_dim=8).train()
    checkpointed = KNNAttentionProcessor(
        3, 16, 4, edge_dim=8, checkpoint_segments=2
    ).train()
    checkpointed.load_state_dict(plain.state_dict())
    assert checkpointed.segments == ((0, 2), (2, 3))
    neighborhood = LocalNeighborhood(index, mask, edge_attr)
    x = torch.randn(8, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = plain(x, neighborhood)
    actual = checkpointed(y, neighborhood)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(y.grad, x.grad, atol=0, rtol=0)
    for (name, a), (_, b) in zip(
        plain.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        # The edge encoder runs inside every segment, so its gradient is the
        # sum of per-segment contributions: equal up to summation order.
        exact = {} if name.startswith("edge_encoder.") else dict(atol=0, rtol=0)
        torch.testing.assert_close(b.grad, a.grad, msg=name, **exact)
    with pytest.raises(ValueError, match="checkpoint_segments"):
        KNNAttentionProcessor(2, 16, 4, checkpoint_segments=3)
    with pytest.raises(ValueError, match="at least one valid"):
        plain(x, LocalNeighborhood(index, torch.zeros_like(mask), edge_attr))


@requires_pyg
def test_processors_are_inert_at_zero_gates_and_train_otherwise():
    torch.manual_seed(3)
    graph = _graph()
    global_embedding = torch.randn(1, 1, 2)
    frozen = _model(
        mesh_pre_residual_gate_init=0.0, mesh_post_residual_gate_init=0.0
    ).eval()
    baseline = _forward(frozen, graph, global_embedding)
    perturbed = graph.clone()
    perturbed.knn_edge_attr = torch.randn_like(graph.knn_edge_attr)
    torch.testing.assert_close(_forward(frozen, perturbed, global_embedding), baseline)
    live = _model().train()
    output = _forward(live, graph, global_embedding)
    assert output.shape == (graph.num_nodes, 5)
    assert not torch.allclose(_forward(live, perturbed, global_embedding), output)
    output.square().mean().backward()
    for name in ("pre_processor", "post_processor"):
        grads = [p.grad for p in getattr(live, name).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert any(g.abs().sum() > 0 for g in grads), name


@requires_pyg
def test_model_checkpoint_segments_and_state_dict_round_trip():
    torch.manual_seed(4)
    graph = _graph(num_nodes=10)
    global_embedding = torch.randn(1, 1, 2)
    plain = _model().train()
    checkpointed = _model(
        num_pre_processor_checkpoint_segments=1,
        num_post_processor_checkpoint_segments=2,
        checkpoint_offloading=True,
    ).train()
    checkpointed.load_state_dict(plain.state_dict(), strict=True)
    expected = _forward(plain, graph, global_embedding)
    actual = _forward(checkpointed, graph, global_embedding)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for (name, a), (_, b) in zip(
        plain.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        exact = {} if ".edge_encoder." in name else dict(atol=0, rtol=0)
        torch.testing.assert_close(b.grad, a.grad, msg=name, **exact)
    reloaded = _model().eval()
    reloaded.load_state_dict(plain.state_dict(), strict=True)
    plain.eval()
    torch.testing.assert_close(
        _forward(reloaded, graph, global_embedding),
        _forward(plain, graph, global_embedding),
    )


@requires_pyg
def test_rejects_mismatched_neighborhoods_and_invalid_settings():
    torch.manual_seed(5)
    graph = _graph()
    global_embedding = torch.randn(1, 1, 2)
    model = _model().eval()
    wrong_metric = graph.clone()
    wrong_metric.knn_metric = "euclidean"
    with pytest.raises(ValueError, match="metric"):
        _forward(model, wrong_metric, global_embedding)
    wrong_k = graph.clone()
    wrong_k.knn_index = graph.knn_index[:, :3]
    wrong_k.knn_mask = graph.knn_mask[:, :3]
    wrong_k.knn_edge_attr = graph.knn_edge_attr[:, :3]
    with pytest.raises(ValueError, match="slots"):
        _forward(model, wrong_k, global_embedding)
    bare = pyg.data.Data(
        x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=9
    )
    with pytest.raises(ValueError, match="knn_index"):
        _forward(model, bare, global_embedding)
    with pytest.raises(ValueError, match="local_neighborhood"):
        _model(local_neighborhood="hops")
    with pytest.raises(ValueError, match="local_knn_k"):
        _model(local_knn_k=0)
    with pytest.raises(ValueError, match="divisible"):
        _model(local_num_heads=5)


# --------------------------------------------------------------------------- #
# Fused kNN attention (no gathered [N, K, H, dh] tensors).           #
# --------------------------------------------------------------------------- #

from physicsnemo.experimental.models.deformer.fused_knn_attention import (  # noqa: E402
    knn_attention,
    triton_available,
)
from physicsnemo.experimental.models.deformer.local_attention import (  # noqa: E402
    KNNAttentionBlock,
)

# bf16 autocast tolerances for the fused path against the bf16 reference. Both
# round the bf16 projections identically; they differ in where the softmax
# weights are rounded (the reference casts them to bf16 before the value and
# edge sums, the fused path keeps them in float32), so the gap is bf16 rounding
# noise, not a systematic error. Elementwise: block output and input gradients.
# Parameter gradients sum bf16 products over every node, so they are compared by
# relative L2 norm against the strict-float32 result: the fused error must be at
# most 1.5x the reference's own bf16 error (+1e-3) and below 5 percent.
BF16_TOLERANCE = dict(atol=5e-2, rtol=5e-2)
BF16_RELATIVE_SLACK, BF16_RELATIVE_FLOOR, BF16_RELATIVE_MAX = 1.5, 1e-3, 5e-2
# With the log-sum-exp detached the softmax is shift invariant per row, so the
# key bias receives no gradient (it only shifts the detached lse feature): its
# gradient is zero analytically and only collects rounding noise. The value and
# edge-bias projections have no bias any more.
ZERO_GRADIENT_PARAMETERS = frozenset({"key.bias"})

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
triton_only = pytest.mark.skipif(
    not triton_available(), reason="requires CUDA + Triton"
)


@contextlib.contextmanager
def _strict_fp32():
    """Disable TF32 so float32 comparisons are exact float32 on both sides.

    NGC PyTorch images set ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1``, which makes the
    reference's float32 einsums run in TF32 (about 3e-4 relative error) while the
    fused kernels compute in true float32.
    """
    if not torch.cuda.is_available():
        yield
        return
    saved = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved[0]
        torch.backends.cudnn.allow_tf32 = saved[1]
        torch.set_float32_matmul_precision(saved[2])


def _relative_error(value, target):
    return ((value - target).norm() / target.norm().clamp_min(1e-12)).item()


def _block_inputs(num_nodes, k, hidden, edge_dim, device, dtype=torch.float32, seed=0):
    """Padded neighborhoods (one broken chain link) plus random latents."""
    _, _, index, mask, _ = _neighborhood(
        num_nodes, k, seed=seed, break_at=num_nodes - 3
    )
    assert not mask.all(), "test needs padded slots"
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, hidden, generator=generator)
    edge_latent = torch.randn(num_nodes, k + 1, edge_dim, generator=generator)
    weight = torch.randn(num_nodes, hidden, generator=generator)
    to = dict(device=device)
    return (
        x.to(dtype=dtype, **to),
        index.to(**to),
        mask.to(**to),
        edge_latent.to(dtype=dtype, **to),
        weight.to(dtype=dtype, **to),
    )


def _block_pair(hidden, heads, edge_dim, device, backend, seed=0):
    torch.manual_seed(seed)
    reference = KNNAttentionBlock(hidden, heads, edge_dim).to(device)
    fused = KNNAttentionBlock(
        hidden, heads, edge_dim, fused=True, fused_backend=backend
    ).to(device)
    fused.load_state_dict(reference.state_dict())
    return reference, fused


def _run_block(block, x, index, mask, edge_latent, weight, autocast=False):
    """Forward and backward; returns output and every gradient by name."""
    block.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_()
    edge_latent = edge_latent.detach().clone().requires_grad_()
    device_type = x.device.type
    with torch.autocast(
        device_type=device_type, dtype=torch.bfloat16, enabled=autocast
    ):
        out = block(x, index, mask, edge_latent)
    (out.float() * weight.float()).sum().backward()
    grads = {"x": x.grad, "edge_latent": edge_latent.grad}
    grads.update({name: p.grad for name, p in block.named_parameters()})
    return out.float(), {k: v.float() for k, v in grads.items()}


def _assert_close_dict(actual, expected, **tolerance):
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], msg=name, **tolerance)


@pytest.mark.parametrize("hidden,heads,edge_dim", [(32, 4, 8), (24, 3, 5)])
def test_fused_torch_backend_matches_reference_fp32_on_cpu(hidden, heads, edge_dim):
    """Forward and all gradients agree with the gather reference, padding included."""
    x, index, mask, edge_latent, weight = _block_inputs(13, 5, hidden, edge_dim, "cpu")
    reference, fused = _block_pair(hidden, heads, edge_dim, "cpu", "torch")
    expected, expected_grads = _run_block(
        reference, x, index, mask, edge_latent, weight
    )
    actual, actual_grads = _run_block(fused, x, index, mask, edge_latent, weight)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    _assert_close_dict(actual_grads, expected_grads, atol=1e-5, rtol=1e-5)
    assert all(torch.isfinite(g).all() for g in actual_grads.values())


def test_fused_functional_passes_gradcheck_in_float64():
    """The manual backward is the exact derivative (non-power-of-two widths)."""
    num_nodes, slots, heads, head_dim, edge_dim = 7, 4, 2, 3, 5
    _, _, index, mask, _ = _neighborhood(num_nodes, slots - 1, seed=3, break_at=4)
    assert not mask.all()
    kwargs = dict(dtype=torch.float64, requires_grad=True)
    query = torch.randn(num_nodes, heads, head_dim, **kwargs)
    key = torch.randn(num_nodes, heads, head_dim, **kwargs)
    value = torch.randn(num_nodes, heads, head_dim, **kwargs)
    bias = torch.randn(num_nodes, slots, heads, **kwargs)
    edge = torch.randn(num_nodes, slots, edge_dim, **kwargs)

    def op(q, k, v, b, e):
        return knn_attention(q, k, v, b, e, index, mask, head_dim**-0.5, "torch")

    assert torch.autograd.gradcheck(
        op, (query, key, value, bias, edge), atol=1e-6, rtol=1e-5
    )


def test_fused_functional_handles_fully_masked_rows_like_reference():
    """A node without valid slots yields zeros and finite gradients (no NaN)."""
    num_nodes, slots, hidden, heads, edge_dim = 6, 3, 16, 2, 4
    index = torch.randint(0, num_nodes, (num_nodes, slots))
    mask = torch.ones(num_nodes, slots, dtype=torch.bool)
    mask[2] = False
    x = torch.randn(num_nodes, hidden)
    edge_latent = torch.randn(num_nodes, slots, edge_dim)
    weight = torch.randn(num_nodes, hidden)
    reference, fused = _block_pair(hidden, heads, edge_dim, "cpu", "torch")
    expected, expected_grads = _run_block(
        reference, x, index, mask, edge_latent, weight
    )
    actual, actual_grads = _run_block(fused, x, index, mask, edge_latent, weight)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    _assert_close_dict(actual_grads, expected_grads, atol=1e-5, rtol=1e-5)


def test_fused_processor_is_neighbor_order_invariant_and_ignores_padding():
    """Processor-level invariances hold with the fused kernel enabled."""
    torch.manual_seed(1)
    num_nodes, k = 11, 5
    _, _, index, mask, edge_attr = _neighborhood(num_nodes, k, break_at=8)
    processor = KNNAttentionProcessor(
        2,
        16,
        4,
        edge_input_dim=4,
        edge_dim=8,
        fused_attention=True,
        fused_backend="torch",
    )
    assert all(block.fused for block in processor.blocks)
    x = torch.randn(num_nodes, 16)
    reference = processor(x, LocalNeighborhood(index, mask, edge_attr))
    assert reference.shape == (num_nodes, 16) and torch.isfinite(reference).all()
    permutation = torch.stack([torch.randperm(k + 1) for _ in range(num_nodes)])
    shuffled = LocalNeighborhood(
        index.gather(1, permutation),
        mask.gather(1, permutation),
        edge_attr.gather(1, permutation[..., None].expand(-1, -1, 4)),
    )
    torch.testing.assert_close(processor(x, shuffled), reference, atol=1e-5, rtol=1e-5)
    corrupted = LocalNeighborhood(
        torch.where(mask, index, torch.randint(0, num_nodes, index.shape)),
        mask,
        torch.where(mask[..., None], edge_attr, torch.randn_like(edge_attr) * 50),
    )
    torch.testing.assert_close(processor(x, corrupted), reference, atol=0, rtol=0)
    # The gather reference and the fused processor agree as whole stacks.
    plain = KNNAttentionProcessor(2, 16, 4, edge_input_dim=4, edge_dim=8)
    plain.load_state_dict(processor.state_dict())
    torch.testing.assert_close(
        plain(x, LocalNeighborhood(index, mask, edge_attr)),
        reference,
        atol=1e-5,
        rtol=1e-5,
    )
    with pytest.raises(ValueError, match="fused_backend"):
        KNNAttentionBlock(16, 4, 8, fused_backend="cuda")


def test_fused_processor_checkpoint_segments_match_plain_execution():
    torch.manual_seed(2)
    _, _, index, mask, edge_attr = _neighborhood(8, 3)
    plain = KNNAttentionProcessor(
        3, 16, 4, edge_dim=8, fused_attention=True, fused_backend="torch"
    ).train()
    checkpointed = KNNAttentionProcessor(
        3,
        16,
        4,
        edge_dim=8,
        checkpoint_segments=2,
        fused_attention=True,
        fused_backend="torch",
    ).train()
    checkpointed.load_state_dict(plain.state_dict())
    neighborhood = LocalNeighborhood(index, mask, edge_attr)
    x = torch.randn(8, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected = plain(x, neighborhood)
    actual = checkpointed(y, neighborhood)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(y.grad, x.grad, atol=0, rtol=0)
    for (name, a), (_, b) in zip(
        plain.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        exact = {} if name.startswith("edge_encoder.") else dict(atol=0, rtol=0)
        torch.testing.assert_close(b.grad, a.grad, msg=name, **exact)


@triton_only
@pytest.mark.parametrize("hidden,heads,edge_dim", [(64, 4, 8), (96, 4, 12)])
def test_fused_triton_matches_reference_fp32_on_cuda(hidden, heads, edge_dim):
    """Triton forward and gradients match the gather reference in float32.

    Key/value gradients use float32 atomics, so only their summation order
    differs from the reference; the tolerance covers that reordering.
    """
    device = "cuda"
    x, index, mask, edge_latent, weight = _block_inputs(
        1000, 8, hidden, edge_dim, device
    )
    reference, fused = _block_pair(hidden, heads, edge_dim, device, "triton")
    with _strict_fp32():
        expected, expected_grads = _run_block(
            reference, x, index, mask, edge_latent, weight
        )
        actual, actual_grads = _run_block(fused, x, index, mask, edge_latent, weight)
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
        _assert_close_dict(actual_grads, expected_grads, atol=2e-4, rtol=2e-4)
        # Padded slots with corrupted IDs / features do not change the result.
        corrupted_index = torch.where(mask, index, torch.randint_like(index, 0, 1000))
        corrupted_edge = torch.where(
            mask[..., None], edge_latent, torch.randn_like(edge_latent) * 50
        )
        torch.testing.assert_close(
            _run_block(fused, x, corrupted_index, mask, corrupted_edge, weight)[0],
            actual,
            atol=0,
            rtol=0,
        )


@triton_only
def test_fused_triton_bf16_autocast_matches_reference_within_tolerance():
    """Under bf16 autocast the fused path is within BF16_TOLERANCE of the reference
    and at least as close to the float32 result as the reference itself."""
    device, hidden, heads, edge_dim = "cuda", 64, 4, 8
    x, index, mask, edge_latent, weight = _block_inputs(
        1000, 8, hidden, edge_dim, device
    )
    reference, fused = _block_pair(hidden, heads, edge_dim, device, "triton")
    with _strict_fp32():
        truth, truth_grads = _run_block(reference, x, index, mask, edge_latent, weight)
    expected, expected_grads = _run_block(
        reference, x, index, mask, edge_latent, weight, autocast=True
    )
    actual, actual_grads = _run_block(
        fused, x, index, mask, edge_latent, weight, autocast=True
    )
    torch.testing.assert_close(actual, expected, **BF16_TOLERANCE)
    for name in ("x", "edge_latent"):
        torch.testing.assert_close(
            actual_grads[name], expected_grads[name], msg=name, **BF16_TOLERANCE
        )
    assert _relative_error(actual, truth) <= (
        BF16_RELATIVE_SLACK * _relative_error(expected, truth) + BF16_RELATIVE_FLOOR
    )
    for name in expected_grads:
        if name in ZERO_GRADIENT_PARAMETERS:
            # Analytically zero gradients (see ZERO_GRADIENT_PARAMETERS): only
            # an absolute comparison of the bf16 noise is meaningful.
            fused_error = (actual_grads[name] - truth_grads[name]).norm().item()
            reference_error = (expected_grads[name] - truth_grads[name]).norm().item()
            assert (
                fused_error
                <= BF16_RELATIVE_SLACK * reference_error + BF16_RELATIVE_FLOOR
            )
            continue
        fused_error = _relative_error(actual_grads[name], truth_grads[name])
        reference_error = _relative_error(expected_grads[name], truth_grads[name])
        assert (
            fused_error <= BF16_RELATIVE_SLACK * reference_error + BF16_RELATIVE_FLOOR
        ), (
            name,
            fused_error,
            reference_error,
        )
        assert fused_error < BF16_RELATIVE_MAX, (name, fused_error)


def test_functional_rejects_neighbor_ids_outside_the_node_range():
    """Valid slots must point into the node range; masked slots may hold anything."""
    n, k, heads, dh, edge_dim = 12, 4, 2, 8, 3
    g = torch.Generator().manual_seed(0)
    q, key, value = (torch.randn(n, heads, dh, generator=g) for _ in range(3))
    bias = torch.randn(n, k, heads, generator=g)
    edge = torch.randn(n, k, edge_dim, generator=g)
    index = torch.randint(0, n, (n, k), generator=g)
    mask = torch.ones(n, k, dtype=torch.bool)
    kwargs = dict(scale=dh**-0.5, backend="torch")
    stale = index.clone()
    stale[0, 1] = n
    with pytest.raises(ValueError, match="outside"):
        knn_attention(q, key, value, bias, edge, stale, mask, **kwargs)
    stale[0, 1] = -1
    with pytest.raises(ValueError, match="outside"):
        knn_attention(q, key, value, bias, edge, stale, mask, **kwargs)
    masked = mask.clone()
    masked[0, 1] = False
    expected = knn_attention(q, key, value, bias, edge, index, masked, **kwargs)
    actual = knn_attention(q, key, value, bias, edge, stale, masked, **kwargs)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


@triton_only
def test_fused_triton_block_compiles_and_matches_eager():
    """torch.compile of the Triton block reproduces eager forward and backward."""
    device, hidden, heads, edge_dim = "cuda", 64, 4, 16
    x, index, mask, edge_latent, weight = _block_inputs(
        256, 8, hidden, edge_dim, device
    )
    block = KNNAttentionBlock(
        hidden, heads, edge_dim, fused=True, fused_backend="triton"
    ).to(device)
    with _strict_fp32():
        expected, expected_grads = _run_block(
            block, x, index, mask, edge_latent, weight
        )
        compiled = torch.compile(block)
        actual, compiled_grads = _run_block(
            compiled, x, index, mask, edge_latent, weight
        )
    actual_grads = {
        name.removeprefix("_orig_mod."): grad for name, grad in compiled_grads.items()
    }
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    # Inductor reorders the reductions of the node-update MLP's weight
    # gradients, so they match eager to float32 summation order, not bitwise;
    # the analytically zero gradient is noise on both sides.
    for name in expected_grads:
        if name in ZERO_GRADIENT_PARAMETERS:
            continue
        error = _relative_error(actual_grads[name], expected_grads[name])
        assert error < 1e-4, (name, error)
    _assert_close_dict(actual_grads, expected_grads, atol=1e-4, rtol=1e-4)


@triton_only
def test_fused_triton_runs_on_a_non_current_device():
    """The kernels launch on the tensors' device, not the current one."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    device, hidden, heads, edge_dim = torch.device("cuda", 1), 64, 4, 8
    x, index, mask, edge_latent, weight = _block_inputs(
        300, 8, hidden, edge_dim, device
    )
    reference, fused = _block_pair(hidden, heads, edge_dim, device, "triton")
    with torch.cuda.device(0), _strict_fp32():
        expected, expected_grads = _run_block(
            reference, x, index, mask, edge_latent, weight
        )
        actual, actual_grads = _run_block(fused, x, index, mask, edge_latent, weight)
    assert actual.device == device
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    _assert_close_dict(actual_grads, expected_grads, atol=2e-4, rtol=2e-4)


@cuda_only
def test_fused_path_uses_a_fraction_of_the_reference_memory():
    """The fused block never materializes the [N, K, H, dh] gathered tensors."""
    device, hidden, heads, edge_dim, num_nodes, k = "cuda", 256, 8, 32, 20000, 32
    torch.manual_seed(0)
    index = torch.randint(0, num_nodes, (num_nodes, k + 1), device=device)
    index[:, 0] = torch.arange(num_nodes, device=device)
    mask = torch.ones_like(index, dtype=torch.bool)
    x = torch.randn(num_nodes, hidden, device=device)
    edge_latent = torch.randn(num_nodes, k + 1, edge_dim, device=device)
    weight = torch.randn(num_nodes, hidden, device=device)
    reference, fused = _block_pair(hidden, heads, edge_dim, device, "auto")

    def peak(block):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        _run_block(block, x, index, mask, edge_latent, weight)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    reference_peak, fused_peak = peak(reference), peak(fused)
    gathered = 2 * num_nodes * (k + 1) * hidden * 4  # two [N, K, D] float32 tensors
    assert reference_peak > gathered
    assert fused_peak < 0.5 * reference_peak


@requires_pyg
def test_fused_flag_matches_the_gather_reference_model():
    """The fused blocks (the default) reproduce the gather-based reference model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(6)
    graph = _graph(num_nodes=12).to(device)
    global_embedding = torch.randn(1, 1, 2, device=device)
    reference = _model(local_attention_fused=False).to(device).eval()
    fused = (
        _model(local_attention_fused=True, local_attention_backend="auto")
        .to(device)
        .eval()
    )
    fused.load_state_dict(reference.state_dict(), strict=True)
    assert all(block.fused for block in fused.pre_processor.blocks)
    assert all(block.fused for block in fused.post_processor.blocks)
    assert not any(block.fused for block in reference.pre_processor.blocks)
    with _strict_fp32():
        expected = _forward(reference, graph, global_embedding)
        actual = _forward(fused, graph, global_embedding)
    tolerance = 1e-4 if device == "cuda" else 1e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    with pytest.raises(ValueError, match="local_attention_backend"):
        _model(local_attention_backend="cuda")
    with pytest.raises(ValueError, match="local_attention_fused"):
        _model(local_attention_fused="yes")
