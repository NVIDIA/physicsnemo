# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fused contact remains equivalent with independently optional XYZ/penalty."""

import copy
import sys
from pathlib import Path

import pytest
import torch

CRASH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH))
from penetration_loss import rollout_penetration_loss  # noqa: E402
from test_deformer_contact import make_model  # noqa: E402
from test_surface_contact_recipe import surface_sample  # noqa: E402


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("xyz", [False, True])
@pytest.mark.parametrize("penalty", [False, True])
def test_fused_contact_options_four_step_bptt(monkeypatch, device, xyz, penalty):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(721)
    model = (
        make_model(
            include_position_features=xyz,
            num_time_steps=5,
            contact_graph_backend="surface",
            contact_search_implementation="warp",
            contact_activation_distance=5.0,
            contact_normal_epsilon=0.1,
            contact_surface_predictive=True,
            contact_surface_material_fan=True,
            include_local_features=True,
            n_hidden_local=4,
            checkpoint_rollout=True,
            checkpoint_contact=True,
        )
        .to(device)
        .train()
    )
    reference = copy.deepcopy(model)
    reference.node_contact_builder.narrow_phase = "torch"
    assert model.node_contact_builder.narrow_phase == "auto"
    assert model.state_dict().keys() == reference.state_dict().keys()
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, reference.state_dict()[key])

    sample, stats = surface_sample(device)
    sample.node_target = torch.zeros(6, 4, 3, device=device)
    sample.graph.contact_faces = torch.tensor([[0, 1, 2, 3]], device=device)
    sample.graph.shell_thickness.fill_(3.0)
    initial = (
        sample.node_features["coords"].detach() * stats["node"]["pos_std"]
        + stats["node"]["pos_mean"]
    ).clone()
    initial[4:, 2] += 20
    sample.graph.penetration_reference_positions = initial
    other = copy.deepcopy(sample)
    fused_calls = []
    original = model.node_contact_builder._fused_candidate_gaps

    def fused(*args):
        fused_calls.append(True)
        return original(*args)

    monkeypatch.setattr(model.node_contact_builder, "_fused_candidate_gaps", fused)
    fields = (
        "edge_index",
        "edge_features",
        "edge_weights",
        "obstacle_mask",
        "source_nodes",
        "source_weights",
    )
    graphs = [[], []]
    handles = []
    for index, net in enumerate((model, reference)):

        def capture(module, args, graph, index=index):
            graphs[index].append(
                {k: getattr(graph, k).detach().clone() for k in fields}
            )

        handles.append(net.node_contact_builder.register_forward_hook(capture))

    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
        ):
            actual = model(sample, stats)
            expected = reference(other, stats)
            loss = actual.square().mean()
            expected_loss = expected.square().mean()
            if penalty:
                cfg = dict(
                    tolerance_mm=0.05, length_scale_mm=1.0, implementation="warp"
                )
                extra, _ = rollout_penetration_loss(actual, sample.graph, stats, cfg)
                expected_extra, _ = rollout_penetration_loss(
                    expected, other.graph, stats, cfg
                )
                assert extra > 0
                assert torch.equal(extra, expected_extra)
                loss = loss + 1e-5 * extra
                expected_loss = expected_loss + 1e-5 * expected_extra
        for handle in handles:
            handle.remove()
        assert actual.shape == (6, 4, 3)
        assert torch.equal(actual, expected)
        assert torch.equal(loss, expected_loss)
        assert len(graphs[0]) == len(graphs[1]) == 4
        for a, b in zip(*graphs):
            assert a["edge_index"].shape[1] > 0
            for key in fields:
                assert torch.equal(a[key], b[key]), key
        assert bool(fused_calls) == (device == "cuda")
        loss.backward()
        expected_loss.backward()
        for key in ("coords", "previous_coords"):
            grad = sample.node_features[key].grad
            assert torch.isfinite(grad).all() and grad.abs().sum() > 0
            assert torch.equal(grad, other.node_features[key].grad), key
        for (name, p), (_, q) in zip(
            model.named_parameters(), reference.named_parameters()
        ):
            assert (p.grad is None) == (q.grad is None), name
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
                assert torch.equal(p.grad, q.grad), name
    finally:
        for handle in handles:
            handle.remove()
        torch.use_deterministic_algorithms(deterministic)


@pytest.mark.parametrize(
    "device,dtype",
    [("cpu", torch.float32), ("cpu", torch.float64), ("cuda", torch.float64)],
)
def test_auto_preserves_cpu_and_fp64_reference_path(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from physicsnemo.experimental.models.meshtransolver import (
        SurfaceContactGraphBuilder,
    )

    builder = SurfaceContactGraphBuilder(narrow_phase="auto")
    assert not builder._use_fused_membership(
        torch.zeros(4, 3, device=device, dtype=dtype)
    )
