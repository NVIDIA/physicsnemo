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

"""Real-model contact rollouts; no stubbed DeFormer parents."""

import copy
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.data import Data

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))

from contact_graph import BumperCylinderContactEncoder  # noqa: E402
from datapipe import SimSample  # noqa: E402
from rollout import MeshGeoFLAREAutoregressive  # noqa: E402


def make_model(**overrides):
    """Construct a small DeFormer contact model with explicit test overrides."""
    kwargs = dict(
        num_time_steps=4,
        dt=0.1,
        functional_dim=4,
        out_dim=3,
        geometry_dim=3,
        global_dim=None,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        n_layers=1,
        use_te=False,
        attention_type="GALE_FA",
        input_dim_edges=4,
        mesh_hidden_dim=16,
        num_pre_processor_layers=1,
        num_post_processor_layers=1,
        num_pre_processor_checkpoint_segments=1,
        num_post_processor_checkpoint_segments=1,
        mesh_context_fusion="pre_post",
        mesh_context_use_global=False,
        mesh_pre_residual_gate_init=0.2,
        mesh_post_residual_gate_init=0.1,
        initial_velocity_mode="previous_coords",
        rollout_steps_from_target=True,
        checkpoint_rollout=True,
        teacher_forcing=False,
        node_input_mode="velocity",
        use_contact=True,
        enable_contact=True,
        enable_node_contact=True,
        enable_cylinder_contact=False,
        contact_graph_backend="nearest_k",
        contact_search_implementation="torch",
        node_contact_radius=10,
        contact_max_neighbors=3,
        contact_gate_init=0.1,
        base_shell_thickness=0,
        contact_include_velocity=True,
        contact_velocity_scale=2,
        contact_smooth_cutoff=True,
    )
    kwargs.update(overrides)
    return MeshGeoFLAREAutoregressive(**kwargs)


def make_sample(device="cpu"):
    """Build a deterministic synthetic crash sample for rollout tests."""
    torch.manual_seed(121)
    n = 6
    coords = (torch.randn(n, 3, device=device) * 0.2).requires_grad_()
    previous = (coords.detach() - 0.01 * torch.randn_like(coords)).requires_grad_()
    source = torch.arange(n - 1, device=device)
    edges = torch.stack(
        (torch.cat((source, source + 1)), torch.cat((source + 1, source)))
    )
    graph = Data(
        edge_index=edges,
        edge_attr=torch.randn(2 * (n - 1), 4, device=device),
        num_nodes=n,
        shell_thickness=torch.linspace(0.1, 0.3, n, device=device),
    )
    sample = SimSample(
        {
            "coords": coords,
            "previous_coords": previous,
            "features": torch.randn(n, 1, device=device),
        },
        torch.randn(n, 3, 3, device=device),
        graph=graph,
    )
    zeros = torch.zeros(1, 3, device=device)
    ones = torch.ones(1, 3, device=device)
    stats = {
        "node": dict(
            pos_mean=ones * 4,
            pos_std=torch.tensor([[2.0, 3.0, 4.0]], device=device),
            norm_vel_mean=zeros,
            norm_vel_std=ones,
            norm_acc_mean=zeros,
            norm_acc_std=ones,
        )
    }
    return sample, stats


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_checkpointed_bptt_and_physical_contact_features(implementation, device):
    """Verify real checkpointed BPTT and physical contact features."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(123)
    model = make_model(contact_search_implementation=implementation).to(device).train()
    sample, stats = make_sample(device)
    contacts = []

    def capture(module, args, kwargs, output):
        output.edge_features.retain_grad()
        output.edge_weights.retain_grad()
        contacts.append((kwargs, output))

    handle = model.node_contact_builder.register_forward_hook(capture, with_kwargs=True)
    output = model(sample, stats)
    handle.remove()
    assert output.shape == (6, 3, 3)
    assert len(contacts) == 3
    assert all(item[0]["topology"] is contacts[0][0]["topology"] for item in contacts)
    # Units: d/dt of normalized coordinates times position std; never add mean.
    expected_velocity = (
        (sample.node_features["coords"] - sample.node_features["previous_coords"])
        / model.dt
        * stats["node"]["pos_std"]
    )
    torch.testing.assert_close(contacts[0][0]["velocities"], expected_velocity)
    first_graph = contacts[0][1]
    source, destination = first_graph.edge_index
    physical = (
        sample.node_features["coords"] * stats["node"]["pos_std"]
        + stats["node"]["pos_mean"]
    )
    torch.testing.assert_close(
        first_graph.edge_features[:, :3],
        (physical[source] - physical[destination]) / 10,
    )
    torch.testing.assert_close(
        first_graph.edge_features[:, 8:11],
        (expected_velocity[source] - expected_velocity[destination]) / 2,
    )
    # Last-frame-only loss must reach earlier contact features and initial state.
    output[:, -1].square().sum().backward()
    assert sample.node_features["coords"].grad.abs().sum() > 0
    assert sample.node_features["previous_coords"].grad.abs().sum() > 0
    for _, graph in contacts:
        assert torch.isfinite(graph.edge_features.grad).all()
        assert graph.edge_features.grad.abs().sum() > 0
        assert torch.isfinite(graph.edge_weights.grad).all()
        assert graph.edge_weights.grad.abs().sum() > 0
    assert model.contact_block.gate.grad.abs() > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpointed_and_uncheckpointed_rollouts_match(device):
    """Verify checkpointed and uncheckpointed rollouts match."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(123)
    model = make_model().to(device).train()
    other = copy.deepcopy(model)
    other.checkpoint_rollout = False
    sample, stats = make_sample(device)
    sample2 = copy.deepcopy(sample)
    result = model(sample, stats)
    reference = other(sample2, stats)
    torch.testing.assert_close(result, reference)
    result[:, -1].square().sum().backward()
    reference[:, -1].square().sum().backward()
    torch.testing.assert_close(
        sample.node_features["coords"].grad, sample2.node_features["coords"].grad
    )
    for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, msg=name)


@pytest.mark.parametrize("mesh_width", [16, 12])
def test_contact_gate_learns_with_zero_mesh_gate_and_zero_contact_is_identity(
    mesh_width,
):
    """Verify contact gate learns with zero mesh gate and zero contact is identity."""
    torch.manual_seed(123)
    model = make_model(
        mesh_pre_residual_gate_init=0, mesh_hidden_dim=mesh_width, contact_gate_init=0
    ).eval()
    sample, stats = make_sample()
    with_contact = model(sample, stats)
    model.enable_contact = False
    empty_contact = model(sample, stats)
    torch.testing.assert_close(with_contact, empty_contact, atol=0, rtol=0)
    with_contact.square().sum().backward()
    assert model.contact_block.gate.grad.abs() > 0
    assert torch.isfinite(model.contact_block.gate.grad)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("outer_checkpoint", [False, True])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_memory_options_preserve_bptt_values_and_gradients(
    device, outer_checkpoint, offload, weighted
):
    """Verify memory options preserve BPTT values and gradients."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(126)
    common = dict(
        checkpoint_rollout=outer_checkpoint,
        contact_smooth_cutoff=weighted,
        num_post_processor_layers=2,
        num_post_processor_checkpoint_segments=2,
    )
    reference = make_model(**common).to(device).train()
    optimized = (
        make_model(
            **{**common, "num_post_processor_checkpoint_segments": 4},
            checkpoint_contact=True,
            num_pre_processor_checkpoint_segments=2,
            checkpoint_offloading=offload,
        )
        .to(device)
        .train()
    )
    # Execution flags must not introduce, rename, or resize learned tensors.
    optimized.load_state_dict(reference.state_dict(), strict=True)
    assert reference.state_dict().keys() == optimized.state_dict().keys()
    sample, stats = make_sample(device)
    other_sample = copy.deepcopy(sample)
    graphs = [[], []]

    def capture_into(destination):
        def capture(module, args, kwargs, output):
            output.edge_features.retain_grad()
            if output.edge_weights is not None:
                output.edge_weights.retain_grad()
            destination.append(output)

        return capture

    handles = [
        model.node_contact_builder.register_forward_hook(
            capture_into(destination), with_kwargs=True
        )
        for model, destination in zip((reference, optimized), graphs)
    ]
    expected = reference(sample, stats)
    actual = optimized(other_sample, stats)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(actual, expected)
    expected[:, -1].square().sum().backward()
    actual[:, -1].square().sum().backward()
    for key in ("coords", "previous_coords"):
        torch.testing.assert_close(
            sample.node_features[key].grad, other_sample.node_features[key].grad
        )
    for (name, parameter), (_, other) in zip(
        reference.named_parameters(), optimized.named_parameters()
    ):
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, msg=name)
    assert len(graphs[0]) == len(graphs[1]) == 3
    for original, updated in zip(*graphs):
        torch.testing.assert_close(
            original.edge_features.grad, updated.edge_features.grad
        )
        assert updated.edge_features.grad.abs().sum() > 0
        if weighted:
            torch.testing.assert_close(
                original.edge_weights.grad, updated.edge_weights.grad
            )
            assert updated.edge_weights.grad.abs().sum() > 0


@pytest.mark.parametrize("offload", [False, True])
def test_memory_overrides_only_change_execution_options(offload):
    """Memory tradeoffs are overrides, not separate experiment files."""
    with initialize_config_dir(version_base=None, config_dir=str(CRASH_DIR / "conf")):
        base = compose(config_name="crash_deformer_contact_autoregressive")
        config = compose(
            config_name="crash_deformer_contact_autoregressive",
            overrides=[f"model.checkpoint_offloading={str(offload).lower()}"],
        )
    actual = OmegaConf.to_container(config, resolve=False)
    expected = OmegaConf.to_container(base, resolve=False)
    actual["model"].pop("checkpoint_offloading")
    expected["model"].pop("checkpoint_offloading")
    assert actual == expected
    assert config.model.checkpoint_contact
    assert config.model.num_pre_processor_checkpoint_segments == 2
    assert config.model.num_post_processor_checkpoint_segments == 4
    assert config.model.checkpoint_offloading == offload


def test_evaluation_never_reads_future_targets_and_post_graph_stays_structural():
    """Verify evaluation never reads future targets and post graph stays structural."""
    model = make_model(teacher_forcing=True).eval()
    sample, stats = make_sample()
    structural_edges = sample.graph.edge_index.clone()
    seen = []

    def post_hook(module, args):
        seen.append(args[-1].edge_index.clone())

    handle = model.post_processor.register_forward_pre_hook(post_hook)
    with torch.no_grad():
        first = model(sample, stats)
        sample.node_target.fill_(1000)
        second = model(sample, stats)
    handle.remove()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert len(seen) == 6
    assert all(torch.equal(edges, structural_edges) for edges in seen)


@pytest.mark.parametrize("checkpoint_contact", [False, True])
def test_full_car_thickness_without_globals_and_empty_kinematic_contact(
    checkpoint_contact,
):
    """Verify full car thickness without globals and empty kinematic contact."""
    model = make_model(checkpoint_contact=checkpoint_contact)
    sample, stats = make_sample()
    torch.testing.assert_close(
        model._physical_shell_thickness(sample, stats, 6), sample.graph.shell_thickness
    )
    sample.graph.shell_thickness = None
    torch.testing.assert_close(
        model._physical_shell_thickness(sample, stats, 6), torch.zeros(6)
    )
    model.enable_node_contact = False
    output = model(sample, stats)
    output.sum().backward()
    assert torch.isfinite(output).all()


def test_stationary_cylinder_kinematics_and_smooth_weights():
    """Verify stationary cylinder kinematics and smooth weights."""
    encoder = BumperCylinderContactEncoder(
        center_x=0,
        radius=1,
        search_distance=1,
        include_velocity=True,
        velocity_scale=2,
        smooth_cutoff=True,
    )
    positions = torch.tensor([[1.5, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad=True)
    velocities = torch.tensor([[-2.0, 0, 0], [-1.0, 0, 0]], requires_grad=True)
    graph = encoder(positions, torch.tensor(0.0), velocities=velocities)
    assert graph.edge_features.shape == (2, 12)
    torch.testing.assert_close(graph.edge_features[:, 8:11], -velocities / 2)
    assert graph.edge_features[0, 11] < 0  # approaching the stationary surface
    assert graph.edge_weights[1] == 0
    (graph.edge_features.square().sum() + graph.edge_weights.sum()).backward()
    assert positions.grad.abs().sum() > 0 and velocities.grad.abs().sum() > 0


def test_contact_disable_override_preserves_model_and_training():
    """Disabling messages alone keeps the contact module for a paired ablation."""
    with initialize_config_dir(version_base=None, config_dir=str(CRASH_DIR / "conf")):
        enabled = compose(config_name="crash_deformer_contact_autoregressive")
        disabled = compose(
            config_name="crash_deformer_contact_autoregressive",
            overrides=["model.enable_contact=false"],
        )
    assert disabled.model.use_contact
    assert not disabled.model.enable_contact
    actual = OmegaConf.to_container(disabled, resolve=False)
    expected = OmegaConf.to_container(enabled, resolve=False)
    actual["model"].pop("enable_contact")
    expected["model"].pop("enable_contact")
    assert actual == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(contact_dim=8),
        dict(exclude_same_component=True),
        dict(contact_graph_backend="legacy"),
    ],
)
def test_incompatible_contact_options_rejected(kwargs):
    """Verify incompatible contact options rejected."""
    with pytest.raises(ValueError):
        make_model(**kwargs)


@pytest.mark.parametrize(
    "device,dtype",
    [("cpu", torch.bfloat16), ("cuda", torch.bfloat16), ("cuda", torch.float16)],
)
@pytest.mark.parametrize("checkpoint_contact", [False, True])
def test_mixed_precision_contact_rollout(device, dtype, checkpoint_contact):
    """Verify mixed precision contact rollout."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = (
        make_model(
            contact_search_implementation="warp", checkpoint_contact=checkpoint_contact
        )
        .to(device)
        .train()
    )
    sample, stats = make_sample(device)
    with torch.autocast(device_type=device, dtype=dtype):
        result = model(sample, stats)
        loss = result[:, -1].float().square().mean()
    loss.backward()
    assert torch.isfinite(result).all()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
