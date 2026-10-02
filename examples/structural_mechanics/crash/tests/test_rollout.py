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

import os
import sys
from typing import Dict

import pytest
import torch

# Ensure we can import modules from the crash example directory
THIS_DIR = os.path.dirname(__file__)
CRASH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CRASH_DIR not in sys.path:
    sys.path.insert(0, CRASH_DIR)

import rollout  # noqa: E402
from datapipe import SimSample  # noqa: E402


def make_sample(
    N: int = 5,
    T: int = 4,
    F: int = 2,
    Fo: int = 3,
    with_globals: bool = False,
) -> SimSample:
    """Build a deterministic synthetic crash sample for rollout tests."""
    torch.manual_seed(0)
    coords = torch.randn(N, 3)
    features = torch.randn(N, F)
    # Ground-truth future positions: [N, T-1, 3] (rollout steps)
    future = torch.randn(N, T - 1, Fo)

    class DummyGraph:
        """Minimal mutable graph container for testing rollout integration."""

        pass

    graph = DummyGraph()
    graph.edge_index = torch.empty((2, 0), dtype=torch.long)
    graph.edge_attr = torch.zeros(0, 4)
    graph.component_id = torch.zeros(N, dtype=torch.long)
    graph.shell_thickness = torch.zeros(N)

    node_inputs: Dict[str, torch.Tensor] = {"coords": coords, "features": features}
    global_features = None
    if with_globals:
        global_features = {
            "velocity_x": torch.tensor(1.0),
            "thickness_scale": torch.tensor(0.9),
            "rwall_origin_y": torch.tensor(-0.2),
        }
    return SimSample(
        node_features=node_inputs,
        node_target=future,
        graph=graph,
        global_features=global_features,
    )


def make_data_stats() -> Dict[str, Dict[str, torch.Tensor]]:
    # Broadcastable stats: [1, 3]
    """Return broadcastable identity statistics and global-feature metadata."""
    zeros = torch.zeros(1, 3)
    ones = torch.ones(1, 3)
    return {
        "node": {
            "pos_mean": zeros,
            "pos_std": ones,
            "norm_vel_mean": zeros,
            "norm_vel_std": ones,
            "norm_acc_mean": zeros,
            "norm_acc_std": ones,
        },
        "edge": {
            "edge_mean": torch.zeros(4),
            "edge_std": torch.ones(4),
        },
        "global_features": {
            "mean": torch.zeros(3),
            "std": torch.ones(3),
            "keys": ["velocity_x", "thickness_scale", "rwall_origin_y"],
        },
    }


@pytest.fixture(autouse=True)
def stub_parent_classes(monkeypatch):
    # Stub Transolver.__init__ and Transolver.forward (for TransolverOneShot)
    """Replace expensive model cores with deterministic rollout-test stubs."""

    def transolver_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)

    def transolver_forward(self, fx=None, embedding=None, time=None):
        assert embedding is not None
        return torch.zeros_like(embedding)

    monkeypatch.setattr(rollout.Transolver, "__init__", transolver_init, raising=True)
    monkeypatch.setattr(rollout.Transolver, "forward", transolver_forward, raising=True)

    # Stub GeoTransolver.__init__ and GeoTransolver.forward
    def geotransolver_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.stub_output_dim = int(kwargs.get("out_dim", 3))

    def geotransolver_forward(
        self,
        local_embedding=None,
        geometry=None,
        local_positions=None,
        global_embedding=None,
    ):
        assert geometry is not None
        self.stub_local_embedding = local_embedding
        self.stub_geometry = geometry
        self.stub_local_positions = local_positions
        self.stub_global_embedding = global_embedding
        return geometry.new_zeros((*geometry.shape[:-1], self.stub_output_dim))

    monkeypatch.setattr(
        rollout.GeoTransolver, "__init__", geotransolver_init, raising=True
    )
    monkeypatch.setattr(
        rollout.GeoTransolver, "forward", geotransolver_forward, raising=True
    )

    # Stub MeshGraphNet.__init__ and MeshGraphNet.forward
    def mgn_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)

    def mgn_forward(self, node_features=None, edge_features=None, graph=None):
        # Return zeros acceleration with shape [N, 3]
        assert node_features is not None
        N = node_features.shape[0]
        return torch.zeros(N, 3, dtype=node_features.dtype, device=node_features.device)

    monkeypatch.setattr(rollout.MeshGraphNet, "__init__", mgn_init, raising=True)
    monkeypatch.setattr(rollout.MeshGraphNet, "forward", mgn_forward, raising=True)

    # Stub FIGConvUNet.__init__ and FIGConvUNet.forward
    def figconvunet_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)

    def figconvunet_forward(self, vertices=None, features=None):
        # Return zeros with shape matching vertices
        # vertices: [B, N, 3], features: [B, N, F]
        # output: [B, N, 3]
        assert vertices is not None
        return torch.zeros_like(vertices), None

    monkeypatch.setattr(rollout.FIGConvUNet, "__init__", figconvunet_init, raising=True)
    monkeypatch.setattr(
        rollout.FIGConvUNet, "forward", figconvunet_forward, raising=True
    )

    # Stub the reusable mesh-attention parents while preserving the wrapper
    # contract (flat full-trajectory output from each core model).
    def mesh_attention_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.stub_output_dim = kwargs["output_dim"]
        self.use_contact = bool(kwargs.get("use_contact", False))

    def mesh_transolver_forward(
        self,
        node_features=None,
        edge_features=None,
        graph=None,
        contact_graph=None,
    ):
        assert node_features is not None
        self.stub_node_features = node_features
        self.stub_edge_features = edge_features
        self.stub_edge_feature_history = getattr(
            self, "stub_edge_feature_history", []
        ) + [edge_features.clone()]
        self.stub_contact_calls = getattr(self, "stub_contact_calls", 0) + 1
        self.stub_contact_graph = contact_graph
        return node_features.new_zeros((node_features.shape[0], self.stub_output_dim))

    def mesh_geo_forward(
        self,
        node_features=None,
        edge_features=None,
        graph=None,
        geometry=None,
        local_positions=None,
        global_embedding=None,
        contact_graph=None,
    ):
        assert node_features is not None
        assert geometry is not None
        assert global_embedding is not None
        self.stub_node_features = node_features
        self.stub_edge_features = edge_features
        self.stub_edge_feature_history = getattr(
            self, "stub_edge_feature_history", []
        ) + [edge_features.clone()]
        self.stub_geometry = geometry
        self.stub_local_positions = local_positions
        self.stub_global_embedding = global_embedding
        self.stub_contact_calls = getattr(self, "stub_contact_calls", 0) + 1
        self.stub_contact_graph = contact_graph
        return node_features.new_zeros((node_features.shape[0], self.stub_output_dim))

    for parent in (
        rollout.MeshTransolver,
        rollout.MeshGeoTransolver,
        rollout.MeshGeoFLARE,
    ):
        monkeypatch.setattr(parent, "__init__", mesh_attention_init, raising=True)
    monkeypatch.setattr(
        rollout.MeshTransolver, "forward", mesh_transolver_forward, raising=True
    )
    monkeypatch.setattr(
        rollout.MeshGeoTransolver, "forward", mesh_geo_forward, raising=True
    )
    monkeypatch.setattr(rollout.MeshGeoFLARE, "forward", mesh_geo_forward, raising=True)


@pytest.mark.parametrize(
    "model_cls",
    [
        rollout.MeshTransolverOneShot,
        rollout.MeshGeoTransolverOneShot,
        rollout.MeshGeoFLAREOneShot,
    ],
)
def test_mesh_attention_one_shot_wrappers(model_cls):
    """Verify mesh attention one shot wrappers."""
    N, T, F, Fo = 7, 6, 2, 5
    sample = make_sample(N=N, T=T, F=F, Fo=Fo, with_globals=True)
    model = model_cls(num_time_steps=T, output_dim=(T - 1) * Fo)

    output = model(sample=sample, data_stats={})

    assert output.shape == (N, T - 1, Fo)
    expected_features = 3 + F if model_cls is rollout.MeshGeoFLAREOneShot else 3 + F + 3
    assert model.stub_node_features.shape == (N, expected_features)
    assert "args" not in model._args["__args__"]
    assert model._args["__args__"]["num_time_steps"] == T
    torch.testing.assert_close(
        output[:, :, :3],
        sample.node_features["coords"].unsqueeze(1).expand(-1, T - 1, -1),
    )
    torch.testing.assert_close(output[:, :, 3:], torch.zeros_like(output[:, :, 3:]))
    if model_cls is not rollout.MeshTransolverOneShot:
        assert model.stub_geometry.shape == (N, 3)
        assert model.stub_global_embedding.shape == (1, 1, 3)
    if model_cls is rollout.MeshGeoFLAREOneShot:
        assert model.stub_local_positions.shape == (N, 3)


@pytest.mark.parametrize(
    "model_cls",
    [
        rollout.MeshTransolverAutoregressive,
        rollout.MeshGeoTransolverAutoregressive,
        rollout.MeshGeoFLAREAutoregressive,
    ],
)
def test_mesh_attention_autoregressive_wrappers_integrate_velocity(model_cls):
    """Verify mesh attention autoregressive wrappers integrate velocity."""
    N, T, Fo = 5, 4, 5
    sample = make_sample(N=N, T=T, F=0, Fo=Fo, with_globals=True)
    sample.global_features["velocity_x"] = torch.tensor(1.0)
    stats = make_data_stats()
    model = model_cls(
        num_time_steps=T,
        dt=0.1,
        velocity_unit_scale=1.0,
        checkpoint_rollout=False,
        enable_contact=False,
        input_dim_nodes=9,
        input_dim_edges=4,
        output_dim=Fo,
    )

    output = model(sample=sample, data_stats=stats)

    assert output.shape == (N, T - 1, Fo)
    assert model.stub_node_features.shape == (N, 9)
    for step in range(T - 1):
        expected = sample.node_features["coords"].clone()
        expected[:, 0] += (step + 1) * 0.1
        torch.testing.assert_close(output[:, step, :3], expected)
    torch.testing.assert_close(output[:, :, 3:], torch.zeros_like(output[:, :, 3:]))


def test_mesh_autoregressive_uses_two_frame_velocity_and_target_window_length():
    """Verify mesh autoregressive uses two frame velocity and target window length."""
    N = 4
    sample = make_sample(N=N, T=3, F=0, Fo=3, with_globals=False)
    sample.node_features["coords"].zero_()
    sample.node_features["previous_coords"] = torch.zeros(N, 3)
    sample.node_features["previous_coords"][:, 0] = -1.0
    model = rollout.MeshTransolverAutoregressive(
        num_time_steps=6,
        dt=1.0,
        initial_velocity_mode="previous_coords",
        rollout_steps_from_target=True,
        checkpoint_rollout=False,
        node_input_mode="velocity",
        enable_contact=False,
        use_contact=False,
        input_dim_nodes=3,
        input_dim_edges=4,
        output_dim=3,
    )

    output = model(sample=sample, data_stats=make_data_stats())

    assert output.shape == (N, 2, 3)
    torch.testing.assert_close(output[:, 0, 0], torch.ones(N))
    torch.testing.assert_close(output[:, 1, 0], torch.full((N,), 2.0))
    torch.testing.assert_close(output[:, :, 1:], torch.zeros(N, 2, 2))


def test_mesh_autoregressive_teacher_forces_training_but_not_eval():
    """Verify mesh autoregressive teacher forces training but not eval."""
    N, T, Fo = 3, 4, 5
    sample = make_sample(N=N, T=T, F=0, Fo=Fo, with_globals=True)
    sample.node_features["coords"] = torch.zeros(N, 3)
    sample.global_features["velocity_x"] = torch.tensor(0.0)
    sample.node_target.zero_()
    sample.node_target[:, 0, 0] = 1.0
    sample.node_target[:, 1, 0] = 3.0
    sample.node_target[:, 2, 0] = 6.0
    model = rollout.MeshGeoFLAREAutoregressive(
        num_time_steps=T,
        dt=0.1,
        velocity_unit_scale=1.0,
        checkpoint_rollout=False,
        teacher_forcing=True,
        enable_contact=False,
        use_contact=False,
        input_dim_nodes=9,
        input_dim_edges=4,
        output_dim=Fo,
    )

    model.train()
    training_output = model(sample=sample, data_stats=make_data_stats())
    expected_training_x = torch.tensor([0.0, 2.0, 5.0]).expand(N, -1)
    torch.testing.assert_close(training_output[:, :, 0], expected_training_x)
    assert model.stub_contact_calls == T - 1
    assert model.stub_contact_graph is None

    model.eval()
    evaluation_output = model(sample=sample, data_stats=make_data_stats())
    torch.testing.assert_close(
        evaluation_output[:, :, :3], torch.zeros_like(evaluation_output[:, :, :3])
    )


def test_mesh_autoregressive_parity_input_and_fixed_reference_edges():
    """Verify mesh autoregressive parity input and fixed reference edges."""
    sample = make_sample(N=3, T=4, F=0, Fo=5, with_globals=True)
    sample.global_features["velocity_x"] = torch.tensor(1.0)
    sample.graph.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    sample.graph.edge_attr = torch.tensor(
        [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0]]
    )
    model = rollout.MeshGeoFLAREAutoregressive(
        num_time_steps=4,
        dt=0.1,
        velocity_unit_scale=1.0,
        checkpoint_rollout=False,
        teacher_forcing=True,
        node_input_mode="velocity",
        enable_contact=False,
        use_contact=False,
        input_dim_nodes=3,
        input_dim_edges=4,
        output_dim=5,
    )

    model(sample=sample, data_stats=make_data_stats())

    expected_velocity = torch.tensor([1.0, 0.0, 0.0]).expand(3, -1)
    assert model.stub_node_features.shape == expected_velocity.shape
    torch.testing.assert_close(model.stub_edge_features, sample.graph.edge_attr)
    assert len(model.stub_edge_feature_history) == 3
    for edge_features in model.stub_edge_feature_history:
        torch.testing.assert_close(edge_features, sample.graph.edge_attr)


def test_mesh_autoregressive_position_velocity_state_omits_global_duplication():
    """Verify mesh autoregressive position velocity state omits global duplication."""
    sample = make_sample(N=3, T=3, F=0, Fo=5, with_globals=True)
    model = rollout.MeshGeoFLAREAutoregressive(
        num_time_steps=3,
        dt=0.1,
        velocity_unit_scale=1.0,
        checkpoint_rollout=False,
        teacher_forcing=True,
        node_input_mode="position_velocity",
        enable_contact=False,
        use_contact=False,
        input_dim_nodes=6,
        input_dim_edges=4,
        output_dim=5,
    )

    model(sample=sample, data_stats=make_data_stats())

    assert model.stub_node_features.shape == (3, 6)
    assert model.stub_global_embedding.shape == (1, 1, 3)


def test_mesh_autoregressive_returns_normalized_acceleration_supervision():
    """Verify mesh autoregressive returns normalized acceleration supervision."""
    sample = make_sample(N=2, T=4, F=0, Fo=5, with_globals=True)
    sample.node_features["coords"].zero_()
    sample.node_target.zero_()
    sample.node_target[:, 0, 0] = 1.0
    sample.node_target[:, 1, 0] = 4.0
    sample.node_target[:, 2, 0] = 9.0
    sample.global_features["velocity_x"] = torch.tensor(0.0)
    model = rollout.MeshGeoFLAREAutoregressive(
        num_time_steps=4,
        dt=1.0,
        velocity_unit_scale=1.0,
        checkpoint_rollout=False,
        teacher_forcing=True,
        node_input_mode="position_velocity",
        enable_contact=False,
        use_contact=False,
        input_dim_nodes=6,
        input_dim_edges=4,
        output_dim=5,
    )

    output = model(
        sample=sample,
        data_stats=make_data_stats(),
        teacher_forcing_probability=1.0,
        return_auxiliary=True,
    )

    assert isinstance(output, rollout.AutoregressiveRolloutOutput)
    assert output.trajectory.shape == (2, 3, 5)
    torch.testing.assert_close(output.normalized_acceleration, torch.zeros(2, 3, 3))
    expected = torch.tensor([1.0, 2.0, 2.0]).view(1, 3, 1).expand(2, -1, -1)
    torch.testing.assert_close(
        output.target_normalized_acceleration[:, :, :1], expected
    )
    torch.testing.assert_close(
        output.target_normalized_acceleration[:, :, 1:], torch.zeros(2, 3, 2)
    )
    assert output.acceleration_supervision_mask.tolist() == [True, True, True]

    closed_loop_output = model(
        sample=sample,
        data_stats=make_data_stats(),
        teacher_forcing_probability=0.0,
        return_auxiliary=True,
    )
    assert closed_loop_output.acceleration_supervision_mask.tolist() == [
        True,
        False,
        False,
    ]


def test_mesh_autoregressive_rejects_invalid_node_input_mode():
    """Verify mesh autoregressive rejects invalid node input mode."""
    kwargs = {
        "num_time_steps": 2,
        "checkpoint_rollout": False,
        "enable_contact": False,
        "input_dim_nodes": 9,
        "input_dim_edges": 4,
        "output_dim": 5,
        "node_input_mode": "bad",
    }
    with pytest.raises(ValueError, match="node_input_mode"):
        rollout.MeshGeoFLAREAutoregressive(**kwargs)


def test_mesh_autoregressive_converts_bumper_velocity_to_mm_per_second():
    """Verify mesh autoregressive converts bumper velocity to mm per second."""
    sample = make_sample(N=3, T=2, F=0, Fo=3, with_globals=True)
    sample.global_features["velocity_x"] = torch.tensor(-5.0)
    model = rollout.MeshTransolverAutoregressive(
        num_time_steps=2,
        dt=5.0e-3,
        velocity_unit_scale=1000.0,
        checkpoint_rollout=False,
        enable_contact=False,
        input_dim_nodes=9,
        input_dim_edges=4,
        output_dim=3,
    )

    output = model(sample=sample, data_stats=make_data_stats())
    expected = sample.node_features["coords"].clone()
    expected[:, 0] -= 25.0
    torch.testing.assert_close(output[:, 0], expected)


def test_mesh_autoregressive_keeps_contact_core_for_matched_ablation():
    """Verify mesh autoregressive keeps contact core for matched ablation."""
    sample = make_sample(N=3, T=2, F=0, Fo=3, with_globals=True)
    model = rollout.MeshTransolverAutoregressive(
        num_time_steps=2,
        checkpoint_rollout=False,
        enable_contact=False,
        use_contact=True,
        input_dim_nodes=9,
        input_dim_edges=4,
        output_dim=3,
    )

    model(sample=sample, data_stats=make_data_stats())

    assert model.use_contact
    assert model.stub_contact_graph is not None
    assert model.stub_contact_graph.edge_index.shape == (2, 0)


def test_mesh_autoregressive_applies_thickness_fallback_per_node():
    """Verify mesh autoregressive applies thickness fallback per node."""
    sample = make_sample(N=3, T=2, F=0, Fo=3, with_globals=True)
    sample.graph.shell_thickness = torch.tensor([1.8, 0.0, 2.2])
    model = rollout.MeshTransolverAutoregressive(
        num_time_steps=2,
        checkpoint_rollout=False,
        enable_contact=False,
        base_shell_thickness=2.0,
        input_dim_nodes=9,
        input_dim_edges=4,
        output_dim=3,
    )

    thickness = model._physical_shell_thickness(sample, make_data_stats(), num_nodes=3)

    torch.testing.assert_close(thickness, torch.tensor([1.8, 1.8, 2.2]))


def test_mesh_autoregressive_rebuilds_cylinder_contact_each_step():
    """Verify mesh autoregressive rebuilds cylinder contact each step."""
    N, T = 4, 4
    sample = make_sample(N=N, T=T, F=0, Fo=3, with_globals=True)
    sample.node_features["coords"] = torch.tensor(
        [
            [-40.0, 0.0, 0.0],
            [-39.0, 0.0, 0.0],
            [300.0, 0.0, 0.0],
            [400.0, 0.0, 0.0],
        ]
    )
    sample.global_features["velocity_x"] = torch.tensor(0.0)
    sample.global_features["rwall_origin_y"] = torch.tensor(0.0)
    model = rollout.MeshTransolverAutoregressive(
        num_time_steps=T,
        checkpoint_rollout=False,
        enable_contact=True,
        enable_node_contact=False,
        enable_cylinder_contact=True,
        input_dim_nodes=9,
        input_dim_edges=4,
        output_dim=3,
    )

    model(sample=sample, data_stats=make_data_stats())

    assert model.stub_contact_calls == T - 1
    assert model.stub_contact_graph is not None
    assert model.stub_contact_graph.obstacle_mask.all()


def test_geotransolver_autoregressive_rollout_eval():
    """Verify geotransolver autoregressive rollout eval."""
    N, T, F = 5, 4, 2
    sample = make_sample(N=N, T=T, F=F)
    stats = make_data_stats()

    model = rollout.GeoTransolverAutoregressiveRolloutTraining(
        dt=5e-3, initial_vel=torch.zeros(1, 3), num_time_steps=T
    )
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (N, T - 1, 3)


def test_geotransolver_teacher_forces_training_but_not_eval():
    """Verify geotransolver teacher forces training but not eval."""
    N, T, Fo = 3, 4, 5
    sample = make_sample(N=N, T=T, F=0, Fo=Fo, with_globals=True)
    sample.node_features["coords"] = torch.zeros(N, 3)
    sample.global_features["velocity_x"] = torch.tensor(0.0)
    sample.node_target.zero_()
    sample.node_target[:, 0, 0] = 1.0
    sample.node_target[:, 1, 0] = 3.0
    sample.node_target[:, 2, 0] = 6.0
    model = rollout.GeoTransolverAutoregressive(
        num_time_steps=T,
        dt=0.1,
        velocity_unit_scale=1.0,
        checkpoint_rollout=False,
        teacher_forcing=True,
        functional_dim=3,
        out_dim=Fo,
    )

    model.train()
    training_output = model(sample=sample, data_stats=make_data_stats())
    expected_training_x = torch.tensor([0.0, 2.0, 5.0]).expand(N, -1)
    torch.testing.assert_close(training_output[:, :, 0], expected_training_x)
    assert model.stub_local_embedding.shape == (1, N, 3)
    assert model.stub_geometry.shape == (1, N, 3)
    assert model.stub_local_positions.shape == (1, N, 3)
    assert model.stub_global_embedding.shape == (1, 1, 3)
    torch.testing.assert_close(
        training_output[:, :, 3:], torch.zeros_like(training_output[:, :, 3:])
    )

    model.eval()
    evaluation_output = model(sample=sample, data_stats=make_data_stats())
    torch.testing.assert_close(
        evaluation_output[:, :, :3], torch.zeros_like(evaluation_output[:, :, :3])
    )


def test_geotransolver_previous_coords_uses_target_length_for_tbptt():
    """Verify geotransolver previous coords uses target length for TBPTT."""
    N, target_steps = 3, 2
    sample = make_sample(N=N, T=target_steps + 1, F=1)
    sample.node_features["coords"] = torch.zeros(N, 3)
    sample.node_features["previous_coords"] = torch.tensor([-0.1, 0.0, 0.0]).expand(
        N, -1
    )
    model = rollout.GeoTransolverAutoregressive(
        num_time_steps=6,
        dt=0.1,
        initial_velocity_mode="previous_coords",
        rollout_steps_from_target=True,
        checkpoint_rollout=False,
        functional_dim=4,
        out_dim=3,
    )
    model.eval()

    output = model(sample=sample, data_stats=make_data_stats())

    assert output.shape == (N, target_steps, 3)
    expected_x = torch.tensor([0.1, 0.2]).expand(N, -1)
    torch.testing.assert_close(output[:, :, 0], expected_x)
    torch.testing.assert_close(output[:, :, 1:], torch.zeros(N, target_steps, 2))


def test_geotransolver_time_conditional_rollout_eval():
    """Verify geotransolver time conditional rollout eval."""
    N, T, F = 6, 5, 3
    sample = make_sample(N=N, T=T, F=F)
    stats = make_data_stats()

    model = rollout.GeoTransolverTimeConditional(num_time_steps=T)
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (N, T - 1, 3)


def test_geotransolver_one_step_rollout_eval():
    """Verify geotransolver one step rollout eval."""
    N, T, F = 7, 6, 1
    sample = make_sample(N=N, T=T, F=F)
    stats = make_data_stats()

    model = rollout.GeoTransolverOneStepRollout(
        dt=5e-3, initial_vel=torch.zeros(1, 3), num_time_steps=T
    )
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (N, T - 1, 3)


@pytest.mark.xfail(
    not hasattr(rollout, "MeshGraphNetAutoregressiveRolloutTraining"),
    reason="MeshGraphNet autoregressive wrapper is not implemented",
    strict=True,
)
def test_meshgraphnet_autoregressive_rollout_eval():
    """Verify meshgraphnet autoregressive rollout eval."""
    N, T, F = 4, 4, 2
    sample = make_sample(N=N, T=T, F=F)
    stats = make_data_stats()

    model = rollout.MeshGraphNetAutoregressiveRolloutTraining(
        dt=5e-3, initial_vel=torch.zeros(1, 3), num_time_steps=T
    )
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (N, T - 1, 3)


@pytest.mark.xfail(
    not hasattr(rollout, "MeshGraphNetTimeConditionalRollout"),
    reason="MeshGraphNet time-conditional wrapper is not implemented",
    strict=True,
)
def test_meshgraphnet_time_conditional_rollout_eval():
    """Verify meshgraphnet time conditional rollout eval."""
    N, T, F = 3, 5, 4
    sample = make_sample(N=N, T=T, F=F)
    stats = make_data_stats()

    model = rollout.MeshGraphNetTimeConditionalRollout(num_time_steps=T)
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (N, T - 1, 3)


@pytest.mark.xfail(
    not hasattr(rollout, "MeshGraphNetOneStepRollout"),
    reason="MeshGraphNet one-step wrapper is not implemented",
    strict=True,
)
def test_meshgraphnet_one_step_rollout_eval():
    """Verify meshgraphnet one step rollout eval."""
    N, T, F = 8, 3, 0
    # allow zero features
    torch.manual_seed(0)
    coords = torch.randn(N, 3)
    future = torch.randn(N, T - 1, 3)

    class DummyGraph:
        """Minimal mutable graph container for testing rollout integration."""

        pass

    graph = DummyGraph()
    graph.edge_attr = torch.zeros(0, 1)

    node_inputs = {"coords": coords, "features": coords.new_zeros((N, 0))}
    sample = SimSample(node_features=node_inputs, node_target=future, graph=graph)
    stats = make_data_stats()

    model = rollout.MeshGraphNetOneStepRollout(
        dt=5e-3, initial_vel=torch.zeros(1, 3), num_time_steps=T
    )
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (N, T - 1, 3)


@pytest.mark.xfail(
    not hasattr(rollout, "FIGConvUNetTimeConditionalRollout"),
    reason="FIGConvUNet time-conditional wrapper is not implemented",
    strict=True,
)
def test_figconvunet_time_conditional_rollout_eval():
    """Verify figconvunet time conditional rollout eval."""
    N, T, F = 6, 5, 3
    sample = make_sample(N=N, T=T, F=F)
    stats = make_data_stats()

    model = rollout.FIGConvUNetTimeConditionalRollout(num_time_steps=T)
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (N, T - 1, 3)


@pytest.mark.xfail(
    not hasattr(rollout, "FIGConvUNetOneStepRollout"),
    reason="FIGConvUNet one-step wrapper is not implemented",
    strict=True,
)
def test_figconvunet_one_step_rollout_eval():
    """Verify figconvunet one step rollout eval."""
    N, T, F = 7, 6, 1
    sample = make_sample(N=N, T=T, F=F)
    stats = make_data_stats()

    model = rollout.FIGConvUNetOneStepRollout(
        dt=5e-3, initial_vel=torch.zeros(1, 3), num_time_steps=T
    )
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (N, T - 1, 3)


@pytest.mark.xfail(
    not hasattr(rollout, "FIGConvUNetAutoregressiveRolloutTraining"),
    reason="FIGConvUNet autoregressive wrapper is not implemented",
    strict=True,
)
def test_figconvunet_autoregressive_rollout_eval():
    """Verify figconvunet autoregressive rollout eval."""
    N, T, F = 5, 4, 2
    sample = make_sample(N=N, T=T, F=F)
    stats = make_data_stats()

    model = rollout.FIGConvUNetAutoregressiveRolloutTraining(
        dt=5e-3, initial_vel=torch.zeros(1, 3), num_time_steps=T
    )
    model.eval()

    out = model.forward(sample=sample, data_stats=stats)
    assert out.shape == (N, T - 1, 3)
