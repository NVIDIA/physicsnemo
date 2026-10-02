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

import copy
import io
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datapipe import CrashBaseDataset, CrashGraphDataset
from material_contact import material_contact_exclusions
from training_state import capture_rng_state, restore_rank_rng, restore_rng_state
from training_state import configure_deterministic_training
from test_deformer_contact import make_model, make_sample
from vtp_reader import _load_vtp_cache, _save_vtp_cache
from physicsnemo.experimental.models.meshtransolver import FunctionalContactGraphBuilder


def test_stationary_quad_patch_does_not_contact_itself():
    """Verify stationary quad patch does not contact itself."""
    points = torch.tensor([[x * 5.0, y * 5.0, 0.0] for y in range(3) for x in range(3)])
    cells = [
        [3 * y + x, 3 * y + x + 1, 3 * (y + 1) + x + 1, 3 * (y + 1) + x]
        for y in range(2)
        for x in range(2)
    ]
    edges = torch.tensor(
        [(cell[i], cell[(i + 1) % 4]) for cell in cells for i in range(4)]
    ).t()
    exclusions = material_contact_exclusions(
        edges, 9, 2, np.array([v for cell in cells for v in [4, *cell]])
    )
    builder = FunctionalContactGraphBuilder(8, 16, "torch")
    topology = builder.prepare_topology(
        9, edges, device="cpu", extra_exclusion_edges=exclusions
    )
    assert builder(points, edges).edge_index.numel() > 0
    assert builder(points, topology=topology).edge_index.numel() == 0


def test_distant_same_component_fold_contact_is_preserved():
    """Verify distant same component fold contact is preserved."""
    points = torch.tensor(
        [
            [0.0, 0, 0],
            [5.0, 0, 0],
            [10.0, 0, 0],
            [15.0, 0, 0],
            [10.0, 5, 0],
            [5.0, 5, 0],
            [0.0, 0.3, 0],
        ]
    )
    edges = torch.stack((torch.arange(6), torch.arange(1, 7)))
    exclusions = material_contact_exclusions(edges, 7, 2)
    builder = FunctionalContactGraphBuilder(1, 16, "torch")
    topology = builder.prepare_topology(
        7, edges, device="cpu", extra_exclusion_edges=exclusions
    )
    graph = builder(points, topology=topology)
    assert set(map(tuple, graph.edge_index.t().tolist())) == {(0, 6), (6, 0)}


def test_polygon_diagonals_beyond_two_hops_are_excluded():
    """Verify polygon diagonals beyond two hops are excluded."""
    edges = torch.stack((torch.arange(6), torch.arange(6).roll(-1)))
    exclusions = material_contact_exclusions(edges, 6, 2, [6, 0, 1, 2, 3, 4, 5])
    assert exclusions.shape == (2, 30)
    assert (exclusions[0] != exclusions[1]).all()


def test_material_exclusion_sparse_scaling():
    """Verify material exclusion sparse scaling."""
    n = 100_000
    edges = torch.stack((torch.arange(n - 1), torch.arange(1, n)))
    exclusions = material_contact_exclusions(edges, n, 2)
    assert exclusions.shape == (2, 4 * n - 6)


def test_topology_cache_requires_elements_and_roundtrips(tmp_path):
    """Verify topology cache requires elements and roundtrips."""
    source = tmp_path / "case.vtp"
    source.touch()
    cache = str(tmp_path / "case.pt")
    args = (
        cache,
        str(source),
        np.array([0, 1]),
        np.array([1, 2]),
        np.zeros((3, 3, 3)),
        {},
    )
    _save_vtp_cache(*args)
    assert _load_vtp_cache(cache, str(source)) is not None
    assert _load_vtp_cache(cache, str(source), True) is None
    packed = np.array([3, 0, 1, 2])
    _save_vtp_cache(*args, mesh_cells=packed)
    np.testing.assert_array_equal(_load_vtp_cache(cache, str(source), True)[4], packed)


def test_real_vtp_datapipe_caches_and_shares_exclusions(tmp_path):
    """Verify real VTP datapipe caches and shares exclusions."""
    import pyvista as pv
    from vtp_reader import Reader

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    points = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [0.0, 1, 0]])
    mesh = pv.PolyData(points, np.array([4, 0, 1, 2, 3]))
    mesh.point_data["thickness"] = np.ones(4)
    for t in range(3):
        mesh.point_data[f"displacement_t0.{t * 5:03d}"] = np.ones((4, 3)) * t
    for i in range(2):
        mesh.save(data_dir / f"Run{i}.vtp")
    options = dict(
        data_dir=str(data_dir),
        num_samples=2,
        num_steps=3,
        initial_history_steps=2,
        static_features=["thickness"],
        contact_exclusion_hops=2,
        contact_require_elements=True,
        stats_dir=str(tmp_path / "stats"),
    )
    reader = Reader(cache_dir=str(tmp_path / "cache"), include_contact_topology=True)
    for _ in range(2):
        dataset = CrashGraphDataset(reader=reader, **options)
        assert dataset.graphs[0].contact_exclusion_edges.shape == (2, 12)
        assert (
            dataset.graphs[0].contact_exclusion_edges
            is dataset.graphs[1].contact_exclusion_edges
        )
        sample = dataset[0].to(torch.device("meta"))
        assert sample.graph.contact_exclusion_edges.device.type == "meta"
        assert dataset.graphs[0].contact_exclusion_edges.device.type == "cpu"
    with pytest.raises(ValueError, match="element connectivity"):
        CrashGraphDataset(reader=Reader(), **options)


def window_dataset():
    """Build a synthetic two-history-frame dataset for seeded BPTT windows."""
    dataset = CrashBaseDataset.__new__(CrashBaseDataset)
    dataset.num_samples = 1
    dataset.num_steps = 26
    dataset.initial_history_steps = 2
    dataset.rollout_window_steps = 4
    dataset.sample_type = "random_time_window"
    dataset.mesh_pos_seq = [torch.arange(26.0).view(26, 1, 1).expand(26, 2, 3)]
    dataset.node_features_data = [torch.ones(2, 1)]
    dataset.dynamic_targets = []
    dataset.target_series_data = [{}]
    dataset.window_seed = 42
    dataset.set_epoch(0)
    return dataset


def test_windows_independent_of_global_rng_order_model_and_restart():
    """Verify windows independent of global RNG order model and restart."""
    data = window_dataset()

    def windows():
        return [int(data.build_xy(0, None, i)[0]["coords"][0, 0]) for i in range(100)]

    before_rng = torch.get_rng_state()
    original = windows()
    assert torch.equal(before_rng, torch.get_rng_state())
    torch.randn(10000)
    make_model()
    assert original == windows()
    data.set_epoch(5)
    fifth = windows()
    assert fifth != original
    fresh = window_dataset()
    fresh.set_epoch(5)
    assert fifth == [
        int(fresh.build_xy(0, None, i)[0]["coords"][0, 0]) for i in range(100)
    ]
    assert set(fifth) <= set(range(1, 22))
    assert len(set(fifth)) >= 15


def test_shared_model_parameters_and_rng_match_no_contact_exactly():
    """Verify shared model parameters and RNG match no contact exactly."""
    torch.manual_seed(42)
    plain = make_model(
        use_contact=False, enable_contact=False, contact_isolate_rng=True
    )
    plain_rng = torch.get_rng_state()
    torch.manual_seed(42)
    contact = make_model(contact_isolate_rng=True)
    assert torch.equal(plain_rng, torch.get_rng_state())
    shared = dict(plain.named_parameters())
    assert shared
    for name, parameter in contact.named_parameters():
        if name in shared:
            assert torch.equal(parameter, shared[name]), name


def test_stochastic_optimizer_restart_replays_exactly():
    """Verify stochastic optimizer restart replays exactly."""
    torch.manual_seed(83)
    random.seed(83)
    np.random.seed(83)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 1)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(94)

    def step():
        x = torch.randn(8, 3, generator=generator) + random.random() + np.random.rand()
        optimizer.zero_grad()
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach().clone()

    step()
    snapshot = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": capture_rng_state({"loader": generator}),
        },
        snapshot,
    )
    losses = [step() for _ in range(5)]
    reference = copy.deepcopy(model.state_dict())
    torch.randn(1000)
    snapshot.seek(0)
    saved = torch.load(snapshot, weights_only=True)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    restore_rank_rng([saved["rng"]], 0, 1, {"loader": generator})
    for expected in losses:
        assert torch.equal(step(), expected)
    for name, value in model.state_dict().items():
        assert torch.equal(value, reference[name])
    with pytest.raises(ValueError, match="world size"):
        restore_rank_rng([saved["rng"]], 0, 2, {"loader": generator})


def test_deterministic_execution_requires_valid_workspace(monkeypatch):
    """Verify deterministic execution requires valid workspace."""
    import os

    old = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        with pytest.raises(RuntimeError, match="before CUDA"):
            configure_deterministic_training(True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
        configure_deterministic_training(True)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
        with pytest.raises(ValueError, match="Unsupported"):
            configure_deterministic_training(True)
        configure_deterministic_training(False)  # legacy policy is not altered
    finally:
        torch.use_deterministic_algorithms(old[0], warn_only=old[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old[2:]


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_v2_real_rollout_bptt(checkpoint, implementation, device):
    """Verify nearest-node contact real rollout BPTT."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = make_model(
        contact_selection_taper=True,
        contact_normal_epsilon=0.1,
        contact_activation_distance=5.0,
        contact_isolate_rng=True,
        checkpoint_rollout=checkpoint,
        checkpoint_contact=checkpoint,
        contact_search_implementation=implementation,
    ).to(device)
    sample, stats = make_sample(device)
    graph = sample.graph
    graph.contact_exclusion_edges = material_contact_exclusions(
        graph.edge_index.cpu(), 6, 2
    ).to(device)
    output = model(sample, stats)
    output[:, -1].square().mean().backward()
    assert torch.isfinite(output).all()
    assert model.contact_block.gate.grad.abs() > 0
    for parameter in model.contact_block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert sample.node_features["coords"].grad.norm() > 0
    assert sample.node_features["previous_coords"].grad.norm() > 0
