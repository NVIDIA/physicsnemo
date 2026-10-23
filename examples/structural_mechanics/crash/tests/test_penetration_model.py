# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

CRASH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH))
from penetration_loss import rollout_penetration_loss  # noqa: E402
from test_deformer_contact import make_model  # noqa: E402
from test_surface_contact_recipe import surface_sample  # noqa: E402


@pytest.mark.parametrize("enabled,epoch", [(False, 100), (True, 0), (True, 99)])
def test_trainer_optional_penalty_weight_and_gradient(monkeypatch, enabled, epoch):
    import penetration_loss
    from train import Trainer

    cfg = dict(max_weight=1e-5, warmup_epochs=50, ramp_epochs=50)
    trainer = SimpleNamespace(
        penetration_cfg=cfg if enabled else None,
        last_loss_components={},
        data_stats={},
    )
    trajectory = torch.tensor([2.0], requires_grad=True)
    base = trajectory.square().sum()
    calls = []

    def penalty(value, graph, stats, settings):
        calls.append((graph, stats, settings))
        return value.pow(3).sum(), {"penetration_active_nodes": value.new_tensor(1)}

    monkeypatch.setattr(penetration_loss, "rollout_penetration_loss", penalty)
    sample = SimpleNamespace(graph=object())
    total = Trainer.add_penetration_loss(trainer, base, trajectory, sample, epoch)
    if not enabled or epoch == 0:
        assert total is base
        assert not calls
        total.backward()
        torch.testing.assert_close(trajectory.grad, torch.tensor([4.0]))
    else:
        assert len(calls) == 1
        assert calls[0] == (sample.graph, trainer.data_stats, cfg)
        torch.testing.assert_close(total, torch.tensor(4.0 + 1e-5 * 8.0))
        total.backward()
        torch.testing.assert_close(trajectory.grad, torch.tensor([4.0 + 1e-5 * 12.0]))
        assert not trainer.last_loss_components["penetration"].requires_grad


@pytest.mark.parametrize("include_position_features", [False, True])
def test_loss_only_config_and_identical_initial_weights(include_position_features):
    overrides = [
        f"model.include_position_features={str(include_position_features).lower()}"
    ]
    with initialize_config_dir(config_dir=str(CRASH / "conf"), version_base="1.3"):
        old = compose(
            config_name="crash_deformer_contact_autoregressive", overrides=overrides
        )
        new = compose(
            config_name="crash_deformer_contact_penetration_autoregressive",
            overrides=overrides,
        )
    restored = OmegaConf.to_container(new, resolve=False)
    restored["experiment_name"] = old.experiment_name
    del restored["training"]["penetration_loss"]
    del restored["datapipe"]["penetration_loss_metadata"]
    assert restored == OmegaConf.to_container(old, resolve=False)
    torch.manual_seed(42)
    a = instantiate(old.model)
    torch.manual_seed(42)
    b = instantiate(new.model)
    assert sum(p.numel() for p in b.parameters()) == 9657399 + (
        1536 if include_position_features else 0
    )
    assert a.state_dict().keys() == b.state_dict().keys()
    for k, v in a.state_dict().items():
        torch.testing.assert_close(v, b.state_dict()[k], atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("include_position_features", [False, True])
def test_penalty_only_backward_through_actual_bptt_and_checkpoint(
    device, include_position_features
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = (
        make_model(
            include_position_features=include_position_features,
            num_time_steps=5,
            contact_graph_backend="surface",
            contact_search_implementation="torch",
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
    other = copy.deepcopy(model)
    other.checkpoint_rollout = other.checkpoint_contact = False
    sample, stats = surface_sample(device)
    sample.node_target = torch.zeros(6, 4, 3, device=device)
    sample.graph.contact_faces = torch.tensor([[0, 1, 2, 3]], device=device)
    sample.graph.shell_thickness.fill_(3.0)
    initial = (
        sample.node_features["coords"].detach() * stats["node"]["pos_std"]
        + stats["node"]["pos_mean"]
    )
    initial = initial.clone()
    initial[4:, 2] += 20
    sample.graph.penetration_reference_positions = initial
    sample2 = copy.deepcopy(sample)
    cfg = dict(tolerance_mm=0.05, length_scale_mm=1.0, implementation="torch")
    with torch.autocast(
        device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
    ):
        out = model(sample, stats)
        expected = other(sample2, stats)
        value, _ = rollout_penetration_loss(out, sample.graph, stats, cfg)
        comparison, _ = rollout_penetration_loss(expected, sample2.graph, stats, cfg)
    assert value > 0
    torch.testing.assert_close(value, comparison)
    value.backward()
    comparison.backward()
    assert sample.node_features["coords"].grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, msg=name)


def test_datapipe_initial_physical_metadata_no_future_or_window_leak(tmp_path):
    import pyvista as pv
    from datapipe import CrashGraphDataset
    from vtp_reader import Reader

    directory = tmp_path / "data"
    directory.mkdir()
    points = np.array([[100.0, 20, 30], [104, 20, 30], [104, 24, 30], [100, 24, 30]])
    mesh = pv.PolyData(points, np.array([4, 0, 1, 2, 3]))
    mesh.point_data["thickness"] = np.ones(4)
    for t in range(8):
        mesh.point_data[f"displacement_t{t * 0.005:.3f}"] = np.ones((4, 3)) * t
    mesh.save(directory / "Run0.vtp")
    dataset = CrashGraphDataset(
        reader=Reader(cache_dir=str(tmp_path / "cache"), include_contact_topology=True),
        data_dir=str(directory),
        num_samples=1,
        num_steps=8,
        initial_history_steps=2,
        sample_type="random_time_window",
        rollout_window_steps=4,
        window_seed=0,
        static_features=["thickness"],
        contact_surface=True,
        contact_require_elements=True,
        penetration_loss_metadata=True,
        stats_dir=str(tmp_path / "stats"),
    )
    reference = torch.tensor(points, dtype=torch.float32)
    torch.testing.assert_close(
        dataset.graphs[0].penetration_reference_positions, reference
    )
    assert not torch.equal(dataset.mesh_pos_seq[0][0], reference)
    for epoch in (0, 1, 25):
        dataset.set_epoch(epoch)
        sample = dataset[0]
        torch.testing.assert_close(
            sample.graph.penetration_reference_positions, reference
        )
        assert "penetration_reference_positions" not in sample.node_features
        assert (
            sample.to(
                torch.device("meta")
            ).graph.penetration_reference_positions.device.type
            == "meta"
        )
        assert dataset.graphs[0].penetration_reference_positions.device.type == "cpu"
