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
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.data import Batch

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))
from surface_topology import SurfaceContactData, surface_faces_from_cells  # noqa: E402
from test_deformer_contact import make_model, make_sample  # noqa: E402


def surface_sample(device):
    """Attach face connectivity to a synthetic closed-loop crash sample."""
    sample, stats = make_sample(device)
    coords = torch.tensor(
        [
            [0.0, 0, 0],
            [1, 0, 0],
            [1, 1, 0.03],
            [0, 1, 0],
            [0.3, 0.4, 0.1],
            [0.7, 0.6, -0.1],
        ],
        device=device,
    )
    sample.node_features["coords"] = coords.requires_grad_()
    sample.node_features["previous_coords"] = (coords.detach() - 0.01).requires_grad_()
    return sample, stats


def test_topology_parser_and_batch_offsets():
    """Verify topology parser and batch offsets."""
    faces = surface_faces_from_cells(np.array([3, 0, 1, 2, 4, 0, 1, 2, 3]), 4)
    torch.testing.assert_close(faces, torch.tensor([[0, 1, 2, 2], [0, 1, 2, 3]]))
    padded = surface_faces_from_cells(np.array([4, 0, 1, 2, 2]), 4)
    torch.testing.assert_close(padded, faces[:1])
    g = SurfaceContactData(
        num_nodes=4,
        contact_faces=faces,
        contact_surface_exclusions=torch.tensor([[3], [0]]),
    )
    batch = Batch.from_data_list([g, g])
    torch.testing.assert_close(batch.contact_faces[2:], faces + 4)
    torch.testing.assert_close(
        batch.contact_surface_exclusions, torch.tensor([[3, 7], [0, 2]])
    )
    for invalid in (
        [3, 0, 1],
        [5, 0, 1, 2, 3, 4],
        [3, 0, 0, 2],
        [3, 0, 1, 5],
        [3, 0, 1, 2, 3, 2, 1, 0],
        [4, 0, 0, 2, 3],
        [3, 0, 1, 2, 4, 2, 1, 0, 0],
    ):
        with pytest.raises(ValueError):
            surface_faces_from_cells(np.array(invalid), 4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("quad", [False, True])
@pytest.mark.parametrize("predictive", [False, True])
@pytest.mark.parametrize("material", [False, True])
def test_real_surface_bptt_checkpoint_equivalence(
    device, implementation, quad, predictive, material
):
    """Verify real surface BPTT checkpoint equivalence."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(721)
    model = (
        make_model(
            contact_graph_backend="surface",
            contact_search_implementation=implementation,
            contact_activation_distance=5.0,
            contact_normal_epsilon=0.1,
            contact_surface_predictive=predictive,
            contact_surface_material_fan=material,
            checkpoint_contact=True,
        )
        .to(device)
        .train()
    )
    reference = copy.deepcopy(model)
    reference.checkpoint_contact = False
    reference.checkpoint_rollout = False
    sample, stats = surface_sample(device)
    sample.graph.contact_faces = torch.tensor(
        [[0, 1, 2, 3]] if quad else [[0, 1, 2]], device=device
    )
    sample2 = copy.deepcopy(sample)
    graphs = []

    def capture(module, args, output):
        output.source_weights.retain_grad()
        output.edge_features.retain_grad()
        graphs.append(output)

    hook = model.node_contact_builder.register_forward_hook(capture)
    result = model(sample, stats)
    hook.remove()
    expected = reference(sample2, stats)
    torch.testing.assert_close(result, expected)
    result[:, -1].square().sum().backward()
    expected[:, -1].square().sum().backward()
    assert len(graphs) == 3
    for g in graphs:
        assert len(g.source_weights) > 0
        assert torch.isfinite(g.source_weights.grad).all()
        assert g.source_weights.grad.abs().sum() > 0
        assert g.edge_features.grad.abs().sum() > 0
    for key in ("coords", "previous_coords"):
        grad = sample.node_features[key].grad
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
        torch.testing.assert_close(grad, sample2.node_features[key].grad)
    for (name, p), (_, q) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, msg=name)
    legacy = make_model()
    assert {k: v.shape for k, v in model.state_dict().items()} == {
        k: v.shape for k, v in legacy.state_dict().items()
    }


def test_surface_recipe_preserves_training_budget():
    """Verify surface recipe preserves training budget."""
    with initialize_config_dir(config_dir=str(CRASH_DIR / "conf"), version_base="1.3"):
        new = compose(
            config_name="crash_deformer_contact_autoregressive",
            overrides=[
                "model.contact_surface_predictive=false",
                "model.contact_surface_material_fan=false",
            ],
        )
        old = compose(config_name="crash_deformer_autoregressive")
    old_training = OmegaConf.to_container(old.training, resolve=False)
    new_training = OmegaConf.to_container(new.training, resolve=False)
    assert all(new_training[key] == value for key, value in old_training.items())
    assert new.model.attention_type == old.model.attention_type
    assert new.model.dt == old.model.dt
    assert new.model.contact_dim == 12
    assert new.model.mesh_hidden_dim == old.model.mesh_hidden_dim
    assert new.datapipe.contact_exclusion_hops is None
    assert new.datapipe.contact_surface


def test_predictive_surface_recipe_preserves_architecture_and_budget():
    """Verify predictive surface recipe preserves architecture and budget."""
    with initialize_config_dir(config_dir=str(CRASH_DIR / "conf"), version_base="1.3"):
        new = compose(config_name="crash_deformer_contact_autoregressive")
        old = compose(
            config_name="crash_deformer_contact_autoregressive",
            overrides=[
                "model.contact_surface_predictive=false",
                "model.contact_surface_material_fan=false",
            ],
        )
    assert new.training == old.training
    assert new.datapipe == old.datapipe
    assert new.model.contact_surface_predictive
    assert new.model.contact_surface_material_fan
    for key in old.model:
        if key not in {"contact_surface_predictive", "contact_surface_material_fan"}:
            assert new.model[key] == old.model[key]
    for backend in ("legacy", "nearest_k"):
        with pytest.raises(ValueError, match="requires the surface backend"):
            make_model(contact_graph_backend=backend, contact_surface_predictive=True)


@pytest.mark.parametrize("material", [False, True])
def test_predictive_surface_does_not_read_future_ground_truth(material):
    """Verify predictive surface does not read future ground truth."""
    model = make_model(
        contact_graph_backend="surface",
        contact_surface_predictive=True,
        contact_surface_material_fan=material,
        contact_activation_distance=5.0,
        contact_normal_epsilon=0.1,
    ).eval()
    sample, stats = surface_sample("cpu")
    sample.graph.contact_faces = torch.tensor([[0, 1, 2, 3]])
    changed = copy.deepcopy(sample)
    changed.node_target = torch.full_like(changed.node_target, 1e6)
    with torch.no_grad():
        actual = model(sample, stats)
        expected = model(changed, stats)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert model.node_contact_builder.prediction_horizon == model.dt


@pytest.mark.parametrize("exclusion", ["incidence", "one_ring"])
def test_surface_datapipe_reads_connectivity_and_batches(tmp_path, exclusion):
    """Verify surface datapipe reads connectivity and batches."""
    import pyvista as pv
    from datapipe import CrashGraphDataset
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
        contact_surface=True,
        contact_surface_exclusion=exclusion,
        contact_require_elements=True,
        stats_dir=str(tmp_path / "stats"),
    )
    reader = Reader(cache_dir=str(tmp_path / "cache"), include_contact_topology=True)
    for _ in range(2):
        dataset = CrashGraphDataset(reader=reader, **options)
        assert dataset.graphs[0].contact_faces is dataset.graphs[1].contact_faces
        if exclusion == "one_ring":
            assert (
                dataset.graphs[0].contact_surface_exclusions
                is dataset.graphs[1].contact_surface_exclusions
            )
            torch.testing.assert_close(
                dataset.graphs[0].contact_surface_exclusions,
                torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]]),
            )
        batched = Batch.from_data_list(dataset.graphs)
        torch.testing.assert_close(
            batched.contact_faces, torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        )
        assert (
            dataset[0].to(torch.device("meta")).graph.contact_faces.device.type
            == "meta"
        )
        assert dataset.graphs[0].contact_faces.device.type == "cpu"
    with pytest.raises(ValueError, match="element connectivity"):
        CrashGraphDataset(reader=Reader(), **options)


def test_surface_bfloat16_autocast_backward():
    """Verify surface bfloat16 autocast backward."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = (
        make_model(
            contact_graph_backend="surface",
            contact_search_implementation="warp",
            contact_activation_distance=5.0,
            contact_normal_epsilon=0.1,
            checkpoint_contact=True,
        )
        .cuda()
        .train()
    )
    sample, stats = surface_sample("cuda")
    sample.graph.contact_faces = torch.tensor([[0, 1, 2, 3]], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(sample, stats)
        loss = output[:, -1].square().sum()
    loss.backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(sample.node_features["coords"].grad).all()
    assert sample.node_features["coords"].grad.abs().sum() > 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_surface_thickness_never_uses_implicit_fallback():
    """Verify surface thickness never uses implicit fallback."""
    model = make_model(
        contact_graph_backend="surface",
        base_shell_thickness=7.0,
        contact_activation_distance=5.0,
        contact_normal_epsilon=0.1,
    )
    sample, stats = surface_sample("cpu")
    sample.graph.shell_thickness[0] = 0.0
    actual = model._physical_shell_thickness(sample, stats, 6)
    torch.testing.assert_close(actual, sample.graph.shell_thickness)
    legacy = make_model(base_shell_thickness=7.0)
    assert legacy._physical_shell_thickness(sample, stats, 6)[0] == 7.0
    del sample.graph.shell_thickness
    with pytest.raises(ValueError, match="no fallback"):
        model._physical_shell_thickness(sample, stats, 6)


def test_surface_datapipe_requires_thickness_even_without_static_feature(tmp_path):
    """Verify surface datapipe requires thickness even without static feature."""
    import pyvista as pv
    from datapipe import CrashGraphDataset
    from vtp_reader import Reader

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    points = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [0.0, 1, 0]])
    mesh = pv.PolyData(points, np.array([4, 0, 1, 2, 3]))
    for t in range(3):
        mesh.point_data[f"displacement_t0.{t * 5:03d}"] = np.ones((4, 3)) * t
    mesh.save(data_dir / "Run0.vtp")
    with pytest.raises(ValueError, match="supplied physical thickness"):
        CrashGraphDataset(
            reader=Reader(
                cache_dir=str(tmp_path / "cache"), include_contact_topology=True
            ),
            data_dir=str(data_dir),
            num_samples=1,
            num_steps=3,
            initial_history_steps=2,
            static_features=[],
            contact_surface=True,
            contact_require_elements=True,
            stats_dir=str(tmp_path / "stats"),
        )
