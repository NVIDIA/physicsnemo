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
from test_deformer_contact import make_model  # noqa: E402
from test_surface_contact_recipe import surface_sample  # noqa: E402
from test_surface_one_ring import strip  # noqa: E402

from physicsnemo.nn.functional.neighbors.reference_geodesic import (  # noqa: E402
    reference_geodesic_exclusions,
)


def test_config_changes_only_static_exclusions():
    """Verify config changes only static exclusions."""
    with initialize_config_dir(config_dir=str(CRASH_DIR / "conf"), version_base="1.3"):
        old = compose(
            config_name="crash_deformer_contact_autoregressive",
            overrides=[
                "datapipe.contact_surface_exclusion=incidence",
                "datapipe.contact_geodesic_gap_min=0.0",
            ],
        )
        new = compose(
            config_name="crash_deformer_contact_autoregressive",
            overrides=["datapipe.contact_geodesic_gap_min=0.0"],
        )
    assert old.model == new.model and old.training == new.training
    assert new.datapipe.contact_surface_exclusion == "reference_geodesic"
    assert new.datapipe.contact_geodesic_gap_min == 0.0
    previous = OmegaConf.to_container(old.datapipe, resolve=False)
    actual = OmegaConf.to_container(new.datapipe, resolve=False)
    assert {
        key: value
        for key, value in actual.items()
        if key != "contact_surface_exclusion"
    } == {
        key: value
        for key, value in previous.items()
        if key != "contact_surface_exclusion"
    }


def test_gap_floor_config_is_explicit_and_independent_of_message_band():
    """Verify gap floor config is explicit and independent of message band."""
    with initialize_config_dir(config_dir=str(CRASH_DIR / "conf"), version_base="1.3"):
        old = compose(
            config_name="crash_deformer_contact_autoregressive",
            overrides=["datapipe.contact_geodesic_gap_min=0.0"],
        )
        new = compose(config_name="crash_deformer_contact_autoregressive")
    assert new.model == old.model and new.training == old.training
    assert new.datapipe.contact_geodesic_gap_min == 5.0
    assert old.datapipe.contact_geodesic_gap_min == 0.0
    assert (
        new.model.contact_activation_distance
        == old.model.contact_activation_distance
        == 5.0
    )


def test_datapipe_uses_physical_initial_geometry_and_content_safe_cache(tmp_path):
    """Verify datapipe uses physical initial geometry and content safe cache."""
    import pyvista as pv
    from datapipe import CrashGraphDataset
    from vtp_reader import Reader

    p, f = strip()
    packed = np.column_stack((np.full(len(f), 4), f.numpy())).ravel()
    raw = tmp_path / "data"
    raw.mkdir()
    references = []
    thicknesses = []
    for i in range(4):
        pos = p.numpy() * (10 if i == 2 else 1)
        thickness = np.full(len(p), 3.0 if i == 3 else 1.0)
        references.append(torch.tensor(pos))
        thicknesses.append(torch.tensor(thickness, dtype=torch.float32))
        mesh = pv.PolyData(pos, packed)
        mesh.point_data["thickness"] = thickness
        for frame in range(3):
            mesh.point_data[f"displacement_t0.{frame * 5:03d}"] = np.full(
                (len(p), 3), frame * (200 if i == 1 else 0.1)
            )
        mesh.save(raw / f"Run{i}.vtp")
    ds = CrashGraphDataset(
        reader=Reader(cache_dir=str(tmp_path / "cache"), include_contact_topology=True),
        data_dir=str(raw),
        num_samples=4,
        num_steps=3,
        initial_history_steps=2,
        static_features=["thickness"],
        contact_surface=True,
        contact_surface_exclusion="reference_geodesic",
        contact_geodesic_cache_dir=str(tmp_path / "geodesic-cache"),
        stats_dir=str(tmp_path / "stats"),
    )
    for i, g in enumerate(ds.graphs):
        t = thicknesses[i]
        expected = reference_geodesic_exclusions(
            references[i], f, t / 2, t[f].amax(1) / 2
        )
        torch.testing.assert_close(g.contact_surface_exclusions, expected)
        torch.testing.assert_close(ds.contact_reference_positions[i], references[i])
    assert (
        ds.graphs[0].contact_surface_exclusions
        is ds.graphs[1].contact_surface_exclusions
    )
    assert (
        ds.graphs[0].contact_surface_exclusions
        is not ds.graphs[2].contact_surface_exclusions
    )
    assert (
        ds.graphs[0].contact_surface_exclusions
        is not ds.graphs[3].contact_surface_exclusions
    )
    assert (
        ds.graphs[2].contact_surface_exclusions.shape[1]
        < ds.graphs[0].contact_surface_exclusions.shape[1]
        < ds.graphs[3].contact_surface_exclusions.shape[1]
    )
    batched = Batch.from_data_list(ds.graphs[:2])
    offset = torch.tensor([[len(p)], [len(f)]])
    torch.testing.assert_close(
        batched.contact_surface_exclusions[
            :, ds.graphs[0].contact_surface_exclusions.shape[1] :
        ],
        ds.graphs[1].contact_surface_exclusions + offset,
    )
    assert (
        ds[0].to(torch.device("meta")).graph.contact_surface_exclusions.device.type
        == "meta"
    )
    reloaded = CrashGraphDataset(
        reader=Reader(cache_dir=str(tmp_path / "cache"), include_contact_topology=True),
        data_dir=str(raw),
        num_samples=4,
        num_steps=3,
        initial_history_steps=2,
        static_features=["thickness"],
        contact_surface=True,
        contact_surface_exclusion="reference_geodesic",
        contact_geodesic_cache_dir=str(tmp_path / "geodesic-cache"),
        stats_dir=str(tmp_path / "stats"),
        stats_mode="load",
    )
    assert reloaded.geodesic_cache.counts == dict(
        memory_hits=1, disk_hits=3, computed=0
    )
    for i, graph in enumerate(reloaded.graphs):
        t = thicknesses[i]
        expected = reference_geodesic_exclusions(
            references[i], f, t / 2, t[f].amax(1) / 2
        )
        torch.testing.assert_close(graph.contact_surface_exclusions, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_geodesic_filter_preserves_bptt_and_checkpoint_gradients(
    device, implementation
):
    """Verify geodesic filter preserves BPTT and checkpoint gradients."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(721)
    model = (
        make_model(
            contact_graph_backend="surface",
            contact_search_implementation=implementation,
            contact_activation_distance=5.0,
            contact_normal_epsilon=0.1,
            contact_surface_predictive=True,
            contact_surface_material_fan=True,
            checkpoint_contact=True,
        )
        .to(device)
        .train()
    )
    reference = copy.deepcopy(model)
    reference.checkpoint_contact = reference.checkpoint_rollout = False
    sample, stats = surface_sample(device)
    f = torch.tensor([[0, 1, 2, 3], [0, 1, 4, 4]])
    p = sample.node_features["coords"].detach().cpu()
    e = reference_geodesic_exclusions(p, f, torch.ones(6), torch.ones(2))
    sample.graph.contact_faces = f.to(device)
    sample.graph.contact_surface_exclusions = e.to(device)
    other = copy.deepcopy(sample)
    seen = []

    def observe(module, args, graph):
        assert graph.edge_index.shape[1] > 0 and (graph.edge_index[1] == 5).all()
        seen.append(graph)

    hook = model.node_contact_builder.register_forward_hook(observe)
    actual = model(sample, stats)
    hook.remove()
    expected = reference(other, stats)
    torch.testing.assert_close(actual, expected)
    actual[:, -1].square().sum().backward()
    expected[:, -1].square().sum().backward()
    assert len(seen) == 3
    for key in ("coords", "previous_coords"):
        grad = sample.node_features[key].grad
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
        torch.testing.assert_close(grad, other.node_features[key].grad)
    for (name, p), (_, q) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, msg=name)
