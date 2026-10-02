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

"""One-ring exclusion ablation: topology, discovery, batching and live BPTT."""

import copy
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch_geometric.data import Batch

CRASH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRASH_DIR))
from surface_topology import (  # noqa: E402
    SurfaceContactData,
    surface_one_ring_exclusions,
)
from test_deformer_contact import make_model  # noqa: E402
from test_surface_contact_recipe import surface_sample  # noqa: E402

from physicsnemo.nn.functional.neighbors.surface_contact import (  # noqa: E402
    node_triangle_candidate_chunks,
)


def strip():
    # Five quads in one connected strip, and a separate triangular surface.
    """Build a small material strip for one-ring exclusion tests."""
    faces = torch.tensor(
        [[2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1] for i in range(5)]
        + [[12, 13, 14, 14]]
    )
    positions = torch.tensor(
        [[float(i), float(j), 0.0] for i in range(6) for j in range(2)]
        + [[0.0, 0.0, 0.1], [1.0, 0.0, 0.1], [0.0, 1.0, 0.1]]
    )
    return positions, faces


def oracle(faces, num_nodes):
    """Compute one-ring node-to-face exclusions with a direct test oracle."""
    cells = [set(f) for f in faces.tolist()]
    return {
        (n, f)
        for n in range(num_nodes)
        for f, cell in enumerate(cells)
        if any(n in c and bool(c & cell) for c in cells)
    }


def test_exact_one_ring_not_whole_component():
    """Verify exact one ring not whole component."""
    positions, faces = strip()
    pairs = surface_one_ring_exclusions(faces, len(positions))
    actual = set(map(tuple, pairs.T.tolist()))
    assert actual == oracle(faces, len(positions))
    assert (0, 0) in actual and (0, 1) in actual
    assert (0, 2) not in actual  # Same component, but outside the one-ring.
    assert (0, 5) not in actual  # Separate surface at almost zero distance.
    assert list(map(tuple, pairs.T.tolist())) == sorted(actual)
    triangle = torch.tensor([[0, 1, 2], [2, 3, 4]])
    padded = torch.cat((triangle, triangle[:, -1:]), 1)
    torch.testing.assert_close(
        surface_one_ring_exclusions(triangle, 6), surface_one_ring_exclusions(padded, 6)
    )
    assert surface_one_ring_exclusions(
        torch.empty((0, 4), dtype=torch.long), 0
    ).shape == (2, 0)


@pytest.mark.parametrize(
    "faces,n",
    [
        (torch.zeros((2, 4)), 4),
        (torch.zeros((2, 2), dtype=torch.long), 4),
        (torch.tensor([[0, 1, 3]]), 3),
        (torch.tensor([[0, -1, 2]]), 3),
        (torch.tensor([[0, 1, 2]]), True),
    ],
)
def test_reject_invalid_topology(faces, n):
    """Verify reject invalid topology."""
    with pytest.raises(ValueError):
        surface_one_ring_exclusions(faces, n)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_filtered_discovery_matches_oracle_and_batch_isolation(device, implementation):
    """Verify filtered discovery matches oracle and batch isolation."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    positions, faces = strip()
    exclusions = surface_one_ring_exclusions(faces, len(positions))
    graph = SurfaceContactData(
        num_nodes=len(positions),
        contact_faces=faces,
        contact_surface_exclusions=exclusions,
    )
    batched = Batch.from_data_list([graph, graph]).to(device)
    pos = torch.cat((positions, positions), 0).to(device)
    # Deliberately shuffled duplicate exclusions test the public functional.
    excl = torch.cat(
        (
            batched.contact_surface_exclusions.flip(1),
            batched.contact_surface_exclusions[:, :3],
        ),
        1,
    )
    options = dict(
        batch=batched.batch,
        implementation=implementation,
        pair_chunk_size=7,
        chunk_size=2,
    )

    def collect(excluded):
        chunks = list(
            node_triangle_candidate_chunks(
                pos,
                batched.contact_faces,
                pos.new_full((len(pos),), 20),
                pos.new_zeros(len(batched.contact_faces)),
                excluded_pairs=excluded,
                **options,
            )
        )
        assert all(x.shape[1] <= 7 for x in chunks)
        return (
            set(map(tuple, torch.cat(chunks, 1).T.cpu().tolist())) if chunks else set()
        )

    raw = collect(None)
    filtered = collect(excl)
    expected = raw - set(map(tuple, excl.T.cpu().tolist()))
    assert filtered == expected and len(filtered) < len(raw)
    assert (0, 2) in filtered and (0, 5) in filtered
    assert (0, 6) not in filtered  # Overlapping but separate batch example.
    assert collect(batched.contact_surface_exclusions[:, :0]) == raw
    all_pairs = torch.tensor(sorted(raw), device=device).T.contiguous()
    assert not collect(all_pairs)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_one_ring_bptt_checkpoint_equivalence(device, implementation):
    """Verify one ring BPTT checkpoint equivalence."""
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
    faces = torch.tensor([[0, 1, 2, 3], [0, 1, 4, 4]])
    sample.graph.contact_faces = faces.to(device)
    sample.graph.contact_surface_exclusions = surface_one_ring_exclusions(faces, 6).to(
        device
    )
    other = copy.deepcopy(sample)
    seen = []

    def capture(module, args, output):
        seen.append(output)
        assert output.edge_index.shape[1] > 0
        # Node 4 is a local material neighbor; only isolated node 5 may query.
        assert (output.edge_index[1] == 5).all()

    hook = model.node_contact_builder.register_forward_hook(capture)
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


def test_ablation_changes_only_exclusions():
    """Verify ablation changes only exclusions."""
    with initialize_config_dir(config_dir=str(CRASH_DIR / "conf"), version_base="1.3"):
        new = compose(
            config_name="crash_deformer_contact_autoregressive",
            overrides=["datapipe.contact_surface_exclusion=one_ring"],
        )
        old = compose(
            config_name="crash_deformer_contact_autoregressive",
            overrides=["datapipe.contact_surface_exclusion=incidence"],
        )
    assert new.model == old.model and new.training == old.training
    assert new.datapipe.contact_surface_exclusion == "one_ring"
    del new.datapipe.contact_surface_exclusion
    del old.datapipe.contact_surface_exclusion
    assert new.datapipe == old.datapipe
