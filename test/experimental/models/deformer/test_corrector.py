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
"""DeFormer's contact-aware corrector on a small shell mesh (CPU)."""

import pytest
import torch

pyg = pytest.importorskip("torch_geometric")

import physicsnemo  # noqa: E402
from physicsnemo.experimental.models.deformer import (  # noqa: E402
    DeFormerCorrector,
    mesh_edge_features,
)

EPS = 1.0e-8


def _plate(nx, ny, z, offset):
    """A quad plate at height ``z``: positions ``[nx*ny, 3]`` and faces ``[F, 4]``."""
    xs, ys = torch.meshgrid(
        torch.arange(nx, dtype=torch.float32) * 10.0,
        torch.arange(ny, dtype=torch.float32) * 10.0,
        indexing="ij",
    )
    pos = torch.stack((xs.reshape(-1), ys.reshape(-1), torch.full((nx * ny,), z)), -1)
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            faces.append([a, a + ny, a + ny + 1, a + 1])
    return pos, torch.tensor(faces, dtype=torch.long) + offset


def _mesh(two_plates: bool = True, gap: float = 20.0):
    """Reference geometry, directed mesh edges with standardized features, facets."""
    pos_a, faces_a = _plate(4, 4, 0.0, 0)
    if two_plates:
        pos_b, faces_b = _plate(4, 4, gap, pos_a.shape[0])
        pos, faces = torch.cat((pos_a, pos_b)), torch.cat((faces_a, faces_b))
    else:
        pos, faces = pos_a, faces_a
    undirected = set()
    for quad in faces.tolist():
        for a, b in zip(quad, quad[1:] + quad[:1]):
            undirected.add((min(a, b), max(a, b)))
    und = torch.tensor(sorted(undirected), dtype=torch.long).t()
    edge_index = torch.cat((und, und.flip(0)), dim=1)
    attr = mesh_edge_features(pos, edge_index)
    mean, std = attr.mean(0), attr.std(0, unbiased=False) + EPS
    graph = pyg.data.Data(edge_index=edge_index, num_nodes=pos.shape[0])
    graph.mesh_edge_index = edge_index
    graph.mesh_edge_attr = (attr - mean) / (std + EPS)
    graph.mesh_edge_mean = mean
    graph.mesh_edge_std = std
    graph.contact_faces = faces
    # incidence exclusions: a node is never paired with its own facets
    node = faces.reshape(-1)
    face = torch.arange(faces.shape[0]).repeat_interleave(4)
    graph.contact_surface_exclusions = torch.stack((node, face))
    graph.shell_thickness = torch.full((pos.shape[0],), 1.0)
    return pos, graph


def _inputs(pos, displacement, time_dim=2):
    g = torch.Generator().manual_seed(0)
    static = torch.rand(pos.shape[0], 1, generator=g)
    code = torch.rand(1, time_dim, generator=g).expand(pos.shape[0], -1)
    node_features = torch.cat((pos, static, code, displacement), dim=-1)
    return node_features, torch.zeros(3), torch.ones(3)


def _model(use_contact, **overrides):
    torch.manual_seed(1)
    kwargs = dict(
        functional_dim=3 + 1 + 2 + 3,
        hidden_dim=16,
        processor_size=2,
        use_contact=use_contact,
        contact_search_implementation="torch",
    )
    kwargs.update(overrides)
    return DeFormerCorrector(**kwargs)


def test_corrector_starts_at_the_teacher_and_trains():
    pos, graph = _mesh(two_plates=False)
    displacement = 0.1 * torch.randn(
        pos.shape[0], 3, generator=torch.Generator().manual_seed(2)
    )
    node_features, mean, std = _inputs(pos, displacement)
    model = _model(use_contact=False)
    out = model(node_features, pos, displacement, graph)
    assert out.shape == pos.shape
    # zero-initialised decoder: the correction is exactly zero at construction
    assert torch.equal(out, pos + displacement)
    (out - pos).square().sum().backward()
    assert model.decoder[-1].weight.grad is not None
    assert torch.isfinite(model.decoder[-1].weight.grad).all()
    # the encoders and the processor receive no gradient through a zero decoder,
    # so a second step must change the correction once the decoder has moved
    with torch.no_grad():
        model.decoder[-1].weight.add_(1e-2)
    moved = model(node_features, pos, displacement, graph)
    assert not torch.equal(moved, pos + displacement)


def test_corrector_discovers_contact_edges_on_the_predicted_shape():
    pos, graph = _mesh(two_plates=True, gap=20.0)
    n = pos.shape[0]
    # the upper plate comes down to 2 units above the lower one: in the 5-unit
    # band, outside the 10-unit reference clearance (the reference gap is 20)
    displacement = torch.zeros(n, 3)
    displacement[n // 2 :, 2] = -18.0
    node_features, mean, std = _inputs(pos, displacement)
    model = _model(use_contact=True)
    index, latent = model.contact_edges(pos + displacement, pos, graph, mean, std)
    assert index is not None and index.shape[0] == 2 and index.shape[1] > 0
    assert latent.shape == (index.shape[1], model.hidden_dim)
    assert index.max() < n
    out = model(node_features, pos, displacement, graph, mean, std)
    assert out.shape == pos.shape
    assert torch.equal(out, pos + displacement)  # zero decoder


def test_corrector_without_active_contact_keeps_contact_parameters_in_the_graph():
    pos, graph = _mesh(two_plates=True, gap=20.0)
    displacement = torch.zeros(pos.shape[0], 3)  # plates stay 20 units apart
    node_features, mean, std = _inputs(pos, displacement)
    model = _model(use_contact=True)
    index, placeholder = model.contact_edges(pos + displacement, pos, graph, mean, std)
    assert index is None and placeholder.ndim == 0 and float(placeholder) == 0.0
    out = model(node_features, pos, displacement, graph, mean, std)
    out.square().sum().backward()
    for name, p in model.contact_edge_encoder.named_parameters():
        assert p.grad is not None, name  # the placeholder keeps the layout per rank


def test_corrector_validates_inputs():
    pos, graph = _mesh(two_plates=False)
    displacement = torch.zeros(pos.shape[0], 3)
    node_features, mean, std = _inputs(pos, displacement)
    model = _model(use_contact=False)
    with pytest.raises(ValueError, match="node_features"):
        model(node_features[:, :-1], pos, displacement, graph)
    with pytest.raises(ValueError, match="teacher_displacement"):
        model(node_features, pos, displacement[:-1], graph)
    bare = pyg.data.Data(edge_index=graph.edge_index, num_nodes=pos.shape[0])
    with pytest.raises(ValueError, match="mesh_edge_index"):
        model(node_features, pos, displacement, bare)
    contact = _model(use_contact=True)
    with pytest.raises(ValueError, match="position_mean"):
        contact(node_features, pos, displacement, graph)
    with pytest.raises(ValueError, match="aggregation"):
        _model(use_contact=False, aggregation="max")
    with pytest.raises(ValueError, match="divisible"):
        _model(use_contact=False, processor_size=3, num_processor_checkpoint_segments=2)


def test_corrector_checkpoint_round_trip(tmp_path):
    pos, graph = _mesh(two_plates=False)
    displacement = 0.1 * torch.randn(
        pos.shape[0], 3, generator=torch.Generator().manual_seed(3)
    )
    node_features, mean, std = _inputs(pos, displacement)
    model = _model(use_contact=False)
    with torch.no_grad():
        model.decoder[-1].weight.add_(1e-2)
    path = tmp_path / "corrector.mdlus"
    model.save(str(path))
    reloaded = physicsnemo.Module.from_checkpoint(str(path))
    assert type(reloaded).__name__ == "DeFormerCorrector"
    with torch.no_grad():
        assert torch.equal(
            model(node_features, pos, displacement, graph),
            reloaded(node_features, pos, displacement, graph),
        )
