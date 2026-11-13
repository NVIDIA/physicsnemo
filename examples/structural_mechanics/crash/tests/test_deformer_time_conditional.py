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

"""CPU smoke tests of the time-conditional DeFormer wrapper on a toy plate mesh."""

import os
import sys
from types import SimpleNamespace

import torch

from physicsnemo.experimental.nn.geodesic_knn import geodesic_knn

THIS_DIR = os.path.dirname(__file__)
CRASH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CRASH_DIR not in sys.path:
    sys.path.insert(0, CRASH_DIR)

from datapipes.datapipe import SimSample  # noqa: E402
from models.rollout import DeFormerTimeConditional  # noqa: E402

K = 4
TIME_DIM = 8
NUM_TIME_STEPS = 4  # three predicted frames


def _plate_graph(nx=4, ny=4):
    """A quad plate with its static geodesic neighbourhoods attached."""
    xs, ys = torch.meshgrid(
        torch.arange(nx, dtype=torch.float32) * 10.0,
        torch.arange(ny, dtype=torch.float32) * 10.0,
        indexing="ij",
    )
    pos = torch.stack((xs.reshape(-1), ys.reshape(-1), torch.zeros(nx * ny)), -1)
    undirected = set()
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            quad = (a, a + ny, a + ny + 1, a + 1)
            for u, v in zip(quad, quad[1:] + quad[:1]):
                undirected.add((min(u, v), max(u, v)))
    und = torch.tensor(sorted(undirected), dtype=torch.long).t()
    edge_index = torch.cat((und, und.flip(0)), dim=1)
    knn = geodesic_knn(pos, edge_index, K)
    own = torch.arange(pos.shape[0])[:, None]
    index = torch.cat((own, knn.index.to(torch.long)), dim=1)
    mask = torch.cat((torch.ones_like(own, dtype=torch.bool), knn.mask), dim=1)
    displacement = pos[index] - pos[:, None]
    edge_attr = torch.cat((displacement, displacement.norm(dim=-1, keepdim=True)), -1)
    graph = SimpleNamespace(
        num_nodes=pos.shape[0],
        knn_index=index,
        knn_mask=mask,
        knn_edge_attr=edge_attr,
        knn_metric="geodesic",
    )
    return pos, graph


def _sample(time=0.0):
    pos, graph = _plate_graph()
    torch.manual_seed(0)
    features = torch.rand(pos.shape[0], 1)  # the thickness channel
    node_features = {
        "coords": pos / 100.0,  # the datapipe serves normalized coordinates
        "features": features,
        "time": torch.tensor(float(time)),
    }
    target = torch.zeros(pos.shape[0], NUM_TIME_STEPS - 1, 3)
    return SimSample(node_features=node_features, node_target=target, graph=graph)


def _data_stats():
    return {"node": {"pos_mean": torch.zeros(1, 3), "pos_std": torch.ones(1, 3)}}


def _model(**overrides):
    torch.manual_seed(1)
    kwargs = dict(
        num_time_steps=NUM_TIME_STEPS,
        time_embedding="fourier",
        time_fourier_bands=2,
        time_dim=TIME_DIM,
        functional_dim=3 + 1 + TIME_DIM,
        out_dim=3,
        geometry_dim=3,
        global_dim=TIME_DIM,  # the time code is the global token
        n_hidden=32,
        n_head=4,
        n_layers=1,
        slice_num=8,
        attention_type="GALE_FA",
        use_te=False,
        num_pre_processor_layers=1,
        num_post_processor_layers=1,
        local_knn_k=K,
        local_neighborhood="geodesic",
        local_edge_dim=8,
        mesh_context_fusion="post",
    )
    kwargs.update(overrides)
    return DeFormerTimeConditional(**kwargs)


def _unzero_heads(model):
    """Give the zero-initialised heads small weights so outputs move."""
    with torch.no_grad():
        model.coarse_decoder[-1].weight.normal_(std=0.01)
        last = [
            m for m in model.final_decoder.modules() if isinstance(m, torch.nn.Linear)
        ][-1]
        last.weight.normal_(std=0.01)


def test_network_starts_exactly_at_the_undeformed_mesh():
    """Zero-initialised heads: every frame equals the initial coordinates."""
    model = _model()
    sample, stats = _sample(time=0.5), _data_stats()
    coords = sample.node_features["coords"]
    model.train()
    frame = model(sample, stats)
    assert frame.shape == coords.shape
    assert torch.equal(frame, coords)
    output = model(sample, stats, return_auxiliary=True)
    assert torch.equal(output.prediction, coords)
    assert torch.equal(output.auxiliary_prediction, coords)
    model.eval()
    with torch.no_grad():
        trajectory = model(_sample(), stats)
    assert trajectory.shape == (coords.shape[0], NUM_TIME_STEPS - 1, 3)
    assert torch.equal(trajectory, coords[:, None].expand_as(trajectory))


def test_training_frame_equals_the_frame_of_the_evaluation_trajectory():
    """A train-mode query at time k / steps reproduces frame k of the rollout."""
    model = _model()
    _unzero_heads(model)
    stats = _data_stats()
    model.eval()
    with torch.no_grad():
        trajectory = model(_sample(), stats)
    assert not torch.equal(trajectory[:, 0], trajectory[:, 1])  # heads moved
    model.train()
    steps = NUM_TIME_STEPS - 1
    with torch.no_grad():
        for k in range(steps):
            frame = model(_sample(time=k / steps), stats)
            torch.testing.assert_close(frame, trajectory[:, k], atol=0, rtol=0)


def test_every_trainable_parameter_receives_a_gradient():
    """One backward populates the gradient of every trainable parameter.

    The base model is built without its output head, so every parameter of
    the model is trainable and takes part in the step.
    """
    model = _model()
    _unzero_heads(model)
    model.train()
    output = model(_sample(time=1 / 3), _data_stats(), return_auxiliary=True)
    (
        output.prediction.square().sum() + output.auxiliary_prediction.square().sum()
    ).backward()
    assert all(p.requires_grad for _, p in model.named_parameters())
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing, missing
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
