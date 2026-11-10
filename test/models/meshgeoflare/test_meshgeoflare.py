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

import pytest
import torch

pyg = pytest.importorskip("torch_geometric")

from physicsnemo.experimental.models.meshgeoflare import MeshGeoFLARE  # noqa: E402
from physicsnemo.experimental.nn.static_knn import geodesic_knn  # noqa: E402
from physicsnemo.models.geotransolver import GeoTransolver  # noqa: E402
from physicsnemo.models.geotransolver.flare_plus_plus import (  # noqa: E402
    _FLAREPlusPlusAttention,
)

K = 4


class _KnnData(pyg.data.Data):
    """PyG data whose per-node neighbourhood tensors batch like node tensors."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "knn_index":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("knn_index", "knn_mask", "knn_edge_attr"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def _graph(num_nodes: int = 9, node_dim: int = 6):
    """A chain mesh with its static geodesic neighbourhoods attached."""
    pos = torch.cumsum(torch.rand(num_nodes, 3) + 0.1, dim=0)
    src = torch.arange(num_nodes - 1)
    edge_index = torch.stack((torch.cat((src, src + 1)), torch.cat((src + 1, src))))
    knn = geodesic_knn(pos, edge_index, K)
    own = torch.arange(num_nodes)[:, None]
    index = torch.cat((own, knn.index), dim=1)
    mask = torch.cat((torch.ones_like(own, dtype=torch.bool), knn.mask), dim=1)
    displacement = pos[index] - pos[:, None]
    edge_attr = torch.cat((displacement, displacement.norm(dim=-1, keepdim=True)), -1)
    graph = _KnnData(
        x=torch.randn(num_nodes, node_dim), edge_index=edge_index, num_nodes=num_nodes
    )
    graph.knn_index, graph.knn_mask, graph.knn_edge_attr = index, mask, edge_attr
    graph.knn_metric = "geodesic"
    return graph


def _model(**overrides):
    kwargs = dict(
        functional_dim=6,
        out_dim=5,
        input_dim_edges=4,
        geometry_dim=3,
        global_dim=2,
        n_hidden=32,
        n_head=4,
        n_layers=2,
        slice_num=8,
        num_pre_processor_layers=1,
        num_post_processor_layers=1,
        local_knn_k=K,
        local_neighborhood="geodesic",
        local_edge_dim=8,
        use_te=False,
    )
    kwargs.update(overrides)
    return MeshGeoFLARE(**kwargs)


def _forward(model, graph, global_embedding=None):
    return model(
        node_features=graph.x,
        graph=graph,
        geometry=graph.x[:, :3],
        global_embedding=global_embedding,
    )


def test_forward_backward():
    torch.manual_seed(7)
    graph = _graph(9)
    model = _model()
    output = _forward(model, graph, torch.randn(1, 1, 2))
    output.square().mean().backward()
    assert output.shape == (9, 5)
    assert torch.isfinite(output).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_variable_graph_batch_matches_independent_forwards():
    torch.manual_seed(11)
    graphs = [_graph(5), _graph(8)]
    batch = pyg.data.Batch.from_data_list(graphs)
    model = _model().eval()
    global_embedding = torch.randn(2, 1, 2)
    with torch.no_grad():
        rng_state = torch.random.get_rng_state()
        batch_output = _forward(model, batch, global_embedding)
        # PhysicsAttention++ uses Gumbel sampling in eval mode. Restore the RNG
        # so the batched and independent calls receive identical samples.
        torch.random.set_rng_state(rng_state)
        separate_outputs = [
            _forward(model, graph, global_embedding[index])
            for index, graph in enumerate(graphs)
        ]
    torch.testing.assert_close(batch_output, torch.cat(separate_outputs, dim=0))


def test_geometry_context_is_required():
    graph = _graph(6)
    with pytest.raises(ValueError, match="geometry is required"):
        _model()(
            node_features=graph.x, graph=graph, global_embedding=torch.randn(1, 1, 2)
        )


def test_global_context_is_required_when_configured():
    graph = _graph(6)
    with pytest.raises(ValueError, match="global_embedding is required"):
        _model()(node_features=graph.x, graph=graph, geometry=graph.x[:, :3])


@pytest.mark.parametrize("attention_type", ["GALE_FA", "GALE_FPP"])
def test_retains_full_geoflare_backbone_and_adds_parameters(attention_type):
    """MeshGeoFLARE keeps every GeoTransolver parameter and its attention backend."""
    common = dict(
        functional_dim=6,
        out_dim=5,
        geometry_dim=3,
        global_dim=2,
        n_hidden=32,
        n_head=4,
        n_layers=2,
        slice_num=8,
        use_te=False,
        attention_type=attention_type,
    )
    backbone = GeoTransolver(**common)
    model = _model(**common)
    state = model.state_dict()
    for name, parameter in backbone.state_dict().items():
        assert name in state
        assert state[name].shape == parameter.shape
    assert sum(p.numel() for p in model.parameters()) > sum(
        p.numel() for p in backbone.parameters()
    )
    assert isinstance(model, GeoTransolver)
    for mesh_block, base_block in zip(model.blocks, backbone.blocks, strict=True):
        assert type(mesh_block.Attn) is type(base_block.Attn)
    if attention_type == "GALE_FPP":
        assert all(
            isinstance(block.Attn, _FLAREPlusPlusAttention) for block in model.blocks
        )


@pytest.mark.parametrize(
    "components", [("blocks",), ("context", "preprocess", "blocks", "output")]
)
def test_upstream_checkpointing_preserves_outputs_and_gradients(components):
    """The hidden-state seam honors upstream component checkpointing."""
    common = dict(
        attention_type="GALE_FA",
        mesh_context_fusion="pre_post",
        mesh_context_use_global=True,
    )
    plain = _model(**common).train()
    checkpointed = _model(
        **common,
        activation_checkpointing=True,
        activation_checkpointing_components=components,
    ).train()
    checkpointed.load_state_dict(plain.state_dict(), strict=True)
    graph = _graph(7)
    global_embedding = torch.randn(1, 1, 2)
    expected = _forward(plain, graph, global_embedding)
    actual = _forward(checkpointed, graph, global_embedding)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for (name, parameter), (other_name, other) in zip(
        plain.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(parameter.grad, other.grad)


def test_zero_gated_mesh_residuals_exactly_match_geoflare_backbone():
    torch.manual_seed(23)
    graph = _graph(9, node_dim=3)
    global_embedding = torch.randn(1, 1, 2)
    common = dict(
        functional_dim=3,
        out_dim=5,
        geometry_dim=3,
        global_dim=2,
        n_hidden=32,
        n_head=4,
        n_layers=2,
        slice_num=8,
        use_te=False,
        include_local_features=True,
        attention_type="GALE_FA",
    )
    geoflare = GeoTransolver(**common).eval()
    meshgeoflare = _model(
        **common,
        mesh_hidden_dim=32,
        mesh_pre_residual_gate_init=0.0,
        mesh_post_residual_gate_init=0.0,
    ).eval()
    incompatible = meshgeoflare.load_state_dict(geoflare.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert "mesh_pre_residual_gate" in incompatible.missing_keys
    assert "mesh_post_residual_gate" in incompatible.missing_keys
    with torch.no_grad():
        expected = geoflare(
            local_embedding=graph.x.unsqueeze(0),
            geometry=graph.x.unsqueeze(0),
            local_positions=graph.x.unsqueeze(0),
            global_embedding=global_embedding,
        ).squeeze(0)
        actual = meshgeoflare(
            node_features=graph.x,
            graph=graph,
            geometry=graph.x,
            local_positions=graph.x,
            global_embedding=global_embedding,
        )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_pre_processor_consumes_encoded_geoflare_latent():
    torch.manual_seed(27)
    graph = _graph(9, node_dim=6)
    model = _model(mesh_context_fusion="pre_post", mesh_context_use_global=True).eval()
    observed = {}

    def capture_processor_input(_module, inputs):
        observed["shape"] = tuple(inputs[0].shape)

    handle = model.pre_processor.register_forward_pre_hook(capture_processor_input)
    try:
        with torch.no_grad():
            _forward(model, graph, torch.randn(1, 1, 2))
    finally:
        handle.remove()
    # Raw functional features have width 6; the local stage must instead receive
    # GeoFLARE's 32-channel encoded node representation.
    assert observed["shape"] == (9, 32)
    assert not hasattr(model, "pre_node_decoder")


def test_latent_adapters_support_distinct_mesh_width():
    torch.manual_seed(28)
    graph = _graph(9, node_dim=6)
    model = _model(mesh_hidden_dim=24).eval()
    observed = {}

    def capture_processor_input(_module, inputs):
        observed["shape"] = tuple(inputs[0].shape)

    handle = model.pre_processor.register_forward_pre_hook(capture_processor_input)
    try:
        with torch.no_grad():
            output = _forward(model, graph, torch.randn(1, 1, 2))
    finally:
        handle.remove()
    assert observed["shape"] == (9, 24)
    assert output.shape == (9, 5)
    assert model.pre_latent_output.bias is None


@pytest.mark.parametrize(
    ("fusion_stage", "use_global"),
    [("post", False), ("pre_post", False), ("pre_post", True)],
)
def test_context_fusion_is_zero_initialized_identity(fusion_stage, use_global):
    torch.manual_seed(29)
    graph = _graph(9)
    global_embedding = torch.randn(1, 1, 2)
    torch.manual_seed(31)
    baseline = _model().eval()
    torch.manual_seed(31)
    fused = _model(
        mesh_context_fusion=fusion_stage, mesh_context_use_global=use_global
    ).eval()
    with torch.no_grad():
        rng_state = torch.random.get_rng_state()
        expected = _forward(baseline, graph, global_embedding)
        torch.random.set_rng_state(rng_state)
        actual = _forward(fused, graph, global_embedding)
    torch.testing.assert_close(actual, expected)
    fused.train()
    torch.random.set_rng_state(rng_state)
    loss = _forward(fused, graph, global_embedding).square().mean()
    loss.backward()
    assert fused.post_context_film.affine.weight.grad is not None
    assert torch.count_nonzero(fused.post_context_film.affine.weight.grad) > 0
    if fusion_stage == "pre_post":
        assert fused.pre_context_film.affine.weight.grad is not None
        assert torch.count_nonzero(fused.pre_context_film.affine.weight.grad) > 0


@pytest.mark.parametrize(
    ("fusion_stage", "use_global"),
    [("post", False), ("pre_post", False), ("pre_post", True)],
)
def test_context_fusion_variable_batch_matches_independent(fusion_stage, use_global):
    torch.manual_seed(37)
    graphs = [_graph(5), _graph(8)]
    batch = pyg.data.Batch.from_data_list(graphs)
    model = _model(
        mesh_context_fusion=fusion_stage, mesh_context_use_global=use_global
    ).eval()
    global_embedding = torch.randn(2, 1, 2)
    with torch.no_grad():
        rng_state = torch.random.get_rng_state()
        batch_output = _forward(model, batch, global_embedding)
        torch.random.set_rng_state(rng_state)
        separate_outputs = [
            _forward(model, graph, global_embedding[index])
            for index, graph in enumerate(graphs)
        ]
    torch.testing.assert_close(batch_output, torch.cat(separate_outputs, dim=0))


def test_context_fusion_parameter_ordering():
    models = [
        _model(),
        _model(mesh_context_fusion="post"),
        _model(mesh_context_fusion="pre_post"),
        _model(mesh_context_fusion="pre_post", mesh_context_use_global=True),
    ]
    parameter_counts = [sum(p.numel() for p in model.parameters()) for model in models]
    assert parameter_counts == sorted(parameter_counts)
    assert len(set(parameter_counts)) == len(parameter_counts)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mesh_context_fusion": "invalid"}, "mesh_context_fusion"),
        ({"mesh_context_use_global": True}, "requires mesh context fusion"),
    ],
)
def test_rejects_invalid_context_fusion(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _model(n_layers=1, **kwargs)


def test_checkpoint_roundtrip(tmp_path):
    model = _model()
    checkpoint = tmp_path / "MeshGeoFLARE.mdlus"
    model.save(checkpoint)
    restored = MeshGeoFLARE.from_checkpoint(checkpoint)
    assert type(restored) is MeshGeoFLARE
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bfloat16_amp_forward_backward():
    graph = _graph(9).cuda()
    model = _model().cuda()
    global_embedding = torch.randn(1, 1, 2, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = _forward(model, graph, global_embedding)
        loss = output.square().mean()
    loss.backward()
    assert output.shape == (9, 5)
    assert torch.isfinite(output).all()
