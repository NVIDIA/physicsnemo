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

from physicsnemo.experimental.models.deformer import (
    DeFormer,
    DeFormerMetaData,
    LocalNeighborhoodData,
)
from physicsnemo.experimental.models.deformer.deformer import BackboneState
from physicsnemo.experimental.nn.geodesic_knn import geodesic_knn
from physicsnemo.models.geotransolver import GeoTransolver
from physicsnemo.models.geotransolver.flare_plus_plus import (
    _FLAREPlusPlusAttention,
)

try:
    import torch_geometric as pyg
except ImportError:  # DeFormer itself only reads attributes off the graph
    pyg = None
requires_pyg = pytest.mark.skipif(pyg is None, reason="requires torch_geometric")

K = 4


class _PlainGraph:
    """Attribute bag standing in for a PyG ``Data`` when PyG is not installed."""

    def __init__(self, **tensors):
        self.__dict__.update(tensors)

    def to(self, device):
        moved = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in self.__dict__.items()
        }
        return _PlainGraph(**moved)

    def cuda(self):
        return self.to("cuda")


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
    data_class = _PlainGraph if pyg is None else LocalNeighborhoodData
    graph = data_class(
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
    return DeFormer(**kwargs)


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


@requires_pyg
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
    """DeFormer keeps every GeoTransolver parameter and its attention backend."""
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
    model = _model(
        **common,
        mesh_hidden_dim=32,
        mesh_pre_residual_gate_init=0.0,
        mesh_post_residual_gate_init=0.0,
    ).eval()
    incompatible = model.load_state_dict(geoflare.state_dict(), strict=False)
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
        actual = model(
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
@requires_pyg
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


def test_forward_with_geometry_features_is_refused():
    """The inherited GeoTransolver entry point would skip the local stages."""
    model = _model()
    with pytest.raises(NotImplementedError, match="forward_with_geometry_features"):
        model.forward_with_geometry_features(
            torch.randn(1, 6, 6), geometry=torch.randn(1, 6, 3)
        )


def test_constructor_rejects_unusable_backbone_options():
    with pytest.raises(ValueError, match="time_input"):
        _model(time_input=True)
    with pytest.raises(ValueError, match="mesh_activation='silu'"):
        _model(recompute_activation=True)
    _model(recompute_activation=True, mesh_activation="silu")
    with pytest.raises(ValueError, match="mesh_context"):
        _model(activation_checkpointing_components=("mesh_context",))


def test_metadata_describes_deformer():
    model = _model()
    assert isinstance(model.meta, DeFormerMetaData)
    assert model.meta.name == "DeFormer"
    assert model.meta.amp and not model.meta.onnx_cpu and not model.meta.jit


def test_return_backbone_exposes_the_full_width_state():
    torch.manual_seed(19)
    graph = _graph(7)
    model = _model(mesh_context_fusion="post", include_local_features=False)
    state = model(
        node_features=graph.x,
        graph=graph,
        geometry=graph.x[:, :3],
        global_embedding=torch.randn(1, 1, 2),
        return_backbone=True,
    )
    assert isinstance(state, BackboneState)
    assert state.point_features.shape == (7, 32)
    assert state.neighborhood.index.shape == (7, K + 1)
    assert state.mesh_context.shape == (7, 32)
    state.point_features.square().mean().backward()
    assert model.pre_processor.blocks[0].query.weight.grad is not None


def test_batched_neighborhoods_must_be_offset_per_graph():
    """Collating two graphs without offsetting knn_index is a batching error."""
    torch.manual_seed(13)
    first, second = _graph(5), _graph(6)

    def collate(offset: bool):
        shift = first.num_nodes if offset else 0
        return _PlainGraph(
            x=torch.cat((first.x, second.x)),
            num_nodes=11,
            knn_index=torch.cat((first.knn_index, second.knn_index + shift)),
            knn_mask=torch.cat((first.knn_mask, second.knn_mask)),
            knn_edge_attr=torch.cat((first.knn_edge_attr, second.knn_edge_attr)),
            knn_metric="geodesic",
            batch=torch.cat(
                (torch.zeros(5, dtype=torch.long), torch.ones(6, dtype=torch.long))
            ),
        )

    model = _model().eval()
    global_embedding = torch.randn(2, 1, 2)
    with torch.no_grad():
        assert _forward(model, collate(True), global_embedding).shape == (11, 5)
        with pytest.raises(ValueError, match="another graph"):
            _forward(model, collate(False), global_embedding)


def test_mesh_context_checkpointing_matches_plain_execution(tmp_path):
    """``mesh_context`` is DeFormer's own checkpoint component."""
    torch.manual_seed(17)
    common = dict(mesh_context_fusion="pre_post", mesh_context_use_global=True)
    plain = _model(**common).train()
    checkpointed = _model(
        **common,
        activation_checkpointing=True,
        activation_checkpointing_components=("mesh_context", "blocks"),
    ).train()
    checkpointed.load_state_dict(plain.state_dict(), strict=True)
    recorded = checkpointed._args["__args__"]["activation_checkpointing_components"]
    assert recorded == ("blocks", "mesh_context")
    graph = _graph(7)
    global_embedding = torch.randn(1, 1, 2)
    expected = _forward(plain, graph, global_embedding)
    actual = _forward(checkpointed, graph, global_embedding)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for (name, parameter), (_, other) in zip(
        plain.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        torch.testing.assert_close(parameter.grad, other.grad, msg=name)
    path = tmp_path / "DeFormer.mdlus"
    checkpointed.save(path)
    restored = DeFormer.from_checkpoint(path)
    assert restored._checkpoint_mesh_context
    assert "blocks" in restored._activation_checkpointing_components


def test_backbone_only_model_builds_no_output_head(tmp_path):
    """``output_head=False`` keeps the backbone and the pre-processor only."""
    torch.manual_seed(23)
    model = _model(output_head=False, mesh_context_fusion="pre_post").train()
    names = {name.split(".")[0] for name, _ in model.named_parameters()}
    for absent in (
        "ln_mlp_out",
        "post_node_encoder",
        "post_processor",
        "post_node_decoder",
        "post_context_film",
        "mesh_post_residual_gate",
    ):
        assert absent not in names, absent
    assert {"pre_processor", "pre_context_film", "mesh_context_encoder"} <= names
    graph = _graph(7)
    global_embedding = torch.randn(1, 1, 2)
    with pytest.raises(ValueError, match="output_head=False"):
        _forward(model, graph, global_embedding)
    state = model(
        node_features=graph.x,
        graph=graph,
        geometry=graph.x[:, :3],
        global_embedding=global_embedding,
        return_backbone=True,
    )
    assert state.point_features.shape == (7, model.backbone_width)
    state.point_features.square().mean().backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []
    decoder = model.make_node_decoder()
    assert decoder(
        state.point_features.new_zeros((7, model.mesh_hidden_dim))
    ).shape == (
        7,
        5,
    )
    path = tmp_path / "backbone.mdlus"
    model.save(path)
    restored = DeFormer.from_checkpoint(path)
    assert not restored.output_head and restored.ln_mlp_out is None
    with pytest.raises(ValueError, match="output_head"):
        _model(output_head="no")


def test_checkpoint_roundtrip(tmp_path):
    model = _model()
    checkpoint = tmp_path / "DeFormer.mdlus"
    model.save(checkpoint)
    restored = DeFormer.from_checkpoint(checkpoint)
    assert type(restored) is DeFormer
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
