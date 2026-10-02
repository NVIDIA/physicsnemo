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

from physicsnemo.experimental.models.meshtransolver import (  # noqa: E402
    CONTACT_FEATURE_DIM,
    ContactGraph,
    MeshGeoFLARE,
    MeshGeoTransolver,
    MeshTransolver,
    SparseContactBlock,
    SparseContactGraphBuilder,
)
from physicsnemo.models.geotransolver import GeoTransolver  # noqa: E402
from physicsnemo.models.geotransolver.flare_plus_plus import (  # noqa: E402
    _FLAREPlusPlusAttention,
)


def _graph(num_nodes: int, node_dim: int = 6, edge_dim: int = 4):
    src = torch.arange(num_nodes - 1)
    dst = src + 1
    edge_index = torch.stack([torch.cat((src, dst)), torch.cat((dst, src))], dim=0)
    return pyg.data.Data(
        x=torch.randn(num_nodes, node_dim),
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.shape[1], edge_dim),
        num_nodes=num_nodes,
    )


def _model(model_cls):
    kwargs = dict(
        input_dim_nodes=6,
        input_dim_edges=4,
        output_dim=5,
        hidden_dim=32,
        num_heads=4,
        slice_num=8,
        num_pre_processor_layers=1,
        num_attention_layers=2,
        num_post_processor_layers=1,
        use_te=False,
    )
    if model_cls is not MeshTransolver:
        kwargs.update(geometry_dim=3, global_dim=2)
    return model_cls(**kwargs)


def _meshgeoflare(
    mesh_context_fusion: str = "none", mesh_context_use_global: bool = False
):
    return MeshGeoFLARE(
        input_dim_nodes=6,
        input_dim_edges=4,
        output_dim=5,
        geometry_dim=3,
        global_dim=2,
        hidden_dim=32,
        num_heads=4,
        slice_num=8,
        num_pre_processor_layers=1,
        num_attention_layers=2,
        num_post_processor_layers=1,
        mesh_context_fusion=mesh_context_fusion,
        mesh_context_use_global=mesh_context_use_global,
        use_te=False,
    )


def _forward(model, graph, global_embedding=None):
    kwargs = {}
    if not isinstance(model, MeshTransolver):
        kwargs.update(
            geometry=graph.x[:, :3],
            global_embedding=global_embedding,
        )
    return model(
        node_features=graph.x,
        edge_features=graph.edge_attr,
        graph=graph,
        **kwargs,
    )


@pytest.mark.parametrize("model_cls", [MeshTransolver, MeshGeoTransolver, MeshGeoFLARE])
def test_mesh_attention_hybrid_forward_backward(model_cls):
    torch.manual_seed(7)
    graph = _graph(9)
    model = _model(model_cls)
    global_embedding = torch.randn(1, 1, 2)

    output = _forward(model, graph, global_embedding)
    output.square().mean().backward()

    assert output.shape == (9, 5)
    assert torch.isfinite(output).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("model_cls", [MeshTransolver, MeshGeoTransolver, MeshGeoFLARE])
def test_variable_graph_batch_matches_independent_forwards(model_cls):
    torch.manual_seed(11)
    graphs = [_graph(5), _graph(8)]
    batch = pyg.data.Batch.from_data_list(graphs)
    model = _model(model_cls).eval()
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


@pytest.mark.parametrize("model_cls", [MeshGeoTransolver, MeshGeoFLARE])
def test_geometry_context_is_required(model_cls):
    graph = _graph(6)
    model = _model(model_cls)

    with pytest.raises(ValueError, match="geometry is required"):
        model(
            node_features=graph.x,
            edge_features=graph.edge_attr,
            graph=graph,
            global_embedding=torch.randn(1, 1, 2),
        )


@pytest.mark.parametrize("model_cls", [MeshGeoTransolver, MeshGeoFLARE])
def test_global_context_is_required_when_configured(model_cls):
    graph = _graph(6)
    model = _model(model_cls)

    with pytest.raises(ValueError, match="global_embedding is required"):
        model(
            node_features=graph.x,
            edge_features=graph.edge_attr,
            graph=graph,
            geometry=graph.x[:, :3],
        )


def test_constructor_rejects_incompatible_hidden_and_head_dimensions():
    with pytest.raises(ValueError, match="divisible by num_heads"):
        MeshTransolver(
            input_dim_nodes=6,
            input_dim_edges=4,
            output_dim=5,
            hidden_dim=30,
            num_heads=8,
            use_te=False,
        )


@pytest.mark.parametrize("attention_type", ["GALE_FA", "GALE_FPP"])
def test_meshgeoflare_retains_full_geoflare_backbone_and_adds_parameters(
    attention_type,
):
    """Each mesh variant retains its own upstream attention backend."""
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
    hybrid = MeshGeoFLARE(
        **common,
        input_dim_edges=4,
        num_pre_processor_layers=1,
        num_post_processor_layers=1,
    )
    hybrid_state = hybrid.state_dict()
    for name, parameter in backbone.state_dict().items():
        assert name in hybrid_state
        assert hybrid_state[name].shape == parameter.shape
    assert sum(p.numel() for p in hybrid.parameters()) > sum(
        p.numel() for p in backbone.parameters()
    )
    assert isinstance(hybrid, GeoTransolver)
    for mesh_block, base_block in zip(hybrid.blocks, backbone.blocks, strict=True):
        assert type(mesh_block.Attn) is type(base_block.Attn)
    if attention_type == "GALE_FPP":
        assert all(
            isinstance(block.Attn, _FLAREPlusPlusAttention) for block in hybrid.blocks
        )


@pytest.mark.parametrize(
    "components", [("blocks",), ("context", "preprocess", "blocks", "output")]
)
def test_meshgeoflare_upstream_checkpointing_preserves_outputs_and_gradients(
    components,
):
    """The hidden-state seam honors upstream component checkpointing."""
    common = dict(
        functional_dim=6,
        out_dim=5,
        geometry_dim=3,
        global_dim=2,
        n_hidden=32,
        n_head=4,
        n_layers=2,
        slice_num=8,
        attention_type="GALE_FA",
        mesh_context_fusion="pre_post",
        mesh_context_use_global=True,
    )
    plain = MeshGeoFLARE(**common).train()
    checkpointed = MeshGeoFLARE(
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
    meshgeoflare = MeshGeoFLARE(
        **common,
        input_dim_edges=4,
        mesh_hidden_dim=32,
        num_pre_processor_layers=1,
        num_post_processor_layers=1,
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
            edge_features=graph.edge_attr,
            graph=graph,
            geometry=graph.x,
            local_positions=graph.x,
            global_embedding=global_embedding,
        )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_meshgeoflare_pre_mpnn_consumes_encoded_geoflare_latent():
    torch.manual_seed(27)
    graph = _graph(9, node_dim=6)
    model = _meshgeoflare("pre_post", True).eval()
    observed = {}

    def capture_processor_input(_module, inputs):
        observed["shape"] = tuple(inputs[0].shape)

    handle = model.pre_processor.register_forward_pre_hook(capture_processor_input)
    try:
        with torch.no_grad():
            _forward(model, graph, torch.randn(1, 1, 2))
    finally:
        handle.remove()

    # Raw functional features have width 6; the mesh stage must instead receive
    # GeoFLARE's 32-channel encoded node representation.
    assert observed["shape"] == (9, 32)
    assert not hasattr(model, "pre_node_decoder")


def test_meshgeoflare_latent_adapters_support_distinct_mesh_width():
    torch.manual_seed(28)
    graph = _graph(9, node_dim=6)
    model = MeshGeoFLARE(
        functional_dim=6,
        out_dim=5,
        input_dim_edges=4,
        geometry_dim=3,
        global_dim=2,
        n_hidden=32,
        mesh_hidden_dim=24,
        n_head=4,
        n_layers=2,
        slice_num=8,
        num_pre_processor_layers=1,
        num_post_processor_layers=1,
        use_te=False,
    ).eval()
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
def test_meshgeoflare_context_fusion_is_zero_initialized_identity(
    fusion_stage, use_global
):
    torch.manual_seed(29)
    graph = _graph(9)
    global_embedding = torch.randn(1, 1, 2)

    torch.manual_seed(31)
    baseline = _meshgeoflare().eval()
    torch.manual_seed(31)
    fused = _meshgeoflare(fusion_stage, use_global).eval()

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
def test_meshgeoflare_context_fusion_variable_batch_matches_independent(
    fusion_stage, use_global
):
    torch.manual_seed(37)
    graphs = [_graph(5), _graph(8)]
    batch = pyg.data.Batch.from_data_list(graphs)
    model = _meshgeoflare(fusion_stage, use_global).eval()
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


def test_meshgeoflare_context_fusion_parameter_ordering():
    models = [
        _meshgeoflare(),
        _meshgeoflare("post"),
        _meshgeoflare("pre_post"),
        _meshgeoflare("pre_post", True),
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
def test_meshgeoflare_rejects_invalid_context_fusion(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MeshGeoFLARE(
            input_dim_nodes=6,
            input_dim_edges=4,
            output_dim=5,
            geometry_dim=3,
            global_dim=2,
            hidden_dim=32,
            num_heads=4,
            num_attention_layers=1,
            use_te=False,
            **kwargs,
        )


def test_sparse_contact_graph_filters_mesh_edges_and_applies_topk():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.6],
        ]
    )
    structural = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    builder = SparseContactGraphBuilder(
        search_radius=1.0,
        max_neighbors=1,
        candidate_neighbors=8,
    )

    contact_graph = builder(positions, structural_edge_index=structural)
    contact_graph.validate(positions.shape[0], CONTACT_FEATURE_DIM)

    source, destination = contact_graph.edge_index
    codes = source * positions.shape[0] + destination
    structural_codes = structural[0] * positions.shape[0] + structural[1]
    assert not torch.isin(codes, structural_codes).any()
    assert torch.bincount(destination, minlength=positions.shape[0]).max() <= 1
    assert torch.isfinite(contact_graph.edge_features).all()


def test_sparse_contact_graph_uses_shell_gap_and_component_filter():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    thickness = torch.tensor([0.2, 0.2])
    builder = SparseContactGraphBuilder(
        search_radius=1.0,
        max_neighbors=2,
        exclude_same_component=True,
    )

    contact_graph = builder(
        positions,
        shell_thickness=thickness,
        component_ids=torch.tensor([0, 1]),
    )

    assert contact_graph.edge_index.shape == (2, 2)
    torch.testing.assert_close(contact_graph.edge_features[:, 4], torch.full((2,), 0.9))


def test_sparse_contact_block_zero_gate_is_identity_and_masks_inactive_nodes():
    torch.manual_seed(19)
    latent = torch.randn(4, 16, requires_grad=True)
    contact_graph = ContactGraph(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_features=torch.randn(1, CONTACT_FEATURE_DIM),
        obstacle_mask=torch.tensor([False]),
    )
    block = SparseContactBlock(hidden_dim=16)

    identity = block(latent, contact_graph)
    torch.testing.assert_close(identity, latent)
    identity.sum().backward()
    assert block.gate.grad is not None

    block.gate.data.fill_(1.0)
    updated = block(latent.detach(), contact_graph)
    torch.testing.assert_close(updated[[0, 2, 3]], latent.detach()[[0, 2, 3]])
    assert not torch.allclose(updated[1], latent.detach()[1])


def test_contact_enabled_model_requires_and_consumes_contact_graph():
    graph = _graph(7)
    model = MeshTransolver(
        input_dim_nodes=6,
        input_dim_edges=4,
        output_dim=5,
        hidden_dim=32,
        num_heads=4,
        slice_num=8,
        num_pre_processor_layers=1,
        num_attention_layers=1,
        num_post_processor_layers=1,
        use_contact=True,
        contact_gate_init=0.1,
        use_te=False,
    )
    with pytest.raises(ValueError, match="contact_graph is required"):
        model(graph.x, graph.edge_attr, graph)

    contact_graph = ContactGraph(
        edge_index=torch.tensor([[0, 2], [1, 1]], dtype=torch.long),
        edge_features=torch.randn(2, CONTACT_FEATURE_DIM),
        obstacle_mask=torch.tensor([False, True]),
    )
    output = model(graph.x, graph.edge_attr, graph, contact_graph=contact_graph)
    assert output.shape == (7, 5)


@pytest.mark.parametrize("model_cls", [MeshTransolver, MeshGeoTransolver, MeshGeoFLARE])
def test_checkpoint_roundtrip(model_cls, tmp_path):
    model = _model(model_cls)
    checkpoint = tmp_path / f"{model_cls.__name__}.mdlus"

    model.save(checkpoint)
    restored = model_cls.from_checkpoint(checkpoint)

    assert type(restored) is model_cls
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("model_cls", [MeshTransolver, MeshGeoTransolver, MeshGeoFLARE])
def test_bfloat16_amp_forward_backward(model_cls):
    graph = _graph(9).cuda()
    model = _model(model_cls).cuda()
    global_embedding = torch.randn(1, 1, 2, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = _forward(model, graph, global_embedding)
        loss = output.square().mean()
    loss.backward()

    assert output.shape == (9, 5)
    assert torch.isfinite(output).all()
