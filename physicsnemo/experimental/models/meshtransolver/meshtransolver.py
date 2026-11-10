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

r"""Hybrid structural-mesh and geometry-aware attention architectures.

``MeshGeoFLARE`` implements DeFormer: a structural MPNN processes the native
GeoTransolver node latent, optional contact messages augment that latent, and
geometry-aware FLARE communicates globally. A post-MPNN refines physical outputs.
The crash reference recipe selects point-space FLARE (``GALE_FA``), geometry
conditioning before and after attention, and predictive node-to-face contact.

The broader mesh-attention family is inspired by ``arXiv:2605.11784``; DeFormer's
latent integration and predictive surface-contact mechanism are independent
extensions, not a claim of exact reproduction. Historical class names and
constructor options are retained for checkpoint/configuration compatibility.
"""

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.utils.checkpoint import checkpoint

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.models.geotransolver import GeoTransolver, GlobalContextBuilder
from physicsnemo.models.meshgraphnet.meshgraphnet import MeshGraphNetProcessor
from physicsnemo.models.transolver.transolver import TransolverBlock
from physicsnemo.nn import GALEBlock, get_activation
from physicsnemo.nn.module.gnn_layers.graph_types import PYG_AVAILABLE
from physicsnemo.nn.module.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from physicsnemo.nn.module.gnn_layers.utils import GraphType

from .contact import CONTACT_FEATURE_DIM, ContactGraph, SparseContactBlock
from .local_attention import LocalNeighborhood, NeighborhoodAttentionProcessor

if PYG_AVAILABLE:
    from physicsnemo.nn.module.gnn_layers.graph_types import PyGData

GlobalProcessor = Literal["transolver", "gale", "gale_fa"]
MeshContextFusionStage = Literal["none", "post", "pre_post"]
LocalProcessor = Literal["mpnn", "knn_attention", "knn_mpnn"]


class BackboneState(NamedTuple):
    """Full-width state after the FLARE blocks, before any output head.

    Returned by :meth:`MeshGeoFLARE.forward` with ``return_backbone=True`` so
    recipes can attach their own post-attention stages (for example a
    latent-preserving time-conditional refinement) without a projection through
    the three-channel output.
    """

    point_features: torch.Tensor
    edge_latent: torch.Tensor | None
    graph: Any
    neighborhood: Any
    mesh_context: torch.Tensor | None


LocalNeighborhoodMetric = Literal["geodesic", "euclidean"]


@dataclass
class MeshAttentionMetaData(ModelMetaData):
    r"""Metadata shared by the hybrid graph-attention models."""

    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    onnx: bool = False
    func_torch: bool = True
    auto_grad: bool = True


class MeshGeometryContextEncoder(nn.Module):
    r"""Build node-aligned mesh context from geometry and GALE geometry tokens.

    Pointwise geometry preserves the node's location while the pooled live GALE
    geometry tokens provide a simulation-level shape summary. Optional normalized
    global parameters can be added for recipe-specific conditioning. The returned
    tensor is aligned with flattened graph nodes and can condition any mesh stage.
    """

    def __init__(
        self,
        geometry_dim: int,
        geometry_token_dim: int,
        hidden_dim: int,
        global_dim: int | None = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if geometry_dim <= 0:
            raise ValueError("geometry_dim must be positive")
        if geometry_token_dim <= 0:
            raise ValueError("geometry_token_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if global_dim is not None and global_dim <= 0:
            raise ValueError("global_dim must be positive when configured")

        self.geometry_dim = geometry_dim
        self.geometry_token_dim = geometry_token_dim
        self.hidden_dim = hidden_dim
        self.global_dim = global_dim

        self.local_geometry = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.geometry_tokens = nn.Sequential(
            nn.Linear(geometry_token_dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_context = (
            nn.Sequential(
                nn.Linear(global_dim, hidden_dim),
                get_activation(activation),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if global_dim is not None
            else None
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        geometry: Float[torch.Tensor, "num_nodes geometry_dim"],
        batch: torch.Tensor,
        geometry_context: Float[
            torch.Tensor, "batch heads context_slices geometry_token_dim"
        ],
        global_embedding: (
            Float[torch.Tensor, "batch global_tokens global_dim"] | None
        ) = None,
    ) -> Float[torch.Tensor, "num_nodes hidden_dim"]:
        if not torch.compiler.is_compiling():
            if geometry.ndim != 2 or geometry.shape[-1] != self.geometry_dim:
                raise ValueError(
                    f"Expected geometry [N, {self.geometry_dim}], got "
                    f"{tuple(geometry.shape)}"
                )
            if batch.ndim != 1 or batch.shape[0] != geometry.shape[0]:
                raise ValueError("batch must contain one graph index per node")
            if (
                geometry_context.ndim != 4
                or geometry_context.shape[-1] != self.geometry_token_dim
            ):
                raise ValueError(
                    "Expected geometry_context [B, H, S, "
                    f"{self.geometry_token_dim}], got "
                    f"{tuple(geometry_context.shape)}"
                )

        graph_context = self.geometry_tokens(geometry_context.mean(dim=(1, 2)))
        node_context = self.local_geometry(geometry)
        node_context = node_context + graph_context.index_select(0, batch)

        if self.global_context is not None:
            if global_embedding is None:
                raise ValueError(
                    "global_embedding is required for global mesh-context fusion"
                )
            if not torch.compiler.is_compiling() and (
                global_embedding.ndim != 3
                or global_embedding.shape[0] != graph_context.shape[0]
                or global_embedding.shape[-1] != self.global_dim
            ):
                raise ValueError(
                    f"Expected global_embedding [B, T, {self.global_dim}], got "
                    f"{tuple(global_embedding.shape)}"
                )
            global_context = self.global_context(global_embedding.mean(dim=1))
            node_context = node_context + global_context.index_select(0, batch)
        elif global_embedding is not None:
            raise ValueError(
                "global_embedding was provided to geometry-only mesh-context fusion"
            )

        return self.output_norm(node_context)


class MeshContextFiLM(nn.Module):
    r"""Zero-initialized residual FiLM adapter for a mesh latent state."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.latent_norm = nn.LayerNorm(hidden_dim)
        self.affine = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(
        self,
        latent: Float[torch.Tensor, "num_nodes hidden_dim"],
        context: Float[torch.Tensor, "num_nodes hidden_dim"],
    ) -> Float[torch.Tensor, "num_nodes hidden_dim"]:
        if not torch.compiler.is_compiling() and (
            latent.shape != context.shape or latent.shape[-1] != self.hidden_dim
        ):
            raise ValueError(
                f"Expected matching [N, {self.hidden_dim}] latent/context tensors, "
                f"got {tuple(latent.shape)} and {tuple(context.shape)}"
            )
        scale, shift = self.affine(context).chunk(2, dim=-1)
        return latent + torch.tanh(scale) * self.latent_norm(latent) + shift


class MeshAttentionHybrid(Module):
    r"""Reusable local-global-local processor for irregular meshes.

    Parameters
    ----------
    input_dim_nodes : int
        Number of input node features.
    input_dim_edges : int
        Number of input structural-edge features.
    output_dim : int
        Number of output channels per node.
    global_processor : {"transolver", "gale", "gale_fa"}
        Global attention mechanism between the pre- and post-MPNN stages.
    geometry_dim : int | None, optional
        Geometry context dimension for GALE processors. Default is ``None``.
    global_dim : int | None, optional
        Global context dimension for GALE processors. Default is ``None``.
    num_pre_processor_layers : int, optional
        Number of structural-mesh message-passing blocks before attention.
        Default is 1.
    num_attention_layers : int, optional
        Number of global attention blocks. Default is 4.
    num_post_processor_layers : int, optional
        Number of structural-mesh message-passing blocks after attention.
        Default is 2.
    hidden_dim : int, optional
        Latent node and edge dimension; must be divisible by ``num_heads``.
        Default is 128.
    num_heads : int, optional
        Number of global-attention heads. Default is 8.
    slice_num : int, optional
        Number of learned physics slices/tokens. Default is 128.
    dropout : float, optional
        Dropout rate of the global attention blocks and context builder.
        Default is 0.0.
    attention_activation : str, optional
        Activation of the attention blocks' feed-forward nets. Default is
        ``"gelu"``.
    attention_mlp_ratio : int, optional
        Feed-forward expansion ratio of the attention blocks. Default is 4.
    plus : bool, optional
        Enable Transolver++ slice parameterization where supported. Default is
        ``False``.
    use_te : bool, optional
        Use the Transformer Engine backend of the attention blocks when
        available. Default is ``False``.
    mesh_activation : str, optional
        Activation of the mesh encoders, decoder and MeshGraphNet processors.
        Default is ``"relu"``.
    num_layers_node_processor, num_layers_edge_processor : int, optional
        Hidden layers of the node and edge MLPs of each MeshGraphNet block.
        Default is 2 each.
    num_layers_node_encoder, num_layers_edge_encoder, num_layers_node_decoder : int, optional
        Hidden layers of the node encoder, edge encoder and node decoder MLPs.
        Default is 2 each.
    aggregation : {"sum", "mean"}, optional
        MeshGraphNet message aggregation. Default is ``"sum"``.
    do_concat_trick : bool, optional
        MeshGraphNet concat trick: replace concat+MLP with MLP+idx+sum. Default
        is ``False``.
    norm_type : {"LayerNorm", "TELayerNorm"}, optional
        Normalization of the mesh MLPs and MeshGraphNet blocks. Default is
        ``"LayerNorm"``.
    num_pre_processor_checkpoint_segments, num_post_processor_checkpoint_segments : int, optional
        Activation-checkpoint segments of the pre- and post-processor; 0
        disables checkpointing. Default is 0.
    checkpoint_offloading : bool, optional
        Offload the processors' checkpointed activations to host memory.
        Default is ``False``.
    recompute_activation : bool, optional
        Recompute the mesh MLP activations during backward instead of storing
        them. Default is ``False``.
    state_mixing_mode : str, optional
        How the GALE blocks blend self- and cross-attention: ``"weighted"``
        (learnable sigmoid gate) or ``"concat_project"``. Default is
        ``"weighted"``.
    use_contact : bool, optional
        Add a :class:`SparseContactBlock` after the pre-processor; forward then
        requires ``contact_graph``. Default is ``False``.
    contact_dim : int, optional
        Contact edge feature width. Default is ``CONTACT_FEATURE_DIM`` (8).
    contact_gate_init : float, optional
        Initial value of the contact block's output gate. Default is 0.0.
    contact_aggregation : {"sum", "mean"}, optional
        Contact message aggregation per destination node. Default is ``"mean"``.

    Forward
    -------
    node_features : torch.Tensor
        Flattened node tensor of shape ``[N, D_node]``.
    edge_features : torch.Tensor
        Flattened structural-edge tensor of shape ``[E, D_edge]``.
    graph : GraphType
        PyG graph or batch. Its optional ``batch`` vector separates simulations.
    geometry : torch.Tensor | None
        Geometry context aligned with flattened nodes, shape ``[N, D_geometry]``.
    global_embedding : torch.Tensor | None
        Per-graph global tokens, shape ``[B, N_global, D_global]``.
    contact_graph : ContactGraph | None
        Sparse contact graph, required when ``use_contact=True``.

    Returns
    -------
    torch.Tensor
        Flattened per-node output of shape ``[N, output_dim]``.

    Notes
    -----
    Mesh message passing can process a flattened PyG batch directly. Global
    attention is instead evaluated once per graph so nodes from unrelated samples
    never share physics tokens. This supports variable node counts without padding.
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        global_processor: GlobalProcessor,
        geometry_dim: int | None = None,
        global_dim: int | None = None,
        num_pre_processor_layers: int = 1,
        num_attention_layers: int = 4,
        num_post_processor_layers: int = 2,
        hidden_dim: int = 128,
        num_heads: int = 8,
        slice_num: int = 128,
        dropout: float = 0.0,
        attention_activation: str = "gelu",
        attention_mlp_ratio: int = 4,
        plus: bool = False,
        use_te: bool = False,
        mesh_activation: str = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        num_layers_node_encoder: int = 2,
        num_layers_edge_encoder: int = 2,
        num_layers_node_decoder: int = 2,
        aggregation: Literal["sum", "mean"] = "sum",
        do_concat_trick: bool = False,
        norm_type: Literal["LayerNorm", "TELayerNorm"] = "LayerNorm",
        num_pre_processor_checkpoint_segments: int = 0,
        num_post_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
        recompute_activation: bool = False,
        state_mixing_mode: str = "weighted",
        use_contact: bool = False,
        contact_dim: int = CONTACT_FEATURE_DIM,
        contact_gate_init: float = 0.0,
        contact_aggregation: Literal["sum", "mean"] = "mean",
    ) -> None:
        super().__init__(meta=MeshAttentionMetaData())

        if input_dim_nodes <= 0:
            raise ValueError("input_dim_nodes must be positive")
        if input_dim_edges <= 0:
            raise ValueError("input_dim_edges must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if num_pre_processor_layers <= 0:
            raise ValueError("num_pre_processor_layers must be positive")
        if num_attention_layers <= 0:
            raise ValueError("num_attention_layers must be positive")
        if num_post_processor_layers <= 0:
            raise ValueError("num_post_processor_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if slice_num <= 0:
            raise ValueError("slice_num must be positive")
        if hidden_dim <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be positive and divisible by num_heads; "
                f"got hidden_dim={hidden_dim}, num_heads={num_heads}"
            )
        if global_processor not in ("transolver", "gale", "gale_fa"):
            raise ValueError(
                "global_processor must be 'transolver', 'gale', or 'gale_fa'; "
                f"got {global_processor!r}"
            )
        if (
            global_processor != "transolver"
            and geometry_dim is None
            and global_dim is None
        ):
            raise ValueError(
                "GALE processors require geometry_dim, global_dim, or both"
            )

        self.input_dim_nodes = input_dim_nodes
        self.input_dim_edges = input_dim_edges
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.global_processor = global_processor
        self.geometry_dim = geometry_dim
        self.global_dim = global_dim
        self.num_pre_processor_layers = num_pre_processor_layers
        self.num_attention_layers = num_attention_layers
        self.num_post_processor_layers = num_post_processor_layers
        self.use_contact = use_contact
        self.contact_dim = contact_dim

        mesh_activation_fn = get_activation(mesh_activation)
        self.node_encoder = MeshGraphMLP(
            input_dim_nodes,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_layers_node_encoder,
            activation_fn=mesh_activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )
        self.edge_encoder = MeshGraphMLP(
            input_dim_edges,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=mesh_activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )
        self.node_decoder = MeshGraphMLP(
            hidden_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_layers_node_decoder,
            activation_fn=mesh_activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )

        processor_kwargs = dict(
            input_dim_node=hidden_dim,
            input_dim_edge=hidden_dim,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type=norm_type,
            activation_fn=mesh_activation_fn,
            do_concat_trick=do_concat_trick,
            checkpoint_offloading=checkpoint_offloading,
        )
        self.pre_processor = MeshGraphNetProcessor(
            processor_size=num_pre_processor_layers,
            num_processor_checkpoint_segments=num_pre_processor_checkpoint_segments,
            **processor_kwargs,
        )
        self.post_processor = MeshGraphNetProcessor(
            processor_size=num_post_processor_layers,
            num_processor_checkpoint_segments=num_post_processor_checkpoint_segments,
            **processor_kwargs,
        )
        self.contact_block = (
            SparseContactBlock(
                hidden_dim=hidden_dim,
                contact_dim=contact_dim,
                gate_init=contact_gate_init,
                aggregation=contact_aggregation,
            )
            if use_contact
            else None
        )

        self.context_builder: GlobalContextBuilder | None = None
        if global_processor == "transolver":
            self.attention_blocks = nn.ModuleList(
                [
                    TransolverBlock(
                        num_heads=num_heads,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        act=attention_activation,
                        mlp_ratio=attention_mlp_ratio,
                        last_layer=False,
                        out_dim=hidden_dim,
                        slice_num=slice_num,
                        spatial_shape=None,
                        use_te=use_te,
                        plus=plus,
                    )
                    for _ in range(num_attention_layers)
                ]
            )
        else:
            self.context_builder = GlobalContextBuilder(
                functional_dims=(hidden_dim,),
                geometry_dim=geometry_dim,
                global_dim=global_dim,
                n_hidden=hidden_dim,
                n_head=num_heads,
                dropout=dropout,
                slice_num=slice_num,
                use_te=use_te,
                plus=plus,
                include_local_features=False,
                structured_shape=None,
            )
            attention_type = "GALE" if global_processor == "gale" else "GALE_FA"
            context_dim = self.context_builder.get_context_dim()
            self.attention_blocks = nn.ModuleList(
                [
                    GALEBlock(
                        num_heads=num_heads,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        act=attention_activation,
                        mlp_ratio=attention_mlp_ratio,
                        slice_num=slice_num,
                        use_te=use_te,
                        plus=plus,
                        context_dim=context_dim,
                        spatial_shape=None,
                        attention_type=attention_type,
                        state_mixing_mode=state_mixing_mode,
                    )
                    for _ in range(num_attention_layers)
                ]
            )

    @staticmethod
    def _batch_index(
        graph: GraphType, num_nodes: int, device: torch.device
    ) -> tuple[torch.Tensor, int]:
        batch = getattr(graph, "batch", None)
        if batch is None:
            return torch.zeros(num_nodes, dtype=torch.long, device=device), 1
        if batch.ndim != 1 or batch.shape[0] != num_nodes:
            raise ValueError(
                "graph.batch must have one entry per node; "
                f"got {tuple(batch.shape)} for {num_nodes} nodes"
            )
        batch = batch.to(device=device, dtype=torch.long)
        if batch.numel() == 0:
            raise ValueError("The graph batch contains no nodes")
        if torch.any(batch < 0):
            raise ValueError("graph.batch entries must be non-negative")
        num_graphs = int(batch.max().item()) + 1
        expected = torch.arange(num_graphs, device=device)
        if not torch.equal(torch.unique(batch), expected):
            raise ValueError("graph.batch IDs must be contiguous and start at zero")
        return batch, num_graphs

    @staticmethod
    def _flatten_geometry(
        geometry: torch.Tensor | None,
        num_nodes: int,
        num_graphs: int,
    ) -> torch.Tensor | None:
        if geometry is None:
            return None
        if geometry.ndim == 2:
            flattened = geometry
        elif geometry.ndim == 3:
            if geometry.shape[0] != num_graphs:
                raise ValueError(
                    "Batched geometry must have one leading entry per graph; "
                    f"got {geometry.shape[0]} for {num_graphs} graphs"
                )
            flattened = geometry.reshape(-1, geometry.shape[-1])
        else:
            raise ValueError(
                "geometry must have shape [N, D] or [B, N, D]; "
                f"got {tuple(geometry.shape)}"
            )
        if flattened.shape[0] != num_nodes:
            raise ValueError(
                "geometry must align with flattened graph nodes; "
                f"got {flattened.shape[0]} geometry rows for {num_nodes} nodes"
            )
        return flattened

    @staticmethod
    def _normalize_global_embedding(
        global_embedding: torch.Tensor | None,
        num_graphs: int,
    ) -> torch.Tensor | None:
        if global_embedding is None:
            return None
        if global_embedding.ndim == 1:
            global_embedding = global_embedding.view(1, 1, -1)
        elif global_embedding.ndim == 2:
            if num_graphs == 1:
                global_embedding = global_embedding.unsqueeze(0)
            elif global_embedding.shape[0] == num_graphs:
                global_embedding = global_embedding.unsqueeze(1)
            else:
                raise ValueError(
                    "A 2D global_embedding for a graph batch must have shape "
                    "[B, D] (one token per graph)"
                )
        elif global_embedding.ndim != 3:
            raise ValueError(
                "global_embedding must have shape [D], [tokens, D], [B, D], "
                f"or [B, tokens, D]; got {tuple(global_embedding.shape)}"
            )
        if global_embedding.shape[0] != num_graphs:
            raise ValueError(
                "global_embedding must have one leading entry per graph; "
                f"got {global_embedding.shape[0]} for {num_graphs} graphs"
            )
        return global_embedding

    def _apply_attention(
        self,
        node_latent: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        geometry: torch.Tensor | None,
        global_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        output = torch.zeros_like(node_latent)
        for graph_idx in range(num_graphs):
            node_index = torch.nonzero(batch == graph_idx, as_tuple=False).flatten()
            latent = node_latent.index_select(0, node_index).unsqueeze(0)

            if self.global_processor == "transolver":
                for block in self.attention_blocks:
                    latent = block(latent)
            else:
                if self.context_builder is None:
                    raise RuntimeError("GALE processor is missing its context builder")
                geometry_i = None
                if geometry is not None:
                    geometry_i = geometry.index_select(0, node_index).unsqueeze(0)
                global_i = None
                if global_embedding is not None:
                    global_i = global_embedding[graph_idx : graph_idx + 1]
                context, _, _ = self.context_builder.build_context(
                    (latent,),
                    local_positions=None,
                    geometry=geometry_i,
                    global_embedding=global_i,
                )
                if context is None:
                    raise ValueError(
                        "GALE attention requires geometry or global context at forward"
                    )
                for block in self.attention_blocks:
                    latent = block((latent,), context)[0]

            output = output.index_copy(0, node_index, latent.squeeze(0))
        return output

    def forward(
        self,
        node_features: Float[torch.Tensor, "num_nodes input_dim_nodes"],
        edge_features: Float[torch.Tensor, "num_edges input_dim_edges"],
        graph: GraphType,
        geometry: Float[torch.Tensor, "*geometry_nodes geometry_dim"] | None = None,
        global_embedding: (
            Float[torch.Tensor, "*global_shape global_dim"] | None
        ) = None,
        contact_graph: ContactGraph | None = None,
    ) -> Float[torch.Tensor, "num_nodes output_dim"]:
        r"""Run mesh encoding, global attention, and mesh refinement."""

        if not torch.compiler.is_compiling():
            if (
                node_features.ndim != 2
                or node_features.shape[1] != self.input_dim_nodes
            ):
                raise ValueError(
                    f"Expected node_features [N, {self.input_dim_nodes}], got "
                    f"{tuple(node_features.shape)}"
                )
            if (
                edge_features.ndim != 2
                or edge_features.shape[1] != self.input_dim_edges
            ):
                raise ValueError(
                    f"Expected edge_features [E, {self.input_dim_edges}], got "
                    f"{tuple(edge_features.shape)}"
                )
            graph_num_nodes = int(graph.num_nodes)
            graph_num_edges = int(graph.num_edges)
            if graph_num_nodes != node_features.shape[0]:
                raise ValueError(
                    f"Graph has {graph_num_nodes} nodes but node_features has "
                    f"{node_features.shape[0]} rows"
                )
            if graph_num_edges != edge_features.shape[0]:
                raise ValueError(
                    f"Graph has {graph_num_edges} edges but edge_features has "
                    f"{edge_features.shape[0]} rows"
                )

        batch, num_graphs = self._batch_index(
            graph, node_features.shape[0], node_features.device
        )
        geometry = self._flatten_geometry(geometry, node_features.shape[0], num_graphs)
        global_embedding = self._normalize_global_embedding(
            global_embedding, num_graphs
        )

        if self.geometry_dim is not None:
            if geometry is None:
                raise ValueError("geometry is required when geometry_dim is configured")
            if geometry.shape[-1] != self.geometry_dim:
                raise ValueError(
                    f"Expected geometry dimension {self.geometry_dim}, got "
                    f"{geometry.shape[-1]}"
                )
        if self.global_dim is not None:
            if global_embedding is None:
                raise ValueError(
                    "global_embedding is required when global_dim is configured"
                )
            if global_embedding.shape[-1] != self.global_dim:
                raise ValueError(
                    f"Expected global dimension {self.global_dim}, got "
                    f"{global_embedding.shape[-1]}"
                )

        node_latent = self.node_encoder(node_features)
        edge_latent = self.edge_encoder(edge_features)
        node_latent = self.pre_processor(node_latent, edge_latent, graph)
        if self.contact_block is not None:
            if contact_graph is None:
                raise ValueError("contact_graph is required when use_contact=True")
            node_latent = self.contact_block(node_latent, contact_graph)
        elif contact_graph is not None:
            raise ValueError("contact_graph was provided but use_contact=False")
        node_latent = self._apply_attention(
            node_latent,
            batch,
            num_graphs,
            geometry,
            global_embedding,
        )
        node_latent = self.post_processor(node_latent, edge_latent, graph)
        return self.node_decoder(node_latent)


class MeshTransolver(MeshAttentionHybrid):
    r"""MeshGraphNet with a Transolver++ global processor."""

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        num_pre_processor_layers: int = 1,
        num_attention_layers: int = 6,
        num_post_processor_layers: int = 2,
        hidden_dim: int = 128,
        num_heads: int = 8,
        slice_num: int = 128,
        plus: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim_nodes=input_dim_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=output_dim,
            global_processor="transolver",
            num_pre_processor_layers=num_pre_processor_layers,
            num_attention_layers=num_attention_layers,
            num_post_processor_layers=num_post_processor_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            slice_num=slice_num,
            plus=plus,
            **kwargs,
        )


class MeshGeoTransolver(MeshAttentionHybrid):
    r"""MeshGraphNet with a GALE geometry-aware global processor."""

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        geometry_dim: int = 3,
        global_dim: int | None = None,
        num_pre_processor_layers: int = 1,
        num_attention_layers: int = 4,
        num_post_processor_layers: int = 2,
        hidden_dim: int = 128,
        num_heads: int = 8,
        slice_num: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim_nodes=input_dim_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=output_dim,
            global_processor="gale",
            geometry_dim=geometry_dim,
            global_dim=global_dim,
            num_pre_processor_layers=num_pre_processor_layers,
            num_attention_layers=num_attention_layers,
            num_post_processor_layers=num_post_processor_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            slice_num=slice_num,
            **kwargs,
        )


class MeshGeoFLARE(GeoTransolver):
    r"""DeFormer: FLARE-family GeoTransolver with additive mesh message passing.

    The ``MeshGeoFLARE`` class name is retained so saved model metadata and
    Hydra targets remain loadable. The reference DeFormer recipe explicitly
    selects ``attention_type="GALE_FA"`` (original FLARE); the class also
    accepts FLARE++ for compatibility with earlier experiments.

    The complete :class:`GeoTransolver` backbone is retained: its input
    projection, geometry/global context builder, FLARE-family blocks, local
    ball-query features, and output projection are all inherited directly. The
    pre-mesh processor operates on the native ``n_hidden`` node embedding after
    GeoTransolver's input projection and before local-feature concatenation and
    FLARE attention. A second mesh processor refines the final physical outputs.
    With contact disabled, setting both mesh residual gates to zero recovers the
    matching GeoFLARE function. The latent pre-processor avoids a
    decode-to-input/re-encode bottleneck. Consequently, every
    fixed-query GeoFLARE parameter is present with the same shape and
    MeshGeoFLARE always has strictly more parameters for a matching configuration.

    ``checkpoint_contact=True`` recomputes the contact residual during backward
    instead of retaining its edge-wide MLP activations alongside later attention
    and structural blocks. It changes neither parameters nor contact selection.

    ``local_processor`` selects the local operator run before and after the
    FLARE blocks. ``"mpnn"`` (the reference recipe) runs MeshGraphNet processors
    on the finite-element edges. ``"knn_attention"`` replaces both processors
    with
    :class:`~physicsnemo.experimental.models.meshtransolver.local_attention.NeighborhoodAttentionProcessor`
    stacks over static reference neighborhoods. ``"knn_mpnn"`` keeps the
    MeshGraphNet processors but runs them on the static kNN slots: every valid
    slot is one directed edge neighbor -> node (the self slot included, as a
    self loop) and the slot features replace the structural edge features. Both
    kNN modes need the datapipe to attach ``knn_index`` (``[N, local_knn_k + 1]``
    including the node itself), ``knn_mask``, ``knn_edge_attr``
    (``[N, local_knn_k + 1, input_dim_edges]``) and optionally ``knn_metric`` to
    the graph; ``local_neighborhood`` names the metric those neighborhoods were
    built with and is checked against ``knn_metric``. Structural
    ``edge_features`` are ignored in the kNN modes. The residual gates, FiLM
    conditioning, contact block and backbone are unchanged, so the model is
    still exactly GeoFLARE at zero gates. The kNN modes are experimental
    options; they do not change the validated MPNN reference recipe.

    ``input_dim_nodes``, ``output_dim``, ``hidden_dim``, ``num_heads``, and
    ``num_attention_layers`` remain accepted as compatibility aliases for the
    original hybrid API.  New configurations should use the native GeoFLARE names
    ``functional_dim``, ``out_dim``, ``n_hidden``, ``n_head``, and ``n_layers``.

    The arguments below are DeFormer's. The remaining constructor arguments
    (``functional_dim``, ``out_dim``, ``geometry_dim``, ``global_dim``,
    ``n_layers``, ``n_hidden``, ``dropout``, ``n_head``, ``act``, ``mlp_ratio``,
    ``slice_num``, ``use_te``, ``time_input``, ``plus``,
    ``include_local_features``, ``radii``, ``neighbors_in_radius``,
    ``n_hidden_local``, ``attention_type``, ``concrete_dropout``,
    ``state_mixing_mode``, ``activation_checkpointing``, ``checkpointing_ratio``
    and ``activation_checkpointing_components``) are GeoTransolver's and are
    documented on :class:`~physicsnemo.models.geotransolver.GeoTransolver`;
    here ``attention_type`` must be ``"GALE_FA"`` or ``"GALE_FPP"``, and
    ``n_hidden``, ``n_head`` and ``n_layers`` default to 256, 8 and 6.

    Parameters
    ----------
    input_dim_edges : int, optional
        Width of the structural edge features (``"mpnn"``) or of the per-slot
        kNN features (``"knn_attention"``, ``"knn_mpnn"``). Default is 4.
    guard_config : dict, optional
        No longer supported; any value raises (use the external GeoTransolver
        OOD guard API). Default is ``None``.
    attn_scale : float, optional
        FLARE attention-logit scale, forwarded to GeoTransolver; ``None`` keeps
        the backend default. Default is ``None``.
    num_pre_processor_layers : int, optional
        Local processor blocks on the latent before the FLARE blocks. Default
        is 1.
    num_post_processor_layers : int, optional
        Local processor blocks on the physical outputs after the FLARE blocks.
        Default is 2.
    mesh_hidden_dim : int, optional
        Latent width of both local processors; ``None`` uses ``n_hidden``. A
        different width adds linear adapters around the pre-processor. Default
        is ``None``.
    mesh_activation : str, optional
        Activation of the mesh encoders, decoders, MeshGraphNet processors and
        mesh context encoder. Default is ``"relu"``.
    num_layers_node_processor, num_layers_edge_processor : int, optional
        Hidden layers of the node and edge MLPs of each MeshGraphNet block.
        Default is 2 each.
    num_layers_node_encoder, num_layers_edge_encoder : int, optional
        Hidden layers of the post-processor node encoder and of the structural
        edge encoder. Default is 2 each.
    num_layers_node_decoder : int, optional
        Hidden layers of the post-processor node decoder. Default is 2.
    aggregation : {"sum", "mean"}, optional
        MeshGraphNet message aggregation. Default is ``"sum"``.
    do_concat_trick : bool, optional
        MeshGraphNet concat trick: replace concat+MLP with MLP+idx+sum. Default
        is ``False``.
    norm_type : {"LayerNorm", "TELayerNorm"}, optional
        Normalization of the mesh MLPs and MeshGraphNet blocks. Default is
        ``"LayerNorm"``.
    num_pre_processor_checkpoint_segments, num_post_processor_checkpoint_segments : int, optional
        Activation-checkpoint segments of the pre- and post-processor; 0
        disables checkpointing. Default is 0.
    checkpoint_offloading : bool, optional
        Offload the local processors' checkpointed activations to host memory.
        Default is ``False``.
    recompute_activation : bool, optional
        Recompute the mesh encoder/decoder MLP activations during backward
        instead of storing them. Default is ``False``.
    mesh_pre_residual_gate_init : float, optional
        Initial value of the learnable gate on the pre-processor's latent
        update; 0.0 starts as exact GeoFLARE. Default is 1.0.
    mesh_post_residual_gate_init : float, optional
        Initial value of the learnable gate on the post-processor's output
        correction. Default is 1.0.
    mesh_context_fusion : {"none", "post", "pre_post"}, optional
        FiLM-condition the post-processor input (``"post"``) or both processor
        inputs (``"pre_post"``) on a mesh context built from the pointwise
        geometry and the live GALE geometry tokens. Requires ``geometry_dim``.
        Default is ``"none"``.
    mesh_context_use_global : bool, optional
        Also feed the global embedding into that mesh context. Requires
        ``global_dim`` and mesh context fusion. Default is ``False``.
    use_contact : bool, optional
        Add a :class:`SparseContactBlock` on the pre-processor latent; forward
        then requires ``contact_graph``. Default is ``False``.
    contact_dim : int, optional
        Contact edge feature width. Default is ``CONTACT_FEATURE_DIM`` (8).
    contact_gate_init : float, optional
        Initial value of the contact block's output gate. Default is 0.0.
    contact_aggregation : {"sum", "mean"}, optional
        Contact message aggregation per destination node. Default is ``"mean"``.
    checkpoint_contact : bool, optional
        Recompute the contact residual during backward instead of storing its
        edge-wide MLP activations; parameters and contact selection are
        unchanged. Default is ``False``.
    contact_isolate_rng : bool, optional
        Initialisation-order isolation for seed-matched ablations: draw the
        contact block's parameters from a dedicated RNG substream derived from
        the current state and restore that state afterwards, so a contact and a
        no-contact model built from one seed share every other initial
        parameter (the autoregressive contact recipe sets it). Default is
        ``False`` (plain initialisation order).
    input_dim_nodes, output_dim, hidden_dim, num_heads, num_attention_layers : int, optional
        Compatibility aliases of ``functional_dim``, ``out_dim``, ``n_hidden``,
        ``n_head`` and ``n_layers``; a conflicting pair raises. Default is
        ``None``.
    local_processor : {"mpnn", "knn_attention", "knn_mpnn"}, optional
        Local operator, see above. Default is ``"mpnn"``.
    local_knn_k : int, optional
        Neighbors per node in the kNN modes, excluding the node's own slot.
        Default is 32.
    local_neighborhood : {"geodesic", "euclidean"}, optional
        Metric the kNN neighborhoods were built with. Default is ``"geodesic"``.
    local_num_heads : int, optional
        Attention heads of ``"knn_attention"``; must divide ``mesh_hidden_dim``.
        ``None`` uses ``n_head``. Default is ``None``.
    local_mlp_ratio : int, optional
        Feed-forward expansion ratio of the kNN attention blocks. Default is 2.
    local_edge_dim : int, optional
        Encoded per-slot edge latent width of the kNN attention. Default is 32.
    local_activation : str, optional
        Activation of the kNN attention's edge encoder and feed-forward nets.
        Default is ``"gelu"``.
    local_attention_fused : bool, optional
        Run the kNN attention with the fused kernel of
        :mod:`~physicsnemo.experimental.models.meshtransolver.fused_neighborhood_attention`
        (same math, no gathered ``[N, K, H, d_h]`` key/value tensors). Default
        is ``False``.
    local_attention_backend : {"auto", "triton", "torch"}, optional
        Backend of the fused kernel; ``"auto"`` picks Triton on CUDA. Default
        is ``"auto"``.
    local_value_norm, local_difference_values, local_update, local_bias_fp32, local_split_key_value, local_dropout, local_concrete_dropout
        Forwarded to every
        :class:`~physicsnemo.experimental.models.meshtransolver.local_attention.NeighborhoodAttentionBlock`
        as ``value_norm``, ``difference_values``, ``update``, ``bias_fp32``,
        ``split_key_value``, ``dropout`` and ``concrete_dropout``; the defaults
        (``True``, ``False``, ``"residual"``, ``False``, ``False``, 0.0,
        ``False``) reproduce the original block.
    """

    def __init__(
        self,
        functional_dim: int | None = None,
        out_dim: int | None = None,
        input_dim_edges: int = 4,
        geometry_dim: int | None = 3,
        global_dim: int | None = None,
        n_layers: int | None = None,
        n_hidden: int | None = None,
        dropout: float = 0.0,
        n_head: int | None = None,
        act: str = "gelu",
        mlp_ratio: int = 4,
        slice_num: int = 32,
        use_te: bool = False,
        time_input: bool = False,
        plus: bool = False,
        include_local_features: bool = False,
        radii: list[float] | None = None,
        neighbors_in_radius: list[int] | None = None,
        n_hidden_local: int = 32,
        guard_config: dict | None = None,
        attention_type: str = "GALE_FPP",
        concrete_dropout: bool = False,
        state_mixing_mode: str = "weighted",
        attn_scale: float | None = None,
        num_pre_processor_layers: int = 1,
        num_post_processor_layers: int = 2,
        mesh_hidden_dim: int | None = None,
        mesh_activation: str = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        num_layers_node_encoder: int = 2,
        num_layers_edge_encoder: int = 2,
        num_layers_node_decoder: int = 2,
        aggregation: Literal["sum", "mean"] = "sum",
        do_concat_trick: bool = False,
        norm_type: Literal["LayerNorm", "TELayerNorm"] = "LayerNorm",
        num_pre_processor_checkpoint_segments: int = 0,
        num_post_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
        recompute_activation: bool = False,
        mesh_pre_residual_gate_init: float = 1.0,
        mesh_post_residual_gate_init: float = 1.0,
        mesh_context_fusion: MeshContextFusionStage = "none",
        mesh_context_use_global: bool = False,
        use_contact: bool = False,
        contact_dim: int = CONTACT_FEATURE_DIM,
        contact_gate_init: float = 0.0,
        contact_aggregation: Literal["sum", "mean"] = "mean",
        checkpoint_contact: bool = False,
        contact_isolate_rng: bool = False,
        input_dim_nodes: int | None = None,
        output_dim: int | None = None,
        hidden_dim: int | None = None,
        num_heads: int | None = None,
        num_attention_layers: int | None = None,
        activation_checkpointing: bool = False,
        checkpointing_ratio: float = 1.0,
        activation_checkpointing_components: tuple[str, ...] | list[str] = ("blocks",),
        local_processor: LocalProcessor = "mpnn",
        local_knn_k: int = 32,
        local_neighborhood: LocalNeighborhoodMetric = "geodesic",
        local_num_heads: int | None = None,
        local_mlp_ratio: int = 2,
        local_edge_dim: int = 32,
        local_activation: str = "gelu",
        local_attention_fused: bool = False,
        local_attention_backend: str = "auto",
        local_value_norm: bool = True,
        local_difference_values: bool = False,
        local_update: str = "residual",
        local_bias_fp32: bool = False,
        local_split_key_value: bool = False,
        local_dropout: float = 0.0,
        local_concrete_dropout: bool = False,
    ) -> None:
        functional_dim = self._resolve_alias(
            functional_dim, input_dim_nodes, "functional_dim", "input_dim_nodes"
        )
        out_dim = self._resolve_alias(out_dim, output_dim, "out_dim", "output_dim")
        n_hidden = self._resolve_alias(
            n_hidden, hidden_dim, "n_hidden", "hidden_dim", default=256
        )
        n_head = self._resolve_alias(
            n_head, num_heads, "n_head", "num_heads", default=8
        )
        n_layers = self._resolve_alias(
            n_layers,
            num_attention_layers,
            "n_layers",
            "num_attention_layers",
            default=6,
        )

        if functional_dim is None:
            raise ValueError("functional_dim (or input_dim_nodes) is required")
        if out_dim is None:
            raise ValueError("out_dim (or output_dim) is required")
        if input_dim_edges <= 0:
            raise ValueError("input_dim_edges must be positive")
        if num_pre_processor_layers <= 0:
            raise ValueError("num_pre_processor_layers must be positive")
        if num_post_processor_layers <= 0:
            raise ValueError("num_post_processor_layers must be positive")
        if attention_type not in {"GALE_FA", "GALE_FPP"}:
            raise ValueError(
                "MeshGeoFLARE requires a FLARE-family GeoTransolver backend; "
                "expected attention_type='GALE_FA' or 'GALE_FPP'"
            )
        if mesh_context_fusion not in {"none", "post", "pre_post"}:
            raise ValueError(
                "mesh_context_fusion must be 'none', 'post', or 'pre_post'; "
                f"got {mesh_context_fusion!r}"
            )
        if mesh_context_fusion != "none" and geometry_dim is None:
            raise ValueError(
                "geometry_dim is required when mesh context fusion is enabled"
            )
        if mesh_context_use_global and mesh_context_fusion == "none":
            raise ValueError(
                "mesh_context_use_global=True requires mesh context fusion"
            )
        if mesh_context_use_global and global_dim is None:
            raise ValueError("global_dim is required for global mesh-context fusion")
        if local_processor not in {"mpnn", "knn_attention", "knn_mpnn"}:
            raise ValueError(
                "local_processor must be 'mpnn', 'knn_attention' or 'knn_mpnn'; "
                f"got {local_processor!r}"
            )
        if local_neighborhood not in {"geodesic", "euclidean"}:
            raise ValueError(
                "local_neighborhood must be 'geodesic' or 'euclidean'; "
                f"got {local_neighborhood!r}"
            )
        if (
            isinstance(local_knn_k, bool)
            or type(local_knn_k) is not int
            or local_knn_k <= 0
        ):
            raise ValueError("local_knn_k must be a positive integer")
        if local_num_heads is not None and (
            isinstance(local_num_heads, bool)
            or type(local_num_heads) is not int
            or local_num_heads <= 0
        ):
            raise ValueError("local_num_heads must be a positive integer or None")
        if not isinstance(local_attention_fused, bool):
            raise ValueError("local_attention_fused must be a bool")
        if local_attention_backend not in {"auto", "triton", "torch"}:
            raise ValueError(
                "local_attention_backend must be 'auto', 'triton' or 'torch'; "
                f"got {local_attention_backend!r}"
            )

        if guard_config is not None:
            raise ValueError(
                "Embedded guard_config is no longer supported. Use the external "
                "GeoTransolver OOD guard API to attach a guard to this model."
            )
        super().__init__(
            functional_dim=functional_dim,
            out_dim=out_dim,
            geometry_dim=geometry_dim,
            global_dim=global_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=act,
            mlp_ratio=mlp_ratio,
            slice_num=slice_num,
            use_te=use_te,
            time_input=time_input,
            plus=plus,
            include_local_features=include_local_features,
            radii=radii,
            neighbors_in_radius=neighbors_in_radius,
            n_hidden_local=n_hidden_local,
            attention_type=attention_type,
            concrete_dropout=concrete_dropout,
            state_mixing_mode=state_mixing_mode,
            attn_scale=attn_scale,
            activation_checkpointing=activation_checkpointing,
            checkpointing_ratio=checkpointing_ratio,
            activation_checkpointing_components=activation_checkpointing_components,
        )
        self.__name__ = "MeshGeoFLARE"

        mesh_hidden_dim = n_hidden if mesh_hidden_dim is None else mesh_hidden_dim
        if mesh_hidden_dim <= 0:
            raise ValueError("mesh_hidden_dim must be positive")

        self.input_dim_nodes = functional_dim
        self.input_dim_edges = input_dim_edges
        self.output_dim = out_dim
        self.hidden_dim = n_hidden
        self.num_attention_layers = n_layers
        self.geometry_dim = geometry_dim
        self.global_dim = global_dim
        self.mesh_hidden_dim = mesh_hidden_dim
        self.num_pre_processor_layers = num_pre_processor_layers
        self.num_post_processor_layers = num_post_processor_layers
        self.mesh_pre_residual_gate = nn.Parameter(
            torch.tensor(float(mesh_pre_residual_gate_init))
        )
        self.mesh_post_residual_gate = nn.Parameter(
            torch.tensor(float(mesh_post_residual_gate_init))
        )
        self.mesh_context_fusion = mesh_context_fusion
        self.mesh_context_use_global = mesh_context_use_global
        self.use_contact = use_contact
        self.contact_dim = contact_dim
        self.checkpoint_contact = bool(checkpoint_contact)

        self.local_processor = local_processor
        self.local_knn_k = int(local_knn_k)
        self.local_neighborhood = local_neighborhood
        self.local_num_heads = n_head if local_num_heads is None else local_num_heads
        self.local_attention_fused = local_attention_fused
        self.local_attention_backend = local_attention_backend
        if local_processor == "knn_attention" and (
            mesh_hidden_dim % self.local_num_heads != 0
        ):
            raise ValueError(
                "mesh_hidden_dim must be divisible by local_num_heads; got "
                f"{mesh_hidden_dim} and {self.local_num_heads}"
            )

        mesh_activation_fn = get_activation(mesh_activation)
        # The structural edge encoder only serves the MeshGraphNet processors.
        self.edge_encoder = (
            MeshGraphMLP(
                input_dim_edges,
                output_dim=mesh_hidden_dim,
                hidden_dim=mesh_hidden_dim,
                hidden_layers=num_layers_edge_encoder,
                activation_fn=mesh_activation_fn,
                norm_type=norm_type,
                recompute_activation=recompute_activation,
            )
            if local_processor in ("mpnn", "knn_mpnn")
            else None
        )
        # The pre-MPNN consumes GeoTransolver's encoded node latent directly.
        # Optional linear adapters retain support for a mesh width that differs
        # from n_hidden without ever projecting through functional_dim.
        self.pre_latent_input = (
            nn.Identity()
            if mesh_hidden_dim == n_hidden
            else nn.Linear(n_hidden, mesh_hidden_dim)
        )
        self.pre_latent_output = (
            nn.Identity()
            if mesh_hidden_dim == n_hidden
            else nn.Linear(mesh_hidden_dim, n_hidden, bias=False)
        )
        self.post_node_encoder = MeshGraphMLP(
            out_dim,
            output_dim=mesh_hidden_dim,
            hidden_dim=mesh_hidden_dim,
            hidden_layers=num_layers_node_encoder,
            activation_fn=mesh_activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )
        self.post_node_decoder = MeshGraphMLP(
            mesh_hidden_dim,
            output_dim=out_dim,
            hidden_dim=mesh_hidden_dim,
            hidden_layers=num_layers_node_decoder,
            activation_fn=mesh_activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )

        if local_processor in ("mpnn", "knn_mpnn"):
            # "knn_mpnn": the same MeshGraphNet processors, run on the static
            # kNN slots (every valid slot is one directed edge neighbor -> node,
            # the self slot included) instead of the finite-element edges.
            processor_kwargs = dict(
                input_dim_node=mesh_hidden_dim,
                input_dim_edge=mesh_hidden_dim,
                num_layers_node=num_layers_node_processor,
                num_layers_edge=num_layers_edge_processor,
                aggregation=aggregation,
                norm_type=norm_type,
                activation_fn=mesh_activation_fn,
                do_concat_trick=do_concat_trick,
                checkpoint_offloading=checkpoint_offloading,
            )
            self._local_processor_kwargs = processor_kwargs
            self.pre_processor = self.make_local_processor(
                num_pre_processor_layers, num_pre_processor_checkpoint_segments
            )
            self.post_processor = self.make_local_processor(
                num_post_processor_layers, num_post_processor_checkpoint_segments
            )
        else:
            # Static reference-neighborhood attention in place of both MPNNs.
            # Checkpoint segments keep their meaning: contiguous block groups
            # recomputed during backward, optionally offloaded to host memory.
            local_kwargs = dict(
                hidden_dim=mesh_hidden_dim,
                num_heads=self.local_num_heads,
                edge_input_dim=input_dim_edges,
                edge_dim=local_edge_dim,
                mlp_ratio=local_mlp_ratio,
                activation=local_activation,
                checkpoint_offloading=checkpoint_offloading,
                fused_attention=local_attention_fused,
                fused_backend=local_attention_backend,
                value_norm=local_value_norm,
                difference_values=local_difference_values,
                update=local_update,
                bias_fp32=local_bias_fp32,
                split_key_value=local_split_key_value,
                dropout=local_dropout,
                concrete_dropout=local_concrete_dropout,
            )
            self._local_processor_kwargs = local_kwargs
            self.pre_processor = self.make_local_processor(
                num_pre_processor_layers, num_pre_processor_checkpoint_segments
            )
            self.post_processor = self.make_local_processor(
                num_post_processor_layers, num_post_processor_checkpoint_segments
            )
        # Keep all shared backbone/context tensors and the outer RNG stream
        # identical to no-contact initialization. Legacy checkpoints retain their
        # old constructor behavior unless this versioned option is enabled.
        contact_rng_state = torch.get_rng_state() if contact_isolate_rng else None
        try:
            if contact_rng_state is not None:
                import hashlib

                # A dedicated deterministic substream also avoids reusing the
                # same random values for contact and the following context MLP.
                seed = int.from_bytes(
                    hashlib.blake2b(
                        contact_rng_state.numpy().tobytes(),
                        digest_size=8,
                        person=b"deformer-contact",
                    ).digest(),
                    "little",
                )
                torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            self.contact_block = (
                SparseContactBlock(
                    hidden_dim=mesh_hidden_dim,
                    contact_dim=contact_dim,
                    gate_init=contact_gate_init,
                    aggregation=contact_aggregation,
                )
                if use_contact
                else None
            )
        finally:
            if contact_rng_state is not None:
                torch.set_rng_state(contact_rng_state)
        if mesh_context_fusion == "none":
            self.mesh_context_encoder = None
            self.pre_context_film = None
            self.post_context_film = None
        else:
            self.mesh_context_encoder = MeshGeometryContextEncoder(
                geometry_dim=geometry_dim,
                geometry_token_dim=n_hidden // n_head,
                hidden_dim=mesh_hidden_dim,
                global_dim=global_dim if mesh_context_use_global else None,
                activation=mesh_activation,
            )
            self.pre_context_film = (
                MeshContextFiLM(mesh_hidden_dim)
                if mesh_context_fusion == "pre_post"
                else None
            )
            self.post_context_film = MeshContextFiLM(mesh_hidden_dim)

    @staticmethod
    def _resolve_alias(
        value: int | None,
        alias: int | None,
        value_name: str,
        alias_name: str,
        default: int | None = None,
    ) -> int | None:
        if value is None:
            return alias if alias is not None else default
        if alias is not None and value != alias:
            raise ValueError(
                f"{value_name}={value} conflicts with {alias_name}={alias}"
            )
        return value

    def forward(
        self,
        node_features: Float[torch.Tensor, "num_nodes input_dim_nodes"],
        edge_features: Float[torch.Tensor, "num_edges input_dim_edges"] | None,
        graph: GraphType,
        geometry: Float[torch.Tensor, "*geometry_nodes geometry_dim"] | None = None,
        global_embedding: (
            Float[torch.Tensor, "*global_shape global_dim"] | None
        ) = None,
        local_positions: (
            Float[torch.Tensor, "*position_nodes spatial_dim"] | None
        ) = None,
        contact_graph: ContactGraph | None = None,
        return_backbone: bool = False,
    ) -> Float[torch.Tensor, "num_nodes output_dim"] | BackboneState:
        r"""Apply latent pre-MPNN, FLARE backbone, and output-space post-MPNN.

        With ``return_backbone=True`` the full-width FLARE output and the
        processor inputs are returned instead (see :class:`BackboneState`).
        """

        if not torch.compiler.is_compiling():
            if (
                node_features.ndim != 2
                or node_features.shape[1] != self.input_dim_nodes
            ):
                raise ValueError(
                    f"Expected node_features [N, {self.input_dim_nodes}], got "
                    f"{tuple(node_features.shape)}"
                )
            if self.local_processor == "mpnn":
                if (
                    edge_features is None
                    or edge_features.ndim != 2
                    or edge_features.shape[1] != self.input_dim_edges
                ):
                    raise ValueError(
                        f"Expected edge_features [E, {self.input_dim_edges}], got "
                        f"{None if edge_features is None else tuple(edge_features.shape)}"
                    )
                if int(graph.num_edges) != edge_features.shape[0]:
                    raise ValueError(
                        f"Graph has {int(graph.num_edges)} edges but edge_features has "
                        f"{edge_features.shape[0]} rows"
                    )
            if int(graph.num_nodes) != node_features.shape[0]:
                raise ValueError(
                    f"Graph has {int(graph.num_nodes)} nodes but node_features has "
                    f"{node_features.shape[0]} rows"
                )

        batch, num_graphs = MeshAttentionHybrid._batch_index(
            graph, node_features.shape[0], node_features.device
        )
        geometry = MeshAttentionHybrid._flatten_geometry(
            geometry, node_features.shape[0], num_graphs
        )
        local_positions = MeshAttentionHybrid._flatten_geometry(
            local_positions, node_features.shape[0], num_graphs
        )
        if local_positions is None:
            local_positions = geometry
        global_embedding = MeshAttentionHybrid._normalize_global_embedding(
            global_embedding, num_graphs
        )

        if self.geometry_dim is not None:
            if geometry is None:
                raise ValueError("geometry is required when geometry_dim is configured")
            if geometry.shape[-1] != self.geometry_dim:
                raise ValueError(
                    f"Expected geometry dimension {self.geometry_dim}, got "
                    f"{geometry.shape[-1]}"
                )
        if self.global_dim is not None:
            if global_embedding is None:
                raise ValueError(
                    "global_embedding is required when global_dim is configured"
                )
            if global_embedding.shape[-1] != self.global_dim:
                raise ValueError(
                    f"Expected global dimension {self.global_dim}, got "
                    f"{global_embedding.shape[-1]}"
                )

        per_graph_inputs = []
        live_geometry_contexts = []
        for graph_idx in range(num_graphs):
            node_index = torch.nonzero(batch == graph_idx, as_tuple=False).flatten()
            geometry_i = (
                None
                if geometry is None
                else geometry.index_select(0, node_index).unsqueeze(0)
            )
            positions_i = (
                None
                if local_positions is None
                else local_positions.index_select(0, node_index).unsqueeze(0)
            )
            global_i = (
                None
                if global_embedding is None
                else global_embedding[graph_idx : graph_idx + 1]
            )
            local_i = node_features.index_select(0, node_index).unsqueeze(0)
            context_state = self._run_checkpointed_component(
                "context",
                self.context_builder.build_context,
                (local_i,),
                None if positions_i is None else (positions_i,),
                geometry_i,
                global_i,
                self.mesh_context_encoder is None,
            )
            if self.mesh_context_encoder is not None:
                if context_state[2] is None:
                    raise ValueError(
                        "Mesh context fusion requires live GALE geometry context"
                    )
                live_geometry_contexts.append(context_state[2])
            per_graph_inputs.append(
                (node_index, geometry_i, positions_i, global_i, context_state)
            )

        mesh_context = None
        if self.mesh_context_encoder is not None:
            if geometry is None:
                raise ValueError("geometry is required for mesh context fusion")
            mesh_context = self.mesh_context_encoder(
                geometry=geometry,
                batch=batch,
                geometry_context=torch.cat(live_geometry_contexts, dim=0),
                global_embedding=(
                    global_embedding if self.mesh_context_use_global else None
                ),
            )

        processor_graph = graph
        if self.local_processor == "mpnn":
            edge_latent = self.edge_encoder(edge_features)
            neighborhood = None
        elif self.local_processor == "knn_mpnn":
            neighborhood = self._local_neighborhood(graph, node_features.shape[0])
            processor_graph, slot_features = self._knn_slot_graph(neighborhood)
            edge_latent = self.edge_encoder(slot_features.to(node_features.dtype))
            neighborhood = None
        else:
            edge_latent = None
            neighborhood = self._local_neighborhood(graph, node_features.shape[0])
        backbone_latent = self._run_checkpointed_component(
            "preprocess", self.preprocess[0], node_features
        )
        pre_seed = self.pre_latent_input(backbone_latent)
        pre_latent = pre_seed
        if self.pre_context_film is not None:
            if mesh_context is None:
                raise RuntimeError("pre-context FiLM has no mesh context")
            pre_latent = self.pre_context_film(pre_latent, mesh_context)
        pre_latent = self._run_local_processor(
            self.pre_processor, pre_latent, edge_latent, processor_graph, neighborhood
        )
        # MeshGraphNetProcessor returns an updated latent state. Gate only its
        # change relative to the encoded seed so alpha=0 is exact GeoFLARE and
        # alpha=1 is the complete pre-MPNN update when widths match.
        mesh_latent = (
            backbone_latent
            + self.mesh_pre_residual_gate
            * self.pre_latent_output(pre_latent - pre_seed)
        )
        if self.contact_block is not None:
            if contact_graph is None:
                raise ValueError("contact_graph is required when use_contact=True")
            if self.checkpoint_contact and self.training and torch.is_grad_enabled():
                # Pass every differentiable feature explicitly. Nested
                # checkpointing keeps contact's large gathered/message tensors
                # out of the outer rollout recomputation's saved activations.
                contact_delta = checkpoint(
                    self._contact_residual,
                    pre_latent,
                    contact_graph.edge_index,
                    contact_graph.edge_features,
                    contact_graph.obstacle_mask,
                    contact_graph.edge_weights,
                    contact_graph.source_nodes,
                    contact_graph.source_weights,
                    use_reentrant=False,
                )
            else:
                contact_delta = self.contact_block(
                    pre_latent, contact_graph, return_correction=True
                )
            # Contact has its own gate. Do not multiply it by the structural
            # pre-MPNN gate, which starts at zero and would suppress its training.
            mesh_latent = mesh_latent + self.pre_latent_output(contact_delta)
        elif contact_graph is not None:
            raise ValueError("contact_graph was provided but use_contact=False")

        core_output = None
        point_features = None
        for (
            node_index,
            geometry_i,
            positions_i,
            global_i,
            context_state,
        ) in per_graph_inputs:
            embedding_states, local_features, _ = context_state
            outputs_i, features_i = self._process_preprocessed_embeddings(
                [mesh_latent.index_select(0, node_index).unsqueeze(0)],
                embedding_states,
                local_features,
                return_point_features=True,
            )
            output_i = outputs_i[0].squeeze(0)
            if core_output is None:
                # Autocast may make the GeoFLARE output fp16/bf16 while the
                # datapipe input remains fp32. Allocate from the actual core
                # output so index_copy never mixes dtypes.
                core_output = output_i.new_zeros(
                    (node_features.shape[0], self.output_dim)
                )
            core_output = core_output.index_copy(0, node_index, output_i)
            if return_backbone:
                features_i = features_i[0].squeeze(0)
                if point_features is None:
                    point_features = features_i.new_zeros(
                        (node_features.shape[0], features_i.shape[-1])
                    )
                point_features = point_features.index_copy(0, node_index, features_i)

        if core_output is None:
            raise ValueError("The graph batch contains no nodes")
        if return_backbone:
            return BackboneState(
                point_features, edge_latent, processor_graph, neighborhood, mesh_context
            )
        post_latent = self.post_node_encoder(core_output)
        if self.post_context_film is not None:
            if mesh_context is None:
                raise RuntimeError("post-context FiLM has no mesh context")
            post_latent = self.post_context_film(post_latent, mesh_context)
        post_latent = self._run_local_processor(
            self.post_processor, post_latent, edge_latent, processor_graph, neighborhood
        )
        return core_output + self.mesh_post_residual_gate * self.post_node_decoder(
            post_latent
        )

    def _local_neighborhood(
        self, graph: GraphType, num_nodes: int
    ) -> LocalNeighborhood:
        """Read and check the datapipe's static reference neighborhoods."""
        index = getattr(graph, "knn_index", None)
        mask = getattr(graph, "knn_mask", None)
        edge_attr = getattr(graph, "knn_edge_attr", None)
        if index is None or mask is None or edge_attr is None:
            raise ValueError(
                f"local_processor={self.local_processor!r} requires graph.knn_index, "
                "graph.knn_mask and graph.knn_edge_attr (enable datapipe.local_knn_k)"
            )
        metric = getattr(graph, "knn_metric", None)
        metrics = (
            set(metric)
            if isinstance(metric, (list, tuple))
            else ({metric} if metric is not None else set())
        )
        if metrics and metrics != {self.local_neighborhood}:
            raise ValueError(
                f"graph neighborhoods were built with metric {sorted(metrics)} but "
                f"the model expects local_neighborhood={self.local_neighborhood!r}"
            )
        if index.ndim != 2 or index.shape[1] != self.local_knn_k + 1:
            raise ValueError(
                f"graph.knn_index must hold local_knn_k={self.local_knn_k} neighbors "
                f"plus the node itself ({self.local_knn_k + 1} slots); got "
                f"{tuple(index.shape)}"
            )
        neighborhood = LocalNeighborhood(index, mask, edge_attr)
        neighborhood.validate(num_nodes, self.input_dim_edges)
        return neighborhood

    def make_local_processor(
        self, num_layers: int, checkpoint_segments: int = 0
    ) -> nn.Module:
        """A fresh local processor of the configured kind, ``num_layers`` deep.

        Used for the pre and post stages at construction and by recipes that
        add further stages of the same operator (for example a deeper
        time-conditional refinement stage).
        """
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.local_processor in ("mpnn", "knn_mpnn"):
            return MeshGraphNetProcessor(
                processor_size=num_layers,
                num_processor_checkpoint_segments=checkpoint_segments,
                **self._local_processor_kwargs,
            )
        return NeighborhoodAttentionProcessor(
            num_layers=num_layers,
            checkpoint_segments=min(checkpoint_segments, num_layers),
            **self._local_processor_kwargs,
        )

    def _run_local_processor(
        self,
        processor: nn.Module,
        latent: torch.Tensor,
        edge_latent: torch.Tensor | None,
        graph: GraphType,
        neighborhood: LocalNeighborhood | None,
    ) -> torch.Tensor:
        if self.local_processor in ("mpnn", "knn_mpnn"):
            return processor(latent, edge_latent, graph)
        return processor(latent, neighborhood)

    @staticmethod
    def _knn_slot_graph(neighborhood: LocalNeighborhood):
        """Directed edges neighbor -> node for every valid slot, with slot features.

        The self slot (zero displacement before standardization) becomes a self
        loop, mirroring the self loops of the structural graph. Padded slots are
        dropped. Returns a PyG graph carrying only connectivity, and the raw
        standardized slot features ``[E, edge_input_dim]`` in slot order.
        """
        index, mask = neighborhood.index, neighborhood.mask
        num_nodes = index.shape[0]
        destination = (
            torch.arange(num_nodes, device=index.device).unsqueeze(1).expand_as(index)
        )
        edge_index = torch.stack((index[mask], destination[mask]), dim=0)
        slot_features = neighborhood.edge_features[mask]
        if not PYG_AVAILABLE:
            raise ImportError(
                "The 'knn_mpnn' local processor builds a PyTorch Geometric graph; "
                "install torch_geometric and torch_scatter (physicsnemo[gnns])."
            )
        return PyGData(edge_index=edge_index, num_nodes=num_nodes), slot_features

    def _contact_residual(
        self,
        node_latent: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        obstacle_mask: torch.Tensor,
        edge_weights: torch.Tensor | None,
        source_nodes: torch.Tensor | None = None,
        source_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pure tensor-input contact stage for nested activation checkpointing."""
        return self.contact_block(
            node_latent,
            ContactGraph(
                edge_index,
                edge_features,
                obstacle_mask,
                edge_weights,
                source_nodes,
                source_weights,
            ),
            return_correction=True,
        )
