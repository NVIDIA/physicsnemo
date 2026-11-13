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

r"""DeFormer: the network of the DeFormer crash surrogate.

A static kNN-attention processor refines the native GeoTransolver
node latent, geometry-aware FLARE communicates globally, and a second
kNN-attention processor refines the physical outputs. The crash
recipe selects point-space FLARE (``GALE_FA``) and geometry conditioning before
and after attention.

The design is inspired by the mesh-attention hybrids of ``arXiv:2605.11784``;
DeFormer's latent integration is an independent extension, not a claim of
exact reproduction.
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch.utils.checkpoint import checkpoint

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.models.geotransolver import GeoTransolver
from physicsnemo.nn import get_activation
from physicsnemo.nn.module.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from physicsnemo.nn.module.gnn_layers.utils import GraphType

from .local_attention import KNNAttentionProcessor, LocalNeighborhood

MeshContextFusionStage = Literal["none", "post", "pre_post"]


class BackboneState(NamedTuple):
    """Full-width state after the FLARE blocks, before any output head.

    Returned by :meth:`DeFormer.forward` with ``return_backbone=True`` so
    recipes can attach their own post-attention stages (for example a
    latent-preserving time-conditional refinement) without a projection through
    the three-channel output.
    """

    point_features: Float[torch.Tensor, "num_nodes backbone_dim"]
    neighborhood: LocalNeighborhood
    mesh_context: Float[torch.Tensor, "num_nodes mesh_hidden_dim"] | None


LocalNeighborhoodMetric = Literal["geodesic", "euclidean"]


@dataclass
class DeFormerMetaData(ModelMetaData):
    r"""Meta data of :class:`DeFormer`.

    The model consumes a graph with static neighbourhoods and runs a Triton
    kernel on CUDA, so neither JIT, CUDA graphs nor ONNX export are supported;
    automatic mixed precision is.
    """

    name: str = "DeFormer"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class MeshGeometryContextEncoder(nn.Module):
    r"""Build node-aligned mesh context from geometry and GALE geometry tokens.

    Pointwise geometry preserves the node's location while the pooled live GALE
    geometry tokens provide a simulation-level shape summary. Optional normalized
    global parameters can be added for recipe-specific conditioning. The returned
    tensor is aligned with flattened graph nodes and can condition any mesh stage.

    Parameters
    ----------
    geometry_dim : int
        Width :math:`C_{geo}` of the pointwise geometry features.
    geometry_token_dim : int
        Width :math:`D_{tok}` of the GALE geometry tokens.
    hidden_dim : int
        Width :math:`D` of the returned mesh context.
    global_dim : int, optional
        Width :math:`C_g` of the global embedding; ``None`` disables the global
        branch. Default is ``None``.
    activation : str, optional
        Activation of the three encoder MLPs. Default is ``"gelu"``.

    Forward
    -------
    geometry : torch.Tensor
        Pointwise geometry of shape :math:`(N, C_{geo})` over the flattened
        graph batch.
    batch : torch.Tensor
        Graph index of every node, shape :math:`(N,)` (``int64``).
    geometry_context : torch.Tensor
        Live GALE geometry tokens of shape :math:`(B, H, S, D_{tok})`, one row
        per graph.
    global_embedding : torch.Tensor, optional
        Global embedding of shape :math:`(B, T, C_g)`; required when
        ``global_dim`` is set and rejected otherwise. Default is ``None``.

    Outputs
    -------
    torch.Tensor
        Node-aligned mesh context of shape :math:`(N, D)`.
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
        batch: Int[torch.Tensor, "num_nodes"],  # noqa: F821
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
    r"""Zero-initialized residual FiLM adapter for a mesh latent state.

    The adapter adds ``tanh(scale) * LayerNorm(latent) + shift`` to the latent,
    with ``scale`` and ``shift`` predicted from the context by one linear layer
    that starts at zero, so a fresh adapter is the identity.

    Parameters
    ----------
    hidden_dim : int
        Width :math:`D` of the latent and of the context.

    Forward
    -------
    latent : torch.Tensor
        Node latent of shape :math:`(N, D)`.
    context : torch.Tensor
        Node-aligned mesh context of shape :math:`(N, D)`.

    Outputs
    -------
    torch.Tensor
        Modulated latent of shape :math:`(N, D)`.
    """

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


def _batch_index(
    graph: GraphType, num_nodes: int, device: torch.device
) -> tuple[torch.Tensor, int]:
    """The per-node graph index of a (possibly batched) graph and the graph count."""
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


def _flatten_geometry(
    geometry: torch.Tensor | None,
    num_nodes: int,
    num_graphs: int,
) -> torch.Tensor | None:
    """``[N, D]`` or ``[B, N, D]`` geometry as ``[N, D]`` aligned with the nodes."""
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
            f"geometry must have shape [N, D] or [B, N, D]; got {tuple(geometry.shape)}"
        )
    if flattened.shape[0] != num_nodes:
        raise ValueError(
            "geometry must align with flattened graph nodes; "
            f"got {flattened.shape[0]} geometry rows for {num_nodes} nodes"
        )
    return flattened


def _normalize_global_embedding(
    global_embedding: torch.Tensor | None,
    num_graphs: int,
) -> torch.Tensor | None:
    """Global tokens as ``[B, tokens, D]`` with one leading entry per graph."""
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


class DeFormer(GeoTransolver):
    r"""DeFormer's network: a FLARE-family GeoTransolver with local mesh stages.

    The DeFormer recipe selects ``attention_type="GALE_FA"`` (original FLARE);
    the class also accepts FLARE++.

    The complete :class:`GeoTransolver` backbone is retained: its input
    projection, geometry/global context builder, FLARE-family blocks, local
    ball-query features, and output projection are all inherited directly. The
    pre-mesh processor operates on the native ``n_hidden`` node embedding after
    GeoTransolver's input projection and before local-feature concatenation and
    FLARE attention. A second mesh processor refines the final physical outputs.
    Setting both mesh residual gates to zero recovers the matching GeoFLARE
    function. The latent pre-processor avoids a
    decode-to-input/re-encode bottleneck. Consequently, every
    fixed-query GeoFLARE parameter is present with the same shape and
    DeFormer always has strictly more parameters for a matching configuration.

    The local operator run before and after the FLARE blocks is a
    :class:`~physicsnemo.experimental.models.deformer.local_attention.KNNAttentionProcessor`
    stack over static reference neighborhoods. It needs the datapipe to attach
    ``knn_index`` (``[N, local_knn_k + 1]`` including the node itself),
    ``knn_mask``, ``knn_edge_attr`` (``[N, local_knn_k + 1, input_dim_edges]``)
    and optionally ``knn_metric`` to the graph; ``local_neighborhood`` names the
    metric those neighborhoods were built with and is checked against
    ``knn_metric``. With both residual gates at zero the model is exactly
    GeoFLARE.

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
        Width of the per-slot kNN edge features (reference displacement and
        distance give four). Default is 4.
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
        Activation of the post-processor node encoder and decoder and of the
        mesh context encoder. Default is ``"relu"``.
    num_layers_node_encoder : int, optional
        Hidden layers of the post-processor node encoder. Default is 2.
    num_layers_node_decoder : int, optional
        Hidden layers of the post-processor node decoder. Default is 2.
    norm_type : {"LayerNorm", "TELayerNorm"}, optional
        Normalization of the mesh MLPs. Default is ``"LayerNorm"``. As in
        :class:`~physicsnemo.models.meshgraphnet.MeshGraphNet`, the LayerNorm
        implementation itself comes from
        :func:`physicsnemo.nn.module.layer_norm.get_layer_norm_class`:
        Transformer Engine's when it is installed and a GPU is present, PyTorch's
        otherwise, with the ``PHYSICSNEMO_FORCE_TE`` environment variable as the
        override (``False`` for a model that runs on the CPU).
    num_pre_processor_checkpoint_segments, num_post_processor_checkpoint_segments : int, optional
        Activation-checkpoint segments of the pre- and post-processor; 0
        disables checkpointing. Default is 0.
    checkpoint_offloading : bool, optional
        Offload the local processors' checkpointed activations to host memory.
        Default is ``False``.
    recompute_activation : bool, optional
        Recompute the mesh encoder/decoder MLP activations during backward
        instead of storing them; :class:`MeshGraphMLP` supports this for
        ``mesh_activation="silu"`` only. Default is ``False``.
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
    local_knn_k : int, optional
        Neighbors per node, excluding the node's own slot. Default is 32.
    local_neighborhood : {"geodesic", "euclidean"}, optional
        Metric the kNN neighborhoods were built with. Default is ``"geodesic"``.
    local_num_heads : int, optional
        Attention heads of the local attention; must divide ``mesh_hidden_dim``.
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
        :mod:`~physicsnemo.experimental.models.deformer.fused_knn_attention`
        (same math, no gathered ``[N, K, H, d_h]`` key/value tensors). Default
        is ``True``.
    local_attention_backend : {"auto", "triton", "torch"}, optional
        Backend of the fused kernel; ``"auto"`` picks Triton on CUDA. Default
        is ``"auto"``.
    local_dropout, local_concrete_dropout
        Forwarded to every
        :class:`~physicsnemo.experimental.models.deformer.local_attention.KNNAttentionBlock`
        as ``dropout`` and ``concrete_dropout``.
    output_head : bool, optional
        Build the output path: GeoTransolver's output projection, the post
        node encoder, the post-processor with its FiLM, the node decoder and
        the post gate. With ``False`` only the backbone and the pre-processor
        are built, :meth:`forward` requires ``return_backbone=True`` and
        recipes attach their own stages with :meth:`make_local_processor`,
        :meth:`make_node_decoder` and :class:`MeshContextFiLM`, so that no
        unused parameter is left in the model. Default is ``True``.

    Forward
    -------
    node_features : torch.Tensor
        Node inputs of shape :math:`(N, D_{in})` over the flattened graph batch
        (``functional_dim`` channels).
    graph : GraphType
        Graph carrying the static neighborhoods: ``knn_index`` of shape
        :math:`(N, K + 1)` (``local_knn_k`` neighbors plus the node itself),
        ``knn_mask`` of shape :math:`(N, K + 1)`, ``knn_edge_attr`` of shape
        :math:`(N, K + 1, D_e)` and optionally ``knn_metric``; a batch of
        graphs carries ``batch`` of shape :math:`(N,)`.
    geometry : torch.Tensor, optional
        Geometry features of shape :math:`(N, C_{geo})` or
        :math:`(B, N_b, C_{geo})`; required when ``geometry_dim`` is set.
        Default is ``None``.
    global_embedding : torch.Tensor, optional
        Global parameters of shape :math:`(B, T, C_g)`, :math:`(B, C_g)` for one
        token per graph, or ``(T, C_g)`` / ``(C_g,)`` for a single graph;
        required when ``global_dim`` is set.
        Default is ``None``.
    local_positions : torch.Tensor, optional
        Positions for the ball-query local features, shape :math:`(N, 3)` or
        :math:`(B, N_b, 3)`; defaults to ``geometry``. Default is ``None``.
    return_backbone : bool, optional
        Return the full-width backbone state instead of the outputs. Default
        is ``False``.

    Outputs
    -------
    torch.Tensor or BackboneState
        Node outputs of shape :math:`(N, D_{out})`; with
        ``return_backbone=True`` a :class:`BackboneState` whose
        ``point_features`` have shape :math:`(N, D_{bb})` (``n_hidden`` plus
        the local-feature width), whose ``neighborhood`` is the
        :class:`~physicsnemo.experimental.models.deformer.local_attention.LocalNeighborhood`
        of the graph (checked unless compiling) and whose ``mesh_context`` has shape :math:`(N, D_{mesh})`
        or is ``None`` without mesh-context fusion.
    """

    def __init__(
        self,
        functional_dim: int,
        out_dim: int,
        input_dim_edges: int = 4,
        geometry_dim: int | None = 3,
        global_dim: int | None = None,
        n_layers: int = 6,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
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
        attention_type: str = "GALE_FPP",
        concrete_dropout: bool = False,
        state_mixing_mode: str = "weighted",
        num_pre_processor_layers: int = 1,
        num_post_processor_layers: int = 2,
        mesh_hidden_dim: int | None = None,
        mesh_activation: str = "relu",
        num_layers_node_encoder: int = 2,
        num_layers_node_decoder: int = 2,
        norm_type: Literal["LayerNorm", "TELayerNorm"] = "LayerNorm",
        num_pre_processor_checkpoint_segments: int = 0,
        num_post_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
        recompute_activation: bool = False,
        mesh_pre_residual_gate_init: float = 1.0,
        mesh_post_residual_gate_init: float = 1.0,
        mesh_context_fusion: MeshContextFusionStage = "none",
        mesh_context_use_global: bool = False,
        activation_checkpointing: bool = False,
        checkpointing_ratio: float = 1.0,
        activation_checkpointing_components: tuple[str, ...] | list[str] = ("blocks",),
        local_knn_k: int = 32,
        local_neighborhood: LocalNeighborhoodMetric = "geodesic",
        local_num_heads: int | None = None,
        local_mlp_ratio: int = 2,
        local_edge_dim: int = 32,
        local_activation: str = "gelu",
        local_attention_fused: bool = True,
        local_attention_backend: str = "auto",
        local_dropout: float = 0.0,
        local_concrete_dropout: bool = False,
        output_head: bool = True,
    ) -> None:
        if input_dim_edges <= 0:
            raise ValueError("input_dim_edges must be positive")
        if num_pre_processor_layers <= 0:
            raise ValueError("num_pre_processor_layers must be positive")
        if num_post_processor_layers <= 0:
            raise ValueError("num_post_processor_layers must be positive")
        if attention_type not in {"GALE_FA", "GALE_FPP"}:
            raise ValueError(
                "DeFormer requires a FLARE-family GeoTransolver backend; "
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
        if not isinstance(output_head, bool):
            raise ValueError("output_head must be a bool")
        if local_attention_backend not in {"auto", "triton", "torch"}:
            raise ValueError(
                "local_attention_backend must be 'auto', 'triton' or 'torch'; "
                f"got {local_attention_backend!r}"
            )

        if time_input:
            raise ValueError(
                "DeFormer.forward takes no time argument; condition on time "
                "through node_features or global_embedding (time_input=False)"
            )
        if recompute_activation and mesh_activation.lower() != "silu":
            raise ValueError(
                "recompute_activation requires mesh_activation='silu' (the only "
                f"activation MeshGraphMLP recomputes); got {mesh_activation!r}"
            )
        # ``mesh_context`` is DeFormer's own checkpoint component: the mesh
        # context encoder is not one of GeoTransolver's.
        components = tuple(activation_checkpointing_components)
        checkpoint_mesh_context = "mesh_context" in components
        backbone_components = tuple(c for c in components if c != "mesh_context")
        if not backbone_components:
            raise ValueError(
                "activation_checkpointing_components must name at least one "
                "GeoTransolver component besides 'mesh_context'"
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
            activation_checkpointing=activation_checkpointing,
            checkpointing_ratio=checkpointing_ratio,
            activation_checkpointing_components=backbone_components,
        )
        self.__name__ = "DeFormer"
        self.meta = DeFormerMetaData()
        self._checkpoint_mesh_context = checkpoint_mesh_context
        if checkpoint_mesh_context:
            self._args["__args__"]["activation_checkpointing_components"] = tuple(
                sorted((*self._activation_checkpointing_components, "mesh_context"))
            )

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
        self.num_pre_processor_checkpoint_segments = (
            num_pre_processor_checkpoint_segments
        )
        self.num_post_processor_checkpoint_segments = (
            num_post_processor_checkpoint_segments
        )
        self.mesh_post_residual_gate_init = float(mesh_post_residual_gate_init)
        self.output_head = output_head
        # Width of the backbone state (hidden plus the local-feature width),
        # read from the output projection GeoTransolver built.
        self.backbone_width = int(self.ln_mlp_out[0][0].normalized_shape[0])
        self.mesh_pre_residual_gate = nn.Parameter(
            torch.tensor(float(mesh_pre_residual_gate_init))
        )
        if output_head:
            self.mesh_post_residual_gate = nn.Parameter(
                torch.tensor(float(mesh_post_residual_gate_init))
            )
        else:
            self.register_parameter("mesh_post_residual_gate", None)
            # The backbone's output projection is never used without a head.
            self.ln_mlp_out = None
        self.mesh_context_fusion = mesh_context_fusion
        self.mesh_context_use_global = mesh_context_use_global

        self.local_knn_k = int(local_knn_k)
        self.local_neighborhood = local_neighborhood
        self.local_num_heads = n_head if local_num_heads is None else local_num_heads
        self.local_attention_fused = local_attention_fused
        self.local_attention_backend = local_attention_backend
        if mesh_hidden_dim % self.local_num_heads != 0:
            raise ValueError(
                "mesh_hidden_dim must be divisible by local_num_heads; got "
                f"{mesh_hidden_dim} and {self.local_num_heads}"
            )

        mesh_activation_fn = get_activation(mesh_activation)
        self._node_decoder_kwargs = dict(
            output_dim=out_dim,
            hidden_dim=mesh_hidden_dim,
            hidden_layers=num_layers_node_decoder,
            activation_fn=mesh_activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )
        # The pre-processor consumes GeoTransolver's encoded node latent directly.
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
        if output_head:
            self.post_node_encoder = MeshGraphMLP(
                out_dim,
                output_dim=mesh_hidden_dim,
                hidden_dim=mesh_hidden_dim,
                hidden_layers=num_layers_node_encoder,
                activation_fn=mesh_activation_fn,
                norm_type=norm_type,
                recompute_activation=recompute_activation,
            )
            self.post_node_decoder = self.make_node_decoder()
        else:
            self.post_node_encoder = None
            self.post_node_decoder = None

        # Static kNN attention over the reference neighborhoods before and after the backbone.
        # Checkpoint segments are contiguous block groups recomputed during
        # backward, optionally offloaded to host memory.
        self._local_processor_kwargs = dict(
            hidden_dim=mesh_hidden_dim,
            num_heads=self.local_num_heads,
            edge_input_dim=input_dim_edges,
            edge_dim=local_edge_dim,
            mlp_ratio=local_mlp_ratio,
            activation=local_activation,
            checkpoint_offloading=checkpoint_offloading,
            fused_attention=local_attention_fused,
            fused_backend=local_attention_backend,
            dropout=local_dropout,
            concrete_dropout=local_concrete_dropout,
        )
        self.pre_processor = self.make_local_processor(
            num_pre_processor_layers, num_pre_processor_checkpoint_segments
        )
        self.post_processor = (
            self.make_local_processor(
                num_post_processor_layers, num_post_processor_checkpoint_segments
            )
            if output_head
            else None
        )
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
            self.post_context_film = (
                MeshContextFiLM(mesh_hidden_dim) if output_head else None
            )

    def forward(
        self,
        node_features: Float[torch.Tensor, "num_nodes input_dim_nodes"],
        graph: GraphType,
        geometry: Float[torch.Tensor, "*geometry_nodes geometry_dim"] | None = None,
        global_embedding: (
            Float[torch.Tensor, "*global_shape global_dim"] | None
        ) = None,
        local_positions: (
            Float[torch.Tensor, "*position_nodes spatial_dim"] | None
        ) = None,
        return_backbone: bool = False,
    ) -> Float[torch.Tensor, "num_nodes output_dim"] | BackboneState:
        r"""Apply the latent pre-processor, the FLARE backbone and the post-processor.

        With ``return_backbone=True`` the full-width FLARE output and the
        processor inputs are returned instead (see :class:`BackboneState`).
        """

        if not self.output_head and not return_backbone:
            raise ValueError(
                "this DeFormer was built with output_head=False and has no output "
                "path; call forward(..., return_backbone=True)"
            )
        if not torch.compiler.is_compiling():
            if (
                node_features.ndim != 2
                or node_features.shape[1] != self.input_dim_nodes
            ):
                raise ValueError(
                    f"Expected node_features [N, {self.input_dim_nodes}], got "
                    f"{tuple(node_features.shape)}"
                )
            if int(graph.num_nodes) != node_features.shape[0]:
                raise ValueError(
                    f"Graph has {int(graph.num_nodes)} nodes but node_features has "
                    f"{node_features.shape[0]} rows"
                )

        batch, num_graphs = _batch_index(
            graph, node_features.shape[0], node_features.device
        )
        geometry = _flatten_geometry(geometry, node_features.shape[0], num_graphs)
        local_positions = _flatten_geometry(
            local_positions, node_features.shape[0], num_graphs
        )
        if local_positions is None:
            local_positions = geometry
        global_embedding = _normalize_global_embedding(global_embedding, num_graphs)

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

        neighborhood = self._local_neighborhood(graph, node_features.shape[0])
        if num_graphs > 1 and not torch.compiler.is_compiling():
            # Neighbourhoods of a batch must have been offset per graph (see
            # LocalNeighborhoodData); a cross-graph ID is a batching error.
            neighbor_graph = batch[neighborhood.index]
            crossing = neighborhood.mask & (neighbor_graph != batch[:, None])
            if bool(crossing.any()):
                raise ValueError(
                    "graph.knn_index refers to nodes of another graph: collate "
                    "with LocalNeighborhoodData (or offset knn_index by the node "
                    "counts) when batching"
                )

        per_graph_inputs = []
        live_geometry_contexts = []
        for graph_idx in range(num_graphs):
            if num_graphs == 1:
                # ``nonzero`` is a dynamic-shape op that splits a compiled
                # forward; one graph needs no selection.
                node_index = torch.arange(
                    node_features.shape[0], device=node_features.device
                )
            else:
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
            context_state = self._build_context(
                (local_i,),
                None if positions_i is None else (positions_i,),
                geometry_i,
                global_i,
                detach_geometry_context=self.mesh_context_encoder is None,
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
            mesh_context = self._encode_mesh_context(
                geometry,
                batch,
                torch.cat(live_geometry_contexts, dim=0),
                global_embedding if self.mesh_context_use_global else None,
            )

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
            self.pre_processor, pre_latent, neighborhood
        )
        # The local processor returns an updated latent state. Gate only its
        # change relative to the encoded seed so alpha=0 is exact GeoFLARE and
        # alpha=1 is the complete pre-processor update when widths match.
        mesh_latent = (
            backbone_latent
            + self.mesh_pre_residual_gate
            * self.pre_latent_output(pre_latent - pre_seed)
        )

        core_output = None
        point_features = None
        for (
            node_index,
            _geometry,
            _positions,
            _global,
            context_state,
        ) in per_graph_inputs:
            embedding_states, local_features, _ = context_state
            outputs_i, features_i = self._process_preprocessed_embeddings(
                [mesh_latent.index_select(0, node_index).unsqueeze(0)],
                embedding_states,
                local_features,
                return_point_features=True,
                compute_outputs=self.output_head,
            )
            if outputs_i is not None:
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

        if core_output is None and point_features is None:
            raise ValueError("The graph batch contains no nodes")
        if return_backbone:
            return BackboneState(point_features, neighborhood, mesh_context)
        post_latent = self.post_node_encoder(core_output)
        if self.post_context_film is not None:
            if mesh_context is None:
                raise RuntimeError("post-context FiLM has no mesh context")
            post_latent = self.post_context_film(post_latent, mesh_context)
        post_latent = self._run_local_processor(
            self.post_processor, post_latent, neighborhood
        )
        return core_output + self.mesh_post_residual_gate * self.post_node_decoder(
            post_latent
        )

    def forward_with_geometry_features(self, *args, **kwargs):
        r"""Not available: DeFormer's entry point is :meth:`forward`.

        The inherited GeoTransolver path would skip the kNN-attention
        stages and the mesh-context conditioning, so it is refused instead of
        silently returning a different model's output.
        """
        raise NotImplementedError(
            "DeFormer does not support forward_with_geometry_features; call "
            "forward(node_features, graph, geometry=..., return_backbone=...)"
        )

    def _encode_mesh_context(
        self,
        geometry: torch.Tensor,
        batch: torch.Tensor,
        geometry_context: torch.Tensor,
        global_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the mesh context encoder, checkpointed when so configured."""
        if (
            self._checkpoint_mesh_context
            and self._activation_checkpointing_enabled
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(
                self.mesh_context_encoder,
                geometry,
                batch,
                geometry_context,
                global_embedding,
                use_reentrant=False,
            )
        return self.mesh_context_encoder(
            geometry, batch, geometry_context, global_embedding
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
                "DeFormer requires graph.knn_index, graph.knn_mask and "
                "graph.knn_edge_attr: static neighbourhoods with local_knn_k "
                "neighbours per node plus the node itself (see "
                "physicsnemo.experimental.nn.geodesic_knn, or "
                "physicsnemo.nn.functional.knn for Euclidean neighbourhoods)"
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
        if not torch.compiler.is_compiling():
            neighborhood.validate(num_nodes, self.input_dim_edges)
        return neighborhood

    def make_node_decoder(self) -> nn.Module:
        """A fresh node decoder of the output head's shape (mesh width to ``out_dim``).

        Built for the model's own head and by recipes that attach their own
        head to a backbone-only model (``output_head=False``).
        """
        return MeshGraphMLP(self.mesh_hidden_dim, **self._node_decoder_kwargs)

    def make_local_processor(
        self, num_layers: int, checkpoint_segments: int = 0
    ) -> nn.Module:
        """A fresh kNN-attention processor, ``num_layers`` deep.

        Used for the pre and post stages at construction and by recipes that
        add further stages of the same operator (for example a deeper
        time-conditional refinement stage).
        """
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        return KNNAttentionProcessor(
            num_layers=num_layers,
            checkpoint_segments=min(checkpoint_segments, num_layers),
            **self._local_processor_kwargs,
        )

    def _run_local_processor(
        self,
        processor: nn.Module,
        latent: torch.Tensor,
        neighborhood: LocalNeighborhood,
    ) -> torch.Tensor:
        return processor(latent, neighborhood)
