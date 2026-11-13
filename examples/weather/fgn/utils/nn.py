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

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet


def nested_to(
    x: torch.Tensor | Mapping | list | tuple | Any, **kwargs
) -> torch.Tensor | dict | list | Any:
    """Move tensors inside a nested structure to a device / dtype."""
    if isinstance(x, Mapping):
        return {k: nested_to(v, **kwargs) for (k, v) in x.items()}
    if isinstance(x, (list, tuple)):
        return [nested_to(v, **kwargs) for v in x]
    if not isinstance(x, torch.Tensor):
        return x
    return x.to(**kwargs)


@dataclass
class MetaData(ModelMetaData):
    """Capability flags shared by all FGN backbone classes."""

    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class FGNDiT(Module, register=True):
    r"""DiT-based backbone for FGN (arXiv:2506.10772 §2.3).

    Patchifies the 721×1440 input before the transformer, giving 16-64×
    memory reduction over full-resolution convolutions.  The latent noise
    vector ``z ~ N(0,I)^latent_dim`` conditions every layer via AdaLN-Zero
    (passed as ``condition`` to DiT), mirroring the paper's global conditional
    layer-norm.  A zero dummy timestep satisfies DiT's ``t`` argument; its
    positional embedding becomes a learned constant bias.

    Parameters
    ----------
    state_channels : int
        Number of prognostic channels :math:`C`.
    history_frames : int, optional, default=2
        Number of past frames :math:`T` concatenated as input.
    background_channels : int, optional, default=0
        Slowly-varying background channels (e.g. SST).
    invariant_channels : int, optional, default=0
        Static invariant channels (e.g. orography, land-sea mask).
    latent_dim : int, optional, default=32
        Dimension of :math:`z` (paper §2.3 uses 32).
    input_height, input_width : int
        Spatial dimensions of the input grid (default 721×1440 for 0.25° ERA5).
    patch_size : int or (int, int), optional, default=(4, 4)
        Spatial patch size.  ``(4, 4)`` → 181×360 = 65 k tokens (16× compression).
    hidden_size : int, optional, default=384
        Transformer hidden dimension.
    depth : int, optional, default=12
        Number of transformer layers.
    num_heads : int, optional, default=8
        Number of attention heads.
    attention_backend : str, optional, default="timm"
        DiT attention backend.  ``"natten2d_rope"`` enables axial 2D RoPE and
        NATTEN windowed attention (requires ``natten`` installed); it also
        disables the learned pos_embed so no pre-padding is needed.
    detokenizer : str, optional, default="proj_reshape_2d_conv"
        Detokenizer variant.  ``"proj_reshape_2d_conv"`` adds a zero-init
        residual conv head after unprojection to suppress checkerboard
        artifacts on spiky channels (e.g. precipitation, vertical velocity).

    Forward
    -------
    history : torch.Tensor
        History state of shape :math:`(B, T, C, H, W)`.
    latent : torch.Tensor
        Noise latent of shape :math:`(B, latent\_dim)`.
    background : torch.Tensor, optional
        Background channels of shape :math:`(B, C_{bg}, H, W)`.
    invariants : torch.Tensor, optional
        Static invariant channels of shape :math:`(B, C_{inv}, H, W)`.

    Outputs
    -------
    torch.Tensor
        Predicted next state of shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from utils.nn import FGNDiT
    >>> model = FGNDiT(state_channels=3, history_frames=2, latent_dim=4,
    ...                input_height=8, input_width=16,
    ...                patch_size=(2, 2), hidden_size=16, depth=2, num_heads=2,
    ...                attention_backend="timm")
    >>> history = torch.randn(1, 2, 3, 8, 16)
    >>> latent = torch.randn(1, 4)
    >>> out = model(history, latent)
    >>> out.shape
    torch.Size([1, 3, 8, 16])
    """

    def __init__(
        self,
        state_channels: int,
        history_frames: int = 2,
        background_channels: int = 0,
        invariant_channels: int = 0,
        latent_dim: int = 32,
        input_height: int = 721,
        input_width: int = 1440,
        patch_size: int | tuple[int, int] = (4, 4),
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 8,
        attention_backend: Literal[
            "timm", "transformer_engine", "natten2d", "natten2d_rope"
        ] = "timm",
        detokenizer: Literal[
            "proj_reshape_2d", "proj_reshape_2d_conv", "hpx_patch_detokenizer"
        ] = "proj_reshape_2d_conv",
    ):
        from physicsnemo.models.dit import DiT

        super().__init__(meta=MetaData())
        self.state_channels = state_channels
        self.history_frames = history_frames
        self.background_channels = background_channels
        self.invariant_channels = invariant_channels
        self.latent_dim = latent_dim

        in_channels = (
            history_frames * state_channels + background_channels + invariant_channels
        )
        ps = (
            tuple(patch_size)
            if isinstance(patch_size, (list, tuple))
            else (patch_size, patch_size)
        )
        # natten2d_rope forces pos_embed="none" inside DiT, so PatchEmbed2D's own
        # internal padding handles non-divisible grids correctly and we don't need
        # to pre-pad.  For all other backends the learned pos_embed is allocated with
        # floor(H/ps) tokens but PatchEmbed2D pads to ceil(H/ps) at runtime, causing
        # a token-count mismatch (ERA5 721 % 4 == 1).  Pre-pad so DiT always receives
        # a divisible input and never triggers that mismatch.
        use_rope = attention_backend == "natten2d_rope"
        pad_h = 0 if use_rope else (-input_height) % ps[0]
        pad_w = 0 if use_rope else (-input_width) % ps[1]
        self._pad_top = pad_h // 2
        self._pad_bottom = pad_h - self._pad_top
        self._pad_left = pad_w // 2
        self._pad_right = pad_w - self._pad_left
        self._crop_h = input_height
        self._crop_w = input_width
        self.backbone = DiT(
            input_size=(input_height + pad_h, input_width + pad_w),
            in_channels=in_channels,
            out_channels=state_channels,
            patch_size=ps,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            condition_dim=latent_dim,
            conditioning_embedder="dit",
            attention_backend=attention_backend,
            detokenizer=detokenizer,
        )

    def forward(
        self,
        history: Float[torch.Tensor, "B T C H W"],
        latent: Float[torch.Tensor, "B latent_dim"],
        background: Float[torch.Tensor, "B C_bg H W"] | None = None,
        invariants: Float[torch.Tensor, "B C_inv H W"] | None = None,
    ) -> Float[torch.Tensor, "B C H W"]:
        r"""Run a forward pass of the FGN DiT backbone."""
        if not torch.compiler.is_compiling():
            if history.ndim != 5:
                raise ValueError(
                    f"Expected history of shape (B, T, C, H, W), "
                    f"got {tuple(history.shape)}"
                )
            if (
                history.shape[1] != self.history_frames
                or history.shape[2] != self.state_channels
            ):
                raise ValueError(
                    f"Expected history frames={self.history_frames}, "
                    f"channels={self.state_channels}, got shape {tuple(history.shape)}"
                )

        batch, frames, channels, height, width = history.shape
        pieces = [history.reshape(batch, frames * channels, height, width)]
        if background is not None:
            pieces.append(background)
        if invariants is not None:
            pieces.append(invariants)
        x = torch.cat(pieces, dim=1)

        if self._pad_top or self._pad_bottom or self._pad_left or self._pad_right:
            x = F.pad(
                x, (self._pad_left, self._pad_right, self._pad_top, self._pad_bottom)
            )

        # t=0 dummy: timestep embedding becomes a learned constant bias;
        # all stochasticity comes from latent z via AdaLN-Zero.
        t = torch.zeros(batch, device=x.device, dtype=torch.float32)
        out = self.backbone(x, t, condition=latent)
        return out[
            ...,
            self._pad_top : self._pad_top + self._crop_h,
            self._pad_left : self._pad_left + self._crop_w,
        ]


class FGNGraphCast(Module, register=True):
    r"""Grid-Mesh-Grid GNN backbone for FGN (arXiv:2506.10772 §2.3).

    Wraps PhysicsNeMo's ``GraphCastNet`` (icosahedral grid-mesh-grid GNN) with
    FGN's stochastic latent conditioning.  The latent vector
    ``z ~ N(0,I)^latent_dim`` is broadcast to shape
    :math:`(B, latent\_dim, H, W)` and concatenated with the other input
    channels before the encoder, making every grid node aware of the global
    noise draw.

    Parameters
    ----------
    state_channels : int
        Number of prognostic state channels :math:`C`.
    history_frames : int, optional, default=2
        Number of past frames :math:`T` concatenated as input.
    background_channels : int, optional, default=0
        Slowly-varying background channels (e.g. SST).
    invariant_channels : int, optional, default=0
        Static invariant channels (e.g. orography, land-sea mask).
    latent_dim : int, optional, default=32
        Dimension of the noise latent :math:`z`.
    input_res : tuple[int, int], optional, default=(721, 1440)
        Spatial :math:`(H, W)` resolution of the lat-lon grid.
    mesh_level : int, optional, default=6
        Icosahedral mesh refinement level (6 → ≈40 km resolution).
    hidden_dim : int, optional, default=768
        Hidden dimension for all GNN layers.
    processor_layers : int, optional, default=16
        Number of message-passing or transformer processor layers.
    processor_type : str, optional, default="MessagePassing"
        ``"MessagePassing"`` for GNN-MP or ``"GraphTransformer"`` for
        attention-based mesh processing.
    num_attention_heads : int, optional, default=4
        Attention heads (only used when ``processor_type="GraphTransformer"``).

    Forward
    -------
    history : torch.Tensor
        History state of shape :math:`(B, T, C, H, W)`.
    latent : torch.Tensor
        Noise latent of shape :math:`(B, latent\_dim)`.
    background : torch.Tensor, optional
        Background channels of shape :math:`(B, C_{bg}, H, W)`.
    invariants : torch.Tensor, optional
        Static invariant channels of shape :math:`(B, C_{inv}, H, W)`.

    Outputs
    -------
    torch.Tensor
        Predicted next state of shape :math:`(B, C, H, W)`.

    Examples
    --------
    >>> import torch
    >>> from utils.nn import FGNGraphCast
    >>> model = FGNGraphCast(state_channels=3, history_frames=2, latent_dim=4,
    ...                      input_res=(4, 8), mesh_level=3,
    ...                      hidden_dim=16, processor_layers=3)
    >>> history = torch.randn(1, 2, 3, 4, 8)
    >>> latent = torch.randn(1, 4)
    >>> out = model(history, latent)
    >>> out.shape
    torch.Size([1, 3, 4, 8])
    """

    def __init__(
        self,
        state_channels: int,
        history_frames: int = 2,
        background_channels: int = 0,
        invariant_channels: int = 0,
        latent_dim: int = 32,
        input_res: tuple[int, int] = (721, 1440),
        mesh_level: int = 6,
        hidden_dim: int = 768,
        processor_layers: int = 16,
        processor_type: Literal[
            "MessagePassing", "GraphTransformer"
        ] = "MessagePassing",
        num_attention_heads: int = 4,
    ):
        super().__init__(meta=MetaData())
        self.state_channels = state_channels
        self.history_frames = history_frames
        self.background_channels = background_channels
        self.invariant_channels = invariant_channels
        self.latent_dim = latent_dim
        self.input_res = input_res

        input_dim = (
            history_frames * state_channels
            + background_channels
            + invariant_channels
            + latent_dim
        )
        self.backbone = GraphCastNet(
            mesh_level=mesh_level,
            input_res=input_res,
            input_dim_grid_nodes=input_dim,
            output_dim_grid_nodes=state_channels,
            hidden_dim=hidden_dim,
            processor_layers=processor_layers,
            processor_type=processor_type,
            num_attention_heads=num_attention_heads,
        )

    def forward(
        self,
        history: Float[torch.Tensor, "B T C H W"],
        latent: Float[torch.Tensor, "B latent_dim"],
        background: Float[torch.Tensor, "B C_bg H W"] | None = None,
        invariants: Float[torch.Tensor, "B C_inv H W"] | None = None,
    ) -> Float[torch.Tensor, "B C H W"]:
        r"""Run a forward pass of the FGN GraphCast backbone."""
        if not torch.compiler.is_compiling():
            if history.ndim != 5:
                raise ValueError(
                    f"Expected history of shape (B, T, C, H, W), "
                    f"got {tuple(history.shape)}"
                )

        batch, frames, channels, height, width = history.shape
        pieces = [history.reshape(batch, frames * channels, height, width)]
        if background is not None:
            pieces.append(background)
        if invariants is not None:
            pieces.append(invariants)
        # Broadcast z to every grid node so each location sees the global noise.
        z_spatial = latent.unsqueeze(-1).unsqueeze(-1).expand(batch, -1, height, width)
        pieces.append(z_spatial)
        x = torch.cat(pieces, dim=1)  # (B, C_in, H, W)

        return self.backbone(x)


def build_model(
    cfg,
    state_channels: int,
    background_channels: int,
    invariant_channels: int,
    input_height: int = 721,
    input_width: int = 1440,
) -> FGNDiT | FGNGraphCast:
    """Instantiate the FGN backbone specified by ``cfg.model.backbone``.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra config with a ``model`` sub-config.
    state_channels : int
        Number of prognostic state channels.
    background_channels : int
        Number of slowly-varying background channels.
    invariant_channels : int
        Number of static invariant channels.
    input_height : int, optional, default=721
        Spatial height of the input grid.
    input_width : int, optional, default=1440
        Spatial width of the input grid.

    Returns
    -------
    FGNDiT or FGNGraphCast
        Constructed model.
    """
    if cfg.model.background_channels not in ("auto", background_channels):
        raise ValueError("config model.background_channels disagrees with dataset")
    if cfg.model.invariant_channels not in ("auto", invariant_channels):
        raise ValueError("config model.invariant_channels disagrees with dataset")

    backbone = str(getattr(cfg.model, "backbone", "dit")).lower()

    if backbone == "graphcast":
        return FGNGraphCast(
            state_channels=state_channels,
            history_frames=int(cfg.model.history_frames),
            background_channels=background_channels,
            invariant_channels=invariant_channels,
            latent_dim=int(cfg.model.latent_dim),
            input_res=(input_height, input_width),
            mesh_level=int(getattr(cfg.model, "mesh_level", 6)),
            hidden_dim=int(getattr(cfg.model, "hidden_dim", 768)),
            processor_layers=int(getattr(cfg.model, "processor_layers", 16)),
            processor_type=str(getattr(cfg.model, "processor_type", "MessagePassing")),
            num_attention_heads=int(getattr(cfg.model, "num_attention_heads", 4)),
        )

    ps: tuple[int, int] = (
        tuple(cfg.model.patch_size) if hasattr(cfg.model, "patch_size") else (4, 4)
    )  # type: ignore[assignment]
    return FGNDiT(
        state_channels=state_channels,
        history_frames=int(cfg.model.history_frames),
        background_channels=background_channels,
        invariant_channels=invariant_channels,
        latent_dim=int(cfg.model.latent_dim),
        input_height=input_height,
        input_width=input_width,
        patch_size=ps,
        hidden_size=int(cfg.model.hidden_size),
        depth=int(cfg.model.depth),
        num_heads=int(cfg.model.num_heads),
        attention_backend=cfg.model.attention_backend,  # type: ignore[arg-type]
        detokenizer=cfg.model.detokenizer,  # type: ignore[arg-type]
    )
