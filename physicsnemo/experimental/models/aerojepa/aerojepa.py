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

"""Top-level AeroJEPA model.

:class:`AeroJEPA` composes an :class:`AeroJEPATrunk` (context encoder +
target encoder + field decoder) and a :class:`PrototypeTokenJEPAHead`
(JEPA predictor) into a single :class:`physicsnemo.core.module.Module`.
Inputs are the geometry context, operating conditions, and query
positions; output is the decoded field at the queries.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from .layers import TokenSet

from .predictor import PrototypeTokenJEPAHead
from .trunk import AeroJEPATrunk


@dataclass
class AeroJEPAMetaData(ModelMetaData):
    r"""Meta-data for the :class:`AeroJEPA` model.

    Attributes
    ----------
    jit : bool
        Whether ``torch.jit.script`` is supported. Default ``False``.
    cuda_graphs : bool
        Whether CUDA graphs are supported. Default ``False``.
    amp : bool
        Whether automatic mixed precision is supported. Default ``True``.
    onnx_cpu : bool
        Whether ONNX export to CPU is supported. Default ``False``.
    onnx_gpu : bool
        Whether ONNX export to GPU is supported. Default ``False``.
    onnx_runtime : bool
        Whether ONNX runtime is supported. Default ``False``.
    var_dim : int
        Variable dimension for physics-informed features. Default ``1``.
    func_torch : bool
        Whether ``torch.func`` is used. Default ``False``.
    auto_grad : bool
        Whether automatic differentiation through the forward pass is
        used by the model. Default ``False``.
    """

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


class AeroJEPA(Module):
    r"""AeroJEPA: Joint-Embedding Predictive Architecture for 3D aerodynamics.

    The model composes a context encoder, a target encoder, a field
    decoder (collectively the :class:`AeroJEPATrunk`), and a JEPA
    predictor head (:class:`PrototypeTokenJEPAHead`). The primary
    inference path is :meth:`forward`:

    1. Encode the geometry context into context tokens and a
       decoder-side global conditioning vector.
    2. Derive the target-token coordinates from the same context
       positions by running the target encoder's tokenizer with a
       placeholder feature tensor (the
       :meth:`build_target_token_coords` helper).
    3. Run the predictor over the context tokens at those target
       positions, conditioned on the operating conditions.
    4. Wrap the predicted features into a :class:`TokenSet` and decode
       the field at the supplied query positions.

    For training-time use, callers run the encoders and decoder
    separately via :meth:`encode_geometry_and_flow` and
    :meth:`decode_field` (or use the underlying :attr:`trunk`
    directly), then compute losses on intermediate outputs.

    Operating conditions enter the model only at the predictor head and
    at the decoder via ``cond_global``; the context and target encoders
    are not conditioned on them.

    Parameters
    ----------
    trunk : AeroJEPATrunk
        The encoder/decoder trunk.
    predictor : PrototypeTokenJEPAHead
        The JEPA predictor head.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.aerojepa.encoders.context import ContextTransformer
    >>> from physicsnemo.experimental.models.aerojepa.encoders.target import TargetTransformer
    >>> from physicsnemo.experimental.models.aerojepa.decoder import QueryTokenDecoder
    >>> from physicsnemo.experimental.models.aerojepa.predictor import PrototypeTokenJEPAHead
    >>> from physicsnemo.experimental.models.aerojepa.trunk import AeroJEPATrunk
    >>> from physicsnemo.experimental.models.aerojepa.aerojepa import AeroJEPA
    >>> _ = torch.manual_seed(0)
    >>> enc_kwargs = dict(
    ...     point_input_dim=3, token_dim=32, max_point_tokens=16,
    ...     tokenizer_strategy='fps', tokenizer_knn_chunk_size=32,
    ...     point_pos_pe_bands=4, num_heads=4, num_layers=2, neighbor_k=4,
    ...     mlp_ratio=2, dropout=0.0,
    ... )
    >>> trunk = AeroJEPATrunk(
    ...     context_encoder=ContextTransformer(**enc_kwargs),
    ...     target_encoder=TargetTransformer(**enc_kwargs),
    ...     decoder=QueryTokenDecoder(
    ...         token_dim=32, hidden_dim=64, num_layers=2, out_dim=4,
    ...         use_sdf=True, cond_dim=4, pe_num_bands=4,
    ...         cross_attention_heads=4, cross_attention_layers=1,
    ...         cross_attention_k=4, query_chunk_size=128,
    ...     ),
    ...     include_geometry_global_in_decoder_cond=False,
    ... )
    >>> predictor = PrototypeTokenJEPAHead(
    ...     token_dim=32, cond_dim=4, depth=2, num_heads=4,
    ...     neighbor_k=4, knn_chunk_size=32, query_pe_bands=4,
    ...     mlp_ratio=2, dropout=0.0,
    ... )
    >>> model = AeroJEPA(trunk=trunk, predictor=predictor).eval()
    >>> field = model(
    ...     context_pos=torch.randn(40, 3),
    ...     context_feat=torch.zeros(40, 0),
    ...     gen_params=torch.randn(4),
    ...     query_pos=torch.randn(30, 3),
    ...     query_sdf=torch.randn(30, 1),
    ... )
    >>> field.shape
    torch.Size([30, 4])
    """

    def __init__(
        self,
        *,
        trunk: AeroJEPATrunk,
        predictor: PrototypeTokenJEPAHead,
    ):
        super().__init__(meta=AeroJEPAMetaData())
        self.trunk = trunk
        self.predictor = predictor

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def context_encoder(self):
        r"""The trunk's context encoder."""
        return self.trunk.context_encoder

    @property
    def target_encoder(self):
        r"""The trunk's target encoder."""
        return self.trunk.target_encoder

    @property
    def decoder(self):
        r"""The trunk's field decoder."""
        return self.trunk.decoder

    @property
    def include_geometry_global_in_decoder_cond(self) -> bool:
        r"""Whether the decoder-side ``cond_global`` includes the context global token."""
        return self.trunk.include_geometry_global_in_decoder_cond

    @property
    def mask_head(self):
        r"""The trunk's optional per-query mask head (``None`` when disabled)."""
        return self.trunk.mask_head

    @property
    def mask_head_use_cond(self) -> bool:
        r"""Whether the mask head concatenates the conditioning vector to its input."""
        return self.trunk.mask_head_use_cond

    # ------------------------------------------------------------------ #
    # Forward-pass API
    # ------------------------------------------------------------------ #

    def encode_geometry(
        self,
        *,
        context_pos: Float[torch.Tensor, "N D_pos"],
        context_feat: Float[torch.Tensor, "N D_feat"],
        gen_params: Float[torch.Tensor, "G"],
    ) -> tuple[TokenSet, torch.Tensor]:
        r"""Encode the geometry context, returning context tokens and ``cond_global``.

        Does not run the target encoder. ``cond_global`` is the
        decoder-side global conditioning vector — it is just
        ``gen_params`` unless the trunk was built with
        ``include_geometry_global_in_decoder_cond=True``, in which case
        the context-global token is concatenated.

        Parameters
        ----------
        context_pos : torch.Tensor
            Context positions of shape ``(N, D_pos)``.
        context_feat : torch.Tensor
            Per-point context features of shape ``(N, D_feat)``.
        gen_params : torch.Tensor
            Operating conditions of shape ``(G,)``.

        Returns
        -------
        context_tokens : TokenSet
            Context tokens emitted by the context encoder.
        cond_global : torch.Tensor
            Flat decoder-side conditioning vector.
        """
        context_out = self.trunk.context_encoder(
            context_pos=context_pos,
            context_feat=context_feat,
        )
        cond_global = gen_params.unsqueeze(0)
        if self.trunk.include_geometry_global_in_decoder_cond:
            context_global = context_out.global_token
            if context_global is None:
                context_global = context_out.tokens.global_token
            if context_global is not None:
                cond_global = torch.cat(
                    [cond_global, context_global.reshape(1, -1)], dim=-1
                )
        return context_out.tokens, cond_global.squeeze(0)

    def encode_geometry_and_flow(
        self,
        *,
        context_pos: torch.Tensor,
        context_feat: torch.Tensor,
        target_surface_pos: torch.Tensor,
        target_surface_main_feat: torch.Tensor,
        target_volume_pos: torch.Tensor,
        target_volume_feat: torch.Tensor,
        gen_params: torch.Tensor,
    ) -> dict[str, object]:
        r"""Run both encoders (training-time path).

        Returns the dict the decoder expects:
        ``{"context_tokens", "target_tokens", "cond_global"}``. See
        :meth:`AeroJEPATrunk.encode_context` for parameter semantics.
        """
        return self.trunk.encode_context(
            context_pos=context_pos,
            context_feat=context_feat,
            target_surface_pos=target_surface_pos,
            target_surface_main_feat=target_surface_main_feat,
            target_volume_pos=target_volume_pos,
            target_volume_feat=target_volume_feat,
            gen_params=gen_params,
        )

    def predict_field_tokens(
        self,
        *,
        context_tokens: TokenSet,
        target_positions: torch.Tensor,
        conditions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Run the JEPA predictor head.

        Parameters
        ----------
        context_tokens : TokenSet
            Context tokens from :meth:`encode_geometry` /
            :meth:`encode_geometry_and_flow`.
        target_positions : torch.Tensor
            Target-token positions of shape ``(T, 3)`` or
            ``(B, T, 3)``. Broadcast to the context batch when its
            leading dim is 1.
        conditions : torch.Tensor, optional
            Conditioning vector forwarded to the predictor.

        Returns
        -------
        torch.Tensor
            Predicted target-token features.
        """
        return self.predictor(
            context_tokens=context_tokens,
            target_positions=target_positions,
            cond=conditions,
        )

    def decode_field(
        self,
        *,
        target_tokens: TokenSet,
        cond_global: torch.Tensor,
        query_pos: Float[torch.Tensor, "Nq D_pos"],
        query_sdf: Float[torch.Tensor, "Nq 1"] | None = None,
        return_mask_logits: bool = False,
    ):
        r"""Decode a target token set at the supplied query positions.

        Thin wrapper around :meth:`AeroJEPATrunk.decode_queries`. When
        ``return_mask_logits`` is ``True`` and the mask head is
        configured, returns ``(pred, mask_logits)``; otherwise returns
        ``pred`` only.
        """
        context = {"target_tokens": target_tokens, "cond_global": cond_global}
        return self.trunk.decode_queries(
            context=context,
            query_pos=query_pos,
            query_sdf=query_sdf,
            return_mask_logits=return_mask_logits,
        )

    @torch.no_grad()
    def decode_field_chunked(
        self,
        *,
        target_tokens: TokenSet,
        cond_global: torch.Tensor,
        query_pos: torch.Tensor,
        query_sdf: torch.Tensor,
        chunk_size: int,
        precision: str = "fp32",
    ) -> torch.Tensor:
        r"""Chunked decode for very large query sets with autocast precision control.

        Splits ``query_pos`` (and ``query_sdf``) into chunks of size
        ``chunk_size``, decodes each chunk under the requested autocast
        precision, and moves the chunk's output to CPU before
        concatenation. Returns a CPU tensor so callers don't have to
        manage device transfers.

        Unlike the decoder's built-in ``query_chunk_size`` (which
        chunks internally but keeps everything on GPU), this method is
        useful when the query set is large enough that even the
        accumulated chunk outputs would exceed VRAM.

        Parameters
        ----------
        target_tokens : TokenSet
            Target tokens produced by the predictor (or by the target
            encoder at training).
        cond_global : torch.Tensor
            Flat decoder-side conditioning vector from
            :meth:`encode_geometry`.
        query_pos : torch.Tensor
            Query positions of shape ``(Nq, 3)``. May be on any device;
            chunks are moved to the model's device on demand.
        query_sdf : torch.Tensor
            Per-query SDF of shape ``(Nq, 1)``.
        chunk_size : int
            Maximum number of queries decoded per chunk.
        precision : str, optional
            ``"fp32"``, ``"fp16"``, or ``"bf16"``. Anything other than
            ``"fp32"`` enables ``torch.autocast`` for the chunk's
            decode. Default ``"fp32"``.

        Returns
        -------
        torch.Tensor
            Decoded field of shape ``(Nq, C)`` on CPU.
        """
        device = next(self.parameters()).device
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        autocast_dtype = dtype_map.get(precision, torch.float32)
        enabled = precision in {"fp16", "bf16"}

        context = {"target_tokens": target_tokens, "cond_global": cond_global}
        preds = []
        n = int(query_pos.shape[0])
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=enabled
        ):
            for st in range(0, n, max(1, int(chunk_size))):
                en = min(st + int(chunk_size), n)
                pred_chunk = self.trunk.decode_queries(
                    context=context,
                    query_pos=query_pos[st:en].to(device),
                    query_sdf=query_sdf[st:en].to(device),
                )
                preds.append(pred_chunk.detach().float().cpu())
        return torch.cat(preds, dim=0)

    def build_target_token_coords(
        self,
        *,
        point_positions: torch.Tensor,
    ) -> torch.Tensor:
        r"""Recover the spatial coordinates the target encoder would emit.

        Runs the target encoder's tokenizer on a placeholder
        zero-feature tensor with the supplied ``point_positions`` and
        returns just the resulting token coordinates. The tokenizer's
        center-selection strategies look only at positions, so the
        placeholder features are harmless.

        Used by :meth:`forward` to derive target-token positions
        internally — callers performing inference do not need to supply
        them. Also exposed publicly so optimization-loop callers can
        cache one set of coordinates and reuse it across many predictor
        evaluations.

        Parameters
        ----------
        point_positions : torch.Tensor
            Positions to feed into the tokenizer of shape ``(N, 3)``.

        Returns
        -------
        torch.Tensor
            Target-token coordinates of shape ``(T, 3)``.

        Raises
        ------
        ValueError
            If the target encoder does not expose the expected
            transformer tokenization path.
        """
        target_encoder = self.trunk.target_encoder
        base_encoder = getattr(target_encoder, "encoder", None)
        if base_encoder is None or not hasattr(base_encoder, "_tokenize_single"):
            raise ValueError(
                "Target encoder does not expose the transformer tokenization path."
            )
        feat_dim = int(base_encoder.feature_in.in_features)
        dummy_features = point_positions.new_zeros(
            (int(point_positions.shape[0]), feat_dim)
        )
        token_coords, _ = base_encoder._tokenize_single(  # noqa: SLF001
            point_positions=point_positions,
            point_features=dummy_features,
        )
        return token_coords

    def forward(
        self,
        *,
        context_pos: Float[torch.Tensor, "N D_pos"],
        context_feat: Float[torch.Tensor, "N D_feat"],
        gen_params: Float[torch.Tensor, "G"],
        query_pos: Float[torch.Tensor, "Nq D_pos"],
        query_sdf: Float[torch.Tensor, "Nq 1"] | None = None,
        conditions: Float[torch.Tensor, "B Cond"] | None = None,
    ) -> Float[torch.Tensor, "Nq C"]:
        if not torch.compiler.is_compiling():
            if context_pos.ndim != 2 or context_feat.ndim != 2:
                raise ValueError(
                    "context_pos and context_feat must be rank 2; "
                    f"got {tuple(context_pos.shape)} and {tuple(context_feat.shape)}."
                )
            if int(context_pos.shape[0]) != int(context_feat.shape[0]):
                raise ValueError(
                    "context_pos and context_feat must agree on the point count; "
                    f"got {int(context_pos.shape[0])} vs {int(context_feat.shape[0])}."
                )

        context_tokens, cond_global = self.encode_geometry(
            context_pos=context_pos,
            context_feat=context_feat,
            gen_params=gen_params,
        )
        target_coords = self.build_target_token_coords(point_positions=context_pos)
        if conditions is None:
            conditions = gen_params
            if conditions.ndim == 1:
                conditions = conditions.unsqueeze(0)
        predicted_features = self.predict_field_tokens(
            context_tokens=context_tokens,
            target_positions=target_coords,
            conditions=conditions,
        )
        if predicted_features.ndim == 3 and int(predicted_features.shape[0]) == 1:
            predicted_features = predicted_features[0]
        target_mask = torch.ones(
            (int(target_coords.shape[0]),),
            device=target_coords.device,
            dtype=torch.bool,
        )
        target_tokens = TokenSet(
            features=predicted_features,
            coords=target_coords,
            mask=target_mask,
            global_token=None,
        )
        return self.decode_field(
            target_tokens=target_tokens,
            cond_global=cond_global,
            query_pos=query_pos,
            query_sdf=query_sdf,
        )

    @torch.no_grad()
    def predict(
        self,
        *,
        context_pos: Float[torch.Tensor, "N D_pos"],
        context_feat: Float[torch.Tensor, "N D_feat"],
        gen_params: Float[torch.Tensor, "G"],
        query_pos: Float[torch.Tensor, "Nq D_pos"],
        query_sdf: Float[torch.Tensor, "Nq 1"] | None = None,
        conditions: Float[torch.Tensor, "B Cond"] | None = None,
    ) -> Float[torch.Tensor, "Nq C"]:
        r"""``@torch.no_grad()`` convenience wrapper around :meth:`forward`."""
        return self.forward(
            context_pos=context_pos,
            context_feat=context_feat,
            gen_params=gen_params,
            query_pos=query_pos,
            query_sdf=query_sdf,
            conditions=conditions,
        )
