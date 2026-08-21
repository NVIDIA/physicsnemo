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

r"""GeoTransolver: Geometry-Aware Physics Attention Transformer.

This module provides the GeoTransolver model, which extends the Transolver architecture
with GALE (Geometry-Aware Latent Embeddings) attention for incorporating geometric
structure and global context throughout the forward pass.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from jaxtyping import Float

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.core.version_check import OptionalImport
from physicsnemo.models.transolver.transolver import _TransolverMlp
from physicsnemo.models.utils.activation_checkpointing import (
    resolve_checkpointing_ratio,
)
from physicsnemo.nn import GALEBlock

from .activation_checkpointing import (
    DEFAULT_CHECKPOINTING_COMPONENTS,
    checkpoint_block,
    parse_checkpointing_components,
    run_checkpointed_component,
    should_checkpoint_block,
    should_checkpoint_component,
)
from .context_projector import GlobalContextBuilder

te = OptionalImport("transformer_engine.pytorch")


@dataclass
class GeoTransolverMetaData(ModelMetaData):
    r"""Data class for storing essential meta data needed for the GeoTransolver model.

    Attributes
    ----------
    name : str
        Model name. Default is ``"GeoTransolver"``.
    jit : bool
        Whether JIT compilation is supported. Default is ``False``.
    cuda_graphs : bool
        Whether CUDA graphs are supported. Default is ``False``.
    amp : bool
        Whether automatic mixed precision is supported. Default is ``True``.
    onnx_cpu : bool
        Whether ONNX export to CPU is supported. Default is ``False``.
    onnx_gpu : bool
        Whether ONNX export to GPU is supported. Default is ``True``.
    onnx_runtime : bool
        Whether ONNX runtime is supported. Default is ``True``.
    var_dim : int
        Variable dimension for physics-informed features. Default is 1.
    func_torch : bool
        Whether torch functions are used. Default is ``False``.
    auto_grad : bool
        Whether automatic differentiation is used. Default is ``False``.
    """

    name: str = "GeoTransolver"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


def _normalize_dim(x: int | Sequence[int]) -> tuple[int, ...]:
    r"""Normalize dimension specification to tuple format.

    Parameters
    ----------
    x : int | Sequence[int]
        Dimension specification as scalar or sequence.

    Returns
    -------
    tuple[int, ...]
        Normalized dimension tuple.

    Raises
    ------
    TypeError
        If ``x`` is not an int or valid sequence.
    """
    # Accept int as scalar
    if isinstance(x, int):
        return (x,)
    # Accept any non-string sequence of ints
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        return tuple(int(v) for v in x)
    raise TypeError(f"Invalid dim specifier {x!r}")


def _normalize_tensor(
    x: torch.Tensor | Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    r"""Normalize tensor input to tuple format.

    Parameters
    ----------
    x : torch.Tensor | Sequence[torch.Tensor]
        Single tensor or sequence of tensors.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Normalized tensor tuple.

    Raises
    ------
    TypeError
        If ``x`` is not a tensor or valid sequence.
    """
    # Accept single tensor
    if isinstance(x, torch.Tensor):
        return (x,)
    if isinstance(x, Sequence):
        return tuple(x)
    raise TypeError("Invalid tensor structure")


def _structured_num_tokens(spatial_shape: tuple[int, ...]) -> int:
    return int(math.prod(spatial_shape))


def _flatten_for_structured(
    t: torch.Tensor,
    spatial_shape: tuple[int, ...],
    name: str,
) -> torch.Tensor:
    """Flatten (B,H,W,C) or (B,H,W,D,C) to (B,N,C); pass through (B,N,C) if N matches.

    Mirrors Transolver's structured flatten/unflatten behavior so the rest of
    GeoTransolver can assume a single token layout (B, N, C).
    """
    n = _structured_num_tokens(spatial_shape)
    if t.ndim == 3:
        if not torch.compiler.is_compiling() and t.shape[1] != n:
            raise ValueError(
                f"{name} token count {t.shape[1]} != structured grid size {n}"
            )
        return t
    if len(spatial_shape) == 2 and t.ndim == 4:
        B, H, W, C = t.shape
        if (H, W) != spatial_shape:
            raise ValueError(
                f"{name} spatial dims {(H, W)} != structured_shape {spatial_shape}"
            )
        return t.reshape(B, n, C)
    if len(spatial_shape) == 3 and t.ndim == 5:
        B, H, W, D, C = t.shape
        if (H, W, D) != spatial_shape:
            raise ValueError(
                f"{name} spatial dims {(H, W, D)} != structured_shape {spatial_shape}"
            )
        return t.reshape(B, n, C)
    raise ValueError(
        f"{name}: expected (B,N,C) with N={n}, or spatial layout matching "
        f"structured_shape {spatial_shape}; got shape {tuple(t.shape)}"
    )


class GeoTransolver(Module):
    r"""GeoTransolver: Geometry-Aware Physics Attention Transformer.

    GeoTransolver is an adaptation of the Transolver architecture, replacing standard
    attention with GALE (Geometry-Aware Latent Embeddings) attention. GALE combines
    physics-aware self-attention on learned state slices with cross-attention to
    geometry and global context embeddings.

    The model projects geometry and global features onto physical state spaces, which
    are then used as context in all transformer blocks. This design enables the model
    to incorporate geometric structure and global information throughout the forward
    pass.

    Parameters
    ----------
    functional_dim : int | tuple[int, ...]
        Dimension of the input values (local embeddings), not including global
        embeddings or geometry features. Input will be projected to ``n_hidden``
        before processing. Can be a single int or tuple for multiple input types.
    out_dim : int | tuple[int, ...]
        Dimension of the output of the model. Must have same length as
        ``functional_dim`` if both are tuples.
    geometry_dim : int | None, optional
        Pointwise dimension of the geometry input features. If provided, geometry
        features will be projected onto physical states and used as context in all
        GALE layers. Default is ``None``.
    global_dim : int | None, optional
        Dimension of the global embedding features. If provided, global features
        will be projected onto physical states and used as context in all GALE
        layers. Default is ``None``.
    n_layers : int, optional
        Number of GALE layers in the model. Default is 4.
    n_hidden : int, optional
        Hidden dimension of the transformer. Default is 256.
    dropout : float, optional
        Dropout rate applied across the GALE layers. Default is 0.0.
    n_head : int, optional
        Number of attention heads in each GALE layer. Must evenly divide
        ``n_hidden`` to yield an integer head dimension. Default is 8.
    act : str, optional
        Activation function name. Default is ``"gelu"``.
    mlp_ratio : int, optional
        Ratio of MLP hidden dimension to ``n_hidden``. Default is 4.
    slice_num : int, optional
        Number of learned physical state slices in the GALE layers, representing
        the number of learned states each layer should project inputs onto.
        Default is 32.
    use_te : bool, optional
        Whether to use Transformer Engine backend when available. Default is ``False``.
    time_input : bool, optional
        Whether to include time embeddings. Default is ``False``.
    plus : bool, optional
        Whether to use Transolver++ features in the GALE layers. Default is ``False``.
    include_local_features : bool, optional
        Whether to include local features in the global context. Default is ``False``.
    radii : list[float], optional
        Radii for the local features. Default is ``[0.05, 0.25]``.
    neighbors_in_radius : list[int], optional
        Neighbors in radius for the local features. Default is ``[8, 32]``.
    n_hidden_local : int, optional
        Hidden dimension for the local features. Default is 32.
    structured_shape : tuple[int, ...] | None, optional
        If set to ``(H, W)`` or ``(H, W, D)``, enables structured 2D/3D paths
        (Conv2d/Conv3d GALE; no ball-query local features). Inputs may be
        flattened :math:`(B, N, C)` with :math:`N = H W` or :math:`H W D`, or
        spatial :math:`(B, H, W, C)` / :math:`(B, H, W, D, C)`. Default is ``None``.
    attention_type : {"GALE", "GALE_FA"}, optional
        Attention implementation used inside each GALE block: ``"GALE"`` for the
        reference version, ``"GALE_FA"`` for the flash-attention one.  Validated
        in :class:`~physicsnemo.nn.GALEBlock`, which raises on any other value.
        Default is ``"GALE"``.
    state_mixing_mode : str, optional
        How to blend self-attention and cross-attention outputs in GALE layers.
        ``"weighted"`` uses a learnable sigmoid-gated weighted sum.
        ``"concat_project"`` concatenates the two along the head dimension and
        projects back with a linear layer. Default is ``"weighted"``.
    activation_checkpointing : bool, optional, default=False
        Whether to enable activation checkpointing during training.
    checkpointing_ratio : float, optional, default=1.0
        Fraction of GALE blocks to checkpoint when
        ``activation_checkpointing=True``. Selected blocks are distributed
        evenly across the block stack.
    activation_checkpointing_components : tuple[str, ...] | list[str], optional
        Components covered when activation checkpointing is enabled. Supported
        values are ``"context"``, ``"preprocess"``, ``"blocks"``, and
        ``"output"``. The default ``("blocks",)`` matches Transolver's
        block-only policy. ``checkpointing_ratio`` applies to the block stack;
        other selected components are either fully checkpointed or disabled.

    Forward
    -------
    local_embedding : torch.Tensor | tuple[torch.Tensor, ...]
        Local embedding: unstructured :math:`(B, N, C)`; structured 2D
        :math:`(B, H, W, C)` or flattened :math:`(B, H W, C)`; structured 3D
        :math:`(B, H, W, D, C)` or flattened. Can be a tuple for multiple input types.
    local_positions : torch.Tensor | tuple[torch.Tensor, ...] | None, optional
        Local positions for each input, each of shape :math:`(B, N, 3)`. Required if
        ``include_local_features=True``. Default is ``None``.
    global_embedding : torch.Tensor | None, optional
        Global embedding of the input data of shape :math:`(B, N_g, C_g)` where
        :math:`N_g` is number of global tokens and :math:`C_g` is ``global_dim``.
        If ``None``, global context is not used. Default is ``None``.
    geometry : torch.Tensor | None, optional
        Geometry features of the input data of shape :math:`(B, N, C_{geo})` where
        :math:`C_{geo}` is ``geometry_dim``. If ``None``, geometry context is not
        used. Default is ``None``.
    time : torch.Tensor | None, optional
        Time embedding (currently not implemented). Default is ``None``.

    Outputs
    -------
    torch.Tensor | tuple[torch.Tensor, ...]
        When ``return_embedding_states=False`` (default): output tensor(s) of
        shape :math:`(B, N, C_{out})`. Returns a single tensor if input was
        a single tensor, or a tuple of tensors if input was a tuple
        (multi-stream). For structured grids, output matches the input
        layout—flattened :math:`(B, N, C_{out})` or spatial
        :math:`(B, H, W, C_{out})` / :math:`(B, H, W, D, C_{out})` when
        inputs were 4D/5D.

        When ``return_embedding_states=True``, returns a 2-tuple
        ``(output, embedding_states)`` where ``output`` follows the same
        rules above, and ``embedding_states`` is of shape
        :math:`(B, H, S, D_c)` (geometry/global context), or ``None`` if no
        context sources were provided.

    Raises
    ------
    ValueError
        If ``n_hidden`` is not evenly divisible by ``n_head``.
    ValueError
        If ``functional_dim`` and ``out_dim`` have different lengths when both
        are tuples.
    NotImplementedError
        If ``time`` is provided (not yet implemented).

    Notes
    -----
    Unstructured mesh uses linear GALE projection; structured ``structured_shape``
    uses the same Conv2d/Conv3d slice projection as :class:`~physicsnemo.models.transolver.Transolver`.
    Ball-query local features are disabled when ``structured_shape`` is set.

    For more details on Transolver, see:

    - `Transolver paper <https://arxiv.org/pdf/2402.02366>`_
    - `Transolver++ paper <https://arxiv.org/pdf/2502.02414>`_

    See Also
    --------
    :class:`~physicsnemo.nn.module.gale.GALE` : The attention mechanism used in GeoTransolver.
    :class:`~physicsnemo.nn.module.gale.GALEBlock` : Transformer block using GALE attention.
    :class:`~physicsnemo.models.geotransolver.context_projector.ContextProjector` : Projects context features onto physical states.

    Examples
    --------
    Basic usage with local embeddings only:

    >>> import torch
    >>> from physicsnemo.models.geotransolver import GeoTransolver
    >>> model = GeoTransolver(
    ...     functional_dim=64,
    ...     out_dim=3,
    ...     n_hidden=256,
    ...     n_layers=4,
    ...     use_te=False,
    ... )
    >>> local_emb = torch.randn(2, 1000, 64)  # (batch, nodes, features)
    >>> output = model(local_emb)
    >>> output.shape
    torch.Size([2, 1000, 3])

    Usage with geometry, global context, and embedding states:

    >>> model = GeoTransolver(
    ...     functional_dim=64,
    ...     out_dim=3,
    ...     geometry_dim=3,
    ...     global_dim=16,
    ...     n_hidden=256,
    ...     n_layers=4,
    ...     use_te=False,
    ... )
    >>> local_emb = torch.randn(2, 1000, 64)
    >>> geometry = torch.randn(2, 1000, 3)  # (batch, nodes, spatial_dim)
    >>> global_emb = torch.randn(2, 1, 16)  # (batch, 1, global_features)
    >>> output = model(local_emb, global_embedding=global_emb, geometry=geometry)
    >>> output.shape
    torch.Size([2, 1000, 3])

    To also retrieve the geometry/global context embeddings:

    >>> output, emb_states = model(
    ...     local_emb,
    ...     global_embedding=global_emb,
    ...     geometry=geometry,
    ...     return_embedding_states=True,
    ... )
    >>> emb_states.shape[0] == 2  # batch dimension preserved
    True

    Structured 2D grid:

    >>> model = GeoTransolver(
    ...     functional_dim=3,
    ...     out_dim=1,
    ...     structured_shape=(8, 8),
    ...     n_hidden=64,
    ...     n_head=4,
    ...     n_layers=2,
    ...     use_te=False,
    ... )
    >>> y = model(torch.randn(2, 8, 8, 3))
    >>> y.shape
    torch.Size([2, 8, 8, 1])
    """

    def __init__(
        self,
        functional_dim: int | tuple[int, ...],
        out_dim: int | tuple[int, ...],
        geometry_dim: int | None = None,
        global_dim: int | None = None,
        n_layers: int = 4,
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
        structured_shape: tuple[int, ...] | None = None,
        attention_type: Literal["GALE", "GALE_FA"] = "GALE",
        concrete_dropout: bool = False,
        state_mixing_mode: str = "weighted",
        activation_checkpointing: bool = False,
        checkpointing_ratio: float = 1.0,
        activation_checkpointing_components: tuple[str, ...] | list[str] = ("blocks",),
    ) -> None:
        super().__init__(meta=GeoTransolverMetaData())
        self.__name__ = "GeoTransolver"

        # Set defaults for mutable arguments
        if radii is None:
            radii = [0.05, 0.25]
        if neighbors_in_radius is None:
            neighbors_in_radius = [8, 32]

        if structured_shape is not None:
            if include_local_features:
                raise ValueError(
                    "include_local_features=True is not supported with structured_shape "
                    "(ball-query path is mesh-only)."
                )
            if len(structured_shape) not in (2, 3):
                raise ValueError(
                    f"structured_shape must have length 2 or 3, got {structured_shape!r}"
                )
            if not all(int(s) > 0 for s in structured_shape):
                raise ValueError(
                    f"structured_shape must be positive ints, got {structured_shape!r}"
                )

        self.include_local_features = include_local_features
        self.use_te = use_te
        self.structured_shape = structured_shape
        self._activation_checkpointing_enabled = activation_checkpointing
        self._activation_checkpointing_ratio = resolve_checkpointing_ratio(
            activation_checkpointing, checkpointing_ratio
        )
        self._activation_checkpointing_components = parse_checkpointing_components(
            activation_checkpointing_components
        )
        # Module.__new__ captures raw constructor values before validation.
        # Normalize Hydra/OmegaConf sequences so .mdlus metadata stays JSON-safe.
        self._args["__args__"]["activation_checkpointing_components"] = tuple(
            sorted(self._activation_checkpointing_components)
        )

        # Validate head dimension compatibility
        if n_head <= 0:
            raise ValueError(f"GeoTransolver requires n_head > 0, got {n_head}")
        if n_hidden % n_head != 0:
            raise ValueError(
                f"GeoTransolver requires n_hidden % n_head == 0, "
                f"got n_hidden={n_hidden}, n_head={n_head}"
            )

        # Normalize dimension specifications to tuples
        functional_dims = _normalize_dim(functional_dim)
        out_dims = _normalize_dim(out_dim)

        # Store radii for hidden dimension calculation
        self.radii = radii if self.include_local_features else []

        # Initialize the context builder - handles all context construction
        self.context_builder = GlobalContextBuilder(
            functional_dims=functional_dims,
            geometry_dim=geometry_dim,
            global_dim=global_dim,
            radii=radii,
            neighbors_in_radius=neighbors_in_radius,
            n_hidden_local=n_hidden_local,
            n_hidden=n_hidden,
            n_head=n_head,
            dropout=dropout,
            slice_num=slice_num,
            use_te=use_te,
            plus=plus,
            include_local_features=self.include_local_features,
            structured_shape=structured_shape,
            concrete_dropout=concrete_dropout,
        )
        context_dim = self.context_builder.get_context_dim()

        # Validate dimension tuple lengths match
        if len(functional_dims) != len(out_dims):
            raise ValueError(
                f"functional_dim and out_dim must be the same length, "
                f"but instead got {len(functional_dims)} and {len(out_dims)}"
            )

        # Input projection MLPs - one per input type
        self.preprocess = nn.ModuleList(
            [
                _TransolverMlp(
                    in_features=f,
                    hidden_features=n_hidden * 2,
                    out_features=n_hidden,
                    act_layer=act,
                    use_te=use_te,
                )
                for f in functional_dims
            ]
        )

        self.n_hidden = n_hidden

        # Compute effective hidden dimension including local features
        effective_hidden = (
            n_hidden + n_hidden_local * len(self.radii)
            if self.include_local_features
            else n_hidden
        )

        # GALE transformer blocks
        self.blocks = nn.ModuleList(
            [
                GALEBlock(
                    num_heads=n_head,
                    hidden_dim=effective_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    slice_num=slice_num,
                    last_layer=(layer_idx == n_layers - 1),
                    use_te=use_te,
                    plus=plus,
                    context_dim=context_dim,
                    spatial_shape=structured_shape,
                    attention_type=attention_type,
                    concrete_dropout=concrete_dropout,
                    state_mixing_mode=state_mixing_mode,
                )
                for layer_idx in range(n_layers)
            ]
        )

        # Output projection layers - one per output type
        if use_te:
            self.ln_mlp_out = nn.ModuleList(
                [
                    te.LayerNormLinear(in_features=effective_hidden, out_features=o)
                    for o in out_dims
                ]
            )
        else:
            self.ln_mlp_out = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(effective_hidden),
                        nn.Linear(effective_hidden, o),
                    )
                    for o in out_dims
                ]
            )

        # Time embedding network (optional, not yet implemented)
        self.time_input = time_input
        if time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.SiLU(),
                nn.Linear(n_hidden, n_hidden),
            )

    def _should_checkpoint_block(self, block_idx: int) -> bool:
        r"""Return whether a GALE block should use activation checkpointing."""
        return should_checkpoint_block(
            block_idx,
            len(self.blocks),
            getattr(self, "_activation_checkpointing_ratio", 0.0),
            getattr(
                self,
                "_activation_checkpointing_components",
                DEFAULT_CHECKPOINTING_COMPONENTS,
            ),
            training=self.training,
        )

    def _should_checkpoint_component(self, component: str) -> bool:
        r"""Return whether a non-fractional component should be checkpointed."""
        return should_checkpoint_component(
            component,
            # Full-object pickles bypass ``__init__`` when loaded. Checkpoints
            # from the first checkpointing implementation have only the ratio;
            # older objects have neither field and therefore remain disabled.
            getattr(
                self,
                "_activation_checkpointing_enabled",
                getattr(self, "_activation_checkpointing_ratio", 0.0) > 0.0,
            ),
            getattr(
                self,
                "_activation_checkpointing_components",
                DEFAULT_CHECKPOINTING_COMPONENTS,
            ),
            training=self.training,
        )

    def _run_checkpointed_component(
        self,
        component: str,
        function: Callable[..., Any],
        *inputs: Any,
    ) -> Any:
        r"""Run a selected component directly or under checkpointing."""
        return run_checkpointed_component(
            function,
            *inputs,
            enabled=self._should_checkpoint_component(component),
            use_te=self.use_te,
            te_module=te,
        )

    def _checkpoint_block(
        self,
        block: GALEBlock,
        x: tuple[torch.Tensor, ...] | list[torch.Tensor],
        embedding_states: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        r"""Checkpoint a multi-stream GALE block with explicit tensor inputs."""
        return checkpoint_block(
            block,
            x,
            embedding_states,
            use_te=self.use_te,
            te_module=te,
        )

    def _build_context(
        self,
        local_embedding: tuple[torch.Tensor, ...],
        local_positions: tuple[torch.Tensor, ...] | None,
        geometry: torch.Tensor | None,
        global_embedding: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        list[torch.Tensor] | None,
        torch.Tensor | None,
    ]:
        r"""Build context directly or under activation checkpointing."""
        return self._run_checkpointed_component(
            "context",
            self.context_builder.build_context,
            local_embedding,
            local_positions,
            geometry,
            global_embedding,
        )

    def forward(
        self,
        local_embedding: (
            Float[torch.Tensor, "batch tokens features"]
            | tuple[Float[torch.Tensor, "batch tokens features"], ...]
        ),
        local_positions: (
            Float[torch.Tensor, "batch tokens spatial_dim"]
            | tuple[Float[torch.Tensor, "batch tokens spatial_dim"], ...]
            | None
        ) = None,
        global_embedding: Float[torch.Tensor, "batch global_tokens global_dim"]
        | None = None,
        geometry: Float[torch.Tensor, "batch tokens geometry_dim"] | None = None,
        time: torch.Tensor | None = None,
        *,
        return_embedding_states: bool = False,
        return_point_features: bool = False,
    ) -> (
        Float[torch.Tensor, "batch tokens out_dim"]
        | tuple[Float[torch.Tensor, "batch tokens out_dim"], ...]
    ):
        r"""Forward pass of the GeoTransolver model.

        The model constructs global context embeddings from geometry and global features
        by projecting them onto physical state spaces. These context embeddings are then
        used in all GALE blocks via cross-attention, allowing geometric and global
        information to guide the learned physical state dynamics.

        Parameters
        ----------
        local_embedding : torch.Tensor | tuple[torch.Tensor, ...]
            Local embedding of the input data of shape :math:`(B, N, C)` where
            :math:`B` is batch size, :math:`N` is number of nodes/tokens, and
            :math:`C` is ``functional_dim``.
        local_positions : torch.Tensor | tuple[torch.Tensor, ...] | None, optional
            Local positions for each input, each of shape :math:`(B, N, 3)`.
            Required if ``include_local_features=True``. Default is ``None``.
        global_embedding : torch.Tensor | None, optional
            Global embedding of shape :math:`(B, N_g, C_g)`. Default is ``None``.
        geometry : torch.Tensor | None, optional
            Geometry features of shape :math:`(B, N, C_{geo})`. Default is ``None``.
        time : torch.Tensor | None, optional
            Time embedding (not yet implemented). Default is ``None``.
        return_embedding_states : bool, optional, keyword-only
            If ``True``, return ``(output, embedding_states)`` instead of just
            ``output``.  The ``embedding_states`` tensor contains geometry/global
            context of shape :math:`(B, H, S, D_c)`.  Default is ``False``.
        return_point_features : bool, optional, keyword-only
            If ``True``, also return the per-point features computed just before
            the output projection (``ln_mlp_out``), of shape
            :math:`(B, N, D_{eff})` where
            :math:`D_{eff} = n\_hidden + n\_hidden\_local \cdot len(radii)`.
            These per-point latents are intended for attaching pointwise heads
            (e.g. a field GP head for per-point uncertainty).  Returned as the
            last element of the output tuple.  Default is ``False``.

        Returns
        -------
        Float[torch.Tensor, "batch tokens out_dim"] | tuple[Float[torch.Tensor, "batch tokens out_dim"], Float[torch.Tensor, "batch heads slices context_dim"]]
            With neither flag set (default): output tensor of shape
            :math:`(B, N, C_{out})`.

            With one flag set: a 2-tuple, ``(output, embedding_states)`` or
            ``(output, point_features)``.

            With both set: the 3-tuple
            ``(output, embedding_states, point_features)``.  The two flags are
            keyword-only, since they share a return signature and reading a
            bare ``True`` at the call site would not say which was meant.

        Raises
        ------
        NotImplementedError
            If ``time`` is provided.
        ValueError
            If input tensors have incorrect dimensions.
        """
        # Track whether input was a single tensor for output format
        single_input = isinstance(local_embedding, torch.Tensor)

        # Time embedding not yet supported
        if time is not None:
            raise NotImplementedError(
                "Time input is not implemented yet. "
                "Error rather than silently ignoring it."
            )

        # Normalize inputs to tuple format
        local_embedding = _normalize_tensor(local_embedding)
        if local_positions is not None:
            local_positions = _normalize_tensor(local_positions)

        unflatten_output = False
        if self.structured_shape is not None:
            unflatten_output = any(le.ndim in (4, 5) for le in local_embedding)
            local_embedding = tuple(
                _flatten_for_structured(
                    le, self.structured_shape, f"local_embedding[{i}]"
                )
                for i, le in enumerate(local_embedding)
            )
            if geometry is not None:
                geometry = _flatten_for_structured(
                    geometry, self.structured_shape, "geometry"
                )
            n_tok = _structured_num_tokens(self.structured_shape)
            for i, le in enumerate(local_embedding):
                if le.shape[1] != n_tok:
                    raise ValueError(
                        f"structured GeoTransolver: all streams must have N={n_tok} tokens; "
                        f"local_embedding[{i}] has N={le.shape[1]}"
                    )

        ### Input validation
        if not torch.compiler.is_compiling():
            if len(local_embedding) == 0:
                raise ValueError("Expected non-empty local_embedding")
            # The model has one input stream per configured functional_dim, and
            # builds one preprocess/output module per stream. Check the caller
            # passed the right number of streams here, once, so a wrong count
            # gives a clear error instead of a confusing crash later (or quietly
            # ignoring extra streams).
            n_streams = len(self.preprocess)
            if len(local_embedding) != n_streams:
                raise ValueError(
                    f"Expected {n_streams} local_embedding stream(s) to match the "
                    f"model's configured functional_dim/out_dim, "
                    f"got {len(local_embedding)}"
                )
            if local_positions is not None and len(local_positions) != len(
                local_embedding
            ):
                raise ValueError(
                    f"Expected local_positions to provide the same number of "
                    f"streams as local_embedding ({len(local_embedding)}), "
                    f"got {len(local_positions)}"
                )
            for i, tensor in enumerate(local_embedding):
                if tensor.ndim != 3:
                    raise ValueError(
                        f"Expected 3D local_embedding tensor (B, N, C) at index {i}, "
                        f"got {tensor.ndim}D tensor with shape {tuple(tensor.shape)}"
                    )
            if geometry is not None and geometry.ndim != 3:
                raise ValueError(
                    f"Expected 3D geometry tensor (B, N, C_geo), "
                    f"got {geometry.ndim}D tensor with shape {tuple(geometry.shape)}"
                )
            if global_embedding is not None and global_embedding.ndim != 3:
                raise ValueError(
                    f"Expected 3D global_embedding tensor (B, N_g, C_g), "
                    f"got {global_embedding.ndim}D tensor with shape {tuple(global_embedding.shape)}"
                )

        # Build context embeddings and extract local features. The third
        # return value (detached geometry latent) is consumed by an optional
        # external OOD guard wrapper via a forward hook, not here.
        embedding_states, local_embedding_bq, _ = self._build_context(
            local_embedding, local_positions, geometry, global_embedding
        )

        # Project inputs to hidden dimension: (B, N, C) -> (B, N, n_hidden)
        x = [
            self._run_checkpointed_component("preprocess", self.preprocess[i], le)
            for i, le in enumerate(local_embedding)
        ]

        # Concatenate local features if enabled
        if self.include_local_features and local_embedding_bq is not None:
            x = [
                torch.cat([x[i], local_embedding_bq[i]], dim=-1) for i in range(len(x))
            ]

        # Pass through GALE transformer blocks with context cross-attention
        for block_idx, block in enumerate(self.blocks):
            if self._should_checkpoint_block(block_idx):
                x = self._checkpoint_block(block, x, embedding_states)
            else:
                x = block(tuple(x), embedding_states)

        # Per-point features just before the output projection. Shape per
        # stream: (B, N, effective_hidden). Captured for pointwise heads.
        point_features = list(x)

        # Project to output dimensions: (B, N, n_hidden) -> (B, N, out_dim)
        x = [
            self._run_checkpointed_component("output", self.ln_mlp_out[i], x[i])
            for i in range(len(x))
        ]

        if self.structured_shape is not None and unflatten_output:
            B = x[0].shape[0]
            for i in range(len(x)):
                if len(self.structured_shape) == 2:
                    H, W = self.structured_shape
                    x[i] = x[i].reshape(B, H, W, -1)
                else:
                    H, W, D_ = self.structured_shape
                    x[i] = x[i].reshape(B, H, W, D_, -1)

        # Return same format as input (single tensor or tuple)
        if single_input:
            x = x[0]
            point_features_out = point_features[0]
        else:
            x = tuple(x)
            point_features_out = tuple(point_features)

        if return_embedding_states and return_point_features:
            return x, embedding_states, point_features_out
        if return_embedding_states:
            return x, embedding_states
        if return_point_features:
            return x, point_features_out
        return x
