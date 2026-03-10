# ignore_header_test
# ruff: noqa: E402

r"""
Transolver model and building blocks for physics-informed neural operator learning.

This module provides the main Transolver model class along with its internal
building blocks (MLP, Transolver_block) for solving PDEs on structured and
unstructured meshes.

This code was modified from https://github.com/thuml/Transolver

The following license is provided from their source,

MIT License

Copyright (c) 2024 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import importlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.core.version_check import check_version_spec
from physicsnemo.nn import Mlp, PositionalEmbedding
from physicsnemo.nn.module.physics_attention import (
    PhysicsAttentionIrregularMesh,
    PhysicsAttentionStructuredMesh2D,
    PhysicsAttentionStructuredMesh3D,
)

TE_AVAILABLE = check_version_spec("transformer_engine", hard_fail=False)
if TE_AVAILABLE:
    te = importlib.import_module("transformer_engine.pytorch")
else:
    te = None


ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class _TransolverMlp(Mlp):
    """Mlp subclass with state dict compatibility for legacy Transolver checkpoints.

    This class provides backward compatibility for loading checkpoints saved with
    the old Transolver MLP class, which used different attribute names:
    - Old: `linear_pre`, `linear_post`, `linears`
    - New: `layers` (Sequential)

    The mapping handles the common case where `n_layers=0` (no hidden layers with
    residual connections), which was the typical usage pattern in Transolver.
    """

    # Mapping from old checkpoint keys to new Mlp keys
    # This assumes the typical usage: n_layers=0, which means just linear_pre -> act -> linear_post
    # Old structure: linear_pre (input->hidden), linear_post (hidden->output)
    # New structure: layers.0 (input->hidden), layers.1 (activation), layers.2 (hidden->output)
    _OLD_TO_NEW_KEYS = {
        "linear_pre.weight": "layers.0.weight",
        "linear_pre.bias": "layers.0.bias",
        "linear_post.weight": "layers.2.weight",
        "linear_post.bias": "layers.2.bias",
    }

    _NEW_TO_OLD_KEYS = {v: k for k, v in _OLD_TO_NEW_KEYS.items()}

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        assign: bool = False,
    ):
        """Load state dict with automatic key remapping for legacy checkpoints.

        This hook is called by PyTorch for each module during load_state_dict().
        We intercept it to remap old-style keys (linear_pre, linear_post) to
        new-style keys (layers.0, layers.2) before the actual loading.
        """
        # Check for old-style keys with this module's prefix
        # e.g., prefix="preprocess." -> look for "preprocess.linear_pre.weight"
        for old_suffix, new_suffix in self._OLD_TO_NEW_KEYS.items():
            old_key = prefix + old_suffix
            new_key = prefix + new_suffix
            if old_key in state_dict and new_key not in state_dict:
                # Remap old key to new key
                state_dict[new_key] = state_dict.pop(old_key)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            assign,
        )


class TransolverBlock(nn.Module):
    r"""
    Transformer encoder block with physics attention mechanism.

    This block replaces standard attention with physics attention, which learns
    to project inputs onto physics-informed slices before applying attention.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    hidden_dim : int
        Hidden dimension of the block.
    dropout : float
        Dropout rate.
    act : str, optional, default="gelu"
        Activation function name.
    mlp_ratio : int, optional, default=4
        Ratio of MLP hidden dimension to ``hidden_dim``.
    last_layer : bool, optional, default=False
        Whether this is the last layer (applies output projection).
    out_dim : int, optional, default=1
        Output dimension (only used if ``last_layer=True``).
    slice_num : int, optional, default=32
        Number of physics slices.
    spatial_shape : tuple[int, ...] | None, optional, default=None
        Spatial shape for structured data. ``None`` for irregular meshes.
    use_te : bool, optional, default=True
        Whether to use transformer engine.
    plus : bool, optional, default=False
        Whether to use Transolver++ variant.

    Forward
    -------
    fx : torch.Tensor
        Input tensor of shape :math:`(B, N, C)` where :math:`B` is batch size,
        :math:`N` is number of tokens, :math:`C` is hidden dimension.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, N, C)`, or :math:`(B, N, C_{out})`
        if ``last_layer=True``.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int = 4,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
        spatial_shape: tuple[int, ...] | None = None,
        use_te: bool = True,
        plus: bool = False,
    ):
        super().__init__()

        if use_te and not TE_AVAILABLE:
            raise ImportError(
                "Transformer Engine is not installed. Please install it with "
                "`pip install transformer-engine`."
            )

        self.last_layer = last_layer

        # Layer normalization before attention
        if use_te:
            self.ln_1 = te.LayerNorm(hidden_dim)
        else:
            self.ln_1 = nn.LayerNorm(hidden_dim)

        # Select appropriate physics attention based on spatial structure
        if spatial_shape is None:
            self.Attn = PhysicsAttentionIrregularMesh(
                hidden_dim,
                heads=num_heads,
                dim_head=hidden_dim // num_heads,
                dropout=dropout,
                slice_num=slice_num,
                use_te=use_te,
                plus=plus,
            )
        else:
            if len(spatial_shape) == 2:
                self.Attn = PhysicsAttentionStructuredMesh2D(
                    hidden_dim,
                    spatial_shape=spatial_shape,
                    heads=num_heads,
                    dim_head=hidden_dim // num_heads,
                    dropout=dropout,
                    slice_num=slice_num,
                    use_te=use_te,
                    plus=plus,
                )
            elif len(spatial_shape) == 3:
                self.Attn = PhysicsAttentionStructuredMesh3D(
                    hidden_dim,
                    spatial_shape=spatial_shape,
                    heads=num_heads,
                    dim_head=hidden_dim // num_heads,
                    dropout=dropout,
                    slice_num=slice_num,
                    use_te=use_te,
                    plus=plus,
                )
            else:
                raise ValueError(
                    f"Unexpected length of spatial shape encountered in Transolver_block: "
                    f"{len(spatial_shape)}. Expected 2 or 3."
                )

        # Feed-forward network with layer norm
        if use_te:
            self.ln_mlp1 = te.LayerNormMLP(
                hidden_size=hidden_dim,
                ffn_hidden_size=hidden_dim * mlp_ratio,
            )
        else:
            self.ln_mlp1 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                _TransolverMlp(
                    in_features=hidden_dim,
                    hidden_features=hidden_dim * mlp_ratio,
                    out_features=hidden_dim,
                    act_layer=act,
                    use_te=False,
                ),
            )

        # Output projection for final layer
        if self.last_layer:
            if use_te:
                self.ln_mlp2 = te.LayerNormLinear(
                    in_features=hidden_dim, out_features=out_dim
                )
            else:
                self.ln_mlp2 = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, out_dim),
                )

    def forward(
        self, fx: Float[torch.Tensor, "B N C"]
    ) -> Float[torch.Tensor, "B N C_out"]:
        r"""
        Forward pass of the Transolver block.

        Parameters
        ----------
        fx : torch.Tensor
            Input tensor of shape :math:`(B, N, C)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, N, C)`, or :math:`(B, N, C_{out})`
            if ``last_layer=True``.
        """
        # Apply physics attention with residual connection
        fx = self.Attn(self.ln_1(fx)) + fx

        # Apply feed-forward network with residual connection
        fx = self.ln_mlp1(fx) + fx

        # Apply output projection if last layer
        if self.last_layer:
            return self.ln_mlp2(fx)
        else:
            return fx


@dataclass
class MetaData(ModelMetaData):
    r"""Metadata for the Transolver model."""

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


class Transolver(Module):
    r"""
    Transolver model for physics-informed neural operator learning.

    Transolver adapts the transformer architecture with a physics-attention
    mechanism replacing standard attention. It can work on both structured
    (2D/3D grids) and unstructured (mesh) data.

    For architecture details, see:

    - `Transolver paper <https://arxiv.org/pdf/2402.02366>`_
    - `Transolver++ paper <https://arxiv.org/pdf/2502.02414>`_

    .. note::

        When using structured data, pass the ``structured_shape`` as a tuple.
        Length-2 tuples are treated as 2D image-like data, length-3 tuples as
        3D volumetric data.

    Parameters
    ----------
    functional_dim : int
        Dimension of input values, not including embeddings.
    out_dim : int
        Dimension of model output.
    embedding_dim : int | None, optional, default=None
        Dimension of input embeddings. Required if ``unified_pos=False``.
    n_layers : int, optional, default=4
        Number of Transolver blocks.
    n_hidden : int, optional, default=256
        Hidden dimension of the transformer.
    dropout : float, optional, default=0.0
        Dropout rate.
    n_head : int, optional, default=8
        Number of attention heads. Must evenly divide ``n_hidden``.
    act : str, optional, default="gelu"
        Activation function name.
    mlp_ratio : int, optional, default=4
        Ratio of MLP hidden dimension to ``n_hidden``.
    slice_num : int, optional, default=32
        Number of physics slices in attention layers.
    unified_pos : bool, optional, default=False
        Whether to use unified positional embeddings (structured data only).
    ref : int, optional, default=8
        Reference grid size for unified position encoding.
    structured_shape : None | tuple[int, ...], optional, default=None
        Shape of structured data. ``None`` for unstructured mesh data.
    use_te : bool, optional, default=True
        Whether to use transformer engine.
    time_input : bool, optional, default=False
        Whether to include time embeddings.
    plus : bool, optional, default=False
        Whether to use Transolver++ variant.

    Forward
    -------
    fx : torch.Tensor
        Functional input tensor of shape :math:`(B, N, C_{in})` for flattened
        data where :math:`B` is batch size, :math:`N` is number of tokens, and
        :math:`C_{in}` is functional dimension. For structured data, shape is
        :math:`(B, H_s, W_s, C_{in})` for 2D or :math:`(B, H_s, W_s, D_s, C_{in})`
        for 3D, where :math:`H_s, W_s, D_s` are spatial dimensions.
    embedding : torch.Tensor | None, optional
        Embedding tensor. Required if ``unified_pos=False``. Shape should
        match ``fx`` spatial dimensions.
    time : torch.Tensor | None, optional
        Time tensor of shape :math:`(B,)` for time-dependent models.

    Outputs
    -------
    torch.Tensor
        Output tensor with same spatial shape as input and :math:`C_{out}`
        features (equal to ``out_dim``).

    Examples
    --------
    Structured 2D data with unified position:

    >>> import torch
    >>> from physicsnemo.models.transolver import Transolver
    >>> model = Transolver(
    ...     functional_dim=3,
    ...     out_dim=1,
    ...     structured_shape=(64, 64),
    ...     unified_pos=True,
    ...     n_hidden=128,
    ...     n_head=4,
    ...     use_te=False,
    ... )
    >>> x = torch.randn(2, 64, 64, 3)
    >>> out = model(x)
    >>> out.shape
    torch.Size([2, 64, 64, 1])

    Unstructured mesh data:

    >>> model = Transolver(
    ...     functional_dim=2,
    ...     embedding_dim=3,
    ...     out_dim=1,
    ...     structured_shape=None,
    ...     unified_pos=False,
    ...     n_hidden=128,
    ...     n_head=4,
    ...     use_te=False,
    ... )
    >>> fx = torch.randn(2, 1000, 2)
    >>> emb = torch.randn(2, 1000, 3)
    >>> out = model(fx, embedding=emb)
    >>> out.shape
    torch.Size([2, 1000, 1])
    """

    def __init__(
        self,
        functional_dim: int,
        out_dim: int,
        embedding_dim: int | None = None,
        n_layers: int = 4,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 4,
        slice_num: int = 32,
        unified_pos: bool = False,
        ref: int = 8,
        structured_shape: None | tuple[int, ...] = None,
        use_te: bool = True,
        time_input: bool = False,
        plus: bool = False,
    ) -> None:
        super().__init__(meta=MetaData())

        self.use_te = use_te

        # Validate hidden dimension and head compatibility
        if not n_hidden % n_head == 0:
            raise ValueError(
                f"Transolver requires n_hidden % n_head == 0, "
                f"but got n_hidden={n_hidden}, n_head={n_head} "
                f"(remainder={n_hidden % n_head})"
            )

        # Validate structured shape if provided
        if structured_shape is not None:
            if len(structured_shape) not in [2, 3]:
                raise ValueError(
                    f"Transolver only supports 2D or 3D structured data, "
                    f"got shape with {len(structured_shape)} dimensions"
                )
            if not all([s > 0 and s == int(s) for s in structured_shape]):
                raise ValueError(
                    f"Transolver requires positive integer shapes, "
                    f"got {structured_shape}"
                )
        else:
            if unified_pos:
                raise ValueError(
                    "Transolver requires structured_shape when using unified_pos=True"
                )

        self.structured_shape = structured_shape
        self.unified_pos = unified_pos

        # Set up positional embeddings
        if unified_pos:
            if structured_shape is None:
                raise ValueError(
                    "Transolver cannot use unified position without "
                    "structured_shape argument (got None)"
                )
            # Register unified position embedding as buffer
            self.register_buffer("embedding", self.get_grid(ref))
            self.embedding_dim = ref * ref
            mlp_input_dimension = functional_dim + ref * ref
        else:
            if embedding_dim is None:
                raise ValueError(
                    "Transolver requires embedding_dim when unified_pos=False"
                )
            self.embedding_dim = embedding_dim
            mlp_input_dimension = functional_dim + embedding_dim

        # Initial projection MLP
        self.preprocess = _TransolverMlp(
            in_features=mlp_input_dimension,
            hidden_features=n_hidden * 2,
            out_features=n_hidden,
            act_layer=act,
            use_te=use_te,
        )

        self.time_input = time_input
        self.n_hidden = n_hidden

        # Time embedding projection
        if time_input:
            self.time_embed = PositionalEmbedding(
                num_channels=n_hidden,
                max_positions=10000,
                endpoint=False,
                learnable=False,
                embed_fn="cos_sin",
            )
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.SiLU(),
                nn.Linear(n_hidden, n_hidden),
            )

        # Build transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransolverBlock(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    out_dim=out_dim,
                    slice_num=slice_num,
                    spatial_shape=structured_shape,
                    last_layer=(_ == n_layers - 1),
                    use_te=use_te,
                    plus=plus,
                )
                for _ in range(n_layers)
            ]
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        r"""Initialize model weights using truncated normal distribution."""
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        r"""
        Initialize weights for a single module.

        Parameters
        ----------
        m : nn.Module
            Module to initialize.
        """
        linear_layers = (nn.Linear,)
        if self.use_te:
            linear_layers = linear_layers + (te.Linear,)

        if isinstance(m, linear_layers):
            nn.init.trunc_normal_(m.weight, std=0.02)  # type: ignore[arg-type]
            if isinstance(m, linear_layers) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # type: ignore[arg-type]

        norm_layers = (nn.LayerNorm, nn.BatchNorm1d)
        if self.use_te:
            norm_layers = norm_layers + (te.LayerNorm,)
        if isinstance(m, norm_layers):
            nn.init.constant_(m.bias, 0)  # type: ignore[arg-type]
            nn.init.constant_(m.weight, 1.0)  # type: ignore[arg-type]

    def get_grid(self, ref: int, batchsize: int = 1) -> torch.Tensor:
        r"""
        Generate unified positional encoding grid for structured 2D data.

        Parameters
        ----------
        ref : int
            Reference grid size for unified position encoding.
        batchsize : int, optional, default=1
            Batch size for the generated grid.

        Returns
        -------
        torch.Tensor
            Positional encoding tensor of shape
            :math:`(B, H \times W, \text{ref}^2)`.
        """
        if self.structured_shape is None:
            raise ValueError(
                "Cannot generate positional encoding grid: structured_shape is None. "
                "This method requires structured_shape to be set."
            )
        size_x, size_y = self.structured_shape

        # Create spatial grid for the structured shape
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1)  # (B, H, W, 2)

        # Create reference grid
        gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridx = gridx.reshape(1, ref, 1, 1).repeat([batchsize, 1, ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ref, 1).repeat([batchsize, ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1)  # (B, ref, ref, 2)

        # Compute distance-based positional encoding
        pos = (
            torch.sqrt(
                torch.sum(
                    (grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :])
                    ** 2,
                    dim=-1,
                )
            )
            .reshape(batchsize, -1, ref * ref)  # Flatten spatial dims
            .contiguous()
        )
        return pos

    def forward(
        self,
        fx: Float[torch.Tensor, "B *spatial C_in"],
        embedding: Float[torch.Tensor, "B *spatial C_emb"] | None = None,
        time: Float[torch.Tensor, " B"] | None = None,
    ) -> Float[torch.Tensor, "B *spatial C_out"]:
        r"""
        Forward pass of the Transolver model.

        Parameters
        ----------
        fx : torch.Tensor
            Functional input tensor. Shape :math:`(B, N, C_{in})` for flattened
            data or :math:`(B, H_s, W_s, C_{in})` for structured 2D, where
            :math:`B` is batch size, :math:`N` is number of tokens, and
            :math:`C_{in}` is functional dimension.
        embedding : torch.Tensor | None, optional
            Embedding tensor. Required if ``unified_pos=False``.
        time : torch.Tensor | None, optional
            Time tensor of shape :math:`(B,)` for time-dependent models.

        Returns
        -------
        torch.Tensor
            Output tensor with same spatial shape as input and :math:`C_{out}`
            features.
        """
        # Input validation (skip during torch.compile for performance)
        if not torch.compiler.is_compiling():
            if fx.ndim < 2:
                raise ValueError(
                    f"Expected input tensor with at least 2 dimensions, "
                    f"got {fx.ndim}D tensor with shape {tuple(fx.shape)}"
                )
            if not self.unified_pos and embedding is None:
                raise ValueError("Embedding is required when unified_pos=False")

        # Track whether we need to unflatten output
        unflatten_output = False
        n_tokens = 0

        if self.unified_pos:
            # Extend unified position embedding to batch size
            emb_buffer: torch.Tensor = self.embedding  # type: ignore[assignment]
            embedding = emb_buffer.repeat(fx.shape[0], 1, 1)

        # Reshape structured data to flattened format if necessary
        if self.structured_shape is not None:
            if len(fx.shape) != 3:
                unflatten_output = True
                fx = fx.reshape(fx.shape[0], -1, fx.shape[-1])
            if embedding is not None and len(embedding.shape) != 3:
                embedding = embedding.reshape(
                    embedding.shape[0], *self.structured_shape, -1
                )
        else:
            if embedding is None:
                raise ValueError("Embedding is required for unstructured data")

        # Store n_tokens for time embedding
        if embedding is not None:
            n_tokens = embedding.shape[1]

        # Concatenate embedding with functional input
        if embedding is not None:
            fx = torch.cat((embedding, fx), -1)

        # Project to hidden dimension
        fx = self.preprocess(fx)

        # Add time embedding if provided
        if time is not None:
            time_emb = self.time_embed(time).unsqueeze(1).repeat(1, n_tokens, 1)
            time_emb = self.time_fc(time_emb)
            fx = fx + time_emb

        # Apply transformer blocks
        for block in self.blocks:
            fx = block(fx)

        # Reshape back to structured format if needed
        if self.structured_shape is not None:
            if unflatten_output:
                fx = fx.reshape(fx.shape[0], *self.structured_shape, -1)

        return fx
