# ignore_header_test
# ruff: noqa: E402
""""""

"""
Transolver model. This code was modified from, https://github.com/thuml/Transolver

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

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

try:
    import transformer_engine.pytorch as te

    TE_AVAILABLE = True
except (ImportError, FileNotFoundError):
    TE_AVAILABLE = False

import physicsnemo  # noqa: F401 for docs

from ..meta import ModelMetaData
from ..module import Module
from .Embedding import timestep_embedding

# from .Physics_Attention import Physics_Attention_Structured_Mesh_2D
from .Physics_Attention import (
    PhysicsAttentionIrregularMesh,
    PhysicsAttentionStructuredMesh2D,
    PhysicsAttentionStructuredMesh3D,
)

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


class MLP(nn.Module):
    def __init__(
        self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True, use_te=True
    ):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res

        self.act = act()

        linear_layer = nn.Linear if not use_te else te.Linear

        self.linear_pre = linear_layer(n_input, n_hidden)
        self.linear_post = linear_layer(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(linear_layer(n_hidden, n_hidden), act())
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        x = self.act(self.linear_pre(x))
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block, replacing standard attention with physics attention."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        spatial_shape: tuple[int, ...] | None = None,
        use_te=True,
        plus: bool = False,
    ):
        super().__init__()

        if use_te and not TE_AVAILABLE:
            raise ImportError(
                "Transformer Engine is not installed. Please install it with `pip install transformer-engine`."
            )

        self.last_layer = last_layer
        if use_te:
            self.ln_1 = te.LayerNorm(hidden_dim)
        else:
            self.ln_1 = nn.LayerNorm(hidden_dim)

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
                raise Exception(
                    f"Unexpected length of spatial shape encountered in Transolver_block: {len(spatial_shape)}"
                )

        if use_te:
            self.ln_mlp1 = te.LayerNormMLP(
                hidden_size=hidden_dim,
                ffn_hidden_size=hidden_dim * mlp_ratio,
            )
        else:
            self.ln_mlp1 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                MLP(
                    hidden_dim,
                    hidden_dim * mlp_ratio,
                    hidden_dim,
                    n_layers=0,
                    res=False,
                    act=act,
                    use_te=False,
                ),
            )
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

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.ln_mlp1(fx) + fx
        if self.last_layer:
            return self.ln_mlp2(fx)
        else:
            return fx


@dataclass
class MetaData(ModelMetaData):
    name: str = "Transolver"
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
    """
    Transolver model, adapted from original transolver code.

    Transolver is an adaptation of the transformer architecture, with a physics-attention
    mechanism replacing the standard attention mechanism.

    For more architecture details, see: https://arxiv.org/pdf/2402.02366 and https://arxiv.org/pdf/2502.02414

    Transolver can work on structured or unstructured data points as a model construction choice:
    - unstructured data (like a mesh) should provide some sort of positional encoding to accompany inputs
    - structured data (2D and 3D grids) can provide positional encodings optionally

    When constructing Transolver, you can choose to use "unified position" or not.  If you select "unified
    position" (`unified_pos=True`), then

    If using structured data, pass the structured shape as a tuple in the model constructor.
    Length 2 tuples are assumed to be image-like, length 3 tuples are assumed to be 3D voxel like.
    Other structured shape sizes are not supported.  Passing a structured_shape of None assumes irregular data.

    Output shape will have the same spatial shape as the input shape, with potentially more features

    Also can support Transolver++ implementation.  When using the distributed algorithm
    of Transolver++, use PhysicsNeMo's ShardTensor implementation to support automatic
    domain parallelism and 2D parallelization (data parallel + domain parallel, for example).

    Note
    ----


    Parameters
    ----------
    functional_dim : int
        The dimension of the input values, not including any embeddings.  No Default.
        Input will be concatenated with embeddings or unified position before processing
        with PhysicsAttention blocks.  Originally known as "fun_dim"
    out_dim : int
        The dimension of the output of the model.  This is a mandatory parameter.
    embedding_dim : int | None
        The spatial dimension of the input data embeddings.  Should include not just
        position but all computed embedding features.  Default is None, but if
        `unified_pos=False` this is a mandatory parameter.  Originally named "space_dim"
    n_layers : int
        The number of transformer PhysicsAttention layers in the model.  Default of 4.
    n_hidden : int
        The hidden dimension of the transformer.  Default of 256.  Projection is made
        from the input data + embeddings in the early preprocessing, before the
        PhysicsAttention layers.
    dropout : float
        The dropout rate, applied across the PhysicsAttention Layers.  Default is 0.0
    n_head : int
        The number of attention heads in each PhysicsAttention Layer.  Default is 8.  Note
        that the number of heads must evenly divide the `n_hidden` parameter to yield an
        integer head dimension.
    act : str
        The activation function, default is gelu.
    mlp_ratio : int
        The ratio of hidden dimension in the MLP, default is 4.  Used in the MLPs in the
        PhysicsAttention Layers.
    slice_num : int
        The number of slices in the PhysicsAttention layers.  Default is 32.  Represents the
        number of learned states each layer should project inputs onto.
    unified_pos : bool
        Whether to use unified positional embeddings.  Unified positions are only available for
        structured data (2D grids, 3D grids).  They are computed once initially, and reused through
        training in place of embeddings.
    ref : int
        The reference dimension size when using unified positions.  Default is 8.  Will be
        used to create a linear grid in spatial dimensions to serve as spatial embeddings.
        If `unified_pos=False`, this value is unused.
    structured_shape : None | tuple(int)
        The shape of the latent space.  If None, assumes irregular latent space.  If not
        `None`, this parameter can only be a length-2 or length-3 tuple of ints.
    use_te: bool
        Whether to use transformer engine backend when possible.
    time_input : bool
        Whether to include time embeddings. Default is false
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
        structured_shape: None | tuple[int] = None,
        use_te: bool = True,
        time_input: bool = False,
        plus: bool = False,
    ) -> None:
        super().__init__(meta=MetaData())
        self.__name__ = "Transolver"

        self.use_te = use_te
        # Check that the hidden dimension and head dimensions are compatible:
        if not n_hidden % n_head == 0:
            raise ValueError(
                f"Transolver requires n_hidden % n_head == 0, but instead got {n_hidden % n_head}"
            )

        # Check the shape of the data, if it's structured data:
        if structured_shape is not None:
            # Has to be 2D or 3D data:
            if len(structured_shape) not in [2, 3]:
                raise ValueError(
                    f"Transolver can only use structured data in 2D or 3D, got {structured_shape}"
                )

            # Ensure it's all integers > 0:
            if not all([s > 0 and s == int(s) for s in structured_shape]):
                raise ValueError(
                    f"Transolver can only use integer shapes > 0, got {structured_shape}"
                )
        else:
            # It's mandatory for unified position:
            if unified_pos:
                raise ValueError(
                    "Transolver requires structured_shape to be passed if using unified_pos=True"
                )

        self.structured_shape = structured_shape

        # If we're using the unified position, create and save the position embeddings:
        self.unified_pos = unified_pos

        if unified_pos:
            if structured_shape is None:
                raise ValueError(
                    "Transolver can not use unified position without a structured_shape argument (got None)"
                )

            # This ensures embedding is tracked by torch and moves to the GPU, and saves/loads
            self.register_buffer("embedding", self.get_grid(ref))
            self.embedding_dim = ref * ref
            mlp_input_dimension = functional_dim + ref * ref

        else:
            self.embedding_dim = embedding_dim
            mlp_input_dimension = functional_dim + embedding_dim

        # This MLP is the initial projection onto the hidden space
        self.preprocess = MLP(
            mlp_input_dimension,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
            use_te=use_te,
        )

        self.time_input = time_input
        self.n_hidden = n_hidden
        if time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden)
            )

        self.blocks = nn.ModuleList(
            [
                Transolver_block(
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

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        linear_layers = (nn.Linear,)
        if self.use_te:
            linear_layers = linear_layers + (te.Linear,)

        if isinstance(m, linear_layers):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, linear_layers) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        norm_layers = (nn.LayerNorm, nn.BatchNorm1d)
        if self.use_te:
            norm_layers = norm_layers + (te.LayerNorm,)
        if isinstance(m, norm_layers):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, ref: int, batchsize: int = 1) -> torch.Tensor:
        """
        Generate a unified positional encoding grid for structured 2D data.

        Parameters
        ----------
        ref : int
            The reference grid size for the unified position encoding.
        batchsize : int, optional
            The batch size for the generated grid (default is 1).

        Returns
        -------
        torch.Tensor
            A tensor of shape (batchsize, H*W, ref*ref) containing the positional encodings,
            where H and W are the spatial dimensions from self.structured_shape.
        """
        size_x, size_y = self.structured_shape
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1)  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridx = gridx.reshape(1, ref, 1, 1).repeat([batchsize, 1, ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ref, 1).repeat([batchsize, ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1)  # B H W 8 8 2

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
        fx: torch.Tensor | None,
        embedding: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transolver model.

        Args:
            fx (torch.Tensor | None): Functional input tensor. For structured data,
                shape should be [B, N, C] or [B, *structure, C]. For unstructured data,
                shape should be [B, N, C]. Can be None if not used.
            embedding (torch.Tensor | None, optional): Embedding tensor. For structured
                data, shape should be [B, N, C] or [B, *structure, C]. For unstructured
                data, shape should be [B, N, C]. Defaults to None.
            time (torch.Tensor | None, optional): Optional time tensor. Shape and usage
                depend on the model configuration. Defaults to None.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.

        """
        if self.unified_pos:
            # Extend the embedding to the batch size:
            embedding = self.embedding.repeat(fx.shape[0], 1, 1)

        # Reshape automatically, if necessary:
        if self.structured_shape is not None:
            unflatten_output = False
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

        # Combine the embedding and functional input:
        if embedding is not None:
            fx = torch.cat((embedding, fx), -1)

        # Apply preprocessing
        fx = self.preprocess(fx)

        if time is not None:
            time_emb = timestep_embedding(time, self.n_hidden).repeat(
                1, embedding.shape[1], 1
            )
            time_emb = self.time_fc(time_emb)
            fx = fx + time_emb

        for i, block in enumerate(self.blocks):
            fx = block(fx)

        if self.structured_shape is not None:
            if unflatten_output:
                fx = fx.reshape(fx.shape[0], *self.structured_shape, -1)

        return fx
