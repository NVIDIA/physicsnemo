# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn as nn

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.version_check import check_version_spec
from physicsnemo.models.transolver.transolver import MLP

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module

from .context_projector import ContextProjector, GlobalContextBuilder
from .gale import GALE_block

# Check optional dependency availability
TE_AVAILABLE = check_version_spec("transformer_engine", "0.1.0", hard_fail=False)
if TE_AVAILABLE:
    import transformer_engine.pytorch as te



@dataclass
class GeoTransolverMetaData(ModelMetaData):
    """
    Data class for storing essential meta data needed for the GeoTransolver model.
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


def _normalize_dim(x):
    # Accept int as scalar
    if isinstance(x, int):
        return (x,)
    # Accept any non-string sequence of ints
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        return tuple(int(v) for v in x)
    raise TypeError(f"Invalid dim specifier {x!r}")


def _normalize_tensor(x):
    # Accept int as scalar
    if isinstance(x, torch.Tensor):
        return (x,)
    if isinstance(x, Sequence):
        return x
    raise TypeError(f"Invalid tensor structure")

class GeoTransolver(Module):
    r"""GeoTransolver: Geometry-Aware Physics Attention Transformer.

    GeoTransolver is an adaptation of the Transolver architecture, replacing standard attention
    with GALE (Geometry-Aware Latent Embeddings) attention. GALE combines physics-aware
    self-attention on learned state slices with cross-attention to geometry and global
    context embeddings.

    The model projects geometry and global features onto physical state spaces, which are
    then used as context in all transformer blocks. This design enables the model to
    incorporate geometric structure and global information throughout the forward pass.

    Parameters
    ----------
    functional_dim : int
        Dimension of the input values (local embeddings), not including global embeddings or geometry features. Input will be projected to ``n_hidden`` before processing.
    out_dim : int
        Dimension of the output of the model.
    geometry_dim : int, optional
        Pointwise dimension of the geometry input features. If provided, geometry features will be projected onto physical states and used as context in all GALE layers. Default is None.
    global_dim : int, optional
        Dimension of the global embedding features. If provided, global features will be projected onto physical states and used as context in all GALE layers. Default is None.
    n_layers : int, optional
        Number of GALE layers in the model. Default is 4.
    n_hidden : int, optional
        Hidden dimension of the transformer. Default is 256.
    dropout : float, optional
        Dropout rate applied across the GALE layers. Default is 0.0.
    n_head : int, optional
        Number of attention heads in each GALE layer. Must evenly divide ``n_hidden`` to yield an integer head dimension. Default is 8.
    act : str, optional
        Activation function name. Default is "gelu".
    mlp_ratio : int, optional
        Ratio of MLP hidden dimension to ``n_hidden``. Default is 4.
    slice_num : int, optional
        Number of learned physical state slices in the GALE layers, representing the number of learned states each layer should project inputs onto. Default is 32.
    use_te : bool, optional
        Whether to use Transformer Engine backend when available. Default is True.
    time_input : bool, optional
        Whether to include time embeddings. Default is False.
    plus : bool, optional
        Whether to use Transolver++ features in the GALE layers. Default is False.
    include_local_features : bool, optional
        Whether to include local features in the global context. Default is False.
    radii : list[float], optional
        Radii for the local features. Default is [0.05, 0.25].
    neighbors_in_radius : list[int], optional
        Neighbors in radius for the local features. Default is [8, 32].
    n_hidden_local : int, optional
        Hidden dimension for the local features. Default is 512.

    Raises
    ------
    ValueError
        If ``n_hidden`` is not evenly divisible by ``n_head``.


    Forward
    ----------
    local_embedding : torch.Tensor
        Local embedding of the input data of shape \((B, N, C)\) where \(B\) is batch size, \(N\) is number of nodes/tokens, and \(C\) is ``functional_dim``. Output will have the same \((B, N)\) shape but with ``out_dim`` channels.
    global_embedding : torch.Tensor, optional
        Global embedding of the input data of shape \((B, N_g, C_g)\) where \(N_g\) is number of global tokens and \(C_g\) is ``global_dim``. If None, global context is not used. Default is None.
    geometry : torch.Tensor, optional
        Geometry features of the input data of shape \((B, N, C_{geo})\) where \(C_{geo}\) is ``geometry_dim``. If None, geometry context is not used. Default is None.
    time : torch.Tensor, optional
        Time embedding (currently not implemented). Default is None.

    Returns
    -------
    torch.Tensor
        Output tensor of shape \((B, N, C_{out})\) where \(C_{out}\) is ``out_dim``.

    Notes
    -----
    GeoTransolver currently supports unstructured mesh input only. Enhancements for image-based
    and voxel-based inputs may be available in the future.

    For more details on Transolver, see:
    - https://arxiv.org/pdf/2402.02366
    - https://arxiv.org/pdf/2502.02414

    See Also
    --------
    :class:`GALE` : The attention mechanism used in GeoTransolver.
    :class:`GALE_block` : Transformer block using GALE attention.
    :class:`ContextProjector` : Projects context features onto physical states.

    Examples
    --------
    Basic usage with local embeddings only:

    >>> import torch
    >>> import physicsnemo
    >>> model = physicsnemo.models.GeoTransolver(
    ...     functional_dim=64,
    ...     out_dim=3,
    ...     n_hidden=256,
    ...     n_layers=4
    ... )
    >>> local_emb = torch.randn(2, 1000, 64)  # (batch, nodes, features)
    >>> output = model(local_emb)
    >>> output.shape
    torch.Size([2, 1000, 3])

    Usage with geometry and global context:

    >>> model = physicsnemo.models.GeoTransolver(
    ...     functional_dim=64,
    ...     out_dim=3,
    ...     geometry_dim=3,
    ...     global_dim=16,
    ...     n_hidden=256,
    ...     n_layers=4
    ... )
    >>> local_emb = torch.randn(2, 1000, 64)
    >>> geometry = torch.randn(2, 1000, 3)  # (batch, nodes, spatial_dim)
    >>> global_emb = torch.randn(2, 1, 16)  # (batch, 1, global_features)
    >>> output = model(local_emb, global_embedding=global_emb, geometry=geometry)
    >>> output.shape
    torch.Size([2, 1000, 3])
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
        use_te: bool = True,
        time_input: bool = False,
        plus: bool = False,
        include_local_features: bool = False,
        radii: list[float] = [0.05, 0.25],
        neighbors_in_radius: list[int] = [8, 32],
        n_hidden_local: int = 32,
    ) -> None:
        super().__init__(meta=GeoTransolverMetaData())
        self.__name__ = "GeoTransolver"

        self.include_local_features = include_local_features

        self.use_te = use_te
        # Check that the hidden dimension and head dimensions are compatible:
        if not n_hidden % n_head == 0:
            raise ValueError(
                f"GeoTransolver requires n_hidden % n_head == 0, but instead got {n_hidden % n_head}"
            )

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
        )
        context_dim = self.context_builder.get_context_dim()
      
        if len(functional_dims) != len(out_dims):
            raise ValueError(
                f"functional_dim and out_dim must be the same length, but instead got {len(functional_dims)} and {len(out_dims)}"
            )
      
        # This MLP is the initial projection onto the hidden space.
        # One per input "type"
        
        self.preprocess = nn.ModuleList(
            [
                MLP(
                    f,
                    n_hidden * 2,
                    n_hidden,
                    n_layers=0,
                    res=False,
                    act=act,
                    use_te=use_te,
                )
                for f in functional_dims
            ]
        )

        self.n_hidden = n_hidden

        self.blocks = nn.ModuleList(
            [
                GALE_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden + n_hidden_local * len(self.radii) if self.include_local_features else n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    slice_num=slice_num,
                    last_layer=(_ == n_layers - 1),
                    use_te=use_te,
                    plus=plus,
                    context_dim=context_dim,
                )
                for _ in range(n_layers)
            ]
        )

        if use_te:
            self.ln_mlp_out = nn.ModuleList(
                [
                    te.LayerNormLinear(
                        in_features=n_hidden + n_hidden_local * len(self.radii) if self.include_local_features else n_hidden, out_features=o
                    ) for o in out_dims
                 ]
            )
        else:
            self.ln_mlp_out = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(n_hidden + n_hidden_local * len(self.radii) if self.include_local_features else n_hidden),
                        nn.Linear(n_hidden + n_hidden_local * len(self.radii) if self.include_local_features else n_hidden, o),
                    )
                    for o in out_dims
                ]
                
            )


        self.time_input = time_input
        if time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden)
            )

    def forward(
        self,
        local_embedding: torch.Tensor | tuple[torch.Tensor, ...],
        local_positions: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        global_embedding: torch.Tensor | None = None,
        geometry: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Forward pass of the GeoTransolver model.

        The model constructs global context embeddings from geometry and global features by
        projecting them onto physical state spaces. These context embeddings are then used
        in all GALE blocks via cross-attention, allowing geometric and global information to
        guide the learned physical state dynamics.
        

        """

        single_input = isinstance(local_embedding, torch.Tensor)
        
        if time is not None:
            raise NotImplementedError("Time input is not implemented yet."
                                      "Error rather than silently ignoring it.")

        local_embedding = _normalize_tensor(local_embedding)
        if local_positions is not None:
            local_positions = _normalize_tensor(local_positions)
        
        # Build context and extract local features using the context builder
        embedding_states, local_embedding_bq = self.context_builder.build_context(
            local_embedding, local_positions, geometry, global_embedding
        )

        # Project inputs to hidden dimension
        x = [self.preprocess[i](le) for i, le in enumerate(local_embedding)]

        # Concatenate local features if enabled
        if self.include_local_features and local_embedding_bq is not None:
            x = [torch.cat([x[i], local_embedding_bq[i]], dim=-1) for i in range(len(x))]

        for block in self.blocks:
            x = block(x, embedding_states)

        # Now, pass the data through the model:
        x = [self.ln_mlp_out[i](x[i]) for i in range(len(x))]

        if single_input:
            # If only one input came in, use just that as output:
            x = x[0]
        else:
            x = tuple(x)

        return x
