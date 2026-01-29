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

r"""Context Projector for GeoTransolver model.

This module provides classes for projecting context features (geometry or global
embeddings) onto learned physical state spaces for use in GALE attention layers.

Classes
-------
ContextProjector
    Projects context features onto physical state slices.
GeometricFeatureProcessor
    Processes geometric features at a single spatial scale using BQWarp.
MultiScaleFeatureExtractor
    Multi-scale geometric feature extraction with minimal complexity.
GlobalContextBuilder
    Orchestrates all context construction for the GeoTransolver model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.models.transolver.Physics_Attention import gumbel_softmax
from physicsnemo.nn import BQWarp
from physicsnemo.nn import Mlp

# Check optional dependency availability
TE_AVAILABLE = check_version_spec("transformer_engine", "0.1.0", hard_fail=False)
if TE_AVAILABLE:
    import transformer_engine.pytorch as te


class ContextProjector(nn.Module):
    r"""Projects context features onto physical state space.

    This context projector is conceptually similar to half of a GALE attention layer.
    It projects context values (geometry or global embeddings) onto a learned physical
    state space, but unlike a full attention layer, it never projects back to the
    original space. The projected features are used as context in all GALE blocks
    of the GeoTransolver model.

    Parameters
    ----------
    dim : int
        Input dimension of the context features.
    heads : int, optional
        Number of projection heads. Default is 8.
    dim_head : int, optional
        Dimension of each projection head. Default is 64.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    slice_num : int, optional
        Number of learned physical state slices. Default is 64.
    use_te : bool, optional
        Whether to use Transformer Engine backend when available. Default is ``True``.
    plus : bool, optional
        Whether to use Transolver++ features. Default is ``False``.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, N, C)` where :math:`B` is batch size,
        :math:`N` is number of tokens, and :math:`C` is number of channels.

    Outputs
    -------
    torch.Tensor
        Slice tokens of shape :math:`(B, H, S, D)` where :math:`H` is number of heads,
        :math:`S` is number of slices, and :math:`D` is head dimension.

    Notes
    -----
    The global features are reused in all blocks of the model, so the learned
    projections must capture globally useful features rather than layer-specific ones.

    See Also
    --------
    :class:`~physicsnemo.experimental.models.geotransolver.gale.GALE` : Full GALE attention layer that uses these projected context features.
    :class:`~physicsnemo.experimental.models.geotransolver.GeoTransolver` : Main model that uses ContextProjector for geometry and global embeddings.

    Examples
    --------
    >>> import torch
    >>> projector = ContextProjector(dim=64, heads=8, dim_head=32, slice_num=32)
    >>> x = torch.randn(2, 100, 64)  # (batch, tokens, features)
    >>> slice_tokens = projector(x)
    >>> slice_tokens.shape
    torch.Size([2, 8, 32, 32])
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = True,
        plus: bool = False,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.plus = plus
        self.scale = dim_head**-0.5
        self.use_te = use_te

        # Choose linear layer implementation based on backend
        linear_layer = te.Linear if (use_te and TE_AVAILABLE) else nn.Linear

        # Input projection layers for query and key
        self.in_project_x = linear_layer(dim, inner_dim)
        if not plus:
            self.in_project_fx = linear_layer(dim, inner_dim)

        # Attention components
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        # Transolver++ adaptive temperature projection
        if plus:
            self.proj_temperature = nn.Sequential(
                linear_layer(self.dim_head, slice_num),
                nn.GELU(),
                linear_layer(slice_num, 1),
                nn.GELU(),
            )

        # Slice projection layer maps from head dimension to slice space
        self.in_project_slice = linear_layer(dim_head, slice_num)

    def project_input_onto_slices(
        self, x: Float[torch.Tensor, "batch tokens channels"]
    ) -> (
        Float[torch.Tensor, "batch heads tokens dim"]
        | tuple[
            Float[torch.Tensor, "batch heads tokens dim"],
            Float[torch.Tensor, "batch heads tokens dim"],
        ]
    ):
        r"""Project the input onto the slice space.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, N, C)` where :math:`B` is batch size,
            :math:`N` is number of tokens, and :math:`C` is number of channels.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            If ``plus=True``, returns single tensor of shape :math:`(B, H, N, D)` where
            :math:`H` is number of heads and :math:`D` is head dimension. If ``plus=False``,
            returns tuple of two tensors both of shape :math:`(B, H, N, D)`, representing
            the query and key projections respectively.
        """
        # Project input to multi-head representation: (B, N, C) -> (B, H, N, D)
        projected_x = rearrange(
            self.in_project_x(x), "B N (h d) -> B h N d", h=self.heads, d=self.dim_head
        )

        if self.plus:
            # Transolver++ uses single projection for both paths
            return projected_x
        else:
            # Standard Transolver uses separate query and key projections
            feature_projection = rearrange(
                self.in_project_fx(x),
                "B N (h d) -> B h N d",
                h=self.heads,
                d=self.dim_head,
            )
            return projected_x, feature_projection

    def compute_slices_from_projections(
        self,
        slice_projections: Float[torch.Tensor, "batch heads tokens slices"],
        fx: Float[torch.Tensor, "batch heads tokens dim"],
    ) -> tuple[
        Float[torch.Tensor, "batch heads tokens slices"],
        Float[torch.Tensor, "batch heads slices dim"],
    ]:
        r"""Compute slice weights and slice tokens from input projections and latent features.

        Parameters
        ----------
        slice_projections : torch.Tensor
            Projected input tensor of shape :math:`(B, H, N, S)` where :math:`B` is batch size,
            :math:`H` is number of heads, :math:`N` is number of tokens, and :math:`S` is number of
            slices, representing the projection of each token onto each slice for each
            attention head.
        fx : torch.Tensor
            Latent feature tensor of shape :math:`(B, H, N, D)` where :math:`D` is head dimension,
            representing the learned states to be aggregated by the slice weights.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - ``slice_weights``: Tensor of shape :math:`(B, H, N, S)`, normalized weights for
              each slice per token and head.
            - ``slice_token``: Tensor of shape :math:`(B, H, S, D)`, aggregated latent features
              for each slice, head, and batch.

        Notes
        -----
        The function computes a temperature-scaled softmax over the slice projections to
        obtain slice weights, then aggregates the latent features for each slice using
        these weights. The aggregated features are normalized by the sum of weights for
        numerical stability.
        """
        # Compute temperature-adjusted softmax weights
        if self.plus:
            # Transolver++ uses adaptive temperature with Gumbel softmax
            temperature = self.temperature + self.proj_temperature(fx)
            clamped_temp = torch.clamp(temperature, min=0.01).to(
                slice_projections.dtype
            )
            slice_weights = gumbel_softmax(slice_projections, clamped_temp)
        else:
            # Standard Transolver uses fixed temperature with regular softmax
            clamped_temp = torch.clamp(self.temperature, min=0.5, max=5).to(
                slice_projections.dtype
            )
            slice_weights = nn.functional.softmax(
                slice_projections / clamped_temp, dim=-1
            )

        # Ensure weights match the computation dtype
        slice_weights = slice_weights.to(slice_projections.dtype)

        # Aggregate features by slice weights with normalization
        # Normalize first to prevent overflow in reduced precision
        slice_norm = slice_weights.sum(2)  # Sum over tokens: (B, H, S)
        normed_weights = slice_weights / (slice_norm[:, :, None, :] + 1e-2)

        # Weighted aggregation: (B, H, S, N) @ (B, H, N, D) -> (B, H, S, D)
        slice_token = torch.matmul(normed_weights.transpose(2, 3), fx)

        return slice_weights, slice_token

    def forward(
        self, x: Float[torch.Tensor, "batch tokens channels"]
    ) -> Float[torch.Tensor, "batch heads slices dim"]:
        r"""Project inputs to physical state slices.

        This performs a partial physics attention operation: it projects the input onto
        learned physical state slices but does not project back to the original space.
        The resulting slice tokens serve as context for GALE attention layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, N, C)` where :math:`B` is batch size, :math:`N` is
            number of tokens, and :math:`C` is number of channels.

        Returns
        -------
        torch.Tensor
            Slice tokens of shape :math:`(B, H, S, D)` where :math:`H` is number of heads,
            :math:`S` is number of slices, and :math:`D` is head dimension.

        Notes
        -----
        This method implements the encoding portion of the physics attention mechanism.
        The slice tokens capture learned physical state representations that are used
        as cross-attention context throughout the model.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if x.ndim != 3:
                raise ValueError(
                    f"Expected 3D input tensor (B, N, C), "
                    f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
                )

        # Project inputs onto learned latent spaces
        if self.plus:
            projected_x = self.project_input_onto_slices(x)
            # Transolver++ reuses the same projection for both paths
            feature_projection = projected_x
        else:
            projected_x, feature_projection = self.project_input_onto_slices(x)

        # Project latent representations onto physical state slices: (B, H, N, D) -> (B, H, N, S)
        slice_projections = self.in_project_slice(projected_x)

        # Compute weighted aggregation of features into slice tokens
        _, slice_tokens = self.compute_slices_from_projections(
            slice_projections, feature_projection
        )

        return slice_tokens


class GeometricFeatureProcessor(nn.Module):
    r"""Processes geometric features at a single spatial scale using BQWarp.

    This is a simple, reusable component that handles neighbor querying and
    feature processing for one radius scale. It encapsulates the BQWarp +
    MLP pattern used throughout the model.

    Parameters
    ----------
    radius : float
        Query radius for neighbor search.
    neighbors_in_radius : int
        Maximum number of neighbors within the radius.
    feature_dim : int
        Dimension of the input features to query.
    hidden_dim : int
        Output dimension after MLP processing.

    Forward
    -------
    query_points : torch.Tensor
        Query coordinates of shape :math:`(B, N, 3)` where :math:`B` is batch size
        and :math:`N` is number of query points.
    key_features : torch.Tensor
        Features to query from of shape :math:`(B, N, C)` where :math:`C` is
        ``feature_dim``.

    Outputs
    -------
    torch.Tensor
        Processed features of shape :math:`(B, N, D)` where :math:`D` is ``hidden_dim``.

    See Also
    --------
    :class:`MultiScaleFeatureExtractor` : Uses multiple GeometricFeatureProcessor instances.
    :class:`~physicsnemo.nn.BQWarp` : The ball query operation used internally.

    Examples
    --------
    >>> import torch
    >>> processor = GeometricFeatureProcessor(
    ...     radius=0.1, neighbors_in_radius=16, feature_dim=3, hidden_dim=64
    ... )
    >>> query_points = torch.randn(2, 100, 3)  # (batch, points, xyz)
    >>> key_features = torch.randn(2, 100, 3)  # (batch, points, features)
    >>> output = processor(query_points, key_features)
    >>> output.shape
    torch.Size([2, 100, 64])
    """

    def __init__(
        self,
        radius: float,
        neighbors_in_radius: int,
        feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        # Ball query for neighbor search within radius
        self.bq_warp = BQWarp(radius=radius, neighbors_in_radius=neighbors_in_radius)

        # MLP to process flattened neighbor features
        self.mlp = Mlp(
            in_features=feature_dim * neighbors_in_radius,
            hidden_features=[hidden_dim, hidden_dim // 2],
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

    def forward(
        self,
        query_points: Float[torch.Tensor, "batch points spatial_dim"],
        key_features: Float[torch.Tensor, "batch points features"],
    ) -> Float[torch.Tensor, "batch points hidden_dim"]:
        r"""Query neighbors and process features.

        Parameters
        ----------
        query_points : torch.Tensor
            Query coordinates of shape :math:`(B, N, 3)` where :math:`B` is batch size
            and :math:`N` is number of query points.
        key_features : torch.Tensor
            Features to query from of shape :math:`(B, N, C)` where :math:`C` is the
            feature dimension.

        Returns
        -------
        torch.Tensor
            Processed features of shape :math:`(B, N, D)` where :math:`D` is the
            hidden dimension.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if query_points.ndim != 3:
                raise ValueError(
                    f"Expected 3D query_points tensor (B, N, 3), "
                    f"got {query_points.ndim}D tensor with shape {tuple(query_points.shape)}"
                )
            if key_features.ndim != 3:
                raise ValueError(
                    f"Expected 3D key_features tensor (B, N, C), "
                    f"got {key_features.ndim}D tensor with shape {tuple(key_features.shape)}"
                )

        # Query neighbors within radius: (B, N, K, C)
        _, neighbors = self.bq_warp(query_points, key_features)

        # Flatten neighbor features for MLP: (B, N, K, C) -> (B, N, K*C)
        neighbors_flat = rearrange(neighbors, "b n k c -> b n (k c)")

        # Process through MLP with tanh activation for bounded output
        return torch.nn.functional.tanh(self.mlp(neighbors_flat))


class MultiScaleFeatureExtractor(nn.Module):
    r"""Multi-scale geometric feature extraction with minimal complexity.

    Manages multiple GeometricFeatureProcessor instances for different radii.
    Provides both tokenized context and concatenated local features.

    Parameters
    ----------
    geometry_dim : int
        Dimension of geometry features.
    radii : list[float]
        Radii for multi-scale processing.
    neighbors_in_radius : list[int]
        Neighbors per radius (must have same length as ``radii``).
    hidden_dim : int
        Hidden dimension for processing.
    n_head : int
        Number of attention heads.
    dim_head : int
        Dimension per head.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    slice_num : int, optional
        Number of slices for context tokenization. Default is 64.
    use_te : bool, optional
        Whether to use Transformer Engine. Default is ``True``.
    plus : bool, optional
        Whether to use Transolver++ features. Default is ``False``.

    Forward
    -------
    This class does not implement a standard ``forward`` method. Instead, use:

    - :meth:`extract_context_features`: Get tokenized features for GALE context.
    - :meth:`extract_local_features`: Get concatenated features for local pathway.

    See Also
    --------
    :class:`GeometricFeatureProcessor` : Single-scale processor used by this class.
    :class:`ContextProjector` : Tokenizer used for context features.
    :class:`GlobalContextBuilder` : High-level builder that uses this class.

    Examples
    --------
    >>> import torch
    >>> extractor = MultiScaleFeatureExtractor(
    ...     geometry_dim=3,
    ...     radii=[0.05, 0.25],
    ...     neighbors_in_radius=[8, 32],
    ...     hidden_dim=32,
    ...     n_head=8,
    ...     dim_head=32,
    ... )
    >>> spatial_coords = torch.randn(2, 100, 3)
    >>> geometry = torch.randn(2, 100, 3)
    >>> context_feats = extractor.extract_context_features(spatial_coords, geometry)
    >>> len(context_feats)  # One per scale
    2
    >>> local_feats = extractor.extract_local_features(spatial_coords, geometry)
    >>> local_feats.shape  # Concatenated across scales
    torch.Size([2, 100, 64])
    """

    def __init__(
        self,
        geometry_dim: int,
        radii: list[float],
        neighbors_in_radius: list[int],
        hidden_dim: int,
        n_head: int,
        dim_head: int,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = True,
        plus: bool = False,
    ) -> None:
        super().__init__()
        self.num_scales = len(radii)

        # One processor per scale for geometric feature extraction
        self.processors = nn.ModuleList(
            [
                GeometricFeatureProcessor(
                    radii[i], neighbors_in_radius[i], geometry_dim, hidden_dim
                )
                for i in range(self.num_scales)
            ]
        )

        # One tokenizer per scale for projecting to context space
        self.tokenizers = nn.ModuleList(
            [
                ContextProjector(
                    hidden_dim, n_head, dim_head, dropout, slice_num, use_te, plus
                )
                for _ in range(self.num_scales)
            ]
        )

    def extract_context_features(
        self,
        spatial_coords: Float[torch.Tensor, "batch points spatial_dim"],
        geometry: Float[torch.Tensor, "batch points geometry_dim"],
    ) -> list[Float[torch.Tensor, "batch heads slices dim"]]:
        r"""Extract and tokenize features for context.

        Parameters
        ----------
        spatial_coords : torch.Tensor
            Spatial coordinates of shape :math:`(B, N, 3)`.
        geometry : torch.Tensor
            Geometry features of shape :math:`(B, N, C_{geo})`.

        Returns
        -------
        list[torch.Tensor]
            List of tokenized context features, one per scale, each of shape
            :math:`(B, H, S, D)`.
        """
        return [
            tokenizer(processor(spatial_coords, geometry))
            for processor, tokenizer in zip(self.processors, self.tokenizers)
        ]

    def extract_local_features(
        self,
        spatial_coords: Float[torch.Tensor, "batch points spatial_dim"],
        geometry: Float[torch.Tensor, "batch points geometry_dim"],
    ) -> Float[torch.Tensor, "batch points total_hidden"]:
        r"""Extract and concatenate features for local pathway.

        Parameters
        ----------
        spatial_coords : torch.Tensor
            Spatial coordinates of shape :math:`(B, N, 3)`.
        geometry : torch.Tensor
            Geometry features of shape :math:`(B, N, C_{geo})`.

        Returns
        -------
        torch.Tensor
            Concatenated local features of shape :math:`(B, N, D_{total})` where
            :math:`D_{total}` is ``hidden_dim * num_scales``.
        """
        return torch.cat(
            [processor(geometry, spatial_coords) for processor in self.processors],
            dim=-1,
        )


class GlobalContextBuilder(nn.Module):
    r"""Orchestrates all context construction with a clean, simple interface.

    Manages geometry tokenization, global embedding tokenization, and optional
    multi-scale local features. This is the main entry point for building context
    in the GeoTransolver model.

    Parameters
    ----------
    functional_dims : tuple[int, ...]
        Dimensions of each functional input type.
    geometry_dim : int | None, optional
        Geometry feature dimension. If ``None``, geometry context is disabled.
        Default is ``None``.
    global_dim : int | None, optional
        Global embedding dimension. If ``None``, global context is disabled.
        Default is ``None``.
    radii : list[float], optional
        Radii for local features. Default is ``[0.05, 0.25]``.
    neighbors_in_radius : list[int], optional
        Neighbors per radius. Default is ``[8, 32]``.
    n_hidden_local : int, optional
        Hidden dim for local features. Default is 32.
    n_hidden : int, optional
        Model hidden dimension. Default is 256.
    n_head : int, optional
        Number of attention heads. Default is 8.
    dropout : float, optional
        Dropout rate. Default is 0.0.
    slice_num : int, optional
        Number of slices for tokenization. Default is 32.
    use_te : bool, optional
        Whether to use Transformer Engine. Default is ``True``.
    plus : bool, optional
        Whether to use Transolver++ features. Default is ``False``.
    include_local_features : bool, optional
        Enable local feature extraction. Default is ``False``.

    Forward
    -------
    This class does not implement a standard ``forward`` method. Instead, use
    :meth:`build_context` to construct context and local features.

    See Also
    --------
    :class:`ContextProjector` : Used for tokenizing geometry and global embeddings.
    :class:`MultiScaleFeatureExtractor` : Used for multi-scale local features.
    :class:`~physicsnemo.experimental.models.geotransolver.GeoTransolver` : Main model that uses this builder.

    Examples
    --------
    >>> import torch
    >>> builder = GlobalContextBuilder(
    ...     functional_dims=(64,),
    ...     geometry_dim=3,
    ...     global_dim=16,
    ...     n_hidden=256,
    ...     n_head=8,
    ... )
    >>> local_embeddings = (torch.randn(2, 100, 64),)
    >>> geometry = torch.randn(2, 100, 3)
    >>> global_embedding = torch.randn(2, 1, 16)
    >>> context, local_feats = builder.build_context(
    ...     local_embeddings, None, geometry, global_embedding
    ... )
    >>> context.shape
    torch.Size([2, 8, 32, 64])
    """

    def __init__(
        self,
        functional_dims: tuple[int, ...],
        geometry_dim: int | None = None,
        global_dim: int | None = None,
        radii: list[float] | None = None,
        neighbors_in_radius: list[int] | None = None,
        n_hidden_local: int = 32,
        n_hidden: int = 256,
        n_head: int = 8,
        dropout: float = 0.0,
        slice_num: int = 32,
        use_te: bool = True,
        plus: bool = False,
        include_local_features: bool = False,
    ) -> None:
        super().__init__()

        # Set defaults for mutable arguments
        if radii is None:
            radii = [0.05, 0.25]
        if neighbors_in_radius is None:
            neighbors_in_radius = [8, 32]

        dim_head = n_hidden // n_head
        context_dim = 0

        # Multi-scale extractors for local features (one per functional dim)
        if geometry_dim is not None and include_local_features:
            self.local_extractors = nn.ModuleList(
                [
                    MultiScaleFeatureExtractor(
                        geometry_dim,
                        radii,
                        neighbors_in_radius,
                        n_hidden_local,
                        n_head,
                        dim_head,
                        dropout,
                        slice_num,
                        use_te,
                        plus,
                    )
                    for _ in functional_dims
                ]
            )
            context_dim += dim_head * len(radii) * len(functional_dims)
        else:
            self.local_extractors = None

        # Geometry tokenizer for global geometry context
        if geometry_dim is not None:
            self.geometry_tokenizer = ContextProjector(
                geometry_dim, n_head, dim_head, dropout, slice_num, use_te, plus
            )
            context_dim += dim_head
        else:
            self.geometry_tokenizer = None

        # Global embedding tokenizer
        if global_dim is not None:
            self.global_tokenizer = ContextProjector(
                global_dim, n_head, dim_head, dropout, slice_num, use_te, plus
            )
            context_dim += dim_head
        else:
            self.global_tokenizer = None

        self._context_dim = context_dim

    def get_context_dim(self) -> int:
        r"""Return total context dimension.

        Returns
        -------
        int
            Total dimension of the concatenated context features.
        """
        return self._context_dim

    def build_context(
        self,
        local_embeddings: tuple[Float[torch.Tensor, "batch tokens features"], ...],
        local_positions: (
            tuple[Float[torch.Tensor, "batch tokens spatial_dim"], ...] | None
        ),
        geometry: Float[torch.Tensor, "batch tokens geometry_dim"] | None = None,
        global_embedding: Float[torch.Tensor, "batch global_tokens global_dim"]
        | None = None,
    ) -> tuple[
        Float[torch.Tensor, "batch heads slices context_dim"] | None,
        list[Float[torch.Tensor, "batch tokens local_features"]] | None,
    ]:
        r"""Build all context and local features.

        Parameters
        ----------
        local_embeddings : tuple[torch.Tensor, ...]
            Input embeddings, each of shape :math:`(B, N, C_i)` where :math:`B` is
            batch size, :math:`N` is number of tokens, and :math:`C_i` is the feature
            dimension for input type :math:`i`.
        local_positions : tuple[torch.Tensor, ...] | None
            Local positions, each of shape :math:`(B, N, 3)`. These are used to query
            neighbors for local features. Required if ``include_local_features=True``.
        geometry : torch.Tensor | None, optional
            Geometry features of shape :math:`(B, N, C_{geo})`. Default is ``None``.
        global_embedding : torch.Tensor | None, optional
            Global embedding of shape :math:`(B, N_g, C_g)`. Default is ``None``.

        Returns
        -------
        tuple[torch.Tensor | None, list[torch.Tensor] | None]
            - ``context``: Concatenated context tensor of shape :math:`(B, H, S, D_c)`
              where :math:`D_c` is the total context dimension, or ``None`` if no
              context sources are provided.
            - ``local_features``: List of local feature tensors, one per input type,
              each of shape :math:`(B, N, D_l)`, or ``None`` if local features are
              disabled.

        Raises
        ------
        ValueError
            If ``local_positions`` is ``None`` but local features are enabled.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            if len(local_embeddings) == 0:
                raise ValueError("Expected non-empty tuple of local embeddings")
            for i, emb in enumerate(local_embeddings):
                if emb.ndim != 3:
                    raise ValueError(
                        f"Expected 3D local_embedding tensor (B, N, C) at index {i}, "
                        f"got {emb.ndim}D tensor with shape {tuple(emb.shape)}"
                    )

        context_parts = []
        local_features = None

        if local_positions is None and self.local_extractors is not None:
            raise ValueError(
                "Local positions are required if local features are enabled."
            )

        # Extract multi-scale features if enabled
        if self.local_extractors is not None and geometry is not None:
            local_features = []
            for i, embedding in enumerate(local_embeddings):
                spatial_coords = local_positions[i]  # Extract coordinates

                # Get tokenized context features from multi-scale extractor
                context_feats = self.local_extractors[i].extract_context_features(
                    spatial_coords, geometry
                )
                context_parts.extend(context_feats)

                # Get concatenated local features for skip connection
                local_feats = self.local_extractors[i].extract_local_features(
                    spatial_coords, geometry
                )
                local_features.append(local_feats)

        # Tokenize geometry features
        if self.geometry_tokenizer is not None and geometry is not None:
            context_parts.append(self.geometry_tokenizer(geometry))

        # Tokenize global embedding
        if self.global_tokenizer is not None and global_embedding is not None:
            context_parts.append(self.global_tokenizer(global_embedding))

        # Concatenate all context features along the last dimension
        context = torch.cat(context_parts, dim=-1) if context_parts else None

        return context, local_features
