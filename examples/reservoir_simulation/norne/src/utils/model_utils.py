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

"""
=====================================================================
                    SHARED MODEL UTILITIES
=====================================================================

Shared factory functions and model classes for FNO and Transolver
surrogate models used across forward, inverse and compare sub-packages.

Public API
----------
FNOModel               - PhysicsNeMo Module wrapping a Fourier Neural Operator.
TransolverModel        - PhysicsNeMo Module wrapping a Transolver.
create_fno_model(...)  - Factory for FNOModel.
create_transolver_model(...) - Factory for TransolverModel.
load_modell(...)       - Load model weights from a checkpoint file.

@Author : Clement Etienam
"""

# Standard Library
from collections import OrderedDict

# Third-party Libraries
import torch
import torch.nn as nn
from torch import Tensor
from physicsnemo.models.fno import FNO
from physicsnemo.models.module import Module
from physicsnemo.models.transolver import Transolver

from utils.logging_utils import setup_logging


class ConvFullyConnectedDecoder(nn.Module):
    """
    Conv-based decoder mirroring PhysicsNeMo Sym's ConvFullyConnectedArch.

    Operates on a grid of shape (B, C_latent, *spatial). Internally flattens
    the spatial dims to 1D and applies Conv1d layers with kernel_size=1.
    Mathematically a per-cell MLP, but matches Sym's exact data flow:
    no weight norm, SiLU activation, Xavier-uniform init.

    Defaults (6 layers x 512 width) match Sym's ConvFullyConnectedArch
    defaults, giving a much higher-capacity per-cell mapping than a
    1-layer 32-wide MLP. Important for sharp-feature fields like pressure
    near wells and water saturation displacement fronts.

    Uses .reshape() rather than .view() for the spatial flatten/unflatten so
    the decoder is robust to non-contiguous input (FNO spectral encoder
    outputs are typically non-contiguous after FFT operations).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_size: int = 256,
        nr_layers: int = 6,
        activation_fn=nn.SiLU,
    ):
        super().__init__()

        layers = []
        layer_in = in_channels
        for _ in range(nr_layers):
            layers.append(nn.Conv1d(layer_in, layer_size, kernel_size=1))
            layers.append(activation_fn())
            layer_in = layer_size

        # Final projection: kernel_size=1, no activation
        layers.append(nn.Conv1d(layer_in, out_channels, kernel_size=1))

        self.net = nn.Sequential(*layers)

        # Sym-style Xavier-uniform init to match FCLayer/Conv1dFCLayer behaviour
        self.apply(self._sym_style_init)

    @staticmethod
    def _sym_style_init(m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C_latent, *spatial)
        x_shape = list(x.size())
        # Flatten spatial dims -> (B, C_latent, prod(spatial))
        # Use reshape (not view) to handle non-contiguous FFT outputs.
        x = x.reshape(x.shape[0], x.shape[1], -1)
        y = self.net(x)
        # Reshape back -> (B, out_channels, *spatial)
        x_shape[1] = y.shape[1]
        return y.reshape(x_shape)


class FNOModel(Module):
    """F N O Model."""

    def __init__(
        self,
        input_dim,
        steppi,
        output_shape,
        device,
        num_layers=4,
        decoder_layers=6,
        decoder_layer_size=256,
        dimension=3,
        latent_channels=32,
        num_fno_layers=4,
        padding=8,
        num_fno_modes=16,
    ):
        super().__init__()

        # Build the FNO. Pass minimal decoder args because we overwrite
        # the decoder immediately after construction.
        self.fno = FNO(
            in_channels=input_dim,
            out_channels=output_shape * steppi,
            decoder_layers=1,
            decoder_layer_size=decoder_layer_size,
            decoder_activation_fn="silu",
            dimension=dimension,
            latent_channels=latent_channels,
            num_fno_layers=num_fno_layers,
            padding=padding,
            num_fno_modes=num_fno_modes,
        ).to(torch.device(device))

        # Replace the FNO's pointwise FullyConnected decoder with the
        # Sym-style Conv1d-based decoder.
        self.fno.decoder_net = ConvFullyConnectedDecoder(
            in_channels=latent_channels,
            out_channels=output_shape * steppi,
            layer_size=decoder_layer_size,
            nr_layers=decoder_layers,
            activation_fn=nn.SiLU,
        ).to(torch.device(device))

        # Patch the FNO forward to skip grid_to_points / points_to_grid
        # reshaping — the conv decoder operates on grids natively.
        # Pass in_channels and dimension explicitly because some versions
        # of FNO don't expose them as instance attributes.
        self._patch_fno_forward(in_channels=input_dim, dimension=dimension)

        self.meta = type("", (), {})()
        self.meta.name = "fno_model"

    def _patch_fno_forward(self, in_channels: int, dimension: int):
        """Bypass pointwise reshape path; conv decoder takes grid input."""
        fno = self.fno

        def new_forward(x):
            if not torch.compiler.is_compiling():
                expected_ndim = dimension + 2
                if x.ndim != expected_ndim or x.shape[1] != in_channels:
                    raise ValueError(
                        f"Expected {expected_ndim}D input "
                        f"(B, {in_channels}, ...) for {dimension}D FNO, "
                        f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
                    )
            # Spectral encoder -> (B, latent_channels, *spatial)
            y_latent = fno.spec_encoder(x)
            # Conv decoder operates directly on the grid
            return fno.decoder_net(y_latent)

        fno.forward = new_forward

    def forward(self, x):
        return self.fno(x)


def create_fno_model(
    input_dim,
    steppi,
    output_shape,
    device,
    num_layers=4,
    decoder_layers=6,
    decoder_layer_size=256,
    dimension=3,
    latent_channels=32,
    num_fno_layers=4,
    padding=8,
    num_fno_modes=16,
):
    """
    Create a Fourier Neural Operator (FNO) with a Sym-equivalent
    ConvFullyConnected decoder (6-layer x 512-wide Conv1d, kernel_size=1,
    no weight norm, SiLU, Xavier-uniform init).

    Defaults reproduce PhysicsNeMo Sym's ConvFullyConnectedArch behaviour
    and give a much higher-capacity per-cell decoder mapping than the
    previous 1 x 32 default. Helps fields with sharp local features such
    as pressure near wells and water saturation fronts.

    Parameters
    ----------
    input_dim : int
        Number of input channels.
    steppi : int
        Number of timesteps in the output.
    output_shape : int
        Number of distinct output fields.
    device : str
        Device to place the model on.
    num_layers : int, optional
        Legacy/unused, kept for backward compatibility. Default 4.
    decoder_layers : int, optional
        Number of conv decoder layers. Default 6.
    decoder_layer_size : int, optional
        Width of each conv decoder layer. Default 512.
    dimension : int, optional
        Spatial dimensionality (1, 2, or 3). Default 3.
    latent_channels : int, optional
        Latent channels from the spectral encoder. Default 32.
    num_fno_layers : int, optional
        Number of spectral conv layers. Default 4.
    padding : int, optional
        FFT padding. Default 8.
    num_fno_modes : int, optional
        Number of Fourier modes retained. Default 16.

    Returns
    -------
    FNOModel
    """
    if dimension not in [1, 2, 3]:
        raise ValueError(f"Invalid dimension: {dimension}. Must be 1, 2 or 3.")
    return FNOModel(
        input_dim=input_dim,
        steppi=steppi,
        output_shape=output_shape,
        device=device,
        num_layers=num_layers,
        decoder_layers=decoder_layers,
        decoder_layer_size=decoder_layer_size,
        dimension=dimension,
        latent_channels=latent_channels,
        num_fno_layers=num_fno_layers,
        padding=padding,
        num_fno_modes=num_fno_modes,
    )


class TransolverModel(Module):
    """Transolver Model."""

    def __init__(
        self,
        functional_dim,
        out_dim,
        device,
        embedding_dim=None,
        n_layers=4,
        n_hidden=60,
        dropout=0.0,
        n_head=12,
        act="gelu",
        mlp_ratio=4,
        slice_num=32,
        unified_pos=True,
        ref=8,
        structured_shape=(46, 112),
        use_te=True,
        time_input=False,
    ):
        """Initialise a TransolverModel wrapping a PhysicsNeMo Transolver backbone.

        Parameters
        ----------
        functional_dim : int
            Dimension of input function values (excluding embeddings).
        out_dim : int
            Output feature dimension per spatial point.
        device : str
            Device string for model placement (e.g., ``"cuda:0"`` or ``"cpu"``).
        embedding_dim : int or None, optional
            Spatial embedding dimension. Default is None.
        n_layers : int, optional
            Number of PhysicsAttention transformer layers. Default is 4.
        n_hidden : int, optional
            Hidden dimension of the transformer. Default is 60.
        dropout : float, optional
            Dropout probability. Default is 0.0.
        n_head : int, optional
            Number of attention heads. Default is 12.
        act : str, optional
            Activation function name. Default is ``"gelu"``.
        mlp_ratio : int, optional
            MLP hidden-to-model-dim ratio. Default is 4.
        slice_num : int, optional
            Number of physics-attention slices. Default is 32.
        unified_pos : bool, optional
            Whether to use unified positional embeddings. Default is True.
        ref : int, optional
            Reference dimension for unified positions. Default is 8.
        structured_shape : tuple of int, optional
            Shape of the structured latent space. Default is ``(46, 112)``.
        use_te : bool, optional
            Whether to use the transformer-engine backend. Default is True.
        time_input : bool, optional
            Whether to include time embeddings. Default is False.
        """
        super().__init__()
        self.transolver = Transolver(
            functional_dim=functional_dim,
            out_dim=out_dim,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            n_head=n_head,
            act=act,
            mlp_ratio=mlp_ratio,
            slice_num=slice_num,
            unified_pos=unified_pos,
            ref=ref,
            structured_shape=structured_shape,
            use_te=use_te,
            time_input=time_input,
        ).to(torch.device(device))
        self.meta = type("", (), {})()
        self.meta.name = "transolver_model"
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, nz, nx, ny, C)
        returns: (B, nz, nx, ny, out_dim)
        """
        B, nz, nx, ny, C = x.shape

        # Flatten the 3D field into 2D slices to feed PhysicsNeMo Transolver
        # x_2d: (B * nz, nx, ny, C)
        x_2d = x.reshape(B * nz, nx, ny, C)

        out_2d = self.transolver(x_2d)
        # out_2d should be (B * nz, nx, ny, out_dim)

        return out_2d.reshape(B, nz, nx, ny, self.out_dim)


def create_transolver_model(
    functional_dim,
    out_dim,
    device,
    embedding_dim=None,
    n_layers=4,
    n_hidden=60,
    dropout=0.0,
    n_head=12,
    act="gelu",
    mlp_ratio=2,
    slice_num=24,
    unified_pos=True,
    ref=8,
    structured_shape=(46, 112),
    use_te=True,
    time_input=False,
):
    """
    Create a Transolver model wrapped in a compatible PhysicsNeMo Module.

    Parameters:
    -----------
    functional_dim : int
        The dimension of the input values, not including any embeddings.
    out_dim : int
        The dimension of the output of the model.
    device : str
        Device to create the model on ('cpu' or 'cuda').
    embedding_dim : int | None, optional
        The spatial dimension of the input data embeddings. Default is None.
    n_layers : int, optional
        The number of transformer PhysicsAttention layers. Default is 4.
    n_hidden : int, optional
        The hidden dimension of the transformer. Default is 60.
    dropout : float, optional
        The dropout rate. Default is 0.0.
    n_head : int, optional
        The number of attention heads. Default is 12.
    act : str, optional
        The activation function. Default is "gelu".
    mlp_ratio : int, optional
        The ratio of hidden dimension in the MLP. Default is 2.
    slice_num : int, optional
        The number of slices in the PhysicsAttention layers. Default is 24.
    unified_pos : bool, optional
        Whether to use unified positional embeddings. Default is True.
    ref : int, optional
        The reference dimension size when using unified positions. Default is 8.
    structured_shape : tuple, optional
        The shape of the latent space for structured data. Default is (46, 112).
    use_te : bool, optional
        Whether to use transformer engine backend. Default is True.
    time_input : bool, optional
        Whether to include time embeddings. Default is False.

    Returns:
    --------
    transolver_model : TransolverModel
        Initialized Transolver model ready for inference or training.
    """
    # Validate arguments
    if n_hidden % n_head != 0:
        raise ValueError(
            f"n_hidden ({n_hidden}) must be divisible by n_head ({n_head})"
        )

    if unified_pos and structured_shape is None:
        raise ValueError("structured_shape must be provided when unified_pos=True")

    if structured_shape is not None and len(structured_shape) not in [2, 3]:
        raise ValueError(f"structured_shape must be 2D or 3D, got {structured_shape}")

    return TransolverModel(
        functional_dim=functional_dim,
        out_dim=out_dim,
        device=device,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        n_hidden=n_hidden,
        dropout=dropout,
        n_head=n_head,
        act=act,
        mlp_ratio=mlp_ratio,
        slice_num=slice_num,
        unified_pos=unified_pos,
        ref=ref,
        structured_shape=structured_shape,
        use_te=use_te,
        time_input=time_input,
    )


def load_modell(model, model_path, is_distributed, device, express, namee):
    """
    Loads a PyTorch model from a checkpoint.

    Parameters:
    -----------
    model : nn.Module
        The PyTorch model instance.
    model_path : str
        Path to the saved model file.
    is_distributed : bool
        Whether the model was trained in a distributed setting.
    device : str
        The device to load the model onto ('cpu' or 'cuda').
    express : int
        If 1, load as a plain state_dict; otherwise load from a checkpoint dict
        with key ``"surrogate_state_dict"``.
    namee : str
        Unused in this canonical form; present for interface consistency.

    Returns:
    --------
    model : nn.Module
        The loaded model.
    """
    logger = setup_logging(__name__)
    logger.info(f"Loading model from: {model_path}")

    if express == 1:
        state_dict = torch.load(model_path, map_location=device)

        # Handle Distributed Data Parallel (Remove `module.` prefix if needed)
        if is_distributed == 1:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:] if k.startswith("module.") else k] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        # Move model to correct device & set to eval mode
        model = model.to(device)
        model.eval()
    else:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["surrogate_state_dict"]
        # Handle Distributed Data Parallel (Remove `module.` prefix if needed)
        if is_distributed == 1:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:] if k.startswith("module.") else k] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        # Move model to correct device & set to eval mode
        model = model.to(device)
        model.eval()

    return model
