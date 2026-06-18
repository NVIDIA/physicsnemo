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
                            MACHINE EXTRACTION
=====================================================================

This module provides machine learning model extraction capabilities for
reservoir simulation forward modeling. It includes functions for creating
FNO models, GNN models, composite models, and managing model operations.

Key Features:
- FNO (Fourier Neural Operator) model creation and management
- GNN (Graph Neural Network) model implementation
- Composite model orchestration
- Model saving and loading utilities
- MLflow integration for experiment tracking

Usage:
    from forward.machine_extract import (
        FNOModel,
        create_fno_model,
        CompositeModel,
        CompositeOptimizer,
        GNNModel,
        create_gnn_model
    )

@Author : Clement Etienam
"""

import os
import sys
import time
import logging
import shutil

from omegaconf import DictConfig
import torch

import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient

from physicsnemo.models.fno import FNO
from physicsnemo.models.module import Module
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.distributed import DistributedManager
from physicsnemo.models.transolver import Transolver

import torch.nn as nn
from torch import Tensor


def setup_logging() -> logging.Logger:
    """Configure and return the main logger with green INFO console output."""
    logger = logging.getLogger("forward problem")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on re-entry

    formatter = logging.Formatter(" %(asctime)s - %(levelname)s - %(message)s")

    # File handler — plain, no colors (keeps log file clean)
    f_handler = logging.FileHandler(filename="read_vectors.log", mode="w")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    # Console handler — colored
    class _ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: "\033[0;36m",  # cyan
            logging.INFO: "\033[0;32m",  # green
            logging.WARNING: "\033[1;33m",  # yellow
            logging.ERROR: "\033[0;31m",  # red
            logging.CRITICAL: "\033[1;31m",  # bold red
        }
        RESET = "\033[0m"

        def format(self, record):
            msg = super().format(record)
            if sys.stdout.isatty():
                color = self.COLORS.get(record.levelno, "")
                return f"{color}{msg}{self.RESET}"
            return msg

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(
        _ColorFormatter(" %(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(c_handler)

    return logger


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
        decoder_layers=6,  # Sym default
        decoder_layer_size=256,  # Sym default
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
    decoder_layers=6,  # Sym default
    decoder_layer_size=256,  # Sym default
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
        The number of transformer PhysicsAttention layers. Default is 8.
    n_hidden : int, optional
        The hidden dimension of the transformer. Default is 256.
    dropout : float, optional
        The dropout rate. Default is 0.0.
    n_head : int, optional
        The number of attention heads. Default is 8.
    act : str, optional
        The activation function. Default is "gelu".
    mlp_ratio : int, optional
        The ratio of hidden dimension in the MLP. Default is 4.
    slice_num : int, optional
        The number of slices in the PhysicsAttention layers. Default is 32.
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


def create_transolver_model_batch(
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
    slice_num=32,
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
        The number of transformer PhysicsAttention layers. Default is 8.
    n_hidden : int, optional
        The hidden dimension of the transformer. Default is 256.
    dropout : float, optional
        The dropout rate. Default is 0.0.
    n_head : int, optional
        The number of attention heads. Default is 8.
    act : str, optional
        The activation function. Default is "gelu".
    mlp_ratio : int, optional
        The ratio of hidden dimension in the MLP. Default is 4.
    slice_num : int, optional
        The number of slices in the PhysicsAttention layers. Default is 32.
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


class CompositeModelo(Module):
    """Composite Modelo."""

    def __init__(self, MODELS, steppi, output_variables, model_type="FNO"):
        """Initialise a CompositeModelo with per-phase surrogate models.

        Parameters
        ----------
        MODELS : dict
            Dictionary mapping uppercase phase keys (``"PRESSURE"``, ``"SGAS"``,
            ``"SATURATION"``, ``"SOIL"``, ``"PEACEMANN"``) to torch.nn.Module instances.
        steppi : int
            Number of output time steps.
        output_variables : list of str
            Subset of phase keys present in ``MODELS`` that should be activated.
        model_type : str, optional
            Backend type for all reservoir heads: ``"FNO"`` or ``"Transolver"``.
            Default is ``"FNO"``.
        """
        super().__init__()
        self.output_variables = output_variables
        self.model_type = model_type
        self.steppi = steppi

        self.models = {}
        self.model_types = {}

        if "PRESSURE" in self.output_variables:
            self.surrogate_pressure = MODELS["PRESSURE"]
            self.models["pressure"] = self.surrogate_pressure
            self.model_types["pressure"] = model_type

        if "SGAS" in self.output_variables:
            self.surrogate_gas = MODELS["SGAS"]
            self.models["gas"] = self.surrogate_gas
            self.model_types["gas"] = model_type

        if "SWAT" in self.output_variables:
            self.surrogate_saturation = MODELS["SATURATION"]
            self.models["saturation"] = self.surrogate_saturation
            self.model_types["saturation"] = model_type

        if "SOIL" in self.output_variables:
            self.surrogate_oil = MODELS["SOIL"]
            self.models["oil"] = self.surrogate_oil
            self.model_types["oil"] = model_type

        self.surrogate_peacemann = MODELS["PEACEMANN"]
        self.models["peacemann"] = self.surrogate_peacemann
        self.model_types["peacemann"] = "FNO"

        # ---- one persistent CUDA stream per head ----
        # Streams allow the GPU to overlap independent kernel launches,
        # so pressure/gas/water/oil forward passes can execute concurrently
        # rather than serialising on the default stream.
        # Falls back gracefully if CUDA is not available.
        if torch.cuda.is_available():
            self._streams = {
                key: torch.cuda.Stream()
                for key in ("pressure", "gas", "saturation", "oil", "peacemann")
            }
        else:
            self._streams = {}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _fno_forward(self, input_tensor, model):
        """Run a direct FNO forward pass with the tensor already in the correct shape.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor compatible with the FNO model's expected shape.
        model : torch.nn.Module
            FNO model instance.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """
        return model(input_tensor)

    def _transolver_forward(self, input_tensor, model):
        """
        Transolver forward with chunked batchxtime processing.
        Input:  (B, steppi, nz, nx, ny, C)
        Output: (B, steppi, nz, nx, ny)
        """
        B, steppi, nz, nx, ny, _C = input_tensor.shape
        batch_chunk_size = 2
        time_chunk_size = 3
        output = torch.zeros(B, steppi, nz, nx, ny, device=input_tensor.device)

        chunk_counter = 0
        for batch_start in range(0, B, batch_chunk_size):
            batch_end = min(batch_start + batch_chunk_size, B)
            for time_start in range(0, steppi, time_chunk_size):
                time_end = min(time_start + time_chunk_size, steppi)
                chunk_counter += 1

                input_chunk = input_tensor[batch_start:batch_end, time_start:time_end]
                b, t, nz_c, nx_c, ny_c, C_c = input_chunk.shape

                x_5d = input_chunk.view(b * t, nz_c, nx_c, ny_c, C_c)
                pred_5d = model(x_5d).squeeze(-1)
                output[batch_start:batch_end, time_start:time_end] = pred_5d.view(
                    b, t, nz_c, nx_c, ny_c
                )

                del input_chunk, x_5d, pred_5d
                if chunk_counter % 3 == 0:
                    torch.cuda.empty_cache()

        return output

    def _run_head(self, input_tensor, model, model_key):
        """Dispatch a single surrogate head to its registered backend.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor to pass to the model.
        model : torch.nn.Module
            Surrogate model instance for this head.
        model_key : str
            Key identifying the head (e.g., ``"pressure"``); used to look up the backend type.

        Returns
        -------
        torch.Tensor
            Output tensor from the dispatched backend.
        """
        if self.model_types.get(model_key, "FNO") == "FNO":
            return self._fno_forward(input_tensor, model)
        return self._transolver_forward(input_tensor, model)

    def _run_head_on_stream(self, input_tensor, model, model_key):
        """Run a surrogate head asynchronously on its dedicated CUDA stream.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor to pass to the model.
        model : torch.nn.Module
            Surrogate model instance for this head.
        model_key : str
            Key identifying the head; used to select the CUDA stream.

        Returns
        -------
        torch.Tensor
            Output tensor produced asynchronously on the head's CUDA stream.
        """
        stream = self._streams.get(model_key)
        if stream is None:
            # CPU fallback — no streams
            return self._run_head(input_tensor, model, model_key)

        with torch.cuda.stream(stream):
            return self._run_head(input_tensor, model, model_key)

    # ------------------------------------------------------------------ #
    # Public forward                                                       #
    # ------------------------------------------------------------------ #

    def forward(self, input_tensor, mode="both", **kwargs):
        """
        Forward pass for the composite model.

        mode : "pressure" | "gas" | "saturation" | "oil" | "peacemann"
             | "all"   — runs all 4 reservoir heads concurrently on separate streams
             | "both"  — legacy alias for "all"

        When mode == "all" / "both" the four reservoir heads are dispatched onto
        dedicated CUDA streams so their kernels can overlap.  The default stream
        is synchronised before returning so callers see fully-computed tensors.
        """
        outputs = {}

        # ---- single-head modes: synchronous, on the default stream ----
        if mode == "pressure":
            outputs["pressure"] = self._run_head(
                input_tensor, self.surrogate_pressure, "pressure"
            )
            return outputs

        if mode == "gas":
            outputs["gas"] = self._run_head(input_tensor, self.surrogate_gas, "gas")
            return outputs

        if mode == "saturation":
            outputs["saturation"] = self._run_head(
                input_tensor, self.surrogate_saturation, "saturation"
            )
            return outputs

        if mode == "oil":
            outputs["oil"] = self._run_head(input_tensor, self.surrogate_oil, "oil")
            return outputs

        if mode == "peacemann":
            outputs["peacemann"] = self._run_head(
                input_tensor, self.surrogate_peacemann, "peacemann"
            )
            return outputs

        # ---- multi-head mode: dispatch each active head onto its own stream ----
        # "both" retained as legacy alias for "all"
        if mode in ("all", "both"):
            active_heads = []

            if "PRESSURE" in self.output_variables:
                active_heads.append(("pressure", self.surrogate_pressure))
            if "SGAS" in self.output_variables:
                active_heads.append(("gas", self.surrogate_gas))
            if "SWAT" in self.output_variables:
                active_heads.append(("saturation", self.surrogate_saturation))
            if "SOIL" in self.output_variables:
                active_heads.append(("oil", self.surrogate_oil))

            # Launch all reservoir heads asynchronously
            for key, model in active_heads:
                outputs[key] = self._run_head_on_stream(input_tensor, model, key)

            # Sync all head streams back to the default stream before returning
            if self._streams:
                default_stream = torch.cuda.current_stream()
                for key, _ in active_heads:
                    default_stream.wait_stream(self._streams[key])

            return outputs

        # ---- unknown mode: raise clearly rather than silently returning {} ----
        raise ValueError(
            f"CompositeModel.forward: unknown mode '{mode}'. "
            "Valid modes: 'pressure', 'gas', 'saturation', 'oil', 'peacemann', 'all', 'both'."
        )


class CompositeModel(Module):
    """Composite Model."""

    def __init__(self, MODELS, steppi, output_variables, model_type="FNO"):
        """Initialise a CompositeModel with per-phase surrogate models.

        Parameters
        ----------
        MODELS : dict
            Dictionary mapping uppercase phase keys to torch.nn.Module instances.
        steppi : int
            Number of output time steps.
        output_variables : list of str
            Subset of phase keys to activate from ``MODELS``.
        model_type : str, optional
            Backend type for reservoir heads: ``"FNO"`` or ``"Transolver"``.
            Default is ``"FNO"``.
        """
        super().__init__()
        self.output_variables = output_variables
        self.model_type = model_type  # Main model type for tracking
        self.steppi = steppi

        # Store models and their types
        self.models = {}
        self.model_types = {}  # Track each model's type individually

        if "PRESSURE" in self.output_variables:
            self.surrogate_pressure = MODELS["PRESSURE"]
            self.models["pressure"] = self.surrogate_pressure
            self.model_types["pressure"] = model_type  # FNO or Transolver

        if "SGAS" in self.output_variables:
            self.surrogate_gas = MODELS["SGAS"]
            self.models["gas"] = self.surrogate_gas
            self.model_types["gas"] = model_type

        if "SWAT" in self.output_variables:
            self.surrogate_saturation = MODELS["SATURATION"]
            self.models["saturation"] = self.surrogate_saturation
            self.model_types["saturation"] = model_type

        if "SOIL" in self.output_variables:
            self.surrogate_oil = MODELS["SOIL"]
            self.models["oil"] = self.surrogate_oil
            self.model_types["oil"] = model_type

        self.surrogate_peacemann = MODELS["PEACEMANN"]
        self.models["peacemann"] = self.surrogate_peacemann
        self.model_types["peacemann"] = "FNO"  # Peacemann is always FNO

    def _handle_3d_to_2d(self, input_tensor, model, model_key):
        """Route a tensor to FNO or Transolver with chunked batch-time processing.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor of shape ``(B, steppi, nz, nx, ny, C)`` for Transolver,
            or any compatible shape for FNO.
        model : torch.nn.Module
            Surrogate model instance.
        model_key : str
            Key used to determine the backend type (``"FNO"`` or otherwise).

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, steppi, nz, nx, ny)`` for Transolver,
            or the FNO model's native output shape.
        """
        model_type = self.model_types.get(model_key, "FNO")
        steppi = self.steppi

        if model_type == "FNO":
            # FNO can handle 3D directly, or input is already 2D
            return model(input_tensor)
        B, _steppi_no, nz, nx, ny, _C = input_tensor.shape

        batch_chunk_size = 2
        time_chunk_size = 3

        # Output: (B, steppi, nz, nx, ny)
        output = torch.zeros(B, steppi, nz, nx, ny, device=input_tensor.device)

        (
            (B + batch_chunk_size - 1)
            // batch_chunk_size
            * (steppi + time_chunk_size - 1)
            // time_chunk_size
        )
        current_chunk = 0

        for batch_start in range(0, B, batch_chunk_size):
            batch_end = min(batch_start + batch_chunk_size, B)

            for time_start in range(0, steppi, time_chunk_size):
                time_end = min(time_start + time_chunk_size, steppi)

                current_chunk += 1

                # input_chunk: (b, t, nz, nx, ny, C)
                input_chunk = input_tensor[batch_start:batch_end, time_start:time_end]
                b, t, nz_c, nx_c, ny_c, C_c = input_chunk.shape

                # Merge batch and time for TransolverModel:
                # x_5d: (b * t, nz, nx, ny, C)
                x_5d = input_chunk.view(b * t, nz_c, nx_c, ny_c, C_c)

                # model(...) returns (b * t, nz, nx, ny, out_dim)
                pred_5d = model(x_5d)
                # Remove out_dim=1 and restore (b, t, nz, nx, ny)
                pred_5d = pred_5d.squeeze(-1)
                pred_3d = pred_5d.view(b, t, nz_c, nx_c, ny_c)

                output[batch_start:batch_end, time_start:time_end] = pred_3d

                # Clean up
                del input_chunk, x_5d, pred_5d, pred_3d
                # if current_chunk % 3 == 0:
                # torch.cuda.empty_cache()

        return output

    def forward(self, input_tensor, mode="both", **kwargs):
        """
        Forward pass for the composite model.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor for the model.
        mode : str
            Which model to use: ``"pressure"``, ``"saturation"``, ``"gas"``,
            ``"oil"``, ``"peacemann"``, or ``"both"``.

        Returns
        -------
        dict
            Outputs from the selected model(s).
        """
        outputs = {}

        if mode in ["pressure", "both"]:
            outputs["pressure"] = self._handle_3d_to_2d(
                input_tensor, self.surrogate_pressure, "pressure"
            )

        if mode in ["gas", "both"]:
            outputs["gas"] = self._handle_3d_to_2d(
                input_tensor, self.surrogate_gas, "gas"
            )

        if mode in ["saturation", "both"]:
            outputs["saturation"] = self._handle_3d_to_2d(
                input_tensor, self.surrogate_saturation, "saturation"
            )

        if mode in ["oil", "both"]:
            outputs["oil"] = self._handle_3d_to_2d(
                input_tensor, self.surrogate_oil, "oil"
            )

        if mode in ["peacemann", "both"]:
            outputs["peacemann"] = self._handle_3d_to_2d(
                input_tensor, self.surrogate_peacemann, "peacemann"
            )

        return outputs


class CompositeModelBatch(Module):
    """Composite Model Batch."""

    def __init__(self, MODELS, steppi, output_variables, model_type="FNO"):
        """Initialise a CompositeModelBatch with per-phase surrogate models.

        Parameters
        ----------
        MODELS : dict
            Dictionary mapping uppercase phase keys to torch.nn.Module instances.
        steppi : int
            Number of output time steps.
        output_variables : list of str
            Subset of phase keys to activate from ``MODELS``.
        model_type : str, optional
            Backend type for reservoir heads: ``"FNO"`` or ``"Transolver"``.
            Default is ``"FNO"``.
        """
        super().__init__()
        self.output_variables = output_variables
        self.model_type = model_type  # Main model type for tracking
        self.steppi = steppi

        # Store models and their types
        self.models = {}
        self.model_types = {}  # Track each model's type individually

        if "PRESSURE" in self.output_variables:
            self.surrogate_pressure = MODELS["PRESSURE"]
            self.models["pressure"] = self.surrogate_pressure
            self.model_types["pressure"] = model_type  # FNO or Transolver

        if "SGAS" in self.output_variables:
            self.surrogate_gas = MODELS["SGAS"]
            self.models["gas"] = self.surrogate_gas
            self.model_types["gas"] = model_type

        if "SWAT" in self.output_variables:
            self.surrogate_saturation = MODELS["SATURATION"]
            self.models["saturation"] = self.surrogate_saturation
            self.model_types["saturation"] = model_type

        if "SOIL" in self.output_variables:
            self.surrogate_oil = MODELS["SOIL"]
            self.models["oil"] = self.surrogate_oil
            self.model_types["oil"] = model_type

        self.surrogate_peacemann = MODELS["PEACEMANN"]
        self.models["peacemann"] = self.surrogate_peacemann
        self.model_types["peacemann"] = "FNO"  # Peacemann is always FNO

    def _handle_3d_to_2d(self, input_tensor, model, model_key):
        """Route a tensor to FNO or Transolver with whole-batch Transolver processing.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor of shape ``(B, steppi, nz, nx, ny, C)`` for Transolver,
            or any compatible shape for FNO.
        model : torch.nn.Module
            Surrogate model instance.
        model_key : str
            Key used to determine the backend type (``"FNO"`` or otherwise).

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, steppi, nz, nx, ny)`` for Transolver,
            or the FNO model's native output shape.
        """
        model_type = self.model_types.get(model_key, "FNO")
        steppi = self.steppi

        if model_type == "FNO":
            # FNO can handle 3D directly, or input is already 2D
            return model(input_tensor)
        B, _steppi_no, nz, nx, ny, _C = input_tensor.shape

        batch_chunk_size = 2
        time_chunk_size = 3

        # Output: (B, steppi, nz, nx, ny)
        output = torch.zeros(B, steppi, nz, nx, ny, device=input_tensor.device)

        (
            (B + batch_chunk_size - 1)
            // batch_chunk_size
            * (steppi + time_chunk_size - 1)
            // time_chunk_size
        )
        current_chunk = 0

        for batch_start in range(0, B, batch_chunk_size):
            batch_end = min(batch_start + batch_chunk_size, B)

            # input_chunk: (b, t, nz, nx, ny, C)
            input_chunk = input_tensor[batch_start:batch_end]
            b, t, nz_c, nx_c, ny_c, C_c = input_chunk.shape

            x_5d = input_chunk.view(b * t, nz_c, nx_c, ny_c, C_c)

            # model(...) returns (b * t, nz, nx, ny, out_dim)
            pred_5d = model(x_5d)
            # Remove out_dim=1 and restore (b, t, nz, nx, ny)
            pred_5d = pred_5d.squeeze(-1)
            pred_3d = pred_5d.view(b, steppi, nz_c, nx_c, ny_c)

            output[batch_start:batch_end] = pred_3d

            # Clean up
            del input_chunk, x_5d, pred_5d, pred_3d
            if current_chunk % 3 == 0:
                torch.cuda.empty_cache()

        return output

    def forward(self, input_tensor, mode="both", **kwargs):
        """Forward pass for the composite batch model.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor for the model.
        mode : str
            Which model to use: ``"pressure"``, ``"saturation"``, ``"gas"``,
            ``"oil"``, ``"peacemann"``, or ``"both"``.

        Returns
        -------
        dict
            Outputs from the selected model(s).
        """
        outputs = {}

        if mode in ["pressure", "both"]:
            outputs["pressure"] = self._handle_3d_to_2d(
                input_tensor, self.surrogate_pressure, "pressure"
            )

        if mode in ["gas", "both"]:
            outputs["gas"] = self._handle_3d_to_2d(
                input_tensor, self.surrogate_gas, "gas"
            )

        if mode in ["saturation", "both"]:
            outputs["saturation"] = self._handle_3d_to_2d(
                input_tensor, self.surrogate_saturation, "saturation"
            )

        if mode in ["oil", "both"]:
            outputs["oil"] = self._handle_3d_to_2d(
                input_tensor, self.surrogate_oil, "oil"
            )

        if mode in ["peacemann", "both"]:
            outputs["peacemann"] = self._handle_3d_to_2d(
                input_tensor, self.surrogate_peacemann, "peacemann"
            )

        return outputs


class CompositeOptimizer:
    """Composite Optimizer."""

    def __init__(self, optimizers):
        """
        Initialize with a dictionary of optimizers.
        Args:
            optimizers (dict): Dictionary with keys as model parts and values as optimizers.
        """
        self.optimizers = optimizers

    def zero_grad(self, set_to_none=True):
        """
        Zero gradients for all optimizers.
        Args:
            set_to_none (bool): Whether to set gradients to None.
        """
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        """Step all optimizers."""
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self):
        """Return state_dict for all optimizers."""
        return {key: opt.state_dict() for key, opt in self.optimizers.items()}

    def load_state_dict(self, state_dict):
        """Load state_dict for all optimizers."""
        for key, opt in self.optimizers.items():
            opt.load_state_dict(state_dict[key])


def safe_rmtree(path, retries=3, delay=1):
    """Remove a directory tree with retry logic for busy-device errors.

    Parameters
    ----------
    path : str
        Filesystem path to the directory to remove.
    retries : int, optional
        Maximum number of removal attempts. Default is 3.
    delay : int, optional
        Seconds to wait between retries. Default is 1.
    """
    logger = setup_logging()
    for _attempt in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
                logger.info(f"Removed folder: {path}")
            else:
                logger.info(f"Folder does not exist: {path}")
            break
        except OSError as e:
            if "Device or resource busy" in str(e):
                logger.info(f"Resource busy: {path}. Retrying...")
                time.sleep(delay)
            else:
                logger.info(f"Failed to remove {path}: {e}")
    else:
        logger.info(f"Failed to remove directory: {path} after {retries} attempts.")


def save_model_to_buffer(model, model_name):
    """Save a model's state dictionary to a file path.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model whose state dict is to be saved.
    model_name : str
        Destination file path for the saved state dictionary.
    """
    torch.save(model.state_dict(), model_name)


def write_buffers_to_disk(model_buffers):
    """Write in-memory model buffers to their corresponding file paths on disk.

    Parameters
    ----------
    model_buffers : dict
        Mapping of file paths (str) to ``io.BytesIO`` buffer objects containing
        serialised model state dictionaries.
    """
    for path, buffer in model_buffers.items():
        buffer.seek(0)  # Reset buffer position before writing
        with open(path, "wb") as f:
            f.write(buffer.read())  # Write buffer content to file
        logger = setup_logging()
        logger.info(f"✅ Model buffer written to disk: {path}")


def InitializeLoggers(cfg: DictConfig) -> tuple[DistributedManager, PythonLogger]:
    """
    Initializes loggers and distributed manager.

    Parameters
    ----------
    cfg : DictConfig
        Config file parameters.

    Returns
    -------
    Tuple[DistributedManager, PythonLogger]
    """
    # Initialize distributed manager
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(dist_manager.rank)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(dist_manager.rank % torch.cuda.device_count())
    logger = setup_logging()
    logger.info(
        f"Process {os.getenv('RANK')}: torch.cuda.device_count() = {torch.cuda.device_count()}"
    )
    logger.info(
        f"Process {os.getenv('RANK')}: Visible GPUs = {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
    )
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device_id = dist_manager.rank % gpu_count  # Map rank to available GPUs
        torch.cuda.set_device(device_id)
        logger.info(
            f"Process {dist_manager.rank} is using GPU {device_id}: {torch.cuda.get_device_name(device_id)}"
        )
    else:
        logger.info(f"Process {dist_manager.rank} is using CPU")
    logger = PythonLogger(
        name=f"PhysicsNeMo Reservoir_Characterisation{dist_manager.rank}"
    )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    experiment_name = "PhysicsNeMo-Reservoir Modelling"
    experiment_id = None
    if dist_manager.rank == 0:
        try:
            tracking_dir = os.path.join(os.getcwd(), "mlruns")
            # Honour cfg.custom.reset_mlruns: "Yes" wipes existing runs,
            # anything else (or missing) preserves them across sessions.
            reset = str(cfg.custom.get("reset_mlruns", "No")).lower() == "yes"
            logger = setup_logging()

            if reset and os.path.exists(tracking_dir):
                logger.warning(
                    f"reset_mlruns=Yes — removing existing directory: {tracking_dir}"
                )
                shutil.rmtree(tracking_dir, onerror=on_rm_error)
                os.makedirs(tracking_dir, exist_ok=True)
            elif not os.path.exists(tracking_dir):
                logger.info(f"Creating mlruns directory: {tracking_dir}")
                os.makedirs(tracking_dir, exist_ok=True)
            else:
                logger.info(f"Reusing existing mlruns directory: {tracking_dir}")

            os.environ["MLFLOW_TRACKING_URI"] = f"file://{tracking_dir}"
            mlflow.set_tracking_uri(f"file://{tracking_dir}")
            # ✅ **Fix: Ensure MlflowClient is correctly used**
            client = MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger = setup_logging()
                logger.info(f"[MLflow] Creating new experiment: {experiment_name}")
                experiment_id = client.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
                logger = setup_logging()
                logger.info(f"[MLflow] Using existing experiment ID: {experiment_id}")
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name="PhysicsNeMo-Training")
            logger = setup_logging()
            logger.info(
                f"[MLflow] Started run for experiment '{experiment_name}' with ID {experiment_id}"
            )
        except Exception as e:
            logger = setup_logging()
            logger.error(f"Failed to initialize MLFlow on rank 0: {e}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return dist_manager, RankZeroLoggingWrapper(logger, dist_manager)


def are_models_equal(model1, model2):
    """Check whether two models have identical parameter values.

    Parameters
    ----------
    model1 : torch.nn.Module
        First model to compare.
    model2 : torch.nn.Module
        Second model to compare.

    Returns
    -------
    bool
        ``True`` if every corresponding parameter tensor is element-wise equal.
    """
    return all(
        torch.equal(param1, param2)
        for param1, param2 in zip(
            model1.state_dict().values(), model2.state_dict().values(), strict=False
        )
    )


def remove_ddp(model):
    """Unwrap a DistributedDataParallel model to expose the underlying module.

    Parameters
    ----------
    model : torch.nn.Module
        Model that may or may not be wrapped in ``DistributedDataParallel``.

    Returns
    -------
    torch.nn.Module
        The inner ``.module`` if ``model`` is DDP-wrapped, otherwise ``model`` itself.
    """
    return (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )


def check_and_remove_dirs(directories, response, logger):
    """
    Checks if directories exist and prompts the user for removal.

    Args:
        directories (list): List of directory paths to check and remove.
    """
    for directory in directories:
        if os.path.exists(directory) and os.path.isdir(directory):
            # response = input(f"Directory '{directory}' exists. Do you want to remove it? (yes/no): ").strip().lower()
            if response == "yes":
                shutil.rmtree(directory)
                logger = setup_logging()
                logger.info(f"Removed: {directory}")
            else:
                logger.info(f"Skipped: {directory}")
        else:
            logger.info(f"Directory '{directory}' does not exist.")


def on_rm_error(func, path, exc_info):
    """Error handler for shutil.rmtree() to handle permission issues.

    Clears the read-only attribute on the target so the retry can delete it.
    On POSIX, this enables owner write; on Windows, it removes the read-only
    flag — which is the well-known fix from the Python docs for "rmtree
    refuses to delete read-only files on Windows".
    """
    import stat

    logger = setup_logging()
    logger.warning(f"Error removing {path}. Retrying...")
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)  # Retry removing
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Failed to remove {path}: {e}")
