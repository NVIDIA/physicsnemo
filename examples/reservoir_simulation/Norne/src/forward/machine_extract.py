"""
SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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

# 🛠 Standard Library
import os
import time
import logging
import shutil
from typing import Tuple

# 🔧 Third-party Libraries
from omegaconf import DictConfig
import torch

# 📊 MLFlow & Logging
import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient

# 🔥 PhyNeMo & ML Libraries
from physicsnemo.models.fno import FNO
from physicsnemo.models.module import Module
from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.distributed import DistributedManager

# 📦 Local Modules


def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    logger = logging.getLogger("Forward problem")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class FNOModel(Module):
    def __init__(
        self,
        input_dim,
        steppi,
        output_shape,
        device,
        num_layers=4,
        decoder_layers=1,
        decoder_layer_size=32,
        dimension=3,
        latent_channels=32,
        num_fno_layers=4,
        padding=8,
        num_fno_modes=16,
    ):
        super().__init__()
        self.fno = FNO(
            in_channels=input_dim,
            out_channels=output_shape * steppi,
            decoder_layers=decoder_layers,
            decoder_layer_size=decoder_layer_size,
            dimension=dimension,
            latent_channels=latent_channels,
            num_fno_layers=num_layers,
            padding=padding,
            num_fno_modes=num_fno_modes,
        ).to(torch.device(device))  # Explicit device conversion
        self.meta = type("", (), {})()  # Empty object
        self.meta.name = "fno_model"

    def forward(self, x):
        return self.fno(x)


def create_fno_model(
    input_dim,
    steppi,
    output_shape,
    device,
    num_layers=4,
    decoder_layers=1,
    decoder_layer_size=32,
    dimension=3,
    latent_channels=32,
    num_fno_layers=4,
    padding=8,
    num_fno_modes=16,
):
    """
    Create a Fourier Neural Operator (FNO) model wrapped in a compatible PhyNeMo Module.

    Parameters:
    -----------
    input_dim : int
        The number of input features (e.g., the number of spatial points).
    steppi : int
        Number of time steps or resolution in the output.
    output_shape : int
        The number of distinct outputs to predict.
    device : str
        Device to create the model on ('cpu' or 'cuda').
    num_layers : int, optional
        The number of layers in the FNO. Default is 4.
    decoder_layers : int, optional
        The number of decoder layers. Default is 1.
    decoder_layer_size : int, optional
        The size of each decoder layer. Default is 32.
    dimension : int, optional
        The dimensionality of the FNO (2D or 3D). Default is 2.
    latent_channels : int, optional
        Number of latent channels. Default is 32.
    num_fno_layers : int, optional
        Number of FNO layers. Default is 4.
    padding : int, optional
        Padding for the FNO. Default is 8.
    num_fno_modes : int, optional
        Number of Fourier modes to use. Default is 16.

    Returns:
    --------
    fno_model : FNOModel
        Initialized FNO model ready for inference or training.
    """
    # Validate arguments
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


class CompositeModel(Module):
    def __init__(self, MODELS, output_variables):
        super().__init__()
        self.output_variables = output_variables
        if "PRESSURE" in self.output_variables:
            self.surrogate_pressure = MODELS["PRESSURE"]  # surrogate_pressure
        if "SGAS" in self.output_variables:
            self.surrogate_gas = MODELS["SGAS"]  # surrogate_pressure
        if "SWAT" in self.output_variables:
            self.surrogate_saturation = MODELS["SATURATION"]
        if "SOIL" in self.output_variables:
            self.surrogate_oil = MODELS["SOIL"]

        self.surrogate_peacemann = MODELS["PEACEMANN"]

    def forward(self, input_tensor, mode="both", **kwargs):
        """
        Forward pass for the composite model.
        Parameters:
        -----------
        input_tensor : torch.Tensor
            Input tensor for the model.
        mode : str
            Which model to use: "pressure", "saturation", or "both".
        Returns:
        --------
        dict
            Outputs from the selected model(s).
        """
        outputs = {}
        if mode in ["pressure", "both"]:
            outputs["pressure"] = self.surrogate_pressure(input_tensor)
        if mode in ["gas", "both"]:
            outputs["gas"] = self.surrogate_gas(input_tensor)

        if mode in ["saturation", "both"]:
            outputs["saturation"] = self.surrogate_saturation(input_tensor)

        if mode in ["oil", "both"]:
            outputs["oil"] = self.surrogate_oil(input_tensor)

        if mode in ["peacemann", "both"]:
            outputs["peacemann"] = self.surrogate_peacemann(input_tensor)

        return outputs


class CompositeOptimizer:
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
    for attempt in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
                logger = setup_logging()
                logger.info(f"Removed folder: {path}")
            else:
                logger.info(f"Folder does not exist: {path}")
            break
        except OSError as e:
            if "Device or resource busy" in str(e):
                logger = setup_logging()
                logger.info(f"Resource busy: {path}. Retrying...")
                time.sleep(delay)
            else:
                logger.info(f"Failed to remove {path}: {e}")
    else:
        logger = setup_logging()
        logger.info(f"Failed to remove directory: {path} after {retries} attempts.")


def save_model_to_buffer(model, model_name):
    torch.save(model.state_dict(), model_name)


def write_buffers_to_disk(model_buffers):
    for path, buffer in model_buffers.items():
        buffer.seek(0)  # Reset buffer position before writing
        with open(path, "wb") as f:
            f.write(buffer.read())  # Write buffer content to file
        logger = setup_logging()
        logger.info(f"✅ Model buffer written to disk: {path}")


def InitializeLoggers(cfg: DictConfig) -> Tuple[DistributedManager, PythonLogger]:
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
    logger = PythonLogger(name=f"PhyNeMo Reservoir_Characterisation{dist_manager.rank}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    experiment_name = "PhyNeMo-Reservoir Modelling"
    experiment_id = None
    if dist_manager.rank == 0:
        try:
            tracking_dir = os.path.join(os.getcwd(), "mlruns")
            if os.path.exists(tracking_dir):
                logger = setup_logging()
                logger.info(f"Removing existing directory: {tracking_dir}")
                os.system(f"rm -rf {tracking_dir}")  # Force delete using shell command
                # shutil.rmtree(tracking_dir)  # Remove directory and all contents
            else:
                os.makedirs(tracking_dir, exist_ok=True)
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
            mlflow.start_run(run_name="PhyNeMo-Training")
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
    return all(
        torch.equal(param1, param2)
        for param1, param2 in zip(
            model1.state_dict().values(), model2.state_dict().values()
        )
    )


def remove_ddp(model):
    return (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )


def broadcast_read(obj):
    """
    Broadcasts a dictionary (or any object) from rank 0 to all processes.
    Uses GLOO backend for non-tensor broadcasting.
    """
    world_size = torch.distributed.get_world_size()
    if world_size > 1:
        # ✅ Do NOT reinitialize the process group
        if torch.distributed.get_backend() != "gloo":
            logger = setup_logging()
            logger.warning(
                f"Warning: broadcast_object_list should use GLOO, but found {torch.distributed.get_backend()}"
            )
        obj_list = [obj]  # Place object inside a list
        torch.distributed.broadcast_object_list(obj_list, src=0)  # ✅ Broadcast
        return obj_list[0]  # Extract received object
    else:
        return obj  # No need to broadcast if single GPU


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
    """Error handler for shutil.rmtree() to handle permission issues."""
    logger = setup_logging()
    logger.warning(f"Error removing {path}. Retrying...")
    try:
        os.chmod(path, 0o777)  # Change permissions to ensure it can be deleted
        func(path)  # Retry removing
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Failed to remove {path}: {e}")
