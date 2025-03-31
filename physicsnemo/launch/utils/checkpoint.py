# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import glob
import re
from pathlib import Path
from typing import Any, Dict, List, NewType, Optional, Union

import torch
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import _LRScheduler

import physicsnemo
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger
from physicsnemo.utils.capture import _StaticCapture

optimizer = NewType("optimizer", torch.optim)
scheduler = NewType("scheduler", _LRScheduler)
scaler = NewType("scaler", GradScaler)

checkpoint_logging = PythonLogger("checkpoint")


def _get_checkpoint_filename(
    path: str,
    base_name: str = "checkpoint",
    index: Union[int, None] = None,
    saving: bool = False,
    model_type: str = "mdlus",
) -> str:
    """Gets the file name /path of checkpoint

    This function has three different ways of providing a checkout filename:
    - If supplied an index this will return the checkpoint name using that index.
    - If index is None and saving is false, this will get the checkpoint with the
    largest index (latest save).
    - If index is None and saving is true, it will return the next valid index file name
    which is calculated by indexing the largest checkpoint index found by one.

    Parameters
    ----------
    path : str
        Path to checkpoints
    base_name: str, optional
        Base file name, by default checkpoint
    index : Union[int, None], optional
        Checkpoint index, by default None
    saving : bool, optional
        Get filename for saving a new checkpoint, by default False
    model_type : str
        Model type, by default "mdlus" for PhysicsNeMo models and "pt" for PyTorch models


    Returns
    -------
    str
        Checkpoint file name
    """
    # Get model parallel rank so all processes in the first model parallel group
    # can save their checkpoint. In the case without model parallelism,
    # model_parallel_rank should be the same as the process rank itself and
    # only rank 0 saves
    if not DistributedManager.is_initialized():
        checkpoint_logging.warning(
            "`DistributedManager` not initialized already. Initializing now, but this might lead to unexpected errors"
        )
        DistributedManager.initialize()
    manager = DistributedManager()
    model_parallel_rank = (
        manager.group_rank("model_parallel")
        if "model_parallel" in manager.group_names
        else 0
    )

    # Input file name
    checkpoint_filename = str(
        Path(path).resolve() / f"{base_name}.{model_parallel_rank}"
    )

    # File extension for PhysicsNeMo models or PyTorch models
    file_extension = ".mdlus" if model_type == "mdlus" else ".pt"

    # If epoch is provided load that file
    if index is not None:
        checkpoint_filename = checkpoint_filename + f".{index}"
        checkpoint_filename += file_extension
    # Otherwise try loading the latest epoch or rolling checkpoint
    else:
        file_names = [
            Path(fname).name
            for fname in glob.glob(
                checkpoint_filename + "*" + file_extension, recursive=False
            )
        ]

        if len(file_names) > 0:
            # If checkpoint from a null index save exists load that
            # This is the most likely line to error since it will fail with
            # invalid checkpoint names
            file_idx = [
                int(
                    re.sub(
                        f"^{base_name}.{model_parallel_rank}.|" + file_extension,
                        "",
                        fname,
                    )
                )
                for fname in file_names
            ]
            file_idx.sort()
            # If we are saving index by 1 to get the next free file name
            if saving:
                checkpoint_filename = checkpoint_filename + f".{file_idx[-1]+1}"
            else:
                checkpoint_filename = checkpoint_filename + f".{file_idx[-1]}"
            checkpoint_filename += file_extension
        else:
            checkpoint_filename += ".0" + file_extension

    return checkpoint_filename


def _unique_model_names(
    models: List[torch.nn.Module],
) -> Dict[str, torch.nn.Module]:
    """Util to clean model names and index if repeat names, will also strip DDP wrappers
    if they exist.

    Parameters
    ----------
    model :  List[torch.nn.Module]
        List of models to generate names for

    Returns
    -------
    Dict[str, torch.nn.Module]
        Dictionary of model names and respective modules
    """
    # Loop through provided models and set up base names
    model_dict = {}
    for model0 in models:
        if hasattr(model0, "module"):
            # Strip out DDP layer
            model0 = model0.module
        # Base name of model is meta.name unless pytorch model
        base_name = model0.__class__.__name__
        if isinstance(model0, physicsnemo.models.Module):
            base_name = model0.meta.name
        # If we have multiple models of the same name, introduce another index
        if base_name in model_dict:
            model_dict[base_name].append(model0)
        else:
            model_dict[base_name] = [model0]

    # Set up unique model names if needed
    output_dict = {}
    for key, model in model_dict.items():
        if len(model) > 1:
            for i, model0 in enumerate(model):
                output_dict[key + str(i)] = model0
        else:
            output_dict[key] = model[0]

    return output_dict


def save_checkpoint(
    path: str,
    models: Union[torch.nn.Module, List[torch.nn.Module], None] = None,
    optimizer: Union[optimizer, None] = None,
    scheduler: Union[scheduler, None] = None,
    scaler: Union[scaler, None] = None,
    epoch: Union[int, None] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Training checkpoint saving utility

    This will save a training checkpoint in the provided path following the file naming
    convention "checkpoint.{model parallel id}.{epoch/index}.mdlus". The load checkpoint
    method in PhysicsNeMo core can then be used to read this file.

    Parameters
    ----------
    path : str
        Path to save the training checkpoint
    models : Union[torch.nn.Module, List[torch.nn.Module], None], optional
        A single or list of PyTorch models, by default None
    optimizer : Union[optimizer, None], optional
        Optimizer, by default None
    scheduler : Union[scheduler, None], optional
        Learning rate scheduler, by default None
    scaler : Union[scaler, None], optional
        AMP grad scaler. Will attempt to save on in static capture if none provided, by
        default None
    epoch : Union[int, None], optional
        Epoch checkpoint to load. If none this will save the checkpoint in the next
        valid index, by default None
    metadata : Optional[Dict[str, Any]], optional
        Additional metadata to save, by default None
    """
    # Create checkpoint directory if it does not exist
    if not Path(path).is_dir():
        checkpoint_logging.warning(
            f"Output directory {path} does not exist, will " "attempt to create"
        )
        Path(path).mkdir(parents=True, exist_ok=True)

    # == Saving model checkpoint ==
    if models:
        if not isinstance(models, list):
            models = [models]
        models = _unique_model_names(models)
        for name, model in models.items():
            # Get model type
            model_type = (
                "mdlus" if isinstance(model, physicsnemo.models.Module) else "pt"
            )

            # Get full file path / name
            file_name = _get_checkpoint_filename(
                path, name, index=epoch, saving=True, model_type=model_type
            )

            # Save state dictionary
            if isinstance(model, physicsnemo.models.Module):
                model.save(file_name)
            else:
                torch.save(model.state_dict(), file_name)
            checkpoint_logging.success(f"Saved model state dictionary: {file_name}")

    # == Saving training checkpoint ==
    checkpoint_dict = {}
    # Optimizer state dict
    if optimizer:
        checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()

    # Scheduler state dict
    if scheduler:
        checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()

    # Scheduler state dict
    if scaler:
        checkpoint_dict["scaler_state_dict"] = scaler.state_dict()
    # Static capture is being used, save its grad scaler
    if _StaticCapture._amp_scalers:
        checkpoint_dict["static_capture_state_dict"] = _StaticCapture.state_dict()

    # Output file name
    output_filename = _get_checkpoint_filename(
        path, index=epoch, saving=True, model_type="pt"
    )
    if epoch:
        checkpoint_dict["epoch"] = epoch
    if metadata:
        checkpoint_dict["metadata"] = metadata

    # Save checkpoint to memory
    if bool(checkpoint_dict):
        torch.save(
            checkpoint_dict,
            output_filename,
        )
        checkpoint_logging.success(f"Saved training checkpoint: {output_filename}")


def load_checkpoint(
    path: str,
    models: Union[torch.nn.Module, List[torch.nn.Module], None] = None,
    optimizer: Union[optimizer, None] = None,
    scheduler: Union[scheduler, None] = None,
    scaler: Union[scaler, None] = None,
    epoch: Union[int, None] = None,
    metadata_dict: Optional[Dict[str, Any]] = {},
    device: Union[str, torch.device] = "cpu",
) -> int:
    """Checkpoint loading utility

    This loader is designed to be used with the save checkpoint utility in PhysicsNeMo
    Launch. Given a path, this method will try to find a checkpoint and load state
    dictionaries into the provided training objects.

    Parameters
    ----------
    path : str
        Path to training checkpoint
    models : Union[torch.nn.Module, List[torch.nn.Module], None], optional
        A single or list of PyTorch models, by default None
    optimizer : Union[optimizer, None], optional
        Optimizer, by default None
    scheduler : Union[scheduler, None], optional
        Learning rate scheduler, by default None
    scaler : Union[scaler, None], optional
        AMP grad scaler, by default None
    epoch : Union[int, None], optional
        Epoch checkpoint to load. If none is provided this will attempt to load the
        checkpoint with the largest index, by default None
    metadata_dict: Optional[Dict[str, Any]], optional
        Dictionary to store metadata from the checkpoint, by default None
    device : Union[str, torch.device], optional
        Target device, by default "cpu"

    Returns
    -------
    int
        Loaded epoch
    """
    # Check if checkpoint directory exists
    if not Path(path).is_dir():
        checkpoint_logging.warning(
            f"Provided checkpoint directory {path} does not exist, skipping load"
        )
        return 0

    # == Loading model checkpoint ==
    if models:
        if not isinstance(models, list):
            models = [models]
        models = _unique_model_names(models)
        for name, model in models.items():
            # Get model type
            model_type = (
                "mdlus" if isinstance(model, physicsnemo.models.Module) else "pt"
            )

            # Get full file path / name
            file_name = _get_checkpoint_filename(
                path, name, index=epoch, model_type=model_type
            )
            if not Path(file_name).exists():
                checkpoint_logging.error(
                    f"Could not find valid model file {file_name}, skipping load"
                )
                continue
            # Load state dictionary
            if isinstance(model, physicsnemo.models.Module):
                model.load(file_name)
            else:
                model.load_state_dict(torch.load(file_name, map_location=device))

            checkpoint_logging.success(
                f"Loaded model state dictionary {file_name} to device {device}"
            )

    # == Loading training checkpoint ==
    checkpoint_filename = _get_checkpoint_filename(path, index=epoch, model_type="pt")
    if not Path(checkpoint_filename).is_file():
        checkpoint_logging.warning(
            "Could not find valid checkpoint file, skipping load"
        )
        return 0

    checkpoint_dict = torch.load(checkpoint_filename, map_location=device)
    checkpoint_logging.success(
        f"Loaded checkpoint file {checkpoint_filename} to device {device}"
    )

    # Optimizer state dict
    if optimizer and "optimizer_state_dict" in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        checkpoint_logging.success("Loaded optimizer state dictionary")

    # Scheduler state dict
    if scheduler and "scheduler_state_dict" in checkpoint_dict:
        scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])
        checkpoint_logging.success("Loaded scheduler state dictionary")

    # Scaler state dict
    if scaler and "scaler_state_dict" in checkpoint_dict:
        scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])
        checkpoint_logging.success("Loaded grad scaler state dictionary")

    if "static_capture_state_dict" in checkpoint_dict:
        _StaticCapture.load_state_dict(checkpoint_dict["static_capture_state_dict"])
        checkpoint_logging.success("Loaded static capture state dictionary")

    epoch = 0
    if "epoch" in checkpoint_dict:
        epoch = checkpoint_dict["epoch"]

    # Update metadata if exists and the dictionary object is provided
    metadata = checkpoint_dict.get("metadata", {})
    for key, value in metadata.items():
        metadata_dict[key] = value

    return epoch
