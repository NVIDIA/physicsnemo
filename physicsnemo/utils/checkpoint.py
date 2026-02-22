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

"""Checkpoint utilities for saving and loading training state.

Provides :func:`save_checkpoint` and :func:`load_checkpoint` for persisting
and restoring model weights, optimizer/scheduler/scaler state, and arbitrary
metadata.  Supports local filesystems and remote stores via ``fsspec``.
"""

import os
import re
from pathlib import Path, PurePath
from typing import Any

import fsspec
import fsspec.utils
import torch
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler

import physicsnemo
from physicsnemo.core.filesystem import LOCAL_CACHE, _download_cached
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.capture import _StaticCapture
from physicsnemo.utils.logging import PythonLogger

checkpoint_logging = PythonLogger("checkpoint")


def _get_checkpoint_filename(
    path: str,
    base_name: str = "checkpoint",
    index: int | None = None,
    saving: bool = False,
    model_type: str = "mdlus",
) -> str:
    r"""Build the filename for a numbered checkpoint.

    Resolution logic:

    * **Explicit index** (``index`` is not ``None``): returns that exact
      checkpoint path.
    * **Latest** (``index is None``, ``saving=False``): scans for existing
      checkpoints and returns the one with the largest index.
    * **Next** (``index is None``, ``saving=True``): returns the path for
      the *next* index after the largest existing one.

    When no existing checkpoints are found, the returned path uses index 0.

    Parameters
    ----------
    path : str
        Directory containing checkpoint files.
    base_name : str, optional
        Stem used in the filename, by default ``"checkpoint"``.
    index : int | None, optional
        Specific checkpoint index to use.  When ``None``, the latest or
        next index is determined automatically.
    saving : bool, optional
        If ``True`` (and ``index is None``), return the *next* available
        filename rather than the latest existing one.  By default ``False``.
    model_type : str, optional
        ``"mdlus"`` for :class:`~physicsnemo.core.Module` models,
        ``"pt"`` for vanilla PyTorch models.  Determines the file
        extension.  By default ``"mdlus"``.

    Returns
    -------
    str
        Fully qualified checkpoint filename.
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

    # Determine input file name. Get absolute file path if Posix path.
    # pathlib does not support custom schemes (eg: msc://...) so only perform resolve() for Posix.
    protocol = fsspec.utils.get_protocol(path)
    fs = fsspec.filesystem(protocol)
    if protocol == "file":
        path = str(Path(path).resolve())
    checkpoint_filename = f"{path}/{base_name}.{model_parallel_rank}"

    # File extension for PhysicsNeMo models or PyTorch models
    file_extension = ".mdlus" if model_type == "mdlus" else ".pt"

    # If epoch is provided load that file
    if index is not None:
        checkpoint_filename = checkpoint_filename + f".{index}"
        checkpoint_filename += file_extension
    # Otherwise try loading the latest epoch or rolling checkpoint
    else:
        file_names = [
            fname for fname in fs.glob(checkpoint_filename + "*" + file_extension)
        ]

        if len(file_names) > 0:
            # If checkpoint from a null index save exists load that
            # This is the most likely line to error since it will fail with
            # invalid checkpoint names

            file_idx = []

            for fname in file_names:
                fname_path = PurePath(fname)
                file_stem = fname_path.name

                pattern = rf"^{re.escape(base_name)}\.{model_parallel_rank}\.(\d+){re.escape(file_extension)}$"
                match = re.match(pattern, file_stem)
                if match:
                    file_idx.append(int(match.group(1)))
            file_idx.sort()
            # If we are saving index by 1 to get the next free file name
            if saving:
                checkpoint_filename = checkpoint_filename + f".{file_idx[-1] + 1}"
            else:
                checkpoint_filename = checkpoint_filename + f".{file_idx[-1]}"
            checkpoint_filename += file_extension
        else:
            checkpoint_filename += ".0" + file_extension

    return checkpoint_filename


def _unique_model_names(
    models: list[torch.nn.Module],
    loading: bool = False,
) -> dict[str, torch.nn.Module]:
    r"""Map a list of models to unique names derived from their class names.

    DDP wrappers (``model.module``) and ``torch.compile`` wrappers
    (``OptimizedModule``) are automatically stripped before naming.
    When multiple models share a class name, a numeric suffix is appended
    (e.g. ``"MyModel0"``, ``"MyModel1"``).

    Parameters
    ----------
    models : list[torch.nn.Module]
        Models to generate names for.
    loading : bool, optional
        When ``True``, emits a warning if a model is already compiled
        (loading into a compiled model can cause issues).  By default
        ``False``.

    Returns
    -------
    dict[str, torch.nn.Module]
        Mapping from unique name to (unwrapped) module.
    """
    # Loop through provided models and set up base names
    model_dict = {}
    for model0 in models:
        if hasattr(model0, "module"):
            # Strip out DDP layer
            model0 = model0.module
        # Strip out torch dynamo wrapper
        if isinstance(model0, torch._dynamo.eval_frame.OptimizedModule):
            model0 = model0._orig_mod
            is_compiled = True
        else:
            is_compiled = False
        # Base name of model is the class name
        base_name = type(model0).__name__
        # Warning in case of attempt to load into a compiled model
        if is_compiled and loading:
            checkpoint_logging.warning(
                f"Model {base_name} is already compiled, consider loading first and then compiling."
            )
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
    path: Path | str,
    models: torch.nn.Module | list[torch.nn.Module] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: GradScaler | None = None,
    epoch: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    r"""Save a training checkpoint to disk (or a remote store).

    Up to two categories of files are created inside ``path``:

    * **Model weights** (when ``models`` is provided) - one file per model:
      ``{class_name}{id}.{mp_rank}.{epoch}.{ext}`` where *ext* is
      ``.mdlus`` for :class:`~physicsnemo.core.Module` instances or
      ``.pt`` for plain PyTorch models.  When several models share a class
      name, a numeric *id* is appended (``"MyModel0"``, ``"MyModel1"``).
    * **Training state** (when any of ``optimizer`` / ``scheduler`` /
      ``scaler`` is provided, or
      :class:`~physicsnemo.utils.capture._StaticCapture` scalers exist):
      ``checkpoint.{mp_rank}.{epoch}.pt`` containing their combined
      ``state_dict`` entries, plus ``epoch`` and ``metadata``.

    Use :func:`load_checkpoint` to restore from these files.
    To instantiate *and* load a model in one step (without pre-constructing
    it), use :meth:`~physicsnemo.core.module.Module.from_checkpoint`.

    Parameters
    ----------
    path : Path | str
        Directory in which to store checkpoint files.  Created
        automatically for local paths if it does not exist.
    models : torch.nn.Module | list[torch.nn.Module] | None, optional
        Model(s) whose weights should be saved.
    optimizer : torch.optim.Optimizer | None, optional
        Optimizer whose ``state_dict`` should be saved.
    scheduler : LRScheduler | None, optional
        Learning-rate scheduler whose ``state_dict`` should be saved.
    scaler : GradScaler | None, optional
        AMP gradient scaler whose ``state_dict`` should be saved.
        If ``None`` but a
        :class:`~physicsnemo.utils.capture._StaticCapture` scaler exists,
        that scaler's state is saved instead.
    epoch : int | None, optional
        Epoch index to embed in the filename and the checkpoint dict.
        When ``None``, the next available index is used.
    metadata : dict[str, Any] | None, optional
        Arbitrary key-value pairs persisted alongside the training state
        (e.g. best validation loss, MLflow run ID).
    """
    path = str(path)
    protocol = fsspec.utils.get_protocol(path)
    fs = fsspec.filesystem(protocol)
    # Create checkpoint directory if it does not exist.
    # Only applicable to Posix filesystems ("file" protocol), not object stores.
    if protocol == "file" and not Path(path).is_dir():
        checkpoint_logging.warning(
            f"Output directory {path} does not exist, will attempt to create"
        )
        Path(path).mkdir(parents=True, exist_ok=True)

    # == Saving model checkpoint ==
    if models:
        if not isinstance(models, list):
            models = [models]
        models = _unique_model_names(models)
        for name, model in models.items():
            # Get model type
            model_type = "mdlus" if isinstance(model, physicsnemo.core.Module) else "pt"

            # Get full file path / name
            file_name = _get_checkpoint_filename(
                path, name, index=epoch, saving=True, model_type=model_type
            )

            # Save state dictionary
            if isinstance(model, physicsnemo.core.Module):
                model.save(file_name)
            else:
                with fs.open(file_name, "wb") as fp:
                    torch.save(model.state_dict(), fp)
            checkpoint_logging.success(f"Saved model state dictionary: {file_name}")

    # == Saving training checkpoint ==
    checkpoint_dict = {}
    # Optimizer state dict
    if optimizer:
        opt_state_dict = optimizer.state_dict()
        # Strip out torch dynamo wrapper prefix
        for pg in opt_state_dict.get("param_groups", []):
            param_names = pg.get("param_names")
            if param_names is None:
                continue
            pg["param_names"] = [pn.removeprefix("_orig_mod.") for pn in param_names]
        checkpoint_dict["optimizer_state_dict"] = opt_state_dict

    # Scheduler state dict
    if scheduler:
        checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()

    # Scaler state dict
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
        with fs.open(output_filename, "wb") as fp:
            torch.save(
                checkpoint_dict,
                fp,
            )
        checkpoint_logging.success(f"Saved training checkpoint: {output_filename}")


def load_checkpoint(
    path: Path | str,
    models: torch.nn.Module | list[torch.nn.Module] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: GradScaler | None = None,
    epoch: int | None = None,
    metadata_dict: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
) -> int:
    r"""Load a training checkpoint saved by :func:`save_checkpoint`.

    Scans ``path`` for checkpoint files and restores state dictionaries
    into the provided training objects.  Objects that are ``None`` are
    silently skipped.

    Parameters
    ----------
    path : Path | str
        Directory containing checkpoint files (local path or ``fsspec``
        URI).  If the directory does not exist, the load is skipped and
        ``0`` is returned.
    models : torch.nn.Module | list[torch.nn.Module] | None, optional
        Model(s) whose ``state_dict`` should be restored.  DDP and
        ``torch.compile`` wrappers are stripped automatically.
    optimizer : torch.optim.Optimizer | None, optional
        Optimizer whose ``state_dict`` should be restored.
    scheduler : LRScheduler | None, optional
        Learning-rate scheduler whose ``state_dict`` should be restored.
    scaler : GradScaler | None, optional
        AMP gradient scaler whose ``state_dict`` should be restored.
    epoch : int | None, optional
        Specific checkpoint index to load.  When ``None``, the checkpoint
        with the largest index (most recent) is loaded.
    metadata_dict : dict[str, Any] | None, optional
        If a ``dict`` is provided, it is updated **in-place** with any
        metadata that was persisted by :func:`save_checkpoint`.
    device : str | torch.device, optional
        Device onto which tensors are mapped during loading.  By default
        ``"cpu"``.

    Returns
    -------
    int
        The epoch stored in the checkpoint.  Returns ``0`` when:

        * The checkpoint directory does not exist.
        * No training-state file is found inside the directory.
        * The training-state file does not contain an ``"epoch"`` key.
    """
    path = str(path)
    fs = fsspec.filesystem(fsspec.utils.get_protocol(path))
    # Check if checkpoint directory exists
    if fs.exists(path):
        if fs.isfile(path):
            raise FileNotFoundError(
                f"Provided checkpoint directory {path} is a file, not directory"
            )
    else:
        checkpoint_logging.warning(
            f"Provided checkpoint directory {path} does not exist, skipping load"
        )
        return 0

    # == Loading model checkpoint ==
    if models:
        if not isinstance(models, list):
            models = [models]
        models = _unique_model_names(models, loading=True)
        for name, model in models.items():
            # Get model type
            model_type = "mdlus" if isinstance(model, physicsnemo.core.Module) else "pt"

            # Get full file path / name
            file_name = _get_checkpoint_filename(
                path, name, index=epoch, model_type=model_type
            )
            if not fs.exists(file_name):
                checkpoint_logging.error(
                    f"Could not find valid model file {file_name}, skipping load"
                )
                continue
            # Load state dictionary
            if isinstance(model, physicsnemo.core.Module):
                model.load(file_name)
            else:
                file_to_load = _cache_if_needed(file_name)
                missing_keys, unexpected_keys = model.load_state_dict(
                    torch.load(file_to_load, map_location=device)
                )
                if missing_keys:
                    checkpoint_logging.warning(
                        f"Missing keys when loading {name}: {missing_keys}"
                    )
                if unexpected_keys:
                    checkpoint_logging.warning(
                        f"Unexpected keys when loading {name}: {unexpected_keys}"
                    )

            checkpoint_logging.success(
                f"Loaded model state dictionary {file_name} to device {device}"
            )

    # == Loading training checkpoint ==
    checkpoint_filename = _get_checkpoint_filename(path, index=epoch, model_type="pt")
    if not fs.exists(checkpoint_filename):
        checkpoint_logging.warning(
            "Could not find valid checkpoint file, skipping load"
        )
        return 0

    file_to_load = _cache_if_needed(checkpoint_filename)
    checkpoint_dict = torch.load(file_to_load, map_location=device)
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

    if metadata_dict is not None:
        metadata_dict.update(checkpoint_dict.get("metadata", {}))

    return epoch


def get_checkpoint_dir(base_dir: Path | str, model_name: str) -> str:
    r"""Build a model-specific checkpoint directory path.

    Returns ``"{base_dir}/checkpoints_{model_name}"``, handling both
    local paths and ``msc://`` URIs.

    Parameters
    ----------
    base_dir : Path | str
        Root directory under which the checkpoint subdirectory is placed.
    model_name : str
        Model name used as the directory suffix.

    Returns
    -------
    str
        Full path to the checkpoint directory.
    """
    base_dir = str(base_dir)
    top_level_dir = f"checkpoints_{model_name}"
    protocol = fsspec.utils.get_protocol(base_dir)
    if protocol == "msc":
        if not base_dir.endswith("/"):
            base_dir += "/"
        return base_dir + top_level_dir
    else:
        return os.path.join(base_dir, top_level_dir)


def _cache_if_needed(path: str) -> str:
    r"""Return a local path for ``path``, downloading to cache if remote.

    For the ``"file"`` protocol the path is returned unchanged.  For remote
    protocols the file is fetched via
    :func:`~physicsnemo.core.filesystem._download_cached` into a
    process-specific cache directory.

    Parameters
    ----------
    path : str
        Checkpoint file path (local or ``fsspec`` URI).

    Returns
    -------
    str
        Local filesystem path suitable for :func:`torch.load`.
    """
    protocol = fsspec.utils.get_protocol(path)
    if protocol == "file":
        return path
    else:
        return _download_cached(
            path,
            recursive=False,
            local_cache_path=os.path.join(LOCAL_CACHE, f"checkpoint_pid_{os.getpid()}"),
        )
