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

"""Training utilities for the GLOBE AirFRANS example.

Contains helpers for device transfer, distributed reduction, checkpointing,
hyperparameter logging, MLflow metric sanitization, and signal handling.
"""

import inspect
import re
import signal
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any, Literal

import git
import mlflow
import numpy as np
import torch
import yaml
from tensordict import TensorDict

import physicsnemo
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.mesh import Mesh


### [torch.compile helpers] ###############################################


def disable_autotune_printing() -> None:
    """Silence the verbose command-line output of ``torch.compile(..., mode="max-autotune")``.

    Uses private ``torch._inductor`` APIs that may change across PyTorch
    versions, so failures are silently ignored.
    """
    try:
        from torch._inductor import config, select_algorithm

        config.max_autotune_report_choices_stats = False
        select_algorithm.PRINT_AUTOTUNE = False
    except (ImportError, AttributeError):
        pass


### [MLflow helpers] ######################################################


def sanitize_metric_name(name: str) -> str:
    """Replace characters not in ``[A-Za-z0-9_.- :/]`` with underscores.

    Leading/trailing whitespace is stripped after replacement. Consecutive
    whitespace is collapsed to a single underscore.

    Args:
        name: Original metric name that may contain special characters.

    Returns:
        Sanitized name safe for MLflow metric keys.

    Examples:
        >>> sanitize_metric_name("ln(1+nut/nu)")
        'ln_1_nut_nu'
        >>> sanitize_metric_name("ΔU/|U_inf|")
        'U_U_inf'
        >>> sanitize_metric_name("C_F,shear")
        'C_F_shear'
    """
    import string

    allowed_chars = set(string.ascii_letters + string.digits + "_-. :")
    sanitized = "".join(c if c in allowed_chars else " " for c in name)
    while "  " in sanitized:
        sanitized = sanitized.replace("  ", " ")
    return sanitized.strip().replace(" ", "_")


### [Package metadata] ####################################################


@cache
def get_physicsnemo_pkg_info() -> dict[str, str | None]:
    """Return the PhysicsNeMo package version and current git commit hash.

    Returns:
        Dictionary with ``"version"`` (package version or ``None``) and
        ``"git_hash"`` (hex SHA or ``None`` if not in a git repository).
    """
    try:
        git_hash = git.Repo(search_parent_directories=True).head.commit.hexsha
    except git.InvalidGitRepositoryError:
        git_hash = None

    return {
        "version": getattr(physicsnemo, "__version__", None),
        "git_hash": git_hash,
    }


### [Checkpoint management] ###############################################


def extract_epoch(path: Path) -> int | None:
    """Extract epoch number from a checkpoint filename ``ClassName.<epoch>.pt``.

    Args:
        path: Checkpoint file path.

    Returns:
        Epoch integer, or ``None`` if the filename doesn't match.

    Examples:
        >>> extract_epoch(Path("Model.100.pt"))
        100
        >>> extract_epoch(Path("dir/MyModel.42.pt"))
        42
        >>> extract_epoch(Path("invalid_name.pt"))
    """
    match = re.match(r".*\.(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def get_latest_checkpoint_path(
    output_dir: Path, only_use_best: bool = False
) -> Path | None:
    """Find the checkpoint with the highest epoch number in *output_dir*.

    Searches ``models/`` and ``models/best_model/`` subdirectories for
    ``*.pt`` files and returns the one with the highest epoch number
    (extracted from the filename).

    Args:
        output_dir: Root output directory containing model checkpoints.
        only_use_best: Only consider checkpoints in ``models/best_model/``.

    Returns:
        Path to the latest checkpoint, or ``None`` if none are found.
    """
    models_dir = output_dir / "models"
    best_model_dir = models_dir / "best_model"

    checkpoint_dirs = [best_model_dir]
    if not only_use_best:
        checkpoint_dirs.append(models_dir)

    checkpoint_paths: list[Path] = []
    for directory in checkpoint_dirs:
        if directory.is_dir():
            checkpoint_paths.extend(directory.glob("*.pt"))

    if not checkpoint_paths:
        return None

    def sort_key(p: Path) -> int:
        epoch = extract_epoch(p)
        return -1 if epoch is None else epoch

    return max(checkpoint_paths, key=sort_key)


def save_training_checkpoint(
    save_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    *,
    extra_state: dict[str, Any] | None = None,
    mlflow_run_id: str | None = None,
    keep_only_latest: bool = False,
) -> Path:
    """Save a training checkpoint to disk.

    The checkpoint file is named ``{ModelClass}.{epoch}.pt`` and contains
    model, optimizer, scheduler, and scaler state dicts, plus any caller-
    provided extra state (e.g. ``best_loss``, ``mlflow_run_id``).

    Args:
        save_dir: Directory to write the checkpoint into.
        epoch: Current epoch number (used in the filename).
        model: The unwrapped (non-DDP) model.
        optimizer: Optimizer whose state is saved.
        scheduler: Learning-rate scheduler whose state is saved.
        scaler: AMP gradient scaler whose state is saved.
        extra_state: Additional key-value pairs to include in the
            checkpoint dict (e.g. ``{"best_loss": 0.5}``).
        mlflow_run_id: Active MLflow run ID, persisted so a resumed
            job can reopen the same run.
        keep_only_latest: If True, delete all other ``.pt`` files in
            *save_dir* after saving.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_path = save_dir / f"{model.__class__.__name__}.{epoch:d}.pt"
    state: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "mlflow_run_id": mlflow_run_id,
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, checkpoint_path)
    if keep_only_latest:
        for old in save_dir.glob("*.pt"):
            if old != checkpoint_path:
                old.unlink()
    return checkpoint_path


def load_training_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> dict[str, Any]:
    """Load a training checkpoint and restore stateful objects in-place.

    Restores the model, optimizer, scheduler, and scaler state dicts from
    the checkpoint. Returns the full checkpoint dict so the caller can
    extract extra fields (``epoch``, ``mlflow_run_id``, ``best_loss``, etc.).

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        model: Model to load weights into (should be the unwrapped model).
        optimizer: Optimizer to restore.
        scheduler: Scheduler to restore.
        scaler: AMP scaler to restore.
        device: Device to map tensors to.

    Returns:
        The raw checkpoint dict. Standard keys include ``"epoch"`` and
        ``"mlflow_run_id"``; extra keys depend on what the saver stored.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


### [Hyperparameter logging] ##############################################


def log_hyperparameters(
    log_dir: Path, model: torch.nn.Module, other_hyperparameters: dict[str, Any]
) -> None:
    """Log model and training hyperparameters to YAML (and MLflow if active).

    Extracts model constructor parameters by introspecting ``__init__`` and
    matching against instance attributes. Complex objects (tensors, devices,
    etc.) are converted to YAML-safe representations.

    Args:
        log_dir: Directory for ``hyperparameters.yaml``. Created if needed.
        model: PyTorch model whose constructor params are logged.
        other_hyperparameters: Additional key-value pairs to log (training
            config, optimizer settings, etc.).
    """

    def to_serializable(obj: Any) -> Any:
        """Recursively convert *obj* to a YAML-safe representation."""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if isinstance(obj, (list, tuple, set)):
            return [to_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {str(to_serializable(k)): to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, Path):
            return str(obj)

        try:
            if isinstance(obj, torch.Tensor):
                return (
                    obj.tolist()
                    if obj.numel() <= 32
                    else f"Tensor(shape={tuple(obj.shape)})"
                )
            if isinstance(obj, torch.device):
                return str(obj)
            if isinstance(obj, torch.dtype):
                return str(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist() if obj.size <= 32 else f"ndarray(shape={obj.shape})"
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to serialize {obj} with error {e}")
            return str(obj)

        return str(obj)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    constructor_params: list[str] = list(
        inspect.signature(type(model).__init__).parameters.keys()
    )[1:]  # Skip 'self'

    model_hyperparameters = {
        param: to_serializable(getattr(model, param))
        for param in constructor_params
        if hasattr(model, param)
    }
    other_hyperparameters = {
        k: to_serializable(v) for k, v in other_hyperparameters.items()
    }

    with open(log_dir / "hyperparameters.yaml", "w") as f:
        yaml.safe_dump(
            {
                "model": model_hyperparameters,
                **other_hyperparameters,
            },
            f,
            default_flow_style=False,
            indent=2,
            sort_keys=False,
        )

    if mlflow.active_run():
        mlflow.log_params(
            {
                **model_hyperparameters,
                **other_hyperparameters,
            }
        )


### [Device transfer] #####################################################

TransferableType = (
    torch.Tensor
    | Mesh
    | TensorDict
    | list
    | tuple
    | dict
    | set
    | float
    | int
    | bool
    | complex
)


def to(
    data: TransferableType, device: torch.device, dtype: torch.dtype | None = None
) -> TransferableType:
    """Recursively transfer nested data structures to a PyTorch device.

    Walks dicts, lists, tuples, and sets, calling ``.to()`` on Tensor,
    TensorDict, and Mesh leaves. Python numeric scalars are converted to
    tensors via ``torch.as_tensor``.

    Args:
        data: Nested structure of tensors, TensorDicts, Meshes, and containers.
        device: Target device.
        dtype: Optional target dtype (applied only to Tensor/TensorDict/Mesh).

    Returns:
        Same structure with all tensor-like leaves on *device*.

    Raises:
        NotImplementedError: If *data* contains an unsupported leaf type.
    """
    if isinstance(data, (torch.Tensor, Mesh, TensorDict)):
        return data.to(device=device, dtype=dtype)
    elif isinstance(data, (float, int, bool, complex)):
        return torch.as_tensor(data, device=device, dtype=dtype)
    elif isinstance(data, list):
        return [to(item, device=device, dtype=dtype) for item in data]
    elif isinstance(data, tuple):
        return tuple(to(item, device=device, dtype=dtype) for item in data)
    elif isinstance(data, dict):
        return {k: to(v, device=device, dtype=dtype) for k, v in data.items()}
    elif isinstance(data, set):
        return {to(item, device=device, dtype=dtype) for item in data}
    else:
        raise NotImplementedError(
            f"`to` doesn't have a device-transfer recipe registered for {type(data)=!r}."
        )


### [Signal handling] #####################################################


def install_graceful_shutdown(rank: int = 0) -> Callable[[], bool]:
    """Install signal handlers for graceful training shutdown.

    Catches SIGTERM, SIGINT, and SIGQUIT. On the first signal a message is
    printed (on rank 0) and an internal flag is set. The training loop can
    poll the returned callable each epoch to decide whether to break.

    Args:
        rank: Distributed rank. Only rank 0 prints the signal message.

    Returns:
        A zero-argument callable that returns ``True`` once a shutdown
        signal has been received.
    """
    received = [False]

    def _handler(signum: int, _frame: Any) -> None:
        if rank == 0:
            print(f"{signal.Signals(signum).name} received; quitting after this epoch.")
        received[0] = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT):
        signal.signal(sig, _handler)

    return lambda: received[0]


if __name__ == "__main__":
    print(get_physicsnemo_pkg_info())
