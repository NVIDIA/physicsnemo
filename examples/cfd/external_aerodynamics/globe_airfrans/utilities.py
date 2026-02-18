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
hyperparameter logging, and MLflow metric sanitization.
"""

import inspect
import re
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
    """Silence the verbose command-line output of ``torch.compile(..., mode="max-autotune")``."""
    from torch._inductor import config, select_algorithm

    config.max_autotune_report_choices_stats = False
    select_algorithm.PRINT_AUTOTUNE = False


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

TransferrableType = (
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
    data: TransferrableType, device: torch.device, dtype: torch.dtype | None = None
) -> TransferrableType:
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
        return torch.as_tensor(data, device=device)
    elif isinstance(data, list):
        return [to(item, device=device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to(item, device=device) for item in data)
    elif isinstance(data, dict):
        return {k: to(v, device=device) for k, v in data.items()}
    elif isinstance(data, set):
        return {to(item, device=device) for item in data}
    else:
        raise NotImplementedError(
            f"`to` doesn't have a device-transfer recipe registered for {type(data)=!r}."
        )


### [Distributed reduction] ###############################################


def reduce_over_ranks(
    x: torch.Tensor,
    op: Literal["mean", "sum", "max", "min"] = "mean",
) -> torch.Tensor:
    """All-reduce *x* across ranks using the specified operation.

    No-op when ``world_size == 1``. Modifies *x* in-place.

    Args:
        x: Tensor to reduce.
        op: Reduction operation.

    Returns:
        The same tensor object, now holding the reduced values.
    """
    if not DistributedManager.is_initialized():
        raise RuntimeError(
            "Distributed manager should be initialized when using reduce_over_ranks"
        )
    dist = DistributedManager()

    if dist.world_size != 1:
        op_map = {
            "mean": torch.distributed.ReduceOp.AVG,
            "sum": torch.distributed.ReduceOp.SUM,
            "max": torch.distributed.ReduceOp.MAX,
            "min": torch.distributed.ReduceOp.MIN,
        }
        torch.distributed.all_reduce(x, op=op_map[op])

    return x


if __name__ == "__main__":
    print(get_physicsnemo_pkg_info())
