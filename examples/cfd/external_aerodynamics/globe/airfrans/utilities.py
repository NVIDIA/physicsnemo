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

Contains helpers for hyperparameter logging, MLflow metric sanitization,
and signal handling.
"""

import inspect
import logging
import signal
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import git
import mlflow
import numpy as np
import torch
import yaml

import physicsnemo
from physicsnemo.utils.logging import PythonLogger

logger = PythonLogger("globe.airfrans.utilities")

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
        _MLFLOW_MAX_PARAM_LENGTH = 6000
        all_params = {**model_hyperparameters, **other_hyperparameters}
        mlflow.log_params(
            {k: v for k, v in all_params.items() if len(str(v)) <= _MLFLOW_MAX_PARAM_LENGTH}
        )


### [Signal handling] #####################################################


def install_graceful_shutdown(rank: int = 0) -> Callable[[], bool]:
    """Install signal handlers for graceful training shutdown.

    Catches SIGTERM, SIGINT, and SIGQUIT. On the first signal a warning is
    logged (on rank 0) and an internal flag is set. The training loop can
    poll the returned callable each epoch to decide whether to break.

    Args:
        rank: Distributed rank. Only rank 0 logs the signal message.

    Returns:
        A zero-argument callable that returns ``True`` once a shutdown
        signal has been received.
    """
    received = [False]

    def _handler(signum: int, _frame: Any) -> None:
        if rank == 0:
            logger.warning(f"{signal.Signals(signum).name} received; quitting after this epoch.")
        received[0] = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT):
        signal.signal(sig, _handler)

    return lambda: received[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(str(get_physicsnemo_pkg_info()))
