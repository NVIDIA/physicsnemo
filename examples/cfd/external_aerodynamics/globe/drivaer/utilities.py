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

"""MLflow and hyperparameter logging utilities for the GLOBE DrivAerML example."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from mlflow.tracking.fluent import active_run, log_params
from tenacity import retry, stop_after_attempt, wait_fixed

from physicsnemo.utils.logging import PythonLogger

logger = PythonLogger("globe.drivaer.utilities")

### [MLflow helpers] ######################################################


resilient = retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    retry_error_callback=lambda rs: logger.warning(
        f"{rs.fn.__name__}() failed after {rs.attempt_number} attempts, skipping."
    ),
)


def sanitize_metric_name(name: str) -> str:
    """Replace characters not in ``[A-Za-z0-9_.- :/]`` with underscores.

    Args:
        name: Original metric name (may contain special characters).

    Returns:
        Sanitized name safe for MLflow metric keys.
    """
    import string

    allowed_chars = set(string.ascii_letters + string.digits + "_-. :")
    sanitized = "".join(c if c in allowed_chars else " " for c in name)
    while "  " in sanitized:
        sanitized = sanitized.replace("  ", " ")
    return sanitized.strip().replace(" ", "_")


### [Hyperparameter logging] ##############################################


def log_hyperparameters(
    log_dir: Path, model: torch.nn.Module, other_hyperparameters: dict[str, Any]
) -> None:
    """Log model and training hyperparameters to YAML (and MLflow if active).

    Extracts model constructor parameters by introspecting ``__init__`` and
    matching against instance attributes.

    Args:
        log_dir: Directory for ``hyperparameters.yaml``.
        model: PyTorch model whose constructor params are logged.
        other_hyperparameters: Additional key-value pairs to log.
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

    ### Use the canonical constructor args captured by Module.__new__,
    ### which is the same source-of-truth that Module.save() serializes.
    ### This avoids the fragile pattern of checking hasattr(model, param),
    ### which silently drops any constructor arg not stored as self.xxx.
    model_hyperparameters = {
        k: to_serializable(v) for k, v in model._args["__args__"].items()
    }
    other_hyperparameters = {
        k: to_serializable(v) for k, v in other_hyperparameters.items()
    }

    with open(log_dir / "hyperparameters.yaml", "w") as f:
        yaml.safe_dump(
            {"model": model_hyperparameters, **other_hyperparameters},
            f,
            default_flow_style=False,
            indent=2,
            sort_keys=False,
        )

    if active_run():
        _MLFLOW_MAX_PARAM_LENGTH = 6000
        all_params = {**model_hyperparameters, **other_hyperparameters}
        log_params(
            {
                k: v
                for k, v in all_params.items()
                if len(str(v)) <= _MLFLOW_MAX_PARAM_LENGTH
            }
        )
