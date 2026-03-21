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

Contains helpers for hyperparameter logging and MLflow metric sanitization.
"""

import inspect
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from mlflow.tracking.fluent import active_run, log_params
from tenacity import retry, stop_after_attempt, wait_fixed

from physicsnemo.utils.logging import PythonLogger

logger = PythonLogger("globe.airfrans.utilities")

### [torch.compile helpers] ###############################################


class CompileDiagnosticsCollector(logging.Filter):
    """Logging filter that captures ``torch.compile`` graph breaks and recompiles.

    Intercepts verbose ``torch._dynamo`` log records (emitted when
    ``torch._logging.set_logs(graph_breaks=True, recompiles=True)`` is
    active), extracts compact summaries, and suppresses the multi-line
    stack traces that would otherwise flood the training log.

    Usage::

        collector = CompileDiagnosticsCollector()
        collector.install()
        torch._logging.set_logs(graph_breaks=True, recompiles=True)
        # ... run one training batch ...
        torch._logging.set_logs(graph_breaks=False, recompiles=False)
        collector.uninstall()
        print(collector.summary())
    """

    def __init__(self) -> None:
        super().__init__()
        self.graph_breaks: dict[str, str] = {}
        self.recompiles: dict[str, str] = {}
        self._pending_break_loc: str | None = None
        self._pending_recompile_loc: str | None = None

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith("torch._dynamo"):
            return True
        if record.levelno >= logging.DEBUG:
            return True

        msg = record.getMessage()

        ### Graph breaks
        if "Graph break in user code at" in msg:
            self._pending_break_loc = msg.split(
                "Graph break in user code at "
            )[-1].strip()
        elif self._pending_break_loc and "Graph Break Reason:" in msg:
            reason = msg.split("Graph Break Reason: ")[-1].strip()
            self.graph_breaks.setdefault(self._pending_break_loc, reason)
            self._pending_break_loc = None

        ### Recompiles
        if "Recompiling function" in msg:
            self._pending_recompile_loc = msg.split(
                "Recompiling function "
            )[-1].strip()
        elif self._pending_recompile_loc and "triggered by" not in msg:
            guard = msg.strip().lstrip("- ")
            self.recompiles.setdefault(self._pending_recompile_loc, guard)
            self._pending_recompile_loc = None

        return False

    def install(self) -> None:
        """Add this filter to all handlers on the root logger."""
        for handler in logging.getLogger().handlers:
            handler.addFilter(self)

    def uninstall(self) -> None:
        """Remove this filter from all handlers on the root logger."""
        for handler in logging.getLogger().handlers:
            handler.removeFilter(self)

    @staticmethod
    def _shorten_path(loc: str) -> str:
        """Reduce ``/long/absolute/path/file.py:123`` to ``file.py:123``."""
        parts = loc.rsplit(":", 1)
        return Path(parts[0]).name + ":" + parts[1] if len(parts) == 2 else loc

    def summary(self) -> str:
        """Format collected diagnostics as a compact multi-line string."""
        sections: list[str] = []

        if self.graph_breaks:
            lines = ["torch.compile graph breaks:"]
            for loc, reason in self.graph_breaks.items():
                short_reason = reason.split("\n")[0][:80]
                lines.append(f"  {self._shorten_path(loc):<40s} {short_reason}")
            sections.append("\n".join(lines))
        else:
            sections.append("torch.compile graph breaks: (none)")

        if self.recompiles:
            lines = ["torch.compile recompiles:"]
            for loc, guard in self.recompiles.items():
                lines.append(f"  {loc}")
                lines.append(f"    reason: {guard[:100]}")
            sections.append("\n".join(lines))

        return "\n".join(sections)


### [Resilient MLflow helpers] ############################################

resilient = retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    retry_error_callback=lambda rs: logger.warning(
        f"{rs.fn.__name__}() failed after {rs.attempt_number} attempts, skipping."
    ),
)


def disable_autotune_printing() -> None:
    """Silence the verbose command-line output of ``torch.compile(..., mode="max-autotune")``.

    Uses private ``torch._inductor`` APIs that may change across PyTorch
    versions, so failures are silently ignored.
    """
    try:
        from torch._inductor import config, select_algorithm

        config.max_autotune_report_choices_stats = False
        select_algorithm.PRINT_AUTOTUNE = False  # ty: ignore[invalid-assignment]
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
