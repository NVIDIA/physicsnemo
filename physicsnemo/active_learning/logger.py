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

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import local
from typing import Any

from termcolor import colored

# Thread-local storage for context information
_context_storage = local()


class ActiveLearningLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically includes active learning iteration context.

    This adapter automatically adds iteration information to log messages
    by accessing the driver's current iteration state.

    See Also
    --------
    Driver : Uses this adapter for logging
    setup_active_learning_logger : Sets up loggers with this adapter
    """

    def __init__(self, logger: logging.Logger, driver_ref: Any = None):
        """Initialize the adapter with a logger and optional driver reference.

        Parameters
        ----------
        logger : :class:`logging.Logger`
            The underlying logger to adapt
        driver_ref : Any, optional
            Reference to the :class:`~physicsnemo.active_learning.driver.Driver` object to get iteration context from
        """
        super().__init__(logger, {})
        self.driver_ref = driver_ref

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Process the log message to add iteration, run ID, and phase context.

        Parameters
        ----------
        msg : str
            The log message
        kwargs : dict[str, Any]
            Additional keyword arguments

        Returns
        -------
        tuple[str, dict[str, Any]]
            Processed message and kwargs
        """
        # Add iteration, run ID, and phase context if driver reference is available
        if self.driver_ref is not None:
            extra = kwargs.get("extra", {})

            # Add iteration context
            if hasattr(self.driver_ref, "active_learning_step_idx"):
                iteration = getattr(self.driver_ref, "active_learning_step_idx", None)
                if iteration is not None:
                    extra["iteration"] = iteration

            # Add run ID context
            if hasattr(self.driver_ref, "run_id"):
                run_id = getattr(self.driver_ref, "run_id", None)
                if run_id is not None:
                    extra["run_id"] = run_id

            # Add current phase context
            if hasattr(self.driver_ref, "current_phase"):
                phase = getattr(self.driver_ref, "current_phase", None)
                if phase is not None:
                    extra["phase"] = phase

            if extra:
                kwargs["extra"] = extra

        return msg, kwargs


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging to files.

    This formatter converts log records to JSON format, including all
    contextual information and metadata for structured analysis.

    See Also
    --------
    setup_active_learning_logger : Uses this formatter for file handlers
    ContextFormatter : Alternative formatter for console output
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format

        Returns
        -------
        str
            JSON-formatted log message
        """
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add contextual information if available
        if hasattr(record, "context"):
            log_entry["context"] = record.context

        if hasattr(record, "caller_object"):
            log_entry["caller_object"] = record.caller_object

        if hasattr(record, "iteration"):
            log_entry["iteration"] = record.iteration

        if hasattr(record, "phase"):
            log_entry["phase"] = record.phase

        extra_keys = list(filter(lambda x: x not in log_entry, record.__dict__.keys()))
        # Add any extra fields
        for key in extra_keys:
            log_entry[key] = record.__dict__[key]

        return json.dumps(log_entry)


def _get_context_stack():
    """Get the context stack for the current thread."""
    if not hasattr(_context_storage, "context_stack"):
        _context_storage.context_stack = []
    return _context_storage.context_stack


class ContextFormatter(logging.Formatter):
    """Standard formatter that includes active learning context information with colors.

    See Also
    --------
    setup_active_learning_logger : Uses this formatter for console handlers
    JSONFormatter : Alternative formatter for file output
    """

    def format(self, record):
        # Build context string
        context_parts = []
        if hasattr(record, "caller_object") and record.caller_object:
            context_parts.append(f"obj:{record.caller_object}")
        if hasattr(record, "run_id") and record.run_id:
            context_parts.append(f"run:{record.run_id}")
        if hasattr(record, "iteration") and record.iteration is not None:
            context_parts.append(f"iter:{record.iteration}")
        if hasattr(record, "phase") and record.phase:
            context_parts.append(f"phase:{record.phase}")
        if hasattr(record, "context") and record.context:
            for key, value in record.context.items():
                context_parts.append(f"{key}:{value}")

        context_str = f"[{', '.join(context_parts)}]" if context_parts else ""

        # Use standard formatting
        base_msg = super().format(record)

        # Add color to the message based on level if termcolor is available
        if colored is not None:
            match record.levelno:
                case level if level >= logging.ERROR:
                    base_msg = colored(base_msg, "red")
                case level if level >= logging.WARNING:
                    base_msg = colored(base_msg, "yellow")
                case level if level >= logging.INFO:
                    base_msg = colored(base_msg, "white")
                case _:  # DEBUG
                    base_msg = colored(base_msg, "cyan")

        # Add colored context string
        if context_str:
            if colored is not None:
                context_str = colored(context_str, "blue")
            base_msg += f" {context_str}"

        return base_msg


class ContextInjectingFilter(logging.Filter):
    """Filter that injects contextual information into log records.

    See Also
    --------
    setup_active_learning_logger : Uses this filter for context injection
    log_context : Context manager that works with this filter
    """

    def filter(self, record):
        # Add context information from thread-local storage
        context_stack = _get_context_stack()
        if context_stack:
            current_context = context_stack[-1]
            if current_context["caller_object"]:
                record.caller_object = current_context["caller_object"]
            if current_context["iteration"] is not None:
                record.iteration = current_context["iteration"]
            if current_context.get("phase"):
                record.phase = current_context["phase"]
            if current_context["context"]:
                record.context = current_context["context"]
        return True


def setup_active_learning_logger(
    name: str,
    run_id: str,
    log_dir: str | Path = Path("active_learning_logs"),
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger with active learning-specific formatting and handlers.

    Parameters
    ----------
    name : str
        Logger name
    run_id : str
        Unique identifier for this run, used in log filename
    log_dir : str or :class:`pathlib.Path`, optional
        Directory to store log files, by default "./logs"
    level : int, optional
        Logging level, by default logging.INFO

    Returns
    -------
    :class:`logging.Logger`
        Configured standard Python logger

    See Also
    --------
    Driver : Uses this function to set up logging
    ActiveLearningLoggerAdapter : Logger adapter for context
    log_context : Context manager for logging

    Example
    -------
    >>> logger = setup_active_learning_logger("experiment", "run_001")
    >>> logger.info("Starting experiment")
    >>> with log_context(caller_object="Trainer", iteration=5):
    ...     logger.info("Training step")
    """
    # Get standard logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Disable propagation to prevent duplicate messages from parent loggers
    logger.propagate = False

    # Create log directory if it doesn't exist
    if isinstance(log_dir, str):
        log_dir_path = Path(log_dir)
    else:
        log_dir_path = log_dir
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Set up console handler with standard formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        ContextFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    console_handler.addFilter(ContextInjectingFilter())
    logger.addHandler(console_handler)

    # Set up file handler with JSON formatting
    log_file = log_dir_path / f"{run_id}.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    file_handler.addFilter(ContextInjectingFilter())
    logger.addHandler(file_handler)

    return logger


@contextmanager
def log_context(
    caller_object: str | None = None,
    iteration: int | None = None,
    phase: str | None = None,
    **kwargs: Any,
):
    """Context manager for adding contextual information to log messages.

    Parameters
    ----------
    caller_object : str, optional
        Name or identifier of the object making the log call
    iteration : int, optional
        Current iteration counter
    phase : str, optional
        Current phase of the active learning process
    **kwargs : Any
        Additional contextual key-value pairs

    See Also
    --------
    ContextInjectingFilter : Filter that injects context into log records
    setup_active_learning_logger : Sets up loggers with context support

    Example
    -------
    >>> from logging import getLogger
    >>> from physicsnemo.active_learning.logger import log_context
    >>> logger = getLogger("my_logger")
    >>> with log_context(caller_object="Trainer", iteration=5, phase="training", epoch=2):
    ...     logger.info("Processing batch")
    """
    context_info = {
        "caller_object": caller_object,
        "iteration": iteration,
        "phase": phase,
        "context": kwargs,
    }

    context_stack = _get_context_stack()
    context_stack.append(context_info)

    try:
        yield
    finally:
        context_stack.pop()
