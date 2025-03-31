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

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.profiler import ProfilerActivity, profile

from .core import PhysicsNeMoProfilerWrapper, _Profiler_Singleton


@dataclass
class TorchProfilerConfig:
    """
    Specific configuration for the pytorch profiler.

    Attributes:
        name: Name identifier for this profiler configuration
        torch_prof_activities: List of PyTorch profiler activities to monitor
        record_shapes: Whether to record tensor shapes
        profile_memory: Whether to profile memory usage
        with_stack: Whether to record stack traces
        with_flops: Whether to record FLOPs
        schedule: Optional scheduling function for the profiler
        on_trace_ready_path: Optional path to save trace files
    """

    name: str = "torch"
    torch_prof_activities: Optional[Tuple[ProfilerActivity, ...]] = None
    record_shapes: bool = True
    with_stack: bool = False
    profile_memory: bool = True
    with_flops: bool = True
    schedule: Optional[Callable] = None
    on_trace_ready_path: Optional[Path] = None


class TorchProfileWrapper(PhysicsNeMoProfilerWrapper, metaclass=_Profiler_Singleton):
    """Wrapper class for PyTorch profiler functionality.

    This class wraps PyTorch's built-in profiler to integrate with PhysicsNeMo's profiling system.
    It supports context manager usage for profiling code blocks.

    Attributes:
        _name: Name identifier for this profiler
        _is_context: Whether this profiler supports context manager usage
        _is_decorator: Whether this profiler supports decorator usage
    """

    _name: str = "torch"

    # Overload any of these:
    _is_context: bool = True
    _is_decorator: bool = False

    def __init__(
        self, config: Optional[TorchProfilerConfig] = None, **config_overrides
    ) -> None:
        """Initialize the PyTorch profiler wrapper.

        Args:
            config: Optional configuration object for the profiler
            **config_overrides: Optional keyword arguments to override config values
        """
        default_config = TorchProfilerConfig()

        # Replace any overrides right into the config:
        if config is None:
            self._config = replace(default_config, **config_overrides)
        else:
            self._config = replace(config, **config_overrides)

        # Configure pytorch profiler here:
        # Set the default profiling activities if not set:
        if self._config.torch_prof_activities is None:
            torch_prof_activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                torch_prof_activities.append(ProfilerActivity.CUDA)
            self._config.torch_prof_activities = torch_prof_activities

        return

    def _standup(self) -> None:
        """Initialize the PyTorch profiler with configured settings."""
        if self._config.on_trace_ready_path is not None:
            on_trace_ready = torch.profiler.tensorboard_trace_handler(
                self._config.on_trace_ready_path
            )
        else:
            on_trace_ready = None

        self._profiler = profile(
            activities=self._config.torch_prof_activities,
            profile_memory=self._config.profile_memory,
            record_shapes=self._config.record_shapes,
            with_stack=self._config.with_stack,
            schedule=self._config.schedule,
            with_flops=self._config.with_flops,
            on_trace_ready=on_trace_ready,
        )

        self._initialized = True

    def finalize(self, output_top: Path) -> None:
        """Finalize profiling and write results to disk.

        Args:
            output_top: Base output directory path for profiling results
        """
        if not self.enabled:
            return

        # Avoid finalizing if we never initialized or already finalized:
        if self.finalized:
            return

        # Get the output directory:
        out_top = self.output_dir(output_top)
        if self._profiler is not None:

            try:
                averages = self._profiler.key_averages()
            except AssertionError:
                # no averages recorded!
                averages = None

            # Write out torch profiling results:
            if averages:
                with open(out_top / Path("cpu_time.txt"), "w") as cpu_times:
                    times = averages.table()
                    cpu_times.write(times)

                with open(out_top / Path("gpu_time.txt"), "w") as gpu_times:
                    times = averages.table(sort_by="cuda_time_total")
                    gpu_times.write(times)

            if self._config.on_trace_ready_path is None:
                # Store the trace
                trace_path = out_top / Path("trace.json")
                self._profiler.export_chrome_trace(str(trace_path))

        # Make this profiler completed:
        self.finalized = True

    def __enter__(self) -> "TorchProfileWrapper":
        """Enter the profiling context.

        Returns:
            Self reference for context manager usage
        """
        self._profiler.__enter__()
        return self

    def __exit__(
        self, *exc: Tuple[Optional[type], Optional[Exception], Optional[str]]
    ) -> None:
        """Exit the profiling context.

        Args:
            *exc: Exception information if an error occurred
        """
        self._profiler.__exit__(*exc)

    def step(self) -> None:
        """Advance the profiler's step counter."""
        self._profiler.step()
