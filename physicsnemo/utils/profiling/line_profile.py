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

from pathlib import Path
from typing import Any, Callable

from .core import PhysicsNeMoProfilerWrapper, _Profiler_Singleton

try:
    from line_profiler import LineProfiler

    lp_avail = True
except ImportError:
    lp_avail = False

import warnings


class LineProfileWrapper(PhysicsNeMoProfilerWrapper, metaclass=_Profiler_Singleton):
    """Wrapper class for line-by-line profiling using line_profiler package.

    This class provides line-level profiling capabilities by wrapping functions
    with the LineProfiler from the line_profiler package. It operates as a decorator
    and can output detailed profiling statistics.
    """

    _name: str = "line_profiler"

    def __init__(self, **config_overrides: Any) -> None:
        """Initialize the line profiler wrapper.

        Args:
            **config_overrides: Optional configuration overrides for the profiler
        """
        # Pytorch is a context and annotation but not a wrapper:
        self._is_context: bool = False
        self._is_decorator: bool = True

    def _standup(self) -> None:
        """Initialize the line profiler instance.

        Sets up the LineProfiler if available, otherwise disables profiling functionality
        with a warning.
        """
        if lp_avail:
            self._profiler = LineProfiler()
        else:
            warnings.warn(
                "Line Profiler was requested by the physicsnemo profiler but "
                "isn't install.  Try `pip install line_profiler`."
            )
            self._profiler = None
            self.enabled = False
        self.initialized = True

    def finalize(self, output_top: Path) -> None:
        """Serialize the line_profiler output to a file.

        Args:
            output_top: Path to the directory where profiling results should be saved
        """
        if not lp_avail:
            return

        if not self.enabled:
            return

        # Avoid finalizing if we never initialized:
        if not self.initialized:
            return

        # Prevent double finalization:
        if self.finalized:
            return

        # Get the output directory:
        out_top = self.output_dir(output_top)
        with open(out_top / Path("profiler_stats.txt"), "w") as stats:
            self._profiler.print_stats(stream=stats, stripzeros=True)

        # Make this profiler completed:
        self.finalized = True

    def __call__(self, fn: Callable) -> Callable:
        """Decorator implementation for profiling functions.

        Args:
            fn: The function to be profiled

        Returns:
            The profiled function wrapped with LineProfiler
        """
        if not lp_avail:
            return fn
        f = self._profiler(fn)
        return f
