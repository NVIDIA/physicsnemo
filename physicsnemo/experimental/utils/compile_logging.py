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

"""Utilities for taming ``torch.compile`` verbosity in training scripts."""

import logging
from pathlib import Path


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
        self.active = True
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
        """Remove this filter from all handlers on the root logger and deactivate."""
        for handler in logging.getLogger().handlers:
            handler.removeFilter(self)
        self.active = False

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


def disable_autotune_printing() -> None:
    """Silence the verbose output of ``torch.compile(..., mode="max-autotune")``.

    Uses private ``torch._inductor`` APIs that may change across PyTorch
    versions, so failures are silently ignored.
    """
    try:
        from torch._inductor import config, select_algorithm

        config.max_autotune_report_choices_stats = False
        select_algorithm.PRINT_AUTOTUNE = False
    except (ImportError, AttributeError):
        pass
