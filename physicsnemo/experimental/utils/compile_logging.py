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
import re
from dataclasses import dataclass
from pathlib import Path

_FRAME_RE = re.compile(r'File "(.+?)", line (\d+), in (.+)')


@dataclass
class _BreakRecord:
    """Internal record for a single graph-break event."""

    reason: str
    user_caller: str
    raw_message: str


class CompileDiagnosticsCollector(logging.Filter):
    """Logging filter that captures ``torch.compile`` graph breaks and recompiles.

    Intercepts verbose ``torch._dynamo`` log records (emitted when
    ``torch._logging.set_logs(graph_breaks=True, recompiles=True)`` is
    active), extracts compact summaries, and suppresses the multi-line
    stack traces that would otherwise flood the training log.

    Usage::

        torch._logging.set_logs(graph_breaks=True, recompiles=True)
        collector = CompileDiagnosticsCollector()
        collector.install()
        # ... run one training batch ...
        torch._logging.set_logs(graph_breaks=False, recompiles=False)
        collector.uninstall()
        print(collector.summary())
        # For full stack traces when investigating:
        # print(collector.detailed_summary())
    """

    def __init__(self) -> None:
        super().__init__()
        self.active = True
        self.graph_breaks: dict[str, _BreakRecord] = {}
        self.recompiles: dict[str, str] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith("torch._dynamo"):
            return True
        if record.levelno >= logging.WARNING:
            return True

        # Each graph break / recompile is a single multi-line log record.
        msg = record.getMessage()
        lines = msg.splitlines()

        if "Graph break in user code at" in msg:
            loc = ""
            for line in lines:
                if "Graph break in user code at" in line:
                    loc = line.split("Graph break in user code at ")[-1].strip()
                    break
            reason = ""
            for line in lines:
                if "Graph Break Reason:" in line:
                    reason = line.split("Graph Break Reason: ")[-1].strip()
                    break
            user_caller = self._extract_user_caller(loc, lines)
            self.graph_breaks.setdefault(
                loc, _BreakRecord(reason=reason, user_caller=user_caller, raw_message=msg)
            )

        elif "Recompiling function" in msg:
            loc = lines[0].split("Recompiling function ")[-1].strip()
            reason = ""
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("- "):
                    reason = stripped[2:].strip()
                    break
            self.recompiles.setdefault(loc, reason)

        return False

    # torch._dynamo sets propagate=False and uses its own StreamHandler,
    # so filters on the root logger never see its records. We install on
    # every handler across the relevant logger hierarchy.
    _LOGGER_NAMES = ("torch._dynamo", "torch", "")

    def install(self) -> None:
        """Add this filter to handlers on torch._dynamo, torch, and root loggers."""
        for name in self._LOGGER_NAMES:
            for handler in logging.getLogger(name).handlers:
                handler.addFilter(self)

    def uninstall(self) -> None:
        """Remove this filter from all installed handlers and deactivate."""
        for name in self._LOGGER_NAMES:
            for handler in logging.getLogger(name).handlers:
                handler.removeFilter(self)
        self.active = False

    @staticmethod
    def _shorten_path(loc: str) -> str:
        """Reduce ``/long/absolute/path/file.py:123`` to ``file.py:123``."""
        parts = loc.rsplit(":", 1)
        return Path(parts[0]).name + ":" + parts[1] if len(parts) == 2 else loc

    @classmethod
    def _extract_user_caller(cls, break_loc: str, lines: list[str]) -> str:
        """Find the nearest user-code caller from the "User code traceback" section.

        Parses standard ``File "path", line N, in func`` frames from the
        Dynamo log message.  Returns the deepest frame whose ``file:line``
        differs from ``break_loc`` (i.e. the caller, not the break itself).
        Falls back to the deepest frame if all frames match.
        """
        # Collect all traceback frames from the message.
        frames: list[tuple[str, str, str]] = []  # (path, line, func)
        for line in lines:
            m = _FRAME_RE.search(line)
            if m:
                frames.append((m.group(1), m.group(2), m.group(3)))
        if not frames:
            return ""

        break_short = cls._shorten_path(break_loc)

        # Walk frames from deepest (last) to shallowest, looking for a
        # frame that is NOT the break location itself.
        for path, lineno, func in reversed(frames):
            frame_short = cls._shorten_path(f"{path}:{lineno}")
            if frame_short != break_short:
                return f"{frame_short} in {func}"

        # All frames match the break location; return the deepest with
        # its function name (still useful as context).
        path, lineno, func = frames[-1]
        return f"{cls._shorten_path(f'{path}:{lineno}')} in {func}"

    def summary(self) -> str:
        """Format collected diagnostics as a compact multi-line string."""
        sections: list[str] = []

        if self.graph_breaks:
            lines = ["torch.compile graph breaks:"]
            for loc, rec in self.graph_breaks.items():
                short_reason = rec.reason.split("\n")[0][:80]
                lines.append(f"  {self._shorten_path(loc):<40s} {short_reason}")
                if rec.user_caller:
                    lines.append(f"    <- {rec.user_caller}")
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

    def detailed_summary(self) -> str:
        """Full Dynamo log messages for each graph break, for deep investigations."""
        if not self.graph_breaks:
            return "No graph breaks recorded."
        sections: list[str] = []
        for loc, rec in self.graph_breaks.items():
            sections.append(f"=== Graph break: {self._shorten_path(loc)} ===")
            sections.append(rec.raw_message)
            sections.append("")
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
