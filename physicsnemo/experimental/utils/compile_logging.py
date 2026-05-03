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


def silence_compile_logs_on_non_zero_ranks(rank: int) -> None:
    """Suppress non-error torch compile logs on every rank except rank 0.

    ``torch.compile`` can emit verbose graph-break and recompile records (50-200
    lines per event) that are identical across data-parallel ranks; when every
    rank's logger forwards them, the merged multi-rank log becomes effectively
    useless for triage. This installs a callable filter on handlers attached to
    the ``torch._dynamo``, ``torch._inductor``, ``torch``, and root loggers
    that drops records below ``logging.WARNING`` when ``rank != 0``. Real
    warnings and errors are preserved on every rank, and rank 0 is left
    untouched so the developer still sees full Dynamo / Inductor output.

    Idempotent and safe to call before or after distributed init, but should
    be called once ``rank`` is known and once per process.

    Parameters
    ----------
    rank : int
        Distributed rank of the current process. ``0`` is a no-op; all other
        ranks get the filter installed.

    Examples
    --------
    >>> import logging
    >>> from physicsnemo.experimental.utils import silence_compile_logs_on_non_zero_ranks
    >>> silence_compile_logs_on_non_zero_ranks(0)  # no-op
    >>> silence_compile_logs_on_non_zero_ranks(1)  # filter installed
    """
    if rank == 0:
        return

    def _drop_non_error_compile_logs(record: logging.LogRecord) -> bool:
        return (
            not record.name.startswith(("torch._dynamo", "torch._inductor"))
            or record.levelno >= logging.WARNING
        )

    for logger_name in ("torch._dynamo", "torch._inductor", "torch", ""):
        for handler in logging.getLogger(logger_name).handlers:
            handler.addFilter(_drop_non_error_compile_logs)


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
