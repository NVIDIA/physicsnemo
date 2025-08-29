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

import builtins
import os
import sys
from functools import cache
from io import StringIO

from physicsnemo.distributed import DistributedManager

# Track original stdout and print function
original_stdout = None
original_print = None
debug_mode = False  # Default to normal mode


@cache
def rank():
    return DistributedManager().rank


def is_master():
    """Check if the current rank is the master rank"""
    return rank() == 0


def set_debug_mode(enabled=True):
    """Enable or disable debug mode for print redirection."""
    global debug_mode
    debug_mode = enabled


def null_print(*args, **kwargs):
    """A no-op print function for non-zero ranks"""
    pass


def rank_prefixed_print(*args, **kwargs):
    """Print function that prefixes output with rank information"""
    if not args:
        # If no arguments, just print the rank prefix
        original_print(f"[RANK {rank()}]", **kwargs)
        return

    # Prepend rank prefix to the first argument
    prefix = f"[RANK {rank()}] "
    first_arg = prefix + str(args[0])

    # Print with the prefixed first argument and remaining args
    original_print(first_arg, *args[1:], **kwargs)


def pytest_configure(config):
    """
    Redirect stdout much earlier in the pytest lifecycle.
    This runs before tests are collected.
    """
    global original_stdout, original_print, debug_mode

    # Check if debug mode is enabled through pytest config or environment
    debug_mode = getattr(config.option, "distributed_debug", False) or os.environ.get(
        "PYTEST_DISTRIBUTED_DEBUG", ""
    ).lower() in ("1", "true", "yes")

    original_print = builtins.print

    # Only modify output for non-master ranks
    if not is_master():
        if debug_mode:
            # In debug mode, replace print with rank-prefixed version
            builtins.print = rank_prefixed_print
        else:
            # In normal mode, redirect stdout and suppress prints
            original_stdout = sys.stdout
            sys.stdout = StringIO()
            builtins.print = null_print


def pytest_unconfigure(config):
    """Ensure stdout and print function restoration"""
    global original_stdout, original_print

    # Always restore the original print function
    if original_print:
        builtins.print = original_print

    # Only restore stdout if it was redirected (non-debug mode)
    if not is_master() and original_stdout and not debug_mode:
        sys.stdout = original_stdout
