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

"""
Lightweight path / working-directory utilities.

This module deliberately has no third-party imports so it can be exercised
in test environments without heavy dependencies.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def pushd(target):
    """Context manager that temporarily changes the working directory.

    The previous working directory is restored on exit, even if the
    ``with``-block raises. This eliminates the silent-corruption class of
    bugs where an exception fires inside an ``os.chdir(target); ...;
    os.chdir(oldfolder)`` pair and leaves the process with the wrong
    current directory.

    Parameters
    ----------
    target : str or os.PathLike
        Directory to ``chdir`` into for the duration of the ``with`` block.

    Yields
    ------
    pathlib.Path
        The absolute resolved target directory, for convenience when callers
        want to construct paths relative to it.

    Examples
    --------
    Replaces the common pattern::

        oldfolder = os.getcwd()
        os.chdir(target)
        try:
            do_thing()
        finally:
            os.chdir(oldfolder)

    with::

        with pushd(target):
            do_thing()
    """
    previous = os.getcwd()
    target_path = Path(target).resolve()
    os.chdir(target_path)
    try:
        yield target_path
    finally:
        os.chdir(previous)
