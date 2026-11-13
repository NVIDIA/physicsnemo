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

"""Triton, the optional dependency of ISLA's CUDA kernels, loaded in one place.

Triton ships with the CUDA wheels of torch and is absent on CPU-only installs. ISLA
therefore treats it as a locally optional import (``CODING_STANDARDS/EXTERNAL_IMPORTS.md``,
*Locally Optional Imports*): every Triton kernel has a torch reference path, which the
model runs when ``HAS_TRITON`` is False. ``HAS_TRITON`` is a soft version check for at
least :data:`TRITON_MIN_VERSION` (:func:`triton_available`, through
:func:`~physicsnemo.core.version_check.check_version_spec`). When it holds, this module
imports ``triton``, ``triton.language`` (``tl``) and its ``libdevice``. The modules
with ``@triton.jit`` kernels take them from here, and define their kernels only then.
"""

import importlib
import importlib.util

from packaging.version import parse

from physicsnemo.core import version_check

#: The oldest Triton ISLA's kernels are tested on.
TRITON_MIN_VERSION = "3.7.1"
### Follow-up: HealDA guards Triton with version_check.OptionalImport, which imports on first
### use and checks no version, while these kernels are defined at import and need the floor
### above. One guard for both needs OptionalImport to take a minimum version.


def triton_available(min_version: str = TRITON_MIN_VERSION) -> bool:
    """Whether a Triton of at least ``min_version`` is installed: by the version of the
    distribution named ``triton`` or, when no distribution of that name is installed
    but the module is (another distribution provides it, such as ``pytorch-triton``),
    by the module's own ``__version__``."""
    if version_check.check_version_spec("triton", min_version, hard_fail=False):
        return True
    if version_check.get_installed_version("triton") is not None:
        return False  # the distribution is older than min_version
    if importlib.util.find_spec("triton") is None:
        return False
    return parse(importlib.import_module("triton").__version__) >= parse(min_version)


triton = tl = libdevice = None
HAS_TRITON = triton_available()
if HAS_TRITON:
    triton = importlib.import_module("triton")
    tl = importlib.import_module("triton.language")
    libdevice = importlib.import_module("triton.language.extra.libdevice")
