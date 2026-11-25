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

"""The skip marks of the ISLA tests that need CUDA, or CUDA and Triton.

A CUDA test carries ``@pytest.mark.cuda`` (the marker the GPU tier selects on)
and one of these skips; a CUDA parameter carries both in its ``marks``. The
``cuda`` marker alone skips nothing: ``test/conftest.py`` and the ISLA
``conftest.py`` only register it.

Triton counts as present when ISLA's one loader (``_triton.HAS_TRITON``) found it."""

import pytest
import torch

from physicsnemo.experimental.models.isla._triton import HAS_TRITON

HAS_CUDA = torch.cuda.is_available()

requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
requires_cuda_triton = pytest.mark.skipif(
    not (HAS_CUDA and HAS_TRITON), reason="needs CUDA and Triton"
)
#: The ``device`` parameter of a test that runs on CPU and, where present, on CUDA.
DEVICES = ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, requires_cuda])]
