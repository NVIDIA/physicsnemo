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

r"""Shared :class:`physicsnemo.core.meta.ModelMetaData` for the AeroJEPA stack.

Every AeroJEPA submodule (trunk, predictor, decoder, encoders) inherits from
:class:`physicsnemo.core.module.Module` so the top-level model serialises
cleanly via ``Module.save()``. They all share a single ``AeroJEPAMetaData``
because the capability flags (no JIT, no CUDA graphs, AMP, no ONNX) are
identical across the stack.
"""

from __future__ import annotations

from dataclasses import dataclass

from physicsnemo.core.meta import ModelMetaData


@dataclass
class AeroJEPAMetaData(ModelMetaData):
    r"""Meta-data for the :class:`AeroJEPA` model and its submodules."""

    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False
