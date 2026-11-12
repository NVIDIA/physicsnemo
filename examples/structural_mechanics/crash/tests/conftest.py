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

import pytest


@pytest.fixture(autouse=True)
def torch_layer_norm(monkeypatch):
    """Use PyTorch's LayerNorm in the mesh MLPs of these CPU tests.

    The models' mesh MLPs take their LayerNorm from
    :func:`physicsnemo.nn.module.layer_norm.get_layer_norm_class`, which picks
    Transformer Engine's implementation whenever the package is installed and a
    GPU is present; that implementation cannot run on the CPU these tests use.
    """
    monkeypatch.setenv("PHYSICSNEMO_FORCE_TE", "False")
