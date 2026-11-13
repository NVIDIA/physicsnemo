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

"""Thread-correct check for whether the current code is being traced."""

import torch


def is_tracing() -> bool:
    """Return whether the current code is being traced into a graph.

    True while Dynamo traces the current frame or while ``torch.export`` runs,
    which is what ``torch.compiler.is_compiling()`` meant before torch 2.13.
    False in eager code, including other threads while one thread compiles:
    since torch 2.13, ``is_compiling()`` stays True in every thread for a
    whole compile session, so it must not guard code that data-loading
    threads run eagerly.
    """
    return torch.compiler.is_dynamo_compiling() or torch.compiler.is_exporting()
