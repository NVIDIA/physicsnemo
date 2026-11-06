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

r"""Optional training loop for diffusion distillation."""

import importlib
from typing import Any

from physicsnemo.core.version_check import check_version_spec

_AVAILABLE = check_version_spec("nvalchemi-toolkit", "0.2.0", hard_fail=False)
if _AVAILABLE:
    DistillationStrategy = importlib.import_module(
        f"{__package__}._alchemi"
    ).DistillationStrategy
else:

    class DistillationStrategy:
        r"""Install ``diffusion-strategy`` to run a distillation training loop."""

        def __init__(self, **kwargs: Any) -> None:
            raise ImportError(
                "PhysicsNeMo DistillationStrategy requires nvalchemi-toolkit>=0.2.0. "
                "Install nvidia-physicsnemo[diffusion-strategy] for the tested revision."
            )


__all__ = ["DistillationStrategy"]
