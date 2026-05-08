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

from .noise_schedulers import (  # noqa: F401
    EDMLogUniformNoiseScheduler,
    EDMNoiseScheduler,
    IDDPMNoiseScheduler,
    LinearGaussianNoiseScheduler,
    NoiseScheduler,
    StudentTEDMNoiseScheduler,
    VENoiseScheduler,
    VPNoiseScheduler,
)


def __getattr__(name: str):
    if name == "DomainParallelNoiseScheduler":
        try:
            from .domain_parallel import DomainParallelNoiseScheduler

            return DomainParallelNoiseScheduler
        except ImportError as exc:
            raise ImportError(
                "DomainParallelNoiseScheduler requires optional distributed "
                "dependencies (physicsnemo.domain_parallel). Check that your "
                "torch version is compatible with this release of PhysicsNeMo. "
                f"Original error: {exc}"
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
