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

"""Backward-compatibility shim for the former monolithic noise schedulers module.

Each noise scheduler now lives in its own module. Import the classes below from
:mod:`physicsnemo.diffusion.noise_schedulers` instead.
"""

import warnings

from .base import NoiseScheduler  # noqa: F401
from .edm import EDMNoiseScheduler  # noqa: F401
from .edm_log_uniform import EDMLogUniformNoiseScheduler  # noqa: F401
from .iddpm import IDDPMNoiseScheduler  # noqa: F401
from .linear_gaussian import LinearGaussianNoiseScheduler  # noqa: F401
from .student_t_edm import StudentTEDMNoiseScheduler  # noqa: F401
from .ve import VENoiseScheduler  # noqa: F401
from .vp import VPNoiseScheduler  # noqa: F401

warnings.warn(
    "The module 'physicsnemo.diffusion.noise_schedulers.noise_schedulers' is "
    "deprecated and will be removed in a future release. The noise scheduler "
    "classes are unchanged, but each one now lives in its own module. Import "
    "them from 'physicsnemo.diffusion.noise_schedulers' instead.",
    DeprecationWarning,
    stacklevel=2,
)
