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

"""Backward-compatibility shim for the former monolithic solvers module.

Each solver now lives in its own module. Import the classes below from
:mod:`physicsnemo.diffusion.samplers` instead.
"""

import warnings

from .base import Solver  # noqa: F401
from .edm_stochastic_euler import EDMStochasticEulerSolver  # noqa: F401
from .edm_stochastic_heun import EDMStochasticHeunSolver  # noqa: F401
from .euler import EulerSolver  # noqa: F401
from .heun import HeunSolver  # noqa: F401

warnings.warn(
    "The module 'physicsnemo.diffusion.samplers.solvers' is deprecated and will "
    "be removed in a future release. The solver classes 'Solver', 'EulerSolver', "
    "'HeunSolver', 'EDMStochasticEulerSolver', and 'EDMStochasticHeunSolver' are "
    "unchanged, but each one now lives in its own module. Import them from "
    "'physicsnemo.diffusion.samplers' instead.",
    DeprecationWarning,
    stacklevel=2,
)
