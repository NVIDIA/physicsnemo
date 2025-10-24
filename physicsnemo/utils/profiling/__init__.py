# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import atexit

from .core import ProfileRegistry
from .interface import Profiler
from .line_profile import LineProfileWrapper
from .torch import TorchProfilerConfig, TorchProfileWrapper


def _register_profilers():
    ProfileRegistry.register_profiler("torch", TorchProfileWrapper)
    ProfileRegistry.register_profiler("line_profile", LineProfileWrapper)
    ProfileRegistry.register_profiler("line_profiler", LineProfileWrapper)


_register_profilers()


p = Profiler()
atexit.register(p.finalize)


# convienence wrappers for profiling and annotation decorators:
annotate = p.annotate
profile = p.__call__
