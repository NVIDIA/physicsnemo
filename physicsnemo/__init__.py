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

# from .datapipes.datapipe import Datapipe  # noqa E402
# from .datapipes.meta import DatapipeMetaData  # noqa E402
from .core.meta import ModelMetaData  # noqa E402
from .core.module import Module  # noqa E402

# This is to ensure warp is quiet at startup:
import warp as wp

wp.config.quiet = True

__version__ = "1.4.0a0"
