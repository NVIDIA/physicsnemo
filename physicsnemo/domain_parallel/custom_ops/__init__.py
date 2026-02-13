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

from physicsnemo.core.version_check import check_version_spec

# Prevent importing this module if the minimum version of pytorch is not met.
ST_AVAILABLE = check_version_spec("torch", "2.6.0a0", hard_fail=False)

if ST_AVAILABLE:
    from ._reductions import mean_wrapper, sum_wrapper
    from ._tensor_ops import unbind_rules
