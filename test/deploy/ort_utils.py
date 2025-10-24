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

import pytest
from packaging.version import Version

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def check_ort_version():
    required_version = Version("1.19.0")

    if ort is None:
        return pytest.mark.skipif(
            True,
            reason="Proper ONNX runtime is not installed. 'pip install onnxruntime onnxruntime_gpu'",
        )

    installed_version = Version(ort.__version__)

    if installed_version < required_version:
        return pytest.mark.skipif(
            True,
            reason="Must install ORT 1.19.0 or later. Other versions might work, but are not \
        tested. If using other versions, ensure that the fix here \
        https://github.com/microsoft/onnxruntime/pull/15662 is present. \
        If the onnxruntime-gpu wheel is not available, please build from source.",
        )

    return pytest.mark.skipif(False, reason="")
