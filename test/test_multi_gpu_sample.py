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
import torch


@pytest.mark.multigpu_dynamic
def test_multi_gpu():
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 1, "Not enough GPUs available for test"

    for i in range(num_gpus):
        with torch.cuda.device(i):
            tensor = torch.tensor([1.0, 2.0, 3.0], device=f"cuda:{i}")
            assert tensor.sum() == 6.0


if __name__ == "__main__":
    pytest.main([__file__])
