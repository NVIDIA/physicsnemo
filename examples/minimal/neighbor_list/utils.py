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


import time
import torch


# Track min time and max memory
class Meter:
    def __init__(self, name: str):
        self.name = name
        self.min_time = float("inf")
        self.max_memory = 0
        self.max_allocated_memory = 0

    def __enter__(self):
        self.start = time.time()
        self.start_memory = torch.cuda.memory_allocated()

    def __exit__(self, *args):
        elapsed_time = time.time() - self.start
        if elapsed_time < self.min_time:
            self.min_time = elapsed_time
        memory = torch.cuda.memory_allocated() - self.start_memory
        if memory > self.max_memory:
            self.max_memory = memory
        max_allocated_memory = torch.cuda.max_memory_allocated()
        if max_allocated_memory > self.max_allocated_memory:
            self.max_allocated_memory = max_allocated_memory
