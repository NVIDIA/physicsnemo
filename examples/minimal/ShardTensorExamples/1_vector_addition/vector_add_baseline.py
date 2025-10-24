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

import torch
import time

# Make a really big tensor:
N = 1_000_000_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.randn(N, device=device)
b = torch.randn(N, device=device)


def f(a, b):
    # This is a truly local operation: no communication is needed.
    return a + b


# run a couple times to warmup:
for i in range(5):
    c = f(a, b)

# Optional: Benchmark it if you like:

# Measure execution time
torch.cuda.synchronize()
start_time = time.time()
for i in range(10):
    c = f(a, b)
torch.cuda.synchronize()
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Execution time for 10 runs: {elapsed_time:.4f} seconds")
