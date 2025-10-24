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

import time


class Timer:
    """Timer."""

    def __init__(self, name=""):
        self.name = name
        self.start = 0

    def tic(self):
        self.start = time.time()

    def toc(self):
        print(f"{self.name} takes {time.time() - self.start:.4f}s")

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print(f"{self.name} takes {time.time() - self.start:.4f}s")
        return False


class MinTimer(Timer):
    """MinTimer."""

    def __init__(self, name=""):
        super().__init__(name=name)
        self.min_time = float("inf")

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        t = time.time() - self.start
        self.min_time = min(self.min_time, t)
        return False
