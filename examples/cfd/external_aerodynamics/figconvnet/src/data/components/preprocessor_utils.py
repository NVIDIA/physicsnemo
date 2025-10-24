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

from typing import Any, Mapping, Optional, List, Callable


class ComposePreprocessors:
    """
    Compose multiple preprocessors into a single callable object
    """

    def __init__(self, preprocessors: Optional[List[Callable]] = None):
        self.preprocessors = preprocessors

    def __call__(self, sample: Mapping[str, Any]):
        if self.preprocessors is not None:
            for preprocessor in self.preprocessors:
                sample = preprocessor(sample)
        return sample


class UnitGaussianNormalizer:
    """Unit Gaussian Normalizer."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


class UniformNormalizer:
    """
    Normalize to [0, 1]
    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def encode(self, x):
        return (x - self.min_val) / (self.max_val - self.min_val)

    def decode(self, x):
        return x * (self.max_val - self.min_val) + self.min_val
