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

"""Optional geometry descriptors and their ordered feature schema.

Import this module explicitly when using mesh descriptors. Generic OOD scorers
have no dependency on it.
"""

from .feature_extraction import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    extract_features,
    feature_hash,
)
from .feature_schema import FeatureSchema
from .mesh_validation import validate_mesh

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "extract_features",
    "feature_hash",
    "FeatureSchema",
    "validate_mesh",
]
