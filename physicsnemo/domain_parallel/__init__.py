# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

r"""Domain parallel utilities for distributed tensor operations.

This module provides the ``ShardTensor`` class and related utilities for
domain-parallel computation across multiple devices. Unlike PyTorch's native
``DTensor``, ``ShardTensor`` supports uneven sharding where different ranks
can have different local tensor sizes.

Key components:

- ``ShardTensor``: A distributed tensor class supporting uneven sharding
- ``ShardTensorSpec``: Specification class tracking sharding metadata
- ``scatter_tensor``: Utility to distribute tensors from a source rank

Note
----
This module requires PyTorch >= 2.6.0. Earlier versions are not supported.
"""

# Minimum PyTorch version requirement for ShardTensor:
# - 2.6.0+ is supported
# - 2.5.x and earlier are not supported

import torch

from physicsnemo.core.version_check import check_version_spec

ST_AVAILABLE = check_version_spec("torch", "2.6.0a0", hard_fail=False)


if ST_AVAILABLE:
    # In minumum versions are met, we can import the shard tensor and spec.

    from ._shard_tensor_spec import ShardTensorSpec
    from .shard_tensor import ShardTensor, scatter_tensor

    def register_custom_ops():
        # These imports will register the custom ops with the ShardTensor class.
        # It's done here to avoid an import cycle.
        from .custom_ops import (
            mean_wrapper,
            sum_wrapper,
            unbind_rules,
        )
        from .shard_utils import register_shard_wrappers

        register_shard_wrappers()

    # Protect the automatic imports by checking cuda is available.
    if torch.cuda.is_available():
        register_custom_ops()

else:
    ShardTensor = None
    ShardTensorSpec = None
    scatter_tensor = None
