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

from physicsnemo.utils.version_check import check_module_requirements

# Prevent importing this module if the minimum version of pytorch is not met.
try:
    check_module_requirements("physicsnemo.distributed.shard_tensor")
    SHARD_TENSOR_AVAILABLE = True

except ImportError:
    pass

if SHARD_TENSOR_AVAILABLE:
    from physicsnemo.distributed.shard_tensor import ShardTensor

    def register_shard_wrappers():
        from .attention_patches import sdpa_wrapper
        from .conv_patches import generic_conv_nd_wrapper
        from .index_ops import (
            index_select_wrapper,
            sharded_select_backward_helper,
            sharded_select_helper,
        )
        from .knn import knn_sharded_wrapper

        # Currently disabled until wrapt is removed
        # from .natten_patches import na2d_wrapper
        from .normalization_patches import group_norm_wrapper
        from .point_cloud_ops import radius_search_wrapper
        from .pooling_patches import generic_avg_pool_nd_wrapper
        from .unary_ops import unsqueeze_wrapper
        from .unpooling_patches import generic_interpolate_wrapper
