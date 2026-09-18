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

"""A (hopefully temporary) fix to use Apex FusedLayerNorm with ShardTensor."""

try:
    import apex.normalization

    IS_APEX_AVAILABLE = True
except ImportError:
    IS_APEX_AVAILABLE = False
import torch

from physicsnemo.domain_parallel.shard_tensor import ShardTensor


def sharded_fused_layernorm_forward(original_forward):
    """Wrapper for Apex FusedLayerNorm that handles ShardTensor inputs."""

    def forward_wrapper(self, input):
        # Check if input is sharded
        if isinstance(input, ShardTensor):
            # Get sharding info
            input_spec = input._spec
            mesh = input_spec.mesh
            placements = input_spec.placements

            # LayerNorm is local operation when normalized dims != sharded dims
            # Extract local tensor and compute
            local_input = input.to_local()
            local_output = original_forward(self, local_input)

            # Wrap back into ShardTensor with same placement
            output = ShardTensor.from_local(
                local_output,
                device_mesh=mesh,
                placements=placements,
                sharding_shapes=input_spec.sharding_shapes(),
            )
            return output
        else:
            # Standard path for non-sharded tensors
            return original_forward(self, input)

    return forward_wrapper


# Monkey-patch the Apex FusedLayerNorm forward method
def register_apex_layernorm_handler():
    """Register the sharded handler for Apex FusedLayerNorm."""
    original_forward = apex.normalization.FusedLayerNorm.forward
    apex.normalization.FusedLayerNorm.forward = sharded_fused_layernorm_forward(
        original_forward
    )


if IS_APEX_AVAILABLE:
    register_apex_layernorm_handler()
