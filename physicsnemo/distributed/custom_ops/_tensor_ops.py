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

check_module_requirements("physicsnemo.distributed.shard_tensor")


from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta  # noqa: E402
from torch.distributed.tensor._op_schema import (  # noqa: E402
    OpSchema,
    OutputSharding,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.utils import register_prop_rule  # noqa: E402
from torch.distributed.tensor.placement_types import (  # noqa: E402
    Partial,
    Replicate,
    Shard,
)

# noqa: E402
from physicsnemo.distributed._shard_tensor_spec import (  # noqa: E402
    _stride_from_contiguous_shape_C_style,
)

aten = torch.ops.aten


@register_prop_rule(aten.unbind.int, schema_info=RuntimeSchemaInfo(1))
def unbind_rules(op_schema: OpSchema) -> OutputSharding:
    """
    Need to add rules for unbinding for stormcast and attention in general
    """

    # We need to get the dimension of the slice.  0 is default.

    args_schema = op_schema.args_schema

    if len(args_schema) > 1:
        dim = args_schema[-1]
    else:
        dim = 0

    # if the chunking dimension is along a dimension that is sharded, we have to handle that.
    # If it's along an unsharded dimension, there is nearly nothing to do.

    input_spec = args_schema[0]

    input_placements = input_spec.placements

    shards = [s for s in input_placements if isinstance(s, Shard)]

    if dim in [i.dim for i in shards]:
        raise Exception("No implementation for unbinding along sharding axis yet.")

    else:
        # We are reducing tensor rank and returning one sharding per tensor:
        original_shape = list(input_spec.shape)
        unbind_dim_shape = original_shape.pop(dim)

        output_stride = _stride_from_contiguous_shape_C_style(original_shape)

        # Need to create a new global meta:
        new_meta = TensorMeta(
            torch.Size(tuple(original_shape)),
            stride=output_stride,
            dtype=input_spec.tensor_meta.dtype,
        )

        # The placements get adjusted too
        new_placements = []
        for p in input_spec.placements:
            if isinstance(p, Replicate):
                new_placements.append(p)
            elif isinstance(p, Shard):
                if p.dim > dim:
                    new_placements.append(Shard(p.dim - 1))
                else:
                    new_placements.append(p)
            elif isinstance(p, Partial):
                raise Exception("Partial placement not supported yet for unbind")

        output_spec_list = [
            DTensorSpec(
                mesh=input_spec.mesh,
                placements=tuple(new_placements),
                tensor_meta=new_meta,
            )
            for _ in range(unbind_dim_shape)
        ]
        return OutputSharding(output_spec_list)
