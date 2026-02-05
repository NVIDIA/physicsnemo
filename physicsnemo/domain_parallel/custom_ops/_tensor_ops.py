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

r"""Custom tensor operations for ShardTensor dispatch.

This module provides propagation rules for tensor operations that need
special handling when applied to ``ShardTensor`` objects. These rules
are registered with PyTorch's DTensor operation dispatch system.
"""

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OutputSharding,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.domain_parallel._shard_tensor_spec import (
    _stride_from_contiguous_shape_C_style,
)

if check_version_spec("torch", "2.10.0"):
    from torch.distributed.tensor._ops.registration import (
        register_prop_rule,
    )
else:
    from torch.distributed.tensor._ops.utils import (
        register_prop_rule,
    )

aten = torch.ops.aten


@register_prop_rule(aten.unbind.int, schema_info=RuntimeSchemaInfo(1))
def unbind_rules(op_schema: OpSchema) -> OutputSharding:
    r"""Propagation rule for ``torch.unbind`` on ShardTensor.

    Computes the output sharding specification when unbinding a sharded tensor
    along a specified dimension. The unbind operation removes one dimension
    from the tensor and returns a tuple of tensors.

    Parameters
    ----------
    op_schema : OpSchema
        The operation schema containing input specifications and arguments.
        Expected to contain:

        - ``args_schema[0]``: Input tensor specification (DTensorSpec)
        - ``args_schema[1]``: Dimension to unbind along (int), defaults to 0

    Returns
    -------
    OutputSharding
        Output sharding specification containing a list of DTensorSpec objects,
        one for each tensor in the unbind result.

    Raises
    ------
    Exception
        If attempting to unbind along a sharded dimension (not yet implemented).
        If attempting to unbind with Partial placement (not yet supported).

    Note
    ----
    This rule is needed for operations like attention in Stormcast and other
    models that unbind tensors along non-sharded dimensions.
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
