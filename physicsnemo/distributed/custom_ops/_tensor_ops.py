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
from torch.distributed.tensor._ops.utils import (  # noqa: E402
    register_prop_rule,
)
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


@register_prop_rule(aten.select.int, schema_info=RuntimeSchemaInfo(1))
def select_rules(op_schema: OpSchema) -> OutputSharding:
    """
    Need to add rules for unbinding for stormcast and attention in general

    select function signature is (self, int dim, symint index)

    select and select_backward just recently went into pytorch.  These functions will disappear
    when that version of pytorch is release.
    """

    args_schema = op_schema.args_schema
    # Select mandates these:
    dim = args_schema[-2]
    # We don't really need the actual index to propagate.  But here's how
    # we'd access it:
    # index = args_schema[-1]

    # if the chunking dimension is along a dimension that is sharded, we have to handle that.
    # If it's along an unsharded dimension, there is nearly nothing to do.

    input_spec = args_schema[0]

    input_placements = input_spec.placements

    shards = [s for s in input_placements if isinstance(s, Shard)]

    # We are reducing tensor rank and returning one sharding per tensor:
    original_shape = list(input_spec.shape)

    output_stride = _stride_from_contiguous_shape_C_style(original_shape)

    if dim in [i.dim for i in shards]:
        raise Exception("No implementation for unbinding along sharding axis yet.")

    else:

        # We are reducing tensor rank:
        original_shape.pop(dim)

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
                raise Exception("Partial placement not supported yet for select")

        output_spec = DTensorSpec(
            mesh=input_spec.mesh,
            placements=tuple(new_placements),
            tensor_meta=new_meta,
        )
        return OutputSharding(output_spec)


@register_prop_rule(aten.select_backward.default, schema_info=RuntimeSchemaInfo(1))
def select_backward_rules(op_schema: OpSchema) -> OutputSharding:
    """
    Need to add rules for unbinding for stormcast and attention in general

    select_backward function signature is:
    Declaration: aten::select_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt index) -> Tensor

    For the backwards pass, the gradients must be shaped and sharded like the input.

    select and select_backward just recently went into pytorch.  These functions will disappear
    when that version of pytorch is release.
    """

    # Args_schema is a tuple[object]... describing the function args.
    args_schema = op_schema.args_schema

    # print(f"Op schema is {op_schema}")
    # print(f"Op is {op_schema.op}")
    # print(f"arg_schema is {args_schema}")

    # print(f"len of op_schema: {len(args_schema)}")
    # print(f"Len of arg_schema: {len(args_schema)}")
    input_spec, input_sizes, dim, index = args_schema

    # print(f"input_spec: {input_spec}")
    # print(f"input_sizes: {input_sizes}")
    # print(f"dim: {dim}")
    # print(f"index: {index}")

    # if the chunking dimension is along a dimension that is sharded, we have to handle that.
    # If it's along an unsharded dimension, there is nearly nothing to do.

    input_placements = input_spec.placements

    shards = [s for s in input_placements if isinstance(s, Shard)]

    output_stride = _stride_from_contiguous_shape_C_style(input_sizes)

    if dim in [i.dim for i in shards]:
        raise Exception(
            "No implementation for select_backwards along sharding axis yet."
        )
    else:
        # Need to create a new global meta:
        new_meta = TensorMeta(
            torch.Size(tuple(input_sizes)),
            stride=output_stride,
            dtype=input_spec.tensor_meta.dtype,
        )

        new_placements = input_placements
        # The placements get adjusted too
        new_placements = []
        for p in input_spec.placements:
            if isinstance(p, Replicate):
                new_placements.append(p)
            elif isinstance(p, Shard):
                if p.dim > dim:
                    new_placements.append(Shard(p.dim + 1))
                else:
                    new_placements.append(p)
            elif isinstance(p, Partial):
                raise Exception("Partial placement not supported yet for select")

        output_spec = DTensorSpec(
            mesh=input_spec.mesh,
            placements=tuple(new_placements),
            tensor_meta=new_meta,
        )

        # print(f"Output meta is {new_meta}")
        # print(f"Output placements are {new_placements}")
        # print(f"Output spec is {output_spec}")

        return OutputSharding(output_spec)


# @register_op_strategy(aten.scatter_add.default, schema_info=RuntimeSchemaInfo(1))
# def scatter_add_strategy(mesh, op_schema: OpSchema) -> OpStrategy:
#     """
#     Strategy for scatter_add operation: scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor

#     The output sharding follows these rules:
#     - If self is sharded on the scatter dimension, we need to ensure both index and src match this sharding pattern
#     - If self is sharded on other dimensions, we preserve those sharding patterns
#     """
#     # Extract schema components
#     args_schema = op_schema.args_schema
#     self_strategy, dim, index_strategy, src_strategy = args_schema

#     assert isinstance(self_strategy, OpStrategy)
#     assert isinstance(dim, int)
#     assert isinstance(index_strategy, OpStrategy)
#     assert isinstance(src_strategy, OpStrategy)

#     # Normalize dimension
#     input_ndim = self_strategy.ndim
#     scatter_dim = normalize_dim(dim, input_ndim)

#     output_strategies: list[PlacementStrategy] = []

#     # Iterate through all possible strategies for self tensor
#     for self_placement_strategy in self_strategy.strategies:
#         self_spec = self_placement_strategy.output_spec

#         # For scatter_add, the output has the same sharding as self
#         output_spec = self_spec

#         # Check if self is sharded on the scatter dimension
#         self_sharded_on_scatter_dim = is_tensor_dim_sharded(self_spec, dim=scatter_dim)

#         # Create appropriate input specs
#         # For scatter_add, the primary constraint is that if self is sharded on the scatter dimension,
#         # then index and src need to be sharded in the same way
#         input_specs = [self_spec]

#         # Add the strategy
#         output_strategies.append(
#             PlacementStrategy(
#                 output_specs=output_spec,
#                 input_specs=tuple(input_specs),
#             )
#         )

#     return OpStrategy(output_strategies)
