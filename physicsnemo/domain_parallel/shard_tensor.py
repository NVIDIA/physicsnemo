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

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Callable, Sequence, cast
from warnings import warn

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, _mesh_resources
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import (
    TensorMeta,
)
from torch.distributed.tensor.placement_types import (
    Placement,
    Replicate,
    Shard,
)

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel._shard_redistribute import (
    ShardRedistribute,
)
from physicsnemo.domain_parallel._shard_tensor_spec import (
    ShardTensorSpec,
    _infer_shard_tensor_spec_from_local_chunks,
    _stride_from_contiguous_shape_C_style,
)
from physicsnemo.utils.profiling import annotate, profile

aten = torch.ops.aten


def _shard_tensor_to_dtensor(st: "ShardTensor") -> DTensor:
    r"""Convert a ShardTensor to a plain DTensor for dispatch.

    Creates a DTensor with the same internal state as the ShardTensor,
    which allows DTensor's dispatch to handle it correctly.

    Parameters
    ----------
    st : ShardTensor
        The ShardTensor to convert.

    Returns
    -------
    DTensor
        A DTensor sharing the same ``_local_tensor`` and ``_spec``.
    """
    dtensor = torch.Tensor._make_wrapper_subclass(
        DTensor,
        st._spec.tensor_meta.shape,
        strides=st._spec.tensor_meta.stride,
        dtype=st.dtype,
        device=st.device,
        layout=st.layout,
        requires_grad=st.requires_grad,
    )
    dtensor._local_tensor = st._local_tensor
    dtensor._spec = st._spec
    return dtensor


def _convert_args_to_dtensor(arg: object) -> object:
    r"""Recursively convert ShardTensors in args to DTensors.

    Parameters
    ----------
    arg : object
        A single argument that may be a ShardTensor, an iterable of
        arguments (e.g. list, tuple), a mapping (e.g. dict) whose
        values are converted, or any other value.

    Returns
    -------
    object
        The argument with any ShardTensors replaced by DTensors.
    """
    # ShardTensor is defined later in this module; the isinstance check
    # is safe because this function is only called at runtime.
    if isinstance(arg, ShardTensor):
        return _shard_tensor_to_dtensor(arg)
    elif isinstance(arg, Mapping):
        return type(arg)({k: _convert_args_to_dtensor(v) for k, v in arg.items()})
    elif isinstance(arg, Iterable) and not isinstance(arg, (str, bytes)):
        converted = [_convert_args_to_dtensor(a) for a in arg]
        return type(arg)(converted)
    return arg


class _ToTorchTensor(torch.autograd.Function):
    r"""Autograd function to convert a ShardTensor to a regular PyTorch tensor.

    This class handles the conversion from ShardTensor to ``torch.Tensor`` in both
    forward and backward passes, maintaining proper gradient flow. Slices the
    ShardTensor to the local component only on the current rank.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: "ShardTensor",
        grad_placements: Sequence[Placement] | None = None,
    ) -> torch.Tensor:
        r"""Convert ShardTensor to torch.Tensor in forward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context for saving tensors/variables for backward.
        input : ShardTensor
            ShardTensor to convert.
        grad_placements : Sequence[Placement], optional
            Sequence of placements to use for gradients.

        Returns
        -------
        torch.Tensor
            Local tensor representation of the ShardTensor.
        """
        ctx.shard_tensor_spec = input._spec
        ctx.grad_placements = grad_placements
        local_tensor = input._local_tensor

        # JUST LIKE DTENSOR:
        # We need to return a fresh Tensor object there as autograd metadata
        # will be inplaced into it. So we don't want to pollute the Tensor
        # object stored in the _local_tensor of this ShardTensor.
        return local_tensor.view_as(local_tensor)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple["ShardTensor", None]:
        r"""Convert gradient torch.Tensor back to ShardTensor in backward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved tensors/variables from forward.
        grad_output : torch.Tensor
            Gradient tensor to convert back to ShardTensor.

        Returns
        -------
        Tuple[ShardTensor, None]
            Tuple containing the ShardTensor gradient and None for
            grad_placements gradient (not differentiable).
        """
        shard_tensor_spec = ctx.shard_tensor_spec
        mesh = shard_tensor_spec.mesh
        if ctx.grad_placements is not None:
            if ctx.grad_placements != shard_tensor_spec.placements:
                grad_placements = ctx.grad_placements
                grad_sharding_shapes = "infer"
            else:
                # If the placements are the same as the input placements,
                # we reuse the sharding sizes from the input placements.
                grad_placements = ctx.grad_placements
                grad_sharding_shapes = shard_tensor_spec._sharding_shapes
        else:
            grad_placements = shard_tensor_spec.placements
            grad_sharding_shapes = shard_tensor_spec._sharding_shapes
        if grad_sharding_shapes is None:
            grad_sharding_shapes = "infer"
        # Generate a spec based on grad outputs and the expected placements:
        grad_tensor_spec = _infer_shard_tensor_spec_from_local_chunks(
            grad_output, mesh, grad_placements, grad_sharding_shapes
        )

        return (
            ShardTensor(
                grad_output, grad_tensor_spec, requires_grad=grad_output.requires_grad
            ),
            None,
        )


class _FromTorchTensor(torch.autograd.Function):
    r"""Autograd function for converting a torch.Tensor to a ShardTensor.

    This class handles the forward and backward passes for converting between
    ``torch.Tensor`` and ShardTensor types, maintaining gradient information.

    Global shape information is inferred using collective communication on
    the specified device mesh.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        local_input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        sharding_shapes: str | dict[int, list[tuple[int, ...]]] = "chunk",
    ) -> "ShardTensor":
        r"""Convert a local torch.Tensor to a ShardTensor in forward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context for saving tensors/variables for backward.
        local_input : torch.Tensor
            Local tensor to convert to ShardTensor.
        device_mesh : DeviceMesh
            Device mesh specifying process groups.
        placements : Tuple[Placement, ...]
            Tuple of placement rules for sharding.
        sharding_shapes : Union[str, Dict[int, List[Tuple[int, ...]]]], default="chunk"
            Controls how shard tensor spec is generated:

            - ``"chunk"``: Use ``torch.chunk`` shapes to infer shapes from
              global shape (no communication).
            - ``"infer"``: Use collective communication to infer shapes from
              mesh neighbors.
            - Manual dict mapping mesh dim to list of shard shapes: Use
              provided shapes. Must pass on each rank.

        Returns
        -------
        ShardTensor
            ShardTensor constructed from the local input tensor.
        """
        ctx.previous_placement = placements
        ctx.previous_mesh = device_mesh

        # This function is simpler than the corresponding DTensor implementation on the surface
        # because under the hood, we have some logic here to infer the sharding shapes.
        shard_tensor_spec = _infer_shard_tensor_spec_from_local_chunks(
            local_input, device_mesh, placements, sharding_shapes
        )

        shard_tensor = ShardTensor(
            local_input,
            shard_tensor_spec,
            requires_grad=local_input.requires_grad,
        )

        return shard_tensor

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: "ShardTensor",
    ) -> tuple[torch.Tensor, None, None, None]:
        r"""Convert gradient ShardTensor back to torch.Tensor in backward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved tensors/variables from forward.
        grad_output : ShardTensor
            Gradient ShardTensor to convert back to torch.Tensor.

        Returns
        -------
        Tuple[torch.Tensor, None, None, None]
            Tuple containing the local tensor gradient, and None for
            device_mesh, placements, and sharding_shapes gradients
            (not differentiable).

        Raises
        ------
        RuntimeError
            If gradient tensor has different placement than original and
            the original placement contains partial placements.
        """
        previous_placement = ctx.previous_placement
        if grad_output.placements != previous_placement:
            # Automatically redistribute to the previous placement as long as it's not a partial.
            if not any(p.is_partial() for p in previous_placement):
                grad_output = grad_output.redistribute(
                    grad_output._spec.mesh, previous_placement
                )
            else:
                raise RuntimeError(
                    "Resharding gradients with partial placements not implemented"
                )

        return grad_output.to_local(), None, None, None


class _PromoteDTensorToShardTensor(torch.autograd.Function):
    r"""Autograd function to promote a DTensor to a ShardTensor while preserving ``grad_fn``.

    When DTensor's ``__torch_function__`` returns a non-leaf DTensor (one that
    has a ``grad_fn``), creating a new ShardTensor via ``_make_wrapper_subclass``
    always produces a leaf — disconnecting it from the autograd graph.

    This function bridges that gap: the forward creates the ShardTensor wrapper,
    and ``apply`` attaches a ``grad_fn`` that connects it back to the original
    DTensor's graph. The backward simply passes gradients through unchanged.

    This is only used at the ``__torch_function__`` level where the DTensor
    result already carries autograd state. At the ``__torch_dispatch__`` level,
    promotion is safe without this because autograd wraps the result afterwards.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        dtensor: DTensor,
        spec: "ShardTensorSpec",
    ) -> "ShardTensor":
        r"""Create a ShardTensor from a DTensor, preserving autograd via ``apply``.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context (unused — no state needed for backward).
        dtensor : DTensor
            The DTensor to promote.
        spec : ShardTensorSpec
            The ShardTensorSpec to use for the new ShardTensor.

        Returns
        -------
        ShardTensor
            A new ShardTensor wrapping the same local data.
        """
        return ShardTensor.__new__(
            ShardTensor,
            local_tensor=dtensor._local_tensor,
            spec=spec,
            requires_grad=False,  # autograd.Function.apply handles this
        )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: "ShardTensor",
    ) -> tuple[DTensor, None]:
        r"""Pass gradient through unchanged.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context (unused).
        grad_output : ShardTensor
            Gradient with respect to the ShardTensor output.

        Returns
        -------
        Tuple[DTensor, None]
            The gradient for the DTensor input, and ``None`` for the spec.
        """
        return grad_output, None


class ShardTensor(DTensor):
    r"""A distributed tensor class with support for uneven data sharding.

    Similar to PyTorch's native ``DTensor`` but with more flexibility for
    uneven data sharding. Leverages a very similar API to ``DTensor``
    (identical where possible) but deliberately tweaks routines to avoid
    implicit assumptions about tensor sharding.

    The key differences from ``DTensor`` are:

    - Supports uneven sharding where different ranks can have different
      local tensor sizes
    - Tracks and propagates shard size information across operations
    - Handles redistribution of unevenly sharded tensors
    - Provides custom collective operations optimized for uneven sharding

    Like ``DTensor``, operations are dispatched through PyTorch's dispatcher
    system. Most operations work by:

    1. Converting inputs to local tensors
    2. Performing the operation locally
    3. Constructing a new ShardTensor with appropriate sharding spec
    4. Handling any needed communication between ranks

    The class provides methods for:

    - Converting to/from local tensors
    - Redistributing between different sharding schemes
    - Performing collective operations like all_gather and reduce_scatter
    - Basic tensor operations that maintain sharding information

    Attributes
    ----------
    _local_tensor : torch.Tensor
        The local tensor data on this rank.
    _spec : ShardTensorSpec
        The specification defining sharding scheme and metadata.
    """

    _local_tensor: torch.Tensor
    _spec: ShardTensorSpec
    __slots__ = ["_local_tensor", "_spec"]

    # For torch.ops.aten operators (low-level dispatch)
    _dispatch_registry: dict[torch._ops.OpOverload, Callable] = {}
    # Fallback by op name (e.g. "aten.neg.default") when the OpOverload
    # passed to __torch_dispatch__ is not the same object as the one used to register.
    _dispatch_registry_by_name: dict[str, Callable] = {}

    # For Python-level functions (torch.mean, tensor.mean, etc.)
    _function_registry: dict[Callable, Callable] = {}

    # For custom functions registered with PyTorch,
    # it is sometimes necessary to match by name.
    # For instance, if you declare an op with
    #
    # @torch.library.custom_op(
    #    "module::function_name", mutates_args=()
    # )
    # def function_external_to_torch(
    #
    # Then, you likely want to register the handler with
    #
    # ShardTensor.register_named_function_handler("module.function_name.default", handler)
    _named_function_registry: dict[str, Callable] = {}

    # Upon construction of any ShardTensor objects, this will be set to true.
    # Wrappers are triggered dynamically, so the wrapping will be pass-through
    # exclusively until true.
    _enable_shard_patches: bool = False

    @classmethod
    def patches_enabled(cls) -> bool:
        r"""Check whether patches are enabled for this class.

        Returns
        -------
        bool
            ``True`` if shard patches are enabled, ``False`` otherwise.
            Default is ``False`` until a ShardTensor is constructed.
        """
        return cls._enable_shard_patches

    @classmethod
    def register_dispatch_handler(
        cls, op: torch._ops.OpOverload, handler: Callable
    ) -> None:
        r"""Register a handler for a specific PyTorch operator in the dispatch system.

        Parameters
        ----------
        op : torch._ops.OpOverload
            The PyTorch operator to register a handler for.
        handler : Callable
            The handler function to call when the operator is invoked.
        """
        cls._dispatch_registry[op] = handler
        cls._dispatch_registry_by_name[str(op)] = handler

    @classmethod
    def register_function_handler(cls, func: Callable, handler: Callable) -> None:
        r"""Register a handler for a Python-level function or method.

        Parameters
        ----------
        func : Callable
            The Python function to register a handler for.
        handler : Callable
            The handler function to call when the function is invoked.
        """
        cls._function_registry[func] = handler

    @classmethod
    def register_named_function_handler(cls, func_name: str, handler: Callable) -> None:
        r"""Register a named function registered via ``torch.library.custom_op``.

        Parameters
        ----------
        func_name : str
            The string name of the custom op (e.g., ``"module.function_name.default"``).
        handler : Callable
            The handler function to call when the function is invoked.
        """
        cls._named_function_registry[func_name] = handler

    @staticmethod
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: ShardTensorSpec,
        *,
        requires_grad: bool,
    ) -> "ShardTensor":
        r"""Construct a new ShardTensor from a local tensor and specification.

        Note that unlike ``DTensor``, ShardTensor will automatically collect
        the shard size information from all participating devices. This enables
        uneven and dynamic sharding.

        Parameters
        ----------
        local_tensor : torch.Tensor
            Local tensor to use as the data.
        spec : ShardTensorSpec
            ShardTensorSpec defining the sharding scheme.
        requires_grad : bool
            Whether the tensor requires gradients.

        Returns
        -------
        ShardTensor
            A new ShardTensor instance.

        Note
        ----
        This implementation is heavily derived from ``torch.distributed.tensor.DTensor``.
        """
        if local_tensor.requires_grad and not requires_grad:
            warn(
                "To construct a new ShardTensor from torch.Tensor, "
                "it's recommended to use local_tensor.detach() and "
                "make requires_grad consistent."
            )

        if spec.tensor_meta is None:
            raise ValueError("TensorMeta should not be None!")

        # Check the sharding information is known:
        ret = torch.Tensor._make_wrapper_subclass(
            cls,
            spec.tensor_meta.shape,
            strides=spec.tensor_meta.stride,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=requires_grad,
        )

        ret._spec = spec
        ret._local_tensor = local_tensor

        cls._enable_shard_patches = True

        return ret

    def __repr__(self) -> str:
        return f"ShardTensor(local_tensor={self._local_tensor}, device_mesh={self._spec.mesh}, placements={self._spec.placements})"

    @classmethod
    def from_dtensor(cls, dtensor: DTensor) -> "ShardTensor":
        r"""Convert a DTensor to a ShardTensor.

        Assumes the DTensor is properly constructed. Since DTensor is locked
        to sharding a tensor according to chunk format, the sharding sizes
        can be inferred with no communication.

        If the DTensor is a non-leaf (has a ``grad_fn``), the autograd graph
        is preserved via :class:`_PromoteDTensorToShardTensor`.

        Parameters
        ----------
        dtensor : DTensor
            DTensor to convert.

        Returns
        -------
        ShardTensor
            Equivalent ShardTensor with the same local tensor and inferred spec.
        """
        return cls._maybe_promote_dtensor(dtensor, ())

    @staticmethod
    def _maybe_promote_dtensor(dtensor, input_args):
        r"""Promote a single DTensor back to ShardTensor if it matches input criteria.

        If ``dtensor`` is already a ShardTensor, it is returned as-is. Otherwise,
        determines a ``ShardTensorSpec`` (reusing an input's spec when possible,
        otherwise inferring one) and creates a new ShardTensor.

        When the DTensor is a non-leaf (has a ``grad_fn``), the promotion goes
        through :class:`_PromoteDTensorToShardTensor` so that the autograd graph
        is preserved. For leaf DTensors, direct construction is used since there
        is no graph to preserve.

        Parameters
        ----------
        dtensor : DTensor
            The DTensor result to promote.
        input_args : tuple
            Original input arguments to search for matching ShardTensors.

        Returns
        -------
        ShardTensor
            Promoted ShardTensor (or the original if already a ShardTensor).
        """
        if isinstance(dtensor, ShardTensor):
            return dtensor

        # Determine the ShardTensorSpec — reuse an input's spec when the
        # tensor_meta and placements match (avoids communication).
        spec = None
        for arg in input_args:
            if (
                isinstance(arg, ShardTensor)
                and dtensor._spec.tensor_meta == arg._spec.tensor_meta
                and dtensor._spec.placements == arg._spec.placements
            ):
                spec = arg._spec
                break

        if spec is None:
            # Infer from DTensor (no communication for chunk-based sharding).
            spec = _infer_shard_tensor_spec_from_local_chunks(
                dtensor._local_tensor,
                dtensor._spec.mesh,
                dtensor._spec.placements,
                sharding_shapes="chunk",
                global_shape=dtensor.shape,
            )

        # Non-leaf DTensors carry a grad_fn from the operation that produced
        # them.  Creating a new ShardTensor via _make_wrapper_subclass would
        # discard that grad_fn (producing a leaf).  Go through the autograd
        # function so that apply() connects the new ShardTensor back to the
        # original graph.
        if dtensor.grad_fn is not None:
            return _PromoteDTensorToShardTensor.apply(dtensor, spec)

        # Leaf DTensors (parameters, buffers, detached tensors) can be
        # constructed directly — there is no autograd graph to preserve.
        return ShardTensor.__new__(
            ShardTensor,
            local_tensor=dtensor._local_tensor,
            spec=spec,
            requires_grad=dtensor.requires_grad,
        )

    @staticmethod
    def _promote_dtensor_results(result, input_args):
        r"""Promote DTensor(s) in a dispatch/function result back to ShardTensor.

        Handles four cases:

        1. Single DTensor — promoted via :meth:`_maybe_promote_dtensor`.
        2. Mapping (e.g. dict) — each value is promoted if it is a DTensor.
        3. Iterable of results — each DTensor element is promoted individually.
        4. Anything else — returned as-is.

        Parameters
        ----------
        result : object
            The result returned by DTensor dispatch or ``__torch_function__``.
        input_args : tuple
            Original input arguments used for matching specs.

        Returns
        -------
        object
            The result with any DTensors promoted to ShardTensors.
        """
        if isinstance(result, DTensor):
            return ShardTensor._maybe_promote_dtensor(result, input_args)

        if isinstance(result, Mapping):
            return type(result)(
                {
                    k: ShardTensor._maybe_promote_dtensor(v, input_args)
                    if isinstance(v, DTensor)
                    else v
                    for k, v in result.items()
                }
            )

        # Exclude str/bytes so we don't iterate over characters.
        if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
            return type(result)(
                ShardTensor._maybe_promote_dtensor(d, input_args)
                if isinstance(d, DTensor)
                else d
                for d in result
            )

        return result

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        with annotate(f"__torch_function___{func.__name__}"):
            # Check for overrides:
            if func in cls._function_registry and cls._enable_shard_patches:
                res = cls._function_registry[func](func, types, args, kwargs)
                return res
            elif (
                str(func) in cls._named_function_registry and cls._enable_shard_patches
            ):
                res = cls._named_function_registry[str(func)](func, types, args, kwargs)
                return res
            # Fall back to the default behavior, but promote any DTensor
            # results back to ShardTensor (matching dispatch behavior):
            result = super().__torch_function__(func, types, args, kwargs)
            return cls._promote_dtensor_results(result, args)

    @classmethod
    @torch._disable_dynamo
    @profile
    def __torch_dispatch__(
        cls,
        func: torch._ops.OpOverload,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> "ShardTensor" | Iterable["ShardTensor"] | object:
        with annotate(f"__torch_dispatch___{func.__name__}"):
            # Leverage DTensor Dispatch as much as possible, but, enable
            # the ability to operate on this output in the future:
            handler = cls._dispatch_registry.get(func)
            if handler is None:
                handler = cls._dispatch_registry_by_name.get(str(func))
            if handler is not None:
                res = handler(*args, **kwargs)
                return res

            # We assume that if we reach this point, the operator has not been
            # intercepted by a wrapper or in the registry.  So the DTensor
            # default behavior is likely to be correct.

            # Convert ShardTensors to DTensors so DTensor's dispatcher
            # receives the types it expects.
            converted_args = tuple(_convert_args_to_dtensor(arg) for arg in args)
            converted_kwargs = {
                k: _convert_args_to_dtensor(v) for k, v in (kwargs or {}).items()
            }

            dispatch_res = DTensor._op_dispatcher.dispatch(
                func, converted_args, converted_kwargs
            )

            # Promote any DTensor results back to ShardTensor.
            return cls._promote_dtensor_results(dispatch_res, args)

    @staticmethod
    def from_local(
        local_tensor: torch.Tensor,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        sharding_shapes: str | dict[int, list[tuple[int, ...]]] = "infer",
    ) -> "ShardTensor":
        r"""Generate a new ShardTensor from local torch tensors.

        Uses device mesh and placements to infer global tensor properties.
        No restriction is made on forcing tensors to have equal shapes locally.
        Instead, the requirement is that tensor shapes could be concatenated
        into a single tensor according to the placements.

        Parameters
        ----------
        local_tensor : torch.Tensor
            Local chunk of tensor. All participating tensors must be of the
            same rank and concatenatable across the mesh dimensions.
        device_mesh : Optional[DeviceMesh], optional
            Target device mesh. If not specified, will use the current mesh.
        placements : Optional[Sequence[Placement]], optional
            Target placements. Must have same number of elements as
            ``device_mesh.ndim``.
        sharding_shapes : Union[str, Dict[int, List[Tuple[int, ...]]]], default="infer"
            Controls how shard tensor spec is generated:

            - ``"chunk"``: Use ``torch.chunk`` shapes to infer shapes from
              global shape (no communication).
            - ``"infer"``: Use collective communication to infer shapes from
              mesh neighbors.
            - Manual dict mapping mesh dim to list of shard shapes: Use
              provided shapes. Must pass on each rank.

        Returns
        -------
        ShardTensor
            A new ShardTensor instance.
        """

        # This implementation follows the pytorch DTensor Implementation Closely.
        device_mesh = device_mesh or _mesh_resources.get_current_mesh()
        device_type = device_mesh.device_type

        # convert the local tensor to desired device base on device mesh's device_type
        if device_type != local_tensor.device.type and not local_tensor.is_meta:
            local_tensor = local_tensor.to(device_type)

        # set default placements to replicated if not specified
        if placements is None:
            placements = [Replicate() for _ in range(device_mesh.ndim)]
        else:
            placements = list(placements)
            for idx, placement in enumerate(placements):
                # normalize shard dim to be positive
                if placement.is_shard():
                    placement = cast(Shard, placement)
                    if placement.dim < 0:
                        placements[idx] = Shard(placement.dim + local_tensor.ndim)

        # `from_local` is differentiable, and the gradient of the dist tensor this function
        # created should flow back the gradients to the local_tensor, so we call an autograd
        # function to construct the dist tensor instead.
        return _FromTorchTensor.apply(  # pyre-ignore[16]: autograd func
            local_tensor,
            device_mesh,
            tuple(placements),
            sharding_shapes,
        )

    def offsets(self, mesh_dim: int | None = None) -> list[int] | int:
        r"""Get offsets of shards along a mesh dimension.

        Parameters
        ----------
        mesh_dim : Optional[int], optional
            Mesh dimension to get offsets for. If ``None``, returns all offsets.

        Returns
        -------
        Union[List[int], int]
            List of offsets for shards along all dimensions, or single offset
            if ``mesh_dim`` is specified.
        """
        return self._spec.offsets(mesh_dim)

    def redistribute(
        self,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        *,
        async_op: bool = False,
    ) -> "ShardTensor":
        r"""Redistribute tensor across device mesh with new placement scheme.

        Like ``DTensor.redistribute`` but uses custom layer for shard
        redistribution that supports uneven sharding.

        Parameters
        ----------
        device_mesh : Optional[DeviceMesh], optional
            Target device mesh. Uses current mesh if ``None``.
        placements : Optional[Sequence[Placement]], optional
            Target placement scheme. Required.
        async_op : bool, default=False
            Whether to run asynchronously.

        Returns
        -------
        ShardTensor
            Redistributed ShardTensor with new placement scheme.

        Raises
        ------
        RuntimeError
            If placements is not specified or contains invalid placements
            (e.g., ``Partial`` placements or negative shard dimensions).
        """

        # if device_mesh is not specified, use the current device_mesh
        device_mesh = device_mesh or self.device_mesh
        # raise error if new placements not specified
        if placements is None:
            raise RuntimeError("placements is needed for redistribute!")

        placements = list(placements)
        for i, placement in enumerate(placements):
            if placement.is_partial():
                raise RuntimeError(
                    "Can not redistribute to Partial, redistributing to Partial is for internal use only!"
                )
            elif isinstance(placement, Shard) and placement.dim < 0:
                # normalize shard dim to be positive
                placements[i] = Shard(placement.dim + self.ndim)
        placements = tuple(placements)

        return ShardRedistribute.apply(self, device_mesh, placements, async_op)

    def to_local(
        self, *, grad_placements: Sequence[Placement] | None = None
    ) -> torch.Tensor:
        r"""Get local tensor from this ShardTensor.

        Parameters
        ----------
        grad_placements : Optional[Sequence[Placement]], optional
            Future layout of gradients. If provided, gradients will be
            constructed with this placement scheme during backward pass.

        Returns
        -------
        torch.Tensor
            Local tensor. Shape may vary between ranks for sharded tensors.
        """

        if not torch.is_grad_enabled():
            return self._local_tensor

        if grad_placements is not None:
            grad_placements = tuple(grad_placements)

        return _ToTorchTensor.apply(self, grad_placements)

    def full_tensor(
        self, *, grad_placements: Sequence[Placement] | None = None
    ) -> torch.Tensor:
        r"""Gather the full tensor from all ranks.

        Redistributes to ``Replicate`` placement on all mesh dimensions and
        returns the local tensor.

        Parameters
        ----------
        grad_placements : Optional[Sequence[Placement]], optional
            Future layout of gradients. If provided, gradients will be
            constructed with this placement scheme during backward pass.

        Returns
        -------
        torch.Tensor
            The full gathered tensor, identical on all ranks.
        """

        redist_res = self.redistribute(
            placements=[Replicate()] * self.device_mesh.ndim, async_op=False
        )
        if grad_placements is not None:
            grad_placements = tuple(grad_placements)
        return _ToTorchTensor.apply(redist_res, grad_placements)

    def backward(self, *args, **kwargs):
        r"""Perform backward pass for ShardTensor.

        Handles the redistribution of the tensor to resolve any partial
        placements before calling backward on the local tensor.

        Parameters
        ----------
        *args
            Positional arguments passed to ``torch.Tensor.backward``.
        **kwargs
            Keyword arguments passed to ``torch.Tensor.backward``.
        """

        # Before calling backward, we need to resolve any partial placements.
        new_placements = []
        needs_redistribute = False
        for placement in self._spec.placements:
            if placement.is_partial():
                new_placements.append(Replicate())
                needs_redistribute = True
            else:
                new_placements.append(placement)

        if needs_redistribute:
            self = self.redistribute(placements=new_placements)

        return self.to_local().backward(*args, **kwargs)


def scatter_tensor(
    tensor: torch.Tensor,
    global_src: int,
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
    global_shape: torch.Size | None = None,
    dtype: torch.dtype | None = None,
    requires_grad: bool = False,
) -> "ShardTensor":
    r"""Distribute a tensor from source rank across devices on the mesh.

    Takes a tensor that exists on a single source rank and distributes it
    across a device mesh according to the specified placement scheme. For
    multi-dimensional meshes, it performs a flattened scatter operation
    before constructing the sharded tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to distribute. Must exist on source rank; can be ``None``
        on other ranks.
    global_src : int
        Global rank ID of the source process.
    mesh : DeviceMesh
        Device mesh defining the process topology.
    placements : Tuple[Placement, ...]
        Tuple of placement specifications defining how to distribute the tensor.
    global_shape : Optional[torch.Size], optional
        Global shape of the tensor. If ``None``, will be broadcast from source.
    dtype : Optional[torch.dtype], optional
        Data type of the tensor. If ``None``, will be broadcast from source.
    requires_grad : bool, default=False
        Whether the resulting ShardTensor requires gradients.

    Returns
    -------
    ShardTensor
        The distributed tensor with specified placements.

    Raises
    ------
    ValueError
        If ``global_src`` is not an integer or not in the mesh.
    """
    dm = DistributedManager()

    if not isinstance(global_src, int):
        raise ValueError("Global source must be an integer rank")
    if global_src not in mesh.mesh:
        raise ValueError("Please specify a tensor source in this mesh")

    is_src = dm.rank == global_src

    # For multi-dimensional meshes, we use a flattened process group
    mesh_group = dm.get_mesh_group(mesh)

    # Broadcast tensor metadata from source
    if global_shape is None or dtype is None:
        if dm.rank == global_src:
            meta = [TensorMeta(tensor.shape, tensor.stride(), tensor.dtype)]
        else:
            meta = [None]

        dist.broadcast_object_list(meta, src=global_src, group=mesh_group)

        local_meta = meta[0]
    else:
        stride = _stride_from_contiguous_shape_C_style(global_shape)
        local_meta = TensorMeta(global_shape, stride, dtype)

    # This needs to be optimized, but I want to get the whole pipeline optimized first.
    # This only gets done when scatter_tensor is called and it should be relatively small
    # in full applications.

    # What isn't optimized?  Broadcasting the full tensor when placement is likely
    # Shard on at least one mesh dimension.  It would be more efficient to iteratively
    # scatter along Shard dimensions.  BUT, the focus is on performance of full applications
    # and this is a once-per-iteration cost.

    # Broadcast the tensor to all ranks
    if tensor is None and not is_src:
        # Tensor is allowed to be none if not on the root rank
        tensor = torch.empty(local_meta.shape, dtype=local_meta.dtype, device=dm.device)

    dist.broadcast(tensor, src=global_src, group=mesh_group)

    # Create a fully-replicated spec:
    spec = ShardTensorSpec(
        mesh=mesh,
        placements=[Replicate() for _ in range(mesh.ndim)],
        tensor_meta=local_meta,
        _sharding_shapes={},
    )

    # Make a "fully-replicated" tensor on all ranks:
    st = ShardTensor.__new__(
        ShardTensor,
        local_tensor=tensor,
        spec=spec,
        requires_grad=requires_grad,
    )

    # Redistribute the tensor to the desired placements:
    st = st.redistribute(mesh, placements, async_op=False)
    # This is an unoptimal step but is functional:
    if requires_grad:
        st = st.detach()
        st.requires_grad = True
    return st
