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

import dataclasses
import enum
import hashlib
import threading
import warnings
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import Callable, Sequence, cast

import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import is_fake
from torch.distributed.device_mesh import DeviceMesh, _mesh_resources
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import (
    TensorMeta,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel._shard_redistribute import (
    ShardRedistribute,
    redistribute_local_shard_tensor,
)
from physicsnemo.domain_parallel._shard_tensor_spec import (
    ShardTensorSpec,
    _infer_shard_tensor_spec_from_local_chunks,
    _stride_from_contiguous_shape_C_style,
    compute_sharding_shapes_from_chunking_global_shape,
)

aten = torch.ops.aten


class TensorPromotionMode(enum.Enum):
    r"""How a plain ``torch.Tensor`` is handled when it meets a
    :class:`ShardTensor` in an intercepted op.

    Such a plain tensor is typically an unsharded model weight (all-gathered by
    FSDP2 in its pre-forward hook, or replicated under DDP).

    Attributes
    ----------
    DISABLED : TensorPromotionMode
        No promotion; plain tensors pass through to DTensor routing unchanged
        (mixing a non-scalar plain tensor with sharded data raises -- the
        historical behavior).
    WARN : TensorPromotionMode
        Promote each plain tensor to a ``Replicate`` distributed tensor on the
        accompanying distributed argument's mesh, warning on every promotion.
    SILENT : TensorPromotionMode
        Same as :attr:`WARN` but without emitting warnings.  The default.
    """

    DISABLED = "disabled"
    WARN = "warn"
    SILENT = "silent"


# ============================================================================
# Layer 1 -- Semi-private conversions (no autograd, no spec inference)
# ============================================================================


def _shard_tensor_to_dtensor(st: "ShardTensor") -> DTensor:
    r"""Convert a ShardTensor to a plain DTensor (no autograd).

    Creates a DTensor sharing the same ``_local_tensor`` and ``_spec``.
    Use for dispatch or inside backward when building a DTensor gradient.
    """
    if hasattr(torch.Tensor, "_dtensor__new__"):
        dtensor = torch.Tensor._dtensor__new__(
            DTensor, st._local_tensor, st._spec, requires_grad=st.requires_grad
        )
    else:
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


def _dtensor_to_shard_tensor(dtensor: DTensor, spec: ShardTensorSpec) -> "ShardTensor":
    r"""Promote a DTensor to a ShardTensor (no autograd).

    Callers must supply a resolved ``spec``.  Use inside backward (with spec
    from ctx) or after resolving a spec via :func:`_resolve_spec_for_dtensor`.
    """
    if isinstance(dtensor, ShardTensor):
        # Shortcut if we're already a ShardTensor:
        return dtensor
    st = ShardTensor.__new__(
        ShardTensor,
        local_tensor=dtensor._local_tensor,
        spec=spec,
        requires_grad=dtensor.requires_grad,
    )
    return st


# ============================================================================
# Layer 2 -- Autograd Functions (use Layer 1 inside fwd / bwd)
# ============================================================================


class _DTensorToShardTensor(torch.autograd.Function):
    r"""Differentiable promotion: DTensor -> ShardTensor.

    This is to always connect the graphs for the backward pass
    when we have to use a fallback option.

    Forward: :func:`_dtensor_to_shard_tensor`.
    Backward: :func:`_shard_tensor_to_dtensor`.
    """

    @staticmethod
    def forward(ctx, dtensor: DTensor, spec: ShardTensorSpec) -> "ShardTensor":
        return _dtensor_to_shard_tensor(dtensor, spec)

    @staticmethod
    def backward(ctx, grad_output: "ShardTensor"):
        return _shard_tensor_to_dtensor(grad_output), None


class _ShardTensorToDTensor(torch.autograd.Function):
    r"""Differentiable conversion: ShardTensor -> DTensor.

    This is to always connect the graphs for the backward pass
    when we have to use a fallback option.

    Forward: :func:`_shard_tensor_to_dtensor` (caches spec).
    Backward: :func:`_dtensor_to_shard_tensor` (reuses cached spec).
    """

    @staticmethod
    def forward(st: "ShardTensor") -> DTensor:
        return _shard_tensor_to_dtensor(st)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        (st,) = inputs
        ctx.shard_tensor_spec = st._spec

    @staticmethod
    def backward(ctx, grad_output: DTensor):
        cached_spec = ctx.shard_tensor_spec
        grad_placements = tuple(grad_output._spec.placements)
        # Keep the cached uneven sharding shapes, but adopt the gradient's
        # placements so a Replicate->Partial flip isn't dropped (which would
        # skip the all-reduce at the plain-tensor boundary). Shard dims are
        # unchanged, so the cached shard shapes stay valid. Likewise adopt
        # the gradient's tensor_meta: the grad shares the primal's shape but
        # not necessarily its stride (grad-of-permute is permute-of-grad),
        # and stamping the primal's stride onto differently-laid-out grad
        # memory breaks downstream .view() calls.
        cached_spec = dataclasses.replace(
            cached_spec,
            placements=grad_placements,
            tensor_meta=grad_output._spec.tensor_meta,
        )
        return (_dtensor_to_shard_tensor(grad_output, cached_spec),)


# ============================================================================
# Layer 3 -- Smart single-tensor converters (auto-diff when grad_fn present)
# ============================================================================


def _resolve_spec_for_dtensor(
    dtensor: DTensor, input_args: tuple = ()
) -> ShardTensorSpec:
    r"""Resolve a ShardTensorSpec for *dtensor*.

    Tries to reuse a spec from a ShardTensor in *input_args* whose
    ``tensor_meta`` and ``placements`` match.  Falls back to chunk-based
    inference (no communication).
    """
    for arg in input_args:
        if (
            isinstance(arg, ShardTensor)
            and dtensor._spec.tensor_meta == arg._spec.tensor_meta
            and dtensor._spec.placements == arg._spec.placements
        ):
            return arg._spec
    spec = _infer_shard_tensor_spec_from_local_chunks(
        dtensor._local_tensor,
        dtensor._spec.mesh,
        dtensor._spec.placements,
        sharding_shapes="chunk",
        global_shape=dtensor.shape,
    )
    # Adopt the result's own tensor_meta: chunk inference fabricates a
    # C-contiguous stride, which is wrong for non-contiguous results (e.g.
    # permute). A lying stride makes the wrapper report is_contiguous()=True
    # -- .contiguous() becomes a no-op -- and downstream .view() calls fail
    # against the actual local layout.
    return dataclasses.replace(spec, tensor_meta=dtensor._spec.tensor_meta)


# This is a thread-safe reentry guard.
# Goal is to prevent recursion into the fallback conversion paths.
_conversion_guard = threading.local()


def _conversion_active() -> bool:
    r"""Return whether ShardTensor<->DTensor conversion is currently active."""
    return getattr(_conversion_guard, "depth", 0) > 0


@contextmanager
def _conversion_scope():
    r"""Re-entrant conversion guard for cast-down/cast-up paths."""
    previous_depth = getattr(_conversion_guard, "depth", 0)
    _conversion_guard.depth = previous_depth + 1
    try:
        yield
    finally:
        if previous_depth == 0:
            delattr(_conversion_guard, "depth")
        else:
            _conversion_guard.depth = previous_depth


def _find_mesh_in_args(*objs: object) -> DeviceMesh | None:
    r"""Return the mesh of the first ``ShardTensor`` found in ``objs``.

    This is the reference mesh used to promote plain tensors. It is only ever
    reached from ShardTensor dispatch, so a ShardTensor is guaranteed to be
    present, and promotion exists to line plain weights up with the sharded
    activation -- so we key off the ShardTensor's mesh. Walks nested
    ``Mapping``/``tuple``/``list`` containers and short-circuits on the first
    match; returns ``None`` if none is present.
    """
    for obj in objs:
        if isinstance(obj, ShardTensor):
            return obj.device_mesh
        if isinstance(obj, Mapping):
            found = _find_mesh_in_args(*obj.values())
            if found is not None:
                return found
        elif isinstance(obj, (tuple, list)):
            found = _find_mesh_in_args(*obj)
            if found is not None:
                return found
    return None


def _needs_promotion(tensor: torch.Tensor) -> bool:
    r"""Whether a plain tensor participating in ShardTensor dispatch must be
    promoted to a ``Replicate`` DTensor.

    Scalars (0-dim) are normally exempt -- DTensor handles them natively via
    implicit replication -- EXCEPT when they carry gradient state: the
    implicit path emits the scalar's cotangent as a DTensor with no
    ``from_local`` bridge to unwrap it, depositing a DTensor ``.grad`` on a
    plain leaf (e.g. a learnable scalar gate). Promoting grad-tracking
    scalars routes them through the differentiable bridge, whose backward
    returns a plain local gradient.
    """
    if tensor.dim() >= 1:
        return True
    return torch.is_grad_enabled() and tensor.requires_grad


def _promote_plain_tensor_to_dtensor(tensor: torch.Tensor, mesh: DeviceMesh) -> DTensor:
    r"""Promote a plain ``torch.Tensor`` to a ``Replicate`` DTensor on ``mesh``.

    Uses the differentiable ``DTensor.from_local`` so that, in backward, the
    promoted tensor's gradient is normalized from ``Partial`` to ``Replicate``
    (an eager all-reduce) before flowing back to the original plain tensor.
    """
    if ShardTensor._promotion_mode is TensorPromotionMode.WARN:
        warnings.warn(
            "ShardTensor auto-promoting a plain torch.Tensor "
            f"(shape={tuple(tensor.shape)}, dtype={tensor.dtype}) to a "
            f"Replicate tensor on mesh {mesh}. This usually means a "
            "non-distributed model weight met a sharded activation.",
            stacklevel=2,
        )
    placements = [Replicate()] * mesh.ndim
    return DTensor.from_local(tensor, mesh, placements)


def _promote_plain_handler_args(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[tuple[object, ...], dict[str, object]]:
    r"""Promote plain tensors in handler args to ``Replicate`` DTensors.

    Applied in ``__torch_function__`` before every registered function
    handler, so handlers uniformly see distributed tensors with a ``_spec``
    -- the same contract the DTensor fallback provides via
    ``_convert_args_to_dtensor``, and with the same rules: promotion honors
    the promotion mode (no-op when ``DISABLED``), requires a reference mesh
    from an accompanying ShardTensor, exempts grad-free scalar tensors (which
    distributed ops handle natively), and promotes only exact
    ``torch.Tensor`` / ``nn.Parameter`` instances. Reuses
    :func:`_promote_plain_tensor_to_dtensor`, including its differentiable
    ``from_local`` bridge back to the plain tensor (e.g. a DDP-replicated
    parameter like FLARE's ``q_global``). Walks only the top level of
    ``args``/``kwargs``.
    """
    if ShardTensor._promotion_mode is TensorPromotionMode.DISABLED:
        return args, kwargs

    mesh = _find_mesh_in_args(args, kwargs)
    if mesh is None:
        return args, kwargs

    def promote(obj: object) -> object:
        # Exact types only: a tensor subclass with its own dispatch behavior
        # (AsyncCollectiveTensor, functional-tensor wrappers, ...) wrapped in
        # a DTensor nests distributed wrappers, and view ops on the nested
        # tensor re-enter this dispatch without terminating.
        if type(obj) in (torch.Tensor, torch.nn.Parameter) and _needs_promotion(obj):
            return _promote_plain_tensor_to_dtensor(obj, mesh)
        return obj

    # Same guard the fallback's conversions run under: ops that
    # DTensor.from_local performs on its input (view_as) must pass through
    # __torch_function__ plainly, not recurse into handlers or the fallback.
    with _conversion_scope():
        return tuple(promote(a) for a in args), {
            k: promote(v) for k, v in kwargs.items()
        }


def _dispatch_fallback_via_dtensor(
    func: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object] | None = None,
) -> object:
    r"""Execute an ATen op through DTensor fallback using PURE data conversion.

    Native Autograd wraps this hook, so we must NOT build an internal graph
    using .apply(). We just do the math and let PyTorch track the outer graph.
    """
    ref_mesh = _find_mesh_in_args(args, kwargs)
    with _conversion_scope():
        converted_args = tuple(
            _convert_args_to_dtensor(arg, use_autograd=False, ref_mesh=ref_mesh)
            for arg in args
        )
        converted_kwargs = {
            k: _convert_args_to_dtensor(v, use_autograd=False, ref_mesh=ref_mesh)
            for k, v in (kwargs or {}).items()
        }

    dispatch_res = func(*converted_args, **(converted_kwargs or {}))

    with _conversion_scope():
        return _convert_results_to_shard_tensor(dispatch_res, args, use_autograd=False)


def _torch_function_fallback_via_dtensor(
    func: Callable,
    args: tuple[object, ...],
    kwargs: dict[str, object] | None = None,
) -> object:
    r"""Execute a __torch_function__ fallback through DTensor safely.

    Because this executes at the Python API level (above Autograd), we MUST
    use autograd functions (.apply) to bridge the tracking manually.
    """
    ref_mesh = _find_mesh_in_args(args, kwargs)
    with _conversion_scope():
        converted_args = tuple(
            _convert_args_to_dtensor(arg, use_autograd=True, ref_mesh=ref_mesh)
            for arg in args
        )
        converted_kwargs = {
            k: _convert_args_to_dtensor(v, use_autograd=True, ref_mesh=ref_mesh)
            for k, v in (kwargs or {}).items()
        }

    with torch._C.DisableTorchFunctionSubclass():
        result = func(*converted_args, **converted_kwargs)

    with _conversion_scope():
        return _convert_results_to_shard_tensor(result, args, use_autograd=True)


# ============================================================================
# Layer 4 -- Recurse utilities (walk args / kwargs / results)
# ============================================================================


def _convert_args_to_dtensor(
    arg: object, use_autograd: bool = False, ref_mesh: DeviceMesh | None = None
) -> object:
    r"""Recursively replace ShardTensors with DTensors.

    If use_autograd is True, uses Layer 2 to preserve the graph connection.

    Plain ``torch.Tensor`` arguments are auto-promoted to a ``Replicate``
    DTensor on ``ref_mesh`` according to ``ShardTensor._promotion_mode`` (see
    :class:`TensorPromotionMode`). Promotion is skipped when the mode is
    ``DISABLED``, when there is no reference mesh, or for grad-free scalar
    (0-dim) tensors (which DTensor handles natively; grad-tracking scalars
    need the differentiable bridge -- see :func:`_needs_promotion`).
    """
    match arg:
        case ShardTensor():
            if use_autograd and arg.requires_grad and torch.is_grad_enabled():
                return _ShardTensorToDTensor.apply(arg)
            return _shard_tensor_to_dtensor(arg)
        case DTensor():
            # DTensor can be iterable; exit early deliberately
            return arg
        case Mapping():
            return type(arg)(
                {
                    k: _convert_args_to_dtensor(v, use_autograd, ref_mesh)
                    for k, v in arg.items()
                }
            )
        case tuple():
            return tuple(
                _convert_args_to_dtensor(a, use_autograd, ref_mesh) for a in arg
            )
        case list():
            return [_convert_args_to_dtensor(a, use_autograd, ref_mesh) for a in arg]
        case torch.Tensor() if (
            ShardTensor._promotion_mode is not TensorPromotionMode.DISABLED
            and ref_mesh is not None
            and _needs_promotion(arg)
        ):
            return _promote_plain_tensor_to_dtensor(arg, ref_mesh)
        case _:
            return arg


def _convert_results_to_shard_tensor(
    result: object, input_args: tuple, use_autograd: bool = False
) -> object:
    r"""Recursively replace DTensors with ShardTensors in an op result.

    If use_autograd is True, uses Layer 2 to preserve the graph connection.
    Handles None returns gracefully for inplace ATen operations.
    """
    if result is None:
        return None

    if isinstance(result, DTensor):
        spec = _resolve_spec_for_dtensor(result, input_args)

        # If autograd graph connection is requested AND the DTensor actually
        # requires tracking (it has a grad_fn or requires_grad is active)
        if (
            use_autograd
            and torch.is_grad_enabled()
            and (result.grad_fn is not None or result.requires_grad)
        ):
            return _DTensorToShardTensor.apply(result, spec)

        return _dtensor_to_shard_tensor(result, spec)

    if isinstance(result, Mapping):
        return type(result)(
            {
                k: _convert_results_to_shard_tensor(v, input_args, use_autograd)
                for k, v in result.items()
            }
        )

    # Preserve PyTorch's structured tuple results so named fields such as
    # ``torch.max(...).indices`` remain available after converting their values.
    if type(result).__module__ == torch.return_types.__name__:
        return type(result)(
            _convert_results_to_shard_tensor(d, input_args, use_autograd)
            for d in result
        )

    # Explicit allowlist mirroring _convert_args_to_dtensor: only walk into
    # tuple / list containers. A generic Iterable check would crash on things
    # like torch.UntypedStorage (iterable over bytes) or torch.Tensor because
    # their constructors don't accept a generator. Other tuple subclasses
    # degrade to plain tuples here.
    if isinstance(result, tuple):
        return tuple(
            _convert_results_to_shard_tensor(d, input_args, use_autograd)
            for d in result
        )

    if isinstance(result, list):
        return [
            _convert_results_to_shard_tensor(d, input_args, use_autograd)
            for d in result
        ]

    return result


class _ToTorchTensor(torch.autograd.Function):
    r"""Autograd function to convert a ShardTensor to a regular PyTorch tensor.

    This class handles the conversion from ShardTensor to ``torch.Tensor`` in both
    forward and backward passes, maintaining proper gradient flow. Slices the
    ShardTensor to the local component only on the current rank.
    """

    @staticmethod
    def forward(
        input: "ShardTensor",
        grad_placements: Sequence[Placement] | None = None,
    ) -> torch.Tensor:
        r"""Convert ShardTensor to torch.Tensor in forward pass.

        Parameters
        ----------
        input : ShardTensor
            ShardTensor to convert.
        grad_placements : Sequence[Placement], optional
            Sequence of placements to use for gradients. When omitted, input
            ``Partial`` placements are normalized to ``Replicate`` because the
            cotangent of a local contribution is fully materialized.

        Returns
        -------
        torch.Tensor
            Local tensor representation of the ShardTensor.
        """

        # Force the local view to inherit the requires_grad state of the ShardTensor
        local_tensor = input._local_tensor
        res = local_tensor.view_as(local_tensor)
        res.requires_grad_(input.requires_grad)
        return res

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        r"""Save the source ShardTensorSpec and optional grad_placements."""
        input, grad_placements = inputs
        ctx.shard_tensor_spec = input._spec
        ctx.grad_placements = grad_placements

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
            # The local view of a Partial value exposes this rank's
            # contribution. Its incoming cotangent is complete local gradient
            # data, not another pending reduction, so emit Replicate metadata.
            grad_placements = tuple(
                Replicate() if placement.is_partial() else placement
                for placement in shard_tensor_spec.placements
            )
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
        local_input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        sharding_shapes: str | dict[int, list[tuple[int, ...]]] = "chunk",
        global_shape: tuple[int, ...] | None = None,
    ) -> "ShardTensor":
        r"""Convert a local torch.Tensor to a ShardTensor in forward pass.

        Parameters
        ----------
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
        global_shape : Optional[Tuple[int, ...]], optional
            Global shape of the full tensor. Required when
            ``sharding_shapes="chunk"``; ignored otherwise.

        Returns
        -------
        ShardTensor
            ShardTensor constructed from the local input tensor.
        """
        # This function is simpler than the corresponding DTensor implementation on the surface
        # because under the hood, we have some logic here to infer the sharding shapes.
        shard_tensor_spec = _infer_shard_tensor_spec_from_local_chunks(
            local_input, device_mesh, placements, sharding_shapes, global_shape
        )

        shard_tensor = ShardTensor(
            local_input,
            shard_tensor_spec,
            requires_grad=local_input.requires_grad,
        )

        return shard_tensor

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        r"""Save the source mesh and placements for the backward redistribute."""
        _local_input, device_mesh, placements, _sharding_shapes, _global_shape = inputs
        ctx.previous_placement = placements
        ctx.previous_mesh = device_mesh

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: "ShardTensor",
    ) -> tuple[torch.Tensor, None, None, None, None]:
        r"""Convert gradient ShardTensor back to torch.Tensor in backward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved tensors/variables from forward.
        grad_output : ShardTensor
            Gradient ShardTensor to convert back to torch.Tensor.

        Returns
        -------
        Tuple[torch.Tensor, None, None, None, None]
            Tuple containing the local tensor gradient, and None for
            device_mesh, placements, sharding_shapes, and global_shape
            gradients (not differentiable).

        Notes
        -----
        No ``Partial`` placement may cross the ShardTensor -> ``torch.Tensor``
        boundary. The gradient is imperatively (eagerly) resolved to the
        forward placement with any ``Partial`` mapped to ``Replicate`` (an
        all-reduce now), so the plain gradient handed back to the original
        tensor is fully reduced -- this is what lets FSDP reduce-scatter / DDP
        all-reduce see a correct gradient.
        """
        previous_placement = ctx.previous_placement
        # Target placement is the forward placement with Partial -> Replicate.
        # redistribute() forbids resharding *to* Partial, so the target is
        # always a valid (non-Partial) placement; reaching it from a Partial
        # grad performs the necessary all-reduce / reduce-scatter eagerly.
        target = tuple(Replicate() if p.is_partial() else p for p in previous_placement)
        if grad_output.placements != target:
            grad_output = grad_output.redistribute(grad_output._spec.mesh, target)

        return grad_output.to_local(), None, None, None, None


def _is_tracing(args: object, kwargs: object = None) -> bool:
    r"""True when any tensor in ``args``/``kwargs`` is fake, i.e. inside a ``torch.compile``
    trace. ``torch.compiler.is_compiling()`` is unreliable here -- ``__torch_function__``
    runs eagerly on fake operands and reads ``False`` -- so detect a ``FakeTensor`` operand
    instead (correctly ``False`` on the real-tensor eager path)."""
    for leaf in torch.utils._pytree.tree_leaves((args, kwargs)):
        if isinstance(leaf, torch.Tensor) and is_fake(leaf):
            return True
    return False


class ShardTensor(torch.Tensor):
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

    A ``Partial`` placement always describes a pending reduction. Its local
    tensor is this rank's contribution to the logical value, not a replicated
    copy of that value. Resolve it with :meth:`redistribute` or
    :meth:`full_tensor` before treating the local data as complete. This rule
    also applies to gradients: backward paths that retain a fully replicated
    gradient label it ``Replicate`` rather than using ``Partial`` as layout-only
    metadata.

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

    # Functions tied to autograd-graph *identity*: they bind a hook/flag to
    # this exact tensor's autograd node, or query gradients for these exact
    # tensor objects. The DTensor fallback would run them on freshly
    # converted copies: hooks would bind to discarded temporaries, and
    # ``torch.autograd.grad`` would query tensors that aren't in the graph
    # (silently None under ``allow_unused=True``). AOTAutograd's joint trace
    # makes exactly that grad call on the subclass primals, so intercepting
    # it made compiled regions return plain-tensor gradients for ShardTensor
    # inputs. __torch_function__ runs these directly on the real tensors.
    _autograd_passthrough_functions: frozenset = frozenset(
        fn
        for fn in (
            getattr(torch.Tensor, "register_hook", None),
            getattr(torch.Tensor, "register_post_accumulate_grad_hook", None),
            getattr(torch.Tensor, "retain_grad", None),
            torch.autograd.grad,
        )
        if fn is not None
    )

    # Upon construction of any ShardTensor objects, this will be set to true.
    # Wrappers are triggered dynamically, so the wrapping will be pass-through
    # exclusively until true.
    _enable_shard_patches: bool = False

    # Controls how plain torch.Tensor arguments are handled when they appear
    # alongside a ShardTensor in an intercepted op (see TensorPromotionMode).
    _promotion_mode: TensorPromotionMode = TensorPromotionMode.SILENT

    # -- Subclass extension points (compile-safe subclassing) --
    # Extra inner-tensor attribute names a subclass carries through flatten/unflatten so
    # per-instance metadata rides as a graph input; the base flatten protocol handles them.
    _extra_inner_tensors: tuple[str, ...] = ()

    # Attribute names copied from an op input onto its eager op-result, re-classing it to
    # the subclass (needs a __dict__-bearing subclass); skipped under compile.
    _subclass_propagated_attrs: tuple[str, ...] = ()

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
    def get_promotion_mode(cls) -> TensorPromotionMode:
        r"""Return the active plain-tensor promotion mode (defaults to ``SILENT``)."""
        return cls._promotion_mode

    @classmethod
    def set_promotion_mode(cls, mode: TensorPromotionMode) -> None:
        r"""Set the plain-tensor promotion mode.

        ``mode`` may be a :class:`TensorPromotionMode` or an equivalent string
        (``"disabled"``, ``"warn"``, ``"silent"``), which is coerced.
        """
        cls._promotion_mode = TensorPromotionMode(mode)

    @classmethod
    @contextmanager
    def promotion_mode(cls, mode: TensorPromotionMode):
        r"""Temporarily set the promotion mode, restoring the previous one on exit."""
        previous = cls._promotion_mode
        cls.set_promotion_mode(mode)
        try:
            yield
        finally:
            cls._promotion_mode = previous

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

        Before the handler is called, plain non-scalar ``torch.Tensor``
        arguments are promoted to ``Replicate`` DTensors on the mesh of the
        accompanying ShardTensor arguments (honoring the promotion mode) --
        the same contract the DTensor fallback provides -- so handlers
        uniformly receive distributed tensors with a ``_spec``.

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
        ret = torch.Tensor._make_wrapper_subclass(
            cls,
            spec.tensor_meta.shape,
            strides=spec.tensor_meta.stride,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=False,
        )

        ret._spec = spec
        ret._local_tensor = local_tensor

        # Set requires_grad AFTER _spec/_local_tensor are assigned, using
        # the C-level setter directly (bypassing __torch_function__ which
        # would convert to DTensor and set on a temporary).
        if requires_grad:
            with torch._C.DisableTorchFunctionSubclass():
                torch.Tensor.requires_grad.__set__(ret, True)

        cls._enable_shard_patches = True
        return ret

    def __repr__(self) -> str:
        return (
            "ShardTensor("
            f"local_tensor={repr(self._local_tensor)}, "
            f"device_mesh={repr(self._spec.mesh)}, "
            f"placements={repr(self._spec.placements)}"
            ")"
        )

    def __str__(self) -> str:
        # Avoid Tensor/DTensor string formatting paths that can re-enter dispatch.
        return self.__repr__()

    def __format__(self, format_spec: str) -> str:
        # Format as plain Python string to bypass tensor formatting internals.
        return format(str(self), format_spec)

    @property
    def device_mesh(self) -> DeviceMesh:
        """Return the :class:`DeviceMesh` that this tensor is distributed over."""
        return self._spec.mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        """Return the placement strategy for each mesh dimension."""
        return self._spec.placements

    # -- Subclass extension hooks --
    # Carry extra flatten context without re-implementing the protocol; both default to no-ops.

    def __subclass_flatten_context__(self) -> object | None:
        r"""Hook: extra per-instance metadata to carry through Dynamo flatten, nested as
        ``(base_ctx, subclass_ctx)``. Must be a graph-constant; default ``None``."""
        return None

    def __subclass_unflatten__(self, subclass_ctx: object) -> None:
        r"""Hook: reattach :meth:`__subclass_flatten_context__` metadata on reconstruction."""
        return None

    def _stable_inner_sentinel(self, cache_attr: str) -> torch.Tensor:
        r"""Per-instance-cached empty int64 tensor for an unset inner slot. Cached because
        AOTAutograd re-flattens an input and needs the same object each time; minted in
        ``_local_tensor``'s device/fake context so it traces correctly."""
        cached = getattr(self, cache_attr, None)
        if cached is None:
            cached = torch.zeros(0, dtype=torch.int64, device=self._local_tensor.device)
            setattr(self, cache_attr, cached)
        return cached

    @classmethod
    def __metadata_guard__(cls, orig: object, other: object) -> bool:
        r"""Dynamo subclass metadata guard: compare only ``(spec, requires_grad)`` and
        ignore any nested subclass context, which may hold non-value-equal objects that
        would otherwise always fail the guard and block compile. Mirrors DTensor."""
        try:
            orig_base = orig[0] if isinstance(orig[0], tuple) else orig
            other_base = other[0] if isinstance(other[0], tuple) else other
            (orig_spec, orig_rg) = orig_base
            (other_spec, other_rg) = other_base
        except (TypeError, ValueError):
            return bool(orig == other)
        return bool(orig_rg == other_rg) and bool(orig_spec == other_spec)

    @classmethod
    def _find_metadata_source(cls, args, kwargs) -> "ShardTensor | None":
        r"""Input instance whose subclass metadata rides onto op outputs: the first
        ``cls``-typed arg already carrying routing, else the first ``cls``-typed arg."""
        sentinel = cls._subclass_propagated_attrs[0]
        fallback = None

        def _scan(vals):
            nonlocal fallback
            for v in vals:
                if isinstance(v, cls):
                    if fallback is None:
                        fallback = v
                    if getattr(v, sentinel, None) is not None:
                        return v
                elif isinstance(v, (tuple, list)):
                    found = _scan(v)
                    if found is not None:
                        return found
            return None

        found = _scan(args)
        if found is None and kwargs:
            found = _scan(kwargs.values())
        return found if found is not None else fallback

    @classmethod
    def _propagate_subclass_metadata(cls, result: object, source: "ShardTensor"):
        r"""Copy ``_subclass_propagated_attrs`` from ``source`` onto any ShardTensor in
        ``result`` lacking them, re-classing a base autowrap result to ``cls`` first."""
        attrs = cls._subclass_propagated_attrs
        if not attrs:
            return result
        sentinel = attrs[0]

        def _apply(t: object) -> None:
            if not isinstance(t, ShardTensor):
                return
            if type(t) is not cls:
                # Re-class a base autowrap result up to the subclass; skip on mismatch.
                if not issubclass(cls, type(t)):
                    return
                try:
                    t.__class__ = cls
                except TypeError:
                    return
            if getattr(t, sentinel, None) is None:
                for attr in attrs:
                    setattr(t, attr, getattr(source, attr))

        if isinstance(result, ShardTensor):
            _apply(result)
        elif isinstance(result, (tuple, list)):
            for r in result:
                _apply(r)
        return result

    def __tensor_flatten__(self):
        inner_names = ["_local_tensor", *self._extra_inner_tensors]
        base_ctx = (self._spec, self.requires_grad)
        subclass_ctx = self.__subclass_flatten_context__()
        if subclass_ctx is None:
            return inner_names, base_ctx
        return inner_names, (base_ctx, subclass_ctx)

    @classmethod
    def __tensor_unflatten__(
        cls, inner_tensors, flatten_spec, outer_size, outer_stride
    ):
        # Accept a subclass's nested ``(base_ctx, subclass_ctx)`` or the base's
        # flat ``(spec, requires_grad)`` context.
        if isinstance(flatten_spec[0], tuple):
            base_ctx, subclass_ctx = flatten_spec
        else:
            base_ctx, subclass_ctx = flatten_spec, None
        spec, requires_grad = base_ctx
        local_tensor = inner_tensors["_local_tensor"]
        unflatten_meta = TensorMeta(
            shape=outer_size,
            stride=outer_stride,
            dtype=spec.tensor_meta.dtype,
        )

        # Reconstruct ``_sharding_shapes`` as plain int tuples. Reuse the incoming shapes
        # only when fully concrete; otherwise re-derive from chunk semantics against
        # ``outer_size`` (pure arithmetic, no collectives). This avoids ``None`` (which
        # forces a blocking, non-traceable shape all-gather) and avoids ``torch.Size`` /
        # SymInt entries, which break AOT tracing and make the spec unhashable.
        def _all_concrete(shapes_by_mesh_dim):
            return all(
                type(dim) is int
                for shapes in shapes_by_mesh_dim.values()
                for shape in shapes
                for dim in shape
            )

        if spec._sharding_shapes is not None and _all_concrete(spec._sharding_shapes):
            sharding_shapes = {
                mesh_dim: tuple(tuple(s) for s in shapes)
                for mesh_dim, shapes in spec._sharding_shapes.items()
            }
        else:
            chunk_shapes = compute_sharding_shapes_from_chunking_global_shape(
                spec.mesh, spec.placements, tuple(outer_size)
            )
            sharding_shapes = {
                mesh_dim: tuple(tuple(s) for s in shapes)
                for mesh_dim, shapes in chunk_shapes.items()
            }
            # Chunk derivation assumes an even (torch.chunk) layout. When the
            # spec arrived with no ``_sharding_shapes`` at all, verify that
            # assumption against the actual local shard: an unevenly sharded
            # tensor whose spec lost its shapes would otherwise get even-chunk
            # collective sizes baked into the compiled graph — silently wrong
            # results, and Dynamo's guards cannot discriminate (None == None).
            # Only checkable when both sides are concrete (static shapes);
            # symbolic recompute is faithful by construction (see above).
            local_shape = tuple(local_tensor.shape)
            coords = spec.mesh.get_coordinate()
            if (
                spec._sharding_shapes is None
                and coords is not None
                and all(type(d) is int for d in local_shape)
            ):
                for mesh_dim in sharding_shapes:
                    expected = sharding_shapes[mesh_dim][coords[mesh_dim]]
                    if (
                        all(type(d) is int for d in expected)
                        and expected != local_shape
                    ):
                        raise RuntimeError(
                            "ShardTensor crossed a torch.compile boundary with "
                            "_sharding_shapes=None, and its local shard shape "
                            f"{local_shape} does not match the even-chunk "
                            f"assumption {expected} (mesh dim {mesh_dim}, "
                            f"global shape {tuple(outer_size)}). This tensor "
                            "is unevenly sharded but its spec lost the shard "
                            "shapes; compiling would bake wrong collective "
                            "sizes into the graph. Construct it with explicit "
                            "or inferred sharding_shapes before compiling."
                        )

        unflatten_spec = ShardTensorSpec(
            mesh=spec.mesh,
            placements=spec.placements,
            tensor_meta=unflatten_meta,
            _local_shape=local_tensor.shape,
            _sharding_shapes=sharding_shapes,
        )
        # Pass ``local_tensor`` through unchanged: do NOT force its ``requires_grad`` to the
        # wrapper's (set via ``__new__`` below). Forcing it makes the inner disagree with the
        # real tensor for a detached op-result local, tripping ``assert_metadata_eq`` when
        # Dynamo re-fakes across a graph break. Matches DTensor (wrapper flag only).
        # ``cls`` reconstructs a subclass as itself -- no ``__class__`` reassignment.
        out = cls.__new__(
            cls,
            local_tensor=local_tensor,
            spec=unflatten_spec,
            requires_grad=requires_grad,
        )
        # Reattach any declared extra inner tensors, then let the subclass
        # reattach its own context. Both are no-ops for base ShardTensor.
        for name in cls._extra_inner_tensors:
            if name in inner_tensors:
                setattr(out, name, inner_tensors[name])
        if subclass_ctx is not None:
            out.__subclass_unflatten__(subclass_ctx)
        return out

    def _stable_hash_for_caching(self) -> str:
        r"""Return a cross-process stable hash for the AOT autograd cache.

        Mirrors ``DTensor._stable_hash_for_caching`` (see the note on tensor
        subclass stable hashing in torch's ``autograd_cache.py``). Without
        it, PT2 falls back to pickling the spec, which is not byte-stable
        across processes -- silent cache misses, recompiling on every warm
        start. Local metadata only; no collectives.
        """
        cache_data = self._spec._stable_hash() + str(self.requires_grad)
        return hashlib.blake2b(cache_data.encode(), digest_size=16).hexdigest()

    # -- AOTAutograd tangent coercion hooks ------------------------------------
    # AOTAutograd records the expected tangent metadata at trace time and
    # validates it at backward runtime. When a forward output has a
    # ``Partial`` placement (typical right after a reduction like sum/mean),
    # the tangent flowing back from ``.backward()`` is materialized as
    # ``Replicate`` and AOT raises:
    #     "During the backward, we encountered a tensor subclass where we
    #      guessed its metadata incorrectly."
    # These hooks mirror DTensor's implementation in
    # ``torch.distributed.tensor._api`` and reconcile the two ends without
    # weakening Partial's meaning:
    # (1) at trace time, resolve a forward Partial value and record Replicate
    #     as the expected tangent layout;
    # (2) at runtime, redistribute the incoming tangent according to its real
    #     placement. A Partial runtime tangent is an unreduced contribution and
    #     resolving it to Replicate must perform the reduction.

    def __coerce_tangent_metadata__(self) -> "ShardTensor":
        """Trace-time hook: coerce this tensor so its metadata matches a tangent.

        Returns ``self`` if no Partial placement is present (no work needed).
        Otherwise redistributes Partial placements to Replicate, which is the
        layout the autograd engine produces for tangents.
        """
        if not any(isinstance(p, Partial) for p in self.placements):
            return self
        new_placements = [
            Replicate() if isinstance(p, Partial) else p for p in self.placements
        ]
        return self.redistribute(
            device_mesh=self.device_mesh, placements=new_placements
        )

    def __coerce_same_metadata_as_tangent__(
        self,
        flatten_spec: tuple,
        expected_type: type | None = None,
    ) -> "ShardTensor | None":
        """Runtime hook: redistribute ``self`` to match the recorded tangent's placements
        and ``_sharding_shapes`` (preserving uneven layouts).

        Subclass-friendly: a subclass whose forward output is the subclass type but
        whose backward tangent arrives as base ``ShardTensor`` would otherwise make
        AOT raise "guessed its metadata incorrectly". So this accepts a subclass's
        nested ``(base_ctx, subclass_ctx)`` flatten context and, for a differing
        ``ShardTensor``-subclass ``expected_type``, rebuilds through *that* type's
        ``__tensor_unflatten__``; a foreign ``expected_type`` keeps DTensor's ``None``.
        """
        # Accept a subclass's nested ``(base_ctx, subclass_ctx)`` context or the
        # base's flat ``(spec, requires_grad)`` context.
        base_ctx = (
            flatten_spec[0] if isinstance(flatten_spec[0], tuple) else flatten_spec
        )
        (spec, _requires_grad) = base_ctx

        # __coerce_tangent_metadata__ resolves Partial before AOT records the
        # expected tangent metadata. Seeing Partial in the recorded target means
        # that normalization was bypassed; stamping that label onto complete data
        # would make the placement lie about whether reduction is pending.
        if any(isinstance(p, Partial) for p in spec.placements):
            raise RuntimeError(
                "AOTAutograd recorded a Partial tangent target for ShardTensor; "
                "__coerce_tangent_metadata__ must normalize expected tangent "
                "placements to Replicate"
            )

        # ``{}`` (fully-replicated / op-result rewrap) and ``None`` (unknown) both
        # carry no per-shard shape info; with matching placements there is nothing
        # to reconcile (strict ``==`` would trigger a data-no-op redistribute).
        if self._spec.placements == spec.placements and (
            (self._spec._sharding_shapes or None) == (spec._sharding_shapes or None)
        ):
            coerced: "ShardTensor" = self
        else:
            # Bypass ``self.redistribute()`` so we can thread the recorded
            # per-tensor-dim shard sizes through to the local redistribute.
            target_sharding_shapes_by_tensor_dim: dict[int, list[int]] = {}
            if spec._sharding_shapes is not None:
                for mesh_dim, placement in enumerate(spec.placements):
                    if (
                        isinstance(placement, Shard)
                        and mesh_dim in spec._sharding_shapes
                    ):
                        shard_shapes = spec._sharding_shapes[mesh_dim]
                        target_sharding_shapes_by_tensor_dim[placement.dim] = [
                            s[placement.dim] for s in shard_shapes
                        ]

            move_spec = ShardTensorSpec(
                mesh=self.device_mesh,
                placements=spec.placements,
                tensor_meta=self._spec.tensor_meta,
                _sharding_shapes=spec._sharding_shapes,
            )
            new_local = redistribute_local_shard_tensor(
                self._local_tensor,
                self._spec,
                move_spec,
                async_op=False,
                target_sharding_shapes=target_sharding_shapes_by_tensor_dim,
            )

            # The recorded target contains no Partial, so this metadata describes
            # the communication performed above rather than relabeling local data.
            target_spec = ShardTensorSpec(
                mesh=self.device_mesh,
                placements=spec.placements,
                tensor_meta=self._spec.tensor_meta,
                _sharding_shapes=spec._sharding_shapes,
            )
            target_spec._local_shape = new_local.shape

            coerced = ShardTensor(
                new_local.contiguous(),
                target_spec,
                requires_grad=self.requires_grad,
            )

        if expected_type is not None and expected_type is not type(coerced):
            # Cross-type: rebuild via the expected ShardTensor subclass's own
            # unflatten (reclass + reattach metadata); a foreign type keeps the
            # DTensor ``None`` convention.
            if isinstance(expected_type, type) and issubclass(
                expected_type, ShardTensor
            ):
                return expected_type.__tensor_unflatten__(
                    {"_local_tensor": coerced._local_tensor},
                    flatten_spec,
                    tuple(coerced.shape),
                    coerced.stride(),
                )
            # Coerce down to a plain tensor at a fully-replicated boundary (local == global, so
            # the local view is lossless); the mirror of the subclass rebuild above.
            if expected_type is torch.Tensor and all(
                isinstance(p, Replicate) for p in coerced._spec.placements
            ):
                return coerced._local_tensor
            return None
        return coerced

    # -- Autograd property overrides -------------------------------------------
    # The C-level requires_grad is authoritative for autograd engine
    # decisions; we read it first and fall back to _local_tensor for the
    # case where _make_wrapper_subclass didn't propagate it correctly.
    # For grad, the autograd engine accumulates at the C level, so we
    # check there first then fall back to _local_tensor.grad.

    @property  # type: ignore[override]
    def requires_grad(self) -> bool:  # type: ignore[override]
        """Whether this tensor requires gradient computation.

        Returns ``True`` if either the wrapper tensor or the underlying local
        tensor has ``requires_grad`` set.
        """
        with torch._C.DisableTorchFunctionSubclass():
            if torch.Tensor.requires_grad.__get__(self):
                return True
        return self._local_tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """Set ``requires_grad`` on both the wrapper and the local tensor.

        Only a *leaf* local can have its flag set: a non-leaf's ``requires_grad``
        is implied by its ``grad_fn`` and assigning to it raises "you can only
        change requires_grad flags of leaf variables" (hit under Dynamo
        meta-tensor conversion on newer PyTorch). Skip the write for a non-leaf
        local, and when it is a no-op. Mirrors the ``.is_leaf`` / ``.grad_fn`` /
        ``.grad_dtype`` shielding.
        """
        with torch._C.DisableTorchFunctionSubclass():
            torch.Tensor.requires_grad.__set__(self, value)
        if self._local_tensor.is_leaf and self._local_tensor.requires_grad != value:
            self._local_tensor.requires_grad = value

    def requires_grad_(self, requires_grad: bool = True) -> "ShardTensor":
        """Set ``requires_grad`` in-place on both the wrapper and local tensor.

        Parameters
        ----------
        requires_grad : bool, optional
            Whether to enable gradient tracking. Default is ``True``.

        Returns
        -------
        ShardTensor
            ``self``, for method chaining.
        """
        with torch._C.DisableTorchFunctionSubclass():
            torch.Tensor.requires_grad.__set__(self, requires_grad)
        if self._local_tensor.requires_grad != requires_grad:
            self._local_tensor.requires_grad_(requires_grad)
        return self

    @property  # type: ignore[override]
    def is_leaf(self) -> bool:  # type: ignore[override]
        """Whether this tensor is a leaf in the autograd graph."""
        with torch._C.DisableTorchFunctionSubclass():
            return torch.Tensor.is_leaf.__get__(self)

    @property  # type: ignore[override]
    def grad_fn(self):  # type: ignore[override]
        """Return the stored grad_fn without re-entering ``__torch_function__``.
        Without this override, ``.grad_fn`` (a C-level getset_descriptor on
        ``torch.Tensor``) re-enters ``ShardTensor.__torch_function__``
        whenever someone reads it, falls back via
        :func:`_torch_function_fallback_via_dtensor`, and the fallback
        constructs a *new* temporary DTensor via
        ``_ShardTensorToDTensor.apply(self)`` -- whose ``.grad_fn`` (a
        ``_ShardTensorToDTensorBackward`` ``BackwardCFunction`` instance)
        is what the caller actually receives. On newer PyTorch that
        node's ``.next_functions`` accessor raises a "legacy access
        pattern" error, which is exactly what makes
        ``AOTAutograd.setup_stacktrace_preservation_hooks`` (and our
        own diagnostic ``dump_grad_fn_chain``) fail when they try to
        walk the autograd graph of a ShardTensor output.
        Mirrors the same shielding pattern already used by ``.is_leaf``
        and ``.grad``.
        """
        with torch._C.DisableTorchFunctionSubclass():
            return torch.Tensor.grad_fn.__get__(self)

    @property  # type: ignore[override]
    def grad_dtype(self):  # type: ignore[override]
        """dtype of this tensor's gradient (newer PyTorch): the local tensor's dtype.

        Overriding shields the read from ``__torch_function__``, which would otherwise fall
        back to a non-leaf DTensor whose ``grad_dtype`` getter raises during Dynamo fake
        conversion. Mirrors ``.grad_fn`` / ``.is_leaf`` / ``.grad``."""
        return self._local_tensor.dtype

    @property  # type: ignore[override]
    def grad(self) -> "ShardTensor | None":  # type: ignore[override]
        """Return the accumulated gradient, wrapped as a :class:`ShardTensor`.

        If the autograd engine stored a distributed gradient, its placements
        are preserved rather than inherited from the primal tensor. Plain local
        gradients cannot carry a pending reduction and therefore normalize any
        primal ``Partial`` placement to ``Replicate``. If no gradient has been
        accumulated yet, returns ``None``.
        """
        with torch._C.DisableTorchFunctionSubclass():
            c_grad = torch.Tensor.grad.__get__(self)
        if c_grad is not None:
            if isinstance(c_grad, ShardTensor):
                return c_grad
            if isinstance(c_grad, DTensor):
                grad_spec = _resolve_spec_for_dtensor(c_grad, (self,))
                return _dtensor_to_shard_tensor(c_grad, grad_spec)
            grad_spec = self._spec
            if any(placement.is_partial() for placement in grad_spec.placements):
                grad_spec = dataclasses.replace(
                    grad_spec,
                    placements=tuple(
                        Replicate() if placement.is_partial() else placement
                        for placement in grad_spec.placements
                    ),
                )
            return ShardTensor.__new__(
                ShardTensor,
                local_tensor=c_grad,
                spec=grad_spec,
                requires_grad=False,
            )
        local_grad = self._local_tensor.grad
        if local_grad is None:
            return None
        grad_spec = self._spec
        if any(placement.is_partial() for placement in grad_spec.placements):
            # A plain local gradient is the complete cotangent of this rank's
            # contribution. It has no distributed pending-reduction metadata
            # of its own, so do not inherit Partial from the primal.
            grad_spec = dataclasses.replace(
                grad_spec,
                placements=tuple(
                    Replicate() if placement.is_partial() else placement
                    for placement in grad_spec.placements
                ),
            )
        return ShardTensor.__new__(
            ShardTensor,
            local_tensor=local_grad,
            spec=grad_spec,
            requires_grad=False,
        )

    @grad.setter
    def grad(self, value: "ShardTensor | torch.Tensor | None") -> None:
        """Set or clear the gradient on both the wrapper and local tensor."""
        if value is None:
            with torch._C.DisableTorchFunctionSubclass():
                torch.Tensor.grad.__set__(self, None)
            self._local_tensor.grad = None
        elif isinstance(value, ShardTensor):
            with torch._C.DisableTorchFunctionSubclass():
                torch.Tensor.grad.__set__(self, value)
            self._local_tensor.grad = value._local_tensor
        else:
            with torch._C.DisableTorchFunctionSubclass():
                torch.Tensor.grad.__set__(self, value)
            self._local_tensor.grad = value

    @classmethod
    def from_dtensor(cls, dtensor: DTensor) -> "ShardTensor":
        r"""Convert a DTensor to a ShardTensor.

        Differentiable when *dtensor* is non-leaf (has a ``grad_fn``).
        Spec is inferred from the DTensor (chunk-based, no communication).

        Parameters
        ----------
        dtensor : DTensor
            DTensor to convert.

        Returns
        -------
        ShardTensor
            Equivalent ShardTensor with the same local tensor and inferred spec.
        """
        if isinstance(dtensor, ShardTensor):
            return dtensor
        spec = _resolve_spec_for_dtensor(dtensor)
        if dtensor.grad_fn is not None:
            return _DTensorToShardTensor.apply(dtensor, spec)
        return _dtensor_to_shard_tensor(dtensor, spec)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if _conversion_active():
            # When converting shard tensor to dtensor, or dtensor to shard tensor,
            # we just run the function without ShardTensor dispatch.
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        if func in cls._autograd_passthrough_functions:
            # Run directly on the ShardTensor so the hook/flag binds to the real
            # autograd node rather than a throwaway DTensor (see
            # _autograd_passthrough_functions for why this matters for FSDP2).
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        if func in cls._function_registry and cls._enable_shard_patches:
            args, kwargs = _promote_plain_handler_args(args, kwargs)
            res = cls._function_registry[func](func, types, args, kwargs)
        elif str(func) in cls._named_function_registry and cls._enable_shard_patches:
            args, kwargs = _promote_plain_handler_args(args, kwargs)
            res = cls._named_function_registry[str(func)](func, types, args, kwargs)
        elif _is_tracing(args, kwargs):
            # Under torch.compile tracing, route unpatched ops straight to
            # ``__torch_dispatch__`` (like DTensor) so the aten-level path -- which AOT
            # handles correctly -- owns the op and keeps the graph differentiable.
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        else:
            res = _torch_function_fallback_via_dtensor(func, args, kwargs)
        # Ride subclass routing metadata onto op outputs (eager only; under compile it
        # travels via the flatten context). No-op for base ShardTensor. ``is_compiling()``
        # reads False inside ``__torch_function__``, so gate on ``_is_tracing`` (as above).
        if cls._subclass_propagated_attrs and not _is_tracing(args, kwargs):
            source = cls._find_metadata_source(args, kwargs)
            if source is not None:
                cls._propagate_subclass_metadata(res, source)
        return res

    @classmethod
    def __torch_dispatch__(
        cls,
        func: torch._ops.OpOverload,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> "ShardTensor" | Iterable["ShardTensor"] | object:
        # Use a handler, if we have one:
        handler = cls._dispatch_registry.get(func)
        if handler is None:
            handler = cls._dispatch_registry_by_name.get(str(func))
        if handler is not None:
            return handler(*args, **kwargs)
        # Otherwise, try the dtensor route:
        return _dispatch_fallback_via_dtensor(func, args, kwargs)

    @staticmethod
    def from_local(
        local_tensor: torch.Tensor,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        sharding_shapes: str | dict[int, list[tuple[int, ...]]] = "infer",
        global_shape: tuple[int, ...] | None = None,
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
              global shape (no communication). Requires ``global_shape``.
            - ``"infer"``: Use collective communication to infer shapes from
              mesh neighbors.
            - Manual dict mapping mesh dim to list of shard shapes: Use
              provided shapes. Must pass on each rank.
        global_shape : Optional[Tuple[int, ...]], optional
            Global shape of the full tensor across all ranks. Required when
            ``sharding_shapes="chunk"`` (it is what makes that mode
            communication-free); ignored for ``"infer"`` and dict modes.

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
            global_shape,
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
            Specifying ``Partial`` declares that each returned local gradient
            is an additive contribution with a pending reduction.

        Returns
        -------
        torch.Tensor
            Local tensor. Shape may vary between ranks for sharded tensors.

        Notes
        -----
        A ``Partial`` placement is not resolved: this returns the unreduced
        local contribution. Use :meth:`full_tensor` if you need a reduced value.
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
            Specifying ``Partial`` declares that each returned local gradient
            is an additive contribution with a pending reduction.

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

        if self.grad_fn is not None:
            return torch.Tensor.backward(self, *args, **kwargs)

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

    # Broadcast the tensor to all ranks.
    # scatter_tensor is an input-boundary utility; keep internal collectives/layout
    # transforms out of autograd and construct the requested leaf explicitly.
    if tensor is None and not is_src:
        # Tensor is allowed to be none if not on the root rank
        tensor = torch.empty(local_meta.shape, dtype=local_meta.dtype, device=dm.device)

    with torch.no_grad():
        dist.broadcast(tensor, src=global_src, group=mesh_group)

    # Create a fully-replicated spec:
    spec = ShardTensorSpec(
        mesh=mesh,
        placements=[Replicate() for _ in range(mesh.ndim)],
        tensor_meta=local_meta,
        _sharding_shapes={},
    )

    with torch.no_grad():
        # Build a replicated ShardTensor and redistribute to the requested
        # placements without recording autograd history.
        st = ShardTensor.__new__(
            ShardTensor,
            local_tensor=tensor,
            spec=spec,
            requires_grad=False,
        )
        st = st.redistribute(mesh, placements, async_op=False)

    if requires_grad:
        # 1. Ensure the local data is a clean leaf
        local_leaf = st._local_tensor.detach().requires_grad_(True)

        # 2. Create the ShardTensor wrapper
        st = ShardTensor.__new__(
            ShardTensor,
            local_tensor=local_leaf,
            spec=st._spec,
            requires_grad=True,
        )

        # 3. CRITICAL: Force the wrapper itself to be a leaf in the autograd graph
        st = st.detach().requires_grad_(True)

    return st


def install_aot_plain_tangent_coercion() -> None:
    r"""Patch ``AOTDispatchAutograd.process_runtime_tangent`` so a plain runtime backward
    tangent is rebuilt into a :class:`ShardTensor` when the graph traced a ShardTensor
    tangent there.

    AOT can coerce a runtime *subclass* tangent but has no hook for a *plain* tensor where
    the graph traced a subclass, so it raises "guessed its metadata incorrectly" when a
    ShardTensor boundary crosses a graph break. The plain cotangent is value-correct at a
    ``Replicate`` boundary, so rebuilding via ``__tensor_unflatten__`` is lossless. Scoped to
    ShardTensor, idempotent, degrades gracefully; a general AOT gap pending an upstream hook."""
    try:
        from torch._functorch._aot_autograd.runtime_wrappers import (
            AOTDispatchAutograd,
        )
        from torch._functorch._aot_autograd.schemas import SubclassCreationMeta
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass
    except Exception:  # pragma: no cover - torch internals moved / unavailable
        return

    _orig = AOTDispatchAutograd.process_runtime_tangent
    if getattr(_orig, "_shardtensor_plain_tangent_shim", False):
        return

    def _process_runtime_tangent(x, meta, *args, **kwargs):
        # ``*args``/``**kwargs`` forward extra params across torch versions; only (x, meta) matter.
        subclass_type = getattr(meta, "original_subclass_type", None)
        if (
            isinstance(x, torch.Tensor)
            and not is_traceable_wrapper_subclass(x)
            and isinstance(meta, SubclassCreationMeta)
            and isinstance(subclass_type, type)
            and issubclass(subclass_type, ShardTensor)
            and "_local_tensor" in getattr(meta, "attrs", ())
        ):
            try:
                inner = {
                    name: (
                        x
                        if name == "_local_tensor"
                        else torch.zeros(0, dtype=torch.int64, device=x.device)
                    )
                    for name in meta.attrs
                }
                x = subclass_type.__tensor_unflatten__(
                    inner, meta.meta, meta.outer_size, meta.outer_stride
                )
            except Exception:  # noqa: S110  # pragma: no cover - graceful fallback
                pass
        return _orig(x, meta, *args, **kwargs)

    _process_runtime_tangent._shardtensor_plain_tangent_shim = True
    AOTDispatchAutograd.process_runtime_tangent = staticmethod(_process_runtime_tangent)


# Install on import so a ShardTensor survives a compiled backward whose cotangent is a plain
# tensor across a graph break. Scoped and graceful, so inert for other compilations.
install_aot_plain_tangent_coercion()
