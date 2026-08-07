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

r"""Reusable tooling for domain-parallel model tests.

Adding a domain-parallel test for a new model should be a few declarative lines:
describe how to build the model, how to build its (full, plain) inputs, and how
to shard those inputs, then hand a :class:`DomainParallelModelCase` to
:func:`run_domain_parallel_model_check`. The driver takes care of distributing
the model, running the distributed and single-GPU reference forward/backward, and
comparing outputs and gradients.

Distribution strategies
------------------------
Unlike ``distribute_module`` (which converts *every* parameter/buffer to a
``DTensor`` and does not match production), these helpers keep the bulk of the
weights as plain tensors and rely on ``ShardTensor`` auto-promotion when a plain
weight meets a domain-sharded activation:

- ``"ddp"``: wrap in :class:`~torch.nn.parallel.DistributedDataParallel`. All
  params stay plain and replicated; DDP broadcasts them from rank 0 and
  all-reduces their gradients. Suitable for models whose parameters are all
  replicated across the domain mesh.
- ``"fsdp_spatial"``: pre-shard only the genuinely spatial params/buffers
  (selected by :func:`default_spatial_param_selector`) as ``DTensor`` on the
  domain mesh, then ``fully_shard`` (FSDP2) the rest over the data-parallel mesh.
  This mirrors ``ParallelHelper.distribute_model`` in the StormCast recipe and is
  required by models with height-sharded positional/RoPE buffers (e.g. DiT).

The spatial-parameter selector is intentionally kept local to the test harness
for now (a copy of the production ``shard_dim_selector``), rather than promoted
into ``physicsnemo.domain_parallel``.
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Shard
from torch.nn.parallel import DistributedDataParallel

from physicsnemo.distributed import DistributedManager
from test.domain_parallel.ops.utils import (
    default_loss_fn,
    numerical_shard_tensor_check,
)

Strategy = Literal["ddp", "fsdp_spatial", "distribute_module"]


def default_spatial_param_selector(name: str) -> int | None:
    r"""Return the dim along which a spatial parameter/buffer should be sharded.

    A local copy of the production ``shard_dim_selector``
    (``examples/weather/stormcast/utils/parallel.py``): positional embeddings
    laid out as :math:`(1, H W, C)` shard the flattened spatial axis (dim 1),
    while DiT RoPE cos/sin tables laid out :math:`(H, W, d)` shard height (dim 0)
    so each rank owns globally-correct rows. Everything else returns ``None``
    (left replicated / plain).

    Parameters
    ----------
    name : str
        The (unqualified) parameter or buffer name.

    Returns
    -------
    int or None
        The shard dimension, or ``None`` when the tensor should not be sharded.
    """
    if any(key in name for key in ("pos_embed", "pos_embd", "spatial_emb")):
        return 1
    if any(key in name for key in ("rope_cos", "rope_sin")):
        return 0
    return None


def shard_spatial_params_(
    model: nn.Module,
    domain_mesh: DeviceMesh,
    selector: Callable[[str], int | None] = default_spatial_param_selector,
) -> nn.Module:
    r"""Pre-shard spatial params/buffers on the domain mesh, in place.

    Walks every submodule and, for each parameter/buffer whose name matches
    ``selector``, replaces it with a domain-mesh ``DTensor`` sharded along the
    selected dim. All other params/buffers are left untouched (plain tensors) so
    ``ShardTensor`` auto-promotion and FSDP2 handle them. Mirrors
    ``ParallelHelper._shard_spatial_params``.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose spatial params/buffers are sharded in place.
    domain_mesh : DeviceMesh
        The domain mesh to shard across.
    selector : Callable[[str], int | None], optional
        Maps a param/buffer name to a shard dim (or ``None``).

    Returns
    -------
    torch.nn.Module
        The same ``model``, mutated in place.
    """
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            shard_dim = selector(name)
            if shard_dim is None:
                continue
            dist_param = nn.Parameter(
                distribute_tensor(param.data, domain_mesh, [Shard(shard_dim)]),
                requires_grad=param.requires_grad,
            )
            module.register_parameter(name, dist_param)
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf is None:
                continue
            shard_dim = selector(name)
            if shard_dim is None:
                continue
            persistent = name not in module._non_persistent_buffers_set
            module.register_buffer(
                name,
                distribute_tensor(buf, domain_mesh, [Shard(shard_dim)]),
                persistent=persistent,
            )
    return model


def wrap_ddp(
    model: nn.Module,
    mesh: DeviceMesh,
    *,
    find_unused_parameters: bool = True,
) -> DistributedDataParallel:
    r"""Wrap ``model`` in DDP over ``mesh``'s process group.

    Parameters stay plain (replicated); DDP broadcasts them from rank 0 at
    construction and all-reduces their gradients in backward. Domain-sharded
    activations are handled by ``ShardTensor`` auto-promotion inside the model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to wrap. Must already be on the local CUDA device.
    mesh : DeviceMesh
        The (1D) mesh whose process group DDP reduces over.
    find_unused_parameters : bool, default=False
        Forwarded to DDP; enable only when a config leaves some params unused.

    Returns
    -------
    torch.nn.parallel.DistributedDataParallel
        The DDP-wrapped model.
    """
    dm = DistributedManager()
    return DistributedDataParallel(
        model,
        device_ids=[dm.local_rank],
        process_group=mesh.get_group(),
        find_unused_parameters=find_unused_parameters,
    )


def wrap_fsdp_spatial(
    model: nn.Module,
    *,
    ddp_mesh: DeviceMesh,
    domain_mesh: DeviceMesh,
    selector: Callable[[str], int | None] = default_spatial_param_selector,
) -> nn.Module:
    r"""Shard spatial params on the domain mesh, then FSDP2 over the ddp mesh.

    The production pattern (``ParallelHelper.distribute_model``): only the
    genuinely spatial params/buffers become domain-mesh ``DTensor``s; the rest
    stay plain and are sharded across the data-parallel mesh by ``fully_shard``.

    Parameters
    ----------
    model : torch.nn.Module
        Model to distribute (mutated in place).
    ddp_mesh : DeviceMesh
        Data-parallel mesh for FSDP2.
    domain_mesh : DeviceMesh
        Domain mesh for the spatial parameter sharding.
    selector : Callable[[str], int | None], optional
        Spatial parameter selector.

    Returns
    -------
    torch.nn.Module
        The FSDP2-wrapped model.
    """
    shard_spatial_params_(model, domain_mesh, selector)
    # FSDP2 rejects non-contiguous parameters; make them contiguous first.
    with torch.no_grad():
        for p in model.parameters():
            if not p.is_contiguous():
                p.data = p.data.contiguous()
    fully_shard(model, mesh=ddp_mesh)
    return model


def distribute_model_for_test(
    model: nn.Module,
    strategy: Strategy,
    *,
    mesh: DeviceMesh | None = None,
    ddp_mesh: DeviceMesh | None = None,
    domain_mesh: DeviceMesh | None = None,
    selector: Callable[[str], int | None] = default_spatial_param_selector,
    find_unused_parameters: bool = True,
) -> nn.Module:
    r"""Distribute ``model`` according to ``strategy``.

    Parameters
    ----------
    model : torch.nn.Module
        Model to distribute.
    strategy : {"ddp", "fsdp_spatial", "distribute_module"}
        Distribution strategy (see module docstring).
    mesh : DeviceMesh, optional
        The mesh used by the ``"ddp"`` and ``"distribute_module"`` strategies.
    ddp_mesh : DeviceMesh, optional
        Data-parallel mesh for ``"fsdp_spatial"``.
    domain_mesh : DeviceMesh, optional
        Domain mesh for ``"fsdp_spatial"`` spatial-param sharding.
    selector : Callable[[str], int | None], optional
        Spatial parameter selector for ``"fsdp_spatial"``.
    find_unused_parameters : bool, default=False
        Forwarded to DDP.

    Returns
    -------
    torch.nn.Module
        The distributed model.
    """
    if strategy == "ddp":
        if mesh is None:
            raise ValueError("strategy='ddp' requires `mesh`")
        return wrap_ddp(model, mesh, find_unused_parameters=find_unused_parameters)
    if strategy == "fsdp_spatial":
        if ddp_mesh is None or domain_mesh is None:
            raise ValueError(
                "strategy='fsdp_spatial' requires `ddp_mesh` and `domain_mesh`"
            )
        return wrap_fsdp_spatial(
            model, ddp_mesh=ddp_mesh, domain_mesh=domain_mesh, selector=selector
        )
    if strategy == "distribute_module":
        from torch.distributed.tensor import distribute_module

        if mesh is None:
            raise ValueError("strategy='distribute_module' requires `mesh`")
        return distribute_module(model, device_mesh=mesh)
    raise ValueError(f"Unknown distribution strategy: {strategy!r}")


@dataclass
class DomainParallelModelCase:
    r"""Declarative description of a domain-parallel model test.

    Parameters
    ----------
    name : str
        Human-readable case id (used for pytest parametrization).
    build_model : Callable[[torch.device], torch.nn.Module]
        Builds the model on the given device. The autouse seed fixture makes
        construction identical across ranks.
    build_inputs : Callable[[torch.device], tuple[tuple, dict]]
        Builds the full (plain, unsharded) positional args and kwargs.
    shard_inputs : Callable[[tuple, dict, DeviceMesh], tuple[tuple, dict]]
        Shards the args/kwargs across the domain mesh (scalars may stay plain).
    strategy : {"ddp", "fsdp_spatial"}, default="ddp"
        Distribution strategy.
    spatial_param_selector : Callable[[str], int | None], optional
        Spatial parameter selector for the ``"fsdp_spatial"`` strategy.
    output_check_fn : Callable[[Any], None] or None, optional
        Optional callback to assert output shape / placements. Must be collective.
    loss_fn : Callable[[Any], torch.Tensor], default=default_loss_fn
        Scalar loss used for the gradient check (must accept the model output,
        including tuple outputs).
    check_grads : bool, default=True
        Whether to run and compare backward gradients.
    find_unused_parameters : bool, default=False
        Forwarded to DDP for configs with unused parameters.
    atol, rtol : float, default=1e-4
        Comparison tolerances.
    """

    name: str
    build_model: Callable[[torch.device], nn.Module]
    build_inputs: Callable[[torch.device], tuple[tuple, dict]]
    shard_inputs: Callable[[tuple, dict, DeviceMesh], tuple[tuple, dict]]
    strategy: Literal["ddp", "fsdp_spatial"] = "ddp"
    spatial_param_selector: Callable[[str], int | None] = default_spatial_param_selector
    output_check_fn: Callable[[Any], None] | None = None
    loss_fn: Callable[[Any], torch.Tensor] = default_loss_fn
    check_grads: bool = True
    find_unused_parameters: bool = True
    atol: float = 1e-4
    rtol: float = 1e-4


def run_domain_parallel_model_check(
    case: DomainParallelModelCase,
    *,
    mesh: DeviceMesh | None = None,
    ddp_mesh: DeviceMesh | None = None,
    domain_mesh: DeviceMesh | None = None,
) -> None:
    r"""Run a full domain-parallel forward/backward equivalence check for a case.

    Builds the model, shards its inputs, distributes it via ``case.strategy``,
    and delegates to :func:`numerical_shard_tensor_check`, which compares the
    distributed forward output and (optionally) gradients against a single-GPU
    reference built from the same weights and gathered inputs.

    Parameters
    ----------
    case : DomainParallelModelCase
        The case to run.
    mesh : DeviceMesh, optional
        The 1D domain mesh for the ``"ddp"`` strategy. Also used to shard inputs
        when ``domain_mesh`` is not given.
    ddp_mesh : DeviceMesh, optional
        Data-parallel submesh for the ``"fsdp_spatial"`` strategy.
    domain_mesh : DeviceMesh, optional
        Domain submesh for the ``"fsdp_spatial"`` strategy (also used to shard
        inputs). Falls back to ``mesh`` when not given.
    """
    dm = DistributedManager()
    device = dm.device

    # Mesh used to scatter the inputs and to run collective assertions.
    shard_mesh = domain_mesh if domain_mesh is not None else mesh
    if shard_mesh is None:
        raise ValueError("run_domain_parallel_model_check requires a mesh")

    model = case.build_model(device)
    # eval() so dropout/other stochastic layers are deterministic across the
    # distributed and single-GPU reference runs.
    model.eval()

    args, kwargs = case.build_inputs(device)
    d_args, d_kwargs = case.shard_inputs(args, kwargs, shard_mesh)

    def distribute_fn(module: nn.Module) -> nn.Module:
        return distribute_model_for_test(
            module,
            case.strategy,
            mesh=mesh,
            ddp_mesh=ddp_mesh,
            domain_mesh=domain_mesh,
            selector=case.spatial_param_selector,
            find_unused_parameters=case.find_unused_parameters,
        )

    numerical_shard_tensor_check(
        mesh=shard_mesh,
        module=model,
        input_args=d_args,
        input_kwargs=d_kwargs,
        check_grads=case.check_grads,
        loss_fn=case.loss_fn,
        atol=case.atol,
        rtol=case.rtol,
        group=shard_mesh.get_group(),
        distribute_fn=distribute_fn,
        output_check_fn=case.output_check_fn,
    )
