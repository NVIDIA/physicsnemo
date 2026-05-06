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

"""Checkpoint utilities for saving and loading training state.

Provides :func:`save_checkpoint` and :func:`load_checkpoint` for persisting
and restoring model weights, optimizer/scheduler/scaler state, and arbitrary
metadata.  Supports local filesystems and remote stores via ``fsspec``.

When models are wrapped with FSDP or use DTensor/ShardTensor parameters,
:func:`save_checkpoint` and :func:`load_checkpoint` automatically use
PyTorch's distributed checkpoint state-dict APIs to gather and scatter
model and optimizer state.  In this *distributed* mode **all ranks** must
call the functions (the collective operations inside the DCP helpers require
it), while only rank 0 performs actual file I/O.
"""

import io
import os
import re
import tarfile
import zipfile
from pathlib import Path, PurePath
from typing import Any

import fsspec
import fsspec.utils
import torch
from torch.amp import GradScaler
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.optim.lr_scheduler import LRScheduler

import physicsnemo
from physicsnemo.core.filesystem import LOCAL_CACHE, _download_cached
from physicsnemo.core.module import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.capture import _StaticCapture
from physicsnemo.utils.logging import PythonLogger

checkpoint_logging = PythonLogger("checkpoint")


# ---------------------------------------------------------------------------
# Distributed-model detection helpers
# ---------------------------------------------------------------------------


def _is_distributed_model(model: torch.nn.Module) -> bool:
    """Return ``True`` when *model* is FSDP-wrapped or has DTensor params."""
    if isinstance(model, FSDP):
        return True
    return any(isinstance(p, DTensor) for p in model.parameters())


def _unwrap_ddp_compile(
    model: torch.nn.Module, loading: bool = False
) -> torch.nn.Module:
    """Strip DDP / DataParallel / ``torch.compile`` wrappers, keep FSDP."""
    if isinstance(
        model,
        (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
    ):
        model = model.module
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        if loading:
            checkpoint_logging.warning(
                f"Model {type(model._orig_mod).__name__} is already compiled, "
                "consider loading first and then compiling."
            )
        model = model._orig_mod
    return model


def _unwrap_fsdp(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap one FSDP layer (if present) to reach the user module."""
    if isinstance(model, FSDP):
        return model.module
    return model


def _cpu_offload_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Move every tensor in *state_dict* to CPU (shallow copy)."""
    out: dict[str, Any] = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cpu()
        elif isinstance(v, dict):
            out[k] = _cpu_offload_state_dict(v)
        else:
            out[k] = v
    return out


def _force_standard_contiguous(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Make positive-dim, non-DTensor tensors standard-contiguous.

    Compensates for a layout-blind broadcast inside DCP's
    ``set_model_state_dict`` / ``set_optimizer_state_dict``: the rank-0→others
    transfer goes through ``dist.broadcast``, which is happy to send a
    ``channels_last`` tensor (its contiguity check passes for that format) but
    moves bytes in *storage* order. Receivers on non-zero ranks allocate via
    ``torch.empty(shape, dtype, device)`` (standard NCHW), so CL bytes land
    in NCHW positions and values are silently permuted on receive.

    Forcing standard contiguity on rank 0 makes the broadcast layout-consistent.
    ``.contiguous()`` is a no-op for tensors that are already standard-contig,
    a value-preserving (logical) copy for ``channels_last`` /
    ``channels_last_3d``, and is intentionally skipped for ``DTensor`` (whose
    contiguity semantics differ).
    """
    out: dict[str, Any] = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and not isinstance(v, DTensor) and v.dim() > 0:
            out[k] = v.contiguous()
        elif isinstance(v, dict):
            out[k] = _force_standard_contiguous(v)
        else:
            out[k] = v
    return out


def _get_dtensor_param_placements(
    model: torch.nn.Module,
) -> dict[str, tuple[Any, tuple[Any, ...]]]:
    """Map parameter names to ``(device_mesh, placements)`` for DTensor params.

    Uses ``get_model_state_dict`` with native (non-full) format so that the
    DCP layer unflattens FlatParameters back to original names and preserves
    each parameter's DTensor placement.  This works correctly for both
    ``use_orig_params=True`` and ``use_orig_params=False``.

    **Collective** — all ranks must call this together.
    """
    native_sd = get_model_state_dict(model, options=StateDictOptions())
    info: dict[str, tuple[Any, tuple[Any, ...]]] = {}
    for name, value in native_sd.items():
        if isinstance(value, DTensor):
            info[name] = (value.device_mesh, tuple(value.placements))
    return info


def _has_non_fsdp_dtensors(
    model: torch.nn.Module,
    dtensor_plc: dict[str, tuple[Any, tuple[Any, ...]]],
) -> bool:
    """Return ``True`` when *dtensor_plc* contains placements not managed by FSDP.

    FSDP with ``FULL_SHARD`` or ``SHARD_GRAD_OP`` wraps parameters as
    DTensors on its own mesh.  ``broadcast_from_rank0`` handles these
    natively, so manual redistribution should be skipped.  Only
    user-created DTensors (e.g. ShardTensor on a separate domain mesh)
    require explicit redistribution.
    """
    if not dtensor_plc:
        return False
    if not isinstance(model, FSDP):
        return True
    if model.sharding_strategy == ShardingStrategy.NO_SHARD:
        return True
    return False


def _redistribute_sd_for_dtensor(
    placements: dict[str, tuple[Any, tuple[Any, ...]]],
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Convert plain tensors in *state_dict* to DTensors matching *placements*.

    Entries whose key appears in *placements* are converted via
    ``distribute_tensor`` so that each rank receives its correct local shard.
    """

    target_device = next(iter(placements.values()))[0].device_type
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or isinstance(value, DTensor):
            out[key] = value
            continue

        if key in placements:
            mesh, plc = placements[key]
            out[key] = distribute_tensor(value.to(mesh.device_type), mesh, list(plc))
        else:
            out[key] = value.to(target_device)
    return out


def _redistribute_optim_sd_for_dtensor(
    placements: dict[str, tuple[Any, tuple[Any, ...]]],
    optim_sd: dict[str, Any],
) -> dict[str, Any]:
    """Shard optimizer state tensors to local chunks matching model placements.

    FSDP's ``optim_state_dict_to_load`` expects each optimizer state tensor
    (``exp_avg``, ``exp_avg_sq``, …) to be a **plain tensor** whose shape
    matches the parameter's *local* shape — not a DTensor.  We use
    ``distribute_tensor(...).to_local()`` to extract each rank's shard.
    Scalar state entries (e.g. ``step``) are left unchanged.
    """
    if "state" not in optim_sd:
        return optim_sd

    target_device = next(iter(placements.values()))[0].device_type

    new_state: dict[str, Any] = {}
    for param_name, param_state in optim_sd["state"].items():
        if not isinstance(param_state, dict):
            new_state[param_name] = param_state
            continue

        new_ps: dict[str, Any] = {}
        mesh_plc = placements.get(param_name)
        for k, v in param_state.items():
            if (
                not isinstance(v, torch.Tensor)
                or isinstance(v, DTensor)
                or v.dim() == 0
            ):
                new_ps[k] = v
            elif mesh_plc is not None:
                mesh, plc = mesh_plc
                new_ps[k] = distribute_tensor(
                    v.to(mesh.device_type), mesh, list(plc)
                ).to_local()
            else:
                new_ps[k] = v.to(target_device)
        new_state[param_name] = new_ps

    return {**optim_sd, "state": new_state}


def _fsdp_uses_flat_param_optim(model: torch.nn.Module | None) -> bool:
    """Return True if *model* is FSDP-wrapped with ``use_orig_params=False``.

    That is the only configuration in which optimizer state goes through
    the FlatParameter flatten/unflatten path (and therefore the only
    configuration affected by the channels_last asymmetry). With
    ``use_orig_params=True`` FSDP exposes per-original-param optimizer
    state and the round-trip is layout-preserving on its own.
    """
    if not isinstance(model, FSDP):
        return False
    return not getattr(model, "_use_orig_params", True)


def _strides_match_channels_last(
    shape: tuple[int, ...] | torch.Size,
    stride: tuple[int, ...],
) -> bool:
    """True iff *stride* matches the canonical NHWC / NDHWC stride formula.

    For 4-D ``(N, C, H, W)`` the channels_last layout has strides
    ``(C*H*W, 1, W*C, C)``; for 5-D ``(N, C, D, H, W)`` channels_last_3d
    has strides ``(D*H*W*C, 1, H*W*C, W*C, C)``. Anything else -- including
    standard-contig tensors that happen to satisfy ``stride[1] == 1`` because
    of trailing-1 dims -- returns False.
    """
    if len(shape) != len(stride):
        return False
    if len(shape) == 4:
        n, c, h, w = shape
        return tuple(stride) == (c * h * w, 1, w * c, c)
    if len(shape) == 5:
        n, c, d, h, w = shape
        return tuple(stride) == (d * h * w * c, 1, h * w * c, w * c, c)
    return False


def _get_cl_param_fqns(opt_model: torch.nn.Module | None) -> set[str]:
    """Return FQNs of FSDP-managed original params recorded as channels_last.

    For every FSDP submodule in *opt_model*, reads ``flat_param._fqns`` /
    ``_shapes`` / ``_strides`` / ``_contiguities`` and returns the set of
    original-parameter FQNs whose ``_contiguities[i] is False`` and whose
    recorded strides match the canonical ``channels_last`` (4-D) or
    ``channels_last_3d`` (5-D) formula. That is the same bit
    ``_get_unflat_views`` consults to decide ``view`` vs ``as_strided`` on
    save -- so ``_contiguities[i] is False`` plus a CL stride pattern is
    exactly the signal that the destination ``FlatParameter`` slot expects
    NHWC storage order at load time.

    Returns an empty set when *opt_model* isn't FSDP+``use_orig_params=False``
    (the only configuration where the flatten/unflatten asymmetry exists).

    Each FQN is built as ``{module_path_to_FSDP}.{flat_param._fqns[i]}``,
    matching DCP's ``_get_fqns`` convention -- specifically, FSDP's
    ``_fsdp_wrapped_module`` segments are stripped from the path so the
    returned FQNs line up with the keys in ``optim_sd["state"]``.

    The ``_orig_mod.`` (``torch.compile``) prefix is also stripped, matching
    the normalization ``save_checkpoint`` applies to optimizer
    ``param_names``.
    """
    if not _fsdp_uses_flat_param_optim(opt_model):
        return set()

    cl_fqns: set[str] = set()
    for module_name, module in opt_model.named_modules():
        if not isinstance(module, FSDP):
            continue
        flat_param = getattr(module, "_flat_param", None)
        if flat_param is None:
            continue
        # DCP's ``_get_fqns`` skips the ``_fsdp_wrapped_module`` attribute
        # when building parameter FQNs; mirror that by removing the segment
        # from the module path.
        path_segments = [
            seg
            for seg in module_name.split(".")
            if seg and seg != "_fsdp_wrapped_module"
        ]
        prefix = ".".join(path_segments)
        if prefix:
            prefix += "."
        for fqn, shape, stride, contig in zip(
            flat_param._fqns,
            flat_param._shapes,
            flat_param._strides,
            flat_param._contiguities,
        ):
            if contig:
                continue
            if _strides_match_channels_last(shape, stride):
                cl_fqns.add((prefix + fqn).removeprefix("_orig_mod."))
    return cl_fqns


def _remap_channels_last_optim_sd(
    opt_model: torch.nn.Module | None,
    optim_sd: dict[str, Any],
) -> dict[str, Any]:
    """Compensate for an FSDP optim-state flatten/unflatten asymmetry.

    PyTorch FSDP with ``use_orig_params=False`` packs and unpacks the
    ``FlatParameter`` asymmetrically for non-truly-contiguous params:

    * Save path (``_get_unflat_views``) uses
      ``as_strided((numel,), (1,))`` -- bytes are read in *storage* order.
    * Load path (``_flatten_tensor_optim_state``) uses ``torch.flatten``
      -- bytes are read in *logical* (row-major) order.

    For a 4-D Conv2d weight in ``channels_last`` format the two orders
    differ, so the round-trip silently corrupts the optimizer state.

    The remap is gated on the **destination** ``FlatParameter`` slot's
    expected byte order (via ``flat_param._contiguities``), not on the
    saved tensor's layout. That's the only signal that always matches what
    the load-side ``_flatten_tensor_optim_state`` will do, so it works for
    every save/load layout combination -- in particular for
    ``FSDP+ShardTensor`` configurations where ``distribute_module`` calls
    ``.contiguous()`` and silently strips channels_last before FSDP wraps,
    making saved tensors standard-contig even though the conceptual model
    has CL conv weights.

    Inputs are also normalized to standard contiguity before the layout
    decision: a CL tensor that *isn't* getting permuted (because the
    destination is non-CL) would otherwise survive into DCP's per-tensor
    ``dist.broadcast`` and hit the same layout-blind broadcast bug
    ``_force_standard_contiguous`` fixes for model state.

    Only fires when *opt_model* is FSDP-wrapped with
    ``use_orig_params=False`` -- with ``use_orig_params=True`` the
    asymmetry doesn't exist and the remap would *cause* the corruption it
    is meant to prevent.

    See ``torch/distributed/fsdp/_optim_utils.py::_flatten_tensor_optim_state``
    and ``_flat_param.py::flatten_tensors``.
    """
    if "state" not in optim_sd:
        return optim_sd
    if not _fsdp_uses_flat_param_optim(opt_model):
        return optim_sd

    cl_fqns = _get_cl_param_fqns(opt_model)

    def _normalize(t: torch.Tensor, is_cl_dest: bool) -> torch.Tensor:
        if isinstance(t, DTensor) or t.dim() == 0:
            return t
        # Force standard contiguity first so any saved-CL bytes are
        # rewritten in NCHW storage order before the layout decision; this
        # makes the subsequent broadcast inside DCP layout-safe whether or
        # not we permute.
        t = t.contiguous()
        if not is_cl_dest:
            return t
        if t.dim() == 4:
            return t.permute(0, 2, 3, 1).contiguous().view(*t.shape)
        if t.dim() == 5:
            return t.permute(0, 2, 3, 4, 1).contiguous().view(*t.shape)
        return t

    new_state: dict[str, Any] = {}
    for pname, pstate in optim_sd["state"].items():
        if not isinstance(pstate, dict):
            new_state[pname] = pstate
            continue
        is_cl_dest = pname.removeprefix("_orig_mod.") in cl_fqns
        new_ps: dict[str, Any] = {}
        for k, v in pstate.items():
            new_ps[k] = _normalize(v, is_cl_dest) if isinstance(v, torch.Tensor) else v
        new_state[pname] = new_ps

    return {**optim_sd, "state": new_state}


def _is_mdlus_archive(path: str) -> bool:
    """Return ``True`` if *path* is a ``.mdlus`` archive (tar or zip containing ``model.pt``)."""
    cached = _cache_if_needed(path)
    if tarfile.is_tarfile(cached):
        with tarfile.open(cached, "r") as tar:
            return "model.pt" in tar.getnames()
    if zipfile.is_zipfile(cached):
        with zipfile.ZipFile(cached, "r") as archive:
            return "model.pt" in archive.namelist()
    return False


def _extract_mdlus_state_dict(
    file_name: str, device: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Read only the ``state_dict`` from a ``.mdlus`` archive."""
    cached = _cache_if_needed(file_name)
    fmt = Module._detect_checkpoint_format(cached)

    if fmt == "tar":
        with tarfile.open(cached, "r") as tar:
            f = tar.extractfile("model.pt")
            return torch.load(
                io.BytesIO(f.read()), map_location=device, weights_only=False
            )
    else:
        with zipfile.ZipFile(cached, "r") as archive:
            model_bytes = archive.read("model.pt")
        return torch.load(
            io.BytesIO(model_bytes), map_location=device, weights_only=False
        )


def _get_checkpoint_filename(
    path: str,
    base_name: str = "checkpoint",
    index: int | None = None,
    saving: bool = False,
    model_type: str = "mdlus",
    distributed: bool = False,
) -> str:
    r"""Build the filename for a numbered checkpoint.

    Resolution logic:

    * **Explicit index** (``index`` is not ``None``): returns that exact
      checkpoint path.
    * **Latest** (``index is None``, ``saving=False``): scans for existing
      checkpoints and returns the one with the largest index.
    * **Next** (``index is None``, ``saving=True``): returns the path for
      the *next* index after the largest existing one.

    When no existing checkpoints are found, the returned path uses index 0.

    Parameters
    ----------
    path : str
        Directory containing checkpoint files.
    base_name : str, optional
        Stem used in the filename, by default ``"checkpoint"``.
    index : int | None, optional
        Specific checkpoint index to use.  When ``None``, the latest or
        next index is determined automatically.
    saving : bool, optional
        If ``True`` (and ``index is None``), return the *next* available
        filename rather than the latest existing one.  By default ``False``.
    model_type : str, optional
        ``"mdlus"`` for :class:`~physicsnemo.core.Module` models,
        ``"pt"`` for vanilla PyTorch models.  Determines the file
        extension.  By default ``"mdlus"``.
    distributed : bool, optional
        When ``True`` the model_parallel_rank component of the filename is
        forced to ``0`` because FSDP/DTensor distribution is handled by the
        DCP APIs, not per-rank files.  By default ``False``.

    Returns
    -------
    str
        Fully qualified checkpoint filename.
    """
    # Get model parallel rank so all processes in the first model parallel group
    # can save their checkpoint. In the case without model parallelism,
    # model_parallel_rank should be the same as the process rank itself and
    # only rank 0 saves
    if not DistributedManager.is_initialized():
        checkpoint_logging.warning(
            "`DistributedManager` not initialized already. Initializing now, but this might lead to unexpected errors"
        )
        DistributedManager.initialize()
    manager = DistributedManager()
    if distributed:
        model_parallel_rank = 0
    else:
        model_parallel_rank = (
            manager.group_rank("model_parallel")
            if "model_parallel" in manager.group_names
            else 0
        )

    # Determine input file name. Get absolute file path if Posix path.
    # pathlib does not support custom schemes (eg: msc://...) so only perform resolve() for Posix.
    protocol = fsspec.utils.get_protocol(path)
    fs = fsspec.filesystem(protocol)
    if protocol == "file":
        path = str(Path(path).resolve())
    checkpoint_filename = f"{path}/{base_name}.{model_parallel_rank}"

    # File extension for PhysicsNeMo models or PyTorch models
    file_extension = ".mdlus" if model_type == "mdlus" else ".pt"

    # If epoch is provided load that file
    if index is not None:
        checkpoint_filename = checkpoint_filename + f".{index}"
        checkpoint_filename += file_extension
    # Otherwise try loading the latest epoch or rolling checkpoint
    else:
        file_names = [
            fname for fname in fs.glob(checkpoint_filename + "*" + file_extension)
        ]

        if len(file_names) > 0:
            # If checkpoint from a null index save exists load that
            # This is the most likely line to error since it will fail with
            # invalid checkpoint names

            file_idx = []

            for fname in file_names:
                fname_path = PurePath(fname)
                file_stem = fname_path.name

                pattern = rf"^{re.escape(base_name)}\.{model_parallel_rank}\.(\d+){re.escape(file_extension)}$"
                match = re.match(pattern, file_stem)
                if match:
                    file_idx.append(int(match.group(1)))
            file_idx.sort()
            # If we are saving index by 1 to get the next free file name
            if saving:
                checkpoint_filename = checkpoint_filename + f".{file_idx[-1] + 1}"
            else:
                checkpoint_filename = checkpoint_filename + f".{file_idx[-1]}"
            checkpoint_filename += file_extension
        else:
            checkpoint_filename += ".0" + file_extension

    return checkpoint_filename


def _unique_model_names(
    models: list[torch.nn.Module],
    loading: bool = False,
) -> dict[str, torch.nn.Module]:
    r"""Map a list of models to unique names derived from their class names.

    DDP and ``torch.compile`` wrappers are stripped, but FSDP wrappers are
    preserved so that the returned modules can be passed to PyTorch's DCP
    state-dict helpers when needed.

    When multiple models share a class name a numeric suffix is appended
    (e.g. ``"MyModel0"``, ``"MyModel1"``).

    Parameters
    ----------
    models : list[torch.nn.Module]
        Models to generate names for.
    loading : bool, optional
        When ``True``, emits a warning if a model is already compiled
        (loading into a compiled model can cause issues).  By default
        ``False``.

    Returns
    -------
    dict[str, torch.nn.Module]
        Mapping from unique name to module (with FSDP intact if present).
    """
    model_dict: dict[str, list[torch.nn.Module]] = {}
    for model0 in models:
        model0 = _unwrap_ddp_compile(model0, loading=loading)
        base_name = type(_unwrap_fsdp(model0)).__name__
        if base_name in model_dict:
            model_dict[base_name].append(model0)
        else:
            model_dict[base_name] = [model0]

    output_dict: dict[str, torch.nn.Module] = {}
    for key, model_list in model_dict.items():
        if len(model_list) > 1:
            for i, m in enumerate(model_list):
                output_dict[key + str(i)] = m
        else:
            output_dict[key] = model_list[0]

    return output_dict


def save_checkpoint(
    path: Path | str,
    models: torch.nn.Module | list[torch.nn.Module] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: GradScaler | None = None,
    epoch: int | None = None,
    metadata: dict[str, Any] | None = None,
    optimizer_model: torch.nn.Module | None = None,
) -> None:
    r"""Save a training checkpoint to disk (or a remote store).

    Up to two categories of files are created inside ``path``:

    * **Model weights** (when ``models`` is provided) - one file per model:
      ``{class_name}{id}.{mp_rank}.{epoch}.{ext}`` where *ext* is
      ``.mdlus`` for :class:`~physicsnemo.core.Module` instances or
      ``.pt`` for plain PyTorch models.  When several models share a class
      name, a numeric *id* is appended (``"MyModel0"``, ``"MyModel1"``).
    * **Training state** (when any of ``optimizer`` / ``scheduler`` /
      ``scaler`` is provided, or
      :class:`~physicsnemo.utils.capture._StaticCapture` scalers exist):
      ``checkpoint.{mp_rank}.{epoch}.pt`` containing their combined
      ``state_dict`` entries, plus ``epoch`` and ``metadata``.

    When any model is FSDP-wrapped or contains DTensor/ShardTensor
    parameters the function enters *distributed* mode: all ranks **must**
    call it, state is gathered via DCP collective helpers, and only rank 0
    writes files.

    Use :func:`load_checkpoint` to restore from these files.
    To instantiate *and* load a model in one step (without pre-constructing
    it), use :meth:`~physicsnemo.core.module.Module.from_checkpoint`.

    Parameters
    ----------
    path : Path | str
        Directory in which to store checkpoint files.  Created
        automatically for local paths if it does not exist.
    models : torch.nn.Module | list[torch.nn.Module] | None, optional
        Model(s) whose weights should be saved.
    optimizer : torch.optim.Optimizer | None, optional
        Optimizer whose ``state_dict`` should be saved.
    scheduler : LRScheduler | None, optional
        Learning-rate scheduler whose ``state_dict`` should be saved.
    scaler : GradScaler | None, optional
        AMP gradient scaler whose ``state_dict`` should be saved.
        If ``None`` but a
        :class:`~physicsnemo.utils.capture._StaticCapture` scaler exists,
        that scaler's state is saved instead.
    epoch : int | None, optional
        Epoch index to embed in the filename and the checkpoint dict.
        When ``None``, the next available index is used.
    metadata : dict[str, Any] | None, optional
        Arbitrary key-value pairs persisted alongside the training state
        (e.g. best validation loss, MLflow run ID).
    optimizer_model : torch.nn.Module | None, optional
        The model whose parameters the ``optimizer`` is tracking so that
        parameter unsharding of optimizer state can be performed correctly.
        Only required when multiple models are provided, and at least one of
        them is a distributed model (FSDP/ShardTensor). When ``None``, the
        first model in ``models`` is used.  Ignored when *not* in distributed
        mode.

    Examples
    --------
    Save a model together with optimizer and scheduler state:

    >>> import tempfile, os, torch
    >>> from physicsnemo.utils.checkpoint import save_checkpoint
    >>> from physicsnemo.models.mlp import FullyConnected
    >>> model = FullyConnected(in_features=32, out_features=64)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     save_checkpoint(tmpdir, models=model, optimizer=optimizer,
    ...                     scheduler=scheduler, epoch=1)
    ...     sorted(f for f in os.listdir(tmpdir))
    ['FullyConnected.0.1.mdlus', 'checkpoint.0.1.pt']

    Save at multiple epochs with additional metadata:

    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     save_checkpoint(tmpdir, models=model, optimizer=optimizer, epoch=1,
    ...                     metadata={"loss": 0.42, "experiment": "run_01"})
    ...     save_checkpoint(tmpdir, models=model, optimizer=optimizer, epoch=2,
    ...                     metadata={"loss": 0.31, "experiment": "run_01"})
    ...     sorted(f for f in os.listdir(tmpdir))
    ['FullyConnected.0.1.mdlus', 'FullyConnected.0.2.mdlus', 'checkpoint.0.1.pt', 'checkpoint.0.2.pt']
    """
    path = str(path)
    protocol = fsspec.utils.get_protocol(path)
    fs = fsspec.filesystem(protocol)

    # Prepare models and detect distributed mode
    named_models: dict[str, torch.nn.Module] = {}
    is_distributed = False
    if models:
        if not isinstance(models, list):
            models = [models]
        named_models = _unique_model_names(models)
        is_distributed = any(_is_distributed_model(m) for m in named_models.values())

    if not DistributedManager.is_initialized():
        checkpoint_logging.warning(
            "`DistributedManager` not initialized already. "
            "Initializing now, but this might lead to unexpected errors"
        )
        DistributedManager.initialize()
    manager = DistributedManager()
    is_rank0 = manager.rank == 0
    should_write = is_rank0 if is_distributed else True

    # Create checkpoint directory (only writing rank)
    if should_write and protocol == "file" and not Path(path).is_dir():
        checkpoint_logging.warning(
            f"Output directory {path} does not exist, will attempt to create"
        )
        Path(path).mkdir(parents=True, exist_ok=True)

    if is_distributed:
        torch.distributed.barrier()

    # == Saving model checkpoint ==
    for name, model in named_models.items():
        inner = _unwrap_fsdp(model)
        model_type = "mdlus" if isinstance(inner, physicsnemo.core.Module) else "pt"
        file_name = _get_checkpoint_filename(
            path,
            name,
            index=epoch,
            saving=True,
            model_type=model_type,
            distributed=is_distributed,
        )

        if _is_distributed_model(model):
            # cpu_offload is handled manually because the DCP option
            # hangs for FSDP NO_SHARD + DTensor topologies.
            options = StateDictOptions(full_state_dict=True)
            state_dict = get_model_state_dict(model, options=options)
            if should_write:
                state_dict = _cpu_offload_state_dict(state_dict)
                if isinstance(inner, physicsnemo.core.Module):
                    inner.save(file_name, _state_dict=state_dict)
                else:
                    with fs.open(file_name, "wb") as fp:
                        torch.save(state_dict, fp)
                checkpoint_logging.success(f"Saved model state dictionary: {file_name}")
        else:
            if should_write:
                if isinstance(inner, physicsnemo.core.Module):
                    inner.save(file_name)
                else:
                    with fs.open(file_name, "wb") as fp:
                        torch.save(model.state_dict(), fp)
                checkpoint_logging.success(f"Saved model state dictionary: {file_name}")

    # == Saving training checkpoint ==
    checkpoint_dict: dict[str, Any] = {}

    if optimizer:
        if is_distributed:
            opt_model = optimizer_model or next(
                (m for m in named_models.values() if _is_distributed_model(m)),
                None,
            )
            if opt_model is not None:
                # cpu_offload is handled manually because the DCP option
                # hangs for FSDP NO_SHARD + DTensor topologies.
                options = StateDictOptions(full_state_dict=True)
                opt_state_dict = get_optimizer_state_dict(
                    opt_model, optimizer, options=options
                )
                if should_write:
                    opt_state_dict = _cpu_offload_state_dict(opt_state_dict)
            else:
                opt_state_dict = optimizer.state_dict()
        else:
            opt_state_dict = optimizer.state_dict()

        # Strip out torch dynamo wrapper prefix
        for pg in opt_state_dict.get("param_groups", []):
            param_names = pg.get("param_names")
            if param_names is None:
                continue
            pg["param_names"] = [pn.removeprefix("_orig_mod.") for pn in param_names]
        checkpoint_dict["optimizer_state_dict"] = opt_state_dict

    if scheduler:
        checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()

    if scaler:
        checkpoint_dict["scaler_state_dict"] = scaler.state_dict()
    if _StaticCapture._amp_scalers:
        checkpoint_dict["static_capture_state_dict"] = _StaticCapture.state_dict()

    output_filename = _get_checkpoint_filename(
        path,
        index=epoch,
        saving=True,
        model_type="pt",
        distributed=is_distributed,
    )
    if epoch:
        checkpoint_dict["epoch"] = epoch
    if metadata:
        checkpoint_dict["metadata"] = metadata

    if bool(checkpoint_dict) and should_write:
        with fs.open(output_filename, "wb") as fp:
            torch.save(checkpoint_dict, fp)
        checkpoint_logging.success(f"Saved training checkpoint: {output_filename}")


def load_checkpoint(
    path: Path | str,
    models: torch.nn.Module | list[torch.nn.Module] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: GradScaler | None = None,
    epoch: int | None = None,
    metadata_dict: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
    optimizer_model: torch.nn.Module | None = None,
) -> int:
    r"""Load a training checkpoint saved by :func:`save_checkpoint`.

    Scans ``path`` for checkpoint files and restores state dictionaries
    into the provided training objects.  Objects that are ``None`` are
    silently skipped.

    When any model is FSDP-wrapped or contains DTensor/ShardTensor
    parameters the function enters *distributed* mode: all ranks **must**
    call it, rank 0 reads files from disk, and model/optimizer state is
    scattered to all ranks via DCP helpers.

    Parameters
    ----------
    path : Path | str
        Directory containing checkpoint files (local path or ``fsspec``
        URI).  If the directory does not exist, the load is skipped and
        ``0`` is returned.
    models : torch.nn.Module | list[torch.nn.Module] | None, optional
        Model(s) whose ``state_dict`` should be restored.  DDP and
        ``torch.compile`` wrappers are stripped automatically.
    optimizer : torch.optim.Optimizer | None, optional
        Optimizer whose ``state_dict`` should be restored.
    scheduler : LRScheduler | None, optional
        Learning-rate scheduler whose ``state_dict`` should be restored.
    scaler : GradScaler | None, optional
        AMP gradient scaler whose ``state_dict`` should be restored.
    epoch : int | None, optional
        Specific checkpoint index to load.  When ``None``, the checkpoint
        with the largest index (most recent) is loaded.
    metadata_dict : dict[str, Any] | None, optional
        If a ``dict`` is provided, it is updated **in-place** with any
        metadata that was persisted by :func:`save_checkpoint`.
    device : str | torch.device, optional
        Device onto which tensors are mapped during loading.  By default
        ``"cpu"``.
    optimizer_model : torch.nn.Module | None, optional
        The model whose parameters the ``optimizer`` is tracking.
        Required by the DCP ``set_optimizer_state_dict`` helper when
        distributed mode is active.  When ``None``, the first model in
        ``models`` is used.  Ignored when *not* in distributed mode.

    Returns
    -------
    int
        The epoch stored in the checkpoint.  Returns ``0`` when:

        * The checkpoint directory does not exist.
        * No training-state file is found inside the directory.
        * The training-state file does not contain an ``"epoch"`` key.

    Examples
    --------
    Save and then restore a model, optimizer, and scheduler from a checkpoint:

    >>> import tempfile, torch
    >>> from physicsnemo.utils.checkpoint import save_checkpoint, load_checkpoint
    >>> from physicsnemo.models.mlp import FullyConnected
    >>> model = FullyConnected(in_features=32, out_features=64)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     save_checkpoint(tmpdir, models=model, optimizer=optimizer,
    ...                     scheduler=scheduler, epoch=1)
    ...     epoch = load_checkpoint(tmpdir, models=model, optimizer=optimizer,
    ...                             scheduler=scheduler)
    ...     epoch
    1

    Load a specific epoch and retrieve saved metadata:

    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     save_checkpoint(tmpdir, models=model, optimizer=optimizer, epoch=1,
    ...                     metadata={"loss": 0.42, "experiment": "run_01"})
    ...     save_checkpoint(tmpdir, models=model, optimizer=optimizer, epoch=2,
    ...                     metadata={"loss": 0.31, "experiment": "run_01"})
    ...     meta = {}
    ...     epoch = load_checkpoint(tmpdir, models=model, optimizer=optimizer,
    ...                             epoch=1, metadata_dict=meta)
    ...     epoch
    1
    >>> meta["loss"]
    0.42
    """
    path = str(path)
    fs = fsspec.filesystem(fsspec.utils.get_protocol(path))

    # Prepare models and detect distributed mode
    named_models: dict[str, torch.nn.Module] = {}
    is_distributed = False
    if models:
        if not isinstance(models, list):
            models = [models]
        named_models = _unique_model_names(models, loading=True)
        is_distributed = any(_is_distributed_model(m) for m in named_models.values())

    if not DistributedManager.is_initialized():
        checkpoint_logging.warning(
            "`DistributedManager` not initialized already. "
            "Initializing now, but this might lead to unexpected errors"
        )
        DistributedManager.initialize()
    manager = DistributedManager()
    is_rank0 = manager.rank == 0

    # ------------------------------------------------------------------
    # Distributed load path -- all ranks participate
    # ------------------------------------------------------------------
    if is_distributed:
        return _load_checkpoint_distributed(
            path=path,
            fs=fs,
            named_models=named_models,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            metadata_dict=metadata_dict,
            device=device,
            optimizer_model=optimizer_model,
            is_rank0=is_rank0,
        )

    # ------------------------------------------------------------------
    # Non-distributed load path
    # ------------------------------------------------------------------
    if fs.exists(path):
        if fs.isfile(path):
            raise FileNotFoundError(
                f"Provided checkpoint directory {path} is a file, not directory"
            )
    else:
        checkpoint_logging.warning(
            f"Provided checkpoint directory {path} does not exist, skipping load"
        )
        return 0

    # == Loading model checkpoint ==
    for name, model in named_models.items():
        inner = _unwrap_fsdp(model)
        model_type = "mdlus" if isinstance(inner, physicsnemo.core.Module) else "pt"
        file_name = _get_checkpoint_filename(
            path, name, index=epoch, model_type=model_type
        )
        if not fs.exists(file_name):
            checkpoint_logging.error(
                f"Could not find valid model file {file_name}, skipping load"
            )
            continue

        if isinstance(inner, physicsnemo.core.Module):
            inner.load(file_name)
        else:
            file_to_load = _cache_if_needed(file_name)
            missing_keys, unexpected_keys = model.load_state_dict(
                torch.load(file_to_load, map_location=device, weights_only=False)
            )
            if missing_keys:
                checkpoint_logging.warning(
                    f"Missing keys when loading {name}: {missing_keys}"
                )
            if unexpected_keys:
                checkpoint_logging.warning(
                    f"Unexpected keys when loading {name}: {unexpected_keys}"
                )

        checkpoint_logging.success(
            f"Loaded model state dictionary {file_name} to device {device}"
        )

    # == Loading training checkpoint ==
    checkpoint_filename = _get_checkpoint_filename(path, index=epoch, model_type="pt")
    if not fs.exists(checkpoint_filename):
        checkpoint_logging.warning(
            "Could not find valid checkpoint file, skipping load"
        )
        return 0

    file_to_load = _cache_if_needed(checkpoint_filename)
    checkpoint_dict = torch.load(file_to_load, map_location=device, weights_only=False)
    checkpoint_logging.success(
        f"Loaded checkpoint file {checkpoint_filename} to device {device}"
    )

    if optimizer and "optimizer_state_dict" in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        checkpoint_logging.success("Loaded optimizer state dictionary")

    if scheduler and "scheduler_state_dict" in checkpoint_dict:
        scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])
        checkpoint_logging.success("Loaded scheduler state dictionary")

    if scaler and "scaler_state_dict" in checkpoint_dict:
        scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])
        checkpoint_logging.success("Loaded grad scaler state dictionary")

    if "static_capture_state_dict" in checkpoint_dict:
        _StaticCapture.load_state_dict(checkpoint_dict["static_capture_state_dict"])
        checkpoint_logging.success("Loaded static capture state dictionary")

    loaded_epoch = 0
    if "epoch" in checkpoint_dict:
        loaded_epoch = checkpoint_dict["epoch"]

    if metadata_dict is not None:
        metadata_dict.update(checkpoint_dict.get("metadata", {}))

    return loaded_epoch


def load_model_weights(
    model: torch.nn.Module,
    weights_path: str,
    device: str | torch.device = "cpu",
) -> None:
    r"""Load model weights from a single checkpoint file.

    Loads a ``.mdlus`` (or ``.pt``) file directly into *model*, handling
    FSDP and DTensor/ShardTensor distribution automatically.  Unlike
    :func:`load_checkpoint` (which expects a checkpoint *directory* with
    numbered files), this function accepts a path to a single file.

    When the model is FSDP-wrapped or has DTensor parameters this is a
    **collective** operation — all ranks must call it.  Rank 0 reads the
    file and state is scattered via DCP helpers.

    Parameters
    ----------
    model : torch.nn.Module
        The model to load weights into.  May be FSDP-wrapped, contain
        DTensor/ShardTensor parameters, or be a plain module.
    weights_path : str
        Path to a ``.mdlus`` or ``.pt`` checkpoint file (local path or
        ``fsspec`` URI).
    device : str | torch.device, optional
        Device for :func:`torch.load` ``map_location``.  By default
        ``"cpu"``.
    """
    model = _unwrap_ddp_compile(model, loading=True)
    is_mdlus = _is_mdlus_archive(weights_path)

    if not _is_distributed_model(model):
        inner = _unwrap_fsdp(model)
        if is_mdlus and isinstance(inner, physicsnemo.core.Module):
            inner.load(weights_path)
        else:
            cached = _cache_if_needed(weights_path)
            if is_mdlus:
                sd = _extract_mdlus_state_dict(weights_path, device)
            else:
                sd = torch.load(cached, map_location=device, weights_only=False)
            inner.load_state_dict(sd)
        checkpoint_logging.success(f"Loaded model weights from {weights_path}")
        return

    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    is_rank0 = DistributedManager().rank == 0

    state_dict: dict[str, Any] = {}
    if is_rank0:
        if is_mdlus:
            state_dict = _extract_mdlus_state_dict(weights_path, device)
        else:
            cached = _cache_if_needed(weights_path)
            state_dict = torch.load(cached, map_location=device, weights_only=False)

    dtensor_plc = _get_dtensor_param_placements(model)
    if _has_non_fsdp_dtensors(model, dtensor_plc):
        sd_list: list[Any] = [state_dict]
        torch.distributed.broadcast_object_list(sd_list, src=0)
        state_dict = _redistribute_sd_for_dtensor(dtensor_plc, sd_list[0])
        options = StateDictOptions(full_state_dict=True)
    else:
        options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)

    set_model_state_dict(model, state_dict, options=options)
    checkpoint_logging.success(f"Loaded model weights from {weights_path}")


# ------------------------------------------------------------------
# Distributed load implementation
# ------------------------------------------------------------------


def _load_checkpoint_distributed(
    *,
    path: str,
    fs: fsspec.AbstractFileSystem,
    named_models: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer | None,
    scheduler: LRScheduler | None,
    scaler: GradScaler | None,
    epoch: int | None,
    metadata_dict: dict[str, Any] | None,
    device: str | torch.device,
    optimizer_model: torch.nn.Module | None,
    is_rank0: bool,
) -> int:
    """Distributed load: rank 0 reads files, DCP broadcasts to all ranks."""
    broadcast_options = StateDictOptions(
        full_state_dict=True, broadcast_from_rank0=True
    )
    full_options = StateDictOptions(full_state_dict=True)

    # --- Rank 0 checks directory existence and loads raw data -----------
    dir_exists = fs.exists(path) and not fs.isfile(path) if is_rank0 else None
    flags: list[Any] = [dir_exists]
    torch.distributed.broadcast_object_list(flags, src=0)
    dir_exists = flags[0]

    if not dir_exists:
        checkpoint_logging.warning(
            f"Provided checkpoint directory {path} does not exist, skipping load"
        )
        return 0

    # --- Load model checkpoints -----------------------------------------
    # Rank 0: determine which model files exist and load their state dicts
    model_file_info: dict[str, str | None] = {}
    model_state_dicts: dict[str, dict[str, Any]] = {}
    if is_rank0:
        for name, model in named_models.items():
            inner = _unwrap_fsdp(model)
            model_type = "mdlus" if isinstance(inner, physicsnemo.core.Module) else "pt"
            file_name = _get_checkpoint_filename(
                path,
                name,
                index=epoch,
                model_type=model_type,
                distributed=True,
            )
            if fs.exists(file_name):
                model_file_info[name] = file_name
                if isinstance(inner, physicsnemo.core.Module):
                    model_state_dicts[name] = _extract_mdlus_state_dict(
                        file_name, device
                    )
                else:
                    file_to_load = _cache_if_needed(file_name)
                    model_state_dicts[name] = torch.load(
                        file_to_load,
                        map_location=device,
                        weights_only=False,
                    )
            else:
                model_file_info[name] = None

    # Broadcast which model files were found
    info_list: list[Any] = [model_file_info]
    torch.distributed.broadcast_object_list(info_list, src=0)
    model_file_info = info_list[0]

    # Distribute model state dicts via DCP
    for name, model in named_models.items():
        if model_file_info.get(name) is None:
            checkpoint_logging.error(
                f"Could not find valid model file for {name}, skipping load"
            )
            continue

        if _is_distributed_model(model):
            # Collective: inspect native state dict for DTensor placements.
            # This is needed because use_orig_params=False flattens DTensor
            # params into a plain FlatParameter, hiding them from inspection.
            dtensor_plc = _get_dtensor_param_placements(model)
            if _has_non_fsdp_dtensors(model, dtensor_plc):
                # broadcast_from_rank0 does not handle user-managed DTensor
                # redistribution (e.g. ShardTensor on a domain mesh), so we
                # broadcast the full state dict ourselves and convert entries
                # to DTensors.
                sd_list: list[Any] = [
                    model_state_dicts.get(name, {}) if is_rank0 else {}
                ]
                torch.distributed.broadcast_object_list(sd_list, src=0)
                sd = _redistribute_sd_for_dtensor(dtensor_plc, sd_list[0])
                set_model_state_dict(model, sd, options=full_options)
            else:
                # FSDP-managed DTensors (FULL_SHARD/SHARD_GRAD_OP) or no
                # DTensors at all — broadcast_from_rank0 handles both. Force
                # standard contiguity on rank 0 first so the per-tensor
                # broadcast inside DCP doesn't permute channels_last params on
                # receive (see ``_force_standard_contiguous`` for the why).
                sd = model_state_dicts.get(name, {}) if is_rank0 else {}
                if is_rank0:
                    sd = _force_standard_contiguous(sd)
                set_model_state_dict(model, sd, options=broadcast_options)
        else:
            # A mix of distributed and non-distributed models is valid
            # (e.g. a main FSDP model alongside a small auxiliary model).
            sd_list = [model_state_dicts.get(name, {}) if is_rank0 else {}]
            torch.distributed.broadcast_object_list(sd_list, src=0)
            inner = _unwrap_fsdp(model)
            inner.load_state_dict(sd_list[0])

        file_name = model_file_info[name]
        checkpoint_logging.success(
            f"Loaded model state dictionary {file_name} to device {device}"
        )

    # --- Load training checkpoint ---------------------------------------
    checkpoint_filename = _get_checkpoint_filename(
        path, index=epoch, model_type="pt", distributed=True
    )

    # Broadcast file existence so all ranks agree on whether to enter the
    # (collective) optimizer load. Without this, a rundir that has model
    # weights but no training checkpoint -- e.g. fine-tuning from a
    # weights-only export -- would have rank 0 enter ``set_optimizer_state_dict``
    # with an empty dict and trip the "missing 'state'" error inside DCP.
    ckpt_exists = fs.exists(checkpoint_filename) if is_rank0 else None
    ckpt_flags: list[Any] = [ckpt_exists]
    torch.distributed.broadcast_object_list(ckpt_flags, src=0)
    ckpt_exists = ckpt_flags[0]

    if not ckpt_exists:
        checkpoint_logging.warning(
            f"No training checkpoint at {checkpoint_filename}; "
            "skipping optimizer/scheduler/scaler load"
        )
        return 0

    checkpoint_dict: dict[str, Any] = {}
    if is_rank0:
        file_to_load = _cache_if_needed(checkpoint_filename)
        checkpoint_dict = torch.load(
            file_to_load, map_location=device, weights_only=False
        )
        checkpoint_logging.success(
            f"Loaded checkpoint file {checkpoint_filename} to device {device}"
        )

    # Optimizer state via DCP (collective)
    if optimizer:
        opt_model = optimizer_model or next(
            (m for m in named_models.values() if _is_distributed_model(m)),
            None,
        )
        optim_sd = checkpoint_dict.get("optimizer_state_dict", {}) if is_rank0 else {}
        if opt_model is not None and _is_distributed_model(opt_model):
            dtensor_plc = _get_dtensor_param_placements(opt_model)
            if _has_non_fsdp_dtensors(opt_model, dtensor_plc):
                osd_list: list[Any] = [optim_sd]
                torch.distributed.broadcast_object_list(osd_list, src=0)
                optim_sd = _redistribute_optim_sd_for_dtensor(dtensor_plc, osd_list[0])
                optim_sd = _remap_channels_last_optim_sd(opt_model, optim_sd)
                set_optimizer_state_dict(
                    opt_model, optimizer, optim_sd, options=full_options
                )
            else:
                # Remap on rank 0 only -- DCP broadcasts the rank-0 dict to
                # the others as part of broadcast_from_rank0.
                optim_sd = _remap_channels_last_optim_sd(opt_model, optim_sd)
                set_optimizer_state_dict(
                    opt_model, optimizer, optim_sd, options=broadcast_options
                )
            checkpoint_logging.success("Loaded optimizer state dictionary")
        elif optim_sd:
            optimizer.load_state_dict(optim_sd)
            checkpoint_logging.success("Loaded optimizer state dictionary")

    # Broadcast remaining training state (scheduler, scaler, epoch, metadata)
    rest: dict[str, Any] = {}
    if is_rank0:
        rest = {k: v for k, v in checkpoint_dict.items() if k != "optimizer_state_dict"}
    rest_list: list[Any] = [rest]
    torch.distributed.broadcast_object_list(rest_list, src=0)
    rest = rest_list[0]

    if scheduler and "scheduler_state_dict" in rest:
        scheduler.load_state_dict(rest["scheduler_state_dict"])
        checkpoint_logging.success("Loaded scheduler state dictionary")

    if scaler and "scaler_state_dict" in rest:
        scaler.load_state_dict(rest["scaler_state_dict"])
        checkpoint_logging.success("Loaded grad scaler state dictionary")

    if "static_capture_state_dict" in rest:
        _StaticCapture.load_state_dict(rest["static_capture_state_dict"])
        checkpoint_logging.success("Loaded static capture state dictionary")

    loaded_epoch = rest.get("epoch", 0)

    if metadata_dict is not None:
        metadata_dict.update(rest.get("metadata", {}))

    return loaded_epoch


def get_checkpoint_dir(base_dir: Path | str, model_name: str) -> str:
    r"""Build a model-specific checkpoint directory path.

    Returns ``"{base_dir}/checkpoints_{model_name}"``, handling both
    local paths and ``msc://`` URIs.

    Parameters
    ----------
    base_dir : Path | str
        Root directory under which the checkpoint subdirectory is placed.
    model_name : str
        Model name used as the directory suffix.

    Returns
    -------
    str
        Full path to the checkpoint directory.
    """
    base_dir = str(base_dir)
    top_level_dir = f"checkpoints_{model_name}"
    protocol = fsspec.utils.get_protocol(base_dir)
    if protocol == "msc":
        if not base_dir.endswith("/"):
            base_dir += "/"
        return base_dir + top_level_dir
    else:
        return os.path.join(base_dir, top_level_dir)


def _cache_if_needed(path: str) -> str:
    r"""Return a local path for ``path``, downloading to cache if remote.

    For the ``"file"`` protocol the path is returned unchanged.  For remote
    protocols the file is fetched via
    :func:`~physicsnemo.core.filesystem._download_cached` into a
    process-specific cache directory.

    Parameters
    ----------
    path : str
        Checkpoint file path (local or ``fsspec`` URI).

    Returns
    -------
    str
        Local filesystem path suitable for :func:`torch.load`.
    """
    protocol = fsspec.utils.get_protocol(path)
    if protocol == "file":
        return path
    else:
        return _download_cached(
            path,
            recursive=False,
            local_cache_path=os.path.join(LOCAL_CACHE, f"checkpoint_pid_{os.getpid()}"),
        )
