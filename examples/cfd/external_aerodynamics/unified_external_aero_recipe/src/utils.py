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

"""Shared utilities for the unified training recipe."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch.amp import autocast

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.optim import CombinedOptimizer

### Recipe-wide type aliases. Re-exported for use in loss.py, metrics.py,
### output_normalize.py, forward_kwargs.py, collate.py, train.py, infer.py,
### and the tests so that ``target_config`` values share a single source of
### truth.
FieldType: TypeAlias = Literal["scalar", "vector"]

### Allowed mixed-precision modes for the autocast context. ``"float8"`` is
### intentionally absent; `get_autocast_context` rejects it at runtime (see
### its error message for the padding rationale).
Precision: TypeAlias = Literal["float32", "float16", "bfloat16"]


def set_seed(seed: int | None, rank: int = 0) -> None:
    """Pin all RNG states for reproducible training.

    When *seed* is not None, seeds Python, NumPy, and PyTorch (CPU + all
    CUDA devices) with ``seed + rank`` so that different ranks diverge
    deterministically.  When *seed* is None this function is a no-op,
    preserving the current (non-deterministic) behaviour.
    """
    if seed is None:
        return
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed % (1 << 31))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_muon_optimizer(
    model: torch.nn.Module, cfg: DictConfig, *, compile_optimizer: bool = False
) -> torch.optim.Optimizer:
    """Build Muon + AdamW combined optimizer.

    Muon handles 2-D parameters (linear/attention weight matrices) while AdamW
    handles everything else (biases, layer-norm, embeddings, etc.).

    Args:
        model: The model (may be DDP-wrapped).
        cfg: Full Hydra config. Reads ``cfg.training.optimizer.*`` for lr,
            weight_decay, betas, and eps.
        compile_optimizer: If True, compile the optimizer step functions
            with ``torch.compile``.
    """
    base_model = model.module if hasattr(model, "module") else model
    muon_params = [p for p in base_model.parameters() if p.ndim == 2]
    other_params = [p for p in base_model.parameters() if p.ndim != 2]

    opt_cfg = cfg.training.optimizer
    lr = opt_cfg.lr
    weight_decay = opt_cfg.get("weight_decay", 1e-4)
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = opt_cfg.get("eps", 1e-8)

    compile_kwargs = {} if compile_optimizer else None

    if muon_params and other_params:
        return CombinedOptimizer(
            [
                torch.optim.Muon(
                    muon_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    adjust_lr_fn="match_rms_adamw",
                ),
                torch.optim.AdamW(
                    other_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps,
                ),
            ],
            torch_compile_kwargs=compile_kwargs,
        )
    elif muon_params:
        opt = torch.optim.Muon(
            muon_params,
            lr=lr,
            weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )
        if compile_optimizer:
            opt.step = torch.compile(opt.step)
        return opt
    else:
        opt = torch.optim.AdamW(
            other_params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
        )
        if compile_optimizer:
            opt.step = torch.compile(opt.step)
        return opt


# ---------------------------------------------------------------------------
# Field type helpers for target configurations
# ---------------------------------------------------------------------------


def field_dim(field_type: FieldType, n_spatial_dims: int = 3) -> int:
    """Number of channels a single ``"scalar"`` or ``"vector"`` field occupies.

    The type tag is always lowercase by contract -- the recipe normalises
    YAML inputs at the LossCalculator / MetricCalculator boundary. Pass
    pre-lowercased strings here.

    Args:
        field_type: ``"scalar"`` or ``"vector"``.
        n_spatial_dims: Dimensionality of vector fields. Default 3.

    Raises:
        ValueError: If ``field_type`` is not ``"scalar"`` or ``"vector"``.
    """
    if field_type == "scalar":
        return 1
    if field_type == "vector":
        return n_spatial_dims
    raise ValueError(
        f"Unknown field type {field_type!r}. Expected 'scalar' or 'vector'."
    )


def align_scalar_shapes(
    p: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align a ``(...)`` / ``(..., 1)`` shape mismatch by squeezing one side.

    Used in scalar-field loss / metric paths where the prediction may
    arrive as ``(B, N, 1)`` (sliced from a concatenated ``(B, N, C)``
    tensor before squeeze) while the target is ``(B, N)`` (per-element
    scalar from a TensorDict), or vice versa. After alignment both
    tensors share the same shape (or were already equal-shape).
    """
    if p.ndim > t.ndim and p.shape[-1] == 1:
        p = p.squeeze(-1)
    elif t.ndim > p.ndim and t.shape[-1] == 1:
        t = t.squeeze(-1)
    return p, t


def validate_field_coverage(
    target_config: dict[str, FieldType],
    pred: TensorDict,
    target: TensorDict,
) -> None:
    """Raise ``KeyError`` if *pred* or *target* is missing any field in *target_config*.

    Shared precondition check at the top of :class:`loss.LossCalculator` and
    :class:`metrics.MetricCalculator`. The error message identifies which
    side (``pred`` vs ``target``) is missing fields so config bugs surface
    against the right tensor.
    """
    for label, source in (("pred", pred), ("target", target)):
        missing = set(target_config) - set(source.keys())
        if missing:
            raise KeyError(f"{label} is missing target fields {sorted(missing)!r}")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def resolve_dict(cfg: DictConfig, path: str) -> dict[str, Any] | None:
    """Resolve `cfg.<path>` to a plain dict, or ``None`` if missing/empty.

    Wraps the OmegaConf incantation
    ``OmegaConf.to_container(OmegaConf.select(cfg, path, default=...), resolve=True) or None``
    that would otherwise repeat at every read site.
    """
    selected = OmegaConf.select(cfg, path, default=OmegaConf.create({}))
    container = OmegaConf.to_container(selected, resolve=True)
    return container or None


# ---------------------------------------------------------------------------
# Mixed-precision autocast
# ---------------------------------------------------------------------------


def get_autocast_context(precision: Precision):
    """Return an autocast context manager for the given precision.

    Args:
        precision: One of ``"float32"``, ``"float16"``, or ``"bfloat16"``.
            ``"float32"`` yields a no-op ``nullcontext``.

    Returns:
        An autocast context manager for the requested precision, or a
        no-op ``nullcontext`` when no casting is needed.

    Raises:
        NotImplementedError: For ``"float8"`` -- intentionally scoped out
            of this recipe; the raised message carries the padding
            rationale.
        ValueError: For any other unrecognized value (e.g. a YAML typo
            like ``"bf16"``), rather than silently running in fp32.
    """
    if precision == "float32":
        return nullcontext()
    elif precision == "float16":
        return autocast("cuda", dtype=torch.float16)
    elif precision == "bfloat16":
        return autocast("cuda", dtype=torch.bfloat16)
    elif precision == "float8":
        raise NotImplementedError(
            "precision='float8' is not supported in this recipe: TE fp8 needs "
            "every GEMM dimension (the per-sample point count and the target "
            "out_dim, e.g. 4) divisible by 16, which is not padded here. Use "
            "float32 / float16 / bfloat16. For an fp8 reference see "
            "examples/cfd/external_aerodynamics/transformer_models "
            "(update_model_params_for_fp8 / pad_input_for_fp8 / "
            "unpad_output_for_fp8) and TE's Fp8Padding / Fp8Unpadding modules."
        )
    else:
        raise ValueError(
            f"Unknown precision {precision!r}; expected one of "
            f"'float32', 'float16', 'bfloat16'."
        )


# ---------------------------------------------------------------------------
# Recursive tensor / mesh device movement
# ---------------------------------------------------------------------------

### Callable types for the recursive walker. ``LeafFn`` is the per-Tensor
### transform (mandatory); ``ContainerFn`` is the optional override
### applied to tensor-aware containers (Mesh / DomainMesh / TensorDict)
### when the default ``container.apply(leaf_fn)`` semantics aren't enough.
LeafFn = Callable[[torch.Tensor], torch.Tensor]
ContainerFn = Callable[[Any], Any]


def _recursive_apply(
    obj: Any,
    leaf_fn: LeafFn,
    *,
    container_fn: ContainerFn | None = None,
) -> Any:
    """Walk a nested structure, applying ``leaf_fn`` to every Tensor leaf.

    Tensor-aware containers (Mesh, DomainMesh, TensorDict) are routed
    through ``container_fn``. By default, ``container_fn`` delegates to
    ``container.apply(leaf_fn)``, which walks every tensor leaf in
    lock-step but does NOT touch container-level metadata
    (``TensorDict.device`` in particular stays at whatever it was).
    Override ``container_fn`` (e.g. ``lambda c: c.to(device)``) when the
    metadata change matters -- ``TensorDict`` treats ``device is None``
    as "leaves may be on any device", so device moves must go through
    ``.to(device)`` to be observable on the container.

    Plain dicts / lists / tuples are walked recursively. Note that
    ``TensorDict`` is matched in the container branch above, so it does
    NOT fall into the ``isinstance(obj, dict)`` branch (it isn't a
    ``dict`` subclass, but the explicit container check is what makes
    this work). Anything else passes through unchanged.
    """
    if container_fn is None:
        container_fn = lambda c: c.apply(leaf_fn)  # noqa: E731
    if isinstance(obj, torch.Tensor):
        return leaf_fn(obj)
    if isinstance(obj, (Mesh, DomainMesh, TensorDict)):
        return container_fn(obj)
    if isinstance(obj, dict):
        return {
            k: _recursive_apply(v, leaf_fn, container_fn=container_fn)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return type(obj)(
            _recursive_apply(v, leaf_fn, container_fn=container_fn) for v in obj
        )
    return obj


def recursive_to_device(obj: Any, device: torch.device | str) -> Any:
    """Move every tensor / Mesh / DomainMesh / TensorDict in a nested value to *device*.

    Containers go through ``.to(device)`` (not ``.apply(...)``) so that
    ``TensorDict.device`` is updated alongside the leaves; otherwise a
    later consumer reading ``td.device`` would still see ``None`` even
    though the underlying tensors have already moved.
    """
    return _recursive_apply(
        obj,
        lambda t: t.to(device),
        container_fn=lambda c: c.to(device),
    )


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------


def make_jsonl_logger(path: str | Path) -> Callable[[dict], None]:
    """Return a logger that appends timestamped JSON records to *path*.

    Each call serialises one ``record`` dict as a JSON line, stamping it
    with a UTC ``ts``. Shared by the trainer and the inference companion
    for their per-run ``metrics.jsonl``.
    """

    def log_jsonl(record: dict) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    return log_jsonl
