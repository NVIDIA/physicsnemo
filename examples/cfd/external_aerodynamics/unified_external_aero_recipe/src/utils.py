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

import random
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from physicsnemo.optim import CombinedOptimizer

if TYPE_CHECKING:
    from nondim import NondimFieldType, NonDimensionalizeByMetadata
    from physicsnemo.datapipes.transforms.mesh import NormalizeMeshFields

### Recipe-wide type aliases. Re-exported for use in loss.py, metrics.py,
### output_normalize.py, forward_kwargs.py, collate.py, train.py, and the
### tests so that ``target_config`` values share a single source of truth.
FieldType: TypeAlias = Literal["scalar", "vector"]

### Default mapping from a field's `target_config` type to the nondim
### recipe used by `NonDimensionalizeByMetadata`. Surface CFD predictions
### follow this convention by default (pressure scalars are non-dim'd as
### Cp; vector fields like wall shear stress are non-dim'd as Cf via the
### dynamic-pressure scaling). Override per-field via
### `nondim_type_overrides` when a field doesn't follow it (e.g.
### temperature scalars or velocity vectors).
_DEFAULT_NONDIM_TYPE_FROM_FIELD_TYPE: dict[FieldType, "NondimFieldType"] = {
    "scalar": "pressure",
    "vector": "stress",
}


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
# Re-dimensionalization (model-space -> physical units)
# ---------------------------------------------------------------------------


def to_physical_units(
    pred_td: TensorDict,
    target_config: dict[str, FieldType],
    normalizer: "NormalizeMeshFields | None",
    nondim_transform: "NonDimensionalizeByMetadata | None",
    metadata: dict[str, Any],
    nondim_type_overrides: dict[str, "NondimFieldType"] | None = None,
) -> TensorDict:
    """Convert a per-field model-space TensorDict back to physical units.

    Symmetric inverse of the dataset's normalize + non-dim pipeline.
    Chains the two existing per-field inverses (rather than going via
    a flat tensor) so the inference path can stay TensorDict-native:

    1. :meth:`NormalizeMeshFields.inverse_td` undoes z-score
       normalization (additive shift / multiplicative scale per field).
    2. :meth:`NonDimensionalizeByMetadata.inverse_td` undoes the
       freestream-conditioned non-dimensionalization
       (e.g. Cp -> p in Pa).

    Either step is skipped if the corresponding transform is ``None``,
    so callers can use this with a partially-configured pipeline (e.g.
    a model that was only normalized, or only non-dim'd).

    Parameters
    ----------
    pred_td : TensorDict
        Predictions keyed by field name, in model-space (post-normalize,
        post-nondim). Each leaf can be any shape.
    target_config : dict[str, FieldType]
        ``{field_name: "scalar"|"vector"}`` for every key in
        ``pred_td``. Drives the default nondim recipe lookup
        (``scalar`` -> ``pressure``, ``vector`` -> ``stress``).
    normalizer : NormalizeMeshFields | None
        The dataset's normalizer (or ``None`` to skip the inverse-norm
        step).
    nondim_transform : NonDimensionalizeByMetadata | None
        The dataset's non-dim transform (or ``None`` to skip the
        inverse-nondim step).
    metadata : dict[str, Any]
        Freestream conditions used to compute reference scales:
        requires ``U_inf`` and ``rho_inf`` (and ``p_inf`` for pressure-
        like fields). ``T_inf`` is required only when a field is mapped
        to ``"temperature"``. Each value is coerced to a 0-D float32
        tensor (matching the convention of
        :func:`nondim.freestream_scales`); float32 scales stay
        precision-stable through the inverse multiply even when
        ``pred_td`` is in bfloat16 / fp16.
    nondim_type_overrides : dict[str, NondimFieldType] | None
        Per-field overrides to the default nondim-type lookup. Use for
        fields where the default ``scalar -> pressure`` /
        ``vector -> stress`` mapping doesn't apply (e.g.
        ``{"temperature": "temperature"}``).

    Returns
    -------
    TensorDict
        New TensorDict (same keys, batch_size, and device as *pred_td*)
        with leaves in physical units. Returned as-is when both
        transforms are ``None`` and *metadata* is empty (no work to do).
    """
    if normalizer is None and (nondim_transform is None or not metadata):
        return pred_td

    out = pred_td
    if normalizer is not None:
        out = normalizer.inverse_td(out)

    if nondim_transform is not None and metadata:
        ### Lazy import keeps the recipe's module-level dependency graph
        ### narrow -- ``utils`` is at the bottom of the import chain and
        ### only ``to_physical_units`` reaches into the non-dim module.
        from nondim import freestream_scales

        ### Build a 0-D TensorDict from the YAML-decoded ``metadata``
        ### dict so we can delegate the q_inf / U_inf_mag math to the
        ### canonical ``freestream_scales`` helper rather than
        ### reimplementing it here. Cast each entry to float32 to
        ### match ``freestream_scales``'s convention (precision-stable
        ### reference scales, even when ``pred_td`` is bfloat16 / fp16).
        metadata_td = TensorDict(
            {k: torch.as_tensor(v, dtype=torch.float32) for k, v in metadata.items()},
        )
        q_inf, p_inf, U_inf_mag, rho_inf, T_inf = freestream_scales(metadata_td)

        ### Resolve each field's nondim recipe: explicit override wins,
        ### otherwise fall back on the scalar/vector default mapping.
        overrides = nondim_type_overrides or {}
        nondim_fields: dict[str, NondimFieldType] = {
            name: overrides.get(name, _DEFAULT_NONDIM_TYPE_FROM_FIELD_TYPE[ftype])
            for name, ftype in target_config.items()
        }
        out = nondim_transform.inverse_td(
            out,
            nondim_fields,
            q_inf,
            p_inf,
            U_inf_mag,
            rho_inf=rho_inf,
            T_inf=T_inf,
        )

    return out
