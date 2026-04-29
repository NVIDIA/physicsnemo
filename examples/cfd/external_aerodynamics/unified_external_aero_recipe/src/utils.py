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
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from omegaconf import DictConfig
from physicsnemo.optim import CombinedOptimizer


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

    Parameters
    ----------
    model : torch.nn.Module
        The model (may be DDP-wrapped).
    cfg : DictConfig
        Full Hydra config.  Reads ``cfg.training.optimizer.*`` for lr,
        weight_decay, betas, and eps.
    compile_optimizer : bool
        If True, compile the optimizer step functions with ``torch.compile``.
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
# Field specification for target configurations
# ---------------------------------------------------------------------------


@dataclass
class FieldSpec:
    """Specification for a single target field.

    Attributes:
        name: Human-readable name for the field (used in metric/loss keys).
        field_type: Either "scalar" or "vector".
        start_index: Starting index in the channel dimension.
        end_index: Ending index (exclusive) in the channel dimension.
    """

    name: str
    field_type: Literal["scalar", "vector"]
    start_index: int
    end_index: int

    @property
    def dim(self) -> int:
        """Number of channels for this field."""
        return self.end_index - self.start_index


def parse_target_config(
    target_config: dict[str, str], n_spatial_dims: int = 3
) -> list[FieldSpec]:
    """Parse target configuration to field specifications.

    Args:
        target_config: Mapping of field names to types ("scalar" or "vector").
                      Order determines channel indices.
        n_spatial_dims: Dimensionality of vector fields. Default is 3.

    Returns:
        List of FieldSpec objects describing each field.

    Raises:
        ValueError: If an unknown field type is specified.

    Example:
        >>> config = {"pressure": "scalar", "velocity": "vector"}
        >>> specs = parse_target_config(config)
        >>> specs[0]
        FieldSpec(name='pressure', field_type='scalar', start_index=0, end_index=1)
        >>> specs[1]
        FieldSpec(name='velocity', field_type='vector', start_index=1, end_index=4)
    """
    specs = []
    current_index = 0

    for name, field_type in target_config.items():
        field_type = field_type.lower()
        if field_type == "scalar":
            dim = 1
        elif field_type == "vector":
            dim = n_spatial_dims
        else:
            raise ValueError(
                f"Unknown field type '{field_type}' for field '{name}'. "
                "Expected 'scalar' or 'vector'."
            )

        specs.append(
            FieldSpec(
                name=name,
                field_type=field_type,
                start_index=current_index,
                end_index=current_index + dim,
            )
        )
        current_index += dim

    return specs


# This function, below, is to turn non-dimensionalized data back into
# dimensional data.  It's useful for inference scripts which may want to compute
# metrics on dimensional data, of course.

# It's not used at the moment.  But it will be, in the near future, so while it's
# dead code currently it won't be for long.

_NONDIM_TYPE_MAP = {"scalar": "pressure", "vector": "stress"}


def _to_physical(
    tensor: torch.Tensor,
    target_config: dict[str, str],
    normalizer,
    nondim_transform,
    metadata: dict,
    nondim_type_overrides: dict[str, str] | None = None,
) -> torch.Tensor:
    """Convert a model-space tensor (normalized + non-dim) back to physical units.

    Chains two inverse operations using the existing transform instances:
    1. ``NormalizeMeshFields.inverse_tensor`` -- undo z-score normalization
    2. ``NonDimensionalizeByMetadata.inverse_tensor`` -- undo non-dimensionalization

    Parameters
    ----------
    nondim_type_overrides : dict or None
        Optional per-field mapping of ``{field_name: nondim_type}`` (e.g.
        ``{"temperature": "temperature", "density": "density"}``).  When
        provided, overrides the default ``_NONDIM_TYPE_MAP`` lookup for
        fields that don't follow the simple scalar=pressure / vector=stress
        convention.
    """
    if not metadata:
        return tensor

    out = tensor
    device, dtype = tensor.device, tensor.dtype

    # Step 1: undo z-score normalization
    if normalizer is not None:
        out = normalizer.inverse_tensor(out, target_config)

    # Step 2: undo non-dimensionalization
    if nondim_transform is not None:
        overrides = nondim_type_overrides or {}
        nondim_fields = {
            name: overrides.get(name, _NONDIM_TYPE_MAP.get(ftype, ftype))
            for name, ftype in target_config.items()
        }
        U_inf = torch.tensor(metadata["U_inf"], dtype=dtype, device=device)
        rho_inf = torch.tensor(metadata["rho_inf"], dtype=dtype, device=device)
        p_inf = torch.tensor(metadata["p_inf"], dtype=dtype, device=device)
        q_inf = 0.5 * rho_inf * (U_inf * U_inf).sum()
        U_inf_mag = (U_inf * U_inf).sum().sqrt()

        T_inf = None
        if "T_inf" in metadata:
            T_inf = torch.tensor(metadata["T_inf"], dtype=dtype, device=device)

        out = nondim_transform.inverse_tensor(
            out,
            nondim_fields,
            q_inf,
            p_inf,
            U_inf_mag,
            rho_inf=rho_inf,
            T_inf=T_inf,
        )

    return out
