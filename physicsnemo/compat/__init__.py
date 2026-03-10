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

"""
This file is meant to provide a compatibility layer for physicsnemo v1

You can do
```
>>> import physicsnemo.compat as physicsnemo
>>> # All previous paths should work.

```
"""

import importlib
import sys
import types
import warnings

COMPAT_MAP = {
    "physicsnemo.utils.filesystem": "physicsnemo.core.filesystem",
    "physicsnemo.utils.version_check": "physicsnemo.core.version_check",
    "physicsnemo.models.meta": "physicsnemo.core.meta",
    "physicsnemo.models.module": "physicsnemo.core.module",
    "physicsnemo.utils.neighbors": "physicsnemo.nn.functional",
    "physicsnemo.utils.sdf": "physicsnemo.nn.functional.sdf",
    "physicsnemo.models.layers": "physicsnemo.nn",
    "physicsnemo.models.layers.activations": "physicsnemo.nn.module.activations",
    "physicsnemo.models.layers.attention_layers": "physicsnemo.nn.module.attention_layers",
    "physicsnemo.models.layers.ball_query": "physicsnemo.nn.module.ball_query",
    "physicsnemo.models.layers.conv_layers": "physicsnemo.nn.module.conv_layers",
    "physicsnemo.models.layers.dgm_layers": "physicsnemo.nn.module.dgm_layers",
    "physicsnemo.models.layers.drop": "physicsnemo.nn.module.drop",
    "physicsnemo.models.layers.fft": "physicsnemo.nn.module.fft",
    "physicsnemo.models.layers.fourier_layers": "physicsnemo.nn.module.fourier_layers",
    "physicsnemo.models.layers.fully_connected_layers": "physicsnemo.nn.module.fully_connected_layers",
    "physicsnemo.models.layers.fused_silu": "physicsnemo.nn.module.fused_silu",
    "physicsnemo.models.layers.interpolation": "physicsnemo.nn.functional.interpolation",
    "physicsnemo.models.layers.kan_layers": "physicsnemo.nn.module.kan_layers",
    "physicsnemo.models.layers.mlp_layers": "physicsnemo.nn.module.mlp_layers",
    "physicsnemo.models.layers.resample_layers": "physicsnemo.nn.module.resample_layers",
    "physicsnemo.models.layers.siren_layers": "physicsnemo.nn.module.siren_layers",
    "physicsnemo.models.layers.spectral_layers": "physicsnemo.nn.module.spectral_layers",
    "physicsnemo.models.layers.transformer_decoder": "physicsnemo.nn.module.transformer_decoder",
    "physicsnemo.models.layers.transformer_layers": "physicsnemo.nn.module.transformer_layers",
    "physicsnemo.models.layers.weight_fact": "physicsnemo.nn.module.weight_fact",
    "physicsnemo.models.layers.weight_norm": "physicsnemo.nn.module.weight_norm",
    "physicsnemo.utils.graphcast": "physicsnemo.models.graphcast.utils",
    "physicsnemo.utils.diffusion": "physicsnemo.diffusion.utils",
    "physicsnemo.models.diffusion.corrdiff_utils": "physicsnemo.diffusion.samplers",
    "physicsnemo.metrics.diffusion": "physicsnemo.diffusion.metrics",
    "physicsnemo.utils.patching": "physicsnemo.diffusion.multi_diffusion.patching",
    "physicsnemo.utils.domino": "physicsnemo.models.domino.utils",
    "physicsnemo.launch.utils.checkpoint": "physicsnemo.utils.checkpoint",
    "physicsnemo.launch.logging": "physicsnemo.utils.logging",
    "physicsnemo.distributed.shard_tensor": "physicsnemo.domain_parallel.shard_tensor",
}

OBJECT_COMPAT_MAP = {
    # Diffusion layers
    "physicsnemo.models.diffusion.AttentionOp": "physicsnemo.nn.AttentionOp",
    "physicsnemo.models.diffusion.layers.Attention": "physicsnemo.nn.UNetAttention",
    "physicsnemo.models.diffusion.Conv2D": "physicsnemo.nn.Conv2d",
    "physicsnemo.models.diffusion.FourierEmbedding": "physicsnemo.nn.FourierEmbedding",
    "physicsnemo.models.diffusion.GroupNorm": "physicsnemo.nn.GroupNorm",
    "physicsnemo.models.diffusion.get_group_norm": "physicsnemo.nn.get_group_norm",
    "physicsnemo.models.diffusion.Linear": "physicsnemo.nn.Linear",
    "physicsnemo.models.diffusion.PositionalEmbedding": "physicsnemo.nn.PositionalEmbedding",
    "physicsnemo.models.diffusion.UNetBlock": "physicsnemo.nn.UNetBlock",
    # Diffusion UNets
    "physicsnemo.models.diffusion.SongUNet": "physicsnemo.models.diffusion_unets.SongUNet",
    "physicsnemo.models.diffusion.SongUNetPosEmbd": "physicsnemo.models.diffusion_unets.SongUNetPosEmbd",
    "physicsnemo.models.diffusion.SongUNetPosLtEmbd": "physicsnemo.models.diffusion_unets.SongUNetPosLtEmbd",
    "physicsnemo.models.diffusion.DhariwalUNet": "physicsnemo.models.diffusion_unets.DhariwalUNet",
    "physicsnemo.models.diffusion.UNet": "physicsnemo.models.diffusion_unets.UNet",
    "physicsnemo.models.diffusion.StormCastUNet": "physicsnemo.models.diffusion_unets.StormCastUNet",
    # Diffusion Preconditioners
    "physicsnemo.models.diffusion.EDMPrecond": "physicsnemo.diffusion.preconditioners.EDMPrecond",
    "physicsnemo.models.diffusion.EDMPrecondSuperResolution": "physicsnemo.diffusion.preconditioners.EDMPrecondSuperResolution",
    "physicsnemo.models.diffusion.EDMPrecondSR": "physicsnemo.diffusion.preconditioners.EDMPrecondSR",
    "physicsnemo.models.diffusion.VEPrecond": "physicsnemo.diffusion.preconditioners.VEPrecond",
    "physicsnemo.models.diffusion.VPPrecond": "physicsnemo.diffusion.preconditioners.VPPrecond",
    "physicsnemo.models.diffusion.iDDPMPrecond": "physicsnemo.diffusion.preconditioners.iDDPMPrecond",
    "physicsnemo.models.diffusion.VEPrecond_dfsr_cond": "physicsnemo.diffusion.preconditioners.VEPrecond_dfsr_cond",
    "physicsnemo.models.diffusion.VEPrecond_dfsr": "physicsnemo.diffusion.preconditioners.VEPrecond_dfsr",
}


def _ensure_parent_packages(module_name: str) -> None:
    """Ensure every parent package of ``module_name`` exists in sys.modules.

    Only the parent chain is created; the final segment (the module itself) is
    not. This allows later code to assign that final name (e.g. to a real
    module alias or a placeholder).

    For each missing parent, a placeholder ``ModuleType`` is created, stored in
    ``sys.modules`` under that parent's full name, and attached as an
    attribute on its own parent (so ``import physicsnemo.models`` can resolve
    ``physicsnemo.models`` both via sys.modules and as ``physicsnemo.models``).
    The root (first segment) is assumed to already exist (e.g. ``physicsnemo``
    is already loaded before compat runs).

    Example: for ``module_name="physicsnemo.models.diffusion"``, this ensures
    ``physicsnemo`` and ``physicsnemo.models`` exist as placeholders if missing.
    It does not create ``physicsnemo.models.diffusion``. So
    ``from physicsnemo.models.diffusion import X`` still requires something else
    to put ``physicsnemo.models.diffusion`` in sys.modules. (see below)

    If ``module_name`` has only one or two segments (e.g. ``"physicsnemo"`` or
    ``"physicsnemo.models"``), the loop over parent packages runs over no
    indices, so nothing is created; that is intentional (the "module" is then
    the top-level or second-level name, and no intermediate chain is needed).
    """
    parts = module_name.split(".")
    for i in range(1, len(parts) - 1):
        parent_name = ".".join(parts[: i + 1])
        if parent_name in sys.modules:
            continue
        # Prefer loading the real parent module/package when it exists.
        try:
            importlib.import_module(parent_name)
            continue
        except ImportError:
            pass
        placeholder = types.ModuleType(parts[i])
        sys.modules[parent_name] = placeholder
        grandparent_name = ".".join(parts[:i])
        grandparent = sys.modules.get(grandparent_name)
        if grandparent is not None:
            setattr(grandparent, parts[i], placeholder)


def _ensure_module_exists(module_name: str) -> types.ModuleType:
    """Ensure ``module_name`` exists in sys.modules, creating a placeholder if needed.

    If the module is already loaded, it is returned. Otherwise the parent
    package chain is ensured (via _ensure_parent_packages), then a placeholder
    module for ``module_name`` is created, registered in sys.modules, and
    attached to its parent, so the full path is importable.

    Example: for ``module_name="physicsnemo.models.diffusion"``, this guarantees
    ``physicsnemo.models.diffusion`` exists (creating a placeholder when the
    module was removed). Callers can then set attributes on it (e.g. compat
    aliases) so ``from physicsnemo.models.diffusion import UNetBlock`` works.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    _ensure_parent_packages(module_name)
    parent_name, child = module_name.rsplit(".", 1)
    parent_mod = sys.modules[parent_name]
    placeholder = types.ModuleType(child)
    sys.modules[module_name] = placeholder
    setattr(parent_mod, child, placeholder)
    return placeholder


def install():
    """Install backward-compatibility shims."""
    for old_name, new_name in COMPAT_MAP.items():
        try:
            new_mod = importlib.import_module(new_name)
        except ImportError:
            warnings.warn(
                f"Failed to import new module '{new_name}' for compat alias '{old_name}'"
            )
            continue

        # Register module alias
        sys.modules[old_name] = new_mod

        # Ensure removed parent packages exist so "from pkg.subpkg import name" works
        _ensure_parent_packages(old_name)

        # Attach the alias on the parent package
        try:
            parent_name, child = old_name.rsplit(".", 1)
            parent_mod = sys.modules[parent_name]
            setattr(parent_mod, child, new_mod)
        except Exception:
            warnings.warn(
                f"Failed to attach '{old_name}' onto its parent for compat alias; using sys.modules only"
            )

        warnings.warn(
            f"[compat] {old_name} is moved; use {new_name} instead",
            DeprecationWarning,
        )

    for old_path, new_path in OBJECT_COMPAT_MAP.items():
        old_module_name, old_obj_name = old_path.rsplit(".", 1)
        new_module_name, new_obj_name = new_path.rsplit(".", 1)

        try:
            new_mod = importlib.import_module(new_module_name)
            new_obj = getattr(new_mod, new_obj_name)
        except (ImportError, AttributeError):
            warnings.warn(
                f"Failed to import '{new_path}' for compat alias '{old_path}'"
            )
            continue

        try:
            old_mod = _ensure_module_exists(old_module_name)
            setattr(old_mod, old_obj_name, new_obj)
        except Exception:
            warnings.warn(
                f"Failed to attach '{old_path}' onto its parent "
                "for compat alias; using sys.modules only"
            )

        warnings.warn(
            f"[compat] {old_path} is moved; use {new_path} instead",
            DeprecationWarning,
        )
