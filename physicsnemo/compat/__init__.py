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
    "physicsnemo.models.layers.interpolation": "physicsnemo.nn.module.interpolation",
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
    "physicsnemo.utils.patching": "physicsnemo.diffusion.multi_diffusion.patching",
    "physicsnemo.utils.domino": "physicsnemo.models.domino.utils",
    "physicsnemo.launch.utils.checkpoint": "physicsnemo.utils.checkpoint",
    "physicsnemo.launch.logging": "physicsnemo.utils.logging",
}


def _ensure_parent_packages(module_name: str) -> None:
    """Ensure every parent package of module_name exists in sys.modules.

    Creates placeholder modules for removed packages (e.g. physicsnemo.launch)
    so that "from physicsnemo.launch.utils import checkpoint" can resolve.
    """
    parts = module_name.split(".")
    for i in range(1, len(parts) - 1):
        parent_name = ".".join(parts[: i + 1])
        if parent_name in sys.modules:
            continue
        placeholder = types.ModuleType(parts[i])
        sys.modules[parent_name] = placeholder
        grandparent_name = ".".join(parts[:i])
        grandparent = sys.modules.get(grandparent_name)
        if grandparent is not None:
            setattr(grandparent, parts[i], placeholder)


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
