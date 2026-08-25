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

"""DLWP HEALPix model building blocks."""

import warnings

from physicsnemo.nn import (
    HEALPixAvgPool,
    HEALPixLayer,
    HEALPixMaxPool,
    HEALPixPadding,
    HEALPixPaddingv2,
)
from physicsnemo.nn.module.hpx import (
    HEALPixFoldFaces,
    HEALPixUnfoldFaces,
)

from .healpix_blocks import (
    BasicConvBlock,
    ConvGRUBlock,
    ConvNeXtBlock,
    DoubleConvNeXtBlock,
    Interpolate,
    Multi_SymmetricConvNeXtBlock,
    SymmetricConvNeXtBlock,
    TransposedConvUpsample,
)
from .healpix_constraints import NonnegativeConstraint
from .healpix_decoder import UNetDecoder
from .healpix_encoder import UNetEncoder
from .normalization import ConditionalLayerNorm

__all__ = [
    "BasicConvBlock",
    "ConvGRUBlock",
    "ConvNeXtBlock",
    "DoubleConvNeXtBlock",
    "Interpolate",
    "Multi_SymmetricConvNeXtBlock",
    "SymmetricConvNeXtBlock",
    "TransposedConvUpsample",
    "UNetDecoder",
    "UNetEncoder",
    "ConditionalLayerNorm",
    "NonnegativeConstraint",
    "HEALPixFoldFaces",
    "HEALPixLayer",
    "HEALPixPadding",
    "HEALPixPaddingv2",
    "HEALPixUnfoldFaces",
    "HEALPixMaxPool",
    "HEALPixAvgPool",
]


def _remap_target(target: str) -> str:
    """
    Remapping methods for backwards compatibility of legacy checkpoints
    """
    explicit = {
        "physicsnemo.models.dlwp_healpix_layers.healpix_encoder.UNetEncoder": "physicsnemo.models.dlwp_healpix.layers.UNetEncoder",
        "physicsnemo.models.dlwp_healpix_layers.healpix_decoder.UNetDecoder": "physicsnemo.models.dlwp_healpix.layers.UNetDecoder",
    }
    if target in explicit:
        return explicit[target]

    if target.startswith("physicsnemo.models.dlwp_healpix_layers.healpix_blocks."):
        cls_name = target.split(".")[-1]
        if cls_name == "AvgPool":
            return "physicsnemo.nn.HEALPixAvgPool"
        if cls_name == "MaxPool":
            return "physicsnemo.nn.HEALPixMaxPool"
        return f"physicsnemo.models.dlwp_healpix.layers.{cls_name}"

    if target.startswith("physicsnemo.models.dlwp_healpix_layers.healpix_encoder."):
        cls_name = target.split(".")[-1]
        return f"physicsnemo.models.dlwp_healpix.layers.{cls_name}"

    if target.startswith("physicsnemo.models.dlwp_healpix_layers.healpix_decoder."):
        cls_name = target.split(".")[-1]
        return f"physicsnemo.models.dlwp_healpix.layers.{cls_name}"

    if target.startswith("physicsnemo.models.dlwp_healpix_layers.healpix_layers."):
        cls_name = target.split(".")[-1]
        return f"physicsnemo.nn.{cls_name}"

    if target.startswith("physicsnemo.models.dlwp_healpix_layers.normalization."):
        cls_name = target.split(".")[-1]
        return f"physicsnemo.models.dlwp_healpix.layers.normalization.{cls_name}"

    if target.startswith("physicsnemo.models.dlwp_healpix_layers.healpix_constraints."):
        cls_name = target.split(".")[-1]
        return f"physicsnemo.models.dlwp_healpix.layers.healpix_constraints.{cls_name}"

    if target.startswith("physicsnemo.models.dlwp_healpix_layers."):
        cls_name = target.split(".")[-1]
        if cls_name == "AvgPool":
            return "physicsnemo.nn.HEALPixAvgPool"
        if cls_name == "MaxPool":
            return "physicsnemo.nn.HEALPixMaxPool"
        if cls_name.startswith("HEALPix"):
            return f"physicsnemo.nn.{cls_name}"
        return f"physicsnemo.models.dlwp_healpix.layers.{cls_name}"

    if target.startswith("physicsnemo.models.dlwp_healpix.layers.healpix_blocks."):
        cls_name = target.split(".")[-1]
        if cls_name == "AvgPool":
            return "physicsnemo.nn.HEALPixAvgPool"
        if cls_name == "MaxPool":
            return "physicsnemo.nn.HEALPixMaxPool"
        return f"physicsnemo.models.dlwp_healpix.layers.{cls_name}"

    if target.startswith("physicsnemo.models.dlwp_healpix.layers.healpix_encoder."):
        cls_name = target.split(".")[-1]
        return f"physicsnemo.models.dlwp_healpix.layers.{cls_name}"

    if target.startswith("physicsnemo.models.dlwp_healpix.layers.healpix_decoder."):
        cls_name = target.split(".")[-1]
        return f"physicsnemo.models.dlwp_healpix.layers.{cls_name}"

    if target.startswith("physicsnemo.models.layers.activations"):
        cls_name = target.split(".")[-1]
        return f"physicsnemo.nn.{cls_name}"

    return target


def _remap_obj(obj):
    """
    Remapping of Dictionary and Hydra DictConfig objects to new targets.
    """
    from omegaconf import DictConfig, OmegaConf

    if isinstance(obj, DictConfig):
        container = OmegaConf.to_container(obj, resolve=False)
        remapped = _remap_obj(container)
        return OmegaConf.create(remapped)
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key == "_target_" and isinstance(value, str):
                out[key] = _remap_target(value)
            else:
                out[key] = _remap_obj(value)
        return out
    if isinstance(obj, list):
        return [_remap_obj(value) for value in obj]
    return obj


_legacy_hydra_targets_warning = "Automatically converting legacy checkpoint with deprecated `dlwp_healpix_layers` Hydra targets. Please update by saving a new checkpoint after loading the legacy checkpoint."

# Registered in __supported_model_checkpoint_version__ for "0.2.0", this fires
# unconditionally on any version-mismatched load (that's how
# Module.from_checkpoint's version-check hook works), including checkpoints
# that saved a stale "0.2.0" tag despite already using the current
# architecture (e.g. a v1.1 checkpoint saved before __model_checkpoint_version__
# was bumped to "0.3.0"). Kept intentionally neutral so it isn't misleading in
# that case; the specific, actionable warning is
# `_legacy_symmetric_skip_and_diagnostic_warning` below, which only fires when
# `_backward_compat_dlesym_v1_args` actually detects and rewrites a legacy
# architecture.
_dlesym_v02_version_mismatch_warning = (
    "Loading a checkpoint tagged with model checkpoint version 0.2.0; checking "
    "for and applying any legacy-architecture backward-compatibility fixes "
    "needed for pre-DLESyM-v1.1 checkpoints. Please update by saving a new "
    "checkpoint after loading the legacy checkpoint."
)

_legacy_symmetric_skip_and_diagnostic_warning = (
    "Automatically converting legacy checkpoint predating the "
    "`SymmetricConvNeXtBlock` skip-connection rule change and the explicit "
    "`is_diagnostic` argument. Please update by saving a new checkpoint after "
    "loading the legacy checkpoint."
)


def _inject_legacy_skip_rule(obj):
    """
    Recursively inject ``legacy_skip_rule=True`` into any ``SymmetricConvNeXtBlock``
    Hydra config found within ``obj``, so reconstructed encoder/decoder conv blocks
    use the identity-vs-conv skip rule that legacy checkpoints were actually trained
    with (``in_channels == latent_channels``) instead of the current rule
    (``in_channels == out_channels``).
    """
    from omegaconf import DictConfig, OmegaConf

    if isinstance(obj, DictConfig):
        container = OmegaConf.to_container(obj, resolve=False)
        return OmegaConf.create(_inject_legacy_skip_rule(container))
    if isinstance(obj, dict):
        out = {key: _inject_legacy_skip_rule(value) for key, value in obj.items()}
        target = out.get("_target_")
        if isinstance(target, str) and target.split(".")[-1].endswith(
            "SymmetricConvNeXtBlock"
        ):
            out["legacy_skip_rule"] = True
        return out
    if isinstance(obj, list):
        return [_inject_legacy_skip_rule(value) for value in obj]
    return obj


def _backward_compat_dlesym_v1_args(args):
    """
    Backfill ``HEALPixUNet``/``HEALPixRecUNet`` arguments for checkpoints saved
    before the DLESyM v1.1 architecture changes (commit ``94b355108``):

    - inject ``legacy_skip_rule=True`` into every ``SymmetricConvNeXtBlock``
      conv_block config in ``encoder``/``decoder``, reconstructing the actual
      trained architecture (Issue 2).
    - backfill ``is_diagnostic`` using the pre-explicit-arg heuristic when the
      checkpoint predates that parameter (Issue 3).

    Both bugs were introduced by the same commit, so whether a checkpoint
    predates it is detected structurally rather than from
    ``__model_checkpoint_version__`` alone: ``is_diagnostic`` only exists as a
    constructor parameter starting with that commit, and ``Module.__new__``
    always captures it (with its default applied) into a saved checkpoint's
    ``__args__`` once it's part of the signature. So its absence here reliably
    means "saved by code older than the commit that introduced both bugs" —
    unlike the checkpoint version tag, which can lag behind: a checkpoint
    saved by patched-but-not-yet-version-bumped code (e.g. an already-correct
    v1.1 checkpoint saved before this fix landed) still carries the old
    version tag, but it *will* have ``is_diagnostic`` in its args and must NOT
    have ``legacy_skip_rule`` forced on, or loading breaks.
    """
    args = dict(args)
    if "is_diagnostic" in args:
        return args

    warnings.warn(_legacy_symmetric_skip_and_diagnostic_warning)

    if args.get("encoder") is not None:
        args["encoder"] = _inject_legacy_skip_rule(args["encoder"])
    if args.get("decoder") is not None:
        args["decoder"] = _inject_legacy_skip_rule(args["decoder"])
    output_time_dim = args.get("output_time_dim")
    input_time_dim = args.get("input_time_dim")
    args["is_diagnostic"] = bool(
        output_time_dim == 1 and input_time_dim is not None and input_time_dim > 1
    )
    return args
