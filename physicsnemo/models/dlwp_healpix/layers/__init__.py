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
from .healpix_decoder import UNetDecoder
from .healpix_encoder import UNetEncoder

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
    "HEALPixFoldFaces",
    "HEALPixLayer",
    "HEALPixPadding",
    "HEALPixPaddingv2",
    "HEALPixUnfoldFaces",
    "HEALPixMaxPool",
    "HEALPixAvgPool",
]


# Remapping methods for backwards compatibility of legacy checkpoints


def _remap_target(target: str) -> str:
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
