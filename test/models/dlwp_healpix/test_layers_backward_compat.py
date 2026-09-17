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

"""Tests for the legacy Hydra ``_target_`` remapping helpers in
``physicsnemo.models.dlwp_healpix.layers`` (``_remap_target``, ``_remap_obj``),
used by ``HEALPixUNet``/``HEALPixRecUNet`` to load ``0.1.0``-era checkpoints.
"""

import omegaconf
import pytest

from physicsnemo.models.dlwp_healpix.layers import (
    _legacy_hydra_targets_warning,
    _remap_obj,
    _remap_target,
)


def test_legacy_hydra_targets_warning_is_nonempty_string():
    assert isinstance(_legacy_hydra_targets_warning, str)
    assert len(_legacy_hydra_targets_warning) > 0


@pytest.mark.parametrize(
    "target,expected",
    [
        # explicit dict entries take precedence over the generic
        # `healpix_encoder.`/`healpix_decoder.` prefix handling below
        (
            "physicsnemo.models.dlwp_healpix_layers.healpix_encoder.UNetEncoder",
            "physicsnemo.models.dlwp_healpix.layers.UNetEncoder",
        ),
        (
            "physicsnemo.models.dlwp_healpix_layers.healpix_decoder.UNetDecoder",
            "physicsnemo.models.dlwp_healpix.layers.UNetDecoder",
        ),
        # `dlwp_healpix_layers.healpix_blocks.` -> generic class
        (
            "physicsnemo.models.dlwp_healpix_layers.healpix_blocks.ConvNeXtBlock",
            "physicsnemo.models.dlwp_healpix.layers.ConvNeXtBlock",
        ),
        # `dlwp_healpix_layers.healpix_blocks.` -> AvgPool/MaxPool special-cased
        (
            "physicsnemo.models.dlwp_healpix_layers.healpix_blocks.AvgPool",
            "physicsnemo.nn.HEALPixAvgPool",
        ),
        (
            "physicsnemo.models.dlwp_healpix_layers.healpix_blocks.MaxPool",
            "physicsnemo.nn.HEALPixMaxPool",
        ),
        # `dlwp_healpix_layers.healpix_encoder.`/`healpix_decoder.` with a
        # non-explicit class name still maps generically
        (
            "physicsnemo.models.dlwp_healpix_layers.healpix_encoder.SomeHelper",
            "physicsnemo.models.dlwp_healpix.layers.SomeHelper",
        ),
        (
            "physicsnemo.models.dlwp_healpix_layers.healpix_decoder.SomeHelper",
            "physicsnemo.models.dlwp_healpix.layers.SomeHelper",
        ),
        # `dlwp_healpix_layers.healpix_layers.` -> physicsnemo.nn
        (
            "physicsnemo.models.dlwp_healpix_layers.healpix_layers.HEALPixLayer",
            "physicsnemo.nn.HEALPixLayer",
        ),
        # `dlwp_healpix_layers.normalization.` -> layers.normalization
        (
            "physicsnemo.models.dlwp_healpix_layers.normalization.ConditionalLayerNorm",
            "physicsnemo.models.dlwp_healpix.layers.normalization.ConditionalLayerNorm",
        ),
        # `dlwp_healpix_layers.healpix_constraints.` -> layers.healpix_constraints
        (
            "physicsnemo.models.dlwp_healpix_layers.healpix_constraints.NonnegativeConstraint",
            "physicsnemo.models.dlwp_healpix.layers.healpix_constraints.NonnegativeConstraint",
        ),
        # bare `dlwp_healpix_layers.` fallback, generic class
        (
            "physicsnemo.models.dlwp_healpix_layers.SomeOtherClass",
            "physicsnemo.models.dlwp_healpix.layers.SomeOtherClass",
        ),
        # bare `dlwp_healpix_layers.` fallback, AvgPool/MaxPool special-cased
        (
            "physicsnemo.models.dlwp_healpix_layers.AvgPool",
            "physicsnemo.nn.HEALPixAvgPool",
        ),
        (
            "physicsnemo.models.dlwp_healpix_layers.MaxPool",
            "physicsnemo.nn.HEALPixMaxPool",
        ),
        # bare `dlwp_healpix_layers.` fallback, class name starting with
        # "HEALPix" maps to physicsnemo.nn
        (
            "physicsnemo.models.dlwp_healpix_layers.HEALPixFoldFaces",
            "physicsnemo.nn.HEALPixFoldFaces",
        ),
        # new-style `dlwp_healpix.layers.healpix_blocks.` still gets
        # normalized to the top-level `layers` module, generic class
        (
            "physicsnemo.models.dlwp_healpix.layers.healpix_blocks.ConvNeXtBlock",
            "physicsnemo.models.dlwp_healpix.layers.ConvNeXtBlock",
        ),
        # new-style `dlwp_healpix.layers.healpix_blocks.` AvgPool/MaxPool
        (
            "physicsnemo.models.dlwp_healpix.layers.healpix_blocks.AvgPool",
            "physicsnemo.nn.HEALPixAvgPool",
        ),
        (
            "physicsnemo.models.dlwp_healpix.layers.healpix_blocks.MaxPool",
            "physicsnemo.nn.HEALPixMaxPool",
        ),
        # new-style `dlwp_healpix.layers.healpix_encoder.`/`healpix_decoder.`
        (
            "physicsnemo.models.dlwp_healpix.layers.healpix_encoder.UNetEncoder",
            "physicsnemo.models.dlwp_healpix.layers.UNetEncoder",
        ),
        (
            "physicsnemo.models.dlwp_healpix.layers.healpix_decoder.UNetDecoder",
            "physicsnemo.models.dlwp_healpix.layers.UNetDecoder",
        ),
        # legacy activations module
        (
            "physicsnemo.models.layers.activations.CappedGELU",
            "physicsnemo.nn.CappedGELU",
        ),
        # anything unrecognized passes through unchanged
        (
            "physicsnemo.models.dlwp_healpix.layers.UNetEncoder",
            "physicsnemo.models.dlwp_healpix.layers.UNetEncoder",
        ),
        (
            "some.unrelated.module.Class",
            "some.unrelated.module.Class",
        ),
    ],
)
def test_remap_target(target, expected):
    assert _remap_target(target) == expected


def test_remap_obj_passthrough_for_scalars():
    assert _remap_obj(3) == 3
    assert _remap_obj("plain string") == "plain string"
    assert _remap_obj(None) is None


def test_remap_obj_dict_remaps_target_key_only():
    obj = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_blocks.ConvNeXtBlock",
        "in_channels": 3,
        "nested": {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.normalization.ConditionalLayerNorm",
            "channel_depth": 4,
        },
    }

    remapped = _remap_obj(obj)

    assert (
        remapped["_target_"] == "physicsnemo.models.dlwp_healpix.layers.ConvNeXtBlock"
    )
    # non-`_target_` keys are recursed into but otherwise left untouched
    assert remapped["in_channels"] == 3
    assert (
        remapped["nested"]["_target_"]
        == "physicsnemo.models.dlwp_healpix.layers.normalization.ConditionalLayerNorm"
    )
    assert remapped["nested"]["channel_depth"] == 4


def test_remap_obj_ignores_non_string_target_value():
    # a `_target_` key whose value isn't a string (e.g. already resolved to
    # a class object) must be recursed into rather than passed to
    # `_remap_target`, which only accepts strings
    obj = {"_target_": 42}
    assert _remap_obj(obj) == {"_target_": 42}


def test_remap_obj_list_remaps_each_element():
    obj = [
        {"_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_blocks.AvgPool"},
        {"_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_blocks.MaxPool"},
        "unchanged",
    ]

    remapped = _remap_obj(obj)

    assert remapped[0]["_target_"] == "physicsnemo.nn.HEALPixAvgPool"
    assert remapped[1]["_target_"] == "physicsnemo.nn.HEALPixMaxPool"
    assert remapped[2] == "unchanged"


def test_remap_obj_dictconfig_roundtrips_through_container():
    cfg = omegaconf.OmegaConf.create(
        {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_encoder.UNetEncoder",
            "conv_block": {
                "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_blocks.ConvNeXtBlock",
            },
        }
    )

    remapped = _remap_obj(cfg)

    assert isinstance(remapped, omegaconf.DictConfig)
    assert remapped._target_ == "physicsnemo.models.dlwp_healpix.layers.UNetEncoder"
    assert (
        remapped.conv_block._target_
        == "physicsnemo.models.dlwp_healpix.layers.ConvNeXtBlock"
    )
