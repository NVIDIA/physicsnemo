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

# NOTE on regression-data generation:
#   ``validate_forward_accuracy`` creates the reference ``.pth`` if missing and
#   then errors. To regenerate references after an intentional model change,
#   delete the corresponding ``data/cross_unet_*_output.pth`` and re-run the
#   test twice (the first pass writes; the second passes).

import random

import pytest
import torch

from physicsnemo.core.module import Module
from physicsnemo.experimental.models.pv_power import CrossUnet
from test.common import validate_checkpoint, validate_forward_accuracy


def _make_inputs(
    device,
    *,
    batch_size: int = 4,
    seq_len: int = 96,
    target_channels: int = 4,
    weather_channels: int = 3,
):
    """Deterministic sample inputs for the standard CrossUnet config."""
    g = torch.Generator(device="cpu").manual_seed(0)
    x_enc = torch.randn(batch_size, seq_len, target_channels, generator=g).to(device)
    w_enc = torch.randn(batch_size, seq_len, weather_channels, generator=g).to(device)
    hist_w = torch.randn(batch_size, seq_len, weather_channels, generator=g).to(device)
    hist_x = torch.randn(batch_size, seq_len, target_channels - 1, generator=g).to(
        device
    )
    return x_enc, w_enc, hist_w, hist_x


@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"],
)
def test_cross_unet_constructor(config):
    """Exercise the CrossUnet constructor over default and custom configurations."""
    if config == "default":
        model = CrossUnet(
            target_channels=4,
            weather_channels=3,
            seq_len=96,
            pred_len=16,
            seg_len=12,
            e_layers=3,
            d_model=32,
            n_heads=4,
            d_ff=64,
        )
        # Default attribute values
        assert model.target_channels == 4
        assert model.weather_channels == 3
        assert model.total_channels == 7
        assert model.seq_len == 96
        assert model.pred_len == 16
        assert model.use_weather is True
        assert model.nonlinear_correlation_proj is False
        assert len(model.encoder.encode_blocks) == 3
        assert len(model.decoder.decode_layers) == 4  # e_layers + 1
    else:
        model = CrossUnet(
            target_channels=3,
            weather_channels=0,
            seq_len=48,
            pred_len=24,
            seg_len=6,
            e_layers=2,
            d_model=24,
            n_heads=3,
            d_ff=48,
            dropout=0.1,
            nonlinear_correlation_proj=True,
            swap_corr_axis=True,
            merge_kind="cnn_merge",
            attention_kind="parallel",
            use_bottleneck_in_decoder=False,
        )
        assert model.target_channels == 3
        assert model.weather_channels == 0
        assert model.total_channels == 3
        assert model.use_weather is False
        assert model.nonlinear_correlation_proj is True
        assert hasattr(model, "channel_proj1")
        assert hasattr(model, "channel_proj2")
        assert len(model.encoder.encode_blocks) == 2
        assert len(model.decoder.decode_layers) == 3
        assert model.decoder.use_bottleneck is False

    # Common invariants
    assert isinstance(model, Module)
    assert hasattr(model, "meta")
    assert model.meta.jit is False
    assert model.meta.cuda_graphs is False


def test_cross_unet_forward_regression(device):
    """Compare forward output against committed reference data."""
    torch.manual_seed(0)
    model = CrossUnet(
        target_channels=4,
        weather_channels=3,
        seq_len=96,
        pred_len=16,
        seg_len=12,
        e_layers=3,
        d_model=32,
        n_heads=4,
        d_ff=64,
    ).to(device)

    x_enc, w_enc, hist_w, hist_x = _make_inputs(device)
    assert validate_forward_accuracy(
        model,
        (x_enc, w_enc, hist_w, hist_x),
        file_name="experimental/models/pv_power/cross_unet/data/cross_unet_default_output.pth",
        atol=1e-3,
    )


def test_cross_unet_forward_no_weather(device):
    """Forward path with ``weather_channels=0`` (correlation built from target only)."""
    torch.manual_seed(0)
    model = CrossUnet(
        target_channels=4,
        weather_channels=0,
        seq_len=48,
        pred_len=24,
        seg_len=6,
        e_layers=2,
        d_model=24,
        n_heads=3,
        d_ff=48,
    ).to(device)

    g = torch.Generator(device="cpu").manual_seed(1)
    x_enc = torch.randn(4, 48, 4, generator=g).to(device)
    hist_x = torch.randn(4, 48, 3, generator=g).to(device)

    out = model(x_enc, None, None, hist_x)
    assert out.shape == (4, 24, 4)


def test_cross_unet_checkpoint(device):
    """Save model_1 to .mdlus and verify model_2 reproduces its forward output."""
    kwargs = dict(
        target_channels=4,
        weather_channels=3,
        seq_len=96,
        pred_len=16,
        seg_len=12,
        e_layers=3,
        d_model=32,
        n_heads=4,
        d_ff=64,
    )
    torch.manual_seed(0)
    model_1 = CrossUnet(**kwargs).to(device)
    torch.manual_seed(1)
    model_2 = CrossUnet(**kwargs).to(device)

    bsize = random.randint(1, 4)
    x_enc, w_enc, hist_w, hist_x = _make_inputs(device, batch_size=bsize)

    assert validate_checkpoint(model_1, model_2, (x_enc, w_enc, hist_w, hist_x))


def test_cross_unet_forward_validation(device):
    """Forward should raise on shape mismatches."""
    model = CrossUnet(
        target_channels=4,
        weather_channels=3,
        seq_len=96,
        pred_len=16,
        seg_len=12,
        e_layers=3,
        d_model=32,
        n_heads=4,
        d_ff=64,
    ).to(device)

    x_enc, w_enc, hist_w, hist_x = _make_inputs(device)

    # Wrong target_channels
    with pytest.raises(ValueError, match="x_enc"):
        model(x_enc[:, :, :3], w_enc, hist_w, hist_x)

    # Missing weather inputs when weather_channels > 0
    with pytest.raises(ValueError, match="weather_channels"):
        model(x_enc, None, None, hist_x)

    # Wrong seq_x_hist channel count
    with pytest.raises(ValueError, match="seq_x_hist"):
        model(x_enc, w_enc, hist_w, hist_x[:, :, :2])


def test_cross_unet_constructor_validation():
    """Constructor should reject obviously invalid inputs."""
    with pytest.raises(ValueError, match="target_channels"):
        CrossUnet(target_channels=0, weather_channels=3, seq_len=96, pred_len=16)
    with pytest.raises(ValueError, match="weather_channels"):
        CrossUnet(target_channels=4, weather_channels=-1, seq_len=96, pred_len=16)
    with pytest.raises(ValueError, match="seq_len|pred_len|seg_len"):
        CrossUnet(target_channels=4, weather_channels=3, seq_len=0, pred_len=16)
    with pytest.raises(ValueError, match="e_layers"):
        CrossUnet(
            target_channels=4, weather_channels=3, seq_len=96, pred_len=16, e_layers=0
        )
