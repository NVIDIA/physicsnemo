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
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from physicsnemo.core.module import Module
from physicsnemo.experimental.models.pv_power import CrossUnet
from physicsnemo.experimental.models.pv_power.embedding import PositionalEmbedding
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
    hist_x = torch.randn(batch_size, seq_len, target_channels, generator=g).to(device)
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
        atol=2e-3,
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
    hist_x = torch.randn(4, 48, 4, generator=g).to(device)

    out = model(x_enc, None, None, hist_x)
    assert out.shape == (4, 24, 4)


def test_cross_unet_source_compatible_correlation_shape(device):
    """Correlation input has one extra target column and returns model-channel mixing."""
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

    g = torch.Generator(device="cpu").manual_seed(2)
    samples = torch.randn(2, 96, model.total_channels + 1, generator=g).to(device)

    corr = model._compute_channel_correlation(samples)
    raw_corr = torch.vmap(lambda sample: torch.corrcoef(sample.T))(samples)
    corr_to_target = torch.nan_to_num(raw_corr[:, :-1, -1], nan=0.0).clamp(min=0.0)
    expected = (
        F.softmax(corr_to_target, dim=-1)
        .unsqueeze(1)
        .repeat(1, model.total_channels, 1)
    )

    assert corr.shape == (2, model.total_channels, model.total_channels)
    assert torch.allclose(corr, expected)


def test_cross_unet_correlation_single_channel_fallback(device):
    """Direct helper calls with a single input channel should not crash."""
    model = CrossUnet(
        target_channels=1,
        weather_channels=0,
        seq_len=4,
        pred_len=4,
        seg_len=2,
        e_layers=1,
        d_model=8,
        n_heads=2,
        d_ff=16,
    ).to(device)

    samples = torch.randn(3, 4, 1, device=device)

    corr = model._compute_channel_correlation(samples)
    assert corr.shape == (3, 1, 1)
    assert torch.allclose(corr, torch.ones_like(corr))


def test_cross_unet_forward_single_target_channel_no_weather(device):
    """Forward should work for the minimal target-only channel configuration."""
    model = CrossUnet(
        target_channels=1,
        weather_channels=0,
        seq_len=12,
        pred_len=6,
        seg_len=6,
        e_layers=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
    ).to(device)

    g = torch.Generator(device="cpu").manual_seed(3)
    x_enc = torch.randn(2, 12, 1, generator=g).to(device)
    hist_x = torch.randn(2, 12, 1, generator=g).to(device)

    out = model(x_enc, None, None, hist_x)
    assert out.shape == (2, 6, 1)


def test_cross_unet_short_horizon_without_bottleneck_decoder(device):
    """Short decoder grids are allowed when the decoder does not add the bottleneck."""
    model = CrossUnet(
        target_channels=2,
        weather_channels=0,
        seq_len=96,
        pred_len=6,
        seg_len=12,
        e_layers=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        use_bottleneck_in_decoder=False,
    ).to(device)

    g = torch.Generator(device="cpu").manual_seed(4)
    x_enc = torch.randn(2, 96, 2, generator=g).to(device)
    hist_x = torch.randn(2, 96, 2, generator=g).to(device)

    out = model(x_enc, None, None, hist_x)
    assert out.shape == (2, 6, 2)


def test_cross_unet_nonlinear_correlation_projection_forward(device):
    """Nonlinear source-style correlation projection should match model channels."""
    model = CrossUnet(
        target_channels=3,
        weather_channels=2,
        seq_len=24,
        pred_len=12,
        seg_len=6,
        e_layers=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        nonlinear_correlation_proj=True,
    ).to(device)

    x_enc, w_enc, hist_w, hist_x = _make_inputs(
        device,
        batch_size=2,
        seq_len=24,
        target_channels=3,
        weather_channels=2,
    )

    out = model(x_enc, w_enc, hist_w, hist_x)
    assert out.shape == (2, 12, 5)


def test_cross_unet_nonlinear_correlation_projection_masks_after_projection(device):
    """P-corr should project raw correlations, mask negatives, then row-normalize."""
    model = CrossUnet(
        target_channels=2,
        weather_channels=0,
        seq_len=4,
        pred_len=4,
        seg_len=2,
        e_layers=1,
        d_model=8,
        n_heads=2,
        d_ff=16,
        nonlinear_correlation_proj=True,
    ).to(device)

    with torch.no_grad():
        proj1_a = model.channel_proj1[0]
        proj1_b = model.channel_proj1[2]
        proj1_a.weight.zero_()
        proj1_a.bias.copy_(
            torch.tensor([8.0, -8.0, 8.0, -8.0, 0.0, 0.0, 0.0, 0.0], device=device)
        )
        proj1_b.weight.copy_(
            torch.tensor(
                [
                    [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                ],
                device=device,
            )
        )
        proj1_b.bias.zero_()

        proj2_a = model.channel_proj2[0]
        proj2_b = model.channel_proj2[2]
        proj2_a.weight.zero_()
        proj2_a.bias.copy_(
            torch.tensor(
                [
                    8.0,
                    -8.0,
                    -8.0,
                    8.0,
                    -8.0,
                    8.0,
                    8.0,
                    -8.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                device=device,
            )
        )
        proj2_b.weight.copy_(
            torch.tensor(
                [
                    [
                        1.0,
                        -1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        1.0,
                        -1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        -1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        -1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                ],
                device=device,
            )
        )
        proj2_b.bias.zero_()

    t = torch.arange(4, dtype=torch.float32, device=device)
    samples = torch.stack([t, -t, t], dim=-1).unsqueeze(0)

    corr = model._compute_channel_correlation(samples)

    assert corr.shape == (1, 2, 2)
    assert torch.all(corr >= 0)
    assert torch.allclose(corr.sum(dim=-1), torch.ones(1, 2, device=device))
    assert corr[0, 0, 1] == 0
    assert corr[0, 1, 0] == 0


def test_cross_unet_positional_embedding_odd_d_model():
    """Odd-width sinusoidal embeddings should construct successfully."""
    embedding = PositionalEmbedding(d_model=3, max_len=8)
    out = embedding(torch.zeros(2, 5, 3))
    assert out.shape == (1, 5, 3)


def test_cross_unet_example_default_config_constructs_model():
    """The example's default Hydra config should describe a runnable model."""
    config_path = (
        Path(__file__).parents[5]
        / "examples/weather/pv_power_cross_unet/conf/config.yaml"
    )
    cfg = OmegaConf.load(config_path)

    model = CrossUnet(
        target_channels=cfg.target_channels,
        weather_channels=cfg.weather_channels,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        seg_len=cfg.seg_len,
        e_layers=cfg.e_layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        nonlinear_correlation_proj=cfg.nonlinear_correlation_proj,
        attention_kind=cfg.attention_kind,
        merge_kind=cfg.merge_kind,
        use_bottleneck_in_decoder=cfg.use_bottleneck_in_decoder,
    )

    assert model.nonlinear_correlation_proj is True


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
        model(x_enc, w_enc, hist_w, hist_x[:, :, :3])


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
