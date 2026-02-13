# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pytest
import torch

from physicsnemo.experimental.models.healda import (
    MultiSensorObsEmbedder,
    SensorEmbedder,
)
from physicsnemo.experimental.models.healda.point_embed import _split_by_sensor
from test import common


def check_all_params_have_gradients(model: torch.nn.Module) -> tuple[bool, list[str]]:
    """Return whether all trainable params got gradients and list missing names."""
    params_without_grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            params_without_grads.append(name)
    return len(params_without_grads) == 0, params_without_grads


def _build_flattened_obs(
    counts: list[list[list[int]]],
    *,
    nchannel_per_sensor: list[int],
    nplatform_per_sensor: list[int],
    npix: int,
    meta_dim: int = 4,
    device: str | None = None,
):
    """Build deterministic flattened tensors and cumulative-end offsets in ``(S, B, T)``."""
    counts_t = torch.as_tensor(counts, dtype=torch.long)
    if counts_t.ndim != 3:
        raise ValueError(
            f"counts must have shape (S, B, T), got {tuple(counts_t.shape)}"
        )

    s, b, t = counts_t.shape
    offsets = torch.cumsum(counts_t.reshape(-1), dim=0).reshape(s, b, t)
    nobs = int(offsets[-1, -1, -1].item()) if offsets.numel() > 0 else 0
    obs = torch.arange(nobs, dtype=torch.float32)
    float_metadata = torch.randn((nobs, meta_dim), dtype=torch.float32)
    pix = torch.randint(0, npix, (nobs,), dtype=torch.long)
    local_channel = torch.zeros((nobs,), dtype=torch.long)
    local_platform = torch.zeros((nobs,), dtype=torch.long)
    obs_type = torch.randint(0, 256, (nobs,), dtype=torch.long)

    row = 0
    for sensor_idx in range(s):
        for batch_idx in range(b):
            for time_idx in range(t):
                n = int(counts_t[sensor_idx, batch_idx, time_idx].item())
                if n == 0:
                    continue

                sl = slice(row, row + n)
                local_channel[sl] = torch.randint(
                    0, nchannel_per_sensor[sensor_idx], (n,), dtype=torch.long
                )
                local_platform[sl] = torch.randint(
                    0, nplatform_per_sensor[sensor_idx], (n,), dtype=torch.long
                )
                row += n

    if device is not None:
        obs = obs.to(device)
        float_metadata = float_metadata.to(device)
        pix = pix.to(device)
        local_channel = local_channel.to(device)
        local_platform = local_platform.to(device)
        obs_type = obs_type.to(device)
        offsets = offsets.to(device)

    return obs, float_metadata, pix, local_channel, local_platform, obs_type, offsets


def test_split_by_sensor():
    counts = [
        [[2, 1], [0, 3]],  # sensor 0
        [[0, 0], [0, 0]],  # sensor 1 (no observations)
        [[1, 0], [2, 1]],  # sensor 2
    ]
    obs, float_metadata, pix, local_channel, local_platform, obs_type, offsets = (
        _build_flattened_obs(
            counts,
            nchannel_per_sensor=[4, 3, 2],
            nplatform_per_sensor=[3, 2, 1],
            npix=12 * 4**3,
            meta_dim=4,
        )
    )

    split = _split_by_sensor(
        obs=obs,
        float_metadata=float_metadata,
        pix=pix,
        local_channel=local_channel,
        local_platform=local_platform,
        obs_type=obs_type,
        offsets=offsets,
    )
    assert len(split) == len(counts)

    expected_lens = [sum(sum(window) for window in sensor) for sensor in counts]
    for sensor_slice, expected_len in zip(split, expected_lens):
        assert sensor_slice[0].shape[0] == expected_len
        assert sensor_slice[-1][-1, -1].item() == expected_len

    # Re-assemble split tensors and verify exact round-trip on the row axis.
    reconstructed_obs = torch.cat([sensor_slice[0] for sensor_slice in split], dim=0)
    reconstructed_meta = torch.cat([sensor_slice[1] for sensor_slice in split], dim=0)
    assert torch.equal(reconstructed_obs, obs)
    assert torch.equal(reconstructed_meta, float_metadata)


@pytest.mark.parametrize("nobs", [0, 64])
def test_sensor_embedder_forward_and_gradients(device, nobs):
    torch.manual_seed(0)
    b, t = 2, 1
    npix = 12 * 4**3
    out_dim = 32
    meta_dim = 4
    nchannel = 8
    nplatform = 3

    if nobs == 0:
        counts = [[[0], [0]]]
    else:
        counts = [[[nobs // 2], [nobs - nobs // 2]]]

    obs, float_metadata, pix, local_channel, local_platform, obs_type, offsets = (
        _build_flattened_obs(
            counts,
            nchannel_per_sensor=[nchannel],
            nplatform_per_sensor=[nplatform],
            npix=npix,
            meta_dim=meta_dim,
            device=device,
        )
    )

    embedder = SensorEmbedder(
        nplatform=nplatform,
        nchannel=nchannel,
        sensor_embed_dim=16,
        output_dim=out_dim,
        meta_dim=meta_dim,
        n_embed=256,
    ).to(device)
    embedder.train()
    # SensorEmbedder expects 2D offsets (B, T); squeeze the single-sensor dim
    out = embedder(
        obs=obs,
        float_metadata=float_metadata,
        pix=pix,
        local_channel=local_channel,
        local_platform=local_platform,
        obs_type=obs_type,
        offsets=offsets.squeeze(0),
        npix=npix,
    )
    assert out.shape == (b, t, npix, out_dim)
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()
    all_have_grads, missing = check_all_params_have_gradients(embedder)
    assert all_have_grads, f"Parameters without gradients (nobs={nobs}): {missing}"


@pytest.mark.parametrize("counts", [[[[3], [2]], [[1], [4]]], [[[0], [0]], [[0], [0]]]])
def test_multisensor_obs_embedding_forward_and_gradients(device, counts):
    torch.manual_seed(0)
    npix = 12 * 4**3
    meta_dim = 4
    fusion_dim = 32
    nchannel_per_sensor = [7, 5]
    nplatform_per_sensor = [3, 2]

    obs, float_metadata, pix, local_channel, local_platform, obs_type, offsets = (
        _build_flattened_obs(
            counts,
            nchannel_per_sensor=nchannel_per_sensor,
            nplatform_per_sensor=nplatform_per_sensor,
            npix=npix,
            meta_dim=meta_dim,
            device=device,
        )
    )

    model = MultiSensorObsEmbedder(
        nchannel_per_sensor=nchannel_per_sensor,
        nplatform_per_sensor=nplatform_per_sensor,
        embed_dim=16,
        meta_dim=meta_dim,
        fusion_dim=fusion_dim,
        torch_compile=False,
    ).to(device)
    model.train()
    out = model(
        obs=obs,
        float_metadata=float_metadata,
        pix=pix,
        local_channel=local_channel,
        local_platform=local_platform,
        obs_type=obs_type,
        offsets=offsets,
        npix=npix,
    )
    assert out.shape == (2, fusion_dim, 1, npix)
    assert torch.isfinite(out).all()  # verify no NaNs

    loss = out.sum()
    loss.backward()
    all_have_grads, missing = check_all_params_have_gradients(model)
    assert all_have_grads, f"Parameters without gradients: {missing}"


def test_multisensor_obs_embedding_forward_accuracy(device):
    """Regression test for MultiSensorObsEmbedder forward output."""
    torch.manual_seed(0)
    npix = 12 * 4**3
    meta_dim = 4
    fusion_dim = 32
    nchannel_per_sensor = [7, 5]
    nplatform_per_sensor = [3, 2]
    counts = [[[3], [2]], [[1], [4]]]
    obs, float_metadata, pix, local_channel, local_platform, obs_type, offsets = (
        _build_flattened_obs(
            counts,
            nchannel_per_sensor=nchannel_per_sensor,
            nplatform_per_sensor=nplatform_per_sensor,
            npix=npix,
            meta_dim=meta_dim,
            device=device,
        )
    )

    model = MultiSensorObsEmbedder(
        nchannel_per_sensor=nchannel_per_sensor,
        nplatform_per_sensor=nplatform_per_sensor,
        embed_dim=16,
        meta_dim=meta_dim,
        fusion_dim=fusion_dim,
    ).to(device)
    model.eval()

    assert common.validate_forward_accuracy(
        model,
        (
            obs,
            float_metadata,
            pix,
            local_channel,
            local_platform,
            obs_type,
            offsets,
            npix,
        ),
        file_name="models/healda/data/point_embed_multisensor_output.pth",
    )
