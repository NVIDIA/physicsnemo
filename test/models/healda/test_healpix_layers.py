# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
import einops
import pytest
import torch

from physicsnemo.models.healda import HPXPatchEmbed, Subdomain


def test_subdomain_select():
    x = torch.tensor([[0]])
    y = torch.tensor([[0]])
    f = torch.tensor([[0]])

    domain = Subdomain(x, y, f, n=32, level=5)

    global_ = torch.arange(12 * 4**domain.level).view(1, 1, 1, -1)
    out = domain.select_from_global(global_)
    assert out.shape == (1, 1, 1, domain.n**2 * domain.num_faces)
    assert torch.all(out.ravel() == torch.arange(domain.n**2))

    global_ = torch.zeros([1, 2, 1, 12 * 4**domain.level])
    out = domain.select_from_global(global_)
    assert out.shape[:-1] == global_.shape[:-1]
    assert out.shape[-1] == domain.n**2


@pytest.mark.parametrize("allow_nans", [False, True])
def test_hpx_patch_embed(allow_nans):
    """Test HPXPatchEmbed forward pass without subdomain."""
    in_channels = 5
    out_channels = 64
    level_fine = 6
    level_coarse = 4

    embed = HPXPatchEmbed(
        in_channels=in_channels,
        out_channels=out_channels,
        level_fine=level_fine,
        level_coarse=level_coarse,
        allow_nans=allow_nans,
    )

    b, t = 2, 3
    npix = 12 * 4**level_fine
    x = torch.randn(b, in_channels, t, npix)

    # Add some NaNs when testing allow_nans mode
    if allow_nans:
        # Add NaNs to ~10% of pixels
        nan_mask = torch.rand(b, in_channels, t, npix) < 0.1
        x[nan_mask] = float("nan")

    second_of_day = torch.randint(0, 86400, (b, t))
    day_of_year = torch.randint(0, 365, (b, t))

    out = embed(x, second_of_day=second_of_day, day_of_year=day_of_year)

    expected_npix = 12 * 4**level_coarse
    assert out.shape == (b, t, expected_npix, out_channels)

    # Verify output is not NaN and has reasonable magnitude (O(1))
    assert not torch.isnan(out).any(), "Output contains NaNs"
    assert torch.isfinite(out).all(), "Output contains non-finite values"
    max_val = torch.abs(out).max().item()
    assert max_val < 100, f"Output magnitude should be O(1), got max = {max_val}"


def test_hpx_patch_embed_empty_patches():
    """Test HPXPatchEmbed with completely empty patches (all NaN) returns null token."""
    in_channels = 5
    out_channels = 64
    level_fine = 6
    level_coarse = 4
    patch_size = 2 ** (level_fine - level_coarse)

    embed = HPXPatchEmbed(
        in_channels=in_channels,
        out_channels=out_channels,
        level_fine=level_fine,
        level_coarse=level_coarse,
        allow_nans=True,
        use_gains=True,
    )
    embed.pos_embed_gain.data.fill_(0)
    embed.calendar_embed_gain.data.fill_(0)
    embed.null_token.data.fill_(0)

    b, t = 2, 3
    side_fine = 2**level_fine
    x = torch.randn(b, in_channels, t, 12, side_fine, side_fine)

    # Set entire patches to be emptyface
    num_empty_patches = 3
    x[:, :, :, :num_empty_patches, :patch_size, :patch_size] = float("nan")

    # Flatten back to expected input shape
    x = einops.rearrange(x, "b c t f x y -> b c t (f x y)")

    second_of_day = torch.randint(0, 86400, (b, t))
    day_of_year = torch.randint(0, 365, (b, t))

    out = embed(x, second_of_day=second_of_day, day_of_year=day_of_year)

    out.sum().backward()

    # Verify output is not NaN and that empty patches are null tokens (zeros)
    assert not torch.isnan(out).any(), "Output contains NaNs"
    assert torch.isfinite(out).all(), "Output contains non-finite values"
    for name, param in embed.named_parameters():
        if param.grad is not None:
            all_valid = torch.isfinite(param.grad).all()
            assert all_valid, f"Parameter {name} has non-finite gradients"

    # At least some patches should be null tokens (zeros) since we made input patches completely empty
    # Check that there are some zero patches (null tokens)
    zero_patches = torch.all(
        out == 0.0, dim=-1
    )  # (b, t, npix) - True where patch is all zeros
    assert zero_patches.any(), (
        "At least some patches should be null tokens (zeros) for empty input patches"
    )


def test_hpx_patch_embed_subdomain():
    """Test HPXPatchEmbed forward pass with subdomain."""
    in_channels = 5
    out_channels = 64
    level_fine = 6
    level_coarse = 4

    embed = HPXPatchEmbed(
        in_channels=in_channels,
        out_channels=out_channels,
        level_fine=level_fine,
        level_coarse=level_coarse,
    )

    b, t = 2, 3
    subdomain = Subdomain(
        x=torch.zeros([b, 12], dtype=torch.int32),
        y=torch.zeros([b, 12], dtype=torch.int32),
        f=torch.arange(12).unsqueeze(0).expand(b, -1),
        n=32,
        level=level_fine,
    )
    subdomain_npix = subdomain.num_faces * subdomain.n * subdomain.n
    x_subdomain = torch.randn(b, in_channels, t, subdomain_npix)
    second_of_day = torch.randint(0, 86400, (b, t))
    day_of_year = torch.randint(0, 365, (b, t))

    out_subdomain = embed(
        x_subdomain,
        second_of_day=second_of_day,
        day_of_year=day_of_year,
        subdomain=subdomain,
    )

    # Subdomain is coarsened by factor 2^(level_fine - level_coarse)
    coarsen_factor = 2 ** (level_fine - level_coarse)
    n_coarse = subdomain.n // coarsen_factor
    expected_subdomain_npix = subdomain.num_faces * n_coarse * n_coarse
    assert out_subdomain.shape == (b, t, expected_subdomain_npix, out_channels)
