# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import hashlib

import pytest
import torch

from physicsnemo.models.healda import ObsDecoder


def test_obs_decoder_forward_pass(regtest):
    """Test ObsDecoder forward pass with realistic inputs."""
    torch.manual_seed(42)

    # Test parameters
    batch_size = 2
    time_steps = 3
    spatial_size = 16  # Coarse spatial dimension
    nobs = 50  # Number of observations
    latent_dim = 128
    metadata_dim = 32
    max_embed_id = 10
    hpx_fine_level = 5
    hpx_in_level = 6
    obs_dim = 1
    hidden_dim = 64

    # Create decoder
    decoder = ObsDecoder(
        latent_dim=latent_dim,
        metadata_dim=metadata_dim,
        max_embed_id=max_embed_id,
        hpx_fine_level=hpx_fine_level,
        hpx_in_level=hpx_in_level,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        use_layer_norm=True,
        dropout=0.1,
    )

    # Create input tensors
    latent = torch.randn(batch_size, time_steps, spatial_size, latent_dim)
    metadata = torch.randn(batch_size, time_steps, nobs, metadata_dim)

    # Create valid pixel indices for the fine latent spatial dimension
    # The fine latent has spatial size = spatial_size * factor = 16 * 2 = 32
    fine_spatial_size = spatial_size * decoder.factor
    pix = torch.randint(0, fine_spatial_size, (batch_size, time_steps, nobs))

    # Create valid embedding IDs
    embed_ids = torch.randint(0, max_embed_id, (batch_size, time_steps, nobs))

    # Forward pass in eval mode for deterministic output
    decoder.eval()
    with torch.no_grad():
        output = decoder(latent, metadata, pix, embed_ids)

        # Verify output shape and properties
        expected_shape = (batch_size, time_steps, nobs, obs_dim)
        assert output.shape == expected_shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.dtype == torch.float32

        # Test that output is deterministic with same inputs
        output2 = decoder(latent, metadata, pix, embed_ids)
        assert torch.allclose(output, output2, atol=1e-6)

        # Regression test with hash-based verification
        print(f"Output shape: {output.shape}", file=regtest)
        print(f"Output mean: {output.mean().item():.6f}", file=regtest)
        print(f"Output std: {output.std().item():.6f}", file=regtest)
        print(f"Output min: {output.min().item():.6f}", file=regtest)
        print(f"Output max: {output.max().item():.6f}", file=regtest)
        output_bytes = output.detach().cpu().numpy().tobytes()
        output_hash = hashlib.md5(output_bytes).hexdigest()
        print(f"Output hash: {output_hash}", file=regtest)

    # Test gradient flow (need to switch back to training mode)
    decoder.train()
    output_grad = decoder(latent, metadata, pix, embed_ids)
    loss = output_grad.sum()
    loss.backward()

    # Check that gradients exist for key parameters
    assert decoder.latent_proj.weight.grad is not None
    assert decoder.embed.weight.grad is not None
    assert decoder.metadata_proj.weight.grad is not None
    assert decoder.mlp[0].weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__])
