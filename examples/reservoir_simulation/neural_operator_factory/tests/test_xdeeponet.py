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

"""Unit tests for xDeepONet model variants (2D and 3D)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xdeeponet import (
    DeepONet,
    DeepONet3D,
    DeepONet3DWrapper,
    DeepONetWrapper,
    MLPBranch,
    SpatialBranch,
    SpatialBranch3D,
    TrunkNet,
)

BRANCH1_SPATIAL = {
    "encoder": {"type": "linear", "activation_fn": "relu"},
    "layers": {
        "num_fourier_layers": 0,
        "num_unet_layers": 1,
        "num_conv_layers": 0,
        "modes1": 4,
        "modes2": 4,
        "kernel_size": 3,
        "dropout": 0.0,
        "unet_impl": "custom",
        "activation_fn": "relu",
    },
}
BRANCH1_MLP = {
    "encoder": {
        "type": "mlp",
        "hidden_width": 32,
        "num_layers": 2,
        "activation_fn": "relu",
    },
    "layers": {"num_fourier_layers": 0, "num_unet_layers": 0, "num_conv_layers": 0},
}
BRANCH2_SPATIAL = {
    "encoder": {"type": "linear", "activation_fn": "relu"},
    "layers": {
        "num_fourier_layers": 0,
        "num_unet_layers": 1,
        "num_conv_layers": 0,
        "modes1": 4,
        "modes2": 4,
        "kernel_size": 3,
        "dropout": 0.0,
        "unet_impl": "custom",
        "activation_fn": "relu",
    },
}
BRANCH2_MLP = {
    "encoder": {
        "type": "mlp",
        "hidden_width": 32,
        "num_layers": 2,
        "activation_fn": "relu",
    },
    "layers": {"num_fourier_layers": 0, "num_unet_layers": 0, "num_conv_layers": 0},
}
TRUNK = {
    "input_type": "time",
    "hidden_width": 32,
    "num_layers": 2,
    "activation_fn": "tanh",
}


def _init_lazy(model, x, **kwargs):
    """Run one forward pass to initialise LazyLinear modules."""
    with torch.no_grad():
        model(x, **kwargs)


class TestTrunkNet:
    """Tests for TrunkNet."""

    def test_output_shape(self):
        """Verify TrunkNet output shape matches expected features."""
        trunk = TrunkNet(in_features=1, out_features=32, hidden_width=16, num_layers=3)
        x = torch.randn(10, 1)
        assert trunk(x).shape == (10, 32)

    def test_grid_input(self):
        """Verify TrunkNet handles multi-dimensional grid input correctly."""
        trunk = TrunkNet(in_features=4, out_features=64, hidden_width=32, num_layers=2)
        x = torch.randn(5, 4)
        assert trunk(x).shape == (5, 64)


class TestMLPBranch:
    """Tests for MLPBranch."""

    def test_output_shape(self):
        """Verify MLPBranch output shape matches expected features."""
        branch = MLPBranch(out_features=32, hidden_width=16, num_layers=3)
        x = torch.randn(2, 50)
        out = branch(x)
        assert out.shape == (2, 32)


class TestSpatialBranch2D:
    """Tests for 2D SpatialBranch."""

    def test_output_shape(self):
        """Verify 2D SpatialBranch output shape matches expected width."""
        branch = SpatialBranch(
            in_channels=5,
            width=16,
            num_unet_layers=1,
            kernel_size=3,
            unet_impl="custom",
            activation_fn="relu",
        )
        x = torch.randn(2, 16, 24, 5)
        _init_lazy(branch, x)
        out = branch(x)
        assert out.shape == (2, 16, 24, 16)


class TestSpatialBranch3D:
    """Tests for 3D SpatialBranch."""

    def test_output_shape(self):
        """Verify 3D SpatialBranch output shape matches expected width."""
        branch = SpatialBranch3D(
            in_channels=5,
            width=16,
            num_unet_layers=1,
            kernel_size=3,
            unet_impl="custom",
            activation_fn="relu",
        )
        x = torch.randn(2, 8, 16, 8, 5)
        _init_lazy(branch, x)
        out = branch(x)
        assert out.shape == (2, 8, 16, 8, 16)


SINGLE_BRANCH_VARIANTS = ["deeponet", "u_deeponet", "conv_deeponet"]
DUAL_BRANCH_VARIANTS = ["mionet", "tno"]


class TestDeepONetWrapper2D:
    """Tests for 2D DeepONet wrapper."""

    @pytest.mark.parametrize("variant", SINGLE_BRANCH_VARIANTS)
    def test_forward_shape_single_branch(self, variant):
        """Verify 2D single-branch forward pass produces correct output shape."""
        B, H, W, T, C = 2, 16, 24, 4, 5
        model = DeepONetWrapper(
            padding=8,
            variant=variant,
            width=32,
            branch1_config=BRANCH1_SPATIAL,
            trunk_config=TRUNK,
        )
        x = torch.randn(B, H, W, T, C)
        _init_lazy(model, x)
        out = model(x)
        assert out.shape == (B, H, W, T)

    @pytest.mark.parametrize("variant", DUAL_BRANCH_VARIANTS)
    def test_forward_shape_dual_branch(self, variant):
        """Verify 2D dual-branch forward pass produces correct output shape."""
        B, H, W, T, C = 2, 16, 24, 4, 5
        model = DeepONetWrapper(
            padding=8,
            variant=variant,
            width=32,
            branch1_config=BRANCH1_SPATIAL,
            branch2_config=BRANCH2_SPATIAL,
            trunk_config=TRUNK,
        )
        x = torch.randn(B, H, W, T, C)
        b2 = torch.randn(B, H, W, T)
        _init_lazy(model, x, x_branch2=b2)
        out = model(x, x_branch2=b2)
        assert out.shape == (B, H, W, T)

    def test_target_times_changes_output_T(self):
        """Verify target_times overrides the temporal output dimension size."""
        B, H, W, T_in, C = 2, 16, 24, 2, 5
        K = 5
        model = DeepONetWrapper(
            padding=8,
            variant="u_deeponet",
            width=32,
            branch1_config=BRANCH1_SPATIAL,
            trunk_config=TRUNK,
        )
        x = torch.randn(B, H, W, T_in, C)
        tt = torch.linspace(0, 1, K)
        _init_lazy(model, x)
        out = model(x, target_times=tt)
        assert out.shape == (B, H, W, K)

    def test_invalid_variant_raises(self):
        """Verify ValueError is raised for an unknown DeepONet variant."""
        with pytest.raises(ValueError, match="Unknown variant"):
            DeepONetWrapper(
                variant="invalid",
                width=32,
                branch1_config=BRANCH1_SPATIAL,
                trunk_config=TRUNK,
            )

    def test_count_params(self):
        """Verify count_params returns a positive parameter count for 2D wrapper."""
        model = DeepONetWrapper(
            padding=8,
            variant="deeponet",
            width=32,
            branch1_config=BRANCH1_SPATIAL,
            trunk_config=TRUNK,
        )
        x = torch.randn(1, 16, 24, 2, 5)
        _init_lazy(model, x)
        assert model.count_params() > 0

    def test_gradient_flow(self):
        """Verify gradients propagate through the 2D DeepONet wrapper."""
        model = DeepONetWrapper(
            padding=8,
            variant="u_deeponet",
            width=32,
            branch1_config=BRANCH1_SPATIAL,
            trunk_config=TRUNK,
        )
        x = torch.randn(1, 16, 24, 2, 5)
        _init_lazy(model, x)
        x = torch.randn(1, 16, 24, 2, 5, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None


BRANCH1_3D = {
    "encoder": {"type": "linear", "activation_fn": "relu"},
    "layers": {
        "num_fourier_layers": 0,
        "num_unet_layers": 1,
        "num_conv_layers": 0,
        "modes1": 4,
        "modes2": 4,
        "modes3": 4,
        "kernel_size": 3,
        "dropout": 0.0,
        "unet_impl": "custom",
        "activation_fn": "relu",
    },
}
BRANCH2_3D = {
    "encoder": {"type": "linear", "activation_fn": "relu"},
    "layers": {
        "num_fourier_layers": 0,
        "num_unet_layers": 1,
        "num_conv_layers": 0,
        "modes1": 4,
        "modes2": 4,
        "modes3": 4,
        "kernel_size": 3,
        "dropout": 0.0,
        "unet_impl": "custom",
        "activation_fn": "relu",
    },
}


class TestDeepONet3DWrapper:
    """Tests for 3D DeepONet wrapper."""

    @pytest.mark.parametrize("variant", SINGLE_BRANCH_VARIANTS)
    def test_forward_shape_single_branch(self, variant):
        """Verify 3D single-branch forward pass produces correct output shape."""
        B, X, Y, Z, T, C = 1, 8, 16, 8, 3, 5
        model = DeepONet3DWrapper(
            padding=8,
            variant=variant,
            width=32,
            branch1_config=BRANCH1_3D,
            trunk_config=TRUNK,
        )
        x = torch.randn(B, X, Y, Z, T, C)
        _init_lazy(model, x)
        out = model(x)
        assert out.shape == (B, X, Y, Z, T)

    def test_tno_requires_branch2(self):
        """Verify TNO variant produces correct output with a second branch."""
        B, X, Y, Z, T, C = 1, 8, 16, 8, 3, 5
        model = DeepONet3DWrapper(
            padding=8,
            variant="tno",
            width=32,
            branch1_config=BRANCH1_3D,
            branch2_config=BRANCH2_3D,
            trunk_config=TRUNK,
        )
        x = torch.randn(B, X, Y, Z, T, C)
        b2 = torch.randn(B, X, Y, Z, 1)
        _init_lazy(model, x, x_branch2=b2)
        out = model(x, x_branch2=b2)
        assert out.shape == (B, X, Y, Z, T)

    def test_target_times_3d(self):
        """Verify target_times overrides the temporal output dimension in 3D."""
        B, X, Y, Z, T_in, C = 1, 8, 16, 8, 1, 5
        K = 4
        model = DeepONet3DWrapper(
            padding=8,
            variant="u_deeponet",
            width=32,
            branch1_config=BRANCH1_3D,
            trunk_config=TRUNK,
        )
        x = torch.randn(B, X, Y, Z, T_in, C)
        tt = torch.linspace(0, 1, K)
        _init_lazy(model, x)
        out = model(x, target_times=tt)
        assert out.shape == (B, X, Y, Z, K)

    def test_count_params_3d(self):
        """Verify count_params returns a positive parameter count for 3D wrapper."""
        model = DeepONet3DWrapper(
            padding=8,
            variant="deeponet",
            width=32,
            branch1_config=BRANCH1_3D,
            trunk_config=TRUNK,
        )
        x = torch.randn(1, 8, 16, 8, 2, 5)
        _init_lazy(model, x)
        assert model.count_params() > 0


class TestHadamardProduct:
    """Verify 3-way Hadamard product for multi-branch variants."""

    def test_mionet_uses_multiplication(self):
        """Verify MIONet variant computes a 3-way Hadamard product correctly."""
        model = DeepONetWrapper(
            variant="mionet",
            width=16,
            branch1_config={
                "encoder": "spatial",
                "num_unet_layers": 0,
                "num_conv_layers": 1,
                "kernel_size": 3,
            },
            branch2_config={"encoder": "mlp", "hidden_width": 16, "num_layers": 2},
            trunk_config={"hidden_width": 16, "num_layers": 2},
            decoder_layers=0,
        )
        x = torch.randn(2, 16, 24, 4, 6)
        b2 = torch.randn(2, 6)
        with torch.no_grad():
            out = model(x, x_branch2=b2)
        assert out.shape == (2, 16, 24, 4)


class TestTemporalProjection:
    """Test temporal_projection decoder mode."""

    def test_2d_temporal_projection_output_shape(self):
        """Verify 2D temporal-projection decoder produces correct output T dimension."""
        K = 3
        model = DeepONet(
            variant="u_deeponet",
            width=16,
            branch1_config={
                "encoder": "spatial",
                "num_unet_layers": 0,
                "num_conv_layers": 1,
                "kernel_size": 3,
            },
            trunk_config={"hidden_width": 16, "num_layers": 2},
            decoder_type="temporal_projection",
            decoder_layers=1,
            decoder_width=16,
        )
        model.set_output_window(K)
        x_branch = torch.randn(2, 16, 24, 4)
        x_time = torch.randn(1, 1)
        with torch.no_grad():
            out = model(x_branch, x_time)
        assert out.shape == (2, 16, 24, K)

    def test_2d_temporal_projection_with_branch2(self):
        """Verify 2D temporal-projection works with a second branch input."""
        K = 5
        model = DeepONet(
            variant="tno",
            width=16,
            branch1_config={
                "encoder": "spatial",
                "num_unet_layers": 0,
                "num_conv_layers": 1,
                "kernel_size": 3,
            },
            branch2_config={
                "encoder": "spatial",
                "num_unet_layers": 0,
                "num_conv_layers": 1,
                "kernel_size": 3,
            },
            trunk_config={"hidden_width": 16, "num_layers": 2},
            decoder_type="temporal_projection",
            decoder_layers=1,
            decoder_width=16,
        )
        model.set_output_window(K)
        x_branch = torch.randn(2, 16, 24, 4)
        x_branch2 = torch.randn(2, 16, 24, 4)
        x_time = torch.randn(1, 1)
        with torch.no_grad():
            out = model(x_branch, x_time, x_branch2=x_branch2)
        assert out.shape == (2, 16, 24, K)

    def test_3d_temporal_projection(self):
        """Verify 3D temporal-projection decoder produces correct output shape."""
        K = 4
        model = DeepONet3D(
            variant="u_deeponet",
            width=8,
            branch1_config={
                "encoder": "spatial",
                "num_unet_layers": 0,
                "num_conv_layers": 1,
                "kernel_size": 3,
            },
            trunk_config={"hidden_width": 8, "num_layers": 2},
            decoder_type="temporal_projection",
            decoder_layers=1,
            decoder_width=8,
        )
        model.set_output_window(K)
        x_branch = torch.randn(2, 8, 8, 8, 4)
        x_time = torch.randn(1, 1)
        with torch.no_grad():
            out = model(x_branch, x_time)
        assert out.shape == (2, 8, 8, 8, K)

    def test_mlp_decoder_still_works(self):
        """Existing mlp decoder path is preserved."""
        model = DeepONet(
            variant="u_deeponet",
            width=16,
            branch1_config={
                "encoder": "spatial",
                "num_unet_layers": 0,
                "num_conv_layers": 1,
                "kernel_size": 3,
            },
            trunk_config={"hidden_width": 16, "num_layers": 2},
            decoder_type="mlp",
            decoder_layers=1,
            decoder_width=16,
        )
        x_branch = torch.randn(2, 16, 24, 4)
        x_time = torch.randn(6, 1)
        with torch.no_grad():
            out = model(x_branch, x_time)
        assert out.shape == (2, 16, 24, 6)

    def test_gradient_flow_temporal_projection(self):
        """Verify gradients propagate through the temporal-projection decoder."""
        K = 3
        model = DeepONet(
            variant="tno",
            width=16,
            branch1_config={
                "encoder": "spatial",
                "num_unet_layers": 0,
                "num_conv_layers": 1,
                "kernel_size": 3,
            },
            branch2_config={
                "encoder": "spatial",
                "num_unet_layers": 0,
                "num_conv_layers": 1,
                "kernel_size": 3,
            },
            trunk_config={"hidden_width": 16, "num_layers": 2},
            decoder_type="temporal_projection",
            decoder_layers=1,
            decoder_width=16,
        )
        model.set_output_window(K)
        x = torch.randn(2, 16, 24, 4, requires_grad=False)
        b2 = torch.randn(2, 16, 24, 4, requires_grad=False)
        t = torch.randn(1, 1)
        out = model(x, t, x_branch2=b2)
        loss = out.sum()
        loss.backward()
        assert model.temporal_head.weight.grad is not None


class TestInternalResolution:
    """Test adaptive pooling in SpatialBranch."""

    def test_2d_internal_resolution(self):
        """Verify 2D SpatialBranch with internal_resolution preserves output shape."""
        from models.xdeeponet import SpatialBranch

        branch = SpatialBranch(
            in_channels=4,
            width=8,
            num_fourier_layers=0,
            num_unet_layers=0,
            num_conv_layers=1,
            kernel_size=3,
            internal_resolution=[16, 24],
        )
        x = torch.randn(2, 32, 48, 4)
        out = branch(x)
        assert out.shape == (2, 32, 48, 8)

    def test_2d_no_internal_resolution(self):
        """Verify 2D SpatialBranch without internal_resolution preserves output shape."""
        from models.xdeeponet import SpatialBranch

        branch = SpatialBranch(
            in_channels=4,
            width=8,
            num_fourier_layers=0,
            num_unet_layers=0,
            num_conv_layers=1,
            kernel_size=3,
            internal_resolution=None,
        )
        x = torch.randn(2, 32, 48, 4)
        out = branch(x)
        assert out.shape == (2, 32, 48, 8)

    def test_3d_internal_resolution(self):
        """Verify 3D SpatialBranch with internal_resolution preserves output shape."""
        from models.xdeeponet import SpatialBranch3D

        branch = SpatialBranch3D(
            in_channels=4,
            width=8,
            num_fourier_layers=0,
            num_unet_layers=0,
            num_conv_layers=1,
            kernel_size=3,
            internal_resolution=[8, 8, 8],
        )
        x = torch.randn(2, 16, 16, 16, 4)
        out = branch(x)
        assert out.shape == (2, 16, 16, 16, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
