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

"""Unit tests for checkpoint utilities."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.checkpoint import build_model_from_config, load_checkpoint, save_checkpoint


class TestBuildModelFromConfig:
    """Tests for build_model_from_config."""

    def test_xdeeponet_u_deeponet_3d(self):
        """Verify building a 3D U-DeepONet model from config."""
        cfg = {
            "model_type": "xdeeponet",
            "dimensions": "4d",
            "variant": "u_deeponet",
            "width": 32,
            "padding": 8,
            "branch1_config": {
                "encoder": "spatial",
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
            "trunk_config": {
                "input_type": "time",
                "hidden_width": 32,
                "num_layers": 2,
                "activation_fn": "tanh",
            },
            "decoder_type": "mlp",
            "decoder_width": 32,
            "decoder_layers": 1,
            "decoder_activation_fn": "relu",
        }
        model, name = build_model_from_config(cfg)
        assert model is not None
        assert "deeponet3d" in name

    def test_xdeeponet_tno_3d(self):
        """Verify building a 3D TNO model from config."""
        cfg = {
            "model_type": "xdeeponet",
            "dimensions": "4d",
            "variant": "tno",
            "width": 32,
            "padding": 8,
            "branch1_config": {
                "encoder": "spatial",
                "num_fourier_layers": 0,
                "num_unet_layers": 1,
                "num_conv_layers": 0,
                "kernel_size": 3,
                "dropout": 0.0,
                "unet_impl": "custom",
                "activation_fn": "relu",
            },
            "branch2_config": {
                "encoder": "spatial",
                "num_fourier_layers": 0,
                "num_unet_layers": 1,
                "num_conv_layers": 0,
                "kernel_size": 3,
                "dropout": 0.0,
                "unet_impl": "custom",
                "activation_fn": "relu",
            },
            "trunk_config": {
                "input_type": "time",
                "hidden_width": 32,
                "num_layers": 2,
                "activation_fn": "tanh",
            },
            "decoder_type": "mlp",
            "decoder_width": 32,
            "decoder_layers": 1,
        }
        model, name = build_model_from_config(cfg)
        assert "tno" in name

    def test_xdeeponet_2d(self):
        """Verify building a 2D U-DeepONet model from config."""
        cfg = {
            "model_type": "xdeeponet",
            "dimensions": "3d",
            "variant": "u_deeponet",
            "width": 32,
            "padding": 8,
            "branch1_config": {
                "encoder": "spatial",
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
            "trunk_config": {
                "input_type": "time",
                "hidden_width": 32,
                "num_layers": 2,
                "activation_fn": "tanh",
            },
        }
        model, name = build_model_from_config(cfg)
        assert "deeponet_" in name  # 2D has no "3d" in name

    def test_xfno_4d(self):
        """Verify building a 4D FNO model from config."""
        cfg = {
            "model_type": "xfno",
            "dimensions": "4d",
            "in_channels": 11,
            "out_channels": 1,
            "width": 16,
            "modes1": 4,
            "modes2": 4,
            "modes3": 4,
            "modes4": 3,
            "num_fno_layers": 2,
            "padding": 8,
        }
        model, name = build_model_from_config(cfg)
        assert "fno4d" in name

    def test_xfno_3d(self):
        """Verify building a 3D FNO model from config."""
        cfg = {
            "model_type": "xfno",
            "dimensions": "3d",
            "in_channels": 12,
            "out_channels": 1,
            "width": 16,
            "modes1": 4,
            "modes2": 4,
            "modes3": 4,
            "num_fno_layers": 2,
            "padding": 8,
        }
        model, name = build_model_from_config(cfg)
        assert model is not None

    def test_unknown_type_raises(self):
        """Verify unknown model_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_model_from_config({"model_type": "invalid"})


class TestSaveLoadCheckpoint:
    """Tests for save/load round-trip."""

    def test_round_trip(self, tmp_path):
        """Verify save/load checkpoint round-trip preserves all fields."""
        cfg = {
            "model_type": "xdeeponet",
            "dimensions": "4d",
            "variant": "u_deeponet",
            "width": 16,
            "padding": 8,
            "branch1_config": {
                "encoder": "spatial",
                "num_fourier_layers": 0,
                "num_unet_layers": 1,
                "num_conv_layers": 0,
                "kernel_size": 3,
                "dropout": 0.0,
                "unet_impl": "custom",
                "activation_fn": "relu",
            },
            "trunk_config": {
                "input_type": "time",
                "hidden_width": 16,
                "num_layers": 2,
                "activation_fn": "tanh",
            },
        }
        model, _ = build_model_from_config(cfg)

        # Dummy forward to init lazy modules
        x = torch.randn(1, 8, 16, 8, 2, 5)
        with torch.no_grad():
            model(x)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        path = tmp_path / "test_ckpt.pth"
        save_checkpoint(
            path=path,
            model=model,
            epoch=42,
            val_loss=0.05,
            metric_key="val_rmse",
            metric_value=0.1,
            model_config=cfg,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        ckpt = load_checkpoint(path)
        assert ckpt["epoch"] == 42
        assert abs(ckpt["val_loss"] - 0.05) < 1e-8
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt
        assert ckpt["model_config"] == cfg

    def test_rebuild_from_checkpoint(self, tmp_path):
        """Save a model, then rebuild from checkpoint config and load weights."""
        cfg = {
            "model_type": "xdeeponet",
            "dimensions": "4d",
            "variant": "u_deeponet",
            "width": 16,
            "padding": 8,
            "branch1_config": {
                "encoder": "spatial",
                "num_fourier_layers": 0,
                "num_unet_layers": 1,
                "num_conv_layers": 0,
                "kernel_size": 3,
                "dropout": 0.0,
                "unet_impl": "custom",
                "activation_fn": "relu",
            },
            "trunk_config": {
                "input_type": "time",
                "hidden_width": 16,
                "num_layers": 2,
                "activation_fn": "tanh",
            },
        }
        model1, _ = build_model_from_config(cfg)
        x = torch.randn(1, 8, 16, 8, 2, 5)
        with torch.no_grad():
            model1(x)

        path = tmp_path / "test_ckpt.pth"
        save_checkpoint(
            path=path,
            model=model1,
            epoch=10,
            val_loss=0.1,
            metric_key="val_rmse",
            metric_value=0.2,
            model_config=cfg,
        )

        # Rebuild from checkpoint
        ckpt = load_checkpoint(path)
        model2, _ = build_model_from_config(ckpt["model_config"])
        with torch.no_grad():
            model2(x)  # init lazy
        model2.load_state_dict(ckpt["model_state_dict"])

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Mismatch in {n1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
