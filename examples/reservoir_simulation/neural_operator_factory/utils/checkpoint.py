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

"""Checkpoint utilities: model reconstruction and save/load helpers."""

import torch
from models.xdeeponet import DeepONet3DWrapper, DeepONetWrapper
from models.xfno import FNO4DNet, UFNONet


def build_model_from_config(model_config: dict, device="cpu"):
    """Reconstruct a model from a saved model_config dict.

    This is the single source of truth for model reconstruction, used by
    both checkpoint resume and evaluation scripts.

    Parameters
    ----------
    model_config : dict
        The ``model_config`` dict saved inside a checkpoint.  Must contain
        at least ``model_type`` and ``dimensions``.
    device : str or torch.device
        Device to place the model on.

    Returns
    -------
    tuple of (model, model_arch_name)
    """
    model_type = model_config["model_type"]
    dimensions = model_config.get("dimensions", "4d")

    if model_type == "xdeeponet":
        variant = model_config.get("variant", "u_deeponet")
        cls = DeepONet3DWrapper if dimensions == "4d" else DeepONetWrapper
        model = cls(
            padding=model_config.get("padding", 8),
            variant=variant,
            width=model_config.get("width", 128),
            branch1_config=model_config.get("branch1_config", {}),
            branch2_config=model_config.get("branch2_config"),
            trunk_config=model_config.get("trunk_config", {}),
            decoder_type=model_config.get("decoder_type", "mlp"),
            decoder_width=model_config.get("decoder_width", 128),
            decoder_layers=model_config.get("decoder_layers", 2),
            decoder_activation_fn=model_config.get("decoder_activation_fn", "relu"),
        )
        if model_config.get("decoder_type") == "temporal_projection":
            K = model_config.get("output_window", 3)
            model.set_output_window(K)

        b1_enc = model_config.get("branch1_config", {}).get("encoder", "spatial")
        encoder = b1_enc.get("type", "linear") if isinstance(b1_enc, dict) else b1_enc
        model_arch_name = model_config.get(
            "model_arch_name",
            f"deeponet{'3d' if dimensions == '4d' else ''}_{variant}_{encoder}",
        )

    elif model_type == "xfno":
        if dimensions == "4d":
            model = FNO4DNet(
                modes1=model_config["modes1"],
                modes2=model_config["modes2"],
                modes3=model_config["modes3"],
                modes4=model_config.get("modes4", 6),
                width=model_config["width"],
                in_channels=model_config["in_channels"],
                out_channels=model_config.get("out_channels", 1),
                num_fno_layers=model_config["num_fno_layers"],
                padding=model_config.get("padding", 8),
                activation_fn=model_config.get("activation_fn", "gelu"),
                lifting_layers=model_config.get("lifting_layers", 1),
                decoder_layers=model_config.get("decoder_layers", 1),
                decoder_width=model_config.get("decoder_width", 128),
                coord_features=model_config.get("coord_features", True),
            )
            model_arch_name = model_config.get("model_arch_name", "fno4d")
        else:
            model = UFNONet(
                modes1=model_config["modes1"],
                modes2=model_config["modes2"],
                modes3=model_config["modes3"],
                width=model_config["width"],
                in_channels=model_config["in_channels"],
                out_channels=model_config.get("out_channels", 1),
                num_fno_layers=model_config["num_fno_layers"],
                num_unet_layers=model_config.get("num_unet_layers", 0),
                num_conv_layers=model_config.get("num_conv_layers", 0),
                padding=model_config.get("padding", 8),
                unet_type=model_config.get("unet_type", "custom"),
                activation_fn=model_config.get("activation_fn", "relu"),
                lifting_type=model_config.get("lifting_type", "mlp"),
                lifting_layers=model_config.get("lifting_layers", 1),
                lifting_width=model_config.get("lifting_width", 36),
                decoder_type=model_config.get("decoder_type", "mlp"),
                decoder_layers=model_config.get("decoder_layers", 1),
                decoder_width=model_config.get("decoder_width", 128),
            )
            model_arch_name = model_config.get("model_arch_name", "ufno")
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")

    return model.to(device), model_arch_name


def save_checkpoint(
    path,
    model,
    epoch: int,
    val_loss: float,
    metric_key: str,
    metric_value: float,
    model_config: dict,
    optimizer=None,
    scheduler=None,
):
    """Save a training checkpoint with all state needed for resume."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    model_to_save = model.module if isinstance(model, DDP) else model
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "val_loss": val_loss,
        metric_key: metric_value,
        "model_config": model_config,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint and return the dict."""
    return torch.load(path, map_location=device, weights_only=False)
