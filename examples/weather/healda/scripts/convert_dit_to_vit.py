"""
Convert a trained DiT checkpoint to as_vit mode (bias-only AdaLN layers).

When DiT is always given noise_labels=0 and class_labels=None, the adaptive
normalization parameters become fixed. This script bakes those constants into
bias terms, allowing the model to run with emb_channels=0.
"""

import dataclasses
from collections import OrderedDict

import models
import torch
import torch.nn as nn
from utils.checkpointing import Checkpoint

from physicsnemo.models.healda.dit import DiT
from physicsnemo.models.healda.embedding import EmbedNoiseLabels


def compute_constant_emb(
    noise_embed: EmbedNoiseLabels,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute the constant embedding for noise_labels=0, class_labels=0 (or None if no labels).
    """
    noise_labels = torch.zeros(1, device=device, dtype=dtype)

    if noise_embed.map_label is not None:
        label_dim = noise_embed.map_label.in_features
        class_labels = torch.zeros(1, label_dim, device=device, dtype=dtype)
    else:
        class_labels = None

    with torch.no_grad():
        emb = noise_embed(noise_labels, class_labels)

    return emb


def bake_linear_layer(
    linear: nn.Linear,
    fixed_input: torch.Tensor,
) -> torch.Tensor:
    """
    Compute W @ fixed_input + b and return as baked bias.
    """
    with torch.no_grad():
        output = linear(fixed_input)  # [1, out_features]
        return output.squeeze(0)  # [out_features]


def convert_state_dict(
    state_dict: dict,
    noise_embed: EmbedNoiseLabels,
    proj_out_1: nn.Linear,
    transformer_blocks: nn.ModuleList,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Convert DiT state dict to as_vit mode by baking fixed embeddings into biases.

    Removes noise_embed params and AdaLN weight matrices, keeps only biases
    with baked values from running the fixed embedding through the layers.
    """
    fixed_emb = compute_constant_emb(noise_embed, device, dtype)

    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        # Skip noise_embed entirely
        if key.startswith("noise_embed."):
            continue

        # Handle AdaLayerNormZero: transformer_blocks.{i}.norm1.linear.{weight,bias}
        if ".norm1.linear." in key:
            if key.endswith(".weight"):
                continue  # Skip weight - Linear(0, out) has no weight
            if key.endswith(".bias"):
                block_idx = int(key.split(".")[1])
                block = transformer_blocks[block_idx]
                baked = bake_linear_layer(block.norm1.linear, fixed_emb)
                new_state_dict[key] = baked
                continue

        # Handle AdaLayerNormTemporalAttn: transformer_blocks.{i}.temporal_attn_norm.linear.{weight,bias}
        if ".temporal_attn_norm.linear." in key:
            if key.endswith(".weight"):
                continue
            if key.endswith(".bias"):
                block_idx = int(key.split(".")[1])
                block = transformer_blocks[block_idx]
                if (
                    hasattr(block, "temporal_attn_norm")
                    and block.temporal_attn_norm is not None
                ):
                    baked = bake_linear_layer(
                        block.temporal_attn_norm.linear, fixed_emb
                    )
                    new_state_dict[key] = baked
                continue

        # Handle proj_out_1
        if key == "proj_out_1.weight":
            continue
        if key == "proj_out_1.bias":
            baked = bake_linear_layer(proj_out_1, fixed_emb)
            new_state_dict[key] = baked
            continue

        # Copy all other parameters unchanged
        new_state_dict[key] = value

    return new_state_dict


def convert_dit_model(
    dit: DiT,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Convert a loaded DiT model's state dict to as_vit format.

    Args:
        dit: DiT model with loaded weights (as_vit=False)
        device: Device for computation
        dtype: Data type for computation

    Returns:
        Converted state dict compatible with DiT(as_vit=True)
    """
    return convert_state_dict(
        state_dict=dit.state_dict(),
        noise_embed=dit.noise_embed,
        proj_out_1=dit.proj_out_1,
        transformer_blocks=dit.transformer_blocks,
        device=device,
        dtype=dtype,
    )


def convert_checkpoint_file(
    input_path: str,
    output_path: str,
    device: str = "cpu",
):
    """
    Convert a DiT checkpoint file to as_vit mode.

    Reads the checkpoint, converts the state dict, updates the config
    to set as_vit=True, and writes a new checkpoint.
    """
    # Read input checkpoint
    with Checkpoint(input_path, mode="r") as ckpt:
        model_config = ckpt.read_model_config()
        state_dict = ckpt.read_model_state_dict()

        # Try to read batch_info if present
        try:
            batch_info = ckpt.read_batch_info()
        except KeyError:
            batch_info = None

    # Create DiT model to get module references
    dit = models.get_model(model_config)
    dit.load_state_dict(state_dict)
    dit.to(device)
    dit.eval()

    # Convert state dict
    new_state_dict = convert_dit_model(dit, torch.device(device), torch.float32)

    # Update config to set as_vit=True (legacy_label_bias irrelevant but set False for clarity)
    new_config = dataclasses.replace(model_config, as_vit=True, legacy_label_bias=False)

    # Create new model with as_vit=True and load converted weights
    new_dit = models.get_model(new_config)
    new_dit.load_state_dict(new_state_dict, strict=False)

    # Write output checkpoint
    with Checkpoint(output_path, mode="w") as ckpt:
        ckpt.write_model(new_dit)
        ckpt.write_model_config(new_config)
        if batch_info is not None:
            ckpt.write_batch_info(batch_info)

    # Count actual parameter reduction
    old_params = sum(v.numel() for v in state_dict.values())
    new_params = sum(v.numel() for v in new_state_dict.values())
    removed = old_params - new_params
    print(f"Saved to {output_path}")
    print(
        f"Parameters: {old_params / 1e6:.2f}M -> {new_params / 1e6:.2f}M ({removed / 1e6:.2f}M removed)"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert DiT checkpoint to ViT like model"
    )
    parser.add_argument("--input", required=True, help="Input checkpoint path (.zip)")
    parser.add_argument("--output", required=True, help="Output checkpoint path (.zip)")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    convert_checkpoint_file(
        input_path=args.input,
        output_path=args.output,
        device=args.device,
    )
"""
python convert_dit_to_vit.py \
    --input /lustre/fsw/coreai_climate_earth2/aaygupta/clean/ufs_root/training-runs/era5-v2-dense-noInfill-10M-fusion512-lrObs1e-4/training-state-010070944-fixed.checkpoint \
    --output /lustre/fsw/coreai_climate_earth2/aaygupta/clean/ufs_root/training-runs/era5-v2-dense-noInfill-10M-fusion512-lrObs1e-4/training-state-010070944-vit.checkpoint \
    --device cuda
"""
