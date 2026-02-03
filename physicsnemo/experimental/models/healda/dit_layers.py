# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
"""
HealDA model and checkpoint migration utilities.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.experimental.models.dit import DiT

from .config import ModelSensorConfig, SensorEmbedderConfig
from .healpix_layers import HPXPatchDetokenizer, HPXPatchTokenizer
from .point_embed import MultiSensorObsEmbedding
from .types import UnifiedObservation


@dataclass
class HealDAMetaData(ModelMetaData):
    """Metadata for HealDA model."""

    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    bf16: bool = True
    onnx: bool = False
    func_torch: bool = False
    auto_grad: bool = False


class HealDA(Module):
    r"""
    HealDA DiT model that composes preprocessor + PNM experimental DiT.
    
    Parameters
    ----------
    in_channels : int
        Number of input state channels.
    out_channels : int
        Number of output channels.
    hidden_size : int, optional, default=1024
        Transformer hidden dimension.
    num_layers : int, optional, default=24
        Number of transformer blocks.
    num_heads : int, optional, default=16
        Number of attention heads.
    mlp_ratio : float, optional, default=4.0
        MLP hidden dim multiplier.
    level_in : int, optional, default=6
        HEALPix input resolution level.
    level_model : int, optional, default=5
        HEALPix model resolution level after patching.
    time_length : int, optional, default=1
        Number of time steps.
    sensor_embedder_config : SensorEmbedderConfig
        Config for observation embedding.
    sensors : dict[str, ModelSensorConfig]
        Sensor configurations for obs embedding.
    condition_channels : int, optional, default=2
        Number of static input channels that go into tokenizer.
        Tokenizer input = condition_channels + fusion_dim.
    qk_norm_type : Literal["RMSNorm", "LayerNorm"], optional, default="RMSNorm"
        QK normalization type. None disables QK normalization.
    drop_path : float, optional, default=0.0
        DropPath rate for stochastic depth.
    dropout : float, optional, default=0.0
        Dropout rate for projection and MLP layers.
    condition_dim : int, optional, default=None
        Conditioning embedding dimension. If None, runs as VIT (no conditioning).
        If set, enables diffusion-style noise/label conditioning.
    noise_channels : int, optional, default=1024
        Channels for noise level positional embedding.
    label_dim : int, optional, default=0
        Dimension of class labels. 0 means no label conditioning.
    label_dropout : float, optional, default=None
        Dropout rate for labels during training.
    attention_backend : str, optional, default="transformer_engine"
        Attention backend to use.
    layernorm_backend : str, optional, default="apex"
        LayerNorm backend to use.
    
    Forward
    -------
    x : torch.Tensor
        Input state tensor of shape :math:`(B, C, T, N_{pix})`.
    t : torch.Tensor
        Timestep tensor of shape :math:`(B,)`.
    unified_obs : UnifiedObservation
        Observation data (required for obs-to-state DA).
    second_of_day : torch.Tensor, optional
        Second of day for calendar embedding.
    day_of_year : torch.Tensor, optional
        Day of year for calendar embedding.
    noise_labels : torch.Tensor, optional
        Noise levels for diffusion conditioning. Required when condition_dim is set.
    class_labels : torch.Tensor, optional
        Class labels for conditioning. Only used when condition_dim is set.
    
    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, T, N_{pix})`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sensor_embedder_config: SensorEmbedderConfig,
        sensors: dict[str, ModelSensorConfig],
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        level_in: int = 6,
        level_model: int = 5,
        time_length: int = 1,
        condition_channels: int = 2,  # Static input channels (e.g. lat/lon)
        qk_norm_type: Optional[Literal["RMSNorm", "LayerNorm"]] = "RMSNorm",
        drop_path: float = 0.0,
        dropout: float = 0.0,
        condition_dim: Optional[int] = None,
        noise_channels: int = 1024,
        label_dim: int = 0,
        label_dropout: Optional[float] = None,
        attention_backend: str = "transformer_engine",
        layernorm_backend: str = "apex",
    ):
        super().__init__(meta=HealDAMetaData())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.level_in = level_in
        self.level_model = level_model
        self.time_length = time_length
        self.condition_channels = condition_channels
        self.condition_dim = condition_dim
        self.label_dim = label_dim

        # Observation encoder (embeds obs and concatenates with state)
        self.obs_embedder = MultiSensorObsEmbedding(
            sensor_embedder_config=sensor_embedder_config,
            sensors=sensors,
            hpx_level=level_in,
        )
        self.fusion_dim = sensor_embedder_config.fusion_dim
        # Tokenizer input: static condition channels + obs embedding (NOT full in_channels)
        tokenizer_in_channels = condition_channels + self.fusion_dim

        # Create tokenizer and detokenizer
        tokenizer = HPXPatchTokenizer(
            in_channels=tokenizer_in_channels,
            hidden_size=hidden_size,
            level_fine=level_in,
            level_coarse=level_model,
        )

        detokenizer = HPXPatchDetokenizer(
            hidden_size=hidden_size,
            out_channels=out_channels,
            level_coarse=level_model,
            level_fine=level_in,
            time_length=time_length,
            condition_dim=condition_dim,
        )

        # Create PNM DiT with custom tokenizer/detokenizer
        npix_coarse = 12 * 4 ** level_model
        attn_kwargs = {"qk_norm_type": qk_norm_type} if qk_norm_type else {}
        
        # HealDA used dropout after attention projection and in MLP, not on attention weights
        block_kwargs = {
            "proj_drop_rate": dropout,
            "mlp_drop_rate": dropout,
        }

        self.dit = DiT(
            input_size=(npix_coarse * time_length,),
            in_channels=tokenizer_in_channels,
            patch_size=(1,),
            tokenizer=tokenizer,
            detokenizer=detokenizer,
            out_channels=out_channels,
            hidden_size=hidden_size,
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attention_backend=attention_backend,
            layernorm_backend=layernorm_backend,
            condition_dim=condition_dim,
            conditioning_embedder="pre_mlp",
            conditioning_embedder_kwargs={
                "label_dim": label_dim,
                "label_dropout": label_dropout,
            },
            drop_path=drop_path,
            attn_kwargs=attn_kwargs,
            block_kwargs=block_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        unified_obs: UnifiedObservation,
        second_of_day: Optional[torch.Tensor] = None,
        day_of_year: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input state (B, C, T, npix)
            t: Timestep/noise_labels (B,) - used for conditioning
            unified_obs: Observation data (required)
            second_of_day: Calendar info
            day_of_year: Calendar info
            class_labels: Class labels for conditioning (only used when label_dim > 0)
            
        Returns:
            Output (B, C_out, T, npix)
        """
        # Embed observations and concatenate with state
        obs_emb = self.obs_embedder(unified_obs)  # (B, fusion_dim, T, npix)
        x = torch.cat([x, obs_emb], dim=1)  # (B, C + fusion_dim, T, npix)

        return self.dit(
            x, t, condition=class_labels,
            tokenizer_kwargs={"second_of_day": second_of_day, "day_of_year": day_of_year},
        )

    @classmethod
    def from_healda_checkpoint(
        cls,
        checkpoint_path: str,
        sensor_embedder_config: SensorEmbedderConfig,
        sensors: dict[str, ModelSensorConfig],
        level_in: int = 6,
        level_model: int = 5,
        device: str = "cuda",
    ) -> "HealDA":
        """
        Load a HealDA model from an old HealDA checkpoint.
        
        Args:
            checkpoint_path: Path to the .checkpoint file
            sensor_embedder_config: Configuration for multi-sensor observation embedder
            sensors: Dictionary mapping sensor names to their configurations
            level_in: HEALPix input resolution level (default: 6)
            level_model: HEALPix model resolution level (default: 5)
            device: Device to load the model on
            
        Returns:
            Instantiated HealDA with weights loaded
        """
        import json
        import zipfile

        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            # Read config
            with zf.open("model.json") as f:
                config = json.load(f)

            # Read state dict
            with zf.open("net_state.pth") as f:
                old_state_dict = torch.load(f, map_location=device, weights_only=True)

        # Extract model config
        hidden_size = 1024  # Default for dit-l
        num_layers = 24
        num_heads = 16

        # Determine condition_dim from as_vit flag
        # as_vit=True means condition_dim=0 (VIT mode, bias-only adaptive modulation)
        condition_dim = 0 if config.get("as_vit", False) else 4 * hidden_size

        # Create model
        model = cls(
            in_channels=config.get("out_channels", 74),
            out_channels=config.get("out_channels", 74),
            sensor_embedder_config=sensor_embedder_config,
            sensors=sensors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            level_in=level_in,
            level_model=level_model,
            time_length=config.get("time_length", 1),
            condition_channels=config.get("condition_channels", 2),
            qk_norm_type="RMSNorm" if config.get("qk_rms_norm", False) else None,
            drop_path=config.get("drop_path", 0.0),
            dropout=config.get("p_dropout", 0.0),
            condition_dim=condition_dim,
        )

        # Convert and load state dict
        new_state_dict = convert_healda_state_dict(
            old_state_dict,
            num_blocks=num_layers,
            hidden_size=hidden_size,
            condition_dim=condition_dim,
        )

        # With condition_dim=0 (VIT mode), shapes now match [n, 0] directly
        model.load_state_dict(new_state_dict, strict=False)
        return model.to(device)


# Weight mapping utilities for checkpoint migration
def map_healda_to_pnm_block_keys(old_key: str, block_idx: int) -> str:
    """
    Map HealDA DiT block state dict key to PNM DiT block key.
    
    Args:
        old_key: Original key like 'transformer_blocks.0.attn1.to_q.weight'
        block_idx: Block index
        
    Returns:
        New key like 'blocks.0.attention.attn_op.qkv.query_weight'
    """
    prefix = f"transformer_blocks.{block_idx}."
    new_prefix = f"dit.blocks.{block_idx}."
    
    if not old_key.startswith(prefix):
        return old_key
    
    suffix = old_key[len(prefix):]
    
    # Mapping table
    mappings = {
        # AdaLN modulation
        "norm1.linear.weight": "adaptive_modulation.1.weight",
        "norm1.linear.bias": "adaptive_modulation.1.bias",
        # Attention Q/K/V
        "attn1.to_q.weight": "attention.attn_op.qkv.query_weight",
        "attn1.to_q.bias": "attention.attn_op.qkv.query_bias",
        "attn1.to_k.weight": "attention.attn_op.qkv.key_weight",
        "attn1.to_k.bias": "attention.attn_op.qkv.key_bias",
        "attn1.to_v.weight": "attention.attn_op.qkv.value_weight",
        "attn1.to_v.bias": "attention.attn_op.qkv.value_bias",
        # Attention output projection
        "attn1.to_out.0.weight": "attention.attn_op.proj.weight",
        "attn1.to_out.0.bias": "attention.attn_op.proj.bias",
        # MLP
        "ff.net.0.proj.weight": "linear.layers.0.weight",
        "ff.net.0.proj.bias": "linear.layers.0.bias",
        "ff.net.2.weight": "linear.layers.2.weight",
        "ff.net.2.bias": "linear.layers.2.bias",
    }
    
    if suffix in mappings:
        return new_prefix + mappings[suffix]
    
    return old_key


def convert_healda_state_dict(
    old_state_dict: dict,
    num_blocks: int = 24,
    hidden_size: int = 1024,
    condition_dim: int = 0,
) -> dict:
    """
    Convert HealDA DiT state dict to PNM DiT format.
    
    Args:
        old_state_dict: Original HealDA state dict
        num_blocks: Number of transformer blocks
        hidden_size: Model hidden dimension
        condition_dim: Conditioning dimension. If 0, model is unconditional
            and empty weights are expanded to zeros.
        
    Returns:
        New state dict compatible with PNM DiT
    """
    new_state_dict = {}
    is_unconditional = condition_dim == 0
    
    for old_key, value in old_state_dict.items():
        # Handle transformer blocks
        if old_key.startswith("transformer_blocks."):
            # Extract block index
            parts = old_key.split(".")
            block_idx = int(parts[1])
            
            new_key = map_healda_to_pnm_block_keys(old_key, block_idx)
            # VIT mode: [n, 0] weights are kept as-is (model now supports condition_dim=0)
            new_state_dict[new_key] = value
            
        # Handle obs embedder (embed_v2_patch -> obs_embedder)
        elif old_key.startswith("embed_v2_patch."):
            suffix = old_key[len("embed_v2_patch."):]
            new_key = f"obs_embedder.{suffix}"
            new_state_dict[new_key] = value
            
        # Handle tokenizer (pos_embed)
        elif old_key.startswith("pos_embed."):
            suffix = old_key[len("pos_embed."):]
            new_key = f"dit.tokenizer.{suffix}"
            new_state_dict[new_key] = value
            
        # Handle detokenizer (patch_decode)
        elif old_key.startswith("patch_decode."):
            suffix = old_key[len("patch_decode."):]
            new_key = f"dit.detokenizer.{suffix}"
            new_state_dict[new_key] = value
            
        # Handle final projection (proj_out_1 -> detokenizer.adaptive_modulation.1)
        elif old_key.startswith("proj_out_1."):
            suffix = old_key[len("proj_out_1."):]
            new_key = f"dit.detokenizer.adaptive_modulation.1.{suffix}"
            # VIT mode: [n, 0] weights are kept as-is (model now supports condition_dim=0)
            new_state_dict[new_key] = value
            
        elif old_key.startswith("norm_out."):
            suffix = old_key[len("norm_out."):]
            new_key = f"dit.detokenizer.norm_out.{suffix}"
            new_state_dict[new_key] = value
            
        # Map noise_embed to dit.conditioning_embedder (not present in VIT mode)
        elif old_key.startswith("noise_embed."):
            if not is_unconditional:
                suffix = old_key[len("noise_embed."):]
                new_state_dict[f"dit.conditioning_embedder.{suffix}"] = value
            
        else:
            # Pass through other keys unchanged  
            new_state_dict[old_key] = value
    
    return new_state_dict
