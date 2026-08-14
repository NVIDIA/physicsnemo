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

from typing import Any, List

import torch
from jaxtyping import Float
from tensordict import TensorDict

from physicsnemo.core import Module
from physicsnemo.models.diffusion_unets import SongUNet
from physicsnemo.nn import PositionalEmbedding


class HRRRUnconditionalUNet(Module):
    r"""Backbone wrapping SongUNet with temporal embedding for the HRRR
    surface diffusion model.

    This wrapper sits between the preconditioner and the raw SongUNet backbone.
    It consumes a TensorDict condition produced by MultiDiffusionModel2D and:

    1. Embeds the scalar temporal conditioning via a learnable
       :class:`~physicsnemo.nn.PositionalEmbedding`.
    2. Concatenates spatial conditioning and positional embeddings (from the
       multi-diffusion wrapper) directly to the input ``x`` along the channel
       dimension.
    3. Calls :class:`~physicsnemo.models.diffusion_unets.SongUNet` with the
       concatenated input and the temporal embedding as ``class_labels``.

    Parameters
    ----------
    img_resolution : list[int]
        Spatial resolution :math:`[H, W]` of the input patches.
    img_channels : int
        Number of image channels (used as ``out_channels`` for the UNet).
    num_condition_channels : int
        Number of spatial conditioning channels concatenated to ``x``.
    num_grid_channels : int
        Number of positional-embedding channels concatenated to ``x``.
    time_embed_channels : int
        Dimensionality of the temporal embedding vector (passed as
        ``label_dim`` to the internal
        :class:`~physicsnemo.models.diffusion_unets.SongUNet`).
    model_channels : int, optional
        Base channel count for the UNet, by default 128.
    channel_mult : list[int], optional
        Per-level channel multipliers, by default ``[1, 2, 2, 2, 2]``.
    use_apex_gn : bool, optional
        Whether to use Apex fused group-norm kernels, by default ``False``.
    amp_mode : bool, optional
        Whether mixed-precision (AMP) training is enabled, by default ``False``.
        Propagated to the internal
        :class:`~physicsnemo.models.diffusion_unets.SongUNet`.

    Forward
    -------
    x : torch.Tensor
        Noisy latent state of shape :math:`(B, C, H, W)`.
    t : torch.Tensor
        Diffusion time tensor of shape :math:`(B,)`.
    condition : TensorDict or None
        TensorDict with keys ``"cond_concat"`` (spatial conditioning,
        shape :math:`(B, C_{cond}, H, W)`) and ``"cond_time"`` (scalar
        temporal conditioning, shape :math:`(B, 1)`). Optionally contains
        ``"positional_embedding"`` (shape :math:`(B, C_{PE}, H, W)`)
        injected by :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionModel2D`.

    Outputs
    -------
    torch.Tensor
        Model output of shape :math:`(B, C_{out}, H, W)`.
    """

    def __init__(
        self,
        img_resolution: List[int],
        img_channels: int,
        num_condition_channels: int,
        num_grid_channels: int,
        time_embed_channels: int,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 2, 2, 2],
        use_apex_gn: bool = False,
        amp_mode: bool = False,
    ) -> None:
        super().__init__()
        self.unet = SongUNet(
            img_resolution=list(img_resolution),
            in_channels=img_channels + num_condition_channels + num_grid_channels,
            out_channels=img_channels,
            label_dim=time_embed_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            attn_resolutions=[img_resolution[0] >> len(channel_mult)],
            use_apex_gn=use_apex_gn,
            amp_mode=amp_mode,
        )
        self.time_embedding = PositionalEmbedding(
            num_channels=time_embed_channels,
            max_positions=365,
            endpoint=True,
            learnable=True,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "B C H W"],
        t: Float[torch.Tensor, " B"],
        condition: TensorDict,
    ) -> Float[torch.Tensor, "B C H W"]:
        cond_time = condition["cond_time"]
        cond_concat = condition["cond_concat"]
        pos_embd = condition["positional_embedding"]

        ct_embed = self.time_embedding(cond_time.squeeze(-1))
        x_and_cond = torch.cat([x, cond_concat, pos_embd], dim=1)

        return self.unet(
            x_and_cond,
            noise_labels=t,
            class_labels=ct_embed,
        )
