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


"""Distillation helpers for CorrDiff models.

This module adapts FastGen's distillation methods for CorrDiff model.
The main additions over the base FastGen models are:

- ``FastGenNet`` wraps the CorrDiff network into FastGen's
  ``FastGenNetwork`` interface, adding SuperPatching2D unfold/fold and
  window smoothing to support super-patch distillation training.
- ``CMModel``, ``SCMModel``, and ``DMD2Model`` override ``build_model``
  to construct the network via ``FastGenNet`` instead of FastGen's
  default path.
- ``DistillLoss`` bridges CorrDiff's training interface to FastGen's ``single_train_step`` API.

See ``https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/README.md`` for the base implementations.
"""

from functools import partial
import torch
from typing import Optional, Tuple, List
from copy import deepcopy
import einops
import numpy as np

from physicsnemo.nn.module import UNetBlock
from physicsnemo.diffusion.multi_diffusion.patching import (
    RandomPatching2D,
    BasePatching2D,
    GridPatching2D,
)

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.methods.consistency_model.CM import CMModel as CMBaseModel
from fastgen.methods.consistency_model.sCM import SCMModel as SCMBaseModel
from fastgen.methods.distribution_matching.dmd2 import DMD2Model as DMD2BaseModel
from fastgen.networks.discriminators import Discriminator_EDM as BaseDiscriminator_EDM
from fastgen.utils import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from scipy.signal import windows

from omegaconf import DictConfig

PRECISION_MAP = {
    "fp32": "float32",
    "torch.float32": "float32",
    "fp16": "float16",
    "torch.float16": "float16",
    "amp-fp16": "float16",
    "amp-bf16": "bfloat16",
}


def change_block(module, attr, value):
    """Set an attribute on UNetBlock modules, used to override block-level settings like dropout."""
    if isinstance(module, UNetBlock):
        assert hasattr(module, attr), f"Attribute {attr} not found in module"
        setattr(module, attr, value)


class DistillLoss:
    """
    Loss function for CorrDiff-distillation training supported by FastGen framework.
    """

    def __init__(self, regression_net, hr_mean_conditioning=False):
        self.regression_net = regression_net
        self.hr_mean_conditioning = hr_mean_conditioning
        self.y_mean = None

    def compute_loss(
        self,
        net,
        y,
        y_lr,
        y_lr_res,
        lead_time_label=None,
        global_index=None,
        augment_labels=None,
        iteration=None,
    ):
        """Compute the distillation loss by delegating to FastGen's single_train_step."""
        assert not any(p.requires_grad for p in [y, y_lr, y_lr_res])
        data = {
            "real": y,
            # low-res (patched), low-res (unpatched), lead_time_label, global_index, augment_labels
            "condition": (
                y_lr,
                y_lr_res,
                lead_time_label,
                global_index,
                augment_labels,
            ),
            "neg_condition": None,
        }

        loss_map, output = net.single_train_step(data, iteration)
        output["y_mean"] = self.y_mean
        return loss_map["total_loss"], loss_map, output

    def patching(self, y, y_lr, batch_size, patching=None):
        """Extract patches from various input tensors"""
        global_index = None
        if patching is not None:
            # Patched residual
            # (batch_size * patch_num, c_out, patch_shape_y, patch_shape_x)
            y = patching.apply(input=y)
            # Patched conditioning on y_lr and interp(img_lr)
            # (batch_size * patch_num, 2*c_in, patch_shape_y, patch_shape_x)
            y_lr = patching.apply(input=y_lr)

            global_index = patching.global_index(batch_size, y.device)

        return y, y_lr, global_index

    def augment(self, img_clean, img_lr, augment_pipe=None):
        """Apply data augmentation jointly to the clean and low-res images."""
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        return y, y_lr, augment_labels

    def regression(self, y_lr_res, y, lead_time_label=None, augment_labels=None):
        """Run the regression network to produce the mean prediction."""
        if lead_time_label is not None:
            return self.regression_net(
                torch.zeros_like(y, device=y.device),
                y_lr_res,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        return self.regression_net(
            torch.zeros_like(y, device=y.device),
            y_lr_res,
            augment_labels=augment_labels,
        )

    def __call__(
        self,
        net,
        img_clean,
        img_lr,
        patching=None,
        lead_time_label=None,
        augment_pipe=None,
        iteration=None,
        use_patch_grad_acc=False,
    ):
        """Compute the full distillation loss"""
        # Safety check: enforce patching object
        if patching and not isinstance(patching, RandomPatching2D):
            raise ValueError("patching must be a 'RandomPatching2D' object.")
        # Safety check: enforce shapes
        if (
            img_clean.shape[0] != img_lr.shape[0]
            or img_clean.shape[2:] != img_lr.shape[2:]
        ):
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and "
                f"img_lr {img_lr.shape}. "
                f"Batch size, height and width must match."
            )

        # augment for conditional generation
        y, y_lr, augment_labels = self.augment(
            img_clean=img_clean, img_lr=img_lr, augment_pipe=augment_pipe
        )
        del img_clean, img_lr
        y_lr_res = y_lr
        batch_size = y.shape[0]

        # if using multi-iterations of patching, switch to optimized version
        if use_patch_grad_acc:
            if self.y_mean is None:
                self.y_mean = self.regression(
                    y_lr_res=y_lr_res,
                    y=y,
                    lead_time_label=lead_time_label,
                    augment_labels=augment_labels,
                )
        # if on full domain, or if using patching without multi-iterations
        else:
            self.y_mean = self.regression(
                y_lr_res=y_lr_res,
                y=y,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )

        y = y - self.y_mean
        assert not y.requires_grad

        if self.hr_mean_conditioning:
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)
            assert not y_lr.requires_grad

        # patchified training
        # conditioning: cat(y_mean, y_lr, input_interp, pos_embd), 4+12+100+4
        # removed patch_embedding_selector due to compilation issue with dynamo.
        y, y_lr, global_index = self.patching(
            y=y, y_lr=y_lr, batch_size=batch_size, patching=patching
        )

        return self.compute_loss(
            net=net,
            y=y,
            y_lr=y_lr,
            y_lr_res=y_lr_res,
            lead_time_label=lead_time_label,
            global_index=global_index,
            augment_labels=augment_labels,
            iteration=iteration,
        )


class SuperPatching2D(BasePatching2D):
    """Patching utlities which decompose superpatch into regular patches for superpatch-distillation training.

    Parameters
    ----------
    img_shape : Tuple[int, int]
        Height and width of the superpatch :math:`(H, W)`.
    patch_shape : Tuple[int, int]
        Height and width of each patch :math:`(H_p, W_p)`.
        Must divide the superpatch dimensions after accounting for overlap.
    overlap_pix : int, optional
        Number of pixels of overlap between adjacent patches, by default 0.
        When non-zero, the ``fuse`` method averages (or applies windowed
        smoothing to) the overlapping regions during reassembly.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        patch_shape: Tuple[int, int],
        overlap_pix: int = 0,
    ):
        super().__init__(img_shape, patch_shape)
        self.overlap_pix = overlap_pix
        self.patch_shape_y = self.patch_shape[0]
        self.patch_shape_x = self.patch_shape[1]
        self.img_shape_y = self.img_shape[0]
        self.img_shape_x = self.img_shape[1]

        self.num_patches_y, remainder_y = divmod(
            self.img_shape_y - self.overlap_pix, self.patch_shape_y - self.overlap_pix
        )
        self.num_patches_x, remainder_x = divmod(
            self.img_shape_x - self.overlap_pix, self.patch_shape_x - self.overlap_pix
        )
        assert remainder_x == 0 and remainder_y == 0

        # Initialize cache for overlap count
        self.overlap_count = None

    def unfold(self, x):
        """
        Wrapper around torch.nn.functional.unfold to extract regular patches from the superpatch.
        """

        # Cast to float
        dtype = x.dtype
        if dtype == torch.int32:
            x = x.view(torch.float32)
        elif dtype == torch.int64:
            x = x.view(torch.float64)

        x = torch.nn.functional.unfold(
            input=x,
            kernel_size=(self.patch_shape_y, self.patch_shape_x),
            stride=(
                self.patch_shape_y - self.overlap_pix,
                self.patch_shape_x - self.overlap_pix,
            ),
        )

        # cast back
        if dtype in [torch.int32, torch.int64]:
            x = x.view(dtype)

        return x

    def fold(self, x):
        """
        Wrapper around torch.nn.functional.fold to fold the regular patches into the superpatch.
        """
        # Cast to float
        dtype = x.dtype
        if dtype == torch.int32:
            x = x.view(torch.float32)
        elif dtype == torch.int64:
            x = x.view(torch.float64)

        x = torch.nn.functional.fold(
            input=x,
            output_size=(self.img_shape_y, self.img_shape_x),
            kernel_size=(self.patch_shape_y, self.patch_shape_x),
            stride=(
                self.patch_shape_y - self.overlap_pix,
                self.patch_shape_x - self.overlap_pix,
            ),
        )

        # cast back
        if dtype in [torch.int32, torch.int64]:
            x = x.view(dtype)

        return x

    def apply(self, input, additional_input=None):
        """
        Unfold the superpatch into regular patches.
        """
        unfold = self.unfold(input)
        unfold = einops.rearrange(
            unfold,
            "b (c p_h p_w) (nb_p_h nb_p_w) -> (nb_p_w nb_p_h b) c p_h p_w",
            p_h=self.patch_shape_y,
            p_w=self.patch_shape_x,
            nb_p_h=self.num_patches_y,
            nb_p_w=self.num_patches_x,
        )
        if additional_input is not None:
            additional_input = torch.nn.functional.interpolate(
                input=additional_input, size=self.patch_shape, mode="bilinear"
            )
            num_super_patches, rem = divmod(input.shape[0], additional_input.shape[0])
            assert rem == 0, (
                f"{additional_input.shape[0]} must be a factor of {input.shape[0]}"
            )
            repeats = self.num_patches_y * self.num_patches_x * num_super_patches
            # repeat each patch in the batch patch_num times
            additional_input = additional_input.repeat(repeats, 1, 1, 1)
            unfold = torch.cat((unfold, additional_input), dim=1)

        return unfold

    def get_overlap_count(self, device, dtype):
        """
        Compute the overlap count for the overlapping pixels.
        """
        # compute overlap count
        ones = torch.ones(
            (1, 1, self.img_shape_y, self.img_shape_x), device=device, dtype=dtype
        )
        overlap_count = self.unfold(ones)
        return self.fold(overlap_count)

    def fuse(self, input, batch_size=None, window=None):
        """
        Fold the regular patches into the superpatch.
        """
        if window is not None:
            if window.shape[0] == 1:
                window = window.tile((input.shape[0], input.shape[1], 1, 1))

            x = einops.rearrange(
                input * window,
                "(nb_p_w nb_p_h b) c p_h p_w -> b (c p_h p_w) (nb_p_h nb_p_w)",
                p_h=self.patch_shape_y,
                p_w=self.patch_shape_x,
                nb_p_h=self.num_patches_y,
                nb_p_w=self.num_patches_x,
            )
            weights = einops.rearrange(
                window,
                "(nb_p_w nb_p_h b) c p_h p_w -> b (c p_h p_w) (nb_p_h nb_p_w)",
                p_h=self.patch_shape_y,
                p_w=self.patch_shape_x,
                nb_p_h=self.num_patches_y,
                nb_p_w=self.num_patches_x,
            )

            # Stitch patches together (by summing over overlapping patches)
            folded = self.fold(x)
            weights = self.fold(weights)
            return folded / weights
        else:
            # Reshape input to make it 3D to apply fold
            x = einops.rearrange(
                input,
                "(nb_p_w nb_p_h b) c p_h p_w -> b (c p_h p_w) (nb_p_h nb_p_w)",
                p_h=self.patch_shape_y,
                p_w=self.patch_shape_x,
                nb_p_h=self.num_patches_y,
                nb_p_w=self.num_patches_x,
            )
            # Stitch patches together (by summing over overlapping patches)
            folded = self.fold(x)

            if self.overlap_count is None:
                self.overlap_count = self.get_overlap_count(
                    device=folded.device, dtype=folded.dtype
                )
            if not (
                self.overlap_count.dtype == folded.dtype
                and self.overlap_count.device == folded.device
            ):
                self.overlap_count = self.overlap_count.to(folded)

            # Normalize by overlap count
            return folded / self.overlap_count


class FastGenNet(FastGenNetwork):
    """
    A wrapper around the FastGenNetwork in FastGen, which enables distilling CorrDiff models with various methods in FastGen framework.
    Supports super-patching training and window smoothing.

    See `fastgen.networks.network.FastGenNetwork` for more details.
    """

    def __init__(
        self,
        net,
        block_kwargs=None,
        patching=None,
        window=None,
        net_pred_type="x0",
        schedule_type="edm",
        **kwargs,
    ):
        super().__init__(
            net_pred_type=net_pred_type, schedule_type=schedule_type, **kwargs
        )
        self.net = net
        self.logvar_linear = torch.nn.Linear(self.net.model.map_noise.num_channels, 1)
        if block_kwargs is not None:
            for attr, value in block_kwargs.items():
                self.apply(partial(change_block, attr=attr, value=value))

        # patching
        if patching is not None and not isinstance(patching, SuperPatching2D):
            raise ValueError("patching must be a 'SuperPatching2D' object.")
        self.patching = patching
        self.window = window

    def forward(
        self,
        y_t,
        t,
        condition=None,
        return_features_early=False,
        feature_indices=None,
        return_logvar=False,
        fwd_pred_type: Optional[str] = None,
    ):
        """Forward pass with superpatch unfold/fold and optional window smoothing."""
        y_lr, y_lr_res, lead_time_label, global_index, augment_labels = condition
        # squeeze all dims after the first one and expand to batchsize
        t = t.squeeze(list(range(1, t.ndim))).expand(y_t.shape[0])
        assert t.shape == (y_t.shape[0],)

        # Preconditioning weights for input
        y_t_in, t_in = y_t, t

        self.net.model.feature_indices = feature_indices
        self.net.model.features = []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, (
                f"{fwd_pred_type} is not supported as fwd_pred_type"
            )

        # superpatch unfolding: superpatches -> regular patches
        if self.patching is not None:
            y_t = self.patching.apply(y_t)
            y_lr = self.patching.apply(y_lr, additional_input=y_lr_res)

            # TODO(jberner): memory optimizations
            global_index = self.patching.apply(global_index)
            t = t.repeat(self.patching.num_patches_y * self.patching.num_patches_x)

        del y_lr_res

        if lead_time_label is not None:
            out = self.net(
                y_t,
                y_lr,
                t,
                embedding_selector=None,
                global_index=global_index,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            out = self.net(
                y_t,
                y_lr,
                t,
                embedding_selector=None,
                global_index=global_index,
                augment_labels=augment_labels,
            )

        # superpatch folding: regular patches -> superpatches
        if self.patching is not None:
            out = self.patching.fuse(out, window=self.window)

        out = self.noise_scheduler.convert_model_output(
            y_t_in,
            out,
            t_in,
            src_pred_type=self.net_pred_type,
            target_pred_type=fwd_pred_type,
        )

        if feature_indices is not None and len(feature_indices) > 0:
            features = self.net.model.features
            # reset features
            self.net.model.features = None
            assert len(features) == len(feature_indices), (
                f"{len(features)} != {len(feature_indices)}"
            )
            if return_features_early:
                return features
            # score and features; score, features
            out = [out, features]

        if return_logvar:
            emb_timestep = self.net.model.map_noise(t.flatten())
            logvar = self.logvar_linear(emb_timestep)
            return out, logvar
        return out


def build(config: DictConfig, use_ema: bool = False):
    """Build a FastGenNet and optional EMA copy from a distillation config."""
    # Patching
    patching = None
    if "patch_shape" in config:
        patching = SuperPatching2D(
            img_shape=config.input_shape[-2:],
            patch_shape=config.patch_shape,
            overlap_pix=config.overlap_pix,
        )
    window = None
    if "window" in config:
        window = config.window

    # Instantiate the generator network
    net = FastGenNet(
        net=config.net,
        patching=patching,
        window=window,
        train_p_mean=config.sample_t_cfg.train_p_mean,
        train_p_std=config.sample_t_cfg.train_p_std,
        min_t=config.sample_t_cfg.min_t,
        max_t=config.sample_t_cfg.max_t,
        net_pred_type="x0",
        schedule_type="edm",
        block_kwargs=config.get("block_kwargs"),
    )

    net.train().requires_grad_(True)

    # initialize EMA network
    ema = None
    if use_ema:
        ema = deepcopy(net)
        ema.eval().requires_grad_(False)

    return net, ema


class CMModel(CMBaseModel):
    """Consistency Model for Corrdiff distillation.

    A wrapper around the FastGen CM model in FastGen framework.
    See `fastgen.methods.consistency_model.CM.CMModel` for more details.

    References:
        - Song et al., 2023: https://arxiv.org/abs/2303.01469
        - Geng et al., 2024: https://arxiv.org/abs/2406.14548
    """

    def build_model(self):
        """Build the student, EMA, and optional teacher networks for consistency model."""
        self.net, self.ema = build(self.config, use_ema=self.use_ema)

        # instantiate the teacher and consistency network
        if self.config.loss_config.use_cd:
            self.teacher = deepcopy(self.net)
            self.teacher.eval().requires_grad_(False)


class SCMModel(SCMBaseModel):
    """Continuous-time Consistency Model with TrigFlow for CorrDiff distillation.

    A wrapper around the FastGen sCM model in FastGen framework.
    See `fastgen.methods.consistency_model.sCM.SCMModel` for more details.

    References:
        - Lu & Song, 2024: https://arxiv.org/abs/2410.11081
    """

    def build_model(self):
        """Build the student, EMA, and optional teacher networks for sCM model."""
        self.net, self.ema = build(self.config, use_ema=self.use_ema)

        # instantiate the teacher and consistency network
        if self.config.loss_config.use_cd:
            self.teacher = deepcopy(self.net)
            self.teacher.eval().requires_grad_(False)
        else:
            # TODO(jberner): remove this once we do not require a teacher anymore
            self.teacher = torch.nn.Identity()


class Discriminator_EDM(BaseDiscriminator_EDM):
    """EDM Discriminator for CorrDiff distillation.

    A wrapper around the FastGen EDM discriminator in FastGen framework.
    See `fastgen.networks.discriminators.Discriminator_EDM` for more details.
    """

    def __init__(
        self,
        feature_indices=None,
        all_res=[32, 16, 8],
        in_channels=256,
    ):
        torch.nn.Module.__init__(self)
        if feature_indices is None:
            feature_indices = {len(all_res) - 1}  # use the middle bottleneck feature
        self.feature_indices = {
            i for i in feature_indices if i < len(all_res)
        }  # make sure feature indices are valid
        self.in_res = [all_res[i] for i in sorted(feature_indices)]
        if not isinstance(in_channels, (list, tuple)):
            in_channels = [in_channels] * len(self.feature_indices)
        self.in_channels = [in_channels[i] for i in sorted(self.feature_indices)]

        self.discriminator_heads = torch.nn.ModuleList()
        for res, in_channels in zip(self.in_res, self.in_channels):
            layers = []
            while res > 8:
                # reduce the resolution by half, until 8x8
                layers.extend(
                    [
                        torch.nn.Conv2d(
                            kernel_size=4,
                            in_channels=in_channels,
                            out_channels=in_channels,
                            stride=2,
                            padding=1,
                        ),
                        torch.nn.GroupNorm(num_groups=32, num_channels=in_channels),
                        torch.nn.SiLU(),
                    ]
                )
                res //= 2

            layers.extend(
                [
                    torch.nn.Conv2d(
                        kernel_size=4,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=2,
                        padding=1,
                    ),
                    # 8x8 -> 4x4
                    torch.nn.GroupNorm(num_groups=32, num_channels=in_channels),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(
                        kernel_size=4,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=4,
                        padding=0,
                    ),
                    # 4x4 -> 1x1
                    torch.nn.GroupNorm(num_groups=32, num_channels=in_channels),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(
                        kernel_size=1,
                        in_channels=in_channels,
                        out_channels=1,
                        stride=1,
                        padding=0,
                    ),
                    # 1x1 -> 1x1
                ]
            )

            # append the layers for current resolution to the discriminator head
            self.discriminator_heads.append(torch.nn.Sequential(*layers))


class DMD2Model(DMD2BaseModel):
    """VSD + GAN for CorrDiff distillation.

    A wrapper around the FastGen DMD2 model in FastGen framework.
    See `fastgen.methods.distribution_matching.dmd2.DMD2Model` for more details.

    References:
        - 	Yin et al., 2024: https://arxiv.org/abs/2405.14867
    """

    def build_model(self):
        """Build the student, teacher, fake-score, and optional discriminator networks for DMD2."""
        self.net, self.ema = build(self.config, use_ema=self.use_ema)

        # instantiate the teacher and consistency network
        self.teacher = deepcopy(self.net)
        self.teacher.eval().requires_grad_(False)

        # instantiate the fake_score
        self.fake_score = deepcopy(self.net)

        if self.config.gan_loss_weight_gen > 0:
            # instantiate the discriminator in DMD2 ({0, 1, 2} are all features)
            self.discriminator = Discriminator_EDM(
                feature_indices={0, 1, 2},
                all_res=[64, 32, 16],
                in_channels=[64, 128, 128],
            )


MODEL_MAP = {
    "cm": CMModel,
    "scm": SCMModel,
    "dmd2": DMD2Model,
}


def get_window_function(
    patch_shape_x, patch_shape_y, window_alpha, type="KBD", **kwargs
):
    """
    Get the window function for the superpatch.
    returns: window function of shape (patch_shape_y, patch_shape_x)
    """
    functions = {
        "uniform": torch.ones,
        "hann": lambda ps: windows.hann(ps, sym=True),
        "hamming": lambda ps: windows.hamming(ps, sym=True),
        "general_hamming": lambda ps: windows.general_hamming(
            ps, window_alpha, sym=True
        ),
        "kaiser": lambda ps: windows.kaiser(ps, beta=window_alpha * np.pi, sym=True),
        "tukey": lambda ps: windows.tukey(ps, alpha=window_alpha, sym=True),
        "gaussian": lambda ps: windows.gaussian(
            ps, std=window_alpha * ps / 2, sym=True
        ),
        "KBD": lambda ps: windows.kaiser_bessel_derived(ps, window_alpha * np.pi),
    }
    if type not in functions.keys():
        raise ValueError(
            f"Unknown window function type {type}. Supported types are {list(functions.keys())}"
        )

    window_x = torch.tensor(functions[type](patch_shape_x), **kwargs)
    window_y = torch.tensor(functions[type](patch_shape_y), **kwargs)
    window = window_x.unsqueeze(0) * window_y.unsqueeze(1)
    return window


def get_scheduler(name, cfg, optimizer):
    """
    Get the scheduler for the CorrDiff-distillation training supported by FastGen framework.
    """
    if name is None or name == "modulus_default":
        scheduler = LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )  # nul scheduler, lr stays constant
    else:
        schedule = getattr(lr_scheduler, name)(**cfg)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=schedule,
        )
    return scheduler


def few_step_sampler(
    net: torch.nn.Module,
    latents: torch.Tensor,
    img_lr: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    patching: Optional[GridPatching2D] = None,
    mean_hr: Optional[torch.Tensor] = None,
    lead_time_label: Optional[torch.Tensor] = None,
    sigma_max: float = 800,
    sigma_mid: List[float] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Few-step sampler for distillation inference.
    """
    # Safety check on type of patching
    if patching is not None and not isinstance(patching, GridPatching2D):
        raise ValueError("patching must be an instance of GridPatching2D.")

    # Safety check: if patching is used then img_lr and latents must have same
    # height and width, otherwise there is mismatch in the number
    # of patches extracted to form the final batch_size.
    if patching:
        if img_lr.shape[-2:] != latents.shape[-2:]:
            raise ValueError(
                f"img_lr and latents must have the same height and width, "
                f"but found {img_lr.shape[-2:]} vs {latents.shape[-2:]}. "
            )
    # img_lr and latents must also have the same batch_size, otherwise mismatch
    # when processed by the network
    if img_lr.shape[0] != latents.shape[0]:
        raise ValueError(
            f"img_lr and latents must have the same batch size, but found "
            f"{img_lr.shape[0]} vs {latents.shape[0]}."
        )
    batch_size = img_lr.shape[0]

    # latents to dtype if specified
    if dtype is not None:
        latents = latents.to(dtype)

    # Time step discretization.
    sigma_mid = [] if sigma_mid is None else sigma_mid
    # t_0 = T, t_N = 0
    # Max noise level (adjust based on what's supported by the network)
    sigma_max = min(sigma_max, net.sigma_max)
    t_steps = torch.tensor(
        [sigma_max] + list(sigma_mid), dtype=latents.dtype, device=latents.device
    )
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    assert torch.all(t_steps[1:] <= t_steps[:-1])

    # conditioning = [mean_hr, img_lr, global_lr, pos_embd]
    x_lr = img_lr
    if mean_hr is not None:
        if mean_hr.shape[-2:] != img_lr.shape[-2:]:
            raise ValueError(
                f"mean_hr and img_lr must have the same height and width, "
                f"but found {mean_hr.shape[-2:]} vs {img_lr.shape[-2:]}."
            )
        x_lr = torch.cat((mean_hr.expand(x_lr.shape[0], -1, -1, -1), x_lr), dim=1)
    x_lr = x_lr.to(latents.device)

    #  input and position padding + patching
    if patching:
        # Patched conditioning [x_lr, mean_hr]
        # (batch_size * patch_num, C_in + C_out, patch_shape_y, patch_shape_x)
        x_lr = patching.apply(input=x_lr, additional_input=img_lr)

        # Function to select the correct positional embedding for each patch
        def patch_embedding_selector(emb):
            """Select and patch positional embeddings."""
            return patching.apply(emb[None].expand(batch_size, -1, -1, -1))
    else:
        patch_embedding_selector = None

    # Sampling steps
    latents = latents * t_steps[0]
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        # latent patching
        if patching:
            latents = patching.apply(input=latents)

        if lead_time_label is not None:
            latents = net(
                latents,
                x_lr,
                t_cur,
                class_labels,
                lead_time_label=lead_time_label,
                embedding_selector=patch_embedding_selector,
            ).to(latents.dtype)
        else:
            latents = net(
                latents,
                x_lr,
                t_cur,
                class_labels,
                embedding_selector=patch_embedding_selector,
            ).to(latents.dtype)

        if patching:
            # Un-patch the denoised image
            # (batch_size, C_out, img_shape_y, img_shape_x)
            latents = patching.fuse(input=latents, batch_size=batch_size)

        if t_next > 0:
            latents = latents + t_next * torch.randn_like(latents)

    return latents
