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

"""Tests for MultiDiffusionModel2D."""

from contextlib import nullcontext
from typing import Any

import pytest
import torch
from jaxtyping import Float, Shaped
from tensordict import TensorDict

from physicsnemo.core import Module
from physicsnemo.diffusion import DiffusionExtraOutputs, DiffusionExtraTimes
from physicsnemo.diffusion.multi_diffusion import MultiDiffusionModel2D
from physicsnemo.diffusion.multi_diffusion.patching import (
    GridPatching2D,
    RandomPatching2D,
)
from physicsnemo.diffusion.preconditioners import EDMPreconditioner

from .conftest import GLOBAL_SEED
from .helpers import (
    Conv2dX0Predictor,
    compare_outputs,
    instantiate_model_deterministic,
    load_or_create_checkpoint,
    load_or_create_reference,
    make_input,
)


class CustomExtraTimes(DiffusionExtraTimes):
    r"""Additional time fields that callers can supply independently."""

    test_t: Float[torch.Tensor, " B"] | None = None


class CustomExtraOutputs(DiffusionExtraOutputs):
    test_pred: Float[torch.Tensor, " B C H W"] | None = None


class StructuredDiffusionModel(Module):
    r"""Linear model with optional destination times and learned loss weights."""

    def __init__(self, output: str = "tuple"):
        super().__init__()
        self.output = output
        self.gain = torch.nn.Parameter(torch.tensor(0.75))

    def forward(
        self,
        x: Float[torch.Tensor, " B *dims"],
        t: Float[torch.Tensor, " B"]
        | tuple[Float[torch.Tensor, " B"], Shaped[DiffusionExtraTimes, " B"]],
        condition: TensorDict | None = None,
        *,
        offset: float = 0.0,
    ) -> (
        Float[torch.Tensor, " B *dims"]
        | tuple[Float[torch.Tensor, " B *dims"], Shaped[DiffusionExtraOutputs, " B"]]
    ):
        r"""Return a prediction, optionally paired with its loss weight."""
        t, extra_times = t if isinstance(t, tuple) else (t, None)
        t_bc = t.reshape(-1, *([1] * (x.ndim - 1)))  # (B, 1, ...)
        prediction = self.gain * x + t_bc + offset
        if extra_times is not None:
            for time in extra_times.to_tensordict().values():
                prediction = prediction + time.reshape(-1, *([1] * (x.ndim - 1)))
        if condition is not None:
            prediction = prediction + condition["shift"]
        if self.output == "tensor":
            return prediction
        if self.output == "custom":
            return prediction, CustomExtraOutputs(
                test_pred=prediction.square(), batch_size=[x.shape[0]]
            )
        return prediction, DiffusionExtraOutputs(
            logvar=self.gain * t, batch_size=[x.shape[0]]
        )


REF_PREFIX = "test_multi_diffusion_models_"

# sigma_data consistent between EDMPreconditioner and EDMNoiseScheduler,
# mirroring the realistic SDA recipe pattern.
SIGMA_DATA = 1.0

# =============================================================================
# Test Model Definitions
# =============================================================================


class UnconditionalConv(Module):
    """Simple unconditional 2D convolutional model."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.net = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, t, condition=None, **kwargs: Any):
        return self.net(x) + t.view(-1, 1, 1, 1)


class ConditionalConv(Module):
    """Conditional model: concatenates condition image to input."""

    def __init__(self, in_channels: int = 6, out_channels: int = 3):
        super().__init__()
        self.net = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t, condition=None, **kwargs: Any):
        img = condition["image"]
        return self.net(torch.cat([x, img], dim=1)) + t.view(-1, 1, 1, 1)


class VecImgCondConv(Module):
    """Conditional model with image (patched) and vector (expanded) inputs."""

    def __init__(self, img_channels: int = 6, out_channels: int = 3, vec_dim: int = 5):
        super().__init__()
        self.net = torch.nn.Conv2d(img_channels, out_channels, kernel_size=3, padding=1)
        self.vec_proj = torch.nn.Linear(vec_dim, out_channels)

    def forward(self, x, t, condition=None, **kwargs: Any):
        img = condition["image"]
        vec = condition["vector"]
        h = self.net(torch.cat([x, img], dim=1))
        return h + self.vec_proj(vec).unsqueeze(-1).unsqueeze(-1) + t.view(-1, 1, 1, 1)


class PosEmbdConv(Module):
    """Conditional model consuming positional embeddings."""

    def __init__(self, in_channels: int = 10, out_channels: int = 3):
        super().__init__()
        self.net = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t, condition=None, **kwargs: Any):
        img = condition["image"]
        pe = condition["positional_embedding"]
        return self.net(torch.cat([x, img, pe], dim=1)) + t.view(-1, 1, 1, 1)


# =============================================================================
# Constants and Configurations
# =============================================================================

# Square image / patch
IMG_H, IMG_W = 16, 16
PATCH_SHAPE = (8, 8)

# Non-square image / patch
IMG_H_NS, IMG_W_NS = 16, 24
PATCH_SHAPE_NS = (8, 12)

BATCH = 2
CHANNELS = 3
PATCH_NUM = 4
VEC_DIM = 5

INPUT_SHAPE = (BATCH, CHANNELS, IMG_H, IMG_W)
INPUT_SHAPE_NS = (BATCH, CHANNELS, IMG_H_NS, IMG_W_NS)

# (config_name, model_cls, model_kwargs, md_kwargs, has_vector_cond)
MD_CONFIGS = [
    (
        "uncond",
        UnconditionalConv,
        {"channels": CHANNELS},
        {},
        False,
    ),
    (
        "cond_patch",
        ConditionalConv,
        {"in_channels": CHANNELS * 2, "out_channels": CHANNELS},
        {"condition_patch": {"image": True}},
        False,
    ),
    (
        "cond_interp",
        ConditionalConv,
        {"in_channels": CHANNELS * 2, "out_channels": CHANNELS},
        {"condition_interp": {"image": True}},
        False,
    ),
    (
        "cond_vec_img",
        VecImgCondConv,
        {"img_channels": CHANNELS * 2, "out_channels": CHANNELS, "vec_dim": VEC_DIM},
        {"condition_patch": {"image": True}},
        True,
    ),
    (
        "posembd_sin",
        PosEmbdConv,
        {"in_channels": CHANNELS + CHANNELS + 4, "out_channels": CHANNELS},
        {
            "condition_patch": {"image": True},
            "positional_embedding": "sinusoidal",
            "channels_positional_embedding": 4,
        },
        False,
    ),
    (
        "posembd_learn",
        PosEmbdConv,
        {"in_channels": CHANNELS + CHANNELS + 4, "out_channels": CHANNELS},
        {
            "condition_patch": {"image": True},
            "positional_embedding": "learnable",
            "channels_positional_embedding": 4,
        },
        False,
    ),
]

EXTRA_TIME_FIELDS = {"extra_r": ("r",), "extra_custom": ("test_t",)}
MD_CONFIGS.extend(
    [
        ("extra_r", StructuredDiffusionModel, {}, {}, False),
        (
            "extra_custom",
            StructuredDiffusionModel,
            {"output": "custom"},
            {"extra_outputs_fuse": {"test_pred": True}},
            False,
        ),
    ]
)


def _make_model_time(config_name, t):
    fields = EXTRA_TIME_FIELDS.get(config_name)
    if fields is None:
        return t
    return t, CustomExtraTimes(
        **{name: t / (index + 2) for index, name in enumerate(fields)},
        batch_size=t.shape,
    )


# Grid patching configs: (img_shape, patch_shape, overlap, boundary, name)
GRID_CONFIGS = [
    ((IMG_H, IMG_W), PATCH_SHAPE, 0, 0, "sq_nooverlap"),
    ((IMG_H, IMG_W), PATCH_SHAPE, 2, 0, "sq_overlap2"),
    ((IMG_H, IMG_W), PATCH_SHAPE, 2, 1, "sq_overlap2_bnd1"),
    ((IMG_H_NS, IMG_W_NS), PATCH_SHAPE_NS, 0, 0, "ns_nooverlap"),
    ((IMG_H_NS, IMG_W_NS), PATCH_SHAPE_NS, 2, 0, "ns_overlap2"),
]


# =============================================================================
# Helpers
# =============================================================================


def _create_md_model(config_name, img_shape=(IMG_H, IMG_W), seed=0):
    """Create a deterministic MultiDiffusionModel2D for the given config."""
    for name, model_cls, model_kwargs, md_kwargs, _ in MD_CONFIGS:
        if name == config_name:
            inner = instantiate_model_deterministic(
                model_cls, seed=seed, **model_kwargs
            )
            return MultiDiffusionModel2D(
                model=inner,
                global_spatial_shape=img_shape,
                **md_kwargs,
            )
    raise ValueError(f"Unknown config: {config_name}")


def _make_condition(config_name, img_shape=(IMG_H, IMG_W), device="cpu"):
    """Create a condition matching the config, or None for unconditional."""
    if config_name == "uncond" or config_name in EXTRA_TIME_FIELDS:
        return None
    H, W = img_shape
    td = {"image": make_input((BATCH, CHANNELS, H, W), seed=99, device=device)}
    for name, _, _, _, has_vector in MD_CONFIGS:
        if name == config_name and has_vector:
            td["vector"] = make_input((BATCH, VEC_DIM), seed=100, device=device)
            break
    return TensorDict(td, batch_size=[BATCH])


def _create_md_model_edm_precond(img_shape=(IMG_H, IMG_W), seed=0):
    """MultiDiffusionModel2D wrapping an EDMPreconditioner.

    Mirrors the realistic SDA recipe: EDMPreconditioner(backbone) is the inner
    model passed to MultiDiffusionModel2D, with sigma_data=SIGMA_DATA consistent
    with the EDMNoiseScheduler used in downstream loss tests.
    """
    backbone = instantiate_model_deterministic(
        Conv2dX0Predictor, seed=seed, channels=CHANNELS
    )
    precond = EDMPreconditioner(backbone, sigma_data=SIGMA_DATA)
    return MultiDiffusionModel2D(model=precond, global_spatial_shape=img_shape)


# =============================================================================
# Constructor Tests
# =============================================================================


class TestConstructor:
    """Tests for MultiDiffusionModel2D constructor and public attributes."""

    @pytest.mark.parametrize(
        "config_name",
        [c[0] for c in MD_CONFIGS],
        ids=[c[0] for c in MD_CONFIGS],
    )
    def test_attributes(self, config_name):
        md = _create_md_model(config_name)
        assert md.global_spatial_shape == (IMG_H, IMG_W)
        assert md._patching is None
        assert isinstance(md.model, Module)

    def test_positional_embedding_sinusoidal(self):
        md = _create_md_model("posembd_sin")
        assert md.pos_embd is not None
        assert md.pos_embd.shape == (4, IMG_H, IMG_W)

    def test_positional_embedding_learnable(self):
        md = _create_md_model("posembd_learn")
        params = [p for p in md.parameters() if p.shape == (4, IMG_H, IMG_W)]
        assert len(params) == 1

    def test_condition_flags_patch(self):
        md = _create_md_model("cond_patch")
        assert md.condition_patch["image"] is True
        assert md.condition_patch["nonexistent"] is False

    def test_condition_flags_interp(self):
        md = _create_md_model("cond_interp")
        assert md.condition_interp["image"] is True
        assert md.condition_interp["nonexistent"] is False
        assert md.condition_patch["image"] is False

    def test_set_random_patching(self):
        md = _create_md_model("uncond")
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        assert isinstance(md._patching, RandomPatching2D)
        assert md._patching.patch_num == PATCH_NUM

    @pytest.mark.parametrize("config_name", ["uncond", *EXTRA_TIME_FIELDS])
    def test_set_grid_patching(self, config_name):
        md = _create_md_model(config_name)
        md.set_grid_patching(patch_shape=PATCH_SHAPE, overlap_pix=2, fuse=True)
        assert isinstance(md._patching, GridPatching2D)
        assert md._fuse is True
        result = md(
            torch.ones(INPUT_SHAPE), _make_model_time(config_name, torch.ones(BATCH))
        )
        prediction, extra_outputs = (
            result if isinstance(result, tuple) else (result, None)
        )
        assert prediction.shape == INPUT_SHAPE
        if extra_outputs is not None:
            assert extra_outputs.batch_size == torch.Size([BATCH])
        if config_name == "extra_custom":
            patches = md.patch_x(torch.ones(INPUT_SHAPE))
            wrong = CustomExtraOutputs(
                test_pred=patches[:, :, :-1, :], batch_size=[patches.shape[0]]
            )
            with pytest.raises(ValueError, match="patch dimensions"):
                md.fuse((patches, wrong), batch_size=BATCH)
        with pytest.raises(ValueError, match="logvar"):
            MultiDiffusionModel2D(
                md.model, (IMG_H, IMG_W), extra_outputs_fuse={"logvar": True}
            )


# =============================================================================
# Non-Regression Tests
# =============================================================================


@pytest.mark.parametrize(
    "config_name",
    [c[0] for c in MD_CONFIGS],
    ids=[c[0] for c in MD_CONFIGS],
)
class TestNonRegression:
    """Non-regression tests for public methods and forward pass."""

    @pytest.mark.parametrize(
        "patch_num", [None, PATCH_NUM], ids=["unpatched", "random"]
    )
    def test_forward_random_non_regression(
        self, deterministic_settings, device, tolerances, config_name, patch_num
    ):
        """Training-like forward with random patching (not pre-patched)."""
        md = _create_md_model(config_name).to(device)
        if patch_num is not None:
            md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=patch_num)

        x0 = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1
        t = _make_model_time(config_name, t)
        condition = _make_condition(config_name, device=device)

        # First call
        with (
            pytest.warns(UserWarning, match="No patching")
            if patch_num is None
            else nullcontext()
        ):
            out1 = md(x0, t, condition=condition)
        out1_values = (
            {"prediction": out1[0], **out1[1].to_tensordict().to_dict()}
            if isinstance(out1, tuple)
            else {"prediction": out1}
        )
        assert out1_values["prediction"].shape == (
            INPUT_SHAPE
            if patch_num is None
            else (patch_num * BATCH, CHANNELS, *PATCH_SHAPE)
        )

        # Re-draw positions, second call
        if patch_num is not None:
            md.reset_patch_indices()
        with (
            pytest.warns(UserWarning, match="No patching")
            if patch_num is None
            else nullcontext()
        ):
            out2 = md(x0, t, condition=condition)
        out2_values = (
            {"prediction": out2[0], **out2[1].to_tensordict().to_dict()}
            if isinstance(out2, tuple)
            else {"prediction": out2}
        )
        assert out2_values["prediction"].shape == out1_values["prediction"].shape

        mode = "unpatched" if patch_num is None else "rand"
        ref_file = f"{REF_PREFIX}{config_name}_fwd_{mode}.pth"
        out1_cpu = (
            {"prediction": out1[0], **out1[1].to_tensordict().to_dict()}
            if isinstance(out1, tuple)
            else {"prediction": out1}
        )
        out1_cpu = {name: value.to(device="cpu") for name, value in out1_cpu.items()}
        out2_cpu = (
            {"prediction": out2[0], **out2[1].to_tensordict().to_dict()}
            if isinstance(out2, tuple)
            else {"prediction": out2}
        )
        out2_cpu = {name: value.to(device="cpu") for name, value in out2_cpu.items()}
        ref = load_or_create_reference(
            ref_file,
            lambda: {
                "out1": out1_cpu,
                "out2": out2_cpu,
            },
        )
        ref_out1_cpu = (
            ref["out1"]
            if isinstance(ref["out1"], dict)
            else {"prediction": ref["out1"]}
        )
        ref_out1_cpu = {
            name: value.to(device="cpu") for name, value in ref_out1_cpu.items()
        }
        torch.testing.assert_close(
            out1_cpu,
            ref_out1_cpu,
            **tolerances,
        )
        ref_out2_cpu = (
            ref["out2"]
            if isinstance(ref["out2"], dict)
            else {"prediction": ref["out2"]}
        )
        ref_out2_cpu = {
            name: value.to(device="cpu") for name, value in ref_out2_cpu.items()
        }
        torch.testing.assert_close(
            out2_cpu,
            ref_out2_cpu,
            **tolerances,
        )

    def test_forward_random_prepatched_non_regression(
        self, deterministic_settings, device, tolerances, config_name
    ):
        """Training-like forward with pre-patched x and t."""
        md = _create_md_model(config_name).to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)

        x0 = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1
        t = _make_model_time(config_name, t)
        condition = _make_condition(config_name, device=device)

        # First call with pre-patched x and t
        x0_patched = md.patch_x(x0)
        t_patched = md.patch_t(t)
        primary, extras = t if isinstance(t, tuple) else (t, None)
        patched_primary, patched_extras = (
            t_patched if isinstance(t_patched, tuple) else (t_patched, None)
        )
        torch.testing.assert_close(patched_primary, primary.repeat(PATCH_NUM))
        if extras is not None:
            assert type(patched_extras) is type(extras)
            assert patched_extras.batch_size == torch.Size([PATCH_NUM * BATCH])
            for name, value in extras.to_tensordict().items():
                torch.testing.assert_close(
                    getattr(patched_extras, name), value.repeat(PATCH_NUM)
                )
        out1 = md(
            x0_patched,
            t_patched,
            condition=condition,
            x_is_patched=True,
            t_is_patched=True,
        )
        out1_values = (
            {"prediction": out1[0], **out1[1].to_tensordict().to_dict()}
            if isinstance(out1, tuple)
            else {"prediction": out1}
        )
        assert out1_values["prediction"].shape == (
            PATCH_NUM * BATCH,
            CHANNELS,
            *PATCH_SHAPE,
        )

        # Re-draw positions, second call
        md.reset_patch_indices()
        x0_patched2 = md.patch_x(x0)
        t_patched2 = md.patch_t(t)
        out2 = md(
            x0_patched2,
            t_patched2,
            condition=condition,
            x_is_patched=True,
            t_is_patched=True,
        )

        ref_file = f"{REF_PREFIX}{config_name}_fwd_rand_prepatched.pth"
        out1_cpu = (
            {"prediction": out1[0], **out1[1].to_tensordict().to_dict()}
            if isinstance(out1, tuple)
            else {"prediction": out1}
        )
        out1_cpu = {name: value.to(device="cpu") for name, value in out1_cpu.items()}
        out2_cpu = (
            {"prediction": out2[0], **out2[1].to_tensordict().to_dict()}
            if isinstance(out2, tuple)
            else {"prediction": out2}
        )
        out2_cpu = {name: value.to(device="cpu") for name, value in out2_cpu.items()}
        ref = load_or_create_reference(
            ref_file,
            lambda: {
                "out1": out1_cpu,
                "out2": out2_cpu,
            },
        )
        ref_out1_cpu = (
            ref["out1"]
            if isinstance(ref["out1"], dict)
            else {"prediction": ref["out1"]}
        )
        ref_out1_cpu = {
            name: value.to(device="cpu") for name, value in ref_out1_cpu.items()
        }
        torch.testing.assert_close(
            out1_cpu,
            ref_out1_cpu,
            **tolerances,
        )
        ref_out2_cpu = (
            ref["out2"]
            if isinstance(ref["out2"], dict)
            else {"prediction": ref["out2"]}
        )
        ref_out2_cpu = {
            name: value.to(device="cpu") for name, value in ref_out2_cpu.items()
        }
        torch.testing.assert_close(
            out2_cpu,
            ref_out2_cpu,
            **tolerances,
        )

    def test_patch_x_non_regression(
        self, deterministic_settings, device, tolerances, config_name
    ):
        """patch_x output matches reference data."""
        md = _create_md_model(config_name).to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)

        out1 = md.patch_x(x)
        md.reset_patch_indices()
        out2 = md.patch_x(x)

        ref_file = f"{REF_PREFIX}{config_name}_patch_x.pth"
        out1_cpu = (
            {"prediction": out1[0], **out1[1].to_tensordict().to_dict()}
            if isinstance(out1, tuple)
            else {"prediction": out1}
        )
        out1_cpu = {name: value.to(device="cpu") for name, value in out1_cpu.items()}
        out2_cpu = (
            {"prediction": out2[0], **out2[1].to_tensordict().to_dict()}
            if isinstance(out2, tuple)
            else {"prediction": out2}
        )
        out2_cpu = {name: value.to(device="cpu") for name, value in out2_cpu.items()}
        ref = load_or_create_reference(
            ref_file,
            lambda: {
                "out1": out1_cpu,
                "out2": out2_cpu,
            },
        )
        ref_out1_cpu = (
            ref["out1"]
            if isinstance(ref["out1"], dict)
            else {"prediction": ref["out1"]}
        )
        ref_out1_cpu = {
            name: value.to(device="cpu") for name, value in ref_out1_cpu.items()
        }
        torch.testing.assert_close(
            out1_cpu,
            ref_out1_cpu,
            **tolerances,
        )
        ref_out2_cpu = (
            ref["out2"]
            if isinstance(ref["out2"], dict)
            else {"prediction": ref["out2"]}
        )
        ref_out2_cpu = {
            name: value.to(device="cpu") for name, value in ref_out2_cpu.items()
        }
        torch.testing.assert_close(
            out2_cpu,
            ref_out2_cpu,
            **tolerances,
        )

    @pytest.mark.parametrize(
        "img_shape,patch_shape,overlap_pix,boundary_pix,grid_name",
        GRID_CONFIGS,
        ids=[c[4] for c in GRID_CONFIGS],
    )
    def test_forward_grid_non_regression(
        self,
        deterministic_settings,
        device,
        tolerances,
        config_name,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        grid_name,
    ):
        """Inference-like forward (grid patching with fuse)."""
        md = _create_md_model(config_name, img_shape=img_shape).to(device)
        md.set_grid_patching(
            patch_shape=patch_shape,
            overlap_pix=overlap_pix,
            boundary_pix=boundary_pix,
            fuse=True,
        )

        H, W = img_shape
        x = make_input((BATCH, CHANNELS, H, W), seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1
        t = _make_model_time(config_name, t)
        condition = _make_condition(config_name, img_shape=img_shape, device=device)
        out = md(x, t, condition=condition)

        out_values = (
            {"prediction": out[0], **out[1].to_tensordict().to_dict()}
            if isinstance(out, tuple)
            else {"prediction": out}
        )
        assert out_values["prediction"].shape == (
            BATCH,
            CHANNELS,
            H,
            W,
        )

        ref_file = f"{REF_PREFIX}{config_name}_{grid_name}_fwd.pth"
        out_cpu = (
            {"prediction": out[0], **out[1].to_tensordict().to_dict()}
            if isinstance(out, tuple)
            else {"prediction": out}
        )
        out_cpu = {name: value.to(device="cpu") for name, value in out_cpu.items()}
        ref = load_or_create_reference(
            ref_file,
            lambda: {"out": out_cpu},
        )
        ref_out_cpu = (
            ref["out"] if isinstance(ref["out"], dict) else {"prediction": ref["out"]}
        )
        ref_out_cpu = {
            name: value.to(device="cpu") for name, value in ref_out_cpu.items()
        }
        torch.testing.assert_close(
            out_cpu,
            ref_out_cpu,
            **tolerances,
        )

    @pytest.mark.parametrize(
        "img_shape,patch_shape,overlap_pix,boundary_pix,grid_name",
        GRID_CONFIGS[:2],
        ids=[c[4] for c in GRID_CONFIGS[:2]],
    )
    def test_forward_grid_condition_prepatched(
        self,
        deterministic_settings,
        device,
        tolerances,
        config_name,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        grid_name,
    ):
        """Grid forward with externally pre-processed condition."""
        if config_name == "uncond" or config_name in EXTRA_TIME_FIELDS:
            pytest.skip("No condition to pre-process for unconditional model.")

        md = _create_md_model(config_name, img_shape=img_shape).to(device)
        md.set_grid_patching(
            patch_shape=patch_shape,
            overlap_pix=overlap_pix,
            boundary_pix=boundary_pix,
            fuse=True,
        )

        H, W = img_shape
        x = make_input((BATCH, CHANNELS, H, W), seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1
        t = _make_model_time(config_name, t)
        condition = _make_condition(config_name, img_shape=img_shape, device=device)

        # Pre-process condition externally
        cond_patched = md.patch_condition(condition)
        out = md(x, t, condition=cond_patched, condition_is_patched=True)

        out_values = (
            {"prediction": out[0], **out[1].to_tensordict().to_dict()}
            if isinstance(out, tuple)
            else {"prediction": out}
        )
        assert out_values["prediction"].shape == (
            BATCH,
            CHANNELS,
            H,
            W,
        )

        ref_file = f"{REF_PREFIX}{config_name}_{grid_name}_fwd_condpatch.pth"
        out_cpu = (
            {"prediction": out[0], **out[1].to_tensordict().to_dict()}
            if isinstance(out, tuple)
            else {"prediction": out}
        )
        out_cpu = {name: value.to(device="cpu") for name, value in out_cpu.items()}
        ref = load_or_create_reference(
            ref_file,
            lambda: {"out": out_cpu},
        )
        ref_out_cpu = (
            ref["out"] if isinstance(ref["out"], dict) else {"prediction": ref["out"]}
        )
        ref_out_cpu = {
            name: value.to(device="cpu") for name, value in ref_out_cpu.items()
        }
        torch.testing.assert_close(
            out_cpu,
            ref_out_cpu,
            **tolerances,
        )

    def test_forward_from_checkpoint(
        self, deterministic_settings, device, tolerances, config_name
    ):
        """Forward from loaded checkpoint matches reference."""

        def create_fn():
            return _create_md_model(config_name)

        ckpt_file = f"{REF_PREFIX}{config_name}.mdlus"
        md = load_or_create_checkpoint(ckpt_file, create_fn).to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)

        x0 = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1
        t = _make_model_time(config_name, t)
        condition = _make_condition(config_name, device=device)

        x0_patched = md.patch_x(x0)
        t_patched = md.patch_t(t)
        out = md(
            x0_patched,
            t_patched,
            condition=condition,
            x_is_patched=True,
            t_is_patched=True,
        )

        ref_file = f"{REF_PREFIX}{config_name}_fwd_rand_prepatched.pth"
        out_cpu = (
            {"prediction": out[0], **out[1].to_tensordict().to_dict()}
            if isinstance(out, tuple)
            else {"prediction": out}
        )
        out_cpu = {name: value.to(device="cpu") for name, value in out_cpu.items()}
        ref = load_or_create_reference(
            ref_file,
            lambda: {"out1": out_cpu},
        )
        ref_out1_cpu = (
            ref["out1"]
            if isinstance(ref["out1"], dict)
            else {"prediction": ref["out1"]}
        )
        ref_out1_cpu = {
            name: value.to(device="cpu") for name, value in ref_out1_cpu.items()
        }
        torch.testing.assert_close(
            out_cpu,
            ref_out1_cpu,
            **tolerances,
        )

    def test_patch_x_from_checkpoint(
        self, deterministic_settings, device, tolerances, config_name
    ):
        """patch_x from loaded checkpoint matches reference produced by fresh instantiation."""

        def create_fn():
            return _create_md_model(config_name)

        ckpt_file = f"{REF_PREFIX}{config_name}.mdlus"
        md = load_or_create_checkpoint(ckpt_file, create_fn).to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        out1 = md.patch_x(x)
        md.reset_patch_indices()
        out2 = md.patch_x(x)

        # Reuse the same reference files created by test_patch_x_non_regression.
        ref_file = f"{REF_PREFIX}{config_name}_patch_x.pth"
        out1_cpu = (
            {"prediction": out1[0], **out1[1].to_tensordict().to_dict()}
            if isinstance(out1, tuple)
            else {"prediction": out1}
        )
        out1_cpu = {name: value.to(device="cpu") for name, value in out1_cpu.items()}
        out2_cpu = (
            {"prediction": out2[0], **out2[1].to_tensordict().to_dict()}
            if isinstance(out2, tuple)
            else {"prediction": out2}
        )
        out2_cpu = {name: value.to(device="cpu") for name, value in out2_cpu.items()}
        ref = load_or_create_reference(
            ref_file,
            lambda: {
                "out1": out1_cpu,
                "out2": out2_cpu,
            },
        )
        ref_out1_cpu = (
            ref["out1"]
            if isinstance(ref["out1"], dict)
            else {"prediction": ref["out1"]}
        )
        ref_out1_cpu = {
            name: value.to(device="cpu") for name, value in ref_out1_cpu.items()
        }
        torch.testing.assert_close(
            out1_cpu,
            ref_out1_cpu,
            **tolerances,
        )
        ref_out2_cpu = (
            ref["out2"]
            if isinstance(ref["out2"], dict)
            else {"prediction": ref["out2"]}
        )
        ref_out2_cpu = {
            name: value.to(device="cpu") for name, value in ref_out2_cpu.items()
        }
        torch.testing.assert_close(
            out2_cpu,
            ref_out2_cpu,
            **tolerances,
        )

    def test_fuse_from_checkpoint(
        self, deterministic_settings, device, tolerances, config_name
    ):
        """fuse from loaded checkpoint produces correct output for no-overlap grid."""

        def create_fn():
            return _create_md_model(config_name)

        ckpt_file = f"{REF_PREFIX}{config_name}.mdlus"
        md = load_or_create_checkpoint(ckpt_file, create_fn).to(device)
        md.set_grid_patching(
            patch_shape=PATCH_SHAPE,
            overlap_pix=0,
            fuse=True,
        )

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        fused = md.fuse(md.patch_x(x), batch_size=BATCH)

        assert fused.shape == x.shape

        ref_file = f"{REF_PREFIX}{config_name}_fuse.pth"
        ref = load_or_create_reference(ref_file, lambda: {"fused": fused.cpu()})
        fused_cpu = (
            {"prediction": fused[0], **fused[1].to_tensordict().to_dict()}
            if isinstance(fused, tuple)
            else {"prediction": fused}
        )
        fused_cpu = {name: value.to(device="cpu") for name, value in fused_cpu.items()}
        ref_fused_cpu = (
            ref["fused"]
            if isinstance(ref["fused"], dict)
            else {"prediction": ref["fused"]}
        )
        ref_fused_cpu = {
            name: value.to(device="cpu") for name, value in ref_fused_cpu.items()
        }
        torch.testing.assert_close(
            fused_cpu,
            ref_fused_cpu,
            **tolerances,
        )


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestGradientFlow:
    """Tests that gradients flow through the model and public methods."""

    @pytest.mark.parametrize("config_name", [c[0] for c in MD_CONFIGS])
    @pytest.mark.parametrize("fuse", [False, True])
    def test_forward_gradient_flow(self, device, config_name, fuse):
        md = _create_md_model(config_name).to(device=device, dtype=torch.float64)
        if fuse:
            md.set_grid_patching(
                patch_shape=PATCH_SHAPE,
                fuse=True,
            )
        else:
            md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)
        x = (
            make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
            .double()
            .requires_grad_(True)
        )
        t = torch.rand(BATCH, device=device, dtype=torch.float64).requires_grad_(True)
        condition = _make_condition(config_name, device=device)
        if condition is not None:
            condition = condition.double()
        fields = EXTRA_TIME_FIELDS.get(config_name)
        values = tuple(
            (t.detach() / (index + 2)).requires_grad_(True)
            for index, _ in enumerate(fields or ())
        )

        def forward(x, t, *values):
            time = (
                t
                if fields is None
                else (
                    t,
                    CustomExtraTimes(**dict(zip(fields, values)), batch_size=[BATCH]),
                )
            )
            output = md(x, time, condition=condition)
            output_values = (
                {"prediction": output[0], **output[1].to_tensordict().to_dict()}
                if isinstance(output, tuple)
                else {"prediction": output}
            )
            return tuple(output_values.values())

        assert torch.autograd.gradcheck(forward, (x, t, *values), fast_mode=True)

    def test_forward_gradient_flow_conditional(self, device):
        md = _create_md_model("cond_patch").to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device).requires_grad_(
            True
        )
        t = torch.rand(BATCH, device=device)
        cond_img = make_input(INPUT_SHAPE, seed=99, device=device).requires_grad_(True)
        condition = TensorDict({"image": cond_img}, batch_size=[BATCH])

        out = md(x, t, condition=condition)
        out.sum().backward()
        assert x.grad is not None
        assert cond_img.grad is not None

    def test_forward_gradient_flow_posembd(self, device):
        md = _create_md_model("posembd_learn").to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device).requires_grad_(
            True
        )
        t = torch.rand(BATCH, device=device)
        condition = _make_condition("posembd_learn", device=device)

        out = md(x, t, condition=condition)
        out.sum().backward()
        assert x.grad is not None
        # Learnable pos_embd should also receive gradients
        assert md.pos_embd.grad is not None

    def test_patch_x_gradient_flow(self, device):
        md = _create_md_model("uncond").to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device).requires_grad_(
            True
        )
        md.patch_x(x).sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

    def test_fuse_gradient_flow(self, device):
        md = _create_md_model("uncond").to(device)
        md.set_grid_patching(
            patch_shape=PATCH_SHAPE,
            fuse=False,
        )

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        patches = md.patch_x(x).detach().requires_grad_(True)
        md.fuse(patches, batch_size=BATCH).sum().backward()
        assert patches.grad is not None and not torch.isnan(patches.grad).any()


# =============================================================================
# torch.compile Tests
# =============================================================================

COMPILE_CONFIGS = [
    "uncond",
    "cond_patch",
    "cond_interp",
    "cond_vec_img",
    *EXTRA_TIME_FIELDS,
]


@pytest.mark.usefixtures("nop_compile")
@pytest.mark.parametrize("config_name", COMPILE_CONFIGS, ids=COMPILE_CONFIGS)
class TestCompile:
    """Tests for torch.compile compatibility across model configurations."""

    def test_forward_random_compile(self, device, config_name):
        """Compiled random-patching forward matches eager; no recompile after
        reset_patch_indices."""
        torch._dynamo.config.error_on_recompile = True
        md = _create_md_model(config_name).to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device).requires_grad_(
            True
        )
        t = torch.rand(BATCH, device=device, requires_grad=True)
        t = _make_model_time(config_name, t)
        condition = _make_condition(config_name, device=device)

        compiled_fn = torch.compile(
            lambda xi, ti: md(xi, ti, condition=condition),
            backend="aot_eager",
            fullgraph=True,
        )

        out_eager = md(x, t, condition=condition)
        out_compiled = compiled_fn(x, t)
        out_eager_cpu = (
            {"prediction": out_eager[0], **out_eager[1].to_tensordict().to_dict()}
            if isinstance(out_eager, tuple)
            else {"prediction": out_eager}
        )
        out_eager_cpu = {
            name: value.to(device="cpu") for name, value in out_eager_cpu.items()
        }
        out_compiled_cpu = (
            {"prediction": out_compiled[0], **out_compiled[1].to_tensordict().to_dict()}
            if isinstance(out_compiled, tuple)
            else {"prediction": out_compiled}
        )
        out_compiled_cpu = {
            name: value.to(device="cpu") for name, value in out_compiled_cpu.items()
        }
        torch.testing.assert_close(
            out_eager_cpu,
            out_compiled_cpu,
        )
        primary_t, extras = t if isinstance(t, tuple) else (t, None)
        inputs = (x, primary_t, *md.parameters())
        out_eager_values = (
            {"prediction": out_eager[0], **out_eager[1].to_tensordict().to_dict()}
            if isinstance(out_eager, tuple)
            else {"prediction": out_eager}
        )
        eager_loss = sum(value.sum() for value in out_eager_values.values())
        out_compiled_values = (
            {"prediction": out_compiled[0], **out_compiled[1].to_tensordict().to_dict()}
            if isinstance(out_compiled, tuple)
            else {"prediction": out_compiled}
        )
        compiled_loss = sum(value.sum() for value in out_compiled_values.values())
        torch.testing.assert_close(
            torch.autograd.grad(compiled_loss, inputs, retain_graph=True),
            torch.autograd.grad(eager_loss, inputs, retain_graph=True),
        )

        # Reset positions and call again — must not recompile
        md.reset_patch_indices()
        out_eager_2 = md(x, t, condition=condition)
        out_compiled_2 = compiled_fn(x, t)
        out_eager_2_cpu = (
            {"prediction": out_eager_2[0], **out_eager_2[1].to_tensordict().to_dict()}
            if isinstance(out_eager_2, tuple)
            else {"prediction": out_eager_2}
        )
        out_eager_2_cpu = {
            name: value.to(device="cpu") for name, value in out_eager_2_cpu.items()
        }
        out_compiled_2_cpu = (
            {
                "prediction": out_compiled_2[0],
                **out_compiled_2[1].to_tensordict().to_dict(),
            }
            if isinstance(out_compiled_2, tuple)
            else {"prediction": out_compiled_2}
        )
        out_compiled_2_cpu = {
            name: value.to(device="cpu") for name, value in out_compiled_2_cpu.items()
        }
        torch.testing.assert_close(
            out_eager_2_cpu,
            out_compiled_2_cpu,
        )

    def test_forward_grid_compile(self, device, config_name):
        """Compiled grid-patching forward matches eager; no recompile on
        second call."""
        torch._dynamo.config.error_on_recompile = True
        md = _create_md_model(config_name).to(device)
        md.set_grid_patching(
            patch_shape=PATCH_SHAPE,
            fuse=True,
        )

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device).requires_grad_(
            True
        )
        t = torch.rand(BATCH, device=device, requires_grad=True)
        t = _make_model_time(config_name, t)
        condition = _make_condition(config_name, device=device)

        compiled_fn = torch.compile(
            lambda xi, ti: md(xi, ti, condition=condition),
            backend="aot_eager",
            fullgraph=True,
        )

        out_eager = md(x, t, condition=condition)
        out_compiled = compiled_fn(x, t)
        out_eager_cpu = (
            {"prediction": out_eager[0], **out_eager[1].to_tensordict().to_dict()}
            if isinstance(out_eager, tuple)
            else {"prediction": out_eager}
        )
        out_eager_cpu = {
            name: value.to(device="cpu") for name, value in out_eager_cpu.items()
        }
        out_compiled_cpu = (
            {"prediction": out_compiled[0], **out_compiled[1].to_tensordict().to_dict()}
            if isinstance(out_compiled, tuple)
            else {"prediction": out_compiled}
        )
        out_compiled_cpu = {
            name: value.to(device="cpu") for name, value in out_compiled_cpu.items()
        }
        torch.testing.assert_close(
            out_eager_cpu,
            out_compiled_cpu,
        )
        primary_t, extras = t if isinstance(t, tuple) else (t, None)
        inputs = (x, primary_t, *md.parameters())
        out_eager_values = (
            {"prediction": out_eager[0], **out_eager[1].to_tensordict().to_dict()}
            if isinstance(out_eager, tuple)
            else {"prediction": out_eager}
        )
        eager_loss = sum(value.sum() for value in out_eager_values.values())
        out_compiled_values = (
            {"prediction": out_compiled[0], **out_compiled[1].to_tensordict().to_dict()}
            if isinstance(out_compiled, tuple)
            else {"prediction": out_compiled}
        )
        compiled_loss = sum(value.sum() for value in out_compiled_values.values())
        torch.testing.assert_close(
            torch.autograd.grad(compiled_loss, inputs, retain_graph=True),
            torch.autograd.grad(eager_loss, inputs, retain_graph=True),
        )

        out_compiled_2 = compiled_fn(x, t)
        out_compiled_2_cpu = (
            {
                "prediction": out_compiled_2[0],
                **out_compiled_2[1].to_tensordict().to_dict(),
            }
            if isinstance(out_compiled_2, tuple)
            else {"prediction": out_compiled_2}
        )
        out_compiled_2_cpu = {
            name: value.to(device="cpu") for name, value in out_compiled_2_cpu.items()
        }
        torch.testing.assert_close(
            out_eager_cpu,
            out_compiled_2_cpu,
        )


# =============================================================================
# Combined Workflow Tests — EDMPreconditioner as inner model
# =============================================================================


class TestWithPreconditionedInnerModel:
    """Tests for MultiDiffusionModel2D wrapping an EDMPreconditioner.

    Verifies the critical wrapping order: EDMPreconditioner(backbone) is the
    inner model passed to MultiDiffusionModel2D, mirroring the realistic SDA
    recipe. Both non-regression (fresh instantiation) and from_checkpoint
    variants compare against the same reference to detect backward-compat breaks.
    """

    def test_forward_random_non_regression(
        self, deterministic_settings, device, tolerances
    ):
        """Forward with random patching matches reference (fresh instantiation)."""
        md = _create_md_model_edm_precond().to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)

        x0 = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1

        out = md(x0, t)
        assert out.shape == (PATCH_NUM * BATCH, CHANNELS, *PATCH_SHAPE)

        ref_file = f"{REF_PREFIX}edm_precond_fwd_rand.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

    def test_forward_from_checkpoint(self, deterministic_settings, device, tolerances):
        """Forward from loaded checkpoint matches same reference as fresh instantiation."""
        md = load_or_create_checkpoint(
            f"{REF_PREFIX}edm_precond.mdlus", _create_md_model_edm_precond
        ).to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)

        x0 = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1

        out = md(x0, t)
        assert out.shape == (PATCH_NUM * BATCH, CHANNELS, *PATCH_SHAPE)

        # Reuse the same reference as test_forward_random_non_regression.
        ref_file = f"{REF_PREFIX}edm_precond_fwd_rand.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)
