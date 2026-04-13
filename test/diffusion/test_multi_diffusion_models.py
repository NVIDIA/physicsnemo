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

from typing import Any

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.core import Module
from physicsnemo.diffusion.multi_diffusion import MultiDiffusionModel2D

from .conftest import GLOBAL_SEED
from .helpers import (
    compare_outputs,
    instantiate_model_deterministic,
    load_or_create_checkpoint,
    load_or_create_reference,
    make_input,
)

REF_PREFIX = "test_multi_diffusion_models_"

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
    if config_name == "uncond":
        return None
    H, W = img_shape
    td = {"image": make_input((BATCH, CHANNELS, H, W), seed=99, device=device)}
    for name, _, _, _, has_vector in MD_CONFIGS:
        if name == config_name and has_vector:
            td["vector"] = make_input((BATCH, VEC_DIM), seed=100, device=device)
            break
    return TensorDict(td, batch_size=[BATCH])


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
        assert md._patching_type is None
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
        assert md._patching_type == "random"
        assert md._patching.patch_num == PATCH_NUM

    def test_set_grid_patching(self):
        md = _create_md_model("uncond")
        md.set_grid_patching(patch_shape=PATCH_SHAPE, overlap_pix=2, fuse=True)
        assert md._patching_type == "grid"
        assert md._fuse is True


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

    def test_forward_random_non_regression(
        self, deterministic_settings, device, tolerances, config_name
    ):
        """Training-like forward with random patching (not pre-patched)."""
        md = _create_md_model(config_name).to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)

        x0 = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1
        condition = _make_condition(config_name, device=device)

        # First call
        out1 = md(x0, t, condition=condition)
        assert out1.shape == (PATCH_NUM * BATCH, CHANNELS, *PATCH_SHAPE)

        # Re-draw positions, second call
        md.reset_patch_indices()
        out2 = md(x0, t, condition=condition)
        assert out2.shape == out1.shape

        ref_file = f"{REF_PREFIX}{config_name}_fwd_rand.pth"
        ref = load_or_create_reference(
            ref_file, lambda: {"out1": out1.cpu(), "out2": out2.cpu()}
        )
        compare_outputs(out1, ref["out1"], **tolerances)
        compare_outputs(out2, ref["out2"], **tolerances)

    def test_forward_random_prepatched_non_regression(
        self, deterministic_settings, device, tolerances, config_name
    ):
        """Training-like forward with pre-patched x and t."""
        md = _create_md_model(config_name).to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)

        x0 = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = make_input((BATCH,), seed=GLOBAL_SEED + 1, device=device).abs() + 0.1
        condition = _make_condition(config_name, device=device)

        # First call with pre-patched x and t
        x0_patched = md.patch_x(x0)
        t_patched = md.patch_t(t)
        out1 = md(
            x0_patched,
            t_patched,
            condition=condition,
            x_is_patched=True,
            t_is_patched=True,
        )
        assert out1.shape == (PATCH_NUM * BATCH, CHANNELS, *PATCH_SHAPE)

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
        ref = load_or_create_reference(
            ref_file, lambda: {"out1": out1.cpu(), "out2": out2.cpu()}
        )
        compare_outputs(out1, ref["out1"], **tolerances)
        compare_outputs(out2, ref["out2"], **tolerances)

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
        ref = load_or_create_reference(
            ref_file, lambda: {"out1": out1.cpu(), "out2": out2.cpu()}
        )
        compare_outputs(out1, ref["out1"], **tolerances)
        compare_outputs(out2, ref["out2"], **tolerances)

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
        condition = _make_condition(config_name, img_shape=img_shape, device=device)
        out = md(x, t, condition=condition)

        assert out.shape == (BATCH, CHANNELS, H, W)

        ref_file = f"{REF_PREFIX}{config_name}_{grid_name}_fwd.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

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
        if config_name == "uncond":
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
        condition = _make_condition(config_name, img_shape=img_shape, device=device)

        # Pre-process condition externally
        cond_patched = md.patch_condition(condition)
        out = md(x, t, condition=cond_patched, condition_is_patched=True)

        assert out.shape == (BATCH, CHANNELS, H, W)

        ref_file = f"{REF_PREFIX}{config_name}_{grid_name}_fwd_condpatch.pth"
        ref = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref["out"], **tolerances)

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
        ref = load_or_create_reference(ref_file, lambda: {"out1": out.cpu()})
        compare_outputs(out, ref["out1"], **tolerances)


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestGradientFlow:
    """Tests that gradients flow through the model and public methods."""

    def test_forward_gradient_flow(self, device):
        md = _create_md_model("uncond").to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device).requires_grad_(
            True
        )
        t = torch.rand(BATCH, device=device)

        out = md(x, t)
        out.sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

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
        md.set_grid_patching(patch_shape=PATCH_SHAPE, fuse=False)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        patches = md.patch_x(x).detach().requires_grad_(True)
        md.fuse(patches, batch_size=BATCH).sum().backward()
        assert patches.grad is not None and not torch.isnan(patches.grad).any()


# =============================================================================
# torch.compile Tests
# =============================================================================

COMPILE_CONFIGS = ["uncond", "cond_patch", "cond_interp", "cond_vec_img"]


@pytest.mark.parametrize("config_name", COMPILE_CONFIGS, ids=COMPILE_CONFIGS)
class TestCompile:
    """Tests for torch.compile compatibility across model configurations."""

    def test_forward_random_compile(self, device, config_name):
        """Compiled random-patching forward matches eager; no recompile after
        reset_patch_indices."""
        torch._dynamo.config.error_on_recompile = True
        md = _create_md_model(config_name).to(device)
        md.set_random_patching(patch_shape=PATCH_SHAPE, patch_num=PATCH_NUM)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = torch.rand(BATCH, device=device)
        condition = _make_condition(config_name, device=device)

        compiled_fn = torch.compile(
            lambda xi, ti: md(xi, ti, condition=condition), fullgraph=True
        )

        out_eager = md(x, t, condition=condition)
        out_compiled = compiled_fn(x, t)
        torch.testing.assert_close(out_eager, out_compiled)

        # Reset positions and call again — must not recompile
        md.reset_patch_indices()
        out_eager_2 = md(x, t, condition=condition)
        out_compiled_2 = compiled_fn(x, t)
        torch.testing.assert_close(out_eager_2, out_compiled_2)

    def test_forward_grid_compile(self, device, config_name):
        """Compiled grid-patching forward matches eager; no recompile on
        second call."""
        torch._dynamo.config.error_on_recompile = True
        md = _create_md_model(config_name).to(device)
        md.set_grid_patching(patch_shape=PATCH_SHAPE, fuse=True)

        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        t = torch.rand(BATCH, device=device)
        condition = _make_condition(config_name, device=device)

        compiled_fn = torch.compile(
            lambda xi, ti: md(xi, ti, condition=condition), fullgraph=True
        )

        out_eager = md(x, t, condition=condition)
        out_compiled = compiled_fn(x, t)
        torch.testing.assert_close(out_eager, out_compiled)

        out_compiled_2 = compiled_fn(x, t)
        torch.testing.assert_close(out_eager, out_compiled_2)
