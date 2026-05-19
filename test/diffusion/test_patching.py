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

"""Tests for diffusion patching utilities."""

from typing import Tuple

import pytest
import torch

from physicsnemo.diffusion.multi_diffusion import (
    BasePatching2D,
    GridPatching2D,
    RandomPatching2D,
)

from .conftest import GLOBAL_SEED
from .helpers import compare_outputs, load_or_create_reference, make_input

# =============================================================================
# Constants and Configurations
# =============================================================================

# Shared image / input dimensions
IMG_H, IMG_W = 32, 48
BATCH_SIZE = 2
CHANNELS = 3
INPUT_SHAPE: Tuple[int, ...] = (BATCH_SIZE, CHANNELS, IMG_H, IMG_W)

# Additional input for testing the additional_input parameter
ADD_CHANNELS = 2
ADD_H, ADD_W = 8, 12
ADD_INPUT_SHAPE: Tuple[int, ...] = (BATCH_SIZE, ADD_CHANNELS, ADD_H, ADD_W)

# RandomPatching2D configs: (img_shape, patch_shape, patch_num, config_name)
RANDOM_PATCHING_CONFIGS = [
    ((IMG_H, IMG_W), (8, 12), 4, "rand_8x12_4p"),
    ((IMG_H, IMG_W), (16, 16), 2, "rand_16x16_2p"),
]

# GridPatching2D configs: (img_shape, patch_shape, overlap_pix, boundary_pix, config_name)
GRID_PATCHING_CONFIGS = [
    ((IMG_H, IMG_W), (16, 16), 0, 0, "grid_16x16_o0_b0"),
    ((IMG_H, IMG_W), (16, 16), 4, 2, "grid_16x16_o4_b2"),
    ((IMG_H, IMG_W), (12, 16), 2, 0, "grid_12x16_o2_b0"),
]

# fmt: off
# Representative (img_shape, patch_shape, overlap_pix, boundary_pix, expected_patch_num)
# tuples sampling the parameter space across all four axes.
PATCH_NUM_CASES = [
    ((32, 32), (8, 8), 0, 0, 16),
    ((32, 32), (8, 8), 2, 1, 49),
    ((32, 32), (12, 12), 0, 2, 16),
    ((32, 32), (16, 16), 0, 0, 4),
    ((32, 32), (16, 24), 4, 2, 8),
    ((32, 48), (8, 8), 4, 0, 96),
    ((32, 48), (12, 12), 2, 1, 24),
    ((32, 48), (16, 16), 0, 0, 6),
    ((32, 48), (16, 24), 4, 2, 12),
    ((48, 64), (8, 8), 0, 0, 48),
    ((48, 64), (12, 12), 4, 2, 88),
    ((48, 64), (16, 16), 2, 2, 24),
    ((64, 64), (16, 16), 4, 2, 49),
    ((64, 64), (16, 24), 0, 0, 12),
]
# fmt: on


# =============================================================================
# BasePatching2D Tests
# =============================================================================


class TestBasePatching2D:
    """Tests for BasePatching2D abstract class behavior."""

    def test_cannot_instantiate_directly(self):
        """BasePatching2D is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract method"):
            BasePatching2D(img_shape=(32, 32), patch_shape=(8, 8))

    def test_subclass_without_apply_cannot_be_instantiated(self):
        """Subclass that does not implement apply() cannot be instantiated."""

        class IncompletePatching(BasePatching2D):
            pass

        with pytest.raises(TypeError, match="abstract method"):
            IncompletePatching(img_shape=(32, 32), patch_shape=(8, 8))

    def test_subclass_with_trivial_apply(self):
        """Subclass with a trivial apply() can be instantiated and used."""

        class IdentityPatching(BasePatching2D):
            def apply(self, input, **kwargs):
                return input

        patcher = IdentityPatching(img_shape=(32, 48), patch_shape=(8, 12))
        assert patcher.img_shape == (32, 48)
        assert patcher.patch_shape == (8, 12)

        x = torch.randn(2, 3, 32, 48)
        out = patcher.apply(x)
        assert torch.equal(out, x)

    def test_fuse_raises_not_implemented(self):
        """Default fuse() raises NotImplementedError."""

        class IdentityPatching(BasePatching2D):
            def apply(self, input, **kwargs):
                return input

        patcher = IdentityPatching(img_shape=(32, 32), patch_shape=(8, 8))
        with pytest.raises(NotImplementedError, match="fuse"):
            patcher.fuse(torch.randn(2, 3, 8, 8))

    def test_invalid_img_shape_dimensionality(self):
        """img_shape must be 2D."""

        class IdentityPatching(BasePatching2D):
            def apply(self, input, **kwargs):
                return input

        with pytest.raises(ValueError, match="img_shape must be 2D"):
            IdentityPatching(img_shape=(32, 32, 32), patch_shape=(8, 8))

    def test_invalid_patch_shape_dimensionality(self):
        """patch_shape must be 2D."""

        class IdentityPatching(BasePatching2D):
            def apply(self, input, **kwargs):
                return input

        with pytest.raises(ValueError, match="patch_shape must be 2D"):
            IdentityPatching(img_shape=(32, 32), patch_shape=(8, 8, 8))

    def test_patch_shape_clamped_when_larger_than_img(self):
        """patch_shape is clamped to img_shape when larger."""

        class IdentityPatching(BasePatching2D):
            def apply(self, input, **kwargs):
                return input

        with pytest.warns(UserWarning, match="larger than"):
            patcher = IdentityPatching(img_shape=(16, 24), patch_shape=(32, 32))
        assert patcher.patch_shape == (16, 24)

    def test_global_index_uses_apply(self):
        """global_index() delegates to apply() and returns integer coordinates."""

        class IdentityPatching(BasePatching2D):
            def apply(self, input, **kwargs):
                return input

        patcher = IdentityPatching(img_shape=(32, 48), patch_shape=(32, 48))
        gi = patcher.global_index(batch_size=2)
        assert gi.dtype == torch.long
        # Shape: (1, 2, H, W) since identity returns input unchanged
        assert gi.shape[1] == 2


# =============================================================================
# RandomPatching2D Tests
# =============================================================================


class TestRandomPatching2D:
    """Tests for RandomPatching2D."""

    # -----------------------------------------------------------------
    # Constructor and attribute tests
    # -----------------------------------------------------------------

    @pytest.mark.parametrize(
        "img_shape,patch_shape,patch_num",
        [
            ((32, 48), (8, 12), 4),
            ((64, 64), (16, 16), 10),
        ],
        ids=["small", "large"],
    )
    def test_constructor_attributes(self, img_shape, patch_shape, patch_num):
        """Test constructor sets attributes correctly."""
        patcher = RandomPatching2D(
            img_shape=img_shape, patch_shape=patch_shape, patch_num=patch_num
        )
        assert patcher.img_shape == img_shape
        assert patcher.patch_shape == patch_shape
        assert patcher.patch_num == patch_num
        assert isinstance(patcher, BasePatching2D)

    def test_set_patch_num(self):
        """set_patch_num() updates patch_num."""
        patcher = RandomPatching2D(img_shape=(32, 48), patch_shape=(8, 12), patch_num=4)
        assert patcher.patch_num == 4

        patcher.set_patch_num(8)
        assert patcher.patch_num == 8

    # -----------------------------------------------------------------
    # Non-regression tests for apply()
    # -----------------------------------------------------------------

    @pytest.mark.parametrize(
        "img_shape,patch_shape,patch_num,config_name",
        RANDOM_PATCHING_CONFIGS,
        ids=[c[3] for c in RANDOM_PATCHING_CONFIGS],
    )
    def test_apply_non_regression(
        self,
        deterministic_settings,
        device,
        tolerances,
        img_shape,
        patch_shape,
        patch_num,
        config_name,
    ):
        """Test apply() output matches reference data."""
        patcher = RandomPatching2D(
            img_shape=img_shape, patch_shape=patch_shape, patch_num=patch_num
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        out = patcher.apply(x)

        assert out.shape == (BATCH_SIZE * patch_num, CHANNELS, *patcher.patch_shape)

        ref_file = f"test_patching_random_{config_name}_apply.pth"
        ref_data = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref_data["out"], **tolerances)

    @pytest.mark.parametrize(
        "img_shape,patch_shape,patch_num,config_name",
        RANDOM_PATCHING_CONFIGS,
        ids=[c[3] for c in RANDOM_PATCHING_CONFIGS],
    )
    def test_apply_with_additional_input_non_regression(
        self,
        deterministic_settings,
        device,
        tolerances,
        img_shape,
        patch_shape,
        patch_num,
        config_name,
    ):
        """Test apply() with additional_input concatenation against reference."""
        patcher = RandomPatching2D(
            img_shape=img_shape, patch_shape=patch_shape, patch_num=patch_num
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        add_input = make_input(ADD_INPUT_SHAPE, seed=GLOBAL_SEED + 1, device=device)
        out = patcher.apply(x, additional_input=add_input)

        expected_channels = CHANNELS + ADD_CHANNELS
        assert out.shape == (
            BATCH_SIZE * patch_num,
            expected_channels,
            *patcher.patch_shape,
        )

        ref_file = f"test_patching_random_{config_name}_apply_add.pth"
        ref_data = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref_data["out"], **tolerances)

    # -----------------------------------------------------------------
    # reset_patch_indices behavior tests (via public methods)
    # -----------------------------------------------------------------

    def test_apply_changes_after_reset_patch_indices(
        self, deterministic_settings, device, tolerances
    ):
        """apply() produces different output after reset_patch_indices()."""
        patcher = RandomPatching2D(
            img_shape=(IMG_H, IMG_W), patch_shape=(8, 12), patch_num=4
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)

        out_before = patcher.apply(x)
        patcher.reset_patch_indices()
        out_after = patcher.apply(x)

        assert not torch.equal(out_before, out_after)

        ref_file = "test_patching_random_apply_after_reset.pth"
        ref_data = load_or_create_reference(ref_file, lambda: {"out": out_after.cpu()})
        compare_outputs(out_after, ref_data["out"], **tolerances)

    def test_global_index_changes_after_reset_patch_indices(
        self, deterministic_settings, device, tolerances
    ):
        """global_index() produces different output after reset_patch_indices()."""
        patcher = RandomPatching2D(
            img_shape=(IMG_H, IMG_W), patch_shape=(8, 12), patch_num=4
        )

        gi_before = patcher.global_index(batch_size=BATCH_SIZE, device=device)
        patcher.reset_patch_indices()
        gi_after = patcher.global_index(batch_size=BATCH_SIZE, device=device)

        assert not torch.equal(gi_before, gi_after)

        ref_file = "test_patching_random_global_index_after_reset.pth"
        ref_data = load_or_create_reference(
            ref_file, lambda: {"global_index": gi_after.cpu()}
        )
        compare_outputs(
            gi_after.float(), ref_data["global_index"].float(), **tolerances
        )

    # -----------------------------------------------------------------
    # Non-regression test for global_index()
    # -----------------------------------------------------------------

    @pytest.mark.parametrize(
        "img_shape,patch_shape,patch_num,config_name",
        RANDOM_PATCHING_CONFIGS,
        ids=[c[3] for c in RANDOM_PATCHING_CONFIGS],
    )
    def test_global_index_non_regression(
        self,
        deterministic_settings,
        device,
        tolerances,
        img_shape,
        patch_shape,
        patch_num,
        config_name,
    ):
        """Test global_index() against reference data."""
        patcher = RandomPatching2D(
            img_shape=img_shape, patch_shape=patch_shape, patch_num=patch_num
        )
        gi = patcher.global_index(batch_size=BATCH_SIZE, device=device)

        assert gi.shape == (patch_num, 2, *patcher.patch_shape)

        ref_file = f"test_patching_random_{config_name}_global_index.pth"
        ref_data = load_or_create_reference(
            ref_file, lambda: {"global_index": gi.cpu()}
        )
        compare_outputs(gi.float(), ref_data["global_index"].float(), **tolerances)

    # -----------------------------------------------------------------
    # Gradient flow tests
    # -----------------------------------------------------------------

    def test_apply_gradient_flow(self, device):
        """Gradients flow through apply()."""
        patcher = RandomPatching2D(
            img_shape=(IMG_H, IMG_W), patch_shape=(8, 12), patch_num=4
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        x = x.requires_grad_(True)

        out = patcher.apply(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_apply_gradient_flow_with_additional_input(self, device):
        """Gradients flow through apply() to both input and additional_input."""
        patcher = RandomPatching2D(
            img_shape=(IMG_H, IMG_W), patch_shape=(8, 12), patch_num=4
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        add_input = make_input(ADD_INPUT_SHAPE, seed=GLOBAL_SEED + 1, device=device)
        x = x.requires_grad_(True)
        add_input = add_input.requires_grad_(True)

        out = patcher.apply(x, additional_input=add_input)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert add_input.grad is not None
        assert not torch.isnan(add_input.grad).any()

    # -----------------------------------------------------------------
    # torch.compile tests
    # -----------------------------------------------------------------
    @pytest.mark.usefixtures("nop_compile")
    def test_apply_torch_compile(self, device):
        """apply() is compatible with torch.compile and does not recompile
        after reset_patch_indices()."""
        torch._dynamo.config.error_on_recompile = True
        patcher = RandomPatching2D(
            img_shape=(IMG_H, IMG_W), patch_shape=(8, 12), patch_num=4
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)

        def apply_fn(tensor):
            return patcher.apply(tensor)

        compiled_fn = torch.compile(apply_fn, fullgraph=True)

        out_eager = patcher.apply(x)
        out_compiled = compiled_fn(x)
        torch.testing.assert_close(out_eager, out_compiled)

        # After resetting patch indices the compiled function should still
        # work without recompilation
        patcher.reset_patch_indices()
        out_eager_2 = patcher.apply(x)
        out_compiled_2 = compiled_fn(x)
        torch.testing.assert_close(out_eager_2, out_compiled_2)


# =============================================================================
# GridPatching2D Tests
# =============================================================================


class TestGridPatching2D:
    """Tests for GridPatching2D."""

    # -----------------------------------------------------------------
    # Constructor and attribute tests
    # -----------------------------------------------------------------

    @pytest.mark.parametrize(
        "img_shape,patch_shape,overlap_pix,boundary_pix",
        [
            ((32, 48), (16, 16), 0, 0),
            ((32, 48), (16, 16), 4, 2),
            ((64, 64), (16, 16), 0, 0),
        ],
        ids=["no-overlap", "overlap-boundary", "large-no-overlap"],
    )
    def test_constructor_attributes(
        self, img_shape, patch_shape, overlap_pix, boundary_pix
    ):
        """Test constructor sets attributes correctly."""
        patcher = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            overlap_pix=overlap_pix,
            boundary_pix=boundary_pix,
        )
        assert patcher.img_shape == img_shape
        assert patcher.patch_shape == patch_shape
        assert patcher.overlap_pix == overlap_pix
        assert patcher.boundary_pix == boundary_pix
        assert isinstance(patcher.patch_num, int)
        assert patcher.patch_num > 0
        assert isinstance(patcher, BasePatching2D)

    @pytest.mark.parametrize(
        "img_shape,patch_shape,overlap_pix,boundary_pix,expected_patch_num",
        PATCH_NUM_CASES,
        ids=[f"img{c[0]}_p{c[1]}_o{c[2]}_b{c[3]}" for c in PATCH_NUM_CASES],
    )
    def test_patch_num_calculation(
        self,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        expected_patch_num,
    ):
        """patch_num matches expected value across representative parameter combinations."""
        patcher = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            overlap_pix=overlap_pix,
            boundary_pix=boundary_pix,
        )
        assert patcher.patch_num == expected_patch_num

    # -----------------------------------------------------------------
    # Non-regression tests for apply()
    # -----------------------------------------------------------------

    @pytest.mark.parametrize(
        "img_shape,patch_shape,overlap_pix,boundary_pix,config_name",
        GRID_PATCHING_CONFIGS,
        ids=[c[4] for c in GRID_PATCHING_CONFIGS],
    )
    def test_apply_non_regression(
        self,
        deterministic_settings,
        device,
        tolerances,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        config_name,
    ):
        """Test apply() output matches reference data."""
        patcher = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            overlap_pix=overlap_pix,
            boundary_pix=boundary_pix,
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        out = patcher.apply(x)

        expected_shape = (
            BATCH_SIZE * patcher.patch_num,
            CHANNELS,
            *patcher.patch_shape,
        )
        assert out.shape == expected_shape

        ref_file = f"test_patching_grid_{config_name}_apply.pth"
        ref_data = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref_data["out"], **tolerances)

    @pytest.mark.parametrize(
        "img_shape,patch_shape,overlap_pix,boundary_pix,config_name",
        GRID_PATCHING_CONFIGS,
        ids=[c[4] for c in GRID_PATCHING_CONFIGS],
    )
    def test_apply_with_additional_input_non_regression(
        self,
        deterministic_settings,
        device,
        tolerances,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        config_name,
    ):
        """Test apply() with additional_input concatenation against reference."""
        patcher = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            overlap_pix=overlap_pix,
            boundary_pix=boundary_pix,
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        add_input = make_input(ADD_INPUT_SHAPE, seed=GLOBAL_SEED + 1, device=device)
        out = patcher.apply(x, additional_input=add_input)

        expected_channels = CHANNELS + ADD_CHANNELS
        expected_shape = (
            BATCH_SIZE * patcher.patch_num,
            expected_channels,
            *patcher.patch_shape,
        )
        assert out.shape == expected_shape

        ref_file = f"test_patching_grid_{config_name}_apply_add.pth"
        ref_data = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref_data["out"], **tolerances)

    # -----------------------------------------------------------------
    # Non-regression tests for fuse()
    # -----------------------------------------------------------------

    @pytest.mark.parametrize(
        "img_shape,patch_shape,overlap_pix,boundary_pix,config_name",
        GRID_PATCHING_CONFIGS,
        ids=[c[4] for c in GRID_PATCHING_CONFIGS],
    )
    def test_fuse_non_regression(
        self,
        deterministic_settings,
        device,
        tolerances,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        config_name,
    ):
        """Test fuse() output matches reference data."""
        patcher = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            overlap_pix=overlap_pix,
            boundary_pix=boundary_pix,
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        patches = patcher.apply(x)
        fused = patcher.fuse(patches, batch_size=BATCH_SIZE)

        assert fused.shape == INPUT_SHAPE

        ref_file = f"test_patching_grid_{config_name}_fuse.pth"
        ref_data = load_or_create_reference(ref_file, lambda: {"out": fused.cpu()})
        compare_outputs(fused, ref_data["out"], **tolerances)

    # -----------------------------------------------------------------
    # Roundtrip test: apply -> fuse ~ identity
    # -----------------------------------------------------------------

    @pytest.mark.parametrize(
        "img_shape,patch_shape,overlap_pix,boundary_pix",
        [
            ((IMG_H, IMG_W), (8, 8), 0, 0),
            ((IMG_H, IMG_W), (16, 16), 4, 0),
            ((IMG_H, IMG_W), (16, 16), 4, 2),
        ],
        ids=["exact-tiling", "with-overlap", "with-overlap-and-boundary"],
    )
    def test_apply_then_fuse_roundtrip(
        self, device, img_shape, patch_shape, overlap_pix, boundary_pix
    ):
        """apply() then fuse() reconstructs the original image."""
        patcher = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            overlap_pix=overlap_pix,
            boundary_pix=boundary_pix,
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        patches = patcher.apply(x)
        fused = patcher.fuse(patches, batch_size=BATCH_SIZE)

        torch.testing.assert_close(fused, x, atol=1e-5, rtol=1e-5)

    # -----------------------------------------------------------------
    # Non-regression test for global_index()
    # -----------------------------------------------------------------

    @pytest.mark.parametrize(
        "img_shape,patch_shape,overlap_pix,boundary_pix,config_name",
        GRID_PATCHING_CONFIGS,
        ids=[c[4] for c in GRID_PATCHING_CONFIGS],
    )
    def test_global_index_non_regression(
        self,
        deterministic_settings,
        device,
        tolerances,
        img_shape,
        patch_shape,
        overlap_pix,
        boundary_pix,
        config_name,
    ):
        """Test global_index() against reference data."""
        patcher = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            overlap_pix=overlap_pix,
            boundary_pix=boundary_pix,
        )
        gi = patcher.global_index(batch_size=BATCH_SIZE, device=device)

        assert gi.shape == (patcher.patch_num, 2, *patcher.patch_shape)

        ref_file = f"test_patching_grid_{config_name}_global_index.pth"
        ref_data = load_or_create_reference(
            ref_file, lambda: {"global_index": gi.cpu()}
        )
        compare_outputs(gi.float(), ref_data["global_index"].float(), **tolerances)

    # -----------------------------------------------------------------
    # Gradient flow tests
    # -----------------------------------------------------------------

    def test_apply_gradient_flow(self, device):
        """Gradients flow through apply()."""
        patcher = GridPatching2D(
            img_shape=(IMG_H, IMG_W),
            patch_shape=(16, 16),
            overlap_pix=0,
            boundary_pix=0,
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        x = x.requires_grad_(True)

        out = patcher.apply(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_fuse_gradient_flow(self, device):
        """Gradients flow through fuse()."""
        patcher = GridPatching2D(
            img_shape=(IMG_H, IMG_W),
            patch_shape=(16, 16),
            overlap_pix=0,
            boundary_pix=0,
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        patches = patcher.apply(x)
        patches = patches.detach().requires_grad_(True)

        fused = patcher.fuse(patches, batch_size=BATCH_SIZE)
        loss = fused.sum()
        loss.backward()

        assert patches.grad is not None
        assert not torch.isnan(patches.grad).any()

    def test_apply_gradient_flow_with_additional_input(self, device):
        """Gradients flow through apply() to both input and additional_input."""
        patcher = GridPatching2D(
            img_shape=(IMG_H, IMG_W),
            patch_shape=(16, 16),
            overlap_pix=0,
            boundary_pix=0,
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        add_input = make_input(ADD_INPUT_SHAPE, seed=GLOBAL_SEED + 1, device=device)
        x = x.requires_grad_(True)
        add_input = add_input.requires_grad_(True)

        out = patcher.apply(x, additional_input=add_input)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert add_input.grad is not None
        assert not torch.isnan(add_input.grad).any()

    # -----------------------------------------------------------------
    # torch.compile tests
    # -----------------------------------------------------------------

    @pytest.mark.usefixtures("nop_compile")
    def test_apply_torch_compile(self, device):
        """apply() is compatible with torch.compile."""
        torch._dynamo.config.error_on_recompile = True
        patcher = GridPatching2D(
            img_shape=(IMG_H, IMG_W),
            patch_shape=(16, 16),
            overlap_pix=0,
            boundary_pix=0,
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)

        def apply_fn(tensor):
            return patcher.apply(tensor)

        compiled_fn = torch.compile(apply_fn, fullgraph=True)

        out_eager = patcher.apply(x)
        out_compiled = compiled_fn(x)
        torch.testing.assert_close(out_eager, out_compiled)

        # Second call should not trigger recompilation
        out_compiled_2 = compiled_fn(x)
        torch.testing.assert_close(out_eager, out_compiled_2)

    @pytest.mark.usefixtures("nop_compile")
    def test_fuse_torch_compile(self, device):
        """fuse() is compatible with torch.compile."""
        torch._dynamo.config.error_on_recompile = True
        patcher = GridPatching2D(
            img_shape=(IMG_H, IMG_W),
            patch_shape=(16, 16),
            overlap_pix=0,
            boundary_pix=0,
        )
        x = make_input(INPUT_SHAPE, seed=GLOBAL_SEED, device=device)
        patches = patcher.apply(x)

        def fuse_fn(tensor):
            return patcher.fuse(tensor, batch_size=BATCH_SIZE)

        compiled_fn = torch.compile(fuse_fn, fullgraph=True)

        out_eager = patcher.fuse(patches, batch_size=BATCH_SIZE)
        out_compiled = compiled_fn(patches)
        torch.testing.assert_close(out_eager, out_compiled)

        # Second call should not trigger recompilation
        out_compiled_2 = compiled_fn(patches)
        torch.testing.assert_close(out_eager, out_compiled_2)
