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

from pathlib import Path

import pytest
import torch

import physicsnemo
from physicsnemo.nn import get_group_norm
from test import common  # noqa: E402


def _instantiate_model(cls, seed: int = 0, **kwargs):
    """
    Helper function to instantiate a model with reproducible random parameters.
    """
    model: physicsnemo.core.Module = cls(**kwargs)
    gen: torch.Generator = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(
                torch.randn(
                    param.shape,
                    generator=gen,
                    dtype=param.dtype,
                )
            )
    return model


class GroupNormModule(physicsnemo.core.Module):
    """
    A wrapper around get_group_norm that has a factory method to create a model with
    reproducible random parameters.
    """

    _overridable_args: set[str] = {"use_apex_gn"}

    def __init__(self, arch_type: str = "gn_type_1", use_apex_gn: bool = False):
        super().__init__()
        C_in = 64
        # Default parameters
        if arch_type == "gn_type_1":
            self.group_norm = get_group_norm(
                num_channels=C_in,
                use_apex_gn=use_apex_gn,
            )
        # Custom parameters based on num_groups
        elif arch_type == "gn_type_2":
            self.group_norm = get_group_norm(
                num_channels=C_in,
                num_groups=2,
                min_channels_per_group=16,
                eps=1e-3,
                use_apex_gn=use_apex_gn,
            )
        # Custom parameters based on num_groups
        elif arch_type == "gn_type_3":
            self.group_norm = get_group_norm(
                num_channels=C_in,
                num_groups=128,  # min_channels_per_group=2 should prevail
                min_channels_per_group=2,
                eps=1e-3,
                use_apex_gn=use_apex_gn,
            )

    factory: classmethod = classmethod(_instantiate_model)

    def forward(self, x):
        return self.group_norm(x)


def generate_data(device: str):
    """
    Helper function to generate data for the test.
    """
    torch.manual_seed(0)
    B, C_in, H, W = 2, 64, 8, 16
    x: torch.Tensor = torch.randn(B, C_in, H, W).to(device)
    return x


@pytest.mark.parametrize("use_apex_gn", [False, True], ids=["no-apexgn", "apexgn"])
@pytest.mark.parametrize(
    "arch_type",
    ["gn_type_1", "gn_type_2", "gn_type_3"],
    ids=["arch1", "arch2", "arch3"],
)
def test_group_norm_non_regression(apex_device, arch_type, use_apex_gn):
    """
    Test that GroupNorm can be instantiated and compare the output with a
    reference output generated with v1.0.1.
    """
    model: GroupNormModule = GroupNormModule.factory(
        arch_type=arch_type,
        use_apex_gn=use_apex_gn,
    ).to(apex_device)

    # Check that the model is instantiated correctly
    if arch_type == "gn_type_1":
        assert model.group_norm.num_groups == 16
        assert model.group_norm.weight.shape == (64,)
        assert model.group_norm.bias.shape == (64,)
        assert model.group_norm.eps == 1e-5
    elif arch_type == "gn_type_2":
        assert model.group_norm.num_groups == 2
        assert model.group_norm.weight.shape == (64,)
        assert model.group_norm.bias.shape == (64,)
        assert model.group_norm.eps == 1e-3
    elif arch_type == "gn_type_3":
        assert model.group_norm.num_groups == 32
        assert model.group_norm.weight.shape == (64,)
        assert model.group_norm.bias.shape == (64,)
        assert model.group_norm.eps == 1e-3

    x: torch.Tensor = generate_data(apex_device)
    out: torch.Tensor = model(x)

    assert common.validate_accuracy(
        out,
        file_name=f"models/diffusion/data/output_diffusion_{arch_type}-v1.0.1.pth",
    )


# TODO : currently only test overrding use_apex_gn from False to True. Need to
# add test to do the opposite (that is load checkpoint with use_apex_gn=True and
# override it to False)
@pytest.mark.parametrize("use_apex_gn", [False, True], ids=["no-apexgn", "apexgn"])
@pytest.mark.parametrize("chkpt_use_apex_gn", [False], ids=["chkpt-no-apexgn"])
@pytest.mark.parametrize(
    "arch_type",
    ["gn_type_1", "gn_type_2", "gn_type_3"],
    ids=["arch1", "arch2", "arch3"],
)
def test_group_norm_non_regression_from_checkpoint(
    apex_device, arch_type, use_apex_gn, chkpt_use_apex_gn
):
    """
    Tests simple loading and non-regression of a checkpoint generated with the
    get_group_norm class. Also tests the API to override ``use_apex_gn`` to
    use Apex-based group norm when loading the checkpoint.
    """
    script_dir = Path(__file__).parent
    file_name: str = str(
        script_dir / Path(f"data/checkpoint_diffusion_{arch_type}-v1.0.1.mdlus")
    )

    model: physicsnemo.core.Module = physicsnemo.core.Module.from_checkpoint(
        file_name=file_name,
        override_args={"use_apex_gn": use_apex_gn},
    ).to(apex_device)

    # Check that the model is instantiated correctly
    if arch_type == "gn_type_1":
        assert model.group_norm.num_groups == 16
        assert model.group_norm.weight.shape == (64,)
        assert model.group_norm.bias.shape == (64,)
        assert model.group_norm.eps == 1e-5
    elif arch_type == "gn_type_2":
        assert model.group_norm.num_groups == 2
        assert model.group_norm.weight.shape == (64,)
        assert model.group_norm.bias.shape == (64,)
        assert model.group_norm.eps == 1e-3
    elif arch_type == "gn_type_3":
        assert model.group_norm.num_groups == 32
        assert model.group_norm.weight.shape == (64,)
        assert model.group_norm.bias.shape == (64,)
        assert model.group_norm.eps == 1e-3

    x: torch.Tensor = generate_data(apex_device)
    out: torch.Tensor = model(x)

    assert common.validate_accuracy(
        out,
        file_name=f"models/diffusion/data/output_diffusion_{arch_type}-v1.0.1.pth",
    )

    # ---------------------------------------------------------------------------
    #   FOR CHECKPOINT AND DATA GENERATION
    # ---------------------------------------------------------------------------

    # For checkpoint and data generation with v1.0.1
    # class GroupNormModule(physicsnemo.Module):
    #     """
    #     A wrapper around GroupNorm that has a factory method to create a model with
    #     reproducible random parameters.
    #     """

    #     def __init__(self, arch_type: str = "gn_type_1"):
    #         super().__init__()
    #         C_in = 64
    #         # Default parameters
    #         if arch_type == "gn_type_1":
    #             self.group_norm = GroupNorm(num_channels=C_in)
    #         # Custom parameters based on num_groups
    #         elif arch_type == "gn_type_2":
    #             self.group_norm = GroupNorm(
    #                 num_channels=C_in,
    #                 num_groups=2,
    #                 min_channels_per_group=16,
    #                 eps=1e-3,
    #             )
    #         # Custom parameters based on min_channels_per_group
    #         elif arch_type == "gn_type_3":
    #             self.group_norm = GroupNorm(
    #                 num_channels=C_in,
    #                 num_groups=128,  # min_channels_per_group=2 should prevail
    #                 min_channels_per_group=2,
    #                 eps=1e-3,
    #             )

    #     factory: classmethod = classmethod(_instantiate_model)

    #     def forward(self, x):
    #         return self.group_norm(x)

    # @pytest.mark.parametrize(
    #     "arch_type",
    #     ["gn_type_1", "gn_type_2", "gn_type_3"],
    #     ids=["arch1", "arch2", "arch3"],
    # )
    # def test_group_norm_generate_data(arch_type):
    #     """
    #     Function to generate data for the GroupNorm tests.
    #     """

    #     model: GroupNormModule = GroupNormModule.factory(arch_type=arch_type).to("cpu")

    #     # Check that the model is instantiated correctly
    #     if arch_type == "gn_type_1":
    #         assert model.group_norm.num_groups == 16
    #         assert model.group_norm.weight.shape == (64,)
    #         assert model.group_norm.bias.shape == (64,)
    #         assert model.group_norm.eps == 1e-5
    #     elif arch_type == "gn_type_2":
    #         assert model.group_norm.num_groups == 2
    #         assert model.group_norm.weight.shape == (64,)
    #         assert model.group_norm.bias.shape == (64,)
    #         assert model.group_norm.eps == 1e-3
    #     elif arch_type == "gn_type_3":
    #         assert model.group_norm.num_groups == 32
    #         assert model.group_norm.weight.shape == (64,)
    #         assert model.group_norm.bias.shape == (64,)
    #         assert model.group_norm.eps == 1e-3

    #     x: torch.Tensor = generate_data("cpu")
    #     out: torch.Tensor = model(x)

    # model.save(f"checkpoint_diffusion_{arch_type}-v1.0.1.mdlus")
    # torch.save(out, f"output_diffusion_{arch_type}-v1.0.1.pth")
