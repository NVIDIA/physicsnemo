# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import logging
from pathlib import Path
from typing import Tuple

import torch

import physicsnemo

from .utils import compare_output

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


@torch.no_grad()
def validate_checkpoint(
    model_1: physicsnemo.Module,
    model_2: physicsnemo.Module,
    in_args: Tuple[Tensor] = (),
    rtol: float = 1e-5,
    atol: float = 1e-5,
    enable_autocast: bool = False,
) -> bool:
    """Check network's checkpoint safely saves and loads the state of the model

    This test will check if a model's state is fully saved in its checkpoint. Two
    seperately initialized models should be provided. One model will load the other's
    checkpoint and produce the same output.

    Parameters
    ----------
    model_1 : physicsnemo.Module
        PhysicsNeMo model to save checkpoint from
    model_2 : physicsnemo.Module
        PhysicsNeMo model to load checkpoint to
    in_args : Tuple[Tensor], optional
        Input arguments, by default ()
    rtol : float, optional
        Relative tolerance of error allowed, by default 1e-5
    atol : float, optional
        Absolute tolerance of error allowed, by default 1e-5
    enable_autocast: bool, optional
        Whether to enable autocast in model forward

    Returns
    -------
    bool
        Test passed
    """
    # First check fail safes of save/load functions
    try:
        model_1.save("folder_does_not_exist/checkpoint.mdlus")
    except IOError:
        pass

    try:
        model_1.load("does_not_exist.mdlus")
    except IOError:
        pass

    # Now test forward passes
    with torch.autocast("cuda", enabled=enable_autocast):
        output_1 = model_1.forward(*in_args)
        output_2 = model_2.forward(*in_args)

    # Model outputs should initially be different
    assert not compare_output(
        output_1, output_2, rtol, atol
    ), "Model outputs should initially be different"

    # Save checkpoint from model 1 and load it into model 2
    model_1.save("checkpoint.mdlus")
    model_2.load("checkpoint.mdlus")

    # Forward with loaded checkpoint
    with torch.autocast("cuda", enabled=enable_autocast):
        output_2 = model_2.forward(*in_args)

    loaded_checkpoint = compare_output(output_1, output_2, rtol, atol)

    # Restore checkpoint with from_checkpoint, checks initialization of model directly from checkpoint
    model_2 = physicsnemo.Module.from_checkpoint("checkpoint.mdlus").to(model_1.device)
    with torch.autocast("cuda", enabled=enable_autocast):
        output_2 = model_2.forward(*in_args)
    restored_checkpoint = compare_output(output_1, output_2, rtol, atol)

    # Delete checkpoint file (it should exist!)
    Path("checkpoint.mdlus").unlink(missing_ok=False)
    return loaded_checkpoint and restored_checkpoint
