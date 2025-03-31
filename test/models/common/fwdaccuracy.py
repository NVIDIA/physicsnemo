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
from typing import Tuple, Union

import torch

import physicsnemo

from .utils import compare_output

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


def save_output(output: Union[Tensor, Tuple[Tensor, ...]], file_name: Path):
    """Saves output of model to file

    Parameters
    ----------
    output : Union[Tensor, Tuple[Tensor, ...]]
        Output from netwrok model
    file_name : Path
        File path

    Raises
    ------
    IOError
        If file path has a parent directory that does not exist
    ValueError
        If model outputs are larger than 10mb
    """
    if not file_name.parent.is_dir():
        raise IOError(
            f"Folder path, {file_name.parent}, for output accuracy data not found"
        )

    # Check size of outputs
    output_size = 0
    for out_tensor in output:
        out_tensor = out_tensor.detach().contiguous().cpu()
        output_size += out_tensor.element_size() * out_tensor.nelement()

    if output_size > 10**7:
        raise ValueError(
            "Outputs are greater than 10mb which is too large for this test"
        )

    output_dict = {i: data.detach().contiguous().cpu() for i, data in enumerate(output)}
    torch.save(output_dict, file_name)


@torch.no_grad()
def validate_forward_accuracy(
    model: physicsnemo.Module,
    in_args: Tuple[Tensor] = (),
    rtol: float = 1e-3,
    atol: float = 1e-3,
    file_name: Union[str, None] = None,
) -> bool:
    """Validates the accuracy of a model's forward pass with a reference output

    The provided model should likely be initialized using a set seed, otherwise this will
    likely fail. If the reference output tensor file cannot be found a new one will be
    create but this test will error. Run the test again to pass it, be sure to commit the
    output tensor file you have contributed a new model.

    Parameters
    ----------
    model : physicsnemo.Module
        PhysicsNeMo module
    in_args : Tuple[Tensor], optional
        Input arguments, by default ()
    rtol : float, optional
        Relative tolerance of error allowed, by default 1e-3
    atol : float, optional
        Absolute tolerance of error allowed, by default 1e-3
    file_name : Union[str, None], optional
        Override the default file name of the stored target output, by default None

    Returns
    -------
    bool
        Test passed

    Raises
    ------
    IOError
        Target output tensor file for this model was not found
    """
    # Perform a foward pass of the model
    output = model.forward(*in_args)
    # Always use tuples for this comparison / saving
    if isinstance(output, Tensor):
        output = (output,)

    # File name / path
    # Output files should live in test/models/data
    if file_name is None:
        file_name = model.meta.name + "_output.pth"
    file_name = (
        Path(__file__).parents[1].resolve() / Path("data") / Path(file_name.lower())
    )
    # If file does not exist, we will create it then error
    # Model should then reproduce it on next pytest run
    if not file_name.exists():
        save_output(output, file_name)
        raise IOError(
            f"Output check file {str(file_name)} wasn't found so one was created. Please re-run the test."
        )
    # Load tensor dictionary and check
    else:
        tensor_dict = torch.load(str(file_name))
        output_target = tuple(
            [value.to(model.device) for value in tensor_dict.values()]
        )

        return compare_output(output, output_target, rtol, atol)
