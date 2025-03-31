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

import io
import logging

import torch
import torch.nn as nn

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from typing import Tuple, Union

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


def check_ort_install(func):
    """Decorator to check if ONNX runtime is installed"""

    def _wrapper_ort_install(*args, **kwargs):
        if ort is None:
            raise ModuleNotFoundError(
                "ONNXRuntime is not installed. 'pip install \
                onnxruntime onnxruntime_gpu'"
            )
        func(*args, **kwargs)
        return func(*args, **kwargs)

    return _wrapper_ort_install


def export_to_onnx_stream(
    model: nn.Module,
    invars: Union[Tensor, Tuple[Tensor, ...]],
    verbose: bool = False,
) -> bytes:
    """Exports PyTorch model to byte stream instead of a file

    Parameters
    ----------
    model : nn.Module
        PyTorch model to export
    invars : Union[Tensor, Tuple[Tensor,...]]
        Input tensor(s)
    verbose : bool, optional
        Print out a human-readable representation of the model, by default False

    Returns
    -------
    bytes
        ONNX model byte stream

    Note
    ----
    Exporting a ONNX model while training when using CUDA graphs will likely break things.
    Because model must be copied to the CPU and back for export.

    Note
    ----
    ONNX exporting can take a longer time when using custom ONNX functions.
    """
    # Move inputs to CPU for ONNX export
    if isinstance(invars, Tensor):
        invars = (invars.detach().cpu(),)
    else:
        invars = tuple([invar.detach().cpu() for invar in invars])
    # Use model's device if provided (PhysicsNeMo modules have this)
    if hasattr(model, "device"):
        model_device = model.device
    elif len(list(model.parameters())) > 0:
        model_device = next(model.parameters()).device
    else:
        model_device = "cpu"

    with io.BytesIO() as onnx_model:
        # Export to ONNX.
        torch.onnx.export(
            model.cpu(),
            invars,
            onnx_model,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            opset_version=15,
            verbose=verbose,
        )
        # Move model back to original device
        model.to(model_device)
        return onnx_model.getvalue()


@check_ort_install
def get_ort_session(
    model: Union[bytes, str],
    device: torch.device = "cuda",
):
    """Create a ORT session for performing inference of an onnx model

    Parameters
    ----------
    model : Union[bytes, str]
        ONNX model byte string or file name/path
    device : torch.device, optional
        Device to run ORT, by default "cuda"

    Returns
    -------
    ort.InferenceSession
        ONNX runtime session
    """
    providers = ["CPUExecutionProvider"]
    if "cuda" in str(device):
        providers = ["CUDAExecutionProvider"] + providers

    # Must run on GPU as Rfft is currently implemented only for GPU.
    ort_sess = ort.InferenceSession(model, providers=providers)
    return ort_sess


@check_ort_install
def run_onnx_inference(
    model: Union[bytes, str],
    invars: Union[Tensor, Tuple[Tensor, ...]],
    device: torch.device = "cuda",
) -> Tuple[Tensor]:
    """Runs ONNX model in ORT session

    Parameters
    ----------
    model : Union[bytes, str]
        ONNX model byte string or file name/path
    invars : Union[Tensor, Tuple[Tensor,...]]
        Input tensors
    device : torch.device, optional
        Device to run ORT, by default "cuda"

    Returns
    -------
    Tuple[Tensor]
        Tuple of output tensors on CPU
    """
    ort_sess = get_ort_session(model, device)
    # fmt: off
    if isinstance(invars, Tensor):
        invars = (invars,)
    ort_inputs = {inp.name: v.detach().cpu().numpy()
                  for inp, v in zip(ort_sess.get_inputs(), invars)}
    # fmt: on
    ort_outputs = ort_sess.run(None, ort_inputs)
    # Convert to tensors
    outputs = tuple([torch.Tensor(v) for v in ort_outputs])
    return outputs
