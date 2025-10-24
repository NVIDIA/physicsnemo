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

import onnx
import pytest
import torch

import physicsnemo

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from pathlib import Path
from typing import Tuple

from physicsnemo.deploy.onnx import export_to_onnx_stream, run_onnx_inference

from .utils import compare_output

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


def check_ort_version():
    if ort is None:
        return pytest.mark.skipif(
            True,
            reason="Proper ONNX runtime is not installed. 'pip install onnxruntime onnxruntime_gpu'",
        )
    elif ort.__version__ != "1.15.1":
        return pytest.mark.skipif(
            True,
            reason="Must install custom ORT 1.15.1. Other versions do not work \
        due to bug in IRFFT: https://github.com/microsoft/onnxruntime/issues/13236",
        )
    else:
        return pytest.mark.skipif(False, reason="")


@torch.no_grad()
def validate_onnx_export(
    model: physicsnemo.Module,
    in_args: Tuple[Tensor] = (),
) -> bool:
    """Check network's ONNX export works

    This just save a ONNX export, loads it back into Python and makes sure its valid.

    Parameters
    ----------
    model_1 : physicsnemo.Module
        PhysicsNeMo model to save checkpoint from
    in_args : Tuple[Tensor], optional
        Input arguments, by default ()

    Returns
    -------
    bool
        Test passed

    Note
    ----
    ONNX must be turned on in the model's meta data for this test to run.
    """
    if not model.meta.onnx_cpu and str(model.device) == "cpu":
        logger.warning("Model not marked as supporting ONNX CPU, skipping")
        return True
    elif not model.meta.onnx_gpu:
        logger.warning("Model not marked as supporting ONNX GPU, skipping")
        return True

    onnx_name = "model.onnx"
    device = model.device
    # Turn on eval mode for model and move it to CPU for export
    model = model.eval().cpu()
    onnx_in_args = tuple([arg.cpu() for arg in in_args])
    torch.onnx.export(
        model.cpu(),
        onnx_in_args,
        onnx_name,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        opset_version=15,
        verbose=False,
    )
    # Load back into python from file
    onnx_model = onnx.load(onnx_name)
    model = model.to(device)
    # Check that the model is well formed
    try:
        onnx.checker.check_model(onnx_model)
        # Delete checkpoint file (it should exist!)
        Path(onnx_name).unlink(missing_ok=False)
        return True
    except onnx.checker.ValidationError as e:
        logger.error("Loaded ONNX model is not well formed: %s" % e)
        # Delete checkpoint file (it should exist!)
        Path(onnx_name).unlink(missing_ok=False)
        return False


@torch.no_grad()
def validate_onnx_runtime(
    model: physicsnemo.Module,
    in_args: Tuple[Tensor, ...] = (),
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> bool:
    """Check network's ONNX export is consistent with PyTorch forward pass using onnxruntime

    This test will check to make sure that ONNX can export a model. It will then execute
    a forward pass of the provide PyTorch model as well as ONNX version using a onnxruntime
    session. It will then check to see if the outputs are the same


    Parameters
    ----------
    model_1 : physicsnemo.Module
        PhysicsNeMo model to save checkpoint from
    in_args : Tuple[Tensor], optional
        Input arguments, by default ()
    rtol : float, optional
        Relative tolerance of error allowed, by default 1e-3
    atol : float, optional
        Absolute tolerance of error allowed, by default 1e-3

    Returns
    -------
    bool
        Test passed

    Note
    ----
    ONNX runtime must be turned on in the model's meta data for this test to run.
    """
    if ort is None:
        logger.warning("ONNX runtime not installed, skipping")
        return True
    if not model.meta.onnx_runtime:
        logger.warning("Model not marked as supporting ONNX runtime, skipping")
        return True
    if not model.meta.onnx_cpu and str(model.device) == "cpu":
        logger.warning("Model not marked as supporting ONNX CPU, skipping")
        return True
    elif not model.meta.onnx_gpu:
        logger.warning("Model not marked as supporting ONNX GPU, skipping")
        return True

    # Now test regular forward pass
    output = model.forward(*in_args)
    if isinstance(output, Tensor):
        output = (output,)

    # Test ONNX forward
    device = model.device
    onnx_model = export_to_onnx_stream(model, in_args)
    output_ort = run_onnx_inference(onnx_model, in_args, device=device)
    output_ort = tuple(output.to(device) for output in output_ort)

    # Model outputs should initially be different
    return compare_output(output, output_ort, rtol, atol)
