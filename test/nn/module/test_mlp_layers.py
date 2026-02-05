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

import torch

from physicsnemo.nn import Mlp
from test.common import (
    validate_forward_accuracy,
)


def test_mlp_forward_accuracy(device):
    torch.manual_seed(7)
    target_device = torch.device(device)

    model = Mlp(in_features=10, hidden_features=20, out_features=5).to(target_device)
    input_tensor = torch.randn(1, 10).to(
        target_device
    )  # Assuming a batch size of 1 for simplicity
    model(input_tensor)

    # Relative to test/
    file_name = "nn/module/data/mlp_output.pth"

    # Tack this on for the test, since model is not a physicsnemo Module:
    model.device = target_device

    assert validate_forward_accuracy(
        model,
        (input_tensor,),
        file_name=file_name,
        atol=1e-3,
    )


def test_mlp_activation_and_dropout():
    model = Mlp(in_features=10, hidden_features=20, out_features=5, drop=0.5)
    input_tensor = torch.randn(2, 10)  # Batch size of 2

    output_tensor = model(input_tensor)

    assert output_tensor.shape == torch.Size([2, 5])


def test_mlp_different_activation():
    model = Mlp(
        in_features=10, hidden_features=20, out_features=7, act_layer=torch.nn.ReLU
    )
    input_tensor = torch.randn(3, 10)  # Batch size of 3

    output_tensor = model(input_tensor)
    assert output_tensor.shape == torch.Size([3, 7])


def test_multiple_hidden_layers():
    model = Mlp(in_features=10, hidden_features=[20, 30], out_features=5)
    input_tensor = torch.randn(4, 10)  # Batch size of 4

    output_tensor = model(input_tensor)
    assert output_tensor.shape == torch.Size([4, 5])


def test_mlp_string_activation():
    """Test that string activation names work correctly."""
    model = Mlp(in_features=10, hidden_features=20, out_features=5, act_layer="gelu")
    input_tensor = torch.randn(2, 10)

    output_tensor = model(input_tensor)
    assert output_tensor.shape == torch.Size([2, 5])


def test_mlp_use_te_false():
    """Test that use_te=False works (default behavior)."""
    model = Mlp(in_features=10, hidden_features=20, out_features=5, use_te=False)
    input_tensor = torch.randn(2, 10)

    output_tensor = model(input_tensor)
    assert output_tensor.shape == torch.Size([2, 5])

    # Verify that standard nn.Linear is used
    assert isinstance(model.layers[0], torch.nn.Linear)


def test_mlp_use_te_unavailable():
    """Test that use_te=True raises error when TE is not available."""
    import importlib.util

    te_available = importlib.util.find_spec("transformer_engine") is not None

    if te_available:
        # If TE is available, this should work
        model = Mlp(in_features=10, hidden_features=20, out_features=5, use_te=True)
        input_tensor = torch.randn(2, 10)
        output_tensor = model(input_tensor)
        assert output_tensor.shape == torch.Size([2, 5])
    else:
        # If TE is not available, this should raise RuntimeError
        import pytest

        with pytest.raises(RuntimeError, match="Transformer Engine is not available"):
            Mlp(in_features=10, hidden_features=20, out_features=5, use_te=True)


def test_mlp_gradient_flow():
    """Test that gradients flow through the MLP."""
    model = Mlp(in_features=10, hidden_features=20, out_features=5)
    input_tensor = torch.randn(2, 10, requires_grad=True)

    output_tensor = model(input_tensor)
    loss = output_tensor.sum()
    loss.backward()

    assert input_tensor.grad is not None
    assert input_tensor.grad.shape == input_tensor.shape


def test_transolver_mlp_checkpoint_compatibility():
    """Test that _TransolverMlp can load legacy checkpoint format."""
    from physicsnemo.models.transolver.transolver import _TransolverMlp

    # Create a new model
    model = _TransolverMlp(
        in_features=10, hidden_features=20, out_features=5, act_layer="gelu"
    )

    # Simulate an old-style checkpoint with legacy key names
    old_state_dict = {
        "linear_pre.weight": torch.randn(20, 10),
        "linear_pre.bias": torch.randn(20),
        "linear_post.weight": torch.randn(5, 20),
        "linear_post.bias": torch.randn(5),
    }

    # This should work without errors - keys are remapped automatically
    model.load_state_dict(old_state_dict)

    # Verify the weights were loaded correctly
    assert model.layers[0].weight.shape == (20, 10)
    assert model.layers[0].bias.shape == (20,)
    assert model.layers[2].weight.shape == (5, 20)
    assert model.layers[2].bias.shape == (5,)

    # Verify forward pass works
    input_tensor = torch.randn(2, 10)
    output = model(input_tensor)
    assert output.shape == (2, 5)


def test_transolver_mlp_new_checkpoint_format():
    """Test that _TransolverMlp can load new checkpoint format."""
    from physicsnemo.models.transolver.transolver import _TransolverMlp

    # Create a new model
    model = _TransolverMlp(
        in_features=10, hidden_features=20, out_features=5, act_layer="gelu"
    )

    # Get the new-style state dict
    new_state_dict = model.state_dict()

    # Create a fresh model and load the new-style checkpoint
    model2 = _TransolverMlp(
        in_features=10, hidden_features=20, out_features=5, act_layer="gelu"
    )
    model2.load_state_dict(new_state_dict)

    # Verify forward pass produces same results
    input_tensor = torch.randn(2, 10)
    torch.manual_seed(42)
    output1 = model(input_tensor)
    torch.manual_seed(42)
    output2 = model2(input_tensor)

    assert torch.allclose(output1, output2)
