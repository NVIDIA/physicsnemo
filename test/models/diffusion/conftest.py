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

import random

import pytest
import torch
import torch._dynamo

from physicsnemo.core.version_check import check_version_spec

_APEX_AVAILABLE = check_version_spec("apex", hard_fail=False)

_GLOBAL_SEED = 42


def _nop_backend(gm, _inputs):
    def forward(*args, **kwargs):
        return gm.forward(*args, **kwargs)

    return forward


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Reset torch._dynamo state between tests to avoid cross-test recompile errors."""
    torch._dynamo.reset()
    torch._dynamo.config.error_on_recompile = False
    yield
    torch._dynamo.reset()
    torch._dynamo.config.error_on_recompile = False


@pytest.fixture
def nop_compile(monkeypatch):
    """Redirect torch.compile to a no-op backend for fast compile-shape tests."""
    original = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, *args, backend=_nop_backend, **kwargs: original(
            fn, *args, backend=backend, **kwargs
        ),
    )


@pytest.fixture
def deterministic_settings():
    """Set deterministic settings for reproducibility, then restore old state."""
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_random_state = random.getstate()

    try:
        random.seed(_GLOBAL_SEED)
        torch.manual_seed(_GLOBAL_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_GLOBAL_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cudnn.deterministic = old_cudnn_deterministic
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        random.setstate(old_random_state)


@pytest.fixture
def apex_device(request, device):
    """
    Fixture that validates apex availability when use_apex_gn=True is used.

    This fixture automatically skips tests when:
    - use_apex_gn=True and apex is not installed
    - use_apex_gn=True and device is "cpu"

    Usage
    -----
    Simply include this fixture in your test function signature alongside
    device and use_apex_gn parameters:

    .. code-block:: python

        @pytest.mark.parametrize("use_apex_gn", [False, True])
        def test_my_model(apex_device, use_apex_gn):
            # apex_device is the validated device
            model = MyModel(use_apex_gn=use_apex_gn).to(apex_device)
            # Test code here

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest request object to access test parameters.
    device : str
        The device fixture (e.g., "cpu", "cuda:0").

    Returns
    -------
    str
        The validated device string.

    Raises
    ------
    pytest.skip
        If apex is required but unavailable, or if device is CPU with apex enabled.
    """
    # Get use_apex_gn from test parameters if it exists
    use_apex_gn = False
    if hasattr(request, "param"):
        # Fixture was parametrized
        use_apex_gn = request.param
    else:
        # Check if use_apex_gn is in the test's parameters
        for param_name in request.fixturenames:
            if param_name == "use_apex_gn":
                try:
                    use_apex_gn = request.getfixturevalue("use_apex_gn")
                    break
                except (pytest.FixtureLookupError, AttributeError):
                    pass

    # Validate apex availability and device compatibility
    if use_apex_gn:
        if not _APEX_AVAILABLE:
            pytest.skip("apex>=0.9.10.dev0 is not installed")
        if device == "cpu":
            pytest.skip("apex group norm not supported on CPU")

    return device
