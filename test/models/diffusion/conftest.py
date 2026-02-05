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

import pytest

from physicsnemo.core.version_check import check_version_spec

_APEX_AVAILABLE = check_version_spec("apex", hard_fail=False)


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
