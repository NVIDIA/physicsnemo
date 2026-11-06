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

"""The UQ namespaces share GP exports and defer missing-dependency errors."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("gpytorch_available", [False, True])
def test_gp_import_paths(gpytorch_available):
    """Both namespaces expose identical symbols, even without gpytorch installed."""
    if gpytorch_available:
        pytest.importorskip("gpytorch")

    # Use a fresh interpreter so simulated absence cannot affect other GP tests.
    code = textwrap.dedent(
        """
        import sys
        from importlib import metadata
        from unittest.mock import patch

        import pytest

        available = sys.argv[1] == "True"
        original_version = metadata.version

        def version(name):
            if not available and name == "gpytorch":
                raise metadata.PackageNotFoundError(name)
            return original_version(name)

        if not available:
            sys.modules["gpytorch"] = None

        with patch.object(metadata, "version", version):
            import physicsnemo.experimental.uq as uq
            from physicsnemo.experimental.uq import gp

            for name in (
                "VariationalGPHead", "GPPrediction",
                "FieldVariationalGPHead", "FieldVariationalGPPrediction",
            ):
                assert getattr(uq, name) is getattr(gp, name)

            for namespace in (uq, gp):
                for name in ("VariationalGPHead", "FieldVariationalGPHead"):
                    head_type = getattr(namespace, name)
                    if available:
                        head_type(input_dim=4, n_train=8, n_inducing=4)
                    else:
                        with pytest.raises(ImportError, match="pip install gpytorch"):
                            head_type(input_dim=4, n_train=8, n_inducing=4)
        """
    )
    result = subprocess.run(  # noqa: S603 - interpreter and snippet are test constants
        [sys.executable, "-c", code, str(gpytorch_available)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
