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


def test_base_subclasses_keep_their_own_count():
    """New opt-in methods must not change the API of unrelated statistics."""
    from physicsnemo.metrics.general import ensemble_metrics as em
    from physicsnemo.metrics.general.histogram import Histogram

    # An unrelated subclass can continue managing its own count.
    class CustomStatistic(em.StreamingStatistic):
        def __init__(self):
            super().__init__((3,))
            self.count = 0

    assert CustomStatistic().count == 0
    for name in ("count", "combine", "reset"):
        assert not hasattr(Histogram, name)


def test_deprecated_base_aliases_and_silent_module_imports():
    """Both old exports resolve to the same base without warning on module import."""
    import importlib
    import warnings

    from physicsnemo.core.warnings import LegacyFeatureWarning
    from physicsnemo.metrics.general import ensemble_metrics as em

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", LegacyFeatureWarning)
        diffusion = importlib.import_module("physicsnemo.diffusion.metrics")
        importlib.reload(diffusion)
    assert not any("EnsembleMetrics" in str(w.message) for w in recorded)
    for module in (em, diffusion):
        with pytest.warns(LegacyFeatureWarning):
            alias = module.EnsembleMetrics
        assert alias is em.StreamingStatistic

        class LegacySubclass(alias):
            pass

        assert isinstance(LegacySubclass((3,)), em.StreamingStatistic)


@pytest.mark.parametrize(
    "module_name,other_exports",
    [
        (
            "physicsnemo.metrics.general.ensemble_metrics",
            ("Mean", "Variance", "Tensor"),
        ),
        (
            "physicsnemo.diffusion.metrics",
            ("Mean", "Variance", "Histogram", "crps", "EDMLoss", "MSEDSMLoss"),
        ),
    ],
)
def test_deprecated_base_wildcard_import(module_name, other_exports):
    """Legacy wildcard imports can still define subclasses of EnsembleMetrics."""
    import importlib

    from physicsnemo.core.warnings import LegacyFeatureWarning
    from physicsnemo.metrics.general.ensemble_metrics import StreamingStatistic

    module = importlib.import_module(module_name)
    namespace = {}
    with pytest.warns(LegacyFeatureWarning, match="EnsembleMetrics is deprecated"):
        exec(  # noqa: S102 - exercise wildcard syntax with fixed test module names
            f"from {module_name} import *\n"
            "class LegacyStatistic(EnsembleMetrics):\n"
            "    pass\n",
            namespace,
        )
    assert namespace["EnsembleMetrics"] is StreamingStatistic
    assert namespace["StreamingStatistic"] is StreamingStatistic
    assert isinstance(namespace["LegacyStatistic"]((3,)), StreamingStatistic)
    for name in other_exports:
        assert namespace[name] is getattr(module, name)
