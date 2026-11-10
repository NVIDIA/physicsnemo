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


def test_deprecated_base_aliases_and_silent_module_imports():
    """The general metrics alias resolves without warning on module import."""
    import importlib
    import warnings

    from physicsnemo.core.warnings import LegacyFeatureWarning
    from physicsnemo.metrics.general import ensemble_metrics as em

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", LegacyFeatureWarning)
        diffusion = importlib.import_module("physicsnemo.diffusion.metrics")
        importlib.reload(diffusion)
    assert not any("EnsembleMetrics" in str(w.message) for w in recorded)
    with pytest.warns(LegacyFeatureWarning):
        alias = em.EnsembleMetrics
    assert alias is em.StreamingStatistic

    class LegacySubclass(alias):
        pass

    assert isinstance(LegacySubclass((3,)), em.StreamingStatistic)


@pytest.mark.parametrize(
    "module_name,other_exports",
    [
        (
            "physicsnemo.metrics.general.ensemble_metrics",
            ("Mean", "Variance"),
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

    if module_name == "physicsnemo.metrics.general.ensemble_metrics":
        assert not {
            "ABC",
            "DistributedManager",
            "LegacyFeatureWarning",
            "Tensor",
            "torch",
            "warnings",
        }.intersection(namespace)
