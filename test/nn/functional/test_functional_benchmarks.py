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

from benchmarks.physicsnemo.nn.functional._spec_utils import (
    PHASE_ORDER,
    build_benchmark_plan,
)
from benchmarks.physicsnemo.nn.functional.plot_functional_benchmarks import (
    BenchmarkSpecData,
    _build_fallback_params,
    _collect_grouped_data,
)
from physicsnemo.core.function_spec import FunctionSpec


class FirstPlottableSpec(FunctionSpec):
    @FunctionSpec.register(name="impl_a", rank=0)
    def impl_a():
        return None

    @FunctionSpec.register(name="impl_b", rank=1)
    def impl_b():
        return None

    @classmethod
    def make_inputs_forward(cls, device="cpu"):
        yield "first_forward_0", (), {}
        yield "first_forward_1", (), {}

    @classmethod
    def make_inputs_backward(cls, device="cpu"):
        yield "first_backward_0", (), {}


class SingleImplementationSpec(FunctionSpec):
    @FunctionSpec.register(name="impl_a", rank=0)
    def impl_a():
        return None

    @classmethod
    def make_inputs_forward(cls, device="cpu"):
        yield "single_forward_0", (), {}

    @classmethod
    def make_inputs_backward(cls, device="cpu"):
        yield "single_backward_0", (), {}


class SecondPlottableSpec(FunctionSpec):
    @FunctionSpec.register(name="impl_a", rank=0)
    def impl_a():
        return None

    @FunctionSpec.register(name="impl_b", rank=1)
    def impl_b():
        return None

    @classmethod
    def make_inputs_forward(cls, device="cpu"):
        yield "second_forward_0", (), {}

    @classmethod
    def make_inputs_backward(cls, device="cpu"):
        yield "second_backward_0", (), {}
        yield "second_backward_1", (), {}


def test_fallback_params_match_benchmark_plan_order():
    specs = (FirstPlottableSpec, SingleImplementationSpec, SecondPlottableSpec)

    runner_keys, _ = build_benchmark_plan(
        device="cpu",
        phases=PHASE_ORDER,
        selected_specs=specs,
    )
    fallback_keys = _build_fallback_params(
        device="cpu",
        phases=PHASE_ORDER,
        selected_specs=specs,
    )

    assert fallback_keys == runner_keys
    assert fallback_keys == [
        ("forward", "FirstPlottableSpec", "impl_a", 0),
        ("forward", "FirstPlottableSpec", "impl_a", 1),
        ("forward", "FirstPlottableSpec", "impl_b", 0),
        ("forward", "FirstPlottableSpec", "impl_b", 1),
        ("backward", "FirstPlottableSpec", "impl_a", 0),
        ("backward", "FirstPlottableSpec", "impl_b", 0),
        ("forward", "SingleImplementationSpec", "impl_a", 0),
        ("backward", "SingleImplementationSpec", "impl_a", 0),
        ("forward", "SecondPlottableSpec", "impl_a", 0),
        ("forward", "SecondPlottableSpec", "impl_b", 0),
        ("backward", "SecondPlottableSpec", "impl_a", 0),
        ("backward", "SecondPlottableSpec", "impl_a", 1),
        ("backward", "SecondPlottableSpec", "impl_b", 0),
        ("backward", "SecondPlottableSpec", "impl_b", 1),
    ]


def test_unlabeled_fallback_values_map_to_runner_keys():
    specs = (FirstPlottableSpec, SingleImplementationSpec, SecondPlottableSpec)
    runner_keys, _ = build_benchmark_plan(
        device="cpu",
        phases=PHASE_ORDER,
        selected_specs=specs,
    )
    fallback_keys = _build_fallback_params(
        device="cpu",
        phases=PHASE_ORDER,
        selected_specs=specs,
    )
    values = [float(index) for index in range(len(runner_keys))]
    labels = [str(key) for key in fallback_keys]
    spec_data = {
        "FirstPlottableSpec": BenchmarkSpecData(
            slug="test/first",
            implementations=("impl_a", "impl_b"),
            labels_by_phase={
                "forward": ["first_forward_0", "first_forward_1"],
                "backward": ["first_backward_0"],
            },
        ),
        "SecondPlottableSpec": BenchmarkSpecData(
            slug="test/second",
            implementations=("impl_a", "impl_b"),
            labels_by_phase={
                "forward": ["second_forward_0"],
                "backward": ["second_backward_0", "second_backward_1"],
            },
        ),
    }

    grouped = _collect_grouped_data(
        values=values,
        labels=labels,
        spec_data=spec_data,
    )
    expected_key = ("backward", "SecondPlottableSpec", "impl_b", 1)
    expected_value = float(runner_keys.index(expected_key))

    assert (
        grouped["backward"]["SecondPlottableSpec"]["second_backward_1"]["impl_b"]
        == expected_value
    )
    assert "SingleImplementationSpec" not in grouped["forward"]
    assert "SingleImplementationSpec" not in grouped["backward"]
