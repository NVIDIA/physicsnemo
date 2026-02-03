# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import warnings

import pytest
import torch

import physicsnemo.core.function_spec as function_spec
from physicsnemo.core.function_spec import FunctionSpec, Implementation


def test_implementation_forwards_call():
    # Test that the implementation forwards the call to the function
    # The __call__ method will probably never be used like this,
    # but leaving the test here for now.
    def impl(x, y, scale=1):
        return (x + y) * scale

    implementation = Implementation(
        name="impl",
        func=impl,
        required_imports=(),
        rank=0,
        baseline=False,
    )
    assert implementation(2, 3, scale=4) == 20


def test_register_and_dispatch_by_rank():
    # Test that default dispatch selects the lowest-rank implementation
    class RankSpec(FunctionSpec):
        @FunctionSpec.register(name="slow", rank=1)
        def slow(x):
            return f"slow-{x}"

        @FunctionSpec.register(name="fast", rank=0, baseline=True)
        def fast(x):
            return f"fast-{x}"

    assert RankSpec.implementations() == ("fast", "slow")
    assert RankSpec.available_implementations() == ("fast", "slow")
    assert RankSpec.dispatch("value") == "fast-value"
    assert RankSpec().dispatch("value") == "fast-value"
    assert RankSpec()("value") == "fast-value"


def test_register_with_staticmethod():
    # Test if staticmethod still works even though
    # this will probably never be used like this.

    class StaticSpec(FunctionSpec):
        @FunctionSpec.register(name="static_after_register", rank=0)
        @staticmethod
        def static_after_register_impl(x):
            return x + 1

        @staticmethod
        @FunctionSpec.register(name="static_before_register", rank=1)
        def static_before_register_impl(x):
            return x + 2

        @FunctionSpec.register(name="static_never_register", rank=2)
        def static_never_register_impl(x):
            return x + 3

    assert StaticSpec.dispatch(1, implementation="static_after_register") == 2
    assert StaticSpec.dispatch(1, implementation="static_before_register") == 3
    assert StaticSpec.dispatch(1, implementation="static_never_register") == 4
    assert StaticSpec.static_after_register_impl(1) == 2
    assert StaticSpec.static_before_register_impl(1) == 3
    assert StaticSpec.static_never_register_impl(1) == 4


def test_make_function_wrapper():
    # Check that make_function keeps all the
    # function attributes like name, qualname, module, and docstring.

    class WrapperSpec(FunctionSpec):
        """WrapperSpec docstring."""

        @FunctionSpec.register(name="impl", rank=0)
        def impl(x):
            return x * 2

    wrapper = WrapperSpec.make_function("wrapper_spec")
    assert wrapper.__name__ == "wrapper_spec"
    assert wrapper.__qualname__ == "wrapper_spec"
    assert wrapper.__module__ == WrapperSpec.__module__
    assert wrapper.__doc__ == WrapperSpec.__doc__
    assert wrapper(3) == 6


def test_dispatch_explicit_implementation():
    # Test that dispatching with an explicit implementation name
    # selects the corresponding implementation.

    class ExplicitSpec(FunctionSpec):
        @FunctionSpec.register(name="a", rank=1)
        def impl_a(x):
            return f"a-{x}"

        @FunctionSpec.register(name="b", rank=0)
        def impl_b(x):
            return f"b-{x}"

    assert ExplicitSpec.dispatch("x", implementation="a") == "a-x"
    assert ExplicitSpec.dispatch("x", implementation="b") == "b-x"

    with pytest.raises(KeyError):
        ExplicitSpec.dispatch("x", implementation="missing")


def test_available_implementations_filters_missing_imports():
    # NOTE: This test could probably be a bit more comprehensive.
    # Leaving this as a TODO for now.

    class ImportSpec(FunctionSpec):
        @FunctionSpec.register(name="present", required_imports=("math",), rank=0)
        def present(x):
            return x

        @FunctionSpec.register(
            name="missing", required_imports=("not_a_real_module",), rank=1
        )
        def missing(x):
            return x

    assert ImportSpec.available_implementations() == ("present",)
    assert ImportSpec.dispatch(3) == 3

    with pytest.raises(ImportError):
        ImportSpec.dispatch(3, implementation="missing")


def test_duplicate_rank_raises():
    # Test that duplicate rank raises an error

    with pytest.raises(ValueError):

        class DuplicateRank(FunctionSpec):
            @FunctionSpec.register(name="a", rank=0)
            def impl_a(x):
                return x

            @FunctionSpec.register(name="b", rank=0)
            def impl_b(x):
                return x


def test_baseline_unique_raises():
    # Test that duplicate baseline raises an error

    with pytest.raises(ValueError):

        class DuplicateBaseline(FunctionSpec):
            @FunctionSpec.register(name="a", rank=0, baseline=True)
            def impl_a(x):
                return x

            @FunctionSpec.register(name="b", rank=1, baseline=True)
            def impl_b(x):
                return x


def test_register_outside_class_body_raises():
    # Test that registering an implementation outside the class body raises an error
    # This is unlikely to happen, but good to leave it
    # as a warning for now.

    def impl(x):
        return x

    with pytest.raises(ValueError):
        impl.__qualname__ = "impl"
        FunctionSpec.register(name="bad")(impl)


def test_dispatch_no_implementations_raises():
    class EmptySpec(FunctionSpec):
        pass

    with pytest.raises(ImportError):
        EmptySpec.dispatch(1)


def test_missing_imports_handling():
    class UnavailableSpec(FunctionSpec):
        @FunctionSpec.register(
            name="missing", required_imports=("not_a_real_module",), rank=0
        )
        def missing(x):
            return x

    with pytest.raises(ImportError):
        UnavailableSpec.dispatch(1)

    assert not FunctionSpec._check_imports(("not_a_real_module",))


def test_make_inputs_and_compare_not_implemented():
    # This test is really just for code coverage.

    with pytest.raises(NotImplementedError):
        FunctionSpec.make_inputs(device="cpu")
    with pytest.raises(NotImplementedError):
        FunctionSpec.compare(output=None, reference=None)


def test_duplicate_name_raises():
    with pytest.raises(ValueError):

        class DuplicateName(FunctionSpec):
            @FunctionSpec.register(name="dup", rank=0)
            def impl_a(x):
                return x

            @FunctionSpec.register(name="dup", rank=1)
            def impl_b(x):
                return x


def test_fallback_warning_once():
    class WarningSpec(FunctionSpec):
        @FunctionSpec.register(
            name="preferred", required_imports=("not_a_real_module",), rank=0
        )
        def preferred(x):
            return x + 1

        @FunctionSpec.register(name="fallback", rank=1)
        def fallback(x):
            return x + 2

    key = WarningSpec._class_key()
    FunctionSpec._fallback_warned.discard(key)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        assert WarningSpec.dispatch(1) == 3
        assert len(recorded) == 1

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        assert WarningSpec.dispatch(1) == 3
        assert len(recorded) == 0


def test_warp_launch_context(monkeypatch, device):
    # Dummy Warp is to avoid actually importing Warp.
    class DummyWarp:
        def stream_from_torch(self, stream):
            return f"stream:{stream}"

    monkeypatch.setattr(
        function_spec.importlib, "import_module", lambda name: DummyWarp()
    )

    if "cuda" in device:
        monkeypatch.setattr(
            torch.cuda,
            "current_stream",
            lambda device=None: "torch-stream",
        )

        class DummyTensor:
            device = torch.device("cuda")

        tensor = DummyTensor()
        expected_device = None
        expected_stream = "stream:torch-stream"
    else:
        tensor = torch.tensor([1.0])
        expected_device = "cpu"
        expected_stream = None

    actual_device, stream = FunctionSpec.warp_launch_context(tensor)
    assert actual_device == expected_device
    assert stream == expected_stream


def test_warp_launch_context_missing_warp(monkeypatch):
    # Also just for code coverage.
    def _raise(name):
        raise ImportError("missing warp")

    monkeypatch.setattr(function_spec.importlib, "import_module", _raise)
    with pytest.raises(ImportError):
        FunctionSpec.warp_launch_context(torch.tensor([1.0]))
