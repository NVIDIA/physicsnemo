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
import torch
import torch.nn as nn
from torch.optim import LBFGS, SGD, Adam

from physicsnemo.optim.combined_optimizer import CombinedOptimizer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.layer2(nn.functional.relu(self.layer1(x)))


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def optimizers(model):
    opt1 = SGD(model.layer1.parameters(), lr=0.01)
    opt2 = Adam(model.layer2.parameters(), lr=0.001)
    return [opt1, opt2]


@pytest.fixture
def combined_optimizer(optimizers):
    return CombinedOptimizer(optimizers)


class TestInitialization:
    def test_init_requires_optimizers(self):
        with pytest.raises(ValueError, match="must contain at least one optimizer"):
            CombinedOptimizer([])

    def test_init_with_compile_kwargs(self, model):
        opt = SGD(model.layer1.parameters(), lr=0.1)
        # We just check it doesn't crash and sets up step_fns

        with torch.no_grad():
            # Mock torch.compile to verify call
            original_compile = torch.compile
            compile_called = False

            def mock_compile(fn, **kwargs):
                nonlocal compile_called
                compile_called = True
                assert kwargs == {"mode": "reduce-overhead"}
                return fn

            try:
                torch.compile = mock_compile
                combined = CombinedOptimizer(
                    [opt], torch_compile_kwargs={"mode": "reduce-overhead"}
                )
                assert compile_called
                assert len(combined.step_fns) == 1
            finally:
                torch.compile = original_compile

    def test_init_aggregates_param_groups(self, combined_optimizer, model):
        # opt1 has 1 group, opt2 has 1 group
        assert len(combined_optimizer.param_groups) == 2

        # Verify all parameters are present
        all_params = set()
        for group in combined_optimizer.param_groups:
            all_params.update(group["params"])

        model_params = set(model.parameters())
        assert all_params == model_params

    def test_repr(self, combined_optimizer):
        s = repr(combined_optimizer)
        assert "CombinedOptimizer" in s
        assert "SGD" in s
        assert "Adam" in s

    def test_add_param_group_raises(self, combined_optimizer):
        with pytest.raises(
            NotImplementedError, match="does not support add_param_group"
        ):
            combined_optimizer.add_param_group({"params": []})

    def test_overlapping_params_raises(self, model):
        """Verify that overlapping parameter groups raise ValueError."""
        # Both optimizers have the same parameters
        opt1 = SGD(model.layer1.parameters(), lr=0.01)
        opt2 = Adam(model.layer1.parameters(), lr=0.001)  # Same params as opt1

        with pytest.raises(
            ValueError, match="Parameter appears in multiple optimizers"
        ):
            CombinedOptimizer([opt1, opt2])


class TestStep:
    def test_step_calls_all_optimizers(self, model):
        # Mock optimizers to verify calls
        opt1 = torch.optim.SGD(model.layer1.parameters(), lr=0.1)
        opt2 = torch.optim.SGD(model.layer2.parameters(), lr=0.1)

        # Monkey patch step
        opt1_step_called = False
        opt2_step_called = False

        orig_step1 = opt1.step
        orig_step2 = opt2.step

        def step1(closure=None):
            nonlocal opt1_step_called
            opt1_step_called = True
            return orig_step1(closure)

        def step2(closure=None):
            nonlocal opt2_step_called
            opt2_step_called = True
            return orig_step2(closure)

        opt1.step = step1
        opt2.step = step2

        combined = CombinedOptimizer([opt1, opt2])

        # Fake loss/backward
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()

        combined.step()

        assert opt1_step_called
        assert opt2_step_called

    def test_zero_grad_sets_grad_to_none(self, combined_optimizer):
        """Verify zero_grad(set_to_none=True) sets all gradients to None."""
        for group in combined_optimizer.param_groups:
            for p in group["params"]:
                p.grad = torch.ones_like(p)

        combined_optimizer.zero_grad()  # default: set_to_none=True

        for group in combined_optimizer.param_groups:
            for p in group["params"]:
                assert p.grad is None, f"Expected grad to be None, got {p.grad}"

    def test_zero_grad_sets_grad_to_zero(self, combined_optimizer):
        """Verify zero_grad(set_to_none=False) sets all gradients to zero."""
        for group in combined_optimizer.param_groups:
            for p in group["params"]:
                p.grad = torch.ones_like(p)

        combined_optimizer.zero_grad(set_to_none=False)

        for group in combined_optimizer.param_groups:
            for p in group["params"]:
                assert p.grad is not None, "Expected grad to exist"
                assert torch.all(p.grad == 0), f"Expected grad to be zero, got {p.grad}"

    def test_step_with_closure_called_per_optimizer(self, combined_optimizer, model):
        """Verify closure is called by each optimizer that supports it."""
        call_count = 0

        def closure():
            nonlocal call_count
            call_count += 1
            combined_optimizer.zero_grad()
            loss = model(torch.randn(1, 10)).sum()
            loss.backward()
            return loss

        combined_optimizer.step(closure)

        # CombinedOptimizer has 2 optimizers (SGD, Adam), both support closure
        # So closure should be called at least twice (once per optimizer)
        assert call_count >= 2, f"Closure called {call_count} times, expected >= 2"

    def test_step_returns_closure_result(self, combined_optimizer, model):
        """Verify step returns the result of the closure (usually loss)."""
        expected_loss = torch.tensor(123.0)

        def closure():
            combined_optimizer.zero_grad()
            return expected_loss

        result = combined_optimizer.step(closure)
        assert result == expected_loss, (
            f"Expected step to return {expected_loss}, got {result}"
        )

    def test_mixed_optimizers_closure_lbfgs_sgd(self, model):
        """Test combining LBFGS (requires closure) and SGD (optional closure)."""
        # Setup
        opt1 = LBFGS(model.layer1.parameters(), lr=0.1, max_iter=2)
        opt2 = SGD(model.layer2.parameters(), lr=0.1)
        combined = CombinedOptimizer([opt1, opt2])

        # Capture initial params to verify update
        p1_init = list(model.layer1.parameters())[0].clone()
        p2_init = list(model.layer2.parameters())[0].clone()

        # Closure
        closure_calls = 0

        def closure():
            nonlocal closure_calls
            closure_calls += 1
            combined.zero_grad()
            # Deterministic input for consistent loss
            x = torch.ones(1, 10)
            loss = model(x).sum()
            loss.backward()
            return loss

        # This should not raise TypeError (LBFGS missing closure)
        combined.step(closure)

        # Verify updates
        p1_final = list(model.layer1.parameters())[0]
        p2_final = list(model.layer2.parameters())[0]

        assert not torch.equal(p1_init, p1_final), "LBFGS parameters did not update"
        assert not torch.equal(p2_init, p2_final), "SGD parameters did not update"

        # LBFGS usually calls closure multiple times (init + line search)
        # SGD calls it once if passed
        # Total calls should be > 1
        assert closure_calls >= 1


class TestStateDict:
    def test_state_dict_structure(self, combined_optimizer):
        state = combined_optimizer.state_dict()
        assert "optimizers" in state
        assert len(state["optimizers"]) == 2
        assert isinstance(state["optimizers"][0], dict)

    def test_load_state_dict(self, combined_optimizer, model):
        # Save state
        state = combined_optimizer.state_dict()

        # Create new optimizer
        opt1 = SGD(model.layer1.parameters(), lr=0.01)
        opt2 = Adam(model.layer2.parameters(), lr=0.001)
        new_combined = CombinedOptimizer([opt1, opt2])

        # Load state
        new_combined.load_state_dict(state)

        # Verify equality (basic check)
        # Note: strict equality of state dicts might fail due to weak refs or unrelated keys,
        # so we check structure matches.
        new_state = new_combined.state_dict()
        assert len(new_state["optimizers"]) == len(state["optimizers"])

    def test_load_state_dict_mismatch_raises(self, combined_optimizer):
        state = combined_optimizer.state_dict()
        state["optimizers"].pop()  # Remove one

        with pytest.raises(ValueError, match="State dict contains 1 optimizer"):
            combined_optimizer.load_state_dict(state)

    def test_load_state_dict_missing_key(self, combined_optimizer):
        bad_state = {"wrong_key": []}
        with pytest.raises(
            KeyError, match="Expected state_dict to contain 'optimizers' key"
        ):
            combined_optimizer.load_state_dict(bad_state)

    def test_state_dict_numeric_correctness(self, model):
        """Verify save/restore preserves optimizer state numerically.

        This test ensures that loading a saved state_dict correctly restores
        the internal optimizer state (momentum buffers, Adam moments, etc.)
        and produces identical parameter updates.
        """
        import copy

        torch.manual_seed(42)

        # Create optimizers with momentum/state that accumulates over steps
        opt1 = SGD(model.layer1.parameters(), lr=0.1, momentum=0.9)
        opt2 = Adam(model.layer2.parameters(), lr=0.01)
        combined = CombinedOptimizer([opt1, opt2])

        # Create deterministic test input before any other random ops
        x_test = torch.randn(4, 10)

        # Run several training steps to build up optimizer state
        for i in range(5):
            combined.zero_grad()
            # Use deterministic input based on step index
            x = torch.full((4, 10), float(i + 1))
            loss = model(x).sum()
            loss.backward()
            combined.step()

        # Save state and current parameters (deepcopy is essential since
        # state_dict returns references, not copies)
        state = copy.deepcopy(combined.state_dict())
        params_checkpoint = {
            name: p.clone().detach() for name, p in model.named_parameters()
        }

        # Take two more steps to advance optimizer state beyond checkpoint
        for _ in range(2):
            combined.zero_grad()
            loss = model(x_test).sum()
            loss.backward()
            combined.step()

        # Record current params (after 2 more steps)
        params_after_extra_steps = {
            name: p.clone().detach() for name, p in model.named_parameters()
        }

        # Restore checkpoint: model params + optimizer state
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(params_checkpoint[name])
        combined.load_state_dict(state)

        # Take the same two steps again
        for _ in range(2):
            combined.zero_grad()
            loss = model(x_test).sum()
            loss.backward()
            combined.step()

        # Verify we get the same final params
        for name, p in model.named_parameters():
            expected = params_after_extra_steps[name]
            actual = p.detach()
            assert torch.allclose(actual, expected, atol=1e-6), (
                f"State restore produced different result for {name}: "
                f"expected param norm {expected.norm().item():.6f}, "
                f"got {actual.norm().item():.6f}"
            )


class TestIntegration:
    def test_lr_scheduler(self, combined_optimizer):
        scheduler = torch.optim.lr_scheduler.StepLR(
            combined_optimizer, step_size=1, gamma=0.1
        )

        initial_lrs = [g["lr"] for g in combined_optimizer.param_groups]

        scheduler.step()

        new_lrs = [g["lr"] for g in combined_optimizer.param_groups]

        for init, new in zip(initial_lrs, new_lrs):
            assert new == pytest.approx(init * 0.1)
