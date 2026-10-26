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

"""Accumulate uncertainty in energy-derived forces without retaining their graphs.

Run: python examples/minimal/uq/force_statistics.py --device cpu
The analytic energy members make the derivative and statistics independently
checkable. Applications supply their own energy models and training workflow.
"""

import argparse

import torch
from torch import nn

from physicsnemo.metrics.general.ensemble_metrics import Variance


class HarmonicEnergy(nn.Module):
    """Small energy model with one trainable stiffness, used only in this example."""

    def __init__(self, stiffness, device):
        super().__init__()
        self.stiffness = nn.Parameter(torch.tensor(stiffness, device=device))

    def forward(self, positions):
        """Return a scalar energy for the complete configuration."""
        return 0.5 * self.stiffness * positions.square().sum()


def main():
    """Compute derivatives first, then accumulate detached field moments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    positions = torch.arange(12, device=args.device, dtype=torch.float32).reshape(4, 3)
    positions = (positions / 10).requires_grad_()
    stiffnesses = torch.tensor([1.0, 2.0, 4.0], device=args.device)
    stats = Variance(mode="tensor", correction=0)
    for stiffness in stiffnesses.tolist():
        model = HarmonicEnergy(stiffness, args.device)
        # Do not wrap energy/force evaluation in torch.no_grad(). The application
        # chooses create_graph=True only if it also needs gradients of the forces.
        energy = model(positions)
        forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
        stats.update(forces, dim=None)
        assert not stats.mean.requires_grad
        assert not stats.finalize().requires_grad
        # Accumulating a detached copy does not detach or consume the caller's
        # force graph. A force-based training loss can still use that graph.
        force_loss = forces.square().mean()
        force_loss.backward()
        torch.testing.assert_close(
            model.stiffness.grad, 2 * stiffness * positions.detach().square().mean()
        )
        positions.grad = None
        del energy, forces, force_loss, model

    torch.testing.assert_close(stats.mean, -stiffnesses.mean() * positions.detach())
    torch.testing.assert_close(
        stats.finalize(),
        stiffnesses.var(correction=0) * positions.detach().square(),
    )
    assert stats.count == len(stiffnesses)
    print("Force derivatives, detached streaming moments, and caller gradients PASS")


if __name__ == "__main__":
    main()
