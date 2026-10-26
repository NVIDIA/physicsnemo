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

"""Compose existing ensemble, Concrete Dropout, and GP outputs with shared moments.

Run from the repo root: python examples/minimal/uq/streaming_predictions.py
Add --conformal to demonstrate pointwise conformal calibration. These small,
untrained models demonstrate interfaces, not uncertainty quality or coverage.
"""

import argparse
from contextlib import contextmanager

import torch
from torch import nn

from physicsnemo.experimental.uq.gp import VariationalGPHead
from physicsnemo.metrics.general.ensemble_metrics import Mean, Variance
from physicsnemo.nn.module.concrete_dropout import ConcreteDropout


@contextmanager
def stochastic_concrete_dropout(model):
    """Enable only Concrete Dropout layers and restore their previous modes."""
    layers = [layer for layer in model.modules() if isinstance(layer, ConcreteDropout)]
    previous = [layer.training for layer in layers]
    try:
        for layer in layers:
            layer.train()
        yield
    finally:
        for layer, training in zip(layers, previous):
            layer.train(training)


def summarize_predictions(predictions):
    """Stream (mean, optional variance) pairs without storing member outputs."""
    between = Variance(mode="tensor", correction=0)
    within = Mean(mode="tensor")
    for mean, variance in predictions:
        between.update(mean, dim=None)
        within.update(
            torch.zeros_like(mean) if variance is None else variance, dim=None
        )
    return between.mean, between.finalize() + within.finalize()


@torch.no_grad()
def main():
    """Exercise each prediction producer through the same accumulator lifecycle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conformal", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(12)
    features = torch.randn(12, 4, device=args.device)

    members = [nn.Linear(4, 1).to(args.device).eval() for _ in range(3)]
    mean, variance = summarize_predictions((model(features), None) for model in members)
    print("ensemble", mean.shape, variance.mean().item())

    dropout_model = (
        nn.Sequential(ConcreteDropout(4), nn.Linear(4, 1)).to(args.device).eval()
    )
    with stochastic_concrete_dropout(dropout_model):
        mean, variance = summarize_predictions(
            (dropout_model(features), None) for _ in range(16)
        )
    print("concrete dropout", mean.shape, variance.mean().item())

    heads = [
        VariationalGPHead(input_dim=4, n_inducing=8, n_train=100).to(args.device)
        for _ in range(3)
    ]

    def gp_predictions(inputs):
        for head in heads:
            prediction = head.predict(inputs)
            yield prediction.mean, prediction.variance

    mean, variance = summarize_predictions(gp_predictions(features))
    print("GP mixture", mean.shape, variance.mean().item())
    single = heads[0].predict(features)
    single_mean, single_var = summarize_predictions([(single.mean, single.variance)])
    torch.testing.assert_close(single_mean, single.mean)
    torch.testing.assert_close(single_var, single.variance)

    if args.conformal:
        from physicsnemo.experimental.uq.conformal import (
            NormalizedErrorScore,
            PointwiseCalibrator,
        )

        calibrator = PointwiseCalibrator(NormalizedErrorScore(), alpha=0.1)
        for _ in range(20):
            calibration_features = torch.randn_like(features)
            prediction, predictive_variance = summarize_predictions(
                gp_predictions(calibration_features)
            )
            # Substitute held-out targets and a fixed field/point ordering in a recipe.
            target = calibration_features.sum(-1)
            calibrator.update(
                prediction, target, aux={"sigma": predictive_variance.sqrt()}
            )
        fitted = calibrator.finalize()
        lower, upper = fitted.predict_interval(mean, aux={"sigma": variance.sqrt()})
        print("conformal intervals", lower.shape, upper.shape)


if __name__ == "__main__":
    main()
