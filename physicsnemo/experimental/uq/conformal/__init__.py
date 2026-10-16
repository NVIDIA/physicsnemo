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

r"""Post-hoc conformal prediction for spatio-temporal fields.

Wrap a trained model's field predictions in calibrated prediction intervals
without retraining it. Pass predictions and targets from a held-out
calibration set to a calibrator, call its ``finalize()`` to get a predictor,
then call ``predictor.predict_interval`` on new model outputs to get
``(lo, hi)`` bounds with a stated coverage level.

:class:`~physicsnemo.experimental.uq.conformal.CellwiseCalibrator` covers
each output element with probability at least :math:`1 - \alpha`. Every
sample must use the same mesh, with the same points in the same order, and
you pass the coordinates as ``points=`` on every call.

Pick the score by what the model outputs:
:class:`~physicsnemo.experimental.uq.conformal.AbsoluteErrorScore` for a
point prediction,
:class:`~physicsnemo.experimental.uq.conformal.NormalizedErrorScore` for a
mean plus a standard deviation in ``aux["sigma"]``, and
:class:`~physicsnemo.experimental.uq.conformal.QuantileRegressionScore` for
quantile heads in ``aux["lo"]`` and ``aux["hi"]``.

The calibrator accepts plain tensors or ``TensorDict`` containers of fields.

Typical usage::

    calibrator = CellwiseCalibrator(AbsoluteErrorScore(), alpha=0.1)
    for pred, target in calibration_set:
        calibrator.update(pred, target, points=points)
    predictor = calibrator.finalize()

    lo, hi = predictor.predict_interval(model_output, points=points)

Save a fitted predictor, load it elsewhere (the file loads with
``weights_only=True``), and check its coverage on held-out data. The
coverage report measures the same quantity the calibrator guarantees::

    predictor.save("predictor.pt", provenance={"dataset": "holdout-v1"})
    predictor = ConformalPredictor.load("predictor.pt")

    accumulator = predictor.coverage_accumulator()
    for pred, target in held_out_set:
        lo, hi = predictor.predict_interval(pred, points=points)
        accumulator.update(lo, hi, target)
    report = accumulator.finalize()

The guarantee assumes the calibration and deployment samples are
exchangeable, for example drawn independently from the same distribution.
If deployment data drifts away from the calibration data, the coverage is
no longer guaranteed.
"""

from .calibrators import CellwiseCalibrator
from .diagnostics import CoverageAccumulator
from .predictors import ConformalPredictor
from .scores import (
    AbsoluteErrorScore,
    NormalizedErrorScore,
    QuantileRegressionScore,
)

__all__ = [
    "AbsoluteErrorScore",
    "CellwiseCalibrator",
    "ConformalPredictor",
    "CoverageAccumulator",
    "NormalizedErrorScore",
    "QuantileRegressionScore",
]
