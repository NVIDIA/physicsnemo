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

"""Network construction and prediction for Rotor37."""

from typing import NamedTuple

import torch
from hydra.utils import instantiate

from physicsnemo.models.geotransolver import GeoTransolver


class Prediction(NamedTuple):
    """Normalized fields, normalized compressor outputs and basis coefficients."""

    fields: torch.Tensor
    globals: torch.Tensor
    coefficients: torch.Tensor


def build_model(model_cfg):
    """Instantiate the configured network with a zero-initialized output layer.

    The untrained network then predicts the training mean for every case. For
    GeoTransolver, the input weights of the case features also start at zero,
    so its sensitivity to each feature grows from zero during training.
    """
    model = instantiate(model_cfg)
    if isinstance(model, GeoTransolver):
        tokenizer = model.context_builder.global_tokenizer
        inputs = (
            model.preprocess[0].layers[0],
            tokenizer.in_project_x,
            tokenizer.in_project_fx,
        )
        output = model.ln_mlp_out[0][1]
    else:
        inputs = ()
        output = model.final_layer.linear
    with torch.no_grad():
        for layer in inputs:
            layer.weight.zero_()
        output.weight.zero_()
        output.bias.zero_()
    return model


def predict(model, batch, decoder):
    """Predict compressor outputs and decoded fields for a batch of cases.

    GeoTransolver uses the case features as its single query token and its
    global context and reads the point context through cross-attention. The
    multilayer perceptron reads the case features only. The first three
    outputs are the normalized compressor outputs and the rest are basis
    coefficients.
    """
    network = getattr(model, "module", model)
    if isinstance(network, GeoTransolver):
        features = batch["features"].unsqueeze(1)
        output = model(features, global_embedding=features, geometry=batch["geometry"])
        output = output[:, 0]
    else:
        output = model(batch["features"])
    coefficients = output[:, 3:]
    return Prediction(decoder(coefficients), output[:, :3], coefficients)
