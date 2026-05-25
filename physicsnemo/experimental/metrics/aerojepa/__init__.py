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

"""JEPA-style losses and regularizers (experimental).

Reusable losses for AeroJEPA and JEPA-style scientific-ML models:
SIGReg / TokenLatentSIGReg latent regularizers, padding-aware masking
helpers, and the reconstruction loss family (MSE, relative-L2, relative-MSE,
and the relative-L2-plus-MSE hybrid).

API stability: experimental. Names and signatures may change between releases
until the design graduates out of ``physicsnemo.experimental``.

References
----------
Giral et al., "AeroJEPA: Learning Semantic Latent Representations for
Scalable 3D Aerodynamic Field Modeling", preprint arXiv:2605.05586 (2026).
"""

from .masking import flatten_valid_token_features, reshape_token_features_for_sigreg
from .reconstruction import (
    MSELoss,
    RelativeL2Loss,
    RelativeL2MSELoss,
    RelativeMSELoss,
    mse_loss,
    relative_l2_loss,
    relative_l2_mse_loss,
    relative_mse_loss,
)
from .sigreg import SIGReg, TokenLatentSIGReg

__all__ = [
    # Masking helpers
    "flatten_valid_token_features",
    "reshape_token_features_for_sigreg",
    # SIGReg
    "SIGReg",
    "TokenLatentSIGReg",
    # Reconstruction (functional + Module pairs)
    "MSELoss",
    "RelativeL2Loss",
    "RelativeL2MSELoss",
    "RelativeMSELoss",
    "mse_loss",
    "relative_l2_loss",
    "relative_l2_mse_loss",
    "relative_mse_loss",
]
