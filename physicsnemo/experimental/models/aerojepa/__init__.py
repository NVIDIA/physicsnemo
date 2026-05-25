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

"""AeroJEPA model and its model-specific subcomponents (experimental).

The :class:`AeroJEPA` model composes a context encoder, a target encoder,
a predictor head, a trunk that wires the encoders and decoder together,
and a query-based field decoder. All five pieces live in this subpackage
(the encoders under ``aerojepa.encoders``). Reusable building blocks the
model is built from — attention blocks, the point-cloud tokenizer, the
Fourier positional encoding, token dataclasses, and the batching/mask/k-NN
helpers — live in :mod:`physicsnemo.experimental.nn.aerojepa`.

API stability: experimental. Names and signatures may change between
releases until the design graduates out of ``physicsnemo.experimental``.

References
----------
Giral et al., "AeroJEPA: Learning Semantic Latent Representations for
Scalable 3D Aerodynamic Field Modeling", preprint arXiv:2605.05586 (2026).
"""

__all__ = []
