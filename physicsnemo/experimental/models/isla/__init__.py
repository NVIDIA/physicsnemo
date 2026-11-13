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

"""ISLA (Invariant Slice Attention): an E(d)-equivariant soft-slice transformer
for steady boundary-value PDE surrogates.

ISLA uses the mesh-native API (boundary Meshes in, a point-cloud Mesh out), with
named inputs and outputs of declared tensor rank
(:class:`~physicsnemo.mesh.fields.RankSpec` and
:class:`~physicsnemo.mesh.fields.FieldSchema`). The network is exactly covariant
by construction. The full model is exactly covariant, up to roundoff, when the
boundary's canonical frame is unique and not close to a tie. At a tie a surface
task stays exact, and an interior task is exact only with every boundary cell a
token, the exact read and a closed boundary. See
:mod:`physicsnemo.experimental.models.isla.model`, *Where the covariance is
exact*."""

from physicsnemo.experimental.models.isla._scales import case_scales
from physicsnemo.experimental.models.isla.model import ISLA, QueryChunks

__all__ = ["ISLA", "QueryChunks", "case_scales"]
