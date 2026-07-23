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

r"""HealDA data-assimilation models and building blocks.

:class:`VideoHealDA` (current) uses :class:`PixelCrossAttention` and
:class:`ObsTokenizerFiLM`. :class:`HealDA` (v1, deprecated) is retained only to
run existing v1 checkpoints; its observation-embedder components are no longer
publicly exported and will be removed in a future release.
"""

from .attention_layers import PixelCrossAttention
from .healda import HealDA
from .obs_context import ObsContext, prepare_obs_context
from .obs_tokenizer import ObsTokenizerFiLM
from .video_dit import VideoDiT
from .video_healda import VideoHealDA

__all__ = [
    "VideoHealDA",
    "HealDA",
    "PixelCrossAttention",
    "ObsTokenizerFiLM",
    "ObsContext",
    "prepare_obs_context",
    "VideoDiT",
]
