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

"""Protocols and type hints for diffusion model interfaces."""

from typing import Any, Protocol, runtime_checkable

import torch
from jaxtyping import Float
from tensordict import TensorDict


@runtime_checkable
class DiffusionModel(Protocol):
    r"""
    Protocol defining the common interface for diffusion models.

    A diffusion model is any neural network or function that transforms a noisy
    state ``x`` at diffusion time (or noise level) ``t`` into a prediction.
    This protocol defines the standard interface that all diffusion models must
    satisfy.

    Any model or function that implements this interface can be used with
    preconditioners, losses, samplers, and other diffusion utilities.

    The interface is **prediction-agnostic**: whether your model predicts
    clean data (:math:`\mathbf{x}_0`), noise (:math:`\epsilon`), score
    (:math:`\nabla \log p`), or velocity (:math:`\mathbf{v}`), the signature
    remains the same.

    The interface supports both conditional and unconditional diffusion models.
    The ``condition`` argument supports different conditioning scenarios:

    - **torch.Tensor**: Use when there is a single conditioning tensor
      (e.g., a class embedding or a single image).
    - **TensorDict**: Use when multiple conditioning tensors are needed,
      possibly with different shapes. The string keys can be used to provide
      semantic information about each conditioning tensor.
    - **None**: Use for unconditional generation or specific scenarios like
      classifier-free guidance where the model should ignore conditioning.

    Examples
    --------
    >>> import torch
    >>> import torch.nn.functional as F
    >>> from physicsnemo.diffusion import DiffusionModel
    >>>
    >>> class Denoiser:
    ...     def __call__(self, x, t, condition=None, **kwargs):
    ...         return F.relu(x)
    ...
    >>> isinstance(Denoiser(), DiffusionModel)
    True
    """

    def __call__(
        self,
        x: Float[torch.Tensor, " B *dims"],
        t: Float[torch.Tensor, " B"],
        condition: Float[torch.Tensor, " B *cond_dims"] | TensorDict | None = None,
        **model_kwargs: Any,
    ) -> Float[torch.Tensor, " B *dims"]:
        r"""
        Forward pass of the diffusion model.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latent state of shape :math:`(B, *)` where :math:`B` is the
            batch size and :math:`*` denotes any number of additional
            dimensions (e.g., channels and spatial dimensions).
        t : torch.Tensor
            Diffusion time or noise level tensor of shape :math:`(B,)`.
        condition : torch.Tensor, TensorDict, or None, optional, default=None
            Conditioning information for the model. If a Tensor or a TensorDict
            is passed, it should have batch size :math:`B` matching that of
            ``x``. Pass ``None`` for an unconditional model.

        **model_kwargs : Any
            Additional keyword arguments specific to the model implementation.

        Returns
        -------
        torch.Tensor
            Model output with the same shape as ``x``.
        """
        ...
