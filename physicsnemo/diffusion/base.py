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
    >>> class Model:
    ...     def __call__(self, x, t, condition=None, **kwargs):
    ...         return F.relu(x)
    ...
    >>> isinstance(Model(), DiffusionModel)
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


@runtime_checkable
class Predictor(Protocol):
    r"""
    Protocol defining a predictor interface for diffusion models.

    A predictor is any callable that takes a noisy state ``x``
    and diffusion time ``t``, and returns a prediction about the clean data or
    the noise. Common types of predictors include x0-predictor (predicts the
    clean data :math:`\mathbf{x}_0`), score-predictor, noise-predictor
    (predicts the noise :math:`\boldsymbol{\epsilon}`), velocity-predictor etc.

    This protocol is **generic** and does not assume any specific type of
    prediction. A predictor can be a trained neural network, a guidance
    function (e.g., classifier-free guidance, DPS-style guidance), or any
    combination thereof.  The exact meaning of the output depends on the
    predictor type and how it is used. Any callable that implements this
    interface can be used as a predictor in sampling utilities.

    This protocol is typically used during inference. For training, which
    often requires additional inputs like conditioning, use the more general
    :class:`DiffusionModel` protocol instead. A :class:`Predictor` can be
    obtained from a :class:`DiffusionModel` by partially applying the
    ``condition`` and any other keyword arguments using
    ``functools.partial``.

    **Relationship to Denoiser:**

    A :class:`Denoiser` is the update function used during sampling (e.g.,
    the right-hand side of an ODE/SDE). It is obtained from a
    :class:`Predictor` via the
    :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`
    factory. A typical case is ODE/SDE-based sampling, where one solves:

    .. math::
        \frac{d\mathbf{x}}{dt} = D(\mathbf{x}, t;\, P(\mathbf{x}, t))

    where :math:`P` is the **predictor** and :math:`D` is the **denoiser**
    that wraps it. This equation captures the essence of how these two
    concepts are related in the framework.

    See Also
    --------
    :class:`Denoiser` : The interface for sampling update functions.
    :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser` :
        Factory to convert a predictor into a denoiser.

    Examples
    --------
    **Example 1:** Convert a trained conditional model into a predictor using
    ``functools.partial``:

    >>> import torch
    >>> from functools import partial
    >>> from tensordict import TensorDict
    >>> from physicsnemo.diffusion import Predictor
    >>>
    >>> class MyModel:
    ...     def __call__(self, x, t, condition=None):
    ...         # x0-predictor: returns estimate of clean data
    ...         # (here assumes conditional normal distribution N(x|y))
    ...         t_bc = t.view(-1, *([1] * (x.ndim - 1)))
    ...         return x / (1 + t_bc**2) + condition["y"]
    ...
    >>> model = MyModel()
    >>> cond = TensorDict({"y": torch.randn(2, 4)}, batch_size=[2])
    >>> x0_predictor = partial(model, condition=cond)
    >>> isinstance(x0_predictor, Predictor)
    True

    **Example 2:** Convert the x0-predictor above into a score-predictor
    (using a simple EDM-like schedule where :math:`\sigma(t) = t` and
    :math:`\alpha(t) = 1`):

    >>> def x0_to_score(x0, x_t, t):
    ...     sigma_sq = t.view(-1, 1) ** 2
    ...     return (x0 - x_t) / sigma_sq
    >>>
    >>> def score_predictor(x, t):
    ...     x0_pred = x0_predictor(x, t)
    ...     return x0_to_score(x0_pred, x, t)
    >>>
    >>> isinstance(score_predictor, Predictor)
    True
    """

    def __call__(
        self,
        x: Float[torch.Tensor, " B *dims"],
        t: Float[torch.Tensor, " B"],
    ) -> Float[torch.Tensor, " B *dims"]:
        r"""
        Forward pass of the predictor.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latent state of shape :math:`(B, *)` where :math:`B` is the
            batch size and :math:`*` denotes any number of additional
            dimensions (e.g., channels and spatial dimensions).
        t : torch.Tensor
            Batched diffusion time tensor of shape :math:`(B,)`.

        Returns
        -------
        torch.Tensor
            Prediction output with the same shape as ``x``. The exact meaning
            depends on the predictor type (x0, score, noise, velocity, etc.).
        """
        ...


@runtime_checkable
class Denoiser(Protocol):
    r"""
    Protocol defining a denoiser interface for diffusion model sampling.

    A denoiser is the **update function** used during sampling. It takes a
    noisy state ``x`` and diffusion time ``t``, and returns the update term
    consumed by a :class:`~physicsnemo.diffusion.samplers.solvers.Solver`.
    For continuous-time methods this is typically the right-hand side of the
    ODE/SDE, but the interface is generic and can support other sampling
    methods as well.

    This is the interface used by
    :class:`~physicsnemo.diffusion.samplers.solvers.Solver` classes and the
    :func:`~physicsnemo.diffusion.samplers.sample` function. Any callable
    that implements this interface can be used as a denoiser.

    **Important distinction from Predictor:**

    - A :class:`Predictor` is any callable that outputs a raw prediction
      (e.g., clean data :math:`\mathbf{x}_0`, score, guidance signal, etc.).
    - A :class:`Denoiser` is the update function derived from one or more
      predictors, used directly by the solver during sampling.

    **Typical workflow:**

    1. Start with one or more :class:`Predictor` instances (e.g. trained model)
    2. Optionally combine predictors (e.g., conditional + guidance scores)
    3. Convert to a :class:`Denoiser` using
       :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser`
    4. Pass the denoiser to
       :func:`~physicsnemo.diffusion.samplers.sample` together with a
       :class:`~physicsnemo.diffusion.samplers.solvers.Solver`

    See Also
    --------
    :class:`Predictor` : The interface for raw predictions.
    :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser` :
        Factory to convert a predictor into a denoiser.
    :func:`~physicsnemo.diffusion.samplers.sample` : The sampling function
        that uses this denoiser interface.

    Examples
    --------
    Manually creating a denoiser from an x0-predictor using a simple EDM-like
    schedule (:math:`\sigma(t)=t`, :math:`\alpha(t)=1`):

    >>> import torch
    >>> from physicsnemo.diffusion import Denoiser
    >>>
    >>> # Start from a predictor (x0-predictor)
    >>> def x0_predictor(x, t):
    ...     t_bc = t.view(-1, *([1] * (x.ndim - 1)))
    ...     return x / (1 + t_bc**2)
    >>>
    >>> # Build a denoiser (ODE RHS) from scratch:
    >>> # score = (x0 - x) / sigma^2,  ODE RHS = -0.5 * g^2 * score
    >>> # For EDM: sigma = t, g^2 = 2*t, so RHS = (x0 - x) / t
    >>> def my_denoiser(x, t):
    ...     x0 = x0_predictor(x, t)
    ...     t_bc = t.view(-1, *([1] * (x.ndim - 1)))
    ...     return (x0 - x) / t_bc
    ...
    >>> isinstance(my_denoiser, Denoiser)
    True
    """

    def __call__(
        self,
        x: Float[torch.Tensor, " B *dims"],
        t: Float[torch.Tensor, " B"],
    ) -> Float[torch.Tensor, " B *dims"]:
        r"""
        Compute the denoising update at the given state and time.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latent state of shape :math:`(B, *)` where :math:`B` is the
            batch size and :math:`*` denotes any number of additional
            dimensions (e.g., channels and spatial dimensions).
        t : torch.Tensor
            Batched diffusion time tensor of shape :math:`(B,)`.
            All batch elements in the latent state ``x`` typically share the
            same diffusion time values, but ``t`` is still required to be a
            batched tensor.

        Returns
        -------
        torch.Tensor
            Denoising update term with the same shape as ``x``.
        """
        ...
