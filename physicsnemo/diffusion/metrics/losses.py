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

"""Denoising score matching losses for diffusion model training."""

from typing import Callable, Literal

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

from physicsnemo.diffusion.base import DiffusionModel
from physicsnemo.diffusion.noise_schedulers import NoiseScheduler


class MSEDSMLoss:
    r"""
    Mean-squared-error denoising score matching loss for training diffusion
    models.

    Implements the denoising score matching objective. Given clean data
    :math:`\mathbf{x}_0`, the loss is:

    .. math::
        \mathcal{L} = \mathbb{E}_{t, \boldsymbol{\epsilon}}
        \left[ w(t) \left\| \hat{\mathbf{x}}_0(\mathbf{x}_t, t)
        - \mathbf{x}_0 \right\|^2 \right]

    All training functionality is centered around a **noise scheduler** that
    must implement the
    :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler` protocol.
    At each training step the noise scheduler provides:

    - **Time sampling** via :meth:`~NoiseScheduler.sample_time`: draws
      random diffusion times :math:`t`.
    - **Noise injection** via :meth:`~NoiseScheduler.add_noise`: produces
      the noisy state :math:`\mathbf{x}_t` from clean data
      :math:`\mathbf{x}_0`.
    - **Loss weighting** via :meth:`~NoiseScheduler.loss_weight`: returns
      the per-sample weight :math:`w(t)`.

    The model can be trained to either directly predict the clean data
    :math:`\hat{\mathbf{x}}_0` (``prediction_type="x0"``, default) or to
    predict the score, which is then converted to an
    :math:`\hat{\mathbf{x}}_0` estimate via a user-provided
    ``score_to_x0_fn`` callback (``prediction_type="score"``).

    The ``model`` argument must satisfy the
    :class:`~physicsnemo.diffusion.DiffusionModel` interface:

    .. code-block:: python

        model(
            x: torch.Tensor,       # Noisy state, shape: (B, *)
            t: torch.Tensor,       # Diffusion time, shape: (B,)
            condition: torch.Tensor | TensorDict | None = None, # Conditioning information, shape: (B, *cond_dims)
            **model_kwargs: Any,
        ) -> torch.Tensor          # Model prediction, shape: (B, *)

    When ``prediction_type="score"``, you must also provide a
    ``score_to_x0_fn`` callback when instantiating the loss, with the
    following signature:

    .. code-block:: python

        score_to_x0_fn(
            score: torch.Tensor,   # Predicted score, shape: (B, *)
            x_t: torch.Tensor,     # Noisy state, shape: (B, *)
            t: torch.Tensor,       # Diffusion time, shape: (B,)
        ) -> torch.Tensor          # Clean data estimate, shape: (B, *)

    For
    :class:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler`
    subclasses, the
    :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.score_to_x0`
    method provides a ready-made ``score_to_x0_fn``.

    Parameters
    ----------
    model : DiffusionModel
        Diffusion model to train. Can be a plain neural network, or a
        model wrapped with a preconditioner (e.g.,
        :class:`~physicsnemo.diffusion.preconditioners.EDMPreconditioner`).
        The output is interpreted according to ``prediction_type``: as a
        clean-data estimate when ``"x0"``, or as a score when ``"score"``.
        Must satisfy the
        :class:`~physicsnemo.diffusion.DiffusionModel` protocol.
    noise_scheduler : NoiseScheduler
        Noise scheduler implementing the
        :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`
        protocol, providing the methods: :meth:`~NoiseScheduler.sample_time`,
        :meth:`~NoiseScheduler.add_noise`, and
        :meth:`~NoiseScheduler.loss_weight`.
    prediction_type : Literal["x0", "score"], default="x0"
        Type of prediction the model outputs. Use ``"x0"`` when the model
        directly predicts clean data (the most common case with standard
        preconditioners). Use ``"score"`` when the model predicts the score,
        in which case ``score_to_x0_fn`` must be provided.
    score_to_x0_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional
        Callback to convert a score prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="score"``. See above for the expected signature.
    reduction : Literal["none", "mean", "sum"], default="mean"
        Reduction to apply to the output: ``"none"`` returns the
        per-element loss, ``"mean"`` returns the mean over all elements,
        ``"sum"`` returns the sum over all elements.

    Raises
    ------
    ValueError
        If ``prediction_type`` is not ``"x0"`` or ``"score"``.
    ValueError
        If ``prediction_type="score"`` and ``score_to_x0_fn`` is ``None``.

    Examples
    --------
    **Example 1:** Standard unconditional x0-predictor training with EDM
    schedule and preconditioner:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    >>> from physicsnemo.diffusion.metrics.losses import MSEDSMLoss
    >>>
    >>> class UnconditionalModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> model = UnconditionalModel()
    >>> scheduler = EDMNoiseScheduler()
    >>> precond = EDMPreconditioner(model)
    >>> loss_fn = MSEDSMLoss(precond, scheduler)
    >>> x0 = torch.randn(4, 3, 8, 8)
    >>> loss = loss_fn(x0)
    >>> loss.shape
    torch.Size([])

    **Example 2:** Conditional training with VP schedule. The model receives
    multiple conditioning tensors (an image and a vector) through a
    ``TensorDict``:

    >>> from physicsnemo.diffusion.noise_schedulers import VPNoiseScheduler
    >>> from physicsnemo.diffusion.preconditioners import VPPreconditioner
    >>> from tensordict import TensorDict
    >>>
    >>> class ConditionalModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.img_net = torch.nn.Conv2d(6, 3, 1)
    ...         self.vec_net = torch.nn.Linear(4, 3 * 8 * 8)
    ...     def forward(self, x, t, condition=None):
    ...         y_img = condition["image"]
    ...         y_vec = self.vec_net(condition["vector"]).view_as(x)
    ...         return self.img_net(torch.cat([x, y_img], dim=1)) + y_vec
    >>>
    >>> cond_model = ConditionalModel()
    >>> scheduler_vp = VPNoiseScheduler()
    >>> precond_vp = VPPreconditioner(cond_model)
    >>> loss_fn = MSEDSMLoss(precond_vp, scheduler_vp)
    >>> condition = TensorDict({
    ...     "image": torch.randn(4, 3, 8, 8),
    ...     "vector": torch.randn(4, 4),
    ... }, batch_size=[4])
    >>> loss = loss_fn(x0, condition=condition)
    >>> loss.shape
    torch.Size([])

    **Example 3:** Training a score-predictor. The model outputs score
    predictions, and ``score_to_x0_fn`` converts them to clean data estimates
    for the loss computation. For linear-Gaussian noise schedulers, the method
    :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.score_to_x0`
    provides a ready-made conversion:

    >>> scheduler = EDMNoiseScheduler()
    >>> loss_fn = MSEDSMLoss(
    ...     model=model,
    ...     noise_scheduler=scheduler,
    ...     prediction_type="score",
    ...     score_to_x0_fn=scheduler.score_to_x0,
    ... )
    >>> loss = loss_fn(x0)
    >>> loss.shape
    torch.Size([])

    **Example 4:** Bare-bones approach without any built-in scheduler or
    preconditioner. This shows how to plug custom components into
    :class:`MSEDSMLoss` by implementing the
    :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler` and
    :class:`~physicsnemo.diffusion.DiffusionModel` protocols from scratch:

    >>> import math
    >>>
    >>> # Custom noise scheduler (EDM-like, sigma(t)=t, alpha(t)=1)
    >>> class MyScheduler:
    ...     def sample_time(self, N, *, device=None, dtype=None):
    ...         return (0.002 * (80 / 0.002) ** torch.rand(N, device=device, dtype=dtype))
    ...     def add_noise(self, x0, time):
    ...         return x0 + time.view(-1, 1, 1, 1) * torch.randn_like(x0)
    ...     def loss_weight(self, t):
    ...         return (t**2 + 0.5**2) / (t * 0.5) ** 2
    ...     def score_to_x0(self, score, x_t, t):
    ...         return x_t + t.view(-1, 1, 1, 1)**2 * score
    ...     def timesteps(self, n, *, device=None, dtype=None):
    ...         return torch.zeros(1)
    ...     def init_latents(self, s, tN, *, device=None, dtype=None):
    ...         return torch.zeros(1)
    ...     def get_denoiser(self, **kw):
    ...         return lambda x, t: x
    >>>
    >>> # Custom model with single-tensor conditioning
    >>> class ConditionalModel:
    ...     def __init__(self):
    ...         self.w = torch.randn(3, 6, 1, 1) * 0.01
    ...     def __call__(self, x, t, condition=None, **kw):
    ...         return torch.nn.functional.conv2d(
    ...             torch.cat([x, condition], dim=1), self.w)
    >>>
    >>> my_scheduler = MyScheduler()
    >>> cond_model = ConditionalModel()
    >>> loss_fn = MSEDSMLoss(cond_model, my_scheduler)
    >>> x0 = torch.randn(2, 3, 8, 8)
    >>> cond = torch.randn(2, 3, 8, 8)  # single-tensor conditioning
    >>> loss = loss_fn(x0, condition=cond)
    >>> loss.shape
    torch.Size([])
    >>>
    >>> # Also works with score prediction + custom conversion
    >>> loss_fn_score = MSEDSMLoss(
    ...     cond_model, my_scheduler,
    ...     prediction_type="score",
    ...     score_to_x0_fn=my_scheduler.score_to_x0,
    ... )
    >>> loss = loss_fn_score(x0, condition=cond)
    >>> loss.shape
    torch.Size([])
    """

    def __init__(
        self,
        model: DiffusionModel,
        noise_scheduler: NoiseScheduler,
        prediction_type: Literal["x0", "score"] = "x0",
        score_to_x0_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        self.model = model
        self.noise_scheduler = noise_scheduler

        if prediction_type == "x0":
            self._to_x0 = lambda prediction, x_t, t: prediction

        elif prediction_type == "score":
            if score_to_x0_fn is None:
                raise ValueError(
                    "score_to_x0_fn must be provided when prediction_type='score'."
                )
            self._to_x0 = score_to_x0_fn

        else:
            raise ValueError(
                f"prediction_type must be 'x0' or 'score', got '{prediction_type}'."
            )

        # Define the reduction callbacks
        _reductions = {
            "none": lambda x: x,
            "mean": lambda x: x.mean(),
            "sum": lambda x: x.sum(),
        }
        if reduction not in _reductions:
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'."
            )
        self._reduce = _reductions[reduction]

    def __call__(
        self,
        x0: Float[Tensor, " B *dims"],
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
    ) -> Float[Tensor, " B *dims"] | Float[Tensor, ""]:
        r"""
        Compute the denoising score matching loss.

        Parameters
        ----------
        x0 : Tensor
            Clean data of shape :math:`(B, *)` where :math:`B` is the batch
            size and :math:`*` denotes any number of additional dimensions.
        condition : Tensor, TensorDict, or None, optional, default=None
            Conditioning information passed to the model. See
            :class:`~physicsnemo.diffusion.DiffusionModel` for details.

        Returns
        -------
        Tensor
            If ``reduction="none"``, the per-element weighted loss with same
            shape :math:`(B, *)` as ``x0``. If ``reduction="mean"``, or
            ``reduction="sum"``, a scalar tensor.
        """
        B = x0.shape[0]
        t = self.noise_scheduler.sample_time(B, device=x0.device, dtype=x0.dtype)
        x_t = self.noise_scheduler.add_noise(x0, t)
        prediction = self.model(x_t, t, condition=condition)
        x0_pred = self._to_x0(prediction, x_t, t)
        loss = (x0_pred - x0) ** 2
        w = self.noise_scheduler.loss_weight(t)
        loss = w.reshape(-1, *([1] * (x0.ndim - 1))) * loss
        return self._reduce(loss)


class WeightedMSEDSMLoss:
    r"""
    Weighted mean-squared-error denoising score matching loss.

    Identical to :class:`MSEDSMLoss` but accepts an additional ``weight``
    argument that multiplies the per-element squared error.

    .. math::
        \mathcal{L} = \mathbb{E}_{t, \boldsymbol{\epsilon}}
        \left[ w(t) \left\| \mathbf{m} \odot
        \left(\hat{\mathbf{x}}_0(\mathbf{x}_t, t)
        - \mathbf{x}_0\right) \right\|^2 \right]

    where :math:`\mathbf{m}` is the element-wise weight (e.g., a binary
    mask). A common use case is masking out certain spatial regions or
    channels of the state.

    .. note::

        The ``weight`` argument is **not** related to the time-dependent loss
        weight :math:`w(t)` provided by the noise scheduler.

    For more details on prediction types, expected signatures, and
    additional examples, see :class:`MSEDSMLoss`.

    Parameters
    ----------
    model : DiffusionModel
        Diffusion model to train. Must satisfy the
        :class:`~physicsnemo.diffusion.DiffusionModel` protocol.
    noise_scheduler : NoiseScheduler
        Noise scheduler implementing the
        :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`
        protocol.
    prediction_type : Literal["x0", "score"], default="x0"
        Type of prediction the model outputs. See :class:`MSEDSMLoss`.
    score_to_x0_fn : callable, optional
        Callback to convert a score prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="score"``.
    reduction : {"none", "mean", "sum"}, default="mean"
        Reduction to apply to the output: ``"none"`` returns the
        per-element loss, ``"mean"`` the mean, ``"sum"`` the sum.

    Examples
    --------
    Apply a spatial mask so the loss is computed only over unmasked regions:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    >>> from physicsnemo.diffusion.metrics.losses import WeightedMSEDSMLoss
    >>>
    >>> class UnconditionalModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> model = UnconditionalModel()
    >>> scheduler = EDMNoiseScheduler()
    >>> precond = EDMPreconditioner(model)
    >>> loss_fn = WeightedMSEDSMLoss(precond, scheduler)
    >>>
    >>> x0 = torch.randn(4, 3, 8, 8)
    >>> # Binary mask: zero out the left half of the spatial domain
    >>> mask = torch.ones(4, 3, 8, 8)
    >>> mask[:, :, :, :4] = 0.0
    >>> loss = loss_fn(x0, weight=mask)
    >>> loss.shape
    torch.Size([])
    """

    def __init__(
        self,
        model: DiffusionModel,
        noise_scheduler: NoiseScheduler,
        prediction_type: Literal["x0", "score"] = "x0",
        score_to_x0_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        self.model = model
        self.noise_scheduler = noise_scheduler

        if prediction_type == "x0":
            self._to_x0 = lambda prediction, x_t, t: prediction

        elif prediction_type == "score":
            if score_to_x0_fn is None:
                raise ValueError(
                    "score_to_x0_fn must be provided when prediction_type='score'."
                )
            self._to_x0 = score_to_x0_fn

        else:
            raise ValueError(
                f"prediction_type must be 'x0' or 'score', got '{prediction_type}'."
            )

        # Define the reduction callbacks
        _reductions = {
            "none": lambda x: x,
            "mean": lambda x: x.mean(),
            "sum": lambda x: x.sum(),
        }
        if reduction not in _reductions:
            raise ValueError(
                f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'."
            )
        self._reduce = _reductions[reduction]

    def __call__(
        self,
        x0: Float[Tensor, " B *dims"],
        weight: Float[Tensor, " B *dims"],
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
    ) -> Float[Tensor, " B *dims"] | Float[Tensor, ""]:
        r"""
        Compute the weighted denoising score matching loss.

        Parameters
        ----------
        x0 : Tensor
            Clean data of shape :math:`(B, *)` where :math:`B` is the batch
            size and :math:`*` denotes any number of additional dimensions.
        weight : Tensor
            Per-element weight of shape :math:`(B, *)`, same shape as
            ``x0``. For binary masking, use 0 for masked elements and 1
            for active elements.
        condition : Tensor, TensorDict, or None, optional, default=None
            Conditioning information passed to the model. See
            :class:`~physicsnemo.diffusion.DiffusionModel` for details.

        Returns
        -------
        Tensor
            If ``reduction="none"``, the per-element weighted loss with same
            shape :math:`(B, *)` as ``x0``. If ``reduction="mean"``, or
            ``reduction="sum"``, a scalar tensor.
        """
        B = x0.shape[0]
        t = self.noise_scheduler.sample_time(B, device=x0.device, dtype=x0.dtype)
        x_t = self.noise_scheduler.add_noise(x0, t)
        prediction = self.model(x_t, t, condition=condition)
        x0_pred = self._to_x0(prediction, x_t, t)
        loss = weight * (x0_pred - x0) ** 2
        w = self.noise_scheduler.loss_weight(t)
        loss = w.reshape(-1, *([1] * (x0.ndim - 1))) * loss
        return self._reduce(loss)
