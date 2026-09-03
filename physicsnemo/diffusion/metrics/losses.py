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

from __future__ import annotations

from typing import Any, Callable, Literal

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

from physicsnemo.diffusion.base import DiffusionModel, PredictorType
from physicsnemo.diffusion.noise_schedulers import NoiseScheduler
from physicsnemo.diffusion.utils.utils import apply_loss_weight


def _check_domain_parallel_scheduler(
    x0: torch.Tensor, scheduler: NoiseScheduler
) -> None:
    """Raise if *x0* is domain-sharded but *scheduler* is not domain-parallel."""
    mesh = getattr(x0, "device_mesh", None)
    if mesh is None:
        return
    from physicsnemo.diffusion.noise_schedulers.domain_parallel import (
        DomainParallelNoiseScheduler,
    )

    if not isinstance(scheduler, DomainParallelNoiseScheduler):
        raise ValueError(
            "x0 is a ShardTensor (domain-parallel) but the noise scheduler "
            "is not a DomainParallelNoiseScheduler. Wrap your scheduler with "
            "DomainParallelNoiseScheduler before passing it to the loss. "
            "See physicsnemo.diffusion.noise_schedulers.DomainParallelNoiseScheduler."
        )


def _check_weight_mesh(weight: torch.Tensor, x0: torch.Tensor) -> None:
    """Raise if *x0* is a DTensor but *weight* is not on the same mesh."""
    mesh = getattr(x0, "device_mesh", None)
    if mesh is None:
        return
    weight_mesh = getattr(weight, "device_mesh", None)
    if weight_mesh is None:
        raise ValueError(
            "x0 is a DTensor (domain-parallel) but weight is a plain tensor. "
            "weight must be a DTensor on the same device mesh as x0. "
            "Shard or replicate weight to match x0 before passing it to the loss."
        )


def _maybe_promote_to_mesh(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Promote *t* to a replicated DTensor on *ref*'s device mesh if needed.

    This is only required for genuine ``DTensor`` references: DTensor dispatch
    raises on mixed plain-tensor/DTensor arithmetic, so a plain ``t`` must be
    lifted to a replicated ``DTensor`` on the same mesh.

    When ``ref`` is a :class:`ShardTensor`, ``t`` is returned unchanged: the
    ShardTensor dispatch path auto-promotes plain operands to replicated tensors
    on its mesh on the way in (see
    :class:`~physicsnemo.domain_parallel.shard_tensor.TensorPromotionMode`), so
    promoting here would be redundant (and would wrongly wrap scalar
    coefficients into DTensors).
    """
    from torch.distributed.tensor import DTensor

    # Only genuine (non-Shard) DTensors need manual promotion here. A
    # ShardTensor subclasses torch.Tensor (not DTensor), so it never matches
    # this check and correctly falls through to its own auto-promoting dispatch.
    if not isinstance(ref, DTensor):
        return t

    from torch.distributed.tensor.placement_types import Replicate

    if isinstance(t, DTensor):
        return t
    return DTensor.from_local(t, device_mesh=ref.device_mesh, placements=[Replicate()])


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

    - **Time sampling** via :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.sample_time`: draws
      random diffusion times :math:`t`.
    - **Noise injection** via :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.add_noise`: produces
      the noisy state :math:`\mathbf{x}_t` from clean data
      :math:`\mathbf{x}_0`.
    - **Loss weighting** via :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.loss_weight`: returns
      the per-sample weight :math:`w(t)`.

    The model can directly predict the clean data
    :math:`\hat{\mathbf{x}}_0` (``prediction_type="x0"``, default), or
    predict the score, noise, or flow (``prediction_type="score"``,
    ``"epsilon"``, or ``"flow"``); the matching user-provided
    ``*_to_x0_fn`` callback then converts the prediction to an
    :math:`\hat{\mathbf{x}}_0` estimate.

    .. warning::

        For domain-parallel training where ``x0`` is a ``ShardTensor``,
        the scheduler **must** be wrapped with
        :class:`~physicsnemo.diffusion.noise_schedulers.DomainParallelNoiseScheduler`
        so that sampled diffusion times are broadcast across spatial
        shards.  Passing a plain scheduler with sharded data will raise a
        ``ValueError`` at runtime.

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
        protocol, providing the methods:
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.sample_time`,
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.add_noise`, and
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.loss_weight`.
    prediction_type : PredictorType, default="x0"
        Type of prediction the model outputs. Use ``"x0"`` when the model
        directly predicts clean data (the most common case with standard
        preconditioners). Use ``"score"`` when the model predicts the score,
        in which case ``score_to_x0_fn`` must be provided. Use ``"epsilon"``
        when the model predicts the noise, which requires
        ``epsilon_to_x0_fn``. Use ``"flow"`` when the model predicts the
        flow (velocity), which requires ``flow_to_x0_fn``.
    score_to_x0_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional
        Callback to convert a score prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="score"``. See above for the expected signature.
    epsilon_to_x0_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional
        Callback to convert an epsilon (noise) prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="epsilon"``. See above for the expected signature.
    flow_to_x0_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional
        Callback to convert a flow (velocity) prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="flow"``, e.g.
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.flow_to_x0`.
    reduction : Literal["none", "mean", "sum"], default="mean"
        Reduction to apply to the output: ``"none"`` returns the
        per-element loss, ``"mean"`` returns the mean over all elements,
        ``"sum"`` returns the sum over all elements. For irregular data
        (padded batches, point clouds, graphs), use ``"none"`` and apply
        your own reduction, such as a masked mean.

    Raises
    ------
    ValueError
        If ``prediction_type`` is not ``"x0"``, ``"score"``, ``"epsilon"``,
        or ``"flow"``.
    ValueError
        If ``prediction_type="score"`` and ``score_to_x0_fn`` is ``None``.
    ValueError
        If ``prediction_type="epsilon"`` and ``epsilon_to_x0_fn`` is ``None``.
    ValueError
        If ``prediction_type="flow"`` and ``flow_to_x0_fn`` is ``None``.

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
    :class:`~physicsnemo.diffusion.DiffusionModel` protocols from scratch.
    It also demonstrates passing externally sampled diffusion times via the
    ``t`` argument for per-sigma-bin loss tracking:

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
    >>>
    >>> # Sample times externally for per-sigma-bin loss tracking
    >>> t = my_scheduler.sample_time(x0.shape[0], device=x0.device, dtype=x0.dtype)
    >>> loss = loss_fn(x0, condition=cond, t=t)
    >>> loss.shape
    torch.Size([])
    >>> t.shape  # t is available for diagnostics after the loss call
    torch.Size([2])
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
        prediction_type: PredictorType = "x0",
        score_to_x0_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        epsilon_to_x0_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        flow_to_x0_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        self.model = model
        self.noise_scheduler = noise_scheduler

        match prediction_type:
            case "x0":
                self._to_x0 = lambda prediction, x_t, t: prediction
            case "score":
                if score_to_x0_fn is None:
                    raise ValueError(
                        "score_to_x0_fn must be provided when prediction_type='score'."
                    )
                self._to_x0 = score_to_x0_fn
            case "epsilon":
                if epsilon_to_x0_fn is None:
                    raise ValueError(
                        "epsilon_to_x0_fn must be provided when prediction_type='epsilon'."
                    )
                self._to_x0 = epsilon_to_x0_fn
            case "flow":
                if flow_to_x0_fn is None:
                    raise ValueError(
                        "flow_to_x0_fn must be provided when prediction_type='flow'."
                    )
                self._to_x0 = flow_to_x0_fn
            case _:
                raise ValueError(
                    f"prediction_type must be 'x0', 'score', 'epsilon', or "
                    f"'flow', got '{prediction_type}'."
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
        t: Float[Tensor, " B"] | None = None,
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        **model_kwargs: Any,
    ) -> Float[Tensor, " B *dims"] | Float[Tensor, ""]:
        r"""
        Compute the denoising score matching loss.

        Parameters
        ----------
        x0 : Tensor
            Clean data of shape :math:`(B, *)` where :math:`B` is the batch
            size and :math:`*` denotes any number of additional dimensions.
        t : Tensor or None, optional, default=None
            Pre-sampled diffusion time values of shape :math:`(B,)`. When
            ``None`` (the default), times are sampled internally via
            :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.sample_time`.
            Passing explicit times is useful when the caller needs access
            to the sampled values for diagnostics (e.g., per-sigma-bin
            loss tracking).
        condition : Tensor, TensorDict, or None, optional, default=None
            Conditioning information passed to the model. See
            :class:`~physicsnemo.diffusion.DiffusionModel` for details.
        **model_kwargs : Any
            Additional keyword arguments forwarded to the model

        Returns
        -------
        Tensor
            If ``reduction="none"``, the per-element weighted loss with same
            shape :math:`(B, *)` as ``x0``. If ``reduction="mean"``, or
            ``reduction="sum"``, a scalar tensor.
        """
        if not torch.compiler.is_compiling():
            _check_domain_parallel_scheduler(x0, self.noise_scheduler)
        B = x0.shape[0]
        if t is None:
            t = self.noise_scheduler.sample_time(B, device=x0.device, dtype=x0.dtype)
        x_t = self.noise_scheduler.add_noise(x0, t)
        prediction = self.model(x_t, t, condition=condition, **model_kwargs)
        x0_pred = self._to_x0(prediction, x_t, t)
        loss = (x0_pred - x0) ** 2
        w = self.noise_scheduler.loss_weight(t)
        w = apply_loss_weight(w, x0.ndim)
        w = _maybe_promote_to_mesh(w, loss)
        loss = w * loss
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

    .. warning::

        For domain-parallel training where ``x0`` is a ``DTensor``
        (e.g., a :class:`~physicsnemo.domain_parallel.ShardTensor`), ``weight`` must also be a ``DTensor``
        on the same device mesh.  The loss function does **not**
        automatically promote ``weight``; callers are responsible for
        sharding or replicating it to match ``x0``.  Passing a plain
        tensor ``weight`` with a sharded ``x0`` will raise a
        ``ValueError`` at runtime.

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
    prediction_type : PredictorType, default="x0"
        Type of prediction the model outputs. See :class:`MSEDSMLoss`.
    score_to_x0_fn : callable, optional
        Callback to convert a score prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="score"``.
    epsilon_to_x0_fn : callable, optional
        Callback to convert an epsilon (noise) prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="epsilon"``.
    flow_to_x0_fn : callable, optional
        Callback to convert a flow prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="flow"``.
    reduction : {"none", "mean", "sum"}, default="mean"
        Reduction to apply to the output: ``"none"`` returns the
        per-element loss, ``"mean"`` the mean, ``"sum"`` the sum. For
        irregular data (padded batches, point clouds, graphs), use
        ``"none"`` and apply your own reduction, such as a masked mean.

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
        prediction_type: PredictorType = "x0",
        score_to_x0_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        epsilon_to_x0_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        flow_to_x0_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        self.model = model
        self.noise_scheduler = noise_scheduler

        match prediction_type:
            case "x0":
                self._to_x0 = lambda prediction, x_t, t: prediction
            case "score":
                if score_to_x0_fn is None:
                    raise ValueError(
                        "score_to_x0_fn must be provided when prediction_type='score'."
                    )
                self._to_x0 = score_to_x0_fn
            case "epsilon":
                if epsilon_to_x0_fn is None:
                    raise ValueError(
                        "epsilon_to_x0_fn must be provided when prediction_type='epsilon'."
                    )
                self._to_x0 = epsilon_to_x0_fn
            case "flow":
                if flow_to_x0_fn is None:
                    raise ValueError(
                        "flow_to_x0_fn must be provided when prediction_type='flow'."
                    )
                self._to_x0 = flow_to_x0_fn
            case _:
                raise ValueError(
                    f"prediction_type must be 'x0', 'score', 'epsilon', or "
                    f"'flow', got '{prediction_type}'."
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
        t: Float[Tensor, " B"] | None = None,
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        **model_kwargs: Any,
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
            for active elements.  When ``x0`` is a ``DTensor``
            (domain-parallel), ``weight`` must also be a ``DTensor`` on
            the same device mesh.
        t : Tensor or None, optional, default=None
            Pre-sampled diffusion time values of shape :math:`(B,)`. When
            ``None`` (the default), times are sampled internally via
            :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.sample_time`.
            Passing explicit times is useful when the caller needs access
            to the sampled values for diagnostics (e.g., per-sigma-bin
            loss tracking).
        condition : Tensor, TensorDict, or None, optional, default=None
            Conditioning information passed to the model. See
            :class:`~physicsnemo.diffusion.DiffusionModel` for details.
        **model_kwargs : Any
            Additional keyword arguments forwarded to the model

        Returns
        -------
        Tensor
            If ``reduction="none"``, the per-element weighted loss with same
            shape :math:`(B, *)` as ``x0``. If ``reduction="mean"``, or
            ``reduction="sum"``, a scalar tensor.
        """
        if not torch.compiler.is_compiling():
            # Validation checks for domain-parallel training
            _check_domain_parallel_scheduler(x0, self.noise_scheduler)
            _check_weight_mesh(weight, x0)
        B = x0.shape[0]
        if t is None:
            t = self.noise_scheduler.sample_time(B, device=x0.device, dtype=x0.dtype)
        x_t = self.noise_scheduler.add_noise(x0, t)
        prediction = self.model(x_t, t, condition=condition, **model_kwargs)
        x0_pred = self._to_x0(prediction, x_t, t)
        loss = weight * (x0_pred - x0) ** 2
        w = self.noise_scheduler.loss_weight(t)
        w = apply_loss_weight(w, x0.ndim)
        w = _maybe_promote_to_mesh(w, loss)
        loss = w * loss
        return self._reduce(loss)


class FlowMatchingLoss:
    r"""
    Flow matching loss with a flow (velocity) regression target.

    Given clean data :math:`\mathbf{x}_0`, the noisy state
    :math:`\mathbf{x}_t` produced by the noise scheduler, and a
    user-provided ``x0_to_flow_fn`` that computes the flow target
    :math:`\mathbf{v}(\mathbf{x}_0, \mathbf{x}_t, t)`, the loss is:

    .. math::
        \mathcal{L} = \mathbb{E}_{t}
        \left[ w(t) \left\| \hat{\mathbf{v}}(\mathbf{x}_t, t)
        - \mathbf{v}(\mathbf{x}_0, \mathbf{x}_t, t) \right\|^2 \right]

    When paired with a
    :class:`~physicsnemo.diffusion.noise_schedulers.RectifiedFlowNoiseScheduler`
    and its ``x0_to_flow`` method as the callback, the target is the
    standard rectified-flow form
    :math:`\mathbf{v} = \boldsymbol{\epsilon} - \mathbf{x}_0` with uniform
    time sampling and unit loss weight. Any noise scheduler works when
    paired with a matching ``x0_to_flow_fn``;
    :class:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler`
    subclasses provide one ready-made.

    .. note::

        For the rectified-flow schedule, the x0-to-flow conversion
        (which also computes the flow target) is singular at
        :math:`t = 0`, and the epsilon/score conversions at :math:`t = 1`;
        restrict sampled times accordingly (e.g.
        ``RectifiedFlowNoiseScheduler(t_min=1e-3)`` for x0 prediction).

    .. warning::

        For domain-parallel training where ``x0`` is a ``ShardTensor``,
        you **must** wrap the scheduler with
        :class:`~physicsnemo.diffusion.noise_schedulers.DomainParallelNoiseScheduler`;
        a plain scheduler with sharded data raises a ``ValueError``.

    .. note::

        Flow matching times live in ``[0, 1]``, while DDPM++/DiT-style
        timestep embedders expect inputs spanning roughly ``[0, 1000]``.
        Scale ``t`` inside your model (e.g. multiply by ``999.0`` before
        the time embedding), identically at training and sampling time — a
        mismatched scale silently shifts the embedding out of its trained
        range.

    Parameters
    ----------
    model : DiffusionModel
        Diffusion model to train. ``prediction_type`` sets how the loss
        interprets the output. Must follow the
        :class:`~physicsnemo.diffusion.DiffusionModel` protocol.
    noise_scheduler : NoiseScheduler
        Noise scheduler implementing the
        :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`
        protocol, used for time sampling, noise injection, and loss
        weighting. Typically a
        :class:`~physicsnemo.diffusion.noise_schedulers.RectifiedFlowNoiseScheduler`.
    prediction_type : PredictorType, default="flow"
        The prediction the model outputs. Use ``"flow"`` when the
        model directly predicts the flow (velocity) (the most common case
        for flow matching). Use ``"x0"``, ``"score"``, or ``"epsilon"``
        when the model predicts clean data, the score, or the noise; the
        matching ``*_to_flow_fn`` callback then converts the prediction to
        a flow estimate.
    x0_to_flow_fn : Callable[[Tensor, Tensor, Tensor], Tensor]
        Callback ``(x0, x_t, t) -> flow`` computing the flow from clean
        data. Always required: it computes the regression target, and also
        converts the prediction when ``prediction_type="x0"``. For
        :class:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler`
        subclasses, pass
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.x0_to_flow`.
    score_to_flow_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional
        Callback to convert a score prediction to a flow estimate. Required
        when ``prediction_type="score"``, e.g.
        :meth:`~physicsnemo.diffusion.noise_schedulers.LinearGaussianNoiseScheduler.score_to_flow`.
    epsilon_to_flow_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional
        Callback to convert an epsilon (noise) prediction to a flow
        estimate. Required when ``prediction_type="epsilon"``.
    reduction : Literal["none", "mean", "sum"], default="mean"
        Reduction to apply to the output: ``"none"`` returns the
        per-element loss, ``"mean"`` returns the mean over all elements,
        ``"sum"`` returns the sum over all elements. For irregular data
        (padded batches, point clouds, graphs), use ``"none"`` and apply
        your own reduction, such as a masked mean.

    Raises
    ------
    ValueError
        If ``prediction_type`` is not ``"flow"``, ``"x0"``, ``"score"``,
        or ``"epsilon"``.
    ValueError
        If ``x0_to_flow_fn`` is missing, or the callback required by
        ``prediction_type`` is missing.

    Examples
    --------
    Train a flow-predictor with the rectified flow schedule:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.noise_schedulers import (
    ...     RectifiedFlowNoiseScheduler,
    ... )
    >>> from physicsnemo.diffusion.metrics.losses import FlowMatchingLoss
    >>>
    >>> class FlowModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> model = FlowModel()
    >>> scheduler = RectifiedFlowNoiseScheduler()
    >>> loss_fn = FlowMatchingLoss(
    ...     model, scheduler, x0_to_flow_fn=scheduler.x0_to_flow
    ... )
    >>> x0 = torch.randn(4, 3, 8, 8)
    >>> loss = loss_fn(x0)
    >>> loss.shape
    torch.Size([])

    See :class:`MSEDSMLoss` for more usage patterns (conditional training,
    pre-sampled times, switching the prediction type); they apply here
    unchanged.
    """

    def __init__(
        self,
        model: DiffusionModel,
        noise_scheduler: NoiseScheduler,
        prediction_type: PredictorType = "flow",
        x0_to_flow_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        score_to_flow_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        epsilon_to_flow_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        self.model = model
        self.noise_scheduler = noise_scheduler

        if x0_to_flow_fn is None:
            raise ValueError(
                "x0_to_flow_fn must be provided: it computes the flow target "
                "from clean data. LinearGaussianNoiseScheduler subclasses "
                "provide it as noise_scheduler.x0_to_flow."
            )
        self._x0_to_flow = x0_to_flow_fn

        match prediction_type:
            case "flow":
                self._to_flow = lambda prediction, x_t, t: prediction
            case "x0":
                self._to_flow = x0_to_flow_fn
            case "score":
                if score_to_flow_fn is None:
                    raise ValueError(
                        "score_to_flow_fn must be provided when prediction_type='score'."
                    )
                self._to_flow = score_to_flow_fn
            case "epsilon":
                if epsilon_to_flow_fn is None:
                    raise ValueError(
                        "epsilon_to_flow_fn must be provided when "
                        "prediction_type='epsilon'."
                    )
                self._to_flow = epsilon_to_flow_fn
            case _:
                raise ValueError(
                    f"prediction_type must be 'flow', 'x0', 'score', or "
                    f"'epsilon', got '{prediction_type}'."
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
        t: Float[Tensor, " B"] | None = None,
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        **model_kwargs: Any,
    ) -> Float[Tensor, " B *dims"] | Float[Tensor, ""]:
        r"""
        Compute the flow matching loss.

        Parameters
        ----------
        x0 : Tensor
            Clean data of shape :math:`(B, *)` where :math:`B` is the batch
            size and :math:`*` denotes any number of extra dimensions.
        t : Tensor or None, optional, default=None
            Pre-sampled diffusion times of shape :math:`(B,)`. When ``None``
            (the default), the loss samples times internally via
            :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.sample_time`.
            Pass explicit times for diagnostics such as per-time-bin loss
            tracking.
        condition : Tensor, TensorDict, or None, optional, default=None
            Conditioning information passed to the model. See
            :class:`~physicsnemo.diffusion.DiffusionModel` for details.
        **model_kwargs : Any
            Extra keyword arguments forwarded to the model

        Returns
        -------
        Tensor
            If ``reduction="none"``, the per-element weighted loss with same
            shape :math:`(B, *)` as ``x0``. If ``reduction="mean"``, or
            ``reduction="sum"``, a scalar tensor.
        """
        if not torch.compiler.is_compiling():
            _check_domain_parallel_scheduler(x0, self.noise_scheduler)
        B = x0.shape[0]
        if t is None:
            t = self.noise_scheduler.sample_time(B, device=x0.device, dtype=x0.dtype)
        x_t = self.noise_scheduler.add_noise(x0, t)
        prediction = self.model(x_t, t, condition=condition, **model_kwargs)
        flow_pred = self._to_flow(prediction, x_t, t)
        flow_target = self._x0_to_flow(x0, x_t, t)
        loss = (flow_pred - flow_target) ** 2
        w = self.noise_scheduler.loss_weight(t)
        w = apply_loss_weight(w, x0.ndim)
        w = _maybe_promote_to_mesh(w, loss)
        loss = w * loss
        return self._reduce(loss)


class WeightedFlowMatchingLoss:
    r"""
    Weighted flow matching loss.

    Identical to :class:`FlowMatchingLoss` but adds a
    ``weight`` argument that multiplies the per-element squared error.

    .. math::
        \mathcal{L} = \mathbb{E}_{t}
        \left[ w(t) \left\| \mathbf{m} \odot
        \left(\hat{\mathbf{v}}(\mathbf{x}_t, t)
        - \mathbf{v}(\mathbf{x}_0, \mathbf{x}_t, t)\right) \right\|^2 \right]

    where :math:`\mathbf{m}` is the element-wise weight (e.g., a binary
    mask over spatial regions, channels, or padded elements of
    variable-size point clouds or graphs). Note that ``weight`` differs
    from the time-dependent loss weight :math:`w(t)` of the noise
    scheduler. See :class:`FlowMatchingLoss` for prediction types,
    signatures, singularities, and the time-scaling caveat.

    .. warning::

        For domain-parallel training where ``x0`` is a ``DTensor``
        (e.g., a :class:`~physicsnemo.domain_parallel.ShardTensor`),
        ``weight`` must be a ``DTensor`` on the same device mesh; a plain
        tensor ``weight`` raises a ``ValueError``.

    Parameters
    ----------
    model : DiffusionModel
        Diffusion model to train. ``prediction_type`` sets how the loss
        interprets the output. Must follow the
        :class:`~physicsnemo.diffusion.DiffusionModel` protocol.
    noise_scheduler : NoiseScheduler
        Noise scheduler implementing the
        :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`
        protocol. See :class:`FlowMatchingLoss`.
    prediction_type : PredictorType, default="flow"
        The prediction the model outputs. See :class:`FlowMatchingLoss`.
    x0_to_flow_fn : callable
        Callback ``(x0, x_t, t) -> flow`` computing the flow from clean
        data. Always required: it computes the regression target. See
        :class:`FlowMatchingLoss`.
    score_to_flow_fn : callable, optional
        Callback to convert a score prediction to a flow estimate. Required
        when ``prediction_type="score"``.
    epsilon_to_flow_fn : callable, optional
        Callback to convert an epsilon (noise) prediction to a flow
        estimate. Required when ``prediction_type="epsilon"``.
    reduction : Literal["none", "mean", "sum"], default="mean"
        Reduction to apply to the output: ``"none"`` returns the
        per-element loss, ``"mean"`` returns the mean over all elements,
        ``"sum"`` returns the sum over all elements. For irregular data
        (padded batches, point clouds, graphs), use ``"none"`` and apply
        your own reduction, such as a masked mean.

    Raises
    ------
    ValueError
        If ``prediction_type`` is not ``"flow"``, ``"x0"``, ``"score"``,
        or ``"epsilon"``.
    ValueError
        If ``x0_to_flow_fn`` is missing, or the callback required by
        ``prediction_type`` is missing.

    Examples
    --------
    Apply a binary mask so the loss covers only unmasked elements
    (e.g., padded atoms in a batch of variable-size point clouds):

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.noise_schedulers import (
    ...     RectifiedFlowNoiseScheduler,
    ... )
    >>> from physicsnemo.diffusion.metrics.losses import WeightedFlowMatchingLoss
    >>>
    >>> class FlowModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> model = FlowModel()
    >>> scheduler = RectifiedFlowNoiseScheduler()
    >>> loss_fn = WeightedFlowMatchingLoss(
    ...     model, scheduler, x0_to_flow_fn=scheduler.x0_to_flow
    ... )
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
        prediction_type: PredictorType = "flow",
        x0_to_flow_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        score_to_flow_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        epsilon_to_flow_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        self.model = model
        self.noise_scheduler = noise_scheduler

        if x0_to_flow_fn is None:
            raise ValueError(
                "x0_to_flow_fn must be provided: it computes the flow target "
                "from clean data. LinearGaussianNoiseScheduler subclasses "
                "provide it as noise_scheduler.x0_to_flow."
            )
        self._x0_to_flow = x0_to_flow_fn

        match prediction_type:
            case "flow":
                self._to_flow = lambda prediction, x_t, t: prediction
            case "x0":
                self._to_flow = x0_to_flow_fn
            case "score":
                if score_to_flow_fn is None:
                    raise ValueError(
                        "score_to_flow_fn must be provided when prediction_type='score'."
                    )
                self._to_flow = score_to_flow_fn
            case "epsilon":
                if epsilon_to_flow_fn is None:
                    raise ValueError(
                        "epsilon_to_flow_fn must be provided when "
                        "prediction_type='epsilon'."
                    )
                self._to_flow = epsilon_to_flow_fn
            case _:
                raise ValueError(
                    f"prediction_type must be 'flow', 'x0', 'score', or "
                    f"'epsilon', got '{prediction_type}'."
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
        t: Float[Tensor, " B"] | None = None,
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        **model_kwargs: Any,
    ) -> Float[Tensor, " B *dims"] | Float[Tensor, ""]:
        r"""
        Compute the weighted flow matching loss.

        Parameters
        ----------
        x0 : Tensor
            Clean data of shape :math:`(B, *)` where :math:`B` is the batch
            size and :math:`*` denotes any number of extra dimensions.
        weight : Tensor
            Per-element weight with the same shape as ``x0`` (0/1 for
            binary masking). When ``x0`` is a ``DTensor``, ``weight`` must
            be a ``DTensor`` on the same device mesh.
        t : Tensor or None, optional, default=None
            Pre-sampled diffusion times of shape :math:`(B,)`. When ``None``
            (the default), the loss samples times internally via
            :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.sample_time`.
            Pass explicit times for diagnostics such as per-time-bin loss
            tracking.
        condition : Tensor, TensorDict, or None, optional, default=None
            Conditioning information passed to the model. See
            :class:`~physicsnemo.diffusion.DiffusionModel` for details.
        **model_kwargs : Any
            Extra keyword arguments forwarded to the model

        Returns
        -------
        Tensor
            If ``reduction="none"``, the per-element weighted loss with same
            shape :math:`(B, *)` as ``x0``. If ``reduction="mean"``, or
            ``reduction="sum"``, a scalar tensor.
        """
        if not torch.compiler.is_compiling():
            _check_domain_parallel_scheduler(x0, self.noise_scheduler)
            _check_weight_mesh(weight, x0)
        B = x0.shape[0]
        if t is None:
            t = self.noise_scheduler.sample_time(B, device=x0.device, dtype=x0.dtype)
        x_t = self.noise_scheduler.add_noise(x0, t)
        prediction = self.model(x_t, t, condition=condition, **model_kwargs)
        flow_pred = self._to_flow(prediction, x_t, t)
        flow_target = self._x0_to_flow(x0, x_t, t)
        loss = weight * (flow_pred - flow_target) ** 2
        w = self.noise_scheduler.loss_weight(t)
        w = apply_loss_weight(w, x0.ndim)
        w = _maybe_promote_to_mesh(w, loss)
        loss = w * loss
        return self._reduce(loss)
