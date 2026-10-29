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

from typing import Any, Literal, Protocol, runtime_checkable

import torch
from jaxtyping import Float, Shaped
from tensordict import TensorClass, TensorDict


class DiffusionExtraTimes(TensorClass):
    r"""Named, batched diffusion times accompanying the primary time tensor.

    This ``tensordict.TensorClass`` container carries named extra times for
    diffusion utilities. A :class:`DiffusionModel` is one possible consumer.

    Most diffusion and flow-matching models take a single diffusion time
    tensor. Models that need more than one diffusion time during training
    or sampling can take ``(t, extra_times)`` instead. The required primary
    time ``t`` stays separate from the optional named fields in this
    container.

    Every supplied time tensor has shape :math:`(B,)` and uses the same
    scheduler coordinates as the primary time. Set ``batch_size=[B]`` to
    match the input batch. For example, utilities reserve ``r`` for the
    destination diffusion time in MeanFlow. Each field is independently
    optional and defaults to ``None``.

    .. note::

        Subclass this container to add time fields for a custom use. Do not
        reuse reserved names for different meanings. The framework does not
        infer the meaning of custom fields; the consuming utility must define
        and handle them.

    Parameters
    ----------
    r : torch.Tensor or None, optional, default=None
        Destination diffusion time for MeanFlow, with shape :math:`(B,)`.
        Leave unset when the model does not use this field.
    batch_size : sequence of int
        Set to ``[B]``, matching the primary time tensor.

    Examples
    --------
    Pair a primary time with the reserved destination time for a MeanFlow
    consumer. The batch sizes of both tensors and the container agree.
    This example shows the tuple passed to a compatible utility:

    >>> import torch
    >>> from physicsnemo.diffusion import DiffusionExtraTimes
    >>> t = torch.ones(2)
    >>> # The reserved r field holds the destination time.
    >>> times = (t, DiffusionExtraTimes(r=torch.zeros(2), batch_size=[2]))
    >>> t, extra_times = times
    >>> t.shape, extra_times.r.shape
    (torch.Size([2]), torch.Size([2]))

    A subclass can add independent optional times for a custom consumer.
    The consumer must interpret these fields and can accept either field
    alone or both together:

    >>> from jaxtyping import Float
    >>> class CustomTimes(DiffusionExtraTimes):
    ...     # The consuming utility defines these custom times.
    ...     r1: Float[torch.Tensor, " B"] | None = None
    ...     r2: Float[torch.Tensor, " B"] | None = None
    >>> first = CustomTimes(r1=torch.zeros(2), batch_size=[2])
    >>> second = CustomTimes(r2=torch.ones(2), batch_size=[2])
    >>> first.r is None, first.r2 is None, second.r1 is None
    (True, True, True)
    >>> first.r1.shape, second.r2.shape
    (torch.Size([2]), torch.Size([2]))
    >>> both = CustomTimes(
    ...     r1=torch.zeros(2), r2=torch.ones(2), batch_size=[2]
    ... )
    >>> both.r1.shape, both.r2.shape
    (torch.Size([2]), torch.Size([2]))
    """

    r: Float[torch.Tensor, " B"] | None = None


class DiffusionExtraOutputs(TensorClass):
    r"""Named, batched auxiliary outputs accompanying a model prediction.

    This ``tensordict.TensorClass`` container carries named extra outputs for
    diffusion utilities. A :class:`DiffusionModel` is one possible producer.

    Most diffusion and flow-matching models return a single tensor that
    predicts clean data, noise, a score, or a velocity, as described by
    :data:`PredictorType`. Models that produce auxiliary outputs for
    training or sampling can return ``(prediction, extra_outputs)`` instead.
    For example, sCM and EDM2 losses use a learned log-scale alongside the
    prediction for adaptive loss weighting. The required prediction stays
    separate from the optional named fields in this container.

    All supplied output tensors share the prediction's batch size
    :math:`B`, but their remaining dimensions can differ. Set
    ``batch_size=[B]`` to describe that shared batch dimension. For example,
    the reserved field ``logvar`` provides a learned log-scale for adaptive
    loss weighting in sCM and EDM2. It does not represent diffusion
    transition variance. Each field is independently optional and defaults
    to ``None``.

    .. note::

        Subclass this container to add outputs for a custom use. Do not reuse
        reserved names for different meanings. The framework does not infer
        the meaning of custom fields; the consuming utility must define and
        handle them.

    Parameters
    ----------
    logvar : torch.Tensor or None, optional, default=None
        Learned log-scale for adaptive loss weighting in sCM and EDM2, with
        shape :math:`(B,)`. Leave unset when the model does not use it.
    batch_size : sequence of int
        Set to ``[B]``, matching the prediction's batch dimension.

    Examples
    --------
    Pair a required prediction with a learned log-scale for a compatible
    loss. The prediction retains its own shape while the container shares
    its batch dimension:

    >>> import torch
    >>> from physicsnemo.diffusion import DiffusionExtraOutputs
    >>> prediction = torch.zeros(2, 3, 5)
    >>> # The reserved logvar field supplies loss weighting metadata.
    >>> output = (
    ...     prediction,
    ...     DiffusionExtraOutputs(logvar=torch.zeros(2), batch_size=[2]),
    ... )
    >>> prediction, extra_outputs = output
    >>> prediction.shape, extra_outputs.logvar.shape
    (torch.Size([2, 3, 5]), torch.Size([2]))

    A subclass can hold independent optional outputs with different feature
    shapes. The consumer decides how to use each field, so these names carry
    no framework-defined meaning:

    >>> from jaxtyping import Float
    >>> class CustomOutputs(DiffusionExtraOutputs):
    ...     # The consuming utility defines these custom outputs.
    ...     features: Float[torch.Tensor, " B C H W"] | None = None
    ...     confidence: Float[torch.Tensor, " B K"] | None = None
    >>> extra = CustomOutputs(
    ...     features=torch.zeros(2, 4, 5, 7),
    ...     confidence=torch.ones(2, 3),
    ...     batch_size=[2],
    ... )
    >>> extra.features.shape, extra.confidence.shape
    (torch.Size([2, 4, 5, 7]), torch.Size([2, 3]))
    >>> extra.logvar is None
    True
    >>> features_only = CustomOutputs(features=extra.features, batch_size=[2])
    >>> features_only.confidence is None
    True
    """

    logvar: Float[torch.Tensor, " B"] | None = None


PredictorType = Literal["x0", "score", "epsilon", "flow"]
"""Type of prediction produced by a diffusion or flow-matching model.

One of:

- ``"x0"``: clean-data prediction :math:`\\hat{\\mathbf{x}}_0`.
- ``"score"``: score :math:`\\nabla_{\\mathbf{x}_t} \\log p(\\mathbf{x}_t)`.
- ``"epsilon"``: noise prediction :math:`\\hat{\\boldsymbol{\\epsilon}}`.
- ``"flow"``: flow matching velocity :math:`\\hat{\\mathbf{v}}
  = d\\mathbf{x}_t/dt`
"""


@runtime_checkable
class DiffusionModel(Protocol):
    r"""
    Protocol defining the common interface for diffusion and flow-matching
    models.

    A diffusion or flow-matching model is any neural network or function that
    transforms a noisy state ``x`` at diffusion time (or noise level) ``t``
    into a prediction. This protocol defines the standard interface that all
    such models must meet.

    Most models take a single diffusion time tensor and return a single
    prediction tensor. Specialized models can extend either side of this
    interface:

    - Pass ``(t, extra_times)`` with :class:`DiffusionExtraTimes` for models
      that accept more than one diffusion time.
    - Return ``(prediction, extra_outputs)`` with
      :class:`DiffusionExtraOutputs` for models that produce auxiliary outputs.

    The primary time and prediction remain required tensors, separate from
    the optional named fields. Use tuples with consumers that support the
    supplied fields. Prediction targets follow :data:`PredictorType`.

    The interface is **prediction-agnostic**: whether your model predicts
    clean data (:math:`\mathbf{x}_0`), noise (:math:`\epsilon`), score
    (:math:`\nabla \log p`), or velocity (:math:`\mathbf{v}`), the signature
    remains the same. This makes the protocol equally suitable for diffusion
    and flow-matching training objectives.

    The interface supports both conditional and unconditional models, for
    both diffusion and flow matching. The ``condition`` argument supports
    different conditioning scenarios:

    - **torch.Tensor**: Use when there is a single conditioning tensor
      (e.g., a class embedding or a single image).
    - **TensorDict**: Use when multiple conditioning tensors are needed,
      possibly with different shapes. The string keys can be used to provide
      semantic information about each conditioning tensor.
    - **None**: Use for unconditional generation or specific scenarios like
      classifier-free guidance where the model should ignore conditioning.

    Examples
    --------
    An ordinary model can take one time tensor and return one prediction.
    The protocol checks that the model exposes the expected call interface:

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

    A compatible model can also consume an extra time and return an auxiliary
    output. This example keeps the prediction as the first tuple element and
    uses the reserved fields with their intended meanings:

    >>> from physicsnemo.diffusion import DiffusionExtraTimes, DiffusionExtraOutputs
    >>> class ExtendedModel:
    ...     def __call__(self, x, t, condition=None, **kwargs):
    ...         t, extra_times = t
    ...         prediction = x * (t - extra_times.r)[:, None]
    ...         extra_outputs = DiffusionExtraOutputs(
    ...             logvar=torch.zeros_like(t), batch_size=[x.shape[0]]
    ...         )
    ...         # The required prediction stays first in the output tuple.
    ...         return prediction, extra_outputs
    >>> x = torch.ones(2, 3)
    >>> t = torch.ones(2)
    >>> times = (t, DiffusionExtraTimes(r=torch.zeros(2), batch_size=[2]))
    >>> prediction, extra_outputs = ExtendedModel()(x, times)
    >>> prediction.shape, extra_outputs.logvar.shape
    (torch.Size([2, 3]), torch.Size([2]))
    """

    def __call__(
        self,
        x: Float[torch.Tensor, " B *dims"],
        t: Float[torch.Tensor, " B"]
        | tuple[Float[torch.Tensor, " B"], Shaped[DiffusionExtraTimes, " B"]],
        condition: Float[torch.Tensor, " B *cond_dims"] | TensorDict | None = None,
        **model_kwargs: Any,
    ) -> (
        Float[torch.Tensor, " B *dims"]
        | tuple[Float[torch.Tensor, " B *dims"], Shaped[DiffusionExtraOutputs, " B"]]
    ):
        r"""
        Forward pass of the diffusion model.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latent state of shape :math:`(B, *)` where :math:`B` is the
            batch size and :math:`*` denotes any number of additional
            dimensions (e.g., channels and spatial dimensions).
        t : torch.Tensor or tuple[torch.Tensor, DiffusionExtraTimes]
            Diffusion time for each sample, with shape :math:`(B,)`.
            Pass ``(t, extra_times)`` to supply extra named times in
            :class:`DiffusionExtraTimes`, each with shape :math:`(B,)`.
        condition : torch.Tensor, TensorDict, or None, optional, default=None
            Conditioning information for the model. If a Tensor or a TensorDict
            is passed, it should have batch size :math:`B` matching that of
            ``x``. Pass ``None`` for an unconditional model.

        **model_kwargs : Any
            Additional keyword arguments specific to the model implementation.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, DiffusionExtraOutputs]
            Prediction with the shape of ``x``, optionally paired with a
            :class:`DiffusionExtraOutputs`. Every call returns a prediction
            tensor; auxiliary tensors share its batch dimension.
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
    :data:`PredictorType` and its use. Any callable that implements this
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
    **Example 1:** A conditional model needs ``y`` to form its clean-data
    estimate. Bind that condition with ``functools.partial`` so sampling code
    can call the resulting predictor with only ``x`` and ``t``:

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

    **Example 2:** A scheduler may need a score rather than the clean-data
    estimate produced above. For the illustrative schedule
    :math:`\sigma(t) = t` and :math:`\alpha(t) = 1`, convert that estimate
    with the score identity while keeping the same predictor call signature:

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

    **Example 3:** A model can return :class:`DiffusionExtraOutputs` for
    training, such as a ``logvar`` field. Wrap the model to select its
    prediction tensor for sampling. Standard denoiser factories require a
    predictor that accepts one diffusion time tensor:

    >>> from physicsnemo.diffusion import DiffusionExtraOutputs
    >>> def model_with_extras(x, t, condition=None):
    ...     return x / 2, DiffusionExtraOutputs(
    ...         logvar=torch.zeros(x.shape[0]), batch_size=[x.shape[0]]
    ...     )
    >>> def prediction_only(x, t):
    ...     prediction, _ = model_with_extras(x, t)
    ...     return prediction
    >>> prediction_only(torch.ones(2, 3), torch.ones(2)).shape
    torch.Size([2, 3])
    >>> isinstance(prediction_only, Predictor)
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
    consumed by a :class:`~physicsnemo.diffusion.samplers.Solver`.
    For continuous-time methods this is typically the right-hand side of the
    ODE/SDE, but the interface is generic and can support other sampling
    methods as well.

    This is the interface used by
    :class:`~physicsnemo.diffusion.samplers.Solver` classes and the
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
       :class:`~physicsnemo.diffusion.samplers.Solver`

    See Also
    --------
    :class:`Predictor` : The interface for raw predictions.
    :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.get_denoiser` :
        Factory to convert a predictor into a denoiser.
    :func:`~physicsnemo.diffusion.samplers.sample` : The sampling function
        that uses this denoiser interface.

    Examples
    --------
    This example builds a denoiser from a clean-data predictor for the
    illustrative schedule :math:`\sigma(t)=t` and :math:`\alpha(t)=1`.
    The returned tensor is the update term consumed by a solver. It uses one
    time tensor, as required by the standard predictor and denoiser interfaces:

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
