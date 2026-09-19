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

"""Patch-based diffusion and flow-matching losses for multi-diffusion training."""

from functools import lru_cache
from typing import Any, Callable, Literal, Tuple

import torch
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

from physicsnemo.diffusion.base import PredictorType
from physicsnemo.diffusion.multi_diffusion.models import MultiDiffusionModel2D
from physicsnemo.diffusion.noise_schedulers import NoiseScheduler
from physicsnemo.diffusion.utils.utils import _unwrap_module, apply_loss_weight


class _CompiledPatchX:
    """Cached ``torch.compile``-d wrapper around
    :meth:`~MultiDiffusionModel2D.patch_x`.

    A separate compiled graph is cached per unique tensor signature
    (shape, dtype, device) so that calls with different shapes do not
    trigger recompilation.
    """

    def __init__(self, model: MultiDiffusionModel2D, *, maxsize: int = 8) -> None:
        self._model = model
        self._cache = lru_cache(maxsize=maxsize)(self._compile_for_sig)

    @staticmethod
    def _sig(t: Tensor) -> Tuple:
        return (tuple(t.shape), t.dtype, t.device)

    @staticmethod
    def _patch_x(model: MultiDiffusionModel2D, x: Tensor) -> Tensor:
        return model.patch_x(x)

    def _compile_for_sig(self, sig: Tuple) -> Callable:
        return torch.compile(self._patch_x)

    def __call__(self, x: Tensor) -> Tensor:
        fn = self._cache(self._sig(x))
        return fn(self._model, x)


class MultiDiffusionMSEDSMLoss:
    r"""Patch-based MSE denoising score matching loss for multi-diffusion
    training.

    As the patch-based counterpart of
    :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss`, it operates on a
    :class:`~physicsnemo.diffusion.multi_diffusion.MultiDiffusionModel2D`
    wrapper and computes the denoising score matching objective independently
    on each patch. A separate diffusion time is sampled per patch, giving
    :math:`P \times B` independent noise levels per training step.

    All training functionality is centered around a **noise scheduler** that
    must implement the
    :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler` protocol.
    At each training step the noise scheduler provides:

    - **Time sampling** via :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.sample_time`: draws
      random diffusion times :math:`t` — one per patch.
    - **Noise injection** via :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.add_noise`: produces
      the noisy state :math:`\mathbf{x}_t` from clean data
      :math:`\mathbf{x}_0`.
    - **Loss weighting** via :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.loss_weight`: returns
      the per-sample weight :math:`w(t)`.  Weights may be scalar
      :math:`(N,)` or per-channel :math:`(N, C)` when the scheduler uses
      per-channel ``sigma_data`` (see
      :class:`~physicsnemo.diffusion.noise_schedulers.EDMNoiseScheduler`).

    The model **must** have a random patching strategy configured via
    :meth:`~MultiDiffusionModel2D.set_random_patching` before using this
    loss.

    .. note::

        By default, each call to the loss **re-draws random patch positions**
        via :meth:`~MultiDiffusionModel2D.reset_patch_indices`. This ensures
        that every training step uses a fresh set of patches. Pass
        ``reset_patch_indices=False`` to the call to disable this behaviour
        (e.g., when patch positions are managed externally).

    .. note::

        If the model has positional embeddings configured, the wrapped model
        must accept a ``TensorDict`` condition containing a
        ``"positional_embedding"`` key.

    For details on prediction types and ``score_to_x0_fn``, see
    :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss`.

    Parameters
    ----------
    model : MultiDiffusionModel2D
        Multi-diffusion model wrapper with random patching configured.
    noise_scheduler : NoiseScheduler
        Noise scheduler implementing the
        :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`
        protocol, providing the methods:
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.sample_time`,
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.add_noise`, and
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.loss_weight`.
    prediction_type : PredictorType, default="x0"
        Type of prediction the model outputs.
    score_to_x0_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional
        Callback to convert a score prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="score"``.
    epsilon_to_x0_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional
        Callback to convert an epsilon (noise) prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="epsilon"``.
    flow_to_x0_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional
        Callback to convert a flow (velocity) prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="flow"``.
    reduction : Literal["none", "mean", "sum"], default="mean"
        Reduction applied to the output.

    Examples
    --------
    **Example 1:** Unconditional model with EDM schedule:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionModel2D,
    ...     MultiDiffusionMSEDSMLoss,
    ... )
    >>>
    >>> class UnconditionalModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> md_model = MultiDiffusionModel2D(
    ...     model=UnconditionalModel(),
    ...     global_spatial_shape=(16, 16),
    ... )
    >>> md_model.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> loss_fn = MultiDiffusionMSEDSMLoss(md_model, EDMNoiseScheduler())
    >>> x0 = torch.randn(2, 3, 16, 16)
    >>> loss = loss_fn(x0)
    >>> loss.shape
    torch.Size([])

    **Example 2:** Conditional model with score prediction and no reduction:

    >>> from tensordict import TensorDict
    >>>
    >>> class ConditionalModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(6, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(torch.cat([x, condition["image"]], dim=1))
    >>>
    >>> cond_md_model = MultiDiffusionModel2D(
    ...     model=ConditionalModel(),
    ...     global_spatial_shape=(16, 16),
    ...     condition_patch={"image": True},
    ... )
    >>> cond_md_model.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> scheduler = EDMNoiseScheduler()
    >>> loss_fn = MultiDiffusionMSEDSMLoss(
    ...     cond_md_model, scheduler,
    ...     prediction_type="score",
    ...     score_to_x0_fn=scheduler.score_to_x0,
    ...     reduction="none",
    ... )
    >>> cond = TensorDict({"image": torch.randn(2, 3, 16, 16)}, batch_size=[2])
    >>> loss = loss_fn(x0, condition=cond)
    >>> loss.shape
    torch.Size([8, 3, 8, 8])

    **Example 3:** Conditional model with learnable positional embeddings:

    >>> class PosEmbdModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         # 12 input channels: 3 (state) + 3 (image) + 6 (positional embedding)
    ...         self.net = torch.nn.Conv2d(12, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         img = condition["image"]
    ...         # The wrapped model is designed to extract the positional embeddings
    ...         # from the condition TensorDict
    ...         pe = condition["positional_embedding"]
    ...         return self.net(torch.cat([x, img, pe], dim=1))
    >>>
    >>> pe_md_model = MultiDiffusionModel2D(
    ...     model=PosEmbdModel(),
    ...     global_spatial_shape=(16, 16),
    ...     positional_embedding="learnable",
    ...     channels_positional_embedding=6,
    ...     condition_patch={"image": True},
    ... )
    >>> pe_md_model.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> loss_fn = MultiDiffusionMSEDSMLoss(pe_md_model, EDMNoiseScheduler())
    >>> cond = TensorDict({"image": torch.randn(2, 3, 16, 16)}, batch_size=[2])
    >>> loss = loss_fn(x0, condition=cond)
    >>> loss.shape
    torch.Size([])

    See Also
    --------
    :class:`~physicsnemo.diffusion.metrics.losses.MSEDSMLoss` :
        Non-patched version of this loss.
    :class:`MultiDiffusionWeightedMSEDSMLoss` :
        Weighted variant that supports per-element masking.
    """

    def __init__(
        self,
        model: MultiDiffusionModel2D,
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
        self._md_model = _unwrap_module(model, MultiDiffusionModel2D)
        self.noise_scheduler = noise_scheduler
        self._compiled_patch_x = _CompiledPatchX(self._md_model)

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
        x0: Float[Tensor, "B C H W"],
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        reset_patch_indices: bool = True,
        t: Float[Tensor, " PB"] | None = None,
        **model_kwargs: Any,
    ) -> Float[Tensor, "P_times_B C Hp Wp"] | Float[Tensor, ""]:
        r"""Compute the multi-diffusion denoising score matching loss.

        Parameters
        ----------
        x0 : Tensor
            Clean data of shape :math:`(B, C, H, W)` at global resolution.
        condition : Tensor, TensorDict, or None, optional, default=None
            Conditioning information at global resolution (batch size
            :math:`B`).
        reset_patch_indices : bool, default=True
            If ``True``, re-draw random patch positions before computing
            the loss. Set to ``False`` when patch positions are managed
            externally.
        t : Tensor or None, optional, default=None
            Pre-sampled diffusion time values of shape :math:`(P \times B,)`
            — one value per patch.  When ``None`` (the default), times are
            sampled internally via
            :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.sample_time`.
            Passing explicit times is useful when the caller needs access
            to the sampled values for diagnostics (e.g., per-sigma-bin
            loss tracking).
        **model_kwargs : Any
            Additional keyword arguments forwarded to the model.

        Returns
        -------
        Tensor
            If ``reduction="none"``, the per-element weighted loss of shape
            :math:`(P \times B, C, H_p, W_p)`. Otherwise a scalar tensor.
        """
        if reset_patch_indices:
            self._md_model.reset_patch_indices()

        # Patch x0 and sample per-patch noise
        x0_patched = self._compiled_patch_x(x0)  # (P*B, C, Hp, Wp)
        PB = x0_patched.shape[0]
        if t is None:
            t = self.noise_scheduler.sample_time(PB, device=x0.device, dtype=x0.dtype)
        x_t = self.noise_scheduler.add_noise(x0_patched, t)

        # Forward with pre-patched x and t
        prediction = self.model(
            x_t,
            t,
            condition=condition,
            x_is_patched=True,
            t_is_patched=True,
            **model_kwargs,
        )

        x0_pred = self._to_x0(prediction, x_t, t)

        loss = (x0_pred - x0_patched) ** 2
        w = self.noise_scheduler.loss_weight(t)
        loss = apply_loss_weight(w, x0_patched.ndim) * loss
        return self._reduce(loss)


class MultiDiffusionWeightedMSEDSMLoss:
    r"""Weighted patch-based MSE denoising score matching loss.

    As the patch-based counterpart of
    :class:`~physicsnemo.diffusion.metrics.losses.WeightedMSEDSMLoss`, this
    class extends :class:`MultiDiffusionMSEDSMLoss` with a ``weight`` tensor
    that multiplies the per-element squared error.

    The ``weight`` tensor is provided at global resolution and is
    automatically patched alongside :math:`\mathbf{x}_0`.

    .. math::
        \mathcal{L} = \mathbb{E}_{t, \boldsymbol{\epsilon}}
        \left[ w(t) \left\| \mathbf{m} \odot
        \left(\hat{\mathbf{x}}_0(\mathbf{x}_t, t)
        - \mathbf{x}_0\right) \right\|^2 \right]

    The noise scheduler's
    :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.loss_weight`
    may return scalar :math:`(N,)` or per-channel :math:`(N, C)` weights
    when the scheduler uses per-channel ``sigma_data`` (see
    :class:`~physicsnemo.diffusion.noise_schedulers.EDMNoiseScheduler`).

    The model **must** have a random patching strategy configured via
    :meth:`~MultiDiffusionModel2D.set_random_patching` before using this
    loss.

    .. note::

        By default, each call to the loss **re-draws random patch positions**
        via :meth:`~MultiDiffusionModel2D.reset_patch_indices`. Pass
        ``reset_patch_indices=False`` to the call to disable this.

    .. note::

        If the model has positional embeddings configured, the wrapped model
        must accept a ``TensorDict`` condition containing a
        ``"positional_embedding"`` key.

    For additional details, see :class:`MultiDiffusionMSEDSMLoss` and
    :class:`~physicsnemo.diffusion.metrics.losses.WeightedMSEDSMLoss`.

    Parameters
    ----------
    model : MultiDiffusionModel2D
        Multi-diffusion model wrapper with random patching configured.
    noise_scheduler : NoiseScheduler
        Noise scheduler implementing the
        :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`
        protocol, providing the methods:
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.sample_time`,
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.add_noise`, and
        :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.loss_weight`.
    prediction_type : PredictorType, default="x0"
        Type of prediction the model outputs.
    score_to_x0_fn : callable, optional
        Callback to convert a score prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="score"``.
    epsilon_to_x0_fn : callable, optional
        Callback to convert an epsilon (noise) prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="epsilon"``.
    flow_to_x0_fn : callable, optional
        Callback to convert a flow (velocity) prediction to an
        :math:`\hat{\mathbf{x}}_0` estimate. Required when
        ``prediction_type="flow"``.
    reduction : {"none", "mean", "sum"}, default="mean"
        Reduction applied to the output.

    Examples
    --------
    **Example 1:** Unconditional model with a spatial mask:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionModel2D,
    ...     MultiDiffusionWeightedMSEDSMLoss,
    ... )
    >>>
    >>> class UnconditionalModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> md_model = MultiDiffusionModel2D(
    ...     model=UnconditionalModel(),
    ...     global_spatial_shape=(16, 16),
    ... )
    >>> md_model.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> loss_fn = MultiDiffusionWeightedMSEDSMLoss(
    ...     md_model, EDMNoiseScheduler()
    ... )
    >>> x0 = torch.randn(2, 3, 16, 16)
    >>> # Weight/mask at global resolution — patched internally by the loss
    >>> mask = torch.ones(2, 3, 16, 16)
    >>> mask[:, :, :, :8] = 0.0
    >>> loss = loss_fn(x0, weight=mask)
    >>> loss.shape
    torch.Size([])

    **Example 2:** Conditional model with learnable positional embeddings
    and a spatial mask:

    >>> from tensordict import TensorDict
    >>>
    >>> class PosEmbdModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         # 12 input channels: 3 (state) + 3 (image) + 6 (positional embedding)
    ...         self.net = torch.nn.Conv2d(12, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         img = condition["image"]
    ...         # The wrapped model is designed to extract the positional embeddings
    ...         # from the condition TensorDict
    ...         pe = condition["positional_embedding"]
    ...         return self.net(torch.cat([x, img, pe], dim=1))
    >>>
    >>> pe_md_model = MultiDiffusionModel2D(
    ...     model=PosEmbdModel(),
    ...     global_spatial_shape=(16, 16),
    ...     positional_embedding="learnable",
    ...     channels_positional_embedding=6,
    ...     condition_patch={"image": True},
    ... )
    >>> pe_md_model.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> loss_fn = MultiDiffusionWeightedMSEDSMLoss(
    ...     pe_md_model, EDMNoiseScheduler()
    ... )
    >>> cond = TensorDict({"image": torch.randn(2, 3, 16, 16)}, batch_size=[2])
    >>> # Weight/mask at global resolution — patched internally by the loss
    >>> mask = torch.ones(2, 3, 16, 16)
    >>> mask[:, :, :, :8] = 0.0
    >>> loss = loss_fn(x0, weight=mask, condition=cond)
    >>> loss.shape
    torch.Size([])

    See Also
    --------
    :class:`~physicsnemo.diffusion.metrics.losses.WeightedMSEDSMLoss` :
        Non-patched weighted loss.
    :class:`MultiDiffusionMSEDSMLoss` :
        Unweighted variant.
    """

    def __init__(
        self,
        model: MultiDiffusionModel2D,
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
        self._md_model = _unwrap_module(model, MultiDiffusionModel2D)
        self.noise_scheduler = noise_scheduler
        self._compiled_patch_x = _CompiledPatchX(self._md_model)

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
        x0: Float[Tensor, "B C H W"],
        weight: Float[Tensor, "B C H W"],
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        reset_patch_indices: bool = True,
        t: Float[Tensor, " PB"] | None = None,
        **model_kwargs: Any,
    ) -> Float[Tensor, "P_times_B C Hp Wp"] | Float[Tensor, ""]:
        r"""Compute the weighted multi-diffusion DSM loss.

        Parameters
        ----------
        x0 : Tensor
            Clean data of shape :math:`(B, C, H, W)` at global resolution.
        weight : Tensor
            Per-element weight of shape :math:`(B, C, H, W)`, same shape as
            ``x0``. Patched automatically alongside :math:`\mathbf{x}_0`.
        condition : Tensor, TensorDict, or None, optional, default=None
            Conditioning information at global resolution.
        reset_patch_indices : bool, default=True
            If ``True``, re-draw random patch positions before computing
            the loss. Set to ``False`` when patch positions are managed
            externally.
        t : Tensor or None, optional, default=None
            Pre-sampled diffusion time values of shape :math:`(P \times B,)`
            — one value per patch.  When ``None`` (the default), times are
            sampled internally via
            :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.sample_time`.
            Passing explicit times is useful when the caller needs access
            to the sampled values for diagnostics (e.g., per-sigma-bin
            loss tracking).
        **model_kwargs : Any
            Additional keyword arguments forwarded to the model.

        Returns
        -------
        Tensor
            If ``reduction="none"``, the per-element weighted loss of shape
            :math:`(P \times B, C, H_p, W_p)`. Otherwise a scalar tensor.
        """
        if not torch.compiler.is_compiling():
            if weight.shape != x0.shape:
                raise ValueError(
                    f"weight shape {tuple(weight.shape)} must match "
                    f"x0 shape {tuple(x0.shape)}."
                )

        if reset_patch_indices:
            self._md_model.reset_patch_indices()

        # Patch x0 and weight, then sample per-patch noise
        x0_patched = self._compiled_patch_x(x0)  # (P*B, C, Hp, Wp)
        weight_patched = self._compiled_patch_x(weight)  # (P*B, C, Hp, Wp)
        PB = x0_patched.shape[0]
        if t is None:
            t = self.noise_scheduler.sample_time(PB, device=x0.device, dtype=x0.dtype)
        x_t = self.noise_scheduler.add_noise(x0_patched, t)

        # Forward with pre-patched x and t
        prediction = self.model(
            x_t,
            t,
            condition=condition,
            x_is_patched=True,
            t_is_patched=True,
            **model_kwargs,
        )

        x0_pred = self._to_x0(prediction, x_t, t)

        loss = weight_patched * (x0_pred - x0_patched) ** 2
        w = self.noise_scheduler.loss_weight(t)
        loss = apply_loss_weight(w, x0_patched.ndim) * loss
        return self._reduce(loss)


class MultiDiffusionFlowMatchingLoss:
    r"""Patch-based mean-squared-error loss for flow-matching training.

    As the patch-based counterpart of
    :class:`~physicsnemo.diffusion.metrics.losses.FlowMatchingLoss`, it extracts
    random patches from clean data and computes the flow-matching loss
    independently on each patch:

    .. math::
        \mathcal{L} = \mathbb{E}_{t}
        \left[ w(t) \left\| \hat{\mathbf{v}}(\mathbf{x}_t, t)
        - \mathbf{v}(\mathbf{x}_0, \mathbf{x}_t, t) \right\|^2 \right].

    Here :math:`\mathbf{v}(\mathbf{x}_0, \mathbf{x}_t, t)` is the flow target,
    or velocity along the scheduler's interpolation path, and
    :math:`\hat{\mathbf{v}}(\mathbf{x}_t, t)` is the model estimate.

    The scheduler samples a separate time for each of the
    :math:`P \times B` patches. The model may predict flow directly or predict
    clean data, score, or noise. The corresponding callback converts these
    alternate predictions to flow.

    Configure a random patching strategy with
    :meth:`~MultiDiffusionModel2D.set_random_patching` before calling this
    loss. See :class:`MultiDiffusionMSEDSMLoss` for details about random patch
    resets, positional embeddings, conditioning, and reductions.

    Parameters
    ----------
    model : MultiDiffusionModel2D
        Multi-diffusion model wrapper with random patching configured.
    noise_scheduler : NoiseScheduler
        Noise scheduler used for time sampling, noise injection, and loss
        weighting. See the
        :class:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler`
        protocol.
    prediction_type : PredictorType, default="flow"
        Prediction produced by the model. Use ``"flow"`` for a velocity
        prediction, ``"x0"`` for clean data, ``"score"`` for a score, or
        ``"epsilon"`` for noise.
    x0_to_flow_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional, default=None
        Callback ``(x0, x_t, t) -> v`` that computes the flow target from
        clean data. You must provide this callback for every prediction type;
        the constructor raises a ``ValueError`` when passed ``None``.
    score_to_flow_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional
        Callback ``(score, x_t, t) -> v`` used when
        ``prediction_type="score"``.
    epsilon_to_flow_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional
        Callback ``(epsilon, x_t, t) -> v`` used when
        ``prediction_type="epsilon"``.
    reduction : Literal["none", "mean", "sum"], default="mean"
        Reduction applied to the per-element loss.

    Examples
    --------
    Train a conditional flow-matching model with positional embeddings:

    >>> import torch
    >>> from tensordict import TensorDict
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionFlowMatchingLoss,
    ...     MultiDiffusionModel2D,
    ... )
    >>> from physicsnemo.diffusion.noise_schedulers import (
    ...     RectifiedFlowNoiseScheduler,
    ... )
    >>>
    >>> class PosEmbdFlowModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         # 12 input channels: 3 state + 3 image + 6 positional embedding
    ...         self.net = torch.nn.Conv2d(12, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         image = condition["image"]
    ...         pos_emb = condition["positional_embedding"]
    ...         return self.net(torch.cat([x, image, pos_emb], dim=1))
    >>>
    >>> model = MultiDiffusionModel2D(
    ...     model=PosEmbdFlowModel(),
    ...     global_spatial_shape=(16, 16),
    ...     positional_embedding="learnable",
    ...     channels_positional_embedding=6,
    ...     condition_patch={"image": True},
    ... )
    >>> model.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> scheduler = RectifiedFlowNoiseScheduler()
    >>> loss_fn = MultiDiffusionFlowMatchingLoss(
    ...     model, scheduler, x0_to_flow_fn=scheduler.x0_to_flow
    ... )
    >>> x0 = torch.randn(2, 3, 16, 16)
    >>> condition = TensorDict(
    ...     {"image": torch.randn(2, 3, 16, 16)}, batch_size=[2]
    ... )
    >>> loss = loss_fn(x0, condition=condition)
    >>> loss.shape
    torch.Size([])

    The :class:`~physicsnemo.diffusion.metrics.losses.FlowMatchingLoss` class
    provides the non-patched version. Use
    :class:`MultiDiffusionWeightedFlowMatchingLoss` for a weighted variant that
    supports per-element masking.
    """

    def __init__(
        self,
        model: MultiDiffusionModel2D,
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
        self._md_model = _unwrap_module(model, MultiDiffusionModel2D)
        self.noise_scheduler = noise_scheduler
        self._compiled_patch_x = _CompiledPatchX(self._md_model)

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
                        "score_to_flow_fn must be provided when "
                        "prediction_type='score'."
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
        x0: Float[Tensor, "B C H W"],
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        reset_patch_indices: bool = True,
        t: Float[Tensor, " PB"] | None = None,
        **model_kwargs: Any,
    ) -> Float[Tensor, "P_times_B C Hp Wp"] | Float[Tensor, ""]:
        r"""Compute the patch-based flow-matching loss.

        Parameters
        ----------
        x0 : Tensor
            Clean data of shape :math:`(B, C, H, W)` at global resolution.
        condition : Tensor, TensorDict, or None, optional, default=None
            Conditioning information at global resolution  (batch size
            :math:`B`).
        reset_patch_indices : bool, default=True
            If ``True``, re-draw random patch positions before computing the
            loss. Set to ``False`` when the caller manages patch positions.
        t : Tensor or None, optional, default=None
            Pre-sampled time values of shape :math:`(P \times B,)`, with one
            value per patch. When ``None`` (the default), the scheduler samples
            times internally via
            :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.sample_time`.
            Passing explicit times is useful when the caller needs access to
            the sampled values for diagnostics, such as per-time-bin loss
            tracking.
        **model_kwargs : Any
            Keyword arguments forwarded to the model.

        Returns
        -------
        Tensor
            Per-element loss of shape :math:`(P \times B, C, H_p, W_p)` when
            ``reduction="none"``. Otherwise, a scalar tensor.
        """
        if reset_patch_indices:
            self._md_model.reset_patch_indices()

        x0_patched = self._compiled_patch_x(x0)  # (P*B, C, Hp, Wp)
        PB = x0_patched.shape[0]
        if t is None:
            t = self.noise_scheduler.sample_time(PB, device=x0.device, dtype=x0.dtype)
        x_t = self.noise_scheduler.add_noise(x0_patched, t)

        prediction = self.model(
            x_t,
            t,
            condition=condition,
            x_is_patched=True,
            t_is_patched=True,
            **model_kwargs,
        )

        flow_pred = self._to_flow(prediction, x_t, t)
        flow_target = self._x0_to_flow(x0_patched, x_t, t)
        loss = (flow_pred - flow_target) ** 2
        w = self.noise_scheduler.loss_weight(t)
        loss = apply_loss_weight(w, x0_patched.ndim) * loss
        return self._reduce(loss)


class MultiDiffusionWeightedFlowMatchingLoss:
    r"""Weighted patch-based mean-squared-error flow-matching loss.

    This class extends :class:`MultiDiffusionFlowMatchingLoss` with a
    per-element ``weight`` tensor. This is the patch-based counterpart to the
    :class:`~physicsnemo.diffusion.metrics.losses.WeightedFlowMatchingLoss`
    class.

    The loss accepts the weight at global resolution and patches it alongside
    clean data:

    .. math::
        \mathcal{L} = \mathbb{E}_{t}
        \left[ w(t) \mathbf{m} \odot
        \left(\hat{\mathbf{v}}(\mathbf{x}_t, t)
        - \mathbf{v}(\mathbf{x}_0, \mathbf{x}_t, t)\right)^2 \right].

    Here :math:`\mathbf{v}(\mathbf{x}_0, \mathbf{x}_t, t)` is the flow target,
    or velocity along the scheduler's interpolation path, and
    :math:`\hat{\mathbf{v}}(\mathbf{x}_t, t)` is the model estimate.

    The element-wise weight :math:`\mathbf{m}` is independent of the
    time-dependent weight :math:`w(t)` supplied by the noise scheduler. See
    :class:`MultiDiffusionFlowMatchingLoss` for the shared patching,
    prediction-conversion, and reduction behavior.

    Parameters
    ----------
    model : MultiDiffusionModel2D
        Multi-diffusion model wrapper with random patching configured.
    noise_scheduler : NoiseScheduler
        Noise scheduler used for time sampling, noise injection, and loss
        weighting.
    prediction_type : PredictorType, default="flow"
        Prediction produced by the model. Use ``"flow"`` for a velocity
        prediction, ``"x0"`` for clean data, ``"score"`` for a score, or
        ``"epsilon"`` for noise.
    x0_to_flow_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional, default=None
        Callback ``(x0, x_t, t) -> v`` that computes the flow target from
        clean data. You must provide this callback for every prediction type;
        the constructor raises a ``ValueError`` when passed ``None``.
    score_to_flow_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional
        Callback ``(score, x_t, t) -> v`` used when
        ``prediction_type="score"``.
    epsilon_to_flow_fn : Callable[[Tensor, Tensor, Tensor], Tensor], optional
        Callback ``(epsilon, x_t, t) -> v`` used when
        ``prediction_type="epsilon"``.
    reduction : Literal["none", "mean", "sum"], default="mean"
        Reduction applied to the per-element loss.

    Examples
    --------
    Exclude the left half of the global domain from the loss:

    >>> import torch
    >>> from physicsnemo.core import Module
    >>> from physicsnemo.diffusion.multi_diffusion import (
    ...     MultiDiffusionModel2D,
    ...     MultiDiffusionWeightedFlowMatchingLoss,
    ... )
    >>> from physicsnemo.diffusion.noise_schedulers import (
    ...     RectifiedFlowNoiseScheduler,
    ... )
    >>>
    >>> class FlowModel(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(3, 3, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>>
    >>> model = MultiDiffusionModel2D(
    ...     FlowModel(), global_spatial_shape=(16, 16)
    ... )
    >>> model.set_random_patching(patch_shape=(8, 8), patch_num=4)
    >>> scheduler = RectifiedFlowNoiseScheduler()
    >>> loss_fn = MultiDiffusionWeightedFlowMatchingLoss(
    ...     model, scheduler, x0_to_flow_fn=scheduler.x0_to_flow
    ... )
    >>> x0 = torch.randn(2, 3, 16, 16)
    >>> weight = torch.ones_like(x0)
    >>> weight[:, :, :, :8] = 0.0
    >>> loss = loss_fn(x0, weight=weight)
    >>> loss.shape
    torch.Size([])

    The
    :class:`~physicsnemo.diffusion.metrics.losses.WeightedFlowMatchingLoss`
    class provides the non-patched version. Use
    :class:`MultiDiffusionFlowMatchingLoss` when masking is not needed.
    """

    def __init__(
        self,
        model: MultiDiffusionModel2D,
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
        self._md_model = _unwrap_module(model, MultiDiffusionModel2D)
        self.noise_scheduler = noise_scheduler
        self._compiled_patch_x = _CompiledPatchX(self._md_model)

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
                        "score_to_flow_fn must be provided when "
                        "prediction_type='score'."
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
        x0: Float[Tensor, "B C H W"],
        weight: Float[Tensor, "B C H W"],
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        reset_patch_indices: bool = True,
        t: Float[Tensor, " PB"] | None = None,
        **model_kwargs: Any,
    ) -> Float[Tensor, "P_times_B C Hp Wp"] | Float[Tensor, ""]:
        r"""Compute the weighted multi-diffusion flow-matching loss.

        Parameters
        ----------
        x0 : Tensor
            Clean data of shape :math:`(B, C, H, W)` at global resolution.
        weight : Tensor
            Per-element weight of shape :math:`(B, C, H, W)`. The loss patches
            the weight alongside ``x0``.
        condition : Tensor, TensorDict, or None, optional, default=None
            Conditioning information at global resolution.
        reset_patch_indices : bool, default=True
            If ``True``, re-draw random patch positions before computing the
            loss.
        t : Tensor or None, optional, default=None
            Pre-sampled time values of shape :math:`(P \times B,)`, with one
            value per patch. When ``None`` (the default), the scheduler samples
            times internally via
            :meth:`~physicsnemo.diffusion.noise_schedulers.NoiseScheduler.sample_time`.
            Passing explicit times is useful when the caller needs access to
            the sampled values for diagnostics, such as per-time-bin loss
            tracking.
        **model_kwargs : Any
            Keyword arguments forwarded to the model.

        Returns
        -------
        Tensor
            Per-element loss of shape :math:`(P \times B, C, H_p, W_p)` when
            ``reduction="none"``. Otherwise, a scalar tensor.
        """
        if not torch.compiler.is_compiling():
            if weight.shape != x0.shape:
                raise ValueError(
                    f"weight shape {tuple(weight.shape)} must match "
                    f"x0 shape {tuple(x0.shape)}."
                )

        if reset_patch_indices:
            self._md_model.reset_patch_indices()

        x0_patched = self._compiled_patch_x(x0)  # (P*B, C, Hp, Wp)
        weight_patched = self._compiled_patch_x(weight)  # (P*B, C, Hp, Wp)
        PB = x0_patched.shape[0]
        if t is None:
            t = self.noise_scheduler.sample_time(PB, device=x0.device, dtype=x0.dtype)
        x_t = self.noise_scheduler.add_noise(x0_patched, t)

        prediction = self.model(
            x_t,
            t,
            condition=condition,
            x_is_patched=True,
            t_is_patched=True,
            **model_kwargs,
        )

        flow_pred = self._to_flow(prediction, x_t, t)
        flow_target = self._x0_to_flow(x0_patched, x_t, t)
        loss = weight_patched * (flow_pred - flow_target) ** 2
        w = self.noise_scheduler.loss_weight(t)
        loss = apply_loss_weight(w, x0_patched.ndim) * loss
        return self._reduce(loss)
