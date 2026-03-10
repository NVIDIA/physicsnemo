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

import math
from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from tensordict import TensorDict

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module


class BaseAffinePreconditioner(Module, ABC):
    r"""
    Abstract base class for diffusion model preconditioners using an affine
    transformation.

    This class provides a standardized interface for implementing
    preconditioners that use affine transformations of the model
    input and output.

    The preconditioner wraps a neural network model :math:`F` and applies
    a preconditioning formula to transform the network output to produce
    the preconditioned output :math:`D(\mathbf{x}, t)` according to:

    .. math::

        D(\mathbf{x}, t) = c_{\text{skip}}(t) \mathbf{x} +
        c_{\text{out}}(t) F(c_{\text{in}}(t) \mathbf{x}, c_{\text{noise}}(t))

    where:

    - :math:`c_{\text{in}}(t)`: Input scaling coefficient
    - :math:`c_{\text{noise}}(t)`: Noise conditioning value
    - :math:`c_{\text{out}}(t)`: Output scaling coefficient
    - :math:`c_{\text{skip}}(t)`: Skip connection scaling coefficient

    and where :math:`\mathbf{x}` is the latent state and :math:`t` is the
    diffusion time.

    The wrapped model :math:`F` must be an instance of
    :class:`~physicsnemo.core.Module` that satisfies the
    :class:`~physicsnemo.diffusion.DiffusionModel` interface, with the
    following signature:

    .. code-block:: python

        model(
            x: torch.Tensor,  # Shape: (B, *)
            t: torch.Tensor,  # Shape: (B,)
            condition: torch.Tensor | TensorDict | None = None,
            **model_kwargs: Any,
        ) -> torch.Tensor  # Shape: (B, *)

    The preconditioner is agnostic to the prediction target of the wrapped
    model :math:`F`. The same preconditioning formula is applied regardless of
    whether the model is an :math:`\mathbf{x}_0`-predictor, an
    :math:`\epsilon`-predictor, a score predictor, or a
    :math:`\mathbf{v}`-predictor.

    .. note::

        The preconditioner itself also satisfies the
        :class:`~physicsnemo.diffusion.DiffusionModel` interface, meaning it
        does not change the signature of the wrapped model :math:`F`, and it
        can be used anywhere a diffusion model is expected.

    Parameters
    ----------
    model : physicsnemo.Module
        The underlying neural network model :math:`F` to wrap with the
        signature described above.
    meta : ModelMetaData, optional
        Meta data class for storing info regarding model, by default None.
        Subclasses can pass their own metadata.

    Forward
    -------
    x : torch.Tensor
        Noisy latent state of shape :math:`(B, *)` where :math:`B` is the
        batch size and :math:`*` denotes any number of additional dimensions.
    t : torch.Tensor
        Diffusion time tensor of shape :math:`(B,)`.
    condition : torch.Tensor, TensorDict, or None, optional, default=None
        Single Tensor or a TensorDict containing conditioning tensors with
        batch size :math:`B` matching that of ``x``. Pass ``None`` for an
        unconditional model.

    **model_kwargs : Any
        Additional keyword arguments passed to the underlying model.

    Outputs
    -------
    torch.Tensor
        Preconditioned model output with the same shape as the original model
        output.

    .. note::

        To implement a new preconditioner, a subclass of
        :class:`BaseAffinePreconditioner` must be defined, and some methods
        have to be implemented:

        - Subclasses must implement the :meth:`compute_coefficients` method to
          define the specific preconditioning scheme.

        - A :meth:`sigma` method can optionally be implemented.
          If a subclass implements the :meth:`sigma` method, the diffusion time
          :math:`t` is first transformed to a noise level :math:`\sigma(t)`
          before being passed to :meth:`compute_coefficients`. This allows
          implementing preconditioners for different time-to-noise-level
          mappings while keeping the same preconditioning interface, in
          particular for
          preconditioning schemes based on noise level (that is
          :math:`c_{\text{in}}(\sigma)`,
          :math:`c_{\text{noise}}(\sigma)`, :math:`c_{\text{out}}(\sigma)`,
          :math:`c_{\text{skip}}(\sigma)` instead of :math:`c_{\text{in}}(t)`,
          :math:`c_{\text{noise}}(t)`, :math:`c_{\text{out}}(t)`,
          :math:`c_{\text{skip}}(t)`).

        - The ``forward`` method of the preconditioner *should not* be
          overriden.

    .. note::

        The arguments ``t`` of the preconditioner forward method is always
        assumed to be the diffusion time. For preconditioning schemes based
        on noise level the noise level :math:`\sigma(t)` is computed internally
        using the :meth:`sigma` method.

    Examples
    --------
    The following example shows how to implement a classical EDM
    preconditioner. For EDM, there is no need to implement the :meth:`sigma`
    method since :math:`\sigma(t) = t` (noise level and diffusion time are the
    same).

    We first define a simple model to wrap:

    >>> import torch
    >>> from tensordict import TensorDict
    >>> from physicsnemo.nn import Module
    >>> class SimpleModel(Module):
    ...     def __init__(self, channels: int):
    ...         super().__init__()
    ...         self.channels = channels
    ...         self.net = torch.nn.Conv2d(channels, channels, 1)
    ...
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)

    Now we define the EDM preconditioner:

    >>> from physicsnemo.diffusion.preconditioners import (
    ...     BaseAffinePreconditioner,
    ... )
    >>> class SimpleEDMPreconditioner(BaseAffinePreconditioner):
    ...     def __init__(self, model, sigma_data: float = 0.5):
    ...         super().__init__(model)
    ...         self.sigma_data = sigma_data
    ...
    ...     def compute_coefficients(self, t: torch.Tensor):
    ...         # For EDM sigma(t) = t, so the argument passed to
    ...         # compute_coefficients is already sigma(t)
    ...         sigma_data = self.sigma_data
    ...         c_skip = sigma_data**2 / (t**2 + sigma_data**2)
    ...         c_out = t * sigma_data / (t**2 + sigma_data**2).sqrt()
    ...         c_in = 1 / (sigma_data**2 + t**2).sqrt()
    ...         c_noise = t.log() / 4
    ...         return c_in, c_noise, c_out, c_skip
    ...
    >>> model = SimpleModel(channels=3)
    >>> precond = SimpleEDMPreconditioner(model, sigma_data=0.5)
    >>> x = torch.randn(2, 3, 16, 16)
    >>> t = torch.rand(2)
    >>> condition = TensorDict({}, batch_size=[2])
    >>> out = precond(x, t, condition)
    >>> out.shape
    torch.Size([2, 3, 16, 16])

    The following example shows how to override the :meth:`sigma` method to
    implement a Variance Exploding (VE) preconditioner where
    :math:`\sigma(t) = \sqrt{t}`.

    >>> class VEPreconditioner(BaseAffinePreconditioner):
    ...     def __init__(self, model):
    ...         super().__init__(model)
    ...
    ...     def sigma(self, t: torch.Tensor) -> torch.Tensor:
    ...         # Override sigma for VE time-to-noise-level mapping
    ...         return t.sqrt()
    ...
    ...     def compute_coefficients(self, sigma: torch.Tensor):
    ...         # Here the argument passed to compute_coefficients is
    ...         # sigma(t) = sqrt(t) due to override of the sigma method
    ...         # due to override of the sigma method
    ...         c_skip = torch.ones_like(sigma)
    ...         c_out = sigma
    ...         c_in = torch.ones_like(sigma)
    ...         c_noise = (0.5 * sigma).log()
    ...         return c_in, c_noise, c_out, c_skip
    ...
    >>> precond_ve = VEPreconditioner(model)
    >>> out_ve = precond_ve(x, t, condition)
    >>> out_ve.shape
    torch.Size([2, 3, 16, 16])

    **Wrapping existing models to satisfy the DiffusionModel interface**

    Some models in PhysicsNeMo have signatures that differ from the
    :class:`~physicsnemo.diffusion.DiffusionModel` interface. Below are
    examples showing how to write thin wrappers to make them compatible
    with preconditioners, including image-based conditioning via channel
    concatenation.

    **Example: Wrapping SongUNet**

    The :class:`~physicsnemo.models.diffusion_unets.SongUNet` model has
    the signature ``forward(x, noise_labels, class_labels, augment_labels)``.
    We wrap it to match ``forward(x, t, condition)``, where ``condition``
    contains both class labels (1D vector) and an image to concatenate
    channel-wise:

    >>> from physicsnemo.models.diffusion_unets import SongUNet
    >>> from physicsnemo.diffusion import DiffusionModel
    >>> from tensordict import TensorDict
    >>> class SongUNetWrapper(Module):
    ...     def __init__(self, img_channels, cond_channels, label_dim, **kwargs):
    ...         super().__init__()
    ...         # in_channels = img_channels + cond_channels for concatenation
    ...         self.net = SongUNet(
    ...             in_channels=img_channels + cond_channels,
    ...             out_channels=img_channels,
    ...             label_dim=label_dim,
    ...             **kwargs,
    ...         )
    ...
    ...     def forward(self, x, t, condition):
    ...         # Concatenate image condition "y" channel-wise to input
    ...         y = condition["y"]  # shape: (B, C_cond, H, W)
    ...         x_cat = torch.cat([x, y], dim=1)
    ...         # Extract 1D vector condition for class_labels
    ...         class_labels = condition["class_labels"]  # shape: (B, label_dim)
    ...         return self.net(x_cat, noise_labels=t, class_labels=class_labels)
    ...
    >>> wrapped = SongUNetWrapper(
    ...     img_channels=2, cond_channels=1, label_dim=4, img_resolution=8
    ... )
    >>> isinstance(wrapped, DiffusionModel)
    True
    >>> x = torch.rand(1, 2, 8, 8)
    >>> t = torch.rand(1)
    >>> condition = TensorDict({
    ...     "y": torch.rand(1, 1, 8, 8),           # image condition
    ...     "class_labels": torch.rand(1, 4),      # 1D vector condition
    ... }, batch_size=[1])
    >>> out = wrapped(x, t, condition)
    >>> out.shape
    torch.Size([1, 2, 8, 8])

    **Example: Wrapping DiT**

    The :class:`~physicsnemo.models.dit.DiT` model has
    the signature ``forward(x, t, condition, ...)``. We wrap it to support
    both image conditioning (via channel concatenation) and vector
    conditioning:

    >>> from physicsnemo.models.dit import DiT
    >>> class DiTWrapper(Module):
    ...     def __init__(self, img_channels, cond_channels, cond_dim, **kwargs):
    ...         super().__init__()
    ...         # in_channels = img_channels + cond_channels for concatenation
    ...         self.net = DiT(
    ...             in_channels=img_channels + cond_channels,
    ...             out_channels=img_channels,
    ...             condition_dim=cond_dim,
    ...             **kwargs,
    ...         )
    ...
    ...     def forward(self, x, t, condition):
    ...         # Concatenate image condition "y" channel-wise to input
    ...         y = condition["y"]  # shape: (B, C_cond, H, W)
    ...         x_cat = torch.cat([x, y], dim=1)
    ...         # Extract 1D vector condition
    ...         vec = condition["vec"]  # shape: (B, cond_dim)
    ...         return self.net(x_cat, t, condition=vec)
    ...
    >>> wrapped_dit = DiTWrapper(
    ...     img_channels=2, cond_channels=1, cond_dim=4,
    ...     input_size=8, patch_size=4, attention_backend="timm",
    ... )
    >>> isinstance(wrapped_dit, DiffusionModel)
    True
    >>> x = torch.rand(1, 2, 8, 8)
    >>> t = torch.rand(1)
    >>> condition = TensorDict({
    ...     "y": torch.rand(1, 1, 8, 8),  # image condition
    ...     "vec": torch.rand(1, 4),       # 1D vector condition
    ... }, batch_size=[1])
    >>> out = wrapped_dit(x, t, condition)
    >>> out.shape
    torch.Size([1, 2, 8, 8])

    **Example: Using ConcatConditionWrapper with a preconditioner**

    The pattern in the previous example, where (spatially-varying) conditioning
    is concatenated to the noised latent state (and possibly vector conditioning
    is also passed as a separate argument) is common across several diffusion
    use-cases.
    Thus for convenience, we provide the wrapper class :class:`~physicsnemo.diffusion.utils.ConcatConditionWrapper`
    to save you the trouble of writing your own wrapper for this common pattern:

    >>> from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    >>> from physicsnemo.diffusion.utils import ConcatConditionWrapper
    >>> base_model = SongUNet(img_resolution=8, in_channels=4, out_channels=3, label_dim=4)
    >>> wrapped_model = ConcatConditionWrapper(base_model)
    >>> precond = EDMPreconditioner(wrapped_model, sigma_data=0.5)
    >>> x = torch.rand(1, 3, 8, 8)
    >>> t = torch.rand(1)
    >>> condition = TensorDict({
    ...     "cond_concat": torch.rand(1, 1, 8, 8),  # image condition
    ...     "cond_vec": torch.rand(1, 4),           # vector condition
    ... }, batch_size=[1])
    >>> out = precond(x, t, condition)
    >>> out.shape
    torch.Size([1, 3, 8, 8])

    The same wrapper can be used with :class:`~physicsnemo.models.dit.DiT`
    backbones as well, with ``cond_vec`` passed to the model's ``condition`` argument.
    """

    def __init__(
        self,
        model: Module,
        meta: ModelMetaData | None = None,
    ) -> None:
        super().__init__()
        self.meta = meta
        self.model = model

    @abstractmethod
    def compute_coefficients(
        self, t: torch.Tensor, /
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Compute the preconditioning coefficients for a given diffusion time
        :math:`t` or noise level :math:`\sigma`.

        This abstract method must be implemented by subclasses to define
        the specific preconditioning scheme.

        Parameters
        ----------
        t : torch.Tensor
            Diffusion time (or noise level if :meth:`sigma` is
            implemented) tensor of shape :math:`(B, 1, ..., 1)` where
            :math:`B` is the batch size and the trailing singleton
            dimensions match the spatial dimensions of the latent state
            ``x`` for broadcasting.

        Returns
        -------
        c_in : torch.Tensor
            Input scaling coefficient of shape :math:`(B, 1, ..., 1)`.
        c_noise : torch.Tensor
            Noise conditioning value of shape :math:`(B, 1, ..., 1)`.
        c_out : torch.Tensor
            Output scaling coefficient of shape :math:`(B, 1, ..., 1)`.
        c_skip : torch.Tensor
            Skip connection scaling coefficient of shape
            :math:`(B, 1, ..., 1)`.
        """
        ...

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Map diffusion time :math:`t` to noise level :math:`\sigma(t)`.

        By default, this is the identity function :math:`\sigma(t) = t`.
        Subclasses can override this to implement preconditioners for different
        time-to-noise-level mappings.

        When overridden, the output of this method is passed to
        :meth:`compute_coefficients` instead of the raw time ``t``.

        Parameters
        ----------
        t : torch.Tensor
            Diffusion time tensor of shape :math:`(B,)` where
            :math:`B` is the batch size.

        Returns
        -------
        torch.Tensor
            Noise level :math:`\sigma(t)` of shape :math:`(B,)`.
        """
        return t

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | TensorDict | None = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        if not torch.compiler.is_compiling():
            B = x.shape[0]
            if t.shape != (B,):
                raise ValueError(
                    f"Expected t to have shape ({B},) matching batch size of "
                    f"x, but got {t.shape}."
                )
            if isinstance(condition, TensorDict):
                if condition.batch_size and condition.batch_size[0] != B:
                    raise ValueError(
                        f"Condition TensorDict has batch size {condition.batch_size[0]} "
                        f"but expected {B} to match x."
                    )
            elif isinstance(condition, torch.Tensor):
                if condition.shape[0] != B:
                    raise ValueError(
                        f"Condition tensor has batch size {condition.shape[0]} "
                        f"but expected {B} to match x."
                    )

        # Map time step to noise level via sigma method
        sigma_t = self.sigma(t).reshape(-1, *([1] * (x.ndim - 1)))

        # Compute preconditioning coefficients
        c_in, c_noise, c_out, c_skip = self.compute_coefficients(sigma_t)

        # Forward through the underlying model
        if condition is not None:
            F_x = self.model(
                c_in * x,
                c_noise.flatten(),
                condition=condition,
                **model_kwargs,
            )
        else:
            F_x = self.model(
                c_in * x,
                c_noise.flatten(),
                **model_kwargs,
            )

        D_x = c_skip * x + c_out * F_x

        return D_x


class VPPreconditioner(BaseAffinePreconditioner):
    r"""
    Variance Preserving (VP) preconditioner.

    Implements the preconditioning scheme from the VP formulation of
    score-based generative models.

    The time-to-noise-level mapping is:

    .. math::

        \sigma(t) = \sqrt{\exp\left(\frac{\beta_d}{2} t^2
        + \beta_{\min} t\right) - 1}

    The preconditioning coefficients are:

    .. math::

        c_{\text{skip}} &= 1 \\
        c_{\text{out}} &= -\sigma \\
        c_{\text{in}} &= \frac{1}{\sqrt{\sigma^2 + 1}} \\
        c_{\text{noise}} &= (M - 1) \cdot \sigma^{-1}(\sigma)
    
    With these coefficients, the preconditioned model output is expected to be
    an :math:`\mathbf{x}_0`-prediction (clean data estimate).
    This preconditioner is not directly compatible for score-prediction
    training or others.
    
    For training, it is usually paired with
    :class:`~physicsnemo.diffusion.metrics.losses.DSMLoss`
    (``prediction_type="x0"``) and
    :class:`~physicsnemo.diffusion.noise_schedulers.VPNoiseScheduler`.

    Parameters
    ----------
    model : physicsnemo.Module
        The underlying neural network model to wrap with signature described in
        :class:`BaseAffinePreconditioner`.
    beta_d : float, optional
        Extent of the noise level schedule, by default 19.9.
    beta_min : float, optional
        Initial slope of the noise level schedule, by default 0.1.
    M : int, optional
        Number of discretization steps in the DDPM formulation,
        by default 1000.

    Forward
    -------
    x : torch.Tensor
        Noisy latent state of shape :math:`(B, *)` where :math:`B` is the
        batch size and :math:`*` denotes any number of additional dimensions.
    t : torch.Tensor
        Diffusion time tensor of shape :math:`(B,)`.
    condition : torch.Tensor, TensorDict, or None, optional, default=None
        Single Tensor or a TensorDict containing conditioning tensors with
        batch size :math:`B` matching that of ``x``. Pass ``None`` for an
        unconditional model.
    **model_kwargs : Any
        Additional keyword arguments passed to the underlying model.

    Outputs
    -------
    torch.Tensor
        Preconditioned model output with the same shape as the original model
        output.

    Note
    ----
    Reference: `Score-Based Generative Modeling through Stochastic
    Differential Equations <https://arxiv.org/abs/2011.13456>`_

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.core import Module
    >>> # Define a simple model satisfying the diffusion model interface
    >>> class SimpleModel(Module):
    ...     def __init__(self, channels: int):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(channels, channels, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>> model = SimpleModel(channels=3)
    >>> precond = VPPreconditioner(model, beta_d=19.9, beta_min=0.1, M=1000)
    >>> x = torch.randn(2, 3, 16, 16)  # batch of 2 images
    >>> t = torch.rand(2)              # diffusion time for each sample
    >>> out = precond(x, t, condition=None)
    >>> out.shape
    torch.Size([2, 3, 16, 16])
    """

    def __init__(
        self,
        model: Module,
        beta_d: float = 19.9,
        beta_min: float = 0.1,
        M: int = 1000,
    ) -> None:
        super().__init__(model)
        self.register_buffer("beta_d", torch.tensor(beta_d))
        self.register_buffer("beta_min", torch.tensor(beta_min))
        self.register_buffer("M", torch.tensor(M))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Compute :math:`\sigma(t)` for the VP formulation.

        Parameters
        ----------
        t : torch.Tensor
            Diffusion time tensor of shape :math:`(B,)`.

        Returns
        -------
        torch.Tensor
            Noise level :math:`\sigma(t)` of shape :math:`(B,)`.
        """
        exponent = 0.5 * self.beta_d * (t**2) + self.beta_min * t
        return (exponent.exp() - 1).sqrt()

    def compute_coefficients(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Compute VP preconditioning coefficients.

        Parameters
        ----------
        sigma : torch.Tensor
            Noise level tensor of shape :math:`(B, 1, ..., 1)`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Preconditioning coefficients (:math:`c_{\text{in}}`,
            :math:`c_{\text{noise}}`, :math:`c_{\text{out}}`,
            :math:`c_{\text{skip}}`) of shape :math:`(B, 1, ..., 1)`.
        """
        c_skip = torch.ones_like(sigma)
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        # Compute t = sigma_inv(sigma)
        t = (
            (self.beta_min**2 + 2 * self.beta_d * (1 + sigma**2).log()).sqrt()
            - self.beta_min
        ) / self.beta_d
        c_noise = (self.M - 1) * t
        return c_in, c_noise, c_out, c_skip


class VEPreconditioner(BaseAffinePreconditioner):
    r"""
    Variance Exploding (VE) preconditioner.

    Implements the preconditioning scheme from the VE formulation of
    score-based generative models.

    For VE, the time-to-noise-level mapping is the identity:
    :math:`\sigma(t) = t`.

    The preconditioning coefficients are:

    .. math::

        c_{\text{skip}} &= 1 \\
        c_{\text{out}} &= \sigma \\
        c_{\text{in}} &= 1 \\
        c_{\text{noise}} &= \log(0.5 \cdot \sigma)
    
    With these coefficients, the preconditioned model output is expected to be an
    :math:`\mathbf{x}_0`-prediction (clean data estimate). This preconditioner
    is not directly compatible for score-prediction training or others.
    
    For training, it is usually paired with
    :class:`~physicsnemo.diffusion.metrics.losses.DSMLoss`
    (``prediction_type="x0"``) and
    :class:`~physicsnemo.diffusion.noise_schedulers.VENoiseScheduler`.

    Parameters
    ----------
    model : physicsnemo.Module
        The underlying neural network model to wrap with signature described in
        :class:`BaseAffinePreconditioner`.

    Forward
    -------
    x : torch.Tensor
        Noisy latent state of shape :math:`(B, *)` where :math:`B` is the
        batch size and :math:`*` denotes any number of additional dimensions.
    t : torch.Tensor
        Diffusion time tensor of shape :math:`(B,)`.
    condition : torch.Tensor, TensorDict, or None, optional, default=None
        Single Tensor or a TensorDict containing conditioning tensors with
        batch size :math:`B` matching that of ``x``. Pass ``None`` for an
        unconditional model.
    **model_kwargs : Any
        Additional keyword arguments passed to the underlying model.

    Outputs
    -------
    torch.Tensor
        Preconditioned model output with the same shape as the original model
        output.

    Note
    ----
    Reference: `Score-Based Generative Modeling through Stochastic
    Differential Equations <https://arxiv.org/abs/2011.13456>`_

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.core import Module
    >>> # Define a simple model satisfying the diffusion model interface
    >>> class SimpleModel(Module):
    ...     def __init__(self, channels: int):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(channels, channels, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>> model = SimpleModel(channels=3)
    >>> precond = VEPreconditioner(model)
    >>> x = torch.randn(2, 3, 16, 16)  # batch of 2 images
    >>> t = torch.rand(2)              # diffusion time for each sample
    >>> out = precond(x, t, condition=None)
    >>> out.shape
    torch.Size([2, 3, 16, 16])
    """

    def __init__(self, model: Module) -> None:
        super().__init__(model)

    def compute_coefficients(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Compute VE preconditioning coefficients.

        Parameters
        ----------
        t : torch.Tensor
            Diffusion time tensor of shape :math:`(B, 1, ..., 1)`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Preconditioning coefficients (:math:`c_{\text{in}}`,
            :math:`c_{\text{noise}}`, :math:`c_{\text{out}}`,
            :math:`c_{\text{skip}}`) of shape :math:`(B, 1, ..., 1)`.
        """
        c_skip = torch.ones_like(t)
        c_out = t
        c_in = torch.ones_like(t)
        c_noise = (0.5 * t).log()
        return c_in, c_noise, c_out, c_skip


class IDDPMPreconditioner(BaseAffinePreconditioner):
    r"""
    Improved DDPM (iDDPM) preconditioner.

    Implements the preconditioning scheme from the improved DDPM
    formulation.
    
    The preconditioning coefficients are:

    .. math::

        c_{\text{skip}} &= 1 \\
        c_{\text{out}} &= -\sigma \\
        c_{\text{in}} &= \frac{1}{\sqrt{\sigma^2 + 1}} \\
        c_{\text{noise}} &= M - 1 - \text{argmin}|\sigma - u_j|

    where :math:`u_j, j = 0, ..., M` are the precomputed noise levels from
    the iDDPM discretization.

    With these coefficients, the preconditioned model output is expected to be
    an :math:`\mathbf{x}_0`-prediction (clean data estimate). This
    preconditioner is not directly compatible for score-prediction training or
    others.
    
    For training, it is usually paired with
    :class:`~physicsnemo.diffusion.metrics.losses.DSMLoss`
    (``prediction_type="x0"``) and
    :class:`~physicsnemo.diffusion.noise_schedulers.IDDPMNoiseScheduler`.

    Parameters
    ----------
    model : physicsnemo.Module
        The underlying neural network model to wrap with signature described in
        :class:`BaseAffinePreconditioner`.
    C_1 : float, optional
        Timestep adjustment at low noise levels, by default 0.001.
    C_2 : float, optional
        Timestep adjustment at high noise levels, by default 0.008.
    M : int, optional
        Number of discretization steps in the DDPM formulation,
        by default 1000.

    Forward
    -------
    x : torch.Tensor
        Noisy latent state of shape :math:`(B, *)` where :math:`B` is the
        batch size and :math:`*` denotes any number of additional dimensions.
    t : torch.Tensor
        Diffusion time tensor of shape :math:`(B,)`.
    condition : torch.Tensor, TensorDict, or None, optional, default=None
        Single Tensor or a TensorDict containing conditioning tensors with
        batch size :math:`B` matching that of ``x``. Pass ``None`` for an
        unconditional model.
    **model_kwargs : Any
        Additional keyword arguments passed to the underlying model.

    Outputs
    -------
    torch.Tensor
        Preconditioned model output with the same shape as the original model
        output.

    Note
    ----
    Reference: `Improved Denoising Diffusion Probabilistic Models
    <https://arxiv.org/abs/2102.09672>`_

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.core import Module
    >>> # Define a simple model satisfying the diffusion model interface
    >>> class SimpleModel(Module):
    ...     def __init__(self, channels: int):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(channels, channels, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>> model = SimpleModel(channels=3)
    >>> precond = IDDPMPreconditioner(model, C_1=0.001, C_2=0.008, M=1000)
    >>> x = torch.randn(2, 3, 16, 16)  # batch of 2 images
    >>> t = torch.rand(2)              # diffusion time for each sample
    >>> out = precond(x, t, condition=None)
    >>> out.shape
    torch.Size([2, 3, 16, 16])
    """

    def __init__(
        self,
        model: Module,
        C_1: float = 0.001,
        C_2: float = 0.008,
        M: int = 1000,
    ) -> None:
        super().__init__(model)
        self.register_buffer("C_1", torch.tensor(C_1))
        self.register_buffer("C_2", torch.tensor(C_2))
        self.register_buffer("M", torch.tensor(M))

        # Precompute the noise level schedule u_j, j = 0, ..., M
        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):
            angle_j = 0.5 * math.pi * j / M / (C_2 + 1)
            angle_jm1 = 0.5 * math.pi * (j - 1) / M / (C_2 + 1)
            alpha_bar_j = math.sin(angle_j) ** 2
            alpha_bar_jm1 = math.sin(angle_jm1) ** 2
            alpha_ratio = alpha_bar_jm1 / alpha_bar_j
            u[j - 1] = ((u[j] ** 2 + 1) / max(alpha_ratio, C_1) - 1).sqrt()
        self.register_buffer("u", u)

    def compute_coefficients(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Compute iDDPM preconditioning coefficients.

        Parameters
        ----------
        t : torch.Tensor
            Diffusion time tensor of shape :math:`(B, 1, ..., 1)`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Preconditioning coefficients (:math:`c_{\text{in}}`,
            :math:`c_{\text{noise}}`, :math:`c_{\text{out}}`,
            :math:`c_{\text{skip}}`) of shape :math:`(B, 1, ..., 1)`.
        """
        c_skip = torch.ones_like(t)
        c_out = -t
        c_in = 1 / (t**2 + 1).sqrt()

        # Round sigma to nearest index in precomputed schedule u
        u: torch.Tensor = self.u  # type: ignore[assignment]
        t_flat = t.reshape(1, -1, 1)
        u_reshaped = u.reshape(1, -1, 1)
        idx = torch.cdist(t_flat, u_reshaped).argmin(2).reshape(t.shape)
        c_noise = self.M - 1 - idx

        return c_in, c_noise, c_out, c_skip


class EDMPreconditioner(BaseAffinePreconditioner):
    r"""
    EDM preconditioner.

    Implements the improved preconditioning scheme proposed in the EDM
    paper.

    For EDM, the time-to-noise-level mapping is the identity:
    :math:`\sigma(t) = t`.

    The preconditioning coefficients are:

    .. math::

        c_{\text{skip}} &= \frac{\sigma_{\text{data}}^2}
            {\sigma^2 + \sigma_{\text{data}}^2} \\
        c_{\text{out}} &= \frac{\sigma \cdot \sigma_{\text{data}}}
            {\sqrt{\sigma^2 + \sigma_{\text{data}}^2}} \\
        c_{\text{in}} &= \frac{1}
            {\sqrt{\sigma_{\text{data}}^2 + \sigma^2}} \\
        c_{\text{noise}} &= \frac{\log(\sigma)}{4}

    With these coefficients, the preconditioned model output is expected to be an
    :math:`\mathbf{x}_0`-prediction (clean data estimate). This preconditioner
    is not directly compatible for score-prediction training or others.
    
    For training, it is usually paired with
    :class:`~physicsnemo.diffusion.metrics.losses.DSMLoss`
    (``prediction_type="x0"``) and
    :class:`~physicsnemo.diffusion.noise_schedulers.EDMNoiseScheduler`.

    Parameters
    ----------
    model : physicsnemo.Module
        The underlying neural network model to wrap with signature described in
        :class:`BaseAffinePreconditioner`.
    sigma_data : float, optional
        Expected standard deviation of the training data, by default 0.5.

    Forward
    -------
    x : torch.Tensor
        Noisy latent state of shape :math:`(B, *)` where :math:`B` is the
        batch size and :math:`*` denotes any number of additional dimensions.
    t : torch.Tensor
        Diffusion time tensor of shape :math:`(B,)`.
    condition : torch.Tensor, TensorDict, or None, optional, default=None
        Single Tensor or a TensorDict containing conditioning tensors with
        batch size :math:`B` matching that of ``x``. Pass ``None`` for an
        unconditional model.
    **model_kwargs : Any
        Additional keyword arguments passed to the underlying model.

    Outputs
    -------
    torch.Tensor
        Preconditioned model output with the same shape as the original model
        output.

    Note
    ----
    Reference: `Elucidating the Design Space of Diffusion-Based
    Generative Models <https://arxiv.org/abs/2206.00364>`_

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.core import Module
    >>> # Define a simple model satisfying the diffusion model interface
    >>> class SimpleModel(Module):
    ...     def __init__(self, channels: int):
    ...         super().__init__()
    ...         self.net = torch.nn.Conv2d(channels, channels, 1)
    ...     def forward(self, x, t, condition=None):
    ...         return self.net(x)
    >>> model = SimpleModel(channels=3)
    >>> precond = EDMPreconditioner(model, sigma_data=0.5)
    >>> x = torch.randn(2, 3, 16, 16)  # batch of 2 images
    >>> t = torch.rand(2)              # diffusion time for each sample
    >>> out = precond(x, t, condition=None)
    >>> out.shape
    torch.Size([2, 3, 16, 16])
    """

    def __init__(
        self,
        model: Module,
        sigma_data: float = 0.5,
    ) -> None:
        super().__init__(model)
        self.register_buffer("sigma_data", torch.tensor(sigma_data))

    def compute_coefficients(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Compute EDM preconditioning coefficients.

        Parameters
        ----------
        t : torch.Tensor
            Diffusion time (or noise level, since they are identical for EDM)
            of shape :math:`(B, 1, ..., 1)`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Preconditioning coefficients (:math:`c_{\text{in}}`,
            :math:`c_{\text{noise}}`, :math:`c_{\text{out}}`,
            :math:`c_{\text{skip}}`) of shape :math:`(B, 1, ..., 1)`.
        """
        sd = self.sigma_data
        c_skip = sd**2 / (t**2 + sd**2)
        c_out = t * sd / (t**2 + sd**2).sqrt()
        c_in = 1 / (sd**2 + t**2).sqrt()
        c_noise = t.log() / 4
        return c_in, c_noise, c_out, c_skip
