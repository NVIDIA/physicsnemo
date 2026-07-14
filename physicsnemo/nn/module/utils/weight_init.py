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

import warnings
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch.nn.init import trunc_normal_ as _torch_trunc_normal_


def trunc_normal_(*args, **kwargs):
    """Deprecated alias for :func:`torch.nn.init.trunc_normal_`.

    This re-export exists only to preserve backward compatibility for code
    that imported ``trunc_normal_`` from ``physicsnemo.nn.module.utils`` (or
    its ``weight_init`` submodule path) prior to v2.1. It will be removed in
    v2.2.0; new code should call :func:`torch.nn.init.trunc_normal_`
    directly.
    """
    warnings.warn(
        "`physicsnemo.nn.module.utils.trunc_normal_` is deprecated and will "
        "be removed in v2.2.0. Use `torch.nn.init.trunc_normal_` directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _torch_trunc_normal_(*args, **kwargs)


_NOISE_PRESETS = ("scaled_normal", "normal")


@torch.no_grad()
def shrink_and_perturb_(
    module: torch.nn.Module,
    shrink: float = 0.5,
    perturb: float = 0.1,
    *,
    noise: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "scaled_normal",
    include: Optional[Callable[[str, torch.Tensor], bool]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.nn.Module:
    r"""Apply *shrink-and-perturb* re-initialization to ``module`` in place.

    For every selected parameter :math:`\theta`, the update is

    .. math:: \theta \leftarrow \lambda\,\theta + p\,\varepsilon,

    where :math:`\lambda` is ``shrink``, :math:`p` is ``perturb``, and
    :math:`\varepsilon` is fresh noise. Shrinking a *pretrained* weight toward
    zero restores the scale statistics and plasticity of a fresh initialization
    while the noise breaks symmetry, yet the shrink term keeps the direction of
    the pretrained features. In warm-started training this often reaches a lower
    loss asymptote than fine-tuning the raw pretrained weights (Ash & Adams,
    *On Warm-Starting Neural Network Training*, NeurIPS 2020).

    The operation is only meaningful on **pretrained** weights: applied to a
    fresh initialization it merely rescales and re-noises random values.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters are perturbed in place. Buffers (e.g.
        batch-norm running statistics) are left untouched.
    shrink : float, optional
        Multiplicative retention factor :math:`\lambda \ge 0` applied to each
        weight. Values in ``[0, 1)`` shrink toward zero; ``1.0`` disables the
        shrink. Default ``0.5``.
    perturb : float, optional
        Noise scale :math:`p \ge 0`. With ``noise="scaled_normal"`` this is the
        noise level relative to each tensor's own standard deviation. Default
        ``0.1``.
    noise : str or callable, optional
        Source of the perturbation :math:`\varepsilon`:

        - ``"scaled_normal"`` (default):
          :math:`\varepsilon = \operatorname{std}(\theta)\,z` with
          :math:`z \sim \mathcal{N}(0, 1)`, i.e. Gaussian noise scaled by the
          per-tensor standard deviation of the pre-shrink weight. Scale aware
          and free of any architectural assumptions (expects at least two
          elements per parameter tensor).
        - ``"normal"``: :math:`\varepsilon = z \sim \mathcal{N}(0, 1)`,
          unscaled.
        - a callable ``(param) -> tensor`` returning :math:`\varepsilon`, with
          the same signature as ``torch.randn_like``. Pass ``torch.randn_like``
          (equivalent to ``"normal"``), the ``randn_like`` of a
          ``StackedRandomGenerator``, any ``torch.nn.init``-style sampler, or
          e.g. ``lambda p: torch.rand_like(p) * 2 - 1`` for a different noise
          distribution.
    include : callable, optional
        Predicate ``(name, param) -> bool`` selecting which parameters to
        perturb. Parameters for which it returns ``False`` are left entirely
        unchanged (not even shrunk). Default: all parameters. For warm-starting,
        pass e.g. ``include=lambda n, p: n in transferred`` to perturb only the
        transferred backbone.
    generator : torch.Generator, optional
        Generator for the built-in Gaussian noise, for reproducibility. Must be
        on the same device as ``module``'s parameters. Ignored when ``noise``
        is a callable.

    Returns
    -------
    torch.nn.Module
        The same ``module``, modified in place (returned for chaining).

    Raises
    ------
    ValueError
        If ``shrink`` or ``perturb`` is negative, or ``noise`` is an unknown
        string.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.nn import shrink_and_perturb_
    >>> model = torch.nn.Linear(4, 4)
    >>> _ = shrink_and_perturb_(model, shrink=0.6, perturb=0.1)
    """
    if shrink < 0.0 or perturb < 0.0:
        raise ValueError(
            f"shrink and perturb must be non-negative, got shrink={shrink}, "
            f"perturb={perturb}"
        )

    if callable(noise):
        noise_fn = noise
    elif noise == "scaled_normal":

        def noise_fn(p: torch.Tensor) -> torch.Tensor:
            z = torch.empty_like(p).normal_(generator=generator)
            return p.detach().std() * z

    elif noise == "normal":

        def noise_fn(p: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(p).normal_(generator=generator)

    else:
        raise ValueError(
            f'Invalid noise "{noise}"; expected a callable or one of {_NOISE_PRESETS}'
        )

    for name, p in module.named_parameters():
        if include is not None and not include(name, p):
            continue
        # eps is computed from the pre-shrink weight (matters for "scaled_normal").
        eps = noise_fn(p)
        p.mul_(shrink).add_(eps, alpha=perturb)
    return module


def _weight_init(shape: tuple, mode: str, fan_in: int, fan_out: int):
    """
    Unified routine for initializing weights and biases.
    This function provides a unified interface for various weight initialization
    strategies like Xavier (Glorot) and Kaiming (He) initializations.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor to initialize. It could represent weights or biases
        of a layer in a neural network.
    mode : str
        The mode/type of initialization to use. Supported values are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
    fan_in : int
        The number of input units in the weight tensor. For convolutional layers,
        this typically represents the number of input channels times the kernel height
        times the kernel width.
    fan_out : int
        The number of output units in the weight tensor. For convolutional layers,
        this typically represents the number of output channels times the kernel height
        times the kernel width.

    Returns
    -------
    torch.Tensor
        The initialized tensor based on the specified mode.

    Raises
    ------
    ValueError
        If the provided `mode` is not one of the supported initialization modes.
    """
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')
