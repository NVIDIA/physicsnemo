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
import warnings
from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
from torch.distributed.tensor import DTensor
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


def _resolve_noise(
    noise: Union[
        Literal["scaled_normal", "normal"], Callable[[torch.Tensor], torch.Tensor]
    ],
    generator: Optional[torch.Generator],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the ``(param) -> noise`` sampler for :func:`shrink_and_perturb_`.

    Presets read the live (pre-shrink) parameter only to *read* its std and
    allocate fresh noise, never to mutate it. A user callable instead receives
    fresh scratch storage carrying the parameter's shape/dtype/device/layout, so
    it can neither alias nor overwrite the weight (supporting both out-of-place
    samplers like ``torch.randn_like`` and in-place ones like
    ``torch.nn.init.normal_``).
    """
    if callable(noise):

        def make_eps(p: torch.Tensor) -> torch.Tensor:
            return noise(torch.empty_like(p))

    elif noise == "scaled_normal":

        def make_eps(p: torch.Tensor) -> torch.Tensor:
            if p.numel() < 2:
                # std is undefined for a single element; shrink only, no noise.
                return torch.zeros_like(p)
            z = torch.randn_like(p, generator=generator)
            return z.mul_(p.detach().std())  # scale in place: one temporary

    elif noise == "normal":

        def make_eps(p: torch.Tensor) -> torch.Tensor:
            return torch.randn_like(p, generator=generator)

    else:
        raise ValueError(
            f'Invalid noise "{noise}"; expected a callable or one of {_NOISE_PRESETS}'
        )

    return make_eps


def _validate_noise(eps: torch.Tensor, p: torch.Tensor, name: str) -> None:
    """Reject noise incompatible with parameter ``name`` before it is applied.

    Checking up front keeps each parameter's update atomic: a bad value raises
    without having partially modified the parameter.
    """
    if eps.shape != p.shape:
        raise ValueError(
            f"noise for parameter '{name}' returned shape "
            f"{tuple(eps.shape)}, expected {tuple(p.shape)}"
        )
    if eps.device != p.device:
        raise ValueError(
            f"noise for parameter '{name}' is on device {eps.device}, "
            f"expected {p.device}"
        )
    if torch.is_complex(eps) and not torch.is_complex(p):
        raise ValueError(
            f"noise for parameter '{name}' is complex but the parameter is real"
        )


@torch.no_grad()
def shrink_and_perturb_(
    module: torch.nn.Module,
    shrink: float = 0.5,
    perturb: float = 0.1,
    *,
    noise: Union[
        Literal["scaled_normal", "normal"], Callable[[torch.Tensor], torch.Tensor]
    ] = "scaled_normal",
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
    https://arxiv.org/pdf/1910.08475

    The operation is only meaningful on **pretrained** weights: applied to a
    fresh initialization it merely rescales and re-noises random values.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters are perturbed in place. Buffers (e.g.
        batch-norm running statistics) are left untouched. Only whole, unsharded
        parameters are supported: a sharded ``DTensor`` parameter (e.g. from
        FSDP2 ``fully_shard`` or tensor parallelism) raises
        ``NotImplementedError``, so apply this before distributed wrapping.
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
          and free of any architectural assumptions. A scalar parameter (a
          single element) has no defined spread and is shrunk only, without
          added noise.
        - ``"normal"``: :math:`\varepsilon = z \sim \mathcal{N}(0, 1)`,
          unscaled.
        - a callable producing :math:`\varepsilon`. It is handed *fresh scratch
          storage* carrying the parameter's shape, dtype, device, and layout
          (never the live parameter, so it can neither alias nor overwrite the
          weight) and must return a tensor of that shape. It may fill the given
          tensor in place (e.g. ``torch.nn.init.normal_``) or return a new one
          (e.g. ``torch.randn_like``, equivalent to ``"normal"``);
          ``lambda p: torch.rand_like(p) * 2 - 1`` gives a different
          distribution.

        The built-in presets sample Gaussian noise and therefore require
        floating-point (or complex) parameters.
    include : callable, optional
        Predicate ``(name, param) -> bool`` selecting which parameters to
        perturb. Parameters for which it returns ``False`` are left entirely
        unchanged (not even shrunk). Default: all parameters. For warm-starting,
        pass e.g. ``include=lambda n, p: n in transferred`` to perturb only the
        transferred backbone. Tied parameters are visited under every alias so
        the predicate can match any checkpoint key, but each underlying tensor
        is updated at most once. A warning is issued if an explicit ``include``
        matches no parameters.
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
        If ``shrink`` or ``perturb`` is negative or non-finite, ``noise`` is an
        unknown string, a preset ``generator`` is on a different device than the
        module's parameters, or a ``noise`` callable returns a tensor whose
        shape or device is incompatible with its parameter (or complex noise for
        a real parameter).
    NotImplementedError
        If any selected parameter is a sharded ``DTensor`` (only whole,
        unsharded tensors are supported).

    Notes
    -----
    Intended warm-start workflow: load the pretrained weights, apply this to the
    transferred parameters, then (re)create the optimizer, and only afterwards
    ``torch.compile`` / wrap in DDP or FSDP. Run it *before* distributed
    wrapping.

    ``"scaled_normal"`` is a deliberately model-agnostic variant: it scales the
    noise by the *pretrained tensor's* own standard deviation rather than by a
    freshly re-initialized network's per-layer initializer variance.
    It needs no architecture knowledge and matches the
    validated production recipe.

    The operation is not transactional across parameters: each parameter's noise
    is validated before that parameter is modified, but if a later parameter
    raises, earlier ones are already updated.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.nn import shrink_and_perturb_
    >>> model = torch.nn.Linear(4, 4)
    >>> model.load_state_dict(pretrained_state)  # doctest: +SKIP
    >>> _ = shrink_and_perturb_(model, shrink=0.6, perturb=0.1)
    """
    if (
        not math.isfinite(shrink)
        or not math.isfinite(perturb)
        or shrink < 0.0
        or perturb < 0.0
    ):
        raise ValueError(
            f"shrink and perturb must be finite and non-negative, got "
            f"shrink={shrink}, perturb={perturb}"
        )

    if any(isinstance(p, DTensor) for p in module.parameters()):
        raise NotImplementedError(
            "shrink_and_perturb_ does not support sharded DTensor parameters "
            "(e.g. from FSDP2 `fully_shard` or tensor parallelism); apply it to "
            "the full model before distributed wrapping."
        )

    # The preset samplers draw on `generator`, which must sit on the device of
    # every parameter it seeds (a callable ignores it, and perturb == 0 draws
    # nothing, so skip the check in those cases).
    if generator is not None and not callable(noise) and perturb != 0.0:
        bad = {p.device for p in module.parameters()} - {generator.device}
        if bad:
            raise ValueError(
                f"generator is on {generator.device}, but the module has "
                f"parameters on {sorted(map(str, bad))}; the generator must "
                f"match the parameter device(s)"
            )

    make_eps = _resolve_noise(noise, generator)

    # Visit tied parameters under every alias so `include` can match any
    # checkpoint key, but update each underlying tensor at most once.
    seen: set[int] = set()
    matched = False
    for name, p in module.named_parameters(remove_duplicate=False):
        if include is not None and not include(name, p):
            continue
        matched = True
        if id(p) in seen:
            continue
        seen.add(id(p))

        if perturb == 0.0:
            # Pure shrink (a no-op at shrink == 1); never draw from the generator.
            if shrink != 1.0:
                p.mul_(shrink)
            continue

        eps = make_eps(p)
        # Validate before mutating so a rejected noise value leaves this
        # parameter untouched (the op is not transactional across parameters).
        _validate_noise(eps, p, name)
        p.mul_(shrink).add_(eps, alpha=perturb)

    if include is not None and not matched:
        warnings.warn(
            "shrink_and_perturb_: `include` matched no parameters; "
            "nothing was modified.",
            stacklevel=2,
        )
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
