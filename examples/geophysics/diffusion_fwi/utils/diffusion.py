# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import inspect
from typing import Any, Dict, List, Tuple, TypeAlias, Literal
from collections.abc import Callable, Sequence

import torch
import nvtx

from physicsnemo.utils.diffusion import StackedRandomGenerator


class _RemovableHandle:
    r"""
    A handle which provides the capability to remove a hook.

    Parameters
    ----------
    hooks_list : List
        The list of hooks from which to remove the hook.

    """

    def __init__(self, hooks_list: List):
        self.hooks_list = hooks_list
        self.id = None

    def remove(self):
        r"""
        Remove the hook from the list of registered hooks.
        """
        if self.id is not None and self.id < len(self.hooks_list):
            self.hooks_list[self.id] = None


class ModelBasedGuidance:
    r"""
    Guidance function for `Diffusion Posterior Sampling (DPS)
    <https://arxiv.org/abs/2209.14687>`_ based on a generic user-defined model
    (possibly non-linear).
    An instance of ``ModelBasedGuidance`` is a callable object that returns
    the likelihood score :math:`\nabla_{\mathbf{x}_t} \log
    p(\mathbf{y}|\mathbf{x}_t)`, where :math:`\mathbf{y}` is some conditioniong
    variable (observation) that can be predicted by a forward model
    :math:`\mathcal{M}` (called ``guide_model``). This guidance enforces
    consistency between the predicted observation
    :math:`\mathcal{M}(\hat{\mathbf{x}}_0 (\mathbf{x}_t))` and the
    observed data :math:`\mathbf{y}`, where :math:`\hat{\mathbf{x}}_0` is an
    estimate of the clean latent state :math:`\mathbf{x}_0`, usually obtained
    by Tweedie's formula.

    The likelihood score follows a `modified version
    <https://arxiv.org/abs/2506.22780>`_ of the implementation introduced in
    `Score-based data assimilation
    <https://proceedings.neurips.cc/paper_files/paper/2023/hash/7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html>`_.
    It is computed as:

    :math:`\dfrac{S}{1 + M} s_l( \mathbf{y} | \mathbf{x}_t)`, where :math:`S`
    and :math:`M` are scaling parameters and :math:`s_l( \mathbf{y} | \mathbf{x}_t)`
    is the likelihood score. Those are computed fined as:

    - :math:`M = |s_l( \mathbf{y} | \mathbf{x}_t)|` is the magnitude of the
      likelihood score.

    - :math:`S = \text{scale} t^{\text{power}}` if :math:`t < 1` else
      :math:`\text{scale}` is the scaled guidance strength.

    - :math:`s_l( \mathbf{y} | \mathbf{x}_t) = \nabla_{\mathbf{x}_t} \dfrac{1}{2} \log
      p(\mathbf{y}|\mathbf{x}_t) = \nabla_{\mathbf{x}_t} \dfrac{- || \mathbf{y}
      - \mathcal{M}(\hat{\mathbf{x}}_0) ||^{\text{ord}}}{2 (\Sigma_y + \Gamma (\sigma_t /
      \mu)^2)} \log p(\mathbf{y}|\hat{\mathbf{x}}_0 (\mathbf{x}_t))` is the
      likelihood score.

    A ``ModelBasedGuidance`` instance is expected to be the most useful when
    passed to a sampler such as the ``EDMStochasticSampler`` class.

    Parameters
    ----------
    guide_model: Callable[[torch.Tensor], torch.Tensor]
        The forward model :math:`\mathcal{M}` that predicts the observation :math:`\mathbf{y}` from
        the clean latent state :math:`\hat{\mathbf{x}}_0`. Should be a callable
        object that takes as input a single tensor ``x_0_hat`` and returns a
        single tensor ``y_x0`` that is the predicted observation.
    std: float, optional, default=0.075
        The standard deviation :math:`\Sigma_y` of the observation noise.
    gamma: float, optional, default=0.05
        The parameter :math:`\Gamma` of the sclaing, that depends on the
        covariance :math:`\Sigma_x` of the prior.
    mu: float, optional, default=1
        The parameter :math:`\mu` that scales the noise level :math:`\sigma_t`.
    scale: float, optional, default=1
        The :math:`\text{scale}` parameter used to compute the guidance
        strength :math:`S`.
    power: float, optional, default=1
        The :math:`\text{power}` parameter used to compute the guidance
        strength :math:`S`.
    norm_ord: float, optional, default=2
        The order of the norm used to compute the error in the likelihood
        score.
    magnitude_scaling: bool, optional, default=True
        Whether to divide the likelihood score by :math:`1 + M`, where
        :math:`M` is its magnitude.
    model_exec_mode: Literal["batched"], optional, default="batched"
        The execution mode of the ``guide_model``. For ``model_exec_mode="batched"``,
        the ``guide_model`` is expected to be a batched function that takes as input
        a tensor ``x_0_hat`` of shape :math:`(B, *_x)` and returns a tensor ``y_x0``
        of shape :math:`(B, *_y)`. The ``guide_model`` is also expected to be
        differentiable with ``torch.autograd.grad``.

    Forward
    -------
    x: torch.Tensor
        The latent state vector of the diffusion model. Should be of shape
        :math:`(B, *_x)`.
    x_0_hat: torch.Tensor
        The estimate of the clean latent state :math:`\hat{\mathbf{x}}_0`.
        Should be of shape :math:`(B, *_x)`.
    sigma: torch.Tensor
        The noise level :math:`\sigma_t`. Should be of shape :math:`(B,)`.
    y: torch.Tensor
        The observed data :math:`\mathbf{y}`. Should be of shape :math:`(B,
        *_y)`. When used with a sampler such as in instance of the
        ``EDMStochasticSampler`` class, this should be passed as a
        ``guidance_args`` argument.

    Outputs
    -------
    torch.Tensor
        The scaled likelihood score of shape :math:`(B, *_x)`.

    """

    def __init__(
        self,
        guide_model: Callable[[torch.Tensor], torch.Tensor],
        std: float = 0.075,
        gamma: float = 0.05,
        mu: float = 1,
        scale: float = 1,
        power: float = 1,
        norm_ord: float = 2,
        magnitude_scaling: bool = True,
        # NOTE: for now only support model that processes batched inputs. Need
        # more execution modes (e.g. vmap-able, single sample, non-pytorch impl,
        # etc.)
        model_exec_mode: Literal["batched"] = "batched",
    ):
        self.guide_model = guide_model
        self.std = std
        self.gamma = gamma
        self.mu = mu
        self.scale = scale
        self.power = power
        self.norm_ord = norm_ord
        self.magnitude_scaling = magnitude_scaling
        self._score_post_hooks = []
        _valid_model_exec_mode = ["batched"]
        if model_exec_mode in _valid_model_exec_mode:
            self.model_exec_mode = model_exec_mode
        else:
            raise ValueError(
                f"'model_exec_mode' should be one of {', '.join(_valid_model_exec_mode)}, "
                f"but got {model_exec_mode}"
            )

    def register_score_post_hook(self, hook):
        r"""
        Register a post-hook to be executed after the log-likelihood score is
        computed.

        The hook should be a callable with the following signature::

            hook(guidance_instance, x, x_0_hat, sigma, y, log_p) -> None or torch.Tensor

        where ``guidance_instance`` is the ``ModelBasedGuidance`` instance and
        ``log_p`` is the computed log-likelihood tensor of shape :math:`(B,
        *_x)`. The other arguments are the same as the forward method, that is
        the latent state ``x``, the estimate of the clean latent state
        ``x_0_hat``, the noise level ``sigma``, and the observed data ``y``.

        If the hook returns a tensor, it will replace the original ``log_p``
        value. If it returns ``None``, the original ``log_p`` is unchanged
        (allows in-place modifications of the ``log_p`` tensor).

        Parameters
        ----------
        hook : Callable[[ModelBasedGuidance, torch.Tensor], None|torch.Tensor]
            The hook function to register. It will be called with the signature
            specified above.

        Returns
        -------
        RemovableHandle
            A handle that can be used to remove the hook by calling
            ``handle.remove()``.

        Example
        -------

        .. doctest::
           :skip:

            >>> def my_hook(guidance, x, x_0_hat, sigma, y, log_p):
            ...     # Apply some transformation to log_p
            ...     return log_p * 2.0
            >>> guidance = ModelBasedGuidance(my_model)
            >>> handle = guidance.register_score_post_hook(my_hook)
            >>> # Later, remove the hook
            >>> handle.remove()

        """
        handle = _RemovableHandle(self._score_post_hooks)
        self._score_post_hooks.append(hook)
        handle.id = len(self._score_post_hooks) - 1
        return handle

    def _log_likelihood(
        self,
        x: torch.Tensor,
        x_0_hat: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Helper function to compute the log-likelihood.
        """
        # Compute L1 error between model prediction and observation
        # NOTE: for now only Tweedie's formula to estimate clean state x_0
        if self.model_exec_mode == "batched":
            B = y.shape[0]
            y_x0: torch.Tensor = self.guide_model(x_0_hat)  # (B, *_y)
        if y_x0.shape != y.shape:
            raise ValueError(
                f"Expected 'guide_model' output and y to have same shape, "
                f"but got {y_x0.shape} and {y.shape}"
            )
        err1 = torch.abs((y - y_x0)) ** self.norm_ord  # (B, *_y)

        # Compute log-likelihood p(y|x_0_hat)
        var = self.std**2 + self.gamma * (sigma / self.mu) ** 2  # (B,)
        log_p = -0.5 * (err1 / var.view(B, *([1] * (y.ndim - 1)))).sum(
            dim=tuple(range(1, y.ndim))
        )  # (B,)

        # Execute post hooks
        for hook in self._score_post_hooks:
            if hook is not None:
                result = hook(self, x, x_0_hat, sigma, y, log_p)
                if result is not None:
                    log_p = result

        return log_p

    def _get_score(
        self,
        x: torch.Tensor,
        x_0_hat: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Helper function to compute the likelihood score.
        """
        # NOTE: the sum + grad trick only woks with independent samples (i.e.
        # no cross-sample coupling)
        if self.model_exec_mode == "batched":
            log_p = self._log_likelihood(x, x_0_hat, sigma, y)
            dlog_p_dx = torch.autograd.grad(
                outputs=log_p.sum(),  # (,)
                inputs=x,  # (B, *_x)
            )[0]  # (B, *_x)
        return dlog_p_dx

    def __call__(
        self,
        x: torch.Tensor,
        x_0_hat: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        B = x.shape[0]
        ndim = x.ndim

        # Parameters validation
        if sigma.shape != (B,):
            raise ValueError(
                f"Expected sigma to have shape {(B,)}, but got {sigma.shape}"
            )
        if y.shape[0] != B:
            raise ValueError(f"Expected y to have batch size {B}, but got {y.shape[0]}")
        if x_0_hat.shape != x.shape:
            raise ValueError(
                f"Expected x_0_hat and x to have same shape, "
                f"but got {x_0_hat.shape} and {x.shape}"
            )

        # Compute likelihood score
        score = self._get_score(x, x_0_hat, sigma, y)  # (B, *_x)

        # Scale the likelihood score
        scale = torch.where(
            sigma < 1, self.scale * sigma.pow(self.power), self.scale
        ).view(B, *([1] * (ndim - 1)))  # (B, 1, ..., 1)
        if self.magnitude_scaling:
            score_mag = torch.abs(score).mean(
                dim=tuple(range(1, ndim)), keepdim=True
            )  # (B, 1, ..., 1)
        else:
            score_mag = 0
        score_scaled = (
            score * scale * sigma.view(B, *([1] * (ndim - 1))) / (1 + score_mag)
        )  # (B, *_x)

        return score_scaled


class DataConsistencyGuidance(ModelBasedGuidance):
    r"""
    ``DataConsistencyGuidance`` is a specific type of ``ModelBasedGuidance``
    where the model :math:`\mathcal{M}` used in the guidance is a (linear)
    measurement operator, defined by a relation of the form :math:`\mathcal{M}
    (\hat{\mathbf{x}}_0) = \text{Mask} (\mathbf{x}_0)`, where :math:`\text{Mask}`
    is a mask operator that selects a subset of clean latent state
    :math:`\hat{\mathbf{x}}_0`.

    This guidance is useful for applications such as image inpainting, channel
    infilling, etc.

    It returns the scaled likelihood score :math:`\nabla_{\mathbf{x}_t} \log
    p(\mathbf{y}|\mathbf{x}_t)`, in a similar way as ``ModelBasedGuidance``.
    Most of the parameters are the same as ``ModelBasedGuidance``.

    Parameters
    ----------
    std: float, optional, default=0.075
        The standard deviation :math:`\Sigma_y` of the observation noise.
    gamma: float, optional, default=0.05
        The parameter :math:`\Gamma` of the sclaing, that depends on the
        covariance :math:`\Sigma_x` of the prior.
    mu: float, optional, default=1
        The parameter :math:`\mu` that scales the noise level :math:`\sigma_t`.
    scale: float, optional, default=1
        The :math:`\text{scale}` parameter used to compute the guidance
        strength :math:`S`.
    power: float, optional, default=1
        The :math:`\text{power}` parameter used to compute the guidance
        strength :math:`S`.
    norm_ord: float, optional, default=2
        The order of the norm used to compute the error in the likelihood
        score.
    magnitude_scaling: bool, optional, default=True
        Whether to divide the likelihood score by :math:`1 + M`, where
        :math:`M` is its magnitude.

    Forward
    -------
    x: torch.Tensor
        The latent state vector of the diffusion model. Should be of shape
        :math:`(B, *_x)`.
    x_0_hat: torch.Tensor
        The estimate of the clean latent state :math:`\hat{\mathbf{x}}_0`.
        Should be of shape :math:`(B, *_x)`.
    sigma: torch.Tensor
        The noise level :math:`\sigma_t`. Should be of shape :math:`(B,)`.
    y: torch.Tensor
        The observed data :math:`\mathbf{y}`. Should have the same shape as
        ``x``, that is :math:`(B, *_x)`. When used with a sampler such as in
        instance of the ``EDMStochasticSampler`` class, this should be passed
        as a ``guidance_args`` argument.
    mask: torch.Tensor
        The measurement operator :math:`\text{Mask}`. Should be a tensor of
        boolean values of shape :math:`(B, *_x)`. Values that are ``True``
        correspond to the observed data, and values that are ``False`` correspond
        to the unobserved data (ignored in the guidance). When used with a
        sampler, this should be passed as a ``guidance_args`` argument.

    Outputs
    -------
    torch.Tensor
        The scaled likelihood score of shape :math:`(B, *_x)`.
    """

    def __init__(
        self,
        std: float = 0.075,
        gamma: float = 0.05,
        mu: float = 1,
        scale: float = 1,
        power: float = 1,
        norm_ord: float = 2,
        magnitude_scaling: bool = True,
    ):
        super().__init__(
            guide_model=lambda x: x,
            std=std,
            gamma=gamma,
            mu=mu,
            scale=scale,
            power=power,
            norm_ord=norm_ord,
            magnitude_scaling=magnitude_scaling,
            model_exec_mode="batched",
        )

    def __call__(
        self,
        x: torch.Tensor,
        x_0_hat: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.shape != x.shape:
            raise ValueError(
                f"Expected mask and x to have same shape, "
                f"but got {mask.shape} and {x.shape}"
            )
        if y.shape != x.shape:
            raise ValueError(
                f"Expected y and x to have same shape, but got {y.shape} and {x.shape}"
            )
        y_masked = torch.where(mask, y, 0.0)  # (B, *_x)
        return super().__call__(self, x, x_0_hat, sigma, y_masked)


# Some type annotations
_Guidance: TypeAlias = (
    ModelBasedGuidance | DataConsistencyGuidance | Callable[..., torch.Tensor]
)
_SamplerFn: TypeAlias = Callable[
    [torch.Tensor, Dict[str, torch.Tensor], Any, Any], torch.Tensor
]
_DiffusionModel = Callable[
    [torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Any, Any], torch.Tensor
]


def DiffusionAdapter(
    model: torch.nn.Module, args_map: Tuple[str, str, Dict[str, str]]
) -> _DiffusionModel:
    r"""
    Creates a thin wrapper around a module to convert it into a
    diffusion model compatible with other diffusion utilities.

    This wrapper modifies the signature of a model's forward method to match the
    expected interface for diffusion models. It converts a model with
    an original signature ``model(arg1, ..., argN, kwarg1=val1, ..., kwargM=valM,
    **model_kwargs)`` into a model with signature
    ``wrapper(x, sigma, condition, wrapper_disabled=False, **wrapper_kwargs)``.

    Parameters
    ----------
    model : torch.nn.Module
        The model to wrap with the diffusion adapter interface.
    args_map : Tuple[str, str, Dict[str, str]]
        A tuple containing 3 elements:
        - First element: the name of the parameter in the original model's forward
          method that the latent state `x` should be mapped to.
        - Second element: the name of the parameter in the original model's forward
          method that the noise level ``sigma`` should be mapped to.
        - Third element: a dictionary mapping keys in the `cond` dictionary
          to parameter names in the original model's forward method.

    Forward
    -------
    x : torch.Tensor
        The latent state of the diffusion model, typically of shape
        :math:`(B, *)`.
    sigma : torch.Tensor
        The noise level :math:`\sigma_t`. Should be of shape :math:`(B,)`.
    cond : Dict[str, torch.Tensor]
        A dictionary of conditioning variables. Keys are strings identifying
        the conditioning variables names, and values are tensors used for
        conditioning.
    wrapper_disabled : bool, optional, default=False
        Flag to disable the wrapper functionality. When ``True``, the forward
        method reverts to the original model's signature.
    **wrapper_kwargs : Any, optional
        Additional arguments to pass to the original model's forward method.
        Should include all arguments from the original signature that are not
        referenced in ``args_map``. This includes both positional and keyword
        arguments from the original signature, all converted to keyword
        arguments.

    Outputs
    -------
    output : Any
        The output from the wrapped model's forward method, with the same
        type and shape as the original model would return.

    Notes
    -----
    This is a thin wrapper that only holds references to the original model's
    attributes. Any modification of attributes in the wrapper is reflected in the
    original model, and vice versa.

    Example
    -------
    >>> class Model(torch.nn.Module):
    >>>    def __init__(self):
    >>>        super().__init__()
    >>>        self.a = torch.tensor(10.0)
    >>>    def forward(self, x, y, z, u=4, v=5, w=6, **kwargs):
    >>>        return self.a * x, self.a * y, self.a * z, self.a * u, self.a * v, self.a * w
    >>> model = Model()
    >>> wrapper = DiffusionAdapter(
    >>>     model=model,
    >>>     args_map=("w", "u", {"j": "x", "k": "v"})
    >>> )
    >>> x = torch.tensor(1)
    >>> y = torch.tensor(2)
    >>> z = torch.tensor(3)
    >>> u = torch.tensor(-1)
    >>> v = torch.tensor(-2)
    >>> w = torch.tensor(-3)
    >>> model(x, y, z, u=u, v=v, w=w)
    (tensor(10.), tensor(20.), tensor(30.), tensor(-10.), tensor(-20.), tensor(-30.))
    >>> # Can be called with modified signature (x, t, cond, **wrapper_kwargs)
    >>> wrapper(x, w, {"j": y, "k": z}, z=u, y=v)
    (tensor(20.), tensor(-20.), tensor(-10.), tensor(-30.), tensor(30.), tensor(10.))
    >>> # Can be called with original signature with wrapper_disabled=True
    >>> wrapper(x, y, z, wrapper_disabled=True, u=u, v=v, w=w)
    (tensor(10.), tensor(20.), tensor(30.), tensor(-10.), tensor(-20.), tensor(-30.))
    """
    # Safety checks: make sure we don't map twice to the same argument (i.e.
    # targets in args_map are unique)
    if len(args_map[2]) != len(set(args_map[2].values())):
        raise ValueError(
            "Cannot map two values in 'cond' to the same target forward argument."
        )
    if any(arg_name == args_map[0] for arg_name in args_map[2].values()):
        raise ValueError(
            "Cannot map 'x' and a value in 'cond' to the same target forward argument."
        )
    if any(arg_name == args_map[1] for arg_name in args_map[2].values()):
        raise ValueError(
            "Cannot map 't' and a value in 'cond' to the same target forward argument."
        )

    # Unbound original origional forward method
    _orig_forward: Callable[..., Any] = model.__class__.forward

    # Signature of original forward method
    sig = inspect.signature(_orig_forward)

    # Placeholders
    _NoArg, _condArg, _kwArg = object(), object(), object()
    _xArg, _sigmaArg = object(), object()

    # Process each parameter in the original forward method signature
    # and do the mapping if the parameter is a target specified  in args_map
    is_mapped: List = [
        False,
        False,
        {k: False for k in args_map[2].keys()},
    ]
    sig_map: Dict[str, Tuple[int, object] | Tuple[int, object, str]] = {}
    for i, p in enumerate(sig.parameters.values()):
        # Skip 'self' argument
        if i == 0:
            continue
        # For now we don't support *args because it's not clear how to pass those
        # to the original forward method
        if p.kind == p.VAR_POSITIONAL:
            raise NotImplementedError("*args is not supported as a forward argument")
        # Avoid conflict with wrapper_disabled in the new forward
        elif p.name == "wrapper_disabled":
            raise ValueError(
                "'wrapper_disabled' kwarg is not supported as a forward argument"
            )
        # Skip **kwargs
        elif p.kind == p.VAR_KEYWORD:
            continue
        # Argument targetted for x (state vector)
        elif p.name == args_map[0]:
            sig_map[p.name] = (i - 1, _xArg)
            is_mapped[0] = True
        # Argument targetted for sigma (noise level)
        elif p.name == args_map[1]:
            sig_map[p.name] = (i - 1, _sigmaArg)
            is_mapped[1] = True
        # Arguments targetted for condition
        elif p.name in args_map[2].values():
            cond_key = next(k for k, v in args_map[2].items() if v == p.name)
            sig_map[p.name] = (i - 1, _condArg, cond_key)
            is_mapped[2][cond_key] = True
        # Signature argument that is not a target in args_map
        else:
            sig_map[p.name] = (i - 1, _kwArg)
    # Safety check: make sure that we mapped all the variables in `args_map`
    if not is_mapped[0] or not is_mapped[1] or not all(is_mapped[2].values()):
        raise ValueError(
            f"Not all variables in 'args_map' were mapped to a forward argument. "
            f"Detail: {is_mapped}"
        )

    # Forward with modified signature
    def _forward(self, *args, wrapper_disabled=False, **kwargs):
        if wrapper_disabled:
            return _orig_forward(self, *args, **kwargs)
        # Extract x (state vector) and condition from args
        x, sigma, cond = args[0], args[1], args[2]

        # Build a list of arguments to pass to the original forward method
        args_and_kwargs = [_NoArg for _ in range(len(sig_map))]
        for param_name, (idx, arg_type, *cond_key) in sig_map.items():
            if arg_type is _xArg:
                args_and_kwargs[idx] = x
            elif arg_type is _sigmaArg:
                args_and_kwargs[idx] = sigma
            elif arg_type is _condArg:
                args_and_kwargs[idx] = cond[cond_key[0]]
            elif arg_type is _kwArg:
                args_and_kwargs[idx] = kwargs.pop(param_name)

        # Safety checks
        if _NoArg in args_and_kwargs:
            raise ValueError("Some arguments are missing from 'args_map' or 'kwargs'")

        return _orig_forward(self, *args_and_kwargs, **kwargs)

    # Build a throw-away subclass that installs the override
    subclass = type(
        f"DiffusionAdapter{model.__class__.__name__}",
        (model.__class__,),
        {"forward": _forward},
    )

    # Allocate a blank instance of that subclass
    proxy = object.__new__(subclass)

    # Point its attribute storage at the original one (shared state)
    proxy.__dict__ = model.__dict__

    return proxy


def generate(
    sampler_fn: _SamplerFn,
    x_channels: int,
    x_resolution: Tuple[int, ...],
    rank_batches: List[List[int]] | List[torch.Tensor],
    cond: Dict[str, torch.Tensor],
    device: torch.device,
    sampler_kwargs: Dict[str, Any] = {},
) -> torch.Tensor:
    r"""
    Function to generate samples from a diffusion model. It starts by
    initializing a batch of noisy latent states :math:`\mathbf{x}_T` and then generates
    a batch of clean samples :math:`\mathbf{x}_0` by applying the ``sampler_fn`` function.
    It supports in addition generation minibatch by minibatch by splitting the
    seeds in ``rank_batches``.

    The ``sampler_fn`` function is expected to have the following signature:
    ``sampler_fn(x, cond, **sampler_kwargs)``, where ``x`` is the latent state and
    ``cond`` is the conditioning variables, as specified below. It should return
    a single tensor corresponding to a batch of generated samples. Typically,
    the ``sampler_fn`` function is an instance of ``EDMStochasticSampler``.

    Parameters
    ----------
    sampler_fn : Callable
        Function used to generate samples from the diffusion model.
    x_channels : int
        Number of channels :math:`C_{\mathbf{x}}` for the latent state
        :math:`\mathbf{x}`.
    x_resolution : Tuple[int, ...]
        Spatial resolution :math:`\mathbf{x}`. For example, for a 2D image it
        should be of the form :math:`(H, W)`, where :math:`H` and :math:`W` are
        the height and width of the image, respectively.
    rank_batches : List[List[int]] | List[torch.Tensor]
        List of mini-batches of seeds to process. Each mini-batch is a list of
        seeds. The mini-batches are generated sequentially, and the final generated
        samples are concatenated across the batch dimension. This is typically used
        to generate large ensembles that do not fit in device memory.
    cond : Dict[str, torch.Tensor]
        Dictionary of conditioning variables. Keys are strings identifying the
        conditioning variables names, and values are tensors used for
        conditional generation. Can be set to ``{}`` for unconditional
        generation.
    device : torch.device
        Device to perform computations.
    sampler_kwargs : Dict[str, Any], optional, default={}
        Additional keyword arguments to pass to the ``sampler_fn`` function.

    Returns
    -------
    torch.Tensor
        Generated samples. Has shape ``(B, x_channels, *x_resolution)``, where
        ``B`` is the total number of seeds in ``rank_batches``.
    """

    # Loop over batches
    x_generated = []
    for batch_seeds in rank_batches:
        with nvtx.annotate(f"generate {len(x_generated)}", color="rapids"):
            B = len(batch_seeds)
            if B == 0:
                continue

            # Initialize random generator, and generate latents
            rnd = StackedRandomGenerator(device, batch_seeds)
            x_T = rnd.randn(
                (B, x_channels) + x_resolution,
                device=device,
            ).to(memory_format=torch.channels_last)

            # Call the sampler function
            x_0: torch.Tensor = sampler_fn(x_T, cond, **sampler_kwargs)

            x_generated.append(x_0)
    return torch.cat(x_generated)


class EDMStochasticSampler:
    r"""
    Stochastic sampler proposed in the `EDM paper
    <https://arxiv.org/abs/2206.00364>`_, with optional guidance.
    The sampler starts from a batch of noisy latent states :math:`\mathbf{x}_T`
    and generates a batch of clean samples :math:`\mathbf{x}_0` by iteratively denoising
    the noisy latent states.

    The diffusion model is expected to be called with:
    ``x_0_hat = model(x, sigma, cond, *model_args, **model_kwargs)``, where ``x`` is the
    latent state, ``sigma`` is the noise level, ``cond`` is the conditioning
    variables, and ``*model_args`` and ``**model_kwargs`` are additional
    arguments to pass to the model (see below for details on the expected
    arguments). The model should be an :math:`\mathbf{x}_0`-prediction model.
    It should return a tensor :math:`\hat{\mathbf{x}}_0` of
    same shape as ``x``, that is an estimate of the clean latent state
    :math:`\mathbf{x}_0`.

    Guidance sampling (e.g. posterior sampling, classifier guidance, etc.) can be
    enabled by passing one or multiple ``guidance`` functions
    to the sampler. The outputs of the guidance functions are summed and added
    to the score function as a correction or drift term.
    Each guidance function must be an instance of the available guidance types (e.g.
    ``ModelBasedGuidance`` for posterior sampling based on consistency with a nonlinear
    model, ``DataConsistencyGuidance`` for guidance based data measured at
    specific locations, etc.)
    For example, in the case of posterior sampling, the guidance function
    should be an instance of ``ModelBasedGuidance`` that returns the
    likelihood score :math:`\nabla_{\mathbf{x}} \log p(\mathbf{y}|\mathbf{x}_t)`,
    which is a tensor of same shape as ``x``, and where :math:`\mathbf{y}` is some
    conditioniong variable.

    Parameters
    ----------
    model: _DiffusionModel
        The denoising diffusion model to use in the sampling process. Should be
        an :math:`\mathbf{x}_0`-prediction model.
    num_steps: int, optional, default=18
        Number of time steps for the sampler.
    sigma_min: float, optional, default=0.002
        Minimum noise level. If the model has a ``sigma_min`` attribute, the
        larger value between the two will be used.
    sigma_max: float, optional, default=800
        Maximum noise level. If the model has a ``sigma_max`` attribute, the
        smaller value between the two will be used.
    rho: float, optional, default=7
        Exponent used in the time step discretization.
    S_churn: float, optional, default=0
        Churn parameter controlling the level of noise added in each step.
    S_min: float, optional, default=0
        Minimum time step for applying churn.
    S_max: float, optional, default=float("inf")
        Maximum time step for applying churn.
    S_noise: float, optional, default=1
        Noise scaling factor applied during the churn step.

    Forward
    -------
    x: torch.Tensor
        The noisy latent state used as the initial input for the sampler.
        Typically pure noise :math:`\mathbf{x}_T`.
        Should have shape :math:`(B, *)`, where :math:`B` is the batch size and
        :math:`*` is any number of dimensions.
    cond: Dict[str, torch.Tensor]
        Dictionary of conditioning variables. Keys are strings identifying the
        conditioning variables names, and values are tensors used for
        conditioning.
    model_args: Tuple, optional, default=()
        Additional positional arguments to pass to the model.
    model_kwargs: Dict[str, Any], optional, default={}
        Additional keyword arguments to pass to the model.
    guidance: _Guidance | Sequence[_Guidance] | None, optional, default=None
        Guidance function that is added as a correction to the score function (computed by
        ``model``). Typically used for posterior sampling, classifier guidance,
        etc. Also support multiple guidance functions by passing a list or tuple.
    guidance_args: Tuple | Sequence[Tuple], optional, default=()
        Additional positional arguments to pass to the guidance function.
        If multiple guidance functions are passed, this should be a list or tuple
        of the same length as the number of guidance functions.
    guidance_kwargs: Dict[str, Any] | Sequence[Dict[str, Any]], optional, default={}
        Additional keyword arguments to pass to the guidance function.
        If multiple guidance functions are passed, this should be a list or tuple
        of the same length as the number of guidance functions.
    guidance_second_order: bool | Sequence[bool], optional, default=False
        Whether to compute the guidance function in the second order
        correction. If ``True``, the guidance function is computed twice, while
        if ``False``, it is computed only once, which can typically save some
        computation for guidance functions that are expensive to compute.
        If multiple guidance functions are passed, this should be
        a list or tuple of the same length as the number of guidance functions.

    Outputs
    -------
    torch.Tensor
        The final clean latent state :math:`\mathbf{x}_0` produced by the
        sampler. Same shape :math:`(B, *)` as ``x``.
    """

    def __init__(
        self,
        model: _DiffusionModel,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 800,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
    ):
        self.model = model
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

    def __call__(
        self,
        x: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        model_args: Tuple = (),
        model_kwargs: Dict[str, Any] = {},
        guidance: _Guidance | Sequence[_Guidance] | None = None,
        guidance_args: Tuple | Sequence[Tuple] = (),
        guidance_kwargs: Dict[str, Any] | Sequence[Dict[str, Any]] = {},
        guidance_second_order: bool | Sequence[bool] = False,
    ) -> torch.Tensor:
        # Set container structures for guidance functions
        if guidance is None:
            guidances = []
            guidances_args = []
            guidances_kwargs = []
            guidances_second_order = []
        elif not isinstance(guidance, (list, tuple)):
            guidances = [guidance]
            guidances_args = [guidance_args]
            guidances_kwargs = [guidance_kwargs]
            guidances_second_order = [guidance_second_order]
        elif (
            isinstance(guidance, (list, tuple))
            and isinstance(guidance_args, (list, tuple))
            and isinstance(guidance_kwargs, (list, tuple))
            and isinstance(guidance_second_order, (list, tuple))
        ):
            guidances = guidance
            guidances_args = guidance_args
            guidances_kwargs = guidance_kwargs
            guidances_second_order = guidance_second_order
        else:
            raise ValueError(
                "When multiple guidance functions are passed, 'guidance', "
                "'guidance_args', 'guidance_kwargs', and 'guidance_second_order' "
                "must be lists or tuples of the same length"
            )
        if not (
            len(guidances)
            == len(guidances_args)
            == len(guidances_kwargs)
            == len(guidances_second_order)
        ):
            raise ValueError(
                f"Number of guidance functions, arguments, keyword "
                f"arguments, and second order correction must match, "
                f"but got {len(guidances)}, {len(guidances_args)}, "
                f"{len(guidances_kwargs)}, and {len(guidances_second_order)}"
            )

        # Determine if we need to differentiate through the model
        req_grad: bool = False
        req_grad_sec_ord: bool = False
        for gd, gd_sec_ord in zip(guidances, guidances_second_order):
            if isinstance(gd, (ModelBasedGuidance, DataConsistencyGuidance)):
                req_grad: bool = True
                if gd_sec_ord:
                    req_grad_sec_ord: bool = True

        B = x.shape[0]

        # Adjust noise levels based on what's supported by the network.
        # Proposed EDM sampler (Algorithm 2) with minor changes to enable
        # posterior sampling
        if hasattr(self.model, "sigma_min"):
            sigma_min = max(self.sigma_min, self.model.sigma_min)
        else:
            sigma_min = self.sigma_min
        if hasattr(self.model, "sigma_max"):
            sigma_max = min(self.sigma_max, self.model.sigma_max)
        else:
            sigma_max = self.sigma_max
        if hasattr(self.model, "round_sigma") and callable(self.model.round_sigma):
            round_sigma = self.model.round_sigma
        else:
            round_sigma = torch.as_tensor

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, device=x.device)
        t_steps = (
            sigma_max ** (1 / self.rho)
            + step_indices
            / (self.num_steps - 1)
            * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat(
            [round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Main sampling loop.
        x_next = x * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # NOTE: break the computational graph here to save memory when
            # computing the guidance terms --> Cannot backpropagate through the
            # sampler, even when guidance is disabled
            x_cur = x_next.detach()

            # Increase noise temporarily.
            gamma = (
                self.S_churn / self.num_steps
                if self.S_min <= t_cur <= self.S_max
                else 0
            )
            t_hat = round_sigma(t_cur + gamma * t_cur)
            x_hat: torch.Tensor = (
                x_cur
                + (t_hat**2 - t_cur**2).sqrt() * self.S_noise * torch.randn_like(x_cur)
            ).to(x.device)

            # Move conditioning to the device
            for key, value in cond.items():
                cond[key] = value.to(x.device)

            # Activate gradient computation if needed for guidance
            with torch.set_grad_enabled(req_grad):
                if req_grad:
                    x_hat_in = x_hat.clone().detach().requires_grad_(True)
                else:
                    x_hat_in = x_hat

                x_0_hat = self.model(
                    x_hat_in,
                    t_hat.expand(
                        B,
                    ),
                    cond,
                    *model_args,
                    **model_kwargs,
                )

                # Guidance terms (e.g. posterior sampling, etc...)
                # Guidance terms required for 2nd order correction are computed
                # twice, while other guidance terms are only computed once to
                # save cost
                gd_sum = 0
                gd_sum_sec_ord = 0
                if guidances:
                    for gd, gd_args, gd_kwargs, gd_sec_ord in zip(
                        guidances,
                        guidances_args,
                        guidances_kwargs,
                        guidances_second_order,
                    ):
                        if isinstance(
                            guidance, (ModelBasedGuidance, DataConsistencyGuidance)
                        ):
                            gd_val = gd(
                                x_hat_in,
                                x_0_hat,
                                t_hat.expand(
                                    B,
                                ),
                                *gd_args,
                                **gd_kwargs,
                            )
                        else:
                            raise ValueError(f"Unsupported guidance type: {type(gd)}")
                        if gd_sec_ord:
                            gd_sum += gd_val
                        else:
                            # Count twice since we only compute once
                            gd_sum_sec_ord += 2 * gd_val

            d_cur = (x_hat - x_0_hat) / t_hat - gd_sum
            x_next = x_hat + (t_next - t_hat) * d_cur

            # 2nd order correction
            if i < self.num_steps - 1:
                x_next = x_next.to(x.device)

                # Activate gradient computation if needed for guidance
                with torch.set_grad_enabled(req_grad_sec_ord):
                    if req_grad_sec_ord:
                        x_next_in = x_next.clone().detach().requires_grad_(True)
                    else:
                        x_next_in = x_next

                    x_0_hat_next = self.model(
                        x_next_in,
                        t_next.expand(
                            B,
                        ),
                        cond,
                        *model_args,
                        **model_kwargs,
                    )

                    # Only recompute guidance terms specifically required in
                    # the 2nd correction
                    if guidances:
                        for gd, gd_args, gd_kwargs, gd_sec_ord in zip(
                            guidances,
                            guidances_args,
                            guidances_kwargs,
                            guidances_second_order,
                        ):
                            if gd_sec_ord:
                                if isinstance(
                                    guidance,
                                    (ModelBasedGuidance, DataConsistencyGuidance),
                                ):
                                    gd_sum_sec_ord += gd(
                                        x_next_in,
                                        x_0_hat_next,
                                        t_next.expand(
                                            B,
                                        ),
                                        *gd_args,
                                        **gd_kwargs,
                                    )
                                else:
                                    raise ValueError(
                                        f"Unsupported guidance type: {type(gd)}"
                                    )

                d_prime = (x_next - x_0_hat_next) / t_next - gd_sum_sec_ord
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        return x_next
