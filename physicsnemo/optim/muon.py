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

"""Fused Muon optimizer.

PyTorch's :class:`torch.optim.Muon` runs the Newton-Schulz orthogonalization
one parameter at a time -- its functional ``muon()`` explicitly raises for the
``foreach`` path -- so a model with many 2-D weight matrices issues hundreds of
tiny, serial, launch-bound matmuls per step. The Newton-Schulz iteration of one
parameter is independent of every other, so parameters that share a shape can be
stacked and orthogonalized together with batched matmuls (``bmm`` / ``baddbmm``).

``torch.bmm`` computes each batch element independently, so the batched result is
the same as the per-parameter loop; only the number of kernel launches changes,
from ``O(num_params * ns_steps)`` to ``O(num_shape_groups * ns_steps)``.

This module provides :class:`Muon`, a drop-in replacement whose constructor
signature, hyperparameter semantics, and ``momentum_buffer`` optimizer-state key
match :class:`torch.optim.Muon`, so checkpoints remain interchangeable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = ["Muon"]

# Constants from Keller Jordan's Muon post (match torch.optim.Muon defaults):
# https://kellerjordan.github.io/posts/muon/
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """Learning-rate adjustment for a parameter, matching ``torch.optim.Muon``."""
    A, B = param_shape[0], param_shape[1]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


def _batched_newton_schulz(
    updates: Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    """Batched Newton-Schulz orthogonalization of a stack of 2-D matrices.

    Performs, for every matrix in the batch independently, the same quintic
    Newton-Schulz iteration as :func:`torch.optim._muon._zeropower_via_newtonschulz`,
    but with batched matmuls so a whole group of equally-shaped parameters is
    orthogonalized in a handful of kernel launches.

    Parameters
    ----------
    updates : torch.Tensor
        Stack of update matrices of shape ``(G, M, N)``.
    ns_coefficients : tuple[float, float, float]
        Quintic polynomial coefficients ``(a, b, c)``.
    ns_steps : int
        Number of Newton-Schulz iterations.
    eps : float
        Numerical-stability floor for the spectral-norm normalization.

    Returns
    -------
    torch.Tensor
        Orthogonalized stack of shape ``(G, M, N)`` in ``bfloat16`` (cast back
        to the parameter dtype by the caller).
    """
    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
    if updates.ndim != 3:
        raise ValueError("Batched Newton-Schulz expects a 3D (G, M, N) tensor")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")

    a, b, c = ns_coefficients
    ortho = updates.bfloat16()

    # Orient so rows <= cols (the Gram matrix is then the smaller M x M), the
    # same orientation rule torch uses per matrix. All matrices in the batch
    # share a shape, so the decision is uniform.
    transpose = ortho.size(1) > ortho.size(2)
    if transpose:
        ortho = ortho.transpose(1, 2)

    # Ensure each matrix's spectral norm is at most 1 (Frobenius is
    # transpose-invariant, so doing it after the orient above is fine).
    norm = ortho.norm(dim=(1, 2), keepdim=True).clamp(min=eps)
    ortho = ortho / norm

    for _ in range(ns_steps):
        gram = torch.bmm(ortho, ortho.transpose(1, 2))
        # b * gram + c * (gram @ gram)
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        # a * ortho + (gram_update @ ortho)
        ortho = torch.baddbmm(ortho, gram_update, ortho, beta=a, alpha=1.0)

    if transpose:
        ortho = ortho.transpose(1, 2)
    return ortho


class Muon(Optimizer):
    r"""Fused Muon optimizer for 2-D parameters.

    Drop-in replacement for :class:`torch.optim.Muon` that batches the
    Newton-Schulz orthogonalization across parameters of the same shape using
    ``torch.bmm`` / ``torch.baddbmm``, and applies the momentum and weight-decay
    updates with the ``torch._foreach_*`` fused kernels. The algorithm,
    hyperparameter defaults, and ``momentum_buffer`` state key match
    :class:`torch.optim.Muon`, so it is numerically equivalent (batched matmuls
    compute each matrix independently) and checkpoint-compatible.

    Muon only optimizes 2-D parameters (linear / attention weight matrices). Use
    a standard optimizer such as AdamW for biases, norms, and embeddings -- for
    example via :class:`physicsnemo.optim.CombinedOptimizer`.

    Parameters
    ----------
    params : iterable
        Iterable of 2-D parameters or parameter-group dicts.
    lr : float, optional
        Learning rate. Default 1e-3.
    weight_decay : float, optional
        Decoupled weight decay. Default 0.1.
    momentum : float, optional
        Momentum factor. Default 0.95.
    nesterov : bool, optional
        Enable Nesterov momentum. Default True.
    ns_coefficients : tuple[float, float, float], optional
        Newton-Schulz quintic coefficients ``(a, b, c)``.
    eps : float, optional
        Numerical-stability term for the spectral-norm normalization.
    ns_steps : int, optional
        Number of Newton-Schulz iterations. Default 5.
    adjust_lr_fn : str, optional
        One of ``"original"`` or ``"match_rms_adamw"``. Default None
        (treated as ``"original"``).
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (
            DEFAULT_A,
            DEFAULT_B,
            DEFAULT_C,
        ),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in (
            "original",
            "match_rms_adamw",
        ):
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        "Muon only supports 2D parameters whereas we found a "
                        f"parameter with size: {p.size()}"
                    )

    @staticmethod
    def _group_params_by_shape(params: list[Tensor]) -> dict[tuple, list[int]]:
        """Bucket parameter indices by ``(shape, dtype, device)``.

        Parameters that share all three can be stacked and orthogonalized in a
        single batched Newton-Schulz call. Insertion order is preserved so the
        batched result maps back to the original parameter order.

        Parameters
        ----------
        params : list[torch.Tensor]
            Parameters (or per-parameter update tensors) to group.

        Returns
        -------
        dict[tuple, list[int]]
            Mapping from ``(tuple(shape), dtype, device)`` to the list of
            indices into *params* that belong to that group.
        """
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, p in enumerate(params):
            groups[(tuple(p.shape), p.dtype, p.device)].append(i)
        return groups

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            if isinstance(lr, Tensor):
                lr = lr.item()
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_coefficients = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            adjust_lr_fn = group["adjust_lr_fn"]

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            momentum_bufs: list[Tensor] = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                if torch.is_complex(p):
                    raise RuntimeError("Muon does not support complex parameters")
                if p.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if p.grad.ndim != 2:
                    raise ValueError("Param gradient must be a 2D matrix")

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p.grad, memory_format=torch.preserve_format
                    )
                momentum_bufs.append(state["momentum_buffer"])

            if not params_with_grad:
                continue

            # Momentum (fused across all shapes): buf = momentum*buf + (1-momentum)*grad
            torch._foreach_lerp_(momentum_bufs, grads, 1 - momentum)
            if nesterov:
                # update = grad + momentum*(buf - grad)
                updates = torch._foreach_lerp(grads, momentum_bufs, momentum)
            else:
                updates = list(momentum_bufs)

            # Decoupled weight decay (fused across all shapes).
            torch._foreach_mul_(params_with_grad, 1 - lr * weight_decay)

            # Group equally-shaped updates and orthogonalize each group with one
            # batched Newton-Schulz, then apply the (per-group, shape-dependent)
            # learning rate.
            groups = self._group_params_by_shape(params_with_grad)

            for (shape, _dtype, _device), idxs in groups.items():
                stacked = torch.stack([updates[i] for i in idxs], dim=0)
                ortho = _batched_newton_schulz(
                    stacked, ns_coefficients, ns_steps, eps
                )
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, torch.Size(shape))

                group_params = [params_with_grad[i] for i in idxs]
                # Cast back to the parameter dtype (NS runs in bf16).
                ortho_list = [
                    ortho[j].to(group_params[j].dtype) for j in range(len(idxs))
                ]
                torch._foreach_add_(group_params, ortho_list, alpha=-adjusted_lr)

        return loss

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (fused Newton-Schulz)"
