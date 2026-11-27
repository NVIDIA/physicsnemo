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

This module provides :class:`Muon`, a subclass of :class:`torch.optim.Muon` that
replaces the per-step orthogonalization with a batched implementation and adds one
constructor keyword, ``cuda_graph``, which replays the step as a CUDA graph. The
other constructor arguments, hyperparameter semantics, validation, and
``momentum_buffer`` optimizer-state key are all inherited from
:class:`torch.optim.Muon`, and ``cuda_graph`` stays out of the ``state_dict``, so
checkpoints remain interchangeable.
"""

from __future__ import annotations

import contextlib
import math
import sys
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Callable

import torch
from torch import Tensor
from torch.optim import Muon as _TorchMuon

__all__ = ["Muon"]

# The param-group entries left out of Muon(cuda_graph=True)'s graph key: the
# graph does not read them (the parameters enter the key through their data
# pointers), so changing them needs no new capture.
_NOT_IN_GRAPH_KEY = frozenset({"params", "lr", "weight_decay", "adjust_lr_fn"})


def _adjust_lr(lr: float, adjust_lr_fn: str | None, shape: torch.Size) -> float:
    """Learning rate of a parameter under :class:`torch.optim.Muon`'s rule.

    Computes the formulas that :class:`torch.optim.Muon` documents, rather
    than calling its private helper: ``sqrt(max(1, A / B))`` for
    ``"original"`` or ``None``, ``0.2 * sqrt(max(A, B))`` for
    ``"match_rms_adamw"``, and ``sqrt(A / B)`` for ``"spectral_unclamped"``
    (torch 2.14 and later). It raises for any other name, so an option that a
    later torch adds cannot change the step silently.

    Parameters
    ----------
    lr : float
        Learning rate of the parameter group.
    adjust_lr_fn : str or None
        One of ``"original"``, ``"match_rms_adamw"``, ``"spectral_unclamped"``
        or None.
    shape : torch.Size
        Shape ``(A, B, ...)`` of the parameter.

    Returns
    -------
    float
        The adjusted learning rate.
    """
    a, b = shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1, a / b))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(a, b))
    elif adjust_lr_fn == "spectral_unclamped":
        ratio = math.sqrt(a / b)
    else:
        raise ValueError(f"Muon does not implement adjust_lr_fn={adjust_lr_fn!r}")
    return lr * ratio


def _check_one_cuda_device(params: Iterable[Tensor]) -> None:
    """Raise unless every parameter is on one CUDA device, as a graph needs."""
    devices = {p.device for p in params}
    if len(devices) != 1 or devices.pop().type != "cuda":
        raise ValueError(
            "Muon(cuda_graph=True) needs every parameter on one CUDA device"
        )


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
        # This is a decision that exactly mirrors upstream pytorch.
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


class Muon(_TorchMuon):
    r"""Fused Muon optimizer for 2-D parameters.

    Subclass of :class:`torch.optim.Muon` that batches the Newton-Schulz
    orthogonalization across parameters of the same shape using ``torch.bmm`` /
    ``torch.baddbmm``, and applies the momentum and weight-decay updates with the
    ``torch._foreach_*`` fused kernels. It adds one constructor keyword,
    ``cuda_graph``, which replays the step as a CUDA graph. Everything else
    about construction, validation, hyperparameter defaults, and the
    ``momentum_buffer`` state key is inherited from :class:`torch.optim.Muon`,
    so it is numerically equivalent (batched matmuls compute each matrix
    independently) and checkpoint-compatible.

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
        One of ``"original"``, ``"match_rms_adamw"`` or, with torch 2.14 and
        later, ``"spectral_unclamped"``. Default None (treated as
        ``"original"``).
    cuda_graph : bool, optional
        Replay the momentum update and the Newton-Schulz orthogonalization as
        one CUDA graph. It saves host time at the cost of extra GPU memory, and
        the result equals the eager step's bit for bit (see Notes). Needs every
        parameter on one CUDA device. Default False.

    Forward
    -------
    Call :meth:`step` after ``loss.backward()`` to apply one optimization step.

    Outputs
    -------
    The optional closure loss returned by :meth:`step`, or ``None``.

    Notes
    -----
    See :class:`torch.optim.Muon` for the full algorithm description.

    With ``cuda_graph=True``, the weight decay and the update run after each
    replay, so a scheduler's new ``lr``, ``weight_decay``, or ``adjust_lr_fn``
    replays the same graph. Other changes, such as new parameters with
    gradients, a new ``momentum``, or a loaded state dict, need a new capture,
    which synchronizes the device and empties PyTorch's memory caches. So Muon
    captures only once such a change has held for a step, first on the third
    step, and a schedule that changes ``momentum`` every step, as ``OneCycleLR``
    does by default, runs eagerly. The graph keeps copies of the gradients and
    of the orthogonalized updates, each the size of the parameters. If a
    capture fails, Muon warns, turns ``cuda_graph`` off, and steps on without
    the graph. Under ``torch.compile``, the graph step runs outside the
    compiled region.

    .. important::
        The fused path stacks equally-shaped matrices and orthogonalizes them
        with batched matmuls, which is only correct for **replicated 2-D
        parameters**. Single-GPU and **DDP** are fully supported and numerically
        equal to :class:`torch.optim.Muon` (DDP gradients are dense, replicated
        tensors, and :meth:`step` runs after ``backward()`` returns, by which
        point DDP's bucketed all-reduce has already been synchronized).

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.optim import Muon
    >>> weights = [torch.nn.Parameter(torch.randn(8, 8)) for _ in range(3)]
    >>> opt = Muon(weights, lr=0.02)
    >>> for w in weights:
    ...     w.grad = torch.randn_like(w)
    >>> _ = opt.step()
    """

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

    def __init__(self, params, *args, cuda_graph: bool = False, **kwargs) -> None:
        super().__init__(params, *args, **kwargs)
        if cuda_graph:
            _check_one_cuda_device(p for g in self.param_groups for p in g["params"])
        self.cuda_graph = cuda_graph
        self._key = self._graph = self._graph_grads = self._graph_updates = None
        self._graph_stream = None

    def __getstate__(self) -> dict:
        return {**super().__getstate__(), "cuda_graph": self.cuda_graph}

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        # load_state_dict also lands here, with a state lacking cuda_graph.
        self.__dict__.setdefault("cuda_graph", False)
        self.__dict__.setdefault("_graph_stream", None)
        # A graph cannot be copied, so a copy captures its own.
        self._key = self._graph = self._graph_grads = self._graph_updates = None

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step.

        With ``cuda_graph`` set and no closure, the step replays a CUDA graph.

        Parameters
        ----------
        closure : Callable[[], float], optional
            Optional callable that reevaluates the model and returns the loss.
            Default None.

        Returns
        -------
        float or None
            The loss returned by *closure*, or ``None`` if no closure was given.
        """
        if closure is None and self.cuda_graph:
            return self._graph_step()
        return self._step(closure)

    @torch.compiler.disable
    def _graph_step(self) -> None:
        """Replay the step as a CUDA graph, capturing it first if needed.

        Steps eagerly inside a capture the caller started, which then records
        the eager step, and while a parameter with a gradient has no momentum
        buffer, so that no buffer is made in a capture. A capture costs more
        than an eager step, so it also steps eagerly when the key (what the
        graph reads) differs from the previous step's, and captures only a key
        that held for two steps.
        """
        if torch.cuda.is_current_stream_capturing():
            return self._step()
        params = [
            p for g in self.param_groups for p in g["params"] if p.grad is not None
        ]
        if not params or any(
            "momentum_buffer" not in self.state.get(p, {}) for p in params
        ):
            return self._step()
        bufs = [self.state[p]["momentum_buffer"] for p in params]
        key = (
            [t.data_ptr() for t in params + bufs],
            # The group dicts themselves too: the update after the replay
            # reads them.
            [
                (id(g), {k: v for k, v in g.items() if k not in _NOT_IN_GRAPH_KEY})
                for g in self.param_groups
            ],
        )
        if key != self._key:
            self._key = key
            self._graph = self._graph_grads = self._graph_updates = None
            return self._step()
        if self._graph is None and not self._capture(params):
            return self._step()
        torch._foreach_copy_(self._graph_grads, [p.grad for p in params])
        self._graph.replay()
        for update in self._graph_updates:
            self._apply(*update)
        return None

    def _capture(self, params: list[Tensor]) -> bool:
        """Capture the momentum update and orthogonalization as a CUDA graph.

        Returns whether the capture succeeded. One that fails anywhere ends
        cleanly, warns, and turns ``cuda_graph`` off.
        """
        _check_one_cuda_device(params)
        # The graph reads the gradients from copies of its own, refilled
        # before every replay.
        grads = [p.grad for p in params]
        graph_grads = [g.clone() for g in grads]
        # One stream for every capture, from torch's high-priority pool, which
        # default-priority users such as loader threads never get: their work
        # on it during a capture would be recorded instead of run. Replayed
        # kernels keep the priority, so they start first when others compete.
        device = params[0].device
        if self._graph_stream is None or self._graph_stream.device != device:
            self._graph_stream = torch.cuda.Stream(device, priority=-1)
        stream = self._graph_stream
        graph = torch.cuda.CUDAGraph()
        outer = sys.exception()  # the caller's error, if it steps in an except
        try:
            for p, g in zip(params, graph_grads):
                p.grad = g
            # "thread_local": other threads, such as a data loader's, may sync
            # meanwhile. The outer stream context restores the current stream
            # when a failed capture skips torch.cuda.graph's exit.
            with (
                torch.cuda.stream(stream),
                torch.cuda.graph(
                    graph, stream=stream, capture_error_mode="thread_local"
                ),
            ):
                updates = [
                    (group, ps, list(shape_updates))
                    for group, ps, shape_updates in self._updates()
                ]
        except Exception as e:
            # A failure in capture_begin, or in capture_end before it ends
            # the capture, leaves the stream capturing: end it here.
            with torch.cuda.stream(stream):
                if torch.cuda.is_current_stream_capturing():
                    with (
                        contextlib.suppress(Exception),
                        warnings.catch_warnings(action="ignore"),
                    ):
                        graph.capture_end()
            # Name the first error: one in the step makes capture_end raise
            # again, about "a previous error during capture".
            cause = e
            while (c := cause.__cause__ or cause.__context__) not in (None, outer):
                cause = c
            reason = f"{type(cause).__name__}: {cause}"
            # Its traceback holds the capture's tensors: free them for the
            # eager step.
            del cause
            self.cuda_graph = False
            warnings.warn(
                "Muon(cuda_graph=True) could not capture its step as a CUDA "
                f"graph on torch {torch.__version__}, so it steps eagerly from "
                f"now on (the same update, more launches). The first error: "
                f"{reason}",
                RuntimeWarning,
            )
            return False
        finally:
            for p, g in zip(params, grads):
                p.grad = g
        self._graph, self._graph_grads = graph, graph_grads
        self._graph_updates = updates
        return True

    def _updates(self) -> Iterator[tuple[dict, list[Tensor], Iterator[tuple]]]:
        """Update each parameter group's momentum and yield its update.

        For each group with gradients, yields ``(group, params,
        shape_updates)``, where ``shape_updates`` yields ``(shape,
        group_params, ortho_list)`` per group of equally-shaped parameters,
        orthogonalized with one batched Newton-Schulz. Both are lazy, so the
        eager step holds one shape group's update at a time.
        """
        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            momentum_bufs: list[Tensor] = []

            # The parameters with a gradient, their gradients and momentum
            # buffers (zero on first use), as torch.optim.Muon collects them.
            for p in group["params"]:
                if p.grad is None:
                    continue
                if torch.is_complex(p):
                    raise RuntimeError("Muon does not support complex parameters")
                if p.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                params.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p.grad, memory_format=torch.preserve_format
                    )
                momentum_bufs.append(state["momentum_buffer"])

            if not params:
                continue

            for g in grads:
                if g.ndim != 2:
                    raise ValueError("Param gradient must be a 2D matrix")

            # Momentum (fused across all shapes): buf = momentum*buf + (1-momentum)*grad
            momentum = group["momentum"]
            torch._foreach_lerp_(momentum_bufs, grads, 1 - momentum)
            if group["nesterov"]:
                # update = grad + momentum*(buf - grad)
                updates = torch._foreach_lerp(grads, momentum_bufs, momentum)
            else:
                updates = list(momentum_bufs)
            yield group, params, self._orthogonalize(group, params, updates)

    def _orthogonalize(
        self, group: dict, params: list[Tensor], updates: list[Tensor]
    ) -> Iterator[tuple[tuple, list[Tensor], tuple[Tensor, ...]]]:
        """Yield each shape group's orthogonalized update (see :meth:`_updates`)."""
        ns_args = group["ns_coefficients"], group["ns_steps"], group["eps"]
        for (shape, _, _), idxs in self._group_params_by_shape(params).items():
            stacked = torch.stack([updates[i] for i in idxs], dim=0)
            ortho = _batched_newton_schulz(stacked, *ns_args)
            group_params = [params[i] for i in idxs]
            # Cast back to the parameter dtype (NS runs in bf16), once per
            # group: the group key includes the dtype. For float32 parameters
            # the cast also makes the updates contiguous, so a tall group
            # (transposed by NS) keeps _foreach_add_ fused.
            ortho_list = ortho.to(
                group_params[0].dtype, memory_format=torch.contiguous_format
            ).unbind(0)
            yield shape, group_params, ortho_list

    @staticmethod
    def _apply(
        group: dict, params: list[Tensor], shape_updates: Iterable[tuple]
    ) -> None:
        """Apply the decoupled weight decay, then each shape group's update,
        at the group's current learning rate and weight decay."""
        lr = group["lr"]
        if isinstance(lr, Tensor):
            lr = lr.item()
        torch._foreach_mul_(params, 1 - lr * group["weight_decay"])
        for shape, group_params, ortho_list in shape_updates:
            adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], torch.Size(shape))
            torch._foreach_add_(group_params, ortho_list, alpha=-adjusted_lr)

    def _step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step eagerly.

        Parameters
        ----------
        closure : Callable[[], float], optional
            Optional callable that reevaluates the model and returns the loss.
            Default None.

        Returns
        -------
        float or None
            The loss returned by *closure*, or ``None`` if no closure was given.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for update in self._updates():
            self._apply(*update)
        return loss

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (fused Newton-Schulz)"
