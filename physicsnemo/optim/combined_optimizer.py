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

from typing import Any, Callable, Sequence

import torch
from torch.optim import Optimizer


class CombinedOptimizer(Optimizer):
    r"""Combine multiple PyTorch optimizers into a single Optimizer-like interface.

    This wrapper allows you to use different optimizers for different parts of a model
    while presenting a unified interface compatible with PyTorch's training loops and
    learning rate schedulers. The ``param_groups`` from all contained optimizers are
    concatenated, enabling schedulers to operate transparently across all parameters.

    Parameters
    ----------
    optimizers : Sequence[torch.optim.Optimizer]
        Sequence of PyTorch Optimizer instances to combine. Each optimizer
        should already be configured with its own parameters and hyperparameters.
        Must contain at least one optimizer.
    torch_compile_kwargs : dict[str, Any], optional
        Optional dictionary of keyword arguments to pass to ``torch.compile()``
        when compiling each optimizer's step function. If None, step functions
        are not compiled. Compiling can improve performance but may affect
        serialization. Default is None.

    Raises
    ------
    ValueError
        If ``optimizers`` is empty, or if any parameter appears in multiple
        optimizers (parameter groups must be disjoint).

    Notes
    -----
    * **Parameter Groups**: The ``param_groups`` attribute aggregates parameter
      groups from all underlying optimizers, making this wrapper compatible with
      learning rate schedulers.
    * **Closure Behavior**: When ``step()`` is called with a closure, the closure
      is passed to each underlying optimizer sequentially. This results in the
      closure being evaluated multiple times (at least once per optimizer), which
      triggers multiple forward and backward passes. This behavior matches calling
      ``step(closure)`` on each optimizer individually.
    * **Dynamic Parameter Addition**: The ``add_param_group()`` method is not
      supported. To add parameters dynamically, add them to the individual
      optimizers before creating the CombinedOptimizer, or create a new instance.
    * **State Access**: The ``state`` attribute inherited from the base class may
      not accurately reflect the optimizer state. Access state through the
      individual optimizers in the ``optimizers`` attribute instead.
    * **Serialization**: The optimizer can be pickled and unpickled. When
      ``torch_compile_kwargs`` is provided, the compiled step functions are
      reconstructed during unpickling.

    Examples
    --------
    Combine Adam for model backbone and SGD for the head:

    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.optim import Adam, SGD
    >>> from physicsnemo.optim import CombinedOptimizer
    >>>
    >>> model = nn.Sequential(
    ...     nn.Linear(10, 20),  # backbone
    ...     nn.ReLU(),
    ...     nn.Linear(20, 2),   # head
    ... )
    >>> backbone_params = list(model[0].parameters())
    >>> head_params = list(model[2].parameters())
    >>>
    >>> opt1 = Adam(backbone_params, lr=1e-4)
    >>> opt2 = SGD(head_params, lr=1e-2, momentum=0.9)
    >>> combined_opt = CombinedOptimizer([opt1, opt2])
    >>>
    >>> # Use with a learning rate scheduler
    >>> scheduler = torch.optim.lr_scheduler.StepLR(combined_opt, step_size=10)
    >>>
    >>> # Standard training loop
    >>> for epoch in range(100):
    ...     combined_opt.zero_grad()
    ...     loss = model(torch.randn(32, 10)).sum()
    ...     loss.backward()
    ...     combined_opt.step()
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizers: Sequence[Optimizer],
        torch_compile_kwargs: dict[str, Any] | None = None,
    ):
        if not optimizers:
            raise ValueError("`optimizers` must contain at least one optimizer.")

        ### Validate that parameter groups are disjoint
        # Having overlapping parameters would cause silent bugs where the same
        # parameter is updated multiple times per step.
        seen_params: set[int] = set()
        for opt_idx, opt in enumerate(optimizers):
            for group_idx, group in enumerate(opt.param_groups):
                for param in group["params"]:
                    param_id = id(param)
                    if param_id in seen_params:
                        raise ValueError(
                            f"Parameter appears in multiple optimizers. "
                            f"Found duplicate in optimizer {opt_idx}, group {group_idx}. "
                            f"Each parameter must belong to exactly one optimizer to avoid "
                            f"being updated multiple times per step."
                        )
                    seen_params.add(param_id)

        self.optimizers = optimizers
        self._torch_compile_kwargs = torch_compile_kwargs

        ### Aggregate parameter groups from all optimizers
        # We pass an empty defaults dict because hyperparameters are managed by
        # the individual optimizers, not this wrapper.
        param_groups = [g for opt in optimizers for g in opt.param_groups]

        # Flag to allow add_param_group during initialization
        self._initializing = True
        try:
            super().__init__(param_groups, defaults={})
        finally:
            self._initializing = False

        ### Setup step functions (optionally compiled)
        if torch_compile_kwargs is None:
            self.step_fns: list[Callable] = [opt.step for opt in optimizers]
        else:
            self.step_fns: list[Callable] = [
                torch.compile(opt.step, **torch_compile_kwargs) for opt in optimizers
            ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        r"""Clear the gradients of all optimized parameters.

        This method delegates to the ``zero_grad()`` method of each underlying
        optimizer.

        Parameters
        ----------
        set_to_none : bool, optional
            If True (default), sets gradients to None instead of zero. This
            reduces memory usage and can improve performance. Matches the
            upstream PyTorch ``Optimizer.zero_grad()`` interface.
        """
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        r"""Perform a single optimization step.

        This method calls the ``step()`` method of each underlying optimizer. If a
        closure is provided, it is passed to each optimizer.

        Parameters
        ----------
        closure : Callable[[], float], optional
            Optional callable that reevaluates the model and returns the loss.
            If provided, it will be passed to each optimizer's step function.
            Default is None.

        Returns
        -------
        float or None
            The loss value returned by the last optimizer that returns a non-None
            value, or None if no closure was provided or no optimizer returned a
            value. When multiple optimizers return values, the result from the
            last optimizer in sequence takes precedence.

        Notes
        -----
        The return value semantics match PyTorch's ``Optimizer.step()`` interface,
        which returns ``float | None``. In practice, most closures return a
        ``torch.Tensor`` loss, and PyTorch optimizers that use the closure will
        call ``.item()`` on it internally before returning.
        """
        loss = None
        for step_fn in self.step_fns:
            if closure is None:
                step_fn()
            else:
                res = step_fn(closure)
                if res is not None:
                    loss = res

        return loss

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        r"""Add a param group to the Optimizer's param_groups.

        This method is not supported for CombinedOptimizer as it would require
        logic to determine which underlying optimizer should handle the new group.

        Parameters
        ----------
        param_group : dict[str, Any]
            The parameter group to add.

        Raises
        ------
        NotImplementedError
            Always raises NotImplementedError unless called during initialization.
        """
        if getattr(self, "_initializing", False):
            super().add_param_group(param_group)
            return

        raise NotImplementedError(
            "CombinedOptimizer does not support add_param_group() after initialization, "
            "since it is ambiguous which optimizer should handle the new group.\n"
            "Add parameters to the underlying optimizers before creating the CombinedOptimizer."
        )

    def state_dict(self) -> dict[str, Any]:
        r"""Return the state of all optimizers as a dictionary.

        The returned dictionary contains the state dictionaries of all underlying
        optimizers, allowing the combined optimizer to be checkpointed and restored.

        Returns
        -------
        dict[str, Any]
            A dictionary with a single key ``"optimizers"`` mapping to a list of
            state dictionaries, one for each underlying optimizer in order.

        Examples
        --------
        >>> import torch
        >>> from physicsnemo.optim import CombinedOptimizer
        >>> param1 = torch.nn.Parameter(torch.randn(3))
        >>> param2 = torch.nn.Parameter(torch.randn(3))
        >>> opt1 = torch.optim.SGD([param1], lr=0.01)
        >>> opt2 = torch.optim.Adam([param2], lr=0.001)
        >>> combined_opt = CombinedOptimizer([opt1, opt2])
        >>> state = combined_opt.state_dict()
        >>> list(state.keys())
        ['optimizers']
        >>> len(state["optimizers"])
        2
        """
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""Load the state of all optimizers from a dictionary.

        This method restores the state of each underlying optimizer from the provided
        state dictionary. The state dictionary must have been created by
        ``state_dict()`` from a CombinedOptimizer with the same number of optimizers.

        Parameters
        ----------
        state_dict : dict[str, Any]
            A dictionary containing optimizer states, as returned by
            ``state_dict()``. Must contain an ``"optimizers"`` key mapping to
            a list of state dictionaries.

        Raises
        ------
        ValueError
            If the number of optimizers in ``state_dict`` does not match
            the number of optimizers in this instance.
        KeyError
            If ``state_dict`` does not contain the expected structure.

        Notes
        -----
        After loading state, the ``param_groups`` attribute is refreshed to
        reflect any changes in the underlying optimizers.
        """
        ### Validate state dict structure
        if "optimizers" not in state_dict:
            raise KeyError(
                "Expected state_dict to contain 'optimizers' key, "
                f"but got keys: {list(state_dict.keys())}"
            )

        optimizer_states = state_dict["optimizers"]
        if len(optimizer_states) != len(self.optimizers):
            raise ValueError(
                f"State dict contains {len(optimizer_states)} optimizer(s), "
                f"but this CombinedOptimizer has {len(self.optimizers)} optimizer(s). "
                "Cannot load state from a different optimizer configuration."
            )

        ### Load state into each underlying optimizer
        for opt, sd in zip(self.optimizers, optimizer_states):
            opt.load_state_dict(sd)

        ### Refresh param_groups to reflect any changes
        self.param_groups = [g for opt in self.optimizers for g in opt.param_groups]

    def __repr__(self) -> str:
        r"""Return a string representation of the CombinedOptimizer.

        Returns
        -------
        str
            A string showing the optimizer types being combined.
        """
        optimizer_types = [opt.__class__.__name__ for opt in self.optimizers]
        return f"CombinedOptimizer({', '.join(optimizer_types)})"
