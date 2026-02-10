from typing import Any, Callable, Sequence
import torch
from torch.optim import Optimizer


class CombinedOptimizer(Optimizer):
    """Combine multiple PyTorch optimizers into a single Optimizer-like interface.

    The wrapper concatenates the *param_groups* from all contained optimizers so
    that learning-rate schedulers (e.g., ReduceLROnPlateau, CosineAnnealingLR)
    operate transparently across every parameter. Only a minimal subset of the
    *torch.optim.Optimizer* API is implemented—extend as needed.
    """

    def __init__(
        self,
        optimizers: Sequence[Optimizer],
        torch_compile_kwargs: dict[str, Any] | None = None,
    ):
        if not optimizers:
            raise ValueError("`optimizers` must contain at least one optimizer.")

        self.optimizers = optimizers

        # Collect parameter groups from all optimizers. We pass an empty
        # *defaults* dict because hyper-parameters are managed by the inner
        # optimizers, not this wrapper.
        param_groups = [g for opt in optimizers for g in opt.param_groups]
        super().__init__(param_groups, defaults={})

        if torch_compile_kwargs is None:
            self.step_fns: list[Callable] = [opt.step for opt in optimizers]
        else:
            self.step_fns: list[Callable] = [
                torch.compile(opt.step, **torch_compile_kwargs) for opt in optimizers
            ]

    def zero_grad(self, *args, **kwargs) -> None:
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    def step(self, closure=None) -> None:
        for step_fn in self.step_fns:
            if closure is None:
                step_fn()
            else:
                step_fn(closure)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)

        self.param_groups = [g for opt in self.optimizers for g in opt.param_groups]
