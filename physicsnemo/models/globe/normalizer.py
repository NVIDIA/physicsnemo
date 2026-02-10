from typing import Literal, Sequence

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    """Shape-aware EWMA normalization in log-space based on mean(abs(x)).

    This module maintains a running estimate of mean(abs(x)) in log-space and
    normalizes inputs by that estimate. The tracked statistic can be a scalar or
    have arbitrary shape, controlled by the `shape` argument:

    - If a dimension in `shape` is 1, that dimension is reduced by computing the
      mean across that axis.
    - If a dimension in `shape` equals the corresponding dimension in the input,
      no reduction is applied for that axis, and the statistic is maintained per
      position along that axis.
    - Exact zeros in the reduced axes are ignored when computing the mean. If an
      entire reduction slice is zero, its mean is treated as 0 and the log-space
      update naturally leaves the running estimate unchanged for that slice.

    In training mode, each forward call:
    1) computes mean(abs(x)) per the reduction rules above (ignoring zeros),
    2) converts it to log-space, and
    3) updates `ln_running_mean` via a numerically stable log-domain EWMA:
       ln(m) = log( momentum * exp(ln_prev) + (1-momentum) * exp(ln_batch) ).

    In eval mode, state is not updated. In both modes, the output is:
        x * exp(-ln_running_mean)

    Optional learnable affine parameters (`weight`, `bias`) can be supplied via
    `learnable_weight_shape` and `learnable_bias_shape`, which are broadcasted to
    the input as usual.

    Args:
        shape: Target shape of the running statistic. Must have the same number
            of dimensions as the input. Dimensions set to 1 indicate a mean will
            be taken across that axis; dimensions equal to the input size keep a
            per-position statistic for that axis.
        momentum: EWMA momentum in [0, 1). Higher values emphasize historical
            estimates more strongly.
        learnable_weight_shape: If provided, creates a learnable multiplicative
            `weight` with the given shape.
        learnable_bias_shape: If provided, creates a learnable additive `bias`
            with the given shape.

    Notes:
        - `ln_running_mean` is stored as a registered buffer so it moves with the
          module across devices and is included in state_dict.
        - On initialization, `ln_running_mean` is zeros (i.e., running mean is 1).
          This makes eval-before-train a no-op normalization.
    """

    def __init__(
        self,
        shape: torch.Size | Sequence[int] = tuple(),
        momentum: float = 0.99,
        learnable_weight_shape: torch.Size | Sequence[int] | None = None,
        learnable_bias_shape: torch.Size | Sequence[int] | None = None,
        initialization_behavior: Literal["no_op", "first_batch"] = "no_op",
        disable: bool = False,
    ):
        ### Validate inputs
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"{momentum= } must satisfy 0 <= momentum < 1")

        super().__init__()

        # Store parameters
        self.shape = shape
        self.momentum = momentum
        self.learnable_weight_shape = learnable_weight_shape
        self.learnable_bias_shape = learnable_bias_shape
        self.initialization_behavior = initialization_behavior
        self.disable = disable

        # Initialize learnable parameters, if applicable
        if not disable:
            if self.learnable_weight_shape is not None:
                self.weight = nn.Parameter(torch.ones(self.learnable_weight_shape))
            if self.learnable_bias_shape is not None:
                self.bias = nn.Parameter(torch.zeros(self.learnable_bias_shape))

        # Uses buffers to track state across devices/checkpoints
        # Start with ln(1.0) = 0.0 so eval before train is a no-op normalization.
        self.register_buffer("ln_running_mean", torch.zeros(shape))
        self.register_buffer("_initialized", torch.tensor(False, dtype=torch.bool))

    def __repr__(self) -> str:
        """Returns a string representation showing the current running mean state.

        Returns:
            str: String in format "Normalizer(ln_running_mean=<tensor>)" displaying
                the log-space running mean estimate.
        """
        return f"Normalizer(ln_running_mean={self.ln_running_mean!r})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize by a log-space EWMA of mean(abs(x)) with shape-aware reduction.

        Behavior depends on `self.training`:
        - Train: update `ln_running_mean` from the current batch and return the
          normalized output.
        - Eval: return normalized output without updating state.

        Input shape semantics:
        - Let `x.shape == s` and `self.shape == t`. It must be that `len(s) == len(t)`.
        - For each dimension `d`, if `t[d] == 1`, the mean over axis `d` is
          computed (ignoring exact zeros). If `t[d] == s[d]`, no reduction is
          performed along `d` and the statistic is per-position.

        Args:
            x: Input tensor with shape matching the configured rank. Reductions
                are performed along axes where `self.shape` has size 1.

        Returns:
            Tensor normalized by the running mean, then optionally affine
            transformed by `weight`/`bias` if present. Shape matches `x.shape`.
        """
        if self.disable:
            return x

        ### Validate inputs
        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            if len(self.shape) != len(x.shape):
                raise ValueError(f"Shape mismatch: {self.shape=!r} != {x.shape=!r}")

        if self.training:
            # Compute abs(x) for statistic updates out of autograd
            values = x.detach().abs()

            values[values == 0.0] = torch.nan  # Allows nanmean to ignore zero values

            # Reduce axes where target shape has size 1, computing a mean that
            # ignores exact zeros. If all values along an axis slice are zero,
            # the mean is treated as zero for that slice.
            for dim, (desired_size, actual_size) in enumerate(zip(self.shape, x.shape)):
                if desired_size != actual_size:
                    if desired_size == 1:
                        values = torch.nanmean(values, dim=dim, keepdim=True)
                    else:
                        raise ValueError(
                            f"Shape mismatch for {dim=!r}: {desired_size=!r} != {actual_size=!r}"
                        )

            values[torch.isnan(values)] = 0.0  # Makes 0 the default value (no-op)

            ln_batch_mean = values.log()

            ### Now, compute the new value

            # Uninitialized case:
            if self.initialization_behavior == "first_batch":
                new_ln_running_mean_if_uninitialized = ln_batch_mean
            elif self.initialization_behavior == "no_op":
                new_ln_running_mean_if_uninitialized = self.ln_running_mean
            else:
                raise ValueError(
                    f"Invalid initialization behavior: {self.initialization_behavior=!r}"
                )

            # Initialized case:
            new_ln_running_mean_if_initialized = (
                self.momentum * self.ln_running_mean
                + (1.0 - self.momentum) * ln_batch_mean
            )

            # Now, update the running mean
            self.ln_running_mean.copy_(
                torch.where(
                    self._initialized,
                    new_ln_running_mean_if_initialized,
                    new_ln_running_mean_if_uninitialized,
                )
            )

        # Divide by running mean via multiplying by exp(-ln_running_mean)
        x = x * torch.exp(-self.ln_running_mean)
        if self.learnable_weight_shape is not None:
            x = x * self.weight
        if self.learnable_bias_shape is not None:
            x = x + self.bias
        return x


if __name__ == "__main__":
    normalizer = Normalizer(shape=(1,), momentum=0.99)

    print("Training mode:")
    normalizer.train()
    torch.manual_seed(0)
    for i in range(10):
        x = (0 if i == 0 else 10) + torch.randn(3)
        y = normalizer(x)
        print(f"{i=!r}, {x=!r}, {y=!r}, {normalizer.ln_running_mean=!r}")

    print("Evaluation mode:")
    normalizer.eval()
    torch.manual_seed(0)
    for i in range(10):
        x = (0 if i == 0 else 10) + torch.randn(3)
        y = normalizer(x)
        print(f"{i=!r}, {x=!r}, {y=!r}, {normalizer.ln_running_mean=!r}")
