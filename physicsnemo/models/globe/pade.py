from typing import Callable, Sequence

import torch
import torch.nn as nn

from physicsnemo.models.globe.mlp import MLP


class Pade(nn.Module):
    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        spectral_norm: bool = False,
        numerator_order: int = 1,
        denominator_order: int = 2,
        share_denominator_across_channels: bool = True,
        use_separate_mlps: bool = True,
    ):
        """
        Pade approximant module with configurable architecture.

        This class constructs a fully-connected feedforward neural network with
        optional batch normalization, dropout, and a user-specified activation
        function. The architecture is defined by a list of layer sizes, where
        each entry specifies the number of neurons in that layer.

        Args:
            layer_sizes (list[int]): List of integers specifying the number of
                units in each layer, including input and output layers. For
                example, [8, 32, 16, 4] creates an MLP with input dimension 8,
                two hidden layers of sizes 32 and 16, and output dimension 4.
            activation_function (nn.Module | Callable[[torch.Tensor],
                torch.Tensor], optional): Activation function to use after each
                hidden layer. Can be a torch.nn module or a callable. Note: to
                work well, the activation function MUST have the property that
                f(x) -> 0 as x -> -inf and f(x) -> x as x -> +inf. This ensures
                proper asymptotic behavior for the Pade approximant. Common
                choices include nn.SiLU(), nn.Mish(), nn.GELU(), nn.Softplus(),
                etc. Defaults to nn.SiLU().
            dropout (float, optional): Dropout probability applied after each
                activation (except the last layer). Set to 0.0 to disable
                dropout. Defaults to 0.0.
            use_batchnorm (bool, optional): If True, applies BatchNorm1d after
                each linear layer (except the last). Defaults to False.
            spectral_norm (bool, optional): If True, the MLPs are hard-constrained
                to have a spectral norm of 1. Defaults to False.
            numerator_order (int, optional): Power to raise the numerator to.
                Defaults to 1.
            denominator_order (int, optional): Power to raise the
                denominator to. Defaults to 2.
            share_denominator_across_channels (bool, optional): If True, uses a
                single scalar denominator for all output channels. If False,
                uses a separate denominator for each output channel. Defaults to
                True.
            use_separate_mlps (bool, optional): If True, uses separate MLPs for
                numerator and denominator. If False, uses a single MLP with
                combined outputs that are split for numerator and denominator.
                Defaults to True.

        Example:
            >>> pade = Pade([10, 64, 32, 3], activation_function=nn.ReLU(), dropout=0.1, use_batchnorm=True)
            >>> x = torch.randn(5, 10)
            >>> y = pade(x)
            >>> print(y.shape)
            torch.Size([5, 3])

        Notes:
            - No activation, dropout, or batch normalization is applied after
              the final output layer.
            - If a callable is provided for `activation_function`, it will be
              wrapped as a module for use in nn.Sequential.
            - For best performance, prefer vectorized input (batch dimension
              first).

        """
        if activation_function is None:
            activation_function = nn.SiLU()

        super().__init__()

        ### Save inputs
        self.layer_sizes = layer_sizes
        self.activation_function = activation_function
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.spectral_norm = spectral_norm
        self.numerator_order = numerator_order
        self.denominator_order = denominator_order
        self.share_denominator_across_channels = share_denominator_across_channels
        self.use_separate_mlps = use_separate_mlps

        ### Create the MLPs
        if use_separate_mlps:
            # Use separate MLPs for numerator and denominator
            numerator_layer_sizes = list(layer_sizes)[:-1] + [
                self.numerator_output_size
            ]
            denominator_layer_sizes = list(layer_sizes)[:-1] + [
                self.denominator_output_size
            ]

            self.numerator_mlp = MLP(
                layer_sizes=numerator_layer_sizes,
                activation_function=activation_function,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                spectral_norm=spectral_norm,
            )
            self.denominator_mlp = MLP(
                layer_sizes=denominator_layer_sizes,
                activation_function=activation_function,
                bias=False,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                spectral_norm=spectral_norm,
            )
        else:
            # Use a single MLP that outputs both numerator and denominator values
            combined_output_size = (
                self.numerator_output_size + self.denominator_output_size
            )
            combined_layer_sizes = list(layer_sizes)[:-1] + [combined_output_size]

            self.combined_mlp = MLP(
                layer_sizes=combined_layer_sizes,
                activation_function=activation_function,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                spectral_norm=spectral_norm,
            )

    @property
    def numerator_output_size(self) -> int:
        """Output dimension of the numerator MLP.

        For Padé approximants, the numerator has the same dimension as the overall
        output (layer_sizes[-1]), allowing each output channel to have its own
        numerator polynomial.

        Returns:
            int: Number of output channels, equal to layer_sizes[-1].
        """
        return self.layer_sizes[-1]

    @property
    def denominator_output_size(self) -> int:
        """Output dimension of the denominator MLP.

        The denominator can either be shared across all output channels (scalar) or
        per-channel, controlled by share_denominator_across_channels. Sharing the
        denominator reduces parameters and couples the decay behavior across channels,
        which is often appropriate for related physical quantities.

        Returns:
            int: 1 if share_denominator_across_channels is True, otherwise layer_sizes[-1].
        """
        return 1 if self.share_denominator_across_channels else self.layer_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the Padé-approximant neural network.

        This method computes a rational function of the form:
            f(x) = sgn(φ_n(x)) |φ_n(x)|^N / (1 + |φ_d(x)|^D)

        where φ_n and φ_d are learnable MLPs (the numerator and denominator subnetworks),
        and N, D are the numerator and denominator orders. This rational structure provides
        strong inductive bias for approximating Green's functions and other physics kernels.

        Key properties of this formulation:

        1. **Sign preservation**: The numerator uses sign-preserving exponentiation,
           allowing the output to be negative when φ_n(x) < 0. This is critical for
           representing dipole-like and oscillatory field patterns.

        2. **Bounded denominator**: The denominator is (1 + |φ_d(x)|^D), always ≥ 1.
           This prevents singularities and limits the Lipschitz constant, improving
           training stability.

        3. **Asymptotic behavior**: When N = D (default: 2/2), the function asymptotes
           to sgn(φ_n) · |φ_n|^N / |φ_d|^D as |φ_d| → ∞, approaching a constant in any
           far-field direction. This is ideal for representing bounded far-field decay.

        4. **Physical motivation**: Green's functions for many elliptic PDEs exhibit
           rational-like behavior: logarithmic or algebraic singularities near sources,
           algebraic decay at infinity. The Padé structure naturally captures this
           without requiring very deep/wide standard MLPs.

        The activation function (typically SiLU) should satisfy f(x) → 0 as x → -∞ and
        f(x) → x as x → +∞ for proper asymptotic behavior of the rational function.

        Args:
            x: Input tensor of shape (batch_size, input_dim), where input_dim must
                match layer_sizes[0] from initialization.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim), where
                output_dim is layer_sizes[-1]. Values are the evaluated Padé approximant.

        Note:
            See paper Section 3.2.3 for mathematical details and motivation for using
            Padé-approximant networks in physics kernel learning.

        Example:
            >>> pade = Pade([10, 64, 64, 3], numerator_order=2, denominator_order=2)
            >>> x = torch.randn(32, 10)
            >>> y = pade(x)  # shape (32, 3)
            >>> # Far-field behavior: when inputs are large, output approaches constant
            >>> x_large = 100 * torch.randn(32, 10)
            >>> y_large = pade(x_large)  # Bounded, not explosive
        """
        if self.use_separate_mlps:
            raw_numerator: torch.Tensor | float = (
                torch.tensor(1.0, device=x.device)
                if self.numerator_order == 0
                else self.numerator_mlp(x)
            )
            raw_denominator: torch.Tensor | float = (
                torch.tensor(1.0, device=x.device)
                if self.denominator_order == 0
                else self.denominator_mlp(x)
            )
        else:
            raw_numerator, raw_denominator = self.combined_mlp(x).split(
                self.numerator_output_size, dim=-1
            )

        def apply_power(x: torch.Tensor, order: int, even: bool) -> torch.Tensor:
            if even:
                return x.abs().pow(order)
            else:
                return x.sign() * x.abs().pow(order)

        numerator = apply_power(raw_numerator, self.numerator_order, even=False)
        denominator = apply_power(raw_denominator, self.denominator_order, even=True)

        # Compute Padé approximant: numerator / (1 + denominator)
        return numerator / (1 + denominator)


if __name__ == "__main__":
    torch.manual_seed(0)

    # Test with separate MLPs
    pade_separate = Pade(
        layer_sizes=[1, 64, 3],
        share_denominator_across_channels=True,
        numerator_order=1,
        denominator_order=2,
        use_separate_mlps=True,
    )

    # Test with combined MLP
    pade_combined = Pade(
        layer_sizes=[1, 64, 3],
        share_denominator_across_channels=True,
        numerator_order=1,
        denominator_order=2,
        use_separate_mlps=False,
    )

    x = 100 * torch.linspace(-1.0, 1.0, 5000).unsqueeze(-1)  # shape (5000, 1)

    y_separate = pade_separate(x)
    y_combined = pade_combined(x)

    print(f"Separate MLPs output shape: {y_separate.shape}")
    print(f"Combined MLP output shape: {y_combined.shape}")

    # Verify they have the same output shape
    assert y_separate.shape == y_combined.shape, "Output shapes should match"  # noqa: S101

    print("Both implementations produce outputs with the correct shape!")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x.detach().numpy(), y_separate.detach().numpy())
    plt.title("Separate MLPs")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(1, 2, 2)
    plt.plot(x.detach().numpy(), y_combined.detach().numpy())
    plt.title("Combined MLP")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.show()
