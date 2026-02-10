from typing import Callable, Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        bias: bool = True,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        spectral_norm: bool = False,
    ):
        """
        Multi-Layer Perceptron (MLP) module with configurable architecture.

        This class constructs a fully-connected feedforward neural network with
        optional batch normalization, dropout, and a user-specified activation
        function. The architecture is defined by a Sequence of layer sizes,
        where each entry specifies the number of neurons in that layer.

        Args:
            layer_sizes (Sequence[int]): Sequence of integers specifying the
            number of units in each layer, including input and output layers.
            For example, [8, 32, 16, 4] creates an MLP with input dimension 8,
            two hidden layers of sizes 32 and 16, and output dimension 4.

            activation_function (nn.Module | Callable[[torch.Tensor],
            torch.Tensor], optional): Activation function to use after each
            hidden layer. Can be a torch.nn module or a callable. Defaults to
            nn.SiLU.

            bias (bool, optional): If True, adds a bias term to the linear layers.
            Defaults to True.

            dropout (float, optional): Dropout probability applied after each
            activation (except the last layer). Set to 0.0 to disable dropout.
            Defaults to 0.0.

            use_batchnorm (bool, optional): If True, applies BatchNorm1d after
            each linear layer (except the last). Defaults to False.

            spectral_norm (bool, optional): If True, the MLP is hard-constrained
            to have a spectral norm of 1. Defaults to False.

        Example:
            >>> mlp = MLP([10, 64, 32, 3], activation_function=nn.ReLU(), dropout=0.1, use_batchnorm=True)
            >>> x = torch.randn(5, 10)
            >>> y = mlp(x)
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
        self.bias = bias
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.spectral_norm = spectral_norm

        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            linear_layer = nn.Linear(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                bias=bias,
            )
            if spectral_norm:
                linear_layer = nn.utils.parametrizations.spectral_norm(
                    module=linear_layer,
                    name="weight",
                )

            layers.append(linear_layer)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation/dropout after last layer
                layers.append(activation_function)
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim), where
                input_dim must match the first entry in `layer_sizes` provided at initialization.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim), where
                output_dim is the last entry in `layer_sizes`.

        Notes:
            - The input must be a 2D tensor (batch dimension first).
            - No activation, dropout, or batch normalization is applied after the final output layer.
        """
        return self.layers(x)


if __name__ == "__main__":
    mlp = MLP(
        layer_sizes=[10, 64, 32, 3],
        activation_function=nn.ReLU(),
        dropout=0.1,
        use_batchnorm=True,
    )
    x = torch.randn(5, 10)
    y = mlp(x)
    print(y.shape)
