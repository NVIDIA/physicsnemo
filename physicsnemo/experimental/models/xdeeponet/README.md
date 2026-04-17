# xDeepONet — the Extended DeepONet Family

`physicsnemo.experimental.models.xdeeponet` provides a unified, config-driven
implementation of eight DeepONet-based architectures for operator learning
on 2D (`(H, W)`) and 3D (`(X, Y, Z)`) spatial domains.  All variants share
the same branch/trunk/decoder design and are selected via a single
`variant` argument on the wrapper classes.

## Supported Variants

| Variant             | Branches | Branch2 input          | Typical use              |
|---------------------|----------|------------------------|--------------------------|
| `deeponet`          | 1        | —                      | Baseline DeepONet        |
| `u_deeponet`        | 1        | —                      | UNet-enhanced branch     |
| `fourier_deeponet`  | 1        | —                      | Spectral branch          |
| `conv_deeponet`     | 1        | —                      | Convolutional branch     |
| `hybrid_deeponet`   | 1        | —                      | Fourier + UNet + Conv    |
| `mionet`            | 2        | Scalar features        | Multi-input operator     |
| `fourier_mionet`    | 2        | Scalar features        | MIONet + Fourier branch  |
| `tno`               | 2        | Previous solution      | Temporal Neural Operator |

All variants are available in both 2D and 3D spatial configurations.

## Quick Start

```python
import torch
from physicsnemo.experimental.models.xdeeponet import DeepONet3DWrapper

model = DeepONet3DWrapper(
    variant="tno",
    width=128,
    padding=8,
    branch1_config={
        "encoder": "spatial",
        "num_fourier_layers": 1,
        "num_unet_layers": 1,
        "modes1": 10, "modes2": 10, "modes3": 8,
        "activation_fn": "tanh",
    },
    branch2_config={
        "encoder": "spatial",
        "num_fourier_layers": 1,
        "num_unet_layers": 1,
        "modes1": 10, "modes2": 10, "modes3": 8,
        "activation_fn": "tanh",
    },
    trunk_config={
        "input_type": "time",
        "hidden_width": 128,
        "num_layers": 8,
        "activation_fn": "tanh",
        "output_activation": False,
    },
    decoder_type="temporal_projection",
    decoder_width=128,
    decoder_layers=2,
)

# Autoregressive bundling: predict K=3 future timesteps from 1 context step
model.set_output_window(K=3)

x = torch.randn(2, 16, 16, 16, 1, 11)   # (B, X, Y, Z, T_in, C)
prev = torch.randn(2, 16, 16, 16, 1)    # previous solution
out = model(x, x_branch2=prev)          # (B, X, Y, Z, 3)
```

## Public API

### Wrappers (recommended entry points)

`DeepONetWrapper` (2D) and `DeepONet3DWrapper` (3D) add two conveniences
on top of the core classes:

1. **Automatic spatial padding** — right-pads inputs to a multiple (default 8)
   so Fourier, UNet, and Conv sub-branches operate on compatible shapes.
   Outputs are cropped back to the original spatial size.
2. **Automatic trunk coordinate extraction** — assembles trunk query
   coordinates from the full input tensor according to
   `trunk_config["input_type"]` (`"time"` or `"grid"`).

### Core classes

`DeepONet` (2D) and `DeepONet3D` (3D) expose the raw architecture without
padding or input extraction; use these when you have already prepared the
spatial branch input and trunk coordinates explicitly.

### Building blocks

`TrunkNet`, `MLPBranch`, `SpatialBranch`, `SpatialBranch3D` are the sub-networks
used internally; they are exported for users who want to assemble custom
variants.

## Branch configuration schema

Each branch is configured via a Python dict.  Two formats are accepted —
the nested format is canonical; the flat format is converted automatically:

**Nested (canonical):**

```python
{
    "encoder": {
        "type": "linear",        # or "mlp" or "conv"
        "hidden_width": 64,      # mlp only
        "num_layers": 2,         # mlp/conv only
        "activation_fn": "tanh",
    },
    "layers": {
        "num_fourier_layers": 1,
        "num_unet_layers": 1,
        "num_conv_layers": 0,
        "modes1": 10, "modes2": 10, "modes3": 8,  # 3D uses modes3
        "kernel_size": 3,
        "dropout": 0.0,
        "activation_fn": "tanh",
    },
    "internal_resolution": [16, 16, 16],  # optional adaptive pooling
    "in_channels": 11,                     # optional (informational)
}
```

**Flat (auto-converted):**

```python
{
    "encoder": "spatial",       # or "mlp"
    "num_fourier_layers": 1,
    "num_unet_layers": 1,
    "num_conv_layers": 0,
    "modes1": 10, "modes2": 10, "modes3": 8,
    "kernel_size": 3,
    "activation_fn": "tanh",
}
```

## Decoder types

- `"mlp"` — query the trunk at each target timestep, apply an MLP decoder
  per-timestep.  Standard DeepONet decoding.
- `"conv"` — per-timestep trunk query followed by a convolutional decoder.
- `"temporal_projection"` — query the trunk once and project the combined
  latent representation to K output timesteps via a learned linear head.
  Fast for autoregressive bundling.  Requires `model.set_output_window(K)`
  before the first forward pass.

## UNet sub-modules

The UNet layers inside the spatial branches use
`physicsnemo.models.unet.UNet` (3D).  For 2D spatial branches, a small
internal adapter tiles a short time axis so the 3D UNet's pooling stages
function correctly, then averages the result back to 2D.

## Padding behaviour

Both wrappers pad spatial dimensions to a multiple of 8 (configurable via
the `padding` argument, which is rounded up to the next multiple of 8).
Padded cells are filled via replicate padding; outputs are cropped back
to the original input shape.

## References

- Lu, L. et al. (2021). "Learning nonlinear operators via DeepONet."
  *Nature Machine Intelligence*, 3, 218-229.
- Jin, P., Meng, S. & Lu, L. (2022). "MIONet: Learning multiple-input
  operators via tensor product." *SIAM J. Sci. Comp.*, 44(6), A3490-A3514.
- Wen, G. et al. (2022). "U-FNO — An enhanced Fourier neural operator-based
  deep-learning model for multiphase flow." *Advances in Water Resources*,
  163, 104180.
- Zhu, M. et al. (2023). "Fourier-DeepONet: Fourier-enhanced deep operator
  networks for full waveform inversion." arXiv:2305.17289.
- Diab, W. & Al Kobaisi, M. (2024). "U-DeepONet: U-Net enhanced deep
  operator network for geologic carbon sequestration."
  *Scientific Reports*, 14, 21298.
- Jiang, Z. et al. (2024). "Fourier-MIONet: Fourier-enhanced multiple-input
  neural operators for multiphase modeling of geological carbon
  sequestration." *Reliability Eng. & System Safety*, 251, 110392.
- Diab, W. & Al Kobaisi, M. (2025). "Temporal neural operator for modeling
  time-dependent physical phenomena." *Scientific Reports*, 15.
