# Scaffolds

Starting skeletons to adapt to the contributor's architecture and shapes. These
encode the standards (`MOD-***`) so the contributor can't miss them. **Verify
the exact import paths against the live repo** before emitting (e.g.
`grep -rn "class ModelMetaData" physicsnemo/core/`), and read a sibling under
`physicsnemo/models/` for current conventions — these skeletons are a starting
point, not a frozen API.

Every new file starts with the SPDX Apache-2.0 header (copy it from any
existing source file).

---

## A. New model — `physicsnemo/experimental/models/<name>/<name>.py`

```python
# <SPDX Apache-2.0 header>
"""<One-line module purpose>."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo import Module
from physicsnemo.core import ModelMetaData
# reuse, don't reinvent (see references/reuse_map.md):
# from physicsnemo.nn import Mlp
# from physicsnemo.nn.module.layer_norm import LayerNorm


@dataclass
class MyModelMetaData(ModelMetaData):
    name: str = "MyModel"
    # Capability flags — default conservative; flip on only once verified.
    amp: bool = True
    cuda_graphs: bool = False
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    auto_grad: bool = False


class MyModel(Module):
    r"""<One-line summary>.

    <Short description of what the model does.>

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_dim : int
        Width of the hidden representation.
    out_channels : int
        Number of output channels.
    depth : int, optional
        Number of blocks. Default ``4``.

    Forward
    -------
    x : torch.Tensor
        Input of shape :math:`(B, N, C_{in})`.

    Outputs
    -------
    torch.Tensor
        Output of shape :math:`(B, N, C_{out})`.
    """

    def __init__(
        self,
        *,                      # keyword-only: explicit, serialization-friendly
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        depth: int = 4,
    ):
        super().__init__(meta=MyModelMetaData())
        # JSON-serializable args only; build submodules HERE, don't accept raw
        # nn.Modules (see references/serialization.md).
        self.in_channels = int(in_channels)
        self.proj_in = nn.Linear(in_channels, hidden_dim)
        self.blocks = nn.ModuleList(
            _MyBlock(hidden_dim) for _ in range(int(depth))  # <- the contributor's math
        )
        self.proj_out = nn.Linear(hidden_dim, out_channels)

    def forward(
        self, x: Float[torch.Tensor, "b n c_in"]
    ) -> Float[torch.Tensor, "b n c_out"]:
        # MOD-005: validate at the API boundary, skipped under torch.compile.
        if not torch.compiler.is_compiling():
            if x.ndim != 3 or x.shape[-1] != self.in_channels:
                raise ValueError(
                    f"Expected x of shape (B, N, {self.in_channels}), got "
                    f"tensor of shape {tuple(x.shape)}"
                )
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h)
        return self.proj_out(h)
```

`__init__.py` (re-export only the public class):
```python
# <SPDX header>
from .my_model import MyModel
```

---

## B. New reusable layer — `physicsnemo/nn/module/<name>.py`

Same shape as the model but subclass `Module` with no `ModelMetaData` (confirm
against a sibling layer — some layers do pass a meta), and **wire the exports**:

```python
# physicsnemo/nn/module/__init__.py   (alphabetical insertion)
from .my_layer import MyLayer

# physicsnemo/nn/__init__.py
from .module.my_layer import MyLayer
```

So users get `from physicsnemo.nn import MyLayer` (`MOD-000a`). Imports inside
the layer must be upward-only — no importing from `physicsnemo/models/`
(`EXT-***`).

---

## C. Wrap an existing PyTorch model — `Module.from_torch`

```python
# <SPDX header>
from physicsnemo import Module
from physicsnemo.core import ModelMetaData
from their_package import TheirNet          # untouched external architecture

# TheirNet.__init__ args MUST be JSON-serializable. If TheirNet injects nested
# nn.Modules, convert each with Module.from_torch first.
PhysicsNeMoNet = Module.from_torch(TheirNet, meta=ModelMetaData())
```

See `references/serialization.md` for the nested-submodule case and the
round-trip verification.

---

## D. Tests — `test/.../test_<name>.py`

Mirror the source path. Class-per-public-class, `device` fixture, the
`test.common` helpers — don't hand-roll.

```python
# <SPDX header>
import pytest
import torch

from physicsnemo.experimental.models.<name> import MyModel
from test.common import validate_checkpoint, validate_forward_accuracy


def _model(in_channels=8, hidden_dim=16, out_channels=4, depth=2):
    return MyModel(
        in_channels=in_channels, hidden_dim=hidden_dim,
        out_channels=out_channels, depth=depth,
    )


class TestMyModel:
    @pytest.mark.parametrize("in_channels, out_channels", [(8, 4), (16, 8)])
    def test_output_shape(self, device, in_channels, out_channels):
        model = _model(in_channels=in_channels, out_channels=out_channels)
        model = model.to(device).eval()
        x = torch.randn(2, 10, in_channels, device=device)
        out = model(x)
        assert out.shape == (2, 10, out_channels)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, device):
        model = _model().to(device).train()
        x = torch.randn(2, 10, 8, device=device, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_invalid_input(self, device):
        model = _model(in_channels=8).to(device).eval()
        with pytest.raises(ValueError):
            model(torch.randn(2, 10, 7, device=device))   # wrong in_channels

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            (dict(in_channels=8, hidden_dim=16, out_channels=4, depth=2),
             dict(in_channels=8)),
            (dict(in_channels=16, hidden_dim=32, out_channels=8, depth=3),
             dict(in_channels=16)),
        ],
    )
    def test_constructor_attributes(self, kwargs, expected):   # MOD-008a
        model = MyModel(**kwargs)
        for name, value in expected.items():
            assert getattr(model, name) == value

    def test_forward_accuracy(self, device):                   # MOD-008b
        torch.manual_seed(0)
        model = _model().to(device).eval()
        x = torch.randn(2, 10, 8, device=device)
        assert validate_forward_accuracy(
            model, (x,),
            file_name="experimental/models/<name>/data/my_model_output.pth",
            rtol=1e-3, atol=1e-3,
        )

    def test_checkpoint(self, device):                         # MOD-008c
        torch.manual_seed(0)
        x = torch.randn(2, 10, 8, device=device)
        assert validate_checkpoint(
            _model().to(device), _model().to(device), (x,)
        )
```

Notes:
- `validate_forward_accuracy` auto-creates the reference `.pth` on first run
  (and errors); run again to pass, then commit the reference file.
- If the model uses the TE-aware `LayerNorm`, add the skip-cpu-under-TE guard
  (see `references/lessons.md` §2).
- Annotate optional tensor args with jaxtyping too; single-token shapes need
  `# noqa: F821` (`lessons.md` §5).
