# Serialization — the `physicsnemo.Module` contract (and a common trap)

This is a common thing external contributors get wrong. Read it before
scaffolding any `__init__`.

## Why `physicsnemo.Module`, not `torch.nn.Module`

`MOD-001` requires model/layer classes to subclass **`physicsnemo.Module`**
(itself a subclass of `torch.nn.Module`). The payoff is `Module.save(...)` /
`physicsnemo.Module.from_checkpoint(...)` / `from_pretrained(...)` and the model
registry — the public way users load models. A plain `torch.nn.Module` gets
none of it.

## The contract

`physicsnemo.Module` captures the `__init__` arguments at construction and
serializes them as JSON (`args.json` inside the `.mdlus` archive). Therefore:

> **Every `__init__` argument must be JSON-serializable** — ints, floats,
> strings, bools, lists/dicts of those, or `None`. The **only** exception is an
> argument that is itself a `physicsnemo.Module` instance (those are recursed
> into and serialized).

Read the exact rule and the mechanism in `physicsnemo/core/module.py` (the
class docstring + the `save` / `_save_process` path) — don't trust this summary
over the source.

### The trap

Passing a **raw `torch.nn.Module`** as a constructor argument (a common
"inject my submodule" pattern) makes `save()` raise:

```
TypeError: Submodule <x> ... is a PyTorch module, which is not supported by
'Module.save'. Please first convert it ... using 'Module.from_torch'.
```

The model trains and runs `forward` fine, so this is easy to miss — it only
fails at `save()` / `from_checkpoint()`, i.e. the moment a user tries to
persist it.

## The two correct constructor patterns

**(A) Constructor-from-config (preferred for new models).** Take
JSON-serializable primitives / nested dicts, and build submodules *internally*:

```python
class MyModel(physicsnemo.Module):
    def __init__(self, *, in_dim: int, hidden_dim: int, depth: int = 4):
        super().__init__(meta=MyModelMetaData())
        self.encoder = Encoder(in_dim, hidden_dim)   # built here, not passed in
        self.blocks = nn.ModuleList(EncoderBlock(hidden_dim) for _ in range(depth))
```

This is also why `MOD-009` (no string-based class selection) and `MOD-010`
(no splatted `**kwargs`) exist — keep the config explicit and serializable.

**(B) Dependency injection — only with `physicsnemo.Module` submodules.** If
the API genuinely needs to accept submodules, they (and every nested submodule)
must be `physicsnemo.Module`, converted from torch via `Module.from_torch`:

```python
from physicsnemo import Module
PNMEncoder = Module.from_torch(TorchEncoder, meta=ModelMetaData())
model = MyModel(encoder=PNMEncoder(in_dim=64))   # serializable
```

## Wrapping an existing external model

For "I already have a `nn.Module`, make it a physicsnemo model":

```python
from physicsnemo import Module
from physicsnemo.core import ModelMetaData
from my_pkg import MyTorchNet                      # their architecture, untouched

PNMNet = Module.from_torch(MyTorchNet, meta=ModelMetaData())
# Now PNMNet(...) is a physicsnemo.Module: save/from_checkpoint/registry work,
# AS LONG AS MyTorchNet.__init__ args are JSON-serializable (or are themselves
# converted physicsnemo.Modules). If MyTorchNet injects nested nn.Modules,
# convert each of them with Module.from_torch first.
```

## Always verify with a round-trip test

Don't assume — prove it. Use `validate_checkpoint` from `test.common` (save
model_1, load into model_2, assert forward outputs match):

```python
from test.common import validate_checkpoint
assert validate_checkpoint(model_1, model_2, (x, ...))
```

If this passes, the serialization contract holds. If construction args aren't
JSON-serializable, it fails here — fix the constructor (pattern A or B), don't
loosen the test.

## `ModelMetaData`

A model declares capabilities via a `ModelMetaData` dataclass passed to
`super().__init__(meta=...)` — `amp`, `cuda_graphs`, `onnx_*`, `auto_grad`,
etc. Default everything conservatively (`False`) and only flip a flag on once
you've verified that capability. Layers in `nn/module/` typically don't need a
meta (call `super().__init__()`); confirm against a sibling layer.
