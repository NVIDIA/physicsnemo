# ignore_header_test
# ruff: noqa: E402

r"""
Transolver model for physics-informed neural operator learning.

This module provides the Transolver model, which adapts the transformer
architecture with a physics-attention mechanism for solving partial
differential equations on both structured and unstructured meshes.

The Transolver model learns to project inputs onto physics-informed slices
before applying attention, enabling efficient learning of physical systems.

This code was modified from https://github.com/thuml/Transolver

The following license is provided from their source,

MIT License

Copyright (c) 2024 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

References
----------
- `Transolver paper <https://arxiv.org/pdf/2402.02366>`_
- `Transolver++ paper <https://arxiv.org/pdf/2502.02414>`_

Examples
--------
Structured 2D data with unified position:

>>> import torch
>>> from physicsnemo.models.transolver import Transolver
>>> model = Transolver(
...     functional_dim=3,
...     out_dim=1,
...     structured_shape=(64, 64),
...     unified_pos=True,
...     n_hidden=128,
...     n_head=4,
...     use_te=False,
... )
>>> x = torch.randn(2, 64, 64, 3)
>>> out = model(x)
>>> out.shape
torch.Size([2, 64, 64, 1])

Unstructured mesh data:

>>> model = Transolver(
...     functional_dim=2,
...     embedding_dim=3,
...     out_dim=1,
...     structured_shape=None,
...     unified_pos=False,
...     n_hidden=128,
...     n_head=4,
...     use_te=False,
... )
>>> fx = torch.randn(2, 1000, 2)
>>> emb = torch.randn(2, 1000, 3)
>>> out = model(fx, embedding=emb)
>>> out.shape
torch.Size([2, 1000, 1])
"""

from .transolver import Transolver
