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

"""Proof that a forced execution path ran: the guard of the patch-then-compare tests.

Many ISLA tests force a path (a resolver patched to name it, a module constant patched
to disqualify it) and compare its outputs with the default path's. A patch reaches the
code only while the code looks the name up where the patch puts it; once the code
moves (into the physicsnemo library, say), such a test compares the default path with
itself and passes vacuously.

:class:`ran` closes that hole. It counts the executions of the code objects of the
functions it is given, through ``sys.monitoring`` (PEP 669), whichever module the
caller looked them up in, and records the arguments and the return value of each call.
A test forces a path inside ``with ran(f, g) as r:`` and then asserts ``r.count(f) > 0``
(the forced code ran) and ``r.count(g) == 0`` (the other did not), or reads
``r.returns(f)`` (a fallback taken: the forced code returned ``None``).

Only eager Python execution is seen: code that ``torch.compile`` traces into a graph
does not run as Python, and a Warp or Triton kernel has no Python frame; give the
Python function that launches it.
"""

import sys
import threading
import types

import pytest
import torch

### sys.monitoring (PEP 669) exists from Python 3.12; on 3.11 a test that uses ran skips.
_monitoring = getattr(sys, "monitoring", None)


def code_of(fn):
    """The code object that runs when ``fn`` is called: a function, a bound method, a
    ``staticmethod`` or ``classmethod``, a ``torch.library.custom_op`` (its Python
    implementation), a ``torch.autograd.Function`` subclass (its ``forward``) or a code
    object itself (a nested function's, from its parent's ``__code__.co_consts``)."""
    if isinstance(fn, types.CodeType):
        return fn
    if isinstance(fn, type) and issubclass(fn, torch.autograd.Function):
        fn = fn.forward
    fn = getattr(fn, "_init_fn", fn)  # torch.library.custom_op's CustomOpDef
    fn = getattr(fn, "__func__", fn)  # bound methods, staticmethod, classmethod
    fn = getattr(fn, "__wrapped__", fn)  # functools.wraps
    code = getattr(fn, "__code__", None)
    if code is None:
        raise TypeError(f"{fn!r} runs no Python code that can be watched")
    return code


class ran:
    """Count and record the calls of ``functions`` while the ``with`` block runs, on
    every thread (autograd's backward runs on its own device threads)."""

    def __init__(self, *functions):
        self._codes = {code_of(f): f for f in functions}
        self._calls = {c: [] for c in self._codes}
        self._returns = {c: [] for c in self._codes}
        self._tool = None

    def _start(self, code, offset):
        frame = sys._getframe(1)
        n = code.co_argcount + code.co_kwonlyargcount
        self._calls[code].append(
            {k: frame.f_locals[k] for k in code.co_varnames[:n] if k in frame.f_locals}
        )

    def _return(self, code, offset, value):
        self._returns[code].append(value)

    def __enter__(self):
        if _monitoring is None:
            pytest.skip("the forced-path guard needs sys.monitoring (Python 3.12+)")
        events = _monitoring.events.PY_START | _monitoring.events.PY_RETURN
        free = [i for i in (3, 4, 5, 2, 1, 0) if _monitoring.get_tool(i) is None]
        if not free:
            raise RuntimeError("every sys.monitoring tool id is taken")
        self._tool = free[0]
        _monitoring.use_tool_id(self._tool, "isla-forced-path")
        _monitoring.register_callback(
            self._tool, _monitoring.events.PY_START, self._start
        )
        _monitoring.register_callback(
            self._tool, _monitoring.events.PY_RETURN, self._return
        )
        for code in self._codes:
            _monitoring.set_local_events(self._tool, code, events)
        return self

    def __exit__(self, *exc):
        for code in self._codes:
            _monitoring.set_local_events(self._tool, code, 0)
        _monitoring.register_callback(self._tool, _monitoring.events.PY_START, None)
        _monitoring.register_callback(self._tool, _monitoring.events.PY_RETURN, None)
        _monitoring.free_tool_id(self._tool)
        self._tool = None
        return False

    def count(self, fn) -> int:
        """How many times ``fn``'s code started."""
        return len(self._calls[code_of(fn)])

    def calls(self, fn) -> list[dict]:
        """The arguments of each call of ``fn``, by parameter name."""
        return self._calls[code_of(fn)]

    def returns(self, fn) -> list:
        """The value each call of ``fn`` returned (calls that raised are absent)."""
        return self._returns[code_of(fn)]


def reaches(monkeypatch, owner, name: str) -> list:
    """Replace the function ``owner.name`` by a wrapper that records each call's
    arguments and calls the original, as a patch-then-compare test's spy or forced
    resolver replaces it; the list it returns fills with one ``(args, kwargs)`` per
    call that the patch reached. Empty after a call that should look ``name`` up in
    ``owner``: the patch does not reach the code."""
    calls = []
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)
    return calls


def in_another_thread_while_compiling(fn):
    """Return ``fn()``, run eagerly in a second thread while this thread compiles.

    Since torch 2.13, ``torch.compiler.is_compiling()`` is True in every thread during
    a compile, so code that guards an eager path with it skips that path here."""
    result = {}

    def backend(gm, example_inputs):
        thread = threading.Thread(target=lambda: result.update(value=fn()))
        thread.start()
        thread.join()
        return gm.forward

    torch._dynamo.reset()
    torch.compile(lambda x: torch.sin(x) + 1, backend=backend)(torch.randn(4))
    return result["value"]
