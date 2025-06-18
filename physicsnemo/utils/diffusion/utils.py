# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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


"""Miscellaneous utility classes and functions."""

import contextlib
import ctypes
import datetime
import fnmatch
import importlib
import inspect
import os
import re
import shutil
import sys
import types
import warnings
from typing import Any, Iterator, List, Tuple, Union

import cftime
import numpy as np
import torch

# ruff: noqa: E722 PERF203 S110 E713 S324


class EasyDict(dict):  # pragma: no cover
    """
    Convenience class that behaves like a dict but allows access with the attribute
    syntax.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class StackedRandomGenerator:  # pragma: no cover
    """
    Wrapper for torch.Generator that allows specifying a different random seed
    for each sample in a minibatch.
    """

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


def parse_int_list(s):  # pragma: no cover
    """
    Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
    """
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# Small util functions
# -------------------------------------------------------------------------------------
def convert_datetime_to_cftime(
    time: datetime.datetime, cls=cftime.DatetimeGregorian
) -> cftime.DatetimeGregorian:
    """Convert a Python datetime object to a cftime DatetimeGregorian object."""
    return cls(time.year, time.month, time.day, time.hour, time.minute, time.second)


def time_range(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    step: datetime.timedelta,
    inclusive: bool = False,
):
    """Like the Python `range` iterator, but with datetimes."""
    t = start_time
    while (t <= end_time) if inclusive else (t < end_time):
        yield t
        t += step


def format_time(seconds: Union[int, float]) -> str:  # pragma: no cover
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(
            s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60
        )


def format_time_brief(seconds: Union[int, float]) -> str:  # pragma: no cover
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m".format(s // (60 * 60), (s // 60) % 60)
    else:
        return "{0}d {1:02}h".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24)


def tuple_product(t: Tuple) -> Any:  # pragma: no cover
    """Calculate the product of the tuple elements."""
    result = 1

    for v in t:
        result *= v

    return result


_str_to_ctype = {
    "uint8": ctypes.c_ubyte,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "int8": ctypes.c_byte,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double,
}


def get_dtype_and_ctype(type_obj: Any) -> Tuple[np.dtype, Any]:  # pragma: no cover
    """
    Given a type name string (or an object having a __name__ attribute), return
    matching Numpy and ctypes types that have the same size in bytes.
    """
    type_str = None

    if isinstance(type_obj, str):
        type_str = type_obj
    elif hasattr(type_obj, "__name__"):
        type_str = type_obj.__name__
    elif hasattr(type_obj, "name"):
        type_str = type_obj.name
    else:
        raise RuntimeError("Cannot infer type name from input")

    if type_str not in _str_to_ctype.keys():
        raise ValueError("Unknown type name: " + type_str)

    my_dtype = np.dtype(type_str)
    my_ctype = _str_to_ctype[type_str]

    if my_dtype.itemsize != ctypes.sizeof(my_ctype):
        raise ValueError(
            "Numpy and ctypes types for '{}' have different sizes!".format(type_str)
        )

    return my_dtype, my_ctype


# Functionality to import modules/objects by name, and call functions by name
# -------------------------------------------------------------------------------------


def get_module_from_obj_name(
    obj_name: str,
) -> Tuple[types.ModuleType, str]:  # pragma: no cover
    """
    Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed).
    """

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [
        (".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)
    ]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)  # may raise ImportError
            get_obj_from_module(module, local_obj_name)  # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name)  # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith(
                "No module named '" + module_name + "'"
            ):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)  # may raise ImportError
            get_obj_from_module(module, local_obj_name)  # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(
    module: types.ModuleType, obj_name: str
) -> Any:  # pragma: no cover
    """
    Traverses the object name and returns the last (rightmost) python object.
    """
    if obj_name == "":
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:  # pragma: no cover
    """
    Finds the python object with the given name.
    """
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(
    *args, func_name: str = None, **kwargs
) -> Any:  # pragma: no cover
    """
    Finds the python object with the given name and calls it as a function.
    """
    if func_name is None:
        raise ValueError("func_name must be specified")
    func_obj = get_obj_by_name(func_name)
    if not callable(func_obj):
        raise ValueError(func_name + " is not callable")
    return func_obj(*args, **kwargs)


def construct_class_by_name(
    *args, class_name: str = None, **kwargs
) -> Any:  # pragma: no cover
    """
    Finds the python class with the given name and constructs it with the given
    arguments.
    """
    return call_func_by_name(*args, func_name=class_name, **kwargs)


def get_module_dir_by_obj_name(obj_name: str) -> str:  # pragma: no cover
    """
    Get the directory path of the module containing the given object name.
    """
    module, _ = get_module_from_obj_name(obj_name)
    return os.path.dirname(inspect.getfile(module))


def is_top_level_function(obj: Any) -> bool:  # pragma: no cover
    """
    Determine whether the given object is a top-level function, i.e., defined at module
    scope using 'def'.
    """
    return callable(obj) and obj.__name__ in sys.modules[obj.__module__].__dict__


def get_top_level_function_name(obj: Any) -> str:  # pragma: no cover
    """
    Return the fully-qualified name of a top-level function.
    """
    if not is_top_level_function(obj):
        raise ValueError("Object is not a top-level function")
    module = obj.__module__
    if module == "__main__":
        module = os.path.splitext(os.path.basename(sys.modules[module].__file__))[0]
    return module + "." + obj.__name__


# File system helpers
# ------------------------------------------------------------------------------------------


def list_dir_recursively_with_ignore(
    dir_path: str, ignores: List[str] = None, add_base_to_relative: bool = False
) -> List[Tuple[str, str]]:  # pragma: no cover
    """
    List all files recursively in a given directory while ignoring given file and
    directory names. Returns list of tuples containing both absolute and relative paths.
    """
    if not os.path.isdir(dir_path):
        raise RuntimeError(f"Directory does not exist: {dir_path}")
    base_name = os.path.basename(os.path.normpath(dir_path))

    if ignores is None:
        ignores = []

    result = []

    for root, dirs, files in os.walk(dir_path, topdown=True):
        for ignore_ in ignores:
            dirs_to_remove = [d for d in dirs if fnmatch.fnmatch(d, ignore_)]

            # dirs need to be edited in-place
            for d in dirs_to_remove:
                dirs.remove(d)

            files = [f for f in files if not fnmatch.fnmatch(f, ignore_)]

        absolute_paths = [os.path.join(root, f) for f in files]
        relative_paths = [os.path.relpath(p, dir_path) for p in absolute_paths]

        if add_base_to_relative:
            relative_paths = [os.path.join(base_name, p) for p in relative_paths]

        if len(absolute_paths) != len(relative_paths):
            raise ValueError("Number of absolute and relative paths do not match")
        result += zip(absolute_paths, relative_paths)

    return result


def copy_files_and_create_dirs(
    files: List[Tuple[str, str]]
) -> None:  # pragma: no cover
    """
    Takes in a list of tuples of (src, dst) paths and copies files.
    Will create all necessary directories.
    """
    for file in files:
        target_dir_name = os.path.dirname(file[1])

        # will create all intermediate-level directories
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)

        shutil.copyfile(file[0], file[1])


# ----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()


def constant(
    value, shape=None, dtype=None, device=None, memory_format=None
):  # pragma: no cover
    """Cached construction of constant tensors"""
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (
        value.shape,
        value.dtype,
        value.tobytes(),
        shape,
        dtype,
        device,
        memory_format,
    )
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


# ----------------------------------------------------------------------------
# Replace NaN/Inf with specified numerical values.

try:
    nan_to_num = torch.nan_to_num  # 1.8.0a0
except AttributeError:

    def nan_to_num(
        input, nan=0.0, posinf=None, neginf=None, *, out=None
    ):  # pylint: disable=redefined-builtin  # pragma: no cover
        """Replace NaN/Inf with specified numerical values"""
        if not isinstance(input, torch.Tensor):
            raise TypeError("input should be a Tensor")
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        if nan != 0:
            raise ValueError("nan_to_num only supports nan=0")
        return torch.clamp(
            input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out
        )


# ----------------------------------------------------------------------------
# Symbolic assert.

try:
    symbolic_assert = torch._assert  # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert  # 1.7.0

# ----------------------------------------------------------------------------
# Context manager to temporarily suppress known warnings in torch.jit.trace().
# Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672


@contextlib.contextmanager
def suppress_tracer_warnings():  # pragma: no cover
    """
    Context manager to temporarily suppress known warnings in torch.jit.trace().
    Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672
    """
    flt = ("ignore", None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)


# ----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().


def assert_shape(tensor, ref_shape):  # pragma: no cover
    """
    Assert that the shape of a tensor matches the given list of integers.
    None indicates that the size of a dimension is allowed to vary.
    Performs symbolic assertion when used in torch.jit.trace().
    """
    if tensor.ndim != len(ref_shape):
        raise AssertionError(
            f"Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}"
        )
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(torch.as_tensor(size), ref_size),
                    f"Wrong size for dimension {idx}",
                )
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(size, torch.as_tensor(ref_size)),
                    f"Wrong size for dimension {idx}: expected {ref_size}",
                )
        elif size != ref_size:
            raise AssertionError(
                f"Wrong size for dimension {idx}: got {size}, expected {ref_size}"
            )


# ----------------------------------------------------------------------------
# Function decorator that calls torch.autograd.profiler.record_function().


def profiled_function(fn):  # pragma: no cover
    """Function decorator that calls torch.autograd.profiler.record_function()."""

    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)

    decorator.__name__ = fn.__name__
    return decorator


class InfiniteSampler(torch.utils.data.Sampler[int]):  # pragma: no cover
    """Sampler for torch.utils.data.DataLoader that loops over the dataset indefinitely.

    This sampler yields indices indefinitely, optionally shuffling items as it goes.
    It can also perform distributed sampling when `rank` and `num_replicas` are
    specified.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to sample from
    rank : int, default=0
        The rank of the current process within num_replicas processes
    num_replicas : int, default=1
        The number of processes participating in distributed sampling
    shuffle : bool, default=True
        Whether to shuffle the indices
    seed : int, default=0
        Random seed for reproducibility when shuffling
    window_size : float, default=0.5
        Fraction of dataset to use as window for shuffling. Must be between 0 and 1.
        A larger window means more thorough shuffling but slower iteration.
    start_idx : int, default=0
        The initial index to use for the sampler. This is used for resuming training.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        rank: int = 0,
        num_replicas: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        window_size: float = 0.5,
        start_idx: int = 0,
    ):
        if not len(dataset) > 0:
            raise ValueError("Dataset must contain at least one item")
        if not num_replicas > 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be non-negative and less than num_replicas")
        if not 0 <= window_size <= 1:
            raise ValueError("window_size must be between 0 and 1")
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size
        self.start_idx = start_idx

    def __iter__(self) -> Iterator[int]:
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = self.start_idx
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


# ----------------------------------------------------------------------------
# Utilities for operating with torch.nn.Module parameters and buffers.


def params_and_buffers(module):  # pragma: no cover
    """Get parameters and buffers of a nn.Module"""
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module instance")
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):  # pragma: no cover
    """Get named parameters and buffers of a nn.Module"""
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module instance")
    return list(module.named_parameters()) + list(module.named_buffers())


@torch.no_grad()
def copy_params_and_buffers(
    src_module, dst_module, require_all=False
):  # pragma: no cover
    """Copy parameters and buffers from a source module to target module"""
    if not isinstance(src_module, torch.nn.Module):
        raise TypeError("src_module must be a torch.nn.Module instance")
    if not isinstance(dst_module, torch.nn.Module):
        raise TypeError("dst_module must be a torch.nn.Module instance")
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        if not ((name in src_tensors) or (not require_all)):
            raise ValueError(f"Missing source tensor for {name}")
        if name in src_tensors:
            tensor.copy_(src_tensors[name])


# ----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.


@contextlib.contextmanager
def ddp_sync(module, sync):  # pragma: no cover
    """
    Context manager for easily enabling/disabling DistributedDataParallel
    synchronization.
    """
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module instance")
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield


# ----------------------------------------------------------------------------
# Check DistributedDataParallel consistency across processes.


def check_ddp_consistency(module, ignore_regex=None):  # pragma: no cover
    """Check DistributedDataParallel consistency across processes."""
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module instance")
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + "." + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        if not (tensor == other).all():
            raise RuntimeError(f"DDP consistency check failed for {fullname}")


# ----------------------------------------------------------------------------
# Print summary table of module hierarchy.


def print_module_summary(
    module, inputs, max_nesting=3, skip_redundant=True
):  # pragma: no cover
    """Print summary table of module hierarchy."""
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module instance")
    if isinstance(module, torch.jit.ScriptModule):
        raise TypeError("module must not be a torch.jit.ScriptModule instance")
    if not isinstance(inputs, (tuple, list)):
        raise TypeError("inputs must be a tuple or list")

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {
            id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs
        }

    # Filter out redundant entries.
    if skip_redundant:
        entries = [
            e
            for e in entries
            if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)
        ]

    # Construct table.
    rows = [
        [type(module).__name__, "Parameters", "Buffers", "Output shape", "Datatype"]
    ]
    rows += [["---"] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = "<top-level>" if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split(".")[-1] for t in e.outputs]
        rows += [
            [
                name + (":0" if len(e.outputs) >= 2 else ""),
                str(param_size) if param_size else "-",
                str(buffer_size) if buffer_size else "-",
                (output_shapes + ["-"])[0],
                (output_dtypes + ["-"])[0],
            ]
        ]
        for idx in range(1, len(e.outputs)):
            rows += [
                [name + f":{idx}", "-", "-", output_shapes[idx], output_dtypes[idx]]
            ]
        param_total += param_size
        buffer_total += buffer_size
    rows += [["---"] * len(rows[0])]
    rows += [["Total", str(param_total), str(buffer_total), "-", "-"]]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    for row in rows:
        print(
            "  ".join(
                cell + " " * (width - len(cell)) for cell, width in zip(row, widths)
            )
        )
    return outputs


# ----------------------------------------------------------------------------
