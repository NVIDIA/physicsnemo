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


"""Miscellaneous utility classes and functions."""

import glob
import hashlib
import html
import os
import tempfile
import contextlib
import urllib
import urllib.request

from typing import Any, Tuple
import requests
import torch
import uuid
import re
import importlib
import sys
import types
import io

# Cache directories
# -------------------------------------------------------------------------------------

_dnnlib_cache_dir = None


def set_cache_dir(path: str) -> None:  # pragma: no cover
    global _dnnlib_cache_dir
    _dnnlib_cache_dir = path


def make_cache_dir_path(*paths: str) -> str:  # pragma: no cover
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if "DNNLIB_CACHE_DIR" in os.environ:
        return os.path.join(os.environ["DNNLIB_CACHE_DIR"], *paths)
    if "HOME" in os.environ:
        return os.path.join(os.environ["HOME"], ".cache", "dnnlib", *paths)
    if "USERPROFILE" in os.environ:
        return os.path.join(os.environ["USERPROFILE"], ".cache", "dnnlib", *paths)
    return os.path.join(tempfile.gettempdir(), ".cache", "dnnlib", *paths)


# URL helpers
# ------------------------------------------------------------------------------------------


def is_url(obj: Any, allow_file_urls: bool = False) -> bool:  # pragma: no cover
    """
    Determine whether the given object is a valid URL string.
    """
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith("file://"):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


def open_url(
    url: str,
    cache_dir: str = None,
    num_attempts: int = 10,
    verbose: bool = True,
    return_filename: bool = False,
    cache: bool = True,
) -> Any:  # pragma: no cover
    """
    Download the given URL and return a binary-mode file object to access the data.
    This code handles unusual file:// patterns that
    arise on Windows:

    file:///c:/foo.txt

    which would translate to a local '/c:/foo.txt' filename that's
    invalid.  Drop the forward slash for such pathnames.

    If you touch this code path, you should test it on both Linux and
    Windows.

    Some internet resources suggest using urllib.request.url2pathname() but
    but that converts forward slashes to backslashes and this causes
    its own set of problems.
    """
    if not num_attempts >= 1:
        raise ValueError("num_attempts must be at least 1")
    if return_filename and (not cache):
        raise ValueError("return_filename requires cache=True")

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match("^[a-z]+://", url):
        return url if return_filename else open(url, "rb")

    if url.startswith("file://"):
        filename = urllib.parse.urlparse(url).path
        if re.match(r"^/[a-zA-Z]:", filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    if not is_url(url):
        raise IOError("Not a URL: " + url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path("downloads")

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [
                                html.unescape(link)
                                for link in content_str.split('"')
                                if "export=download" in link
                            ]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError(
                                "Google Drive download quota exceeded -- please try again later"
                            )

                    match = re.search(
                        r'filename="([^"]*)"',
                        res.headers.get("Content-Disposition", ""),
                    )
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        safe_name = safe_name[: min(len(safe_name), 128)]
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(
            cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name
        )
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file)  # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    if return_filename:
        raise ValueError("return_filename requires cache=True")
    return io.BytesIO(url_data)


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


# ----------------------------------------------------------------------------
# Check DistributedDataParallel consistency across processes.


def named_params_and_buffers(module):  # pragma: no cover
    """Get named parameters and buffers of a nn.Module"""
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module instance")
    return list(module.named_parameters()) + list(module.named_buffers())


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
            tensor = torch.nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        if not (tensor == other).all():
            raise RuntimeError(f"DDP consistency check failed for {fullname}")


# ----------------------------------------------------------------------------
# Utilities for operating with torch.nn.Module parameters and buffers.


def params_and_buffers(module):  # pragma: no cover
    """Get parameters and buffers of a nn.Module"""
    if not isinstance(module, torch.nn.Module):
        raise TypeError("module must be a torch.nn.Module instance")
    return list(module.parameters()) + list(module.buffers())


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


# ----------------------------------------------------------------------------
# Function decorator that calls torch.autograd.profiler.record_function().


def profiled_function(fn):  # pragma: no cover
    """Function decorator that calls torch.autograd.profiler.record_function()."""

    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)

    decorator.__name__ = fn.__name__
    return decorator
