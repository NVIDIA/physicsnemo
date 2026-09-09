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

"""Smoke tests for the ViT training-loop example.

Each test launches ``training_script.py`` through ``torchrun`` exactly the way
a user would, with small/fast settings, and asserts the run completes and
writes a results CSV.  Tests that need more GPUs than the machine has are
skipped, so the full matrix runs on a 4-GPU node.

Run from this directory:

    uv run pytest test_training_script.py -v
"""

import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).parent.resolve()
SCRIPT = EXAMPLE_DIR / "training_script.py"

# Keep every run small: one image size, minimal iterations.  Warmup must stay
# >= 1 so torch.compile compilation happens outside the timed region.
FAST_ARGS = [
    "--image_size_start",
    "512",
    "--image_size_stop",
    "512",
    "--num_warmup",
    "2",
    "--num_iterations",
    "3",
]

# 15 minutes: generous enough for the inductor-compiled configurations.
TIMEOUT_S = 900


def run_example(nproc, extra_args, workdir):
    """Launch the training script under torchrun and return the process result.

    Runs with ``workdir`` as the working directory so each test gets its own
    ``results/`` output, with the example directory on PYTHONPATH so the
    script's ``model``/``utils`` imports resolve.  ``--standalone`` picks a
    free rendezvous port, so sequential tests never collide.
    """
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={nproc}",
        str(SCRIPT),
        *FAST_ARGS,
        *extra_args,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{EXAMPLE_DIR}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(EXAMPLE_DIR)
    )
    return subprocess.run(
        cmd,
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )


def assert_success(result, workdir):
    """Assert the run exited cleanly and produced a well-formed results CSV."""
    assert result.returncode == 0, (
        f"torchrun exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    csv_files = list((workdir / "results").glob("*.csv"))
    assert len(csv_files) == 1, f"expected exactly one results CSV, got {csv_files}"
    with open(csv_files[0], newline="") as f:
        rows = list(csv.reader(f))
    # Header plus one row per image size (FAST_ARGS sweeps a single size).
    assert len(rows) == 2, f"expected header + 1 result row, got {rows}"
    assert "OOM" not in rows[1], f"benchmark row reported OOM: {rows[1]}"


def require_gpus(n):
    """Skip the current test unless at least ``n`` CUDA devices are available."""
    if torch.cuda.device_count() < n:
        pytest.skip(f"requires {n} GPUs, found {torch.cuda.device_count()}")


# (test id, nproc, script arguments) - the full smoke matrix.  Covers every
# wrapper path in training_script.py: bare module, DDP, domain-parallel with
# the broadcast path, plain DDP combined with domain parallelism, FSDP2
# (including the degenerate size-1 shard), whole-model and regional
# torch.compile, and the inference-only no-grad path.
CONFIGS = [
    ("single_gpu", 1, ["--batch_size", "1"]),
    ("ddp", 4, ["--ddp_size", "4", "--batch_size", "4"]),
    ("domain", 4, ["--domain_size", "4", "--batch_size", "1"]),
    (
        "ddp_x_domain",
        4,
        ["--ddp_size", "2", "--domain_size", "2", "--batch_size", "2"],
    ),
    ("fsdp", 4, ["--ddp_size", "4", "--batch_size", "4", "--fsdp"]),
    (
        "fsdp_x_domain",
        4,
        ["--ddp_size", "2", "--domain_size", "2", "--batch_size", "2", "--fsdp"],
    ),
    # ddp_size == 1: FSDP2 applies a degenerate size-1 shard, so parameters
    # are DTensors (including the domain-sharded pos_embed) with no ddp comms.
    (
        "fsdp_degenerate_shard",
        4,
        ["--domain_size", "4", "--ddp_size", "1", "--batch_size", "1", "--fsdp"],
    ),
    ("compile_ddp", 4, ["--ddp_size", "4", "--batch_size", "4", "--compile"]),
    # domain_size > 1 + --compile exercises the regional-compile path (ring
    # attention stays eager).
    ("compile_domain", 4, ["--domain_size", "4", "--batch_size", "1", "--compile"]),
    (
        "compile_fsdp_x_domain",
        4,
        [
            "--ddp_size",
            "2",
            "--domain_size",
            "2",
            "--batch_size",
            "2",
            "--fsdp",
            "--compile",
        ],
    ),
    (
        "inference_only_compile_domain",
        4,
        [
            "--domain_size",
            "4",
            "--batch_size",
            "1",
            "--compile",
            "--inference_only",
        ],
    ),
    # 3D: the transformer stack is dimension-agnostic; one domain-parallel
    # config exercises the 3D patch embed + volumetric sharding. Smaller
    # image size (the trailing args override FAST_ARGS - argparse last-wins).
    (
        "vit_3d_domain",
        2,
        [
            "--dimension",
            "3",
            "--domain_size",
            "2",
            "--batch_size",
            "1",
            "--image_size_start",
            "64",
            "--image_size_stop",
            "64",
        ],
    ),
]

# Alternate architectures get a compact matrix (the full wrapper/compile
# matrix above runs on ViT; these cover each architecture's sharded-op paths:
# halo convolutions for conv, neighborhood-attention halos for natten).
for _m in ("conv", "natten"):
    CONFIGS += [
        (
            f"{_m}_domain",
            4,
            ["--model", _m, "--domain_size", "4", "--batch_size", "1"],
        ),
        (
            f"{_m}_ddp_x_domain",
            4,
            [
                "--model",
                _m,
                "--ddp_size",
                "2",
                "--domain_size",
                "2",
                "--batch_size",
                "2",
            ],
        ),
        (
            f"{_m}_fsdp_x_domain",
            4,
            [
                "--model",
                _m,
                "--ddp_size",
                "2",
                "--domain_size",
                "2",
                "--batch_size",
                "2",
                "--fsdp",
            ],
        ),
        (
            f"{_m}_compile_domain",
            4,
            ["--model", _m, "--domain_size", "4", "--batch_size", "1", "--compile"],
        ),
    ]

# 3D variants. NOTE natten needs a larger volume: at 128^3 with patch 8 the
# patch grid is 16 per axis, so a 2-way shard leaves 8 >= kernel_size 7 rows
# per rank (smaller volumes deadlock the halo exchange).
CONFIGS += [
    (
        "conv_3d_domain",
        2,
        [
            "--model",
            "conv",
            "--dimension",
            "3",
            "--domain_size",
            "2",
            "--batch_size",
            "1",
            "--image_size_start",
            "64",
            "--image_size_stop",
            "64",
        ],
    ),
    (
        "natten_3d_domain",
        2,
        [
            "--model",
            "natten",
            "--dimension",
            "3",
            "--domain_size",
            "2",
            "--batch_size",
            "1",
            "--image_size_start",
            "128",
            "--image_size_stop",
            "128",
        ],
    ),
]


@pytest.mark.parametrize(
    "nproc,extra_args", [(n, a) for _, n, a in CONFIGS], ids=[c[0] for c in CONFIGS]
)
def test_training_script(nproc, extra_args, tmp_path):
    """Run one configuration from the distributed training smoke-test matrix."""
    require_gpus(nproc)
    if "natten" in extra_args and importlib.util.find_spec("natten") is None:
        pytest.skip("optional natten package not installed")
    result = run_example(nproc, extra_args, tmp_path)
    assert_success(result, tmp_path)


def test_fsdp_rejects_single_process(tmp_path):
    """--fsdp in a single-process job has no mesh and must fail loudly."""
    require_gpus(1)
    result = run_example(1, ["--batch_size", "1", "--fsdp"], tmp_path)
    assert result.returncode != 0, "--fsdp on a single process should error"
    assert "requires a distributed run" in result.stdout + result.stderr
