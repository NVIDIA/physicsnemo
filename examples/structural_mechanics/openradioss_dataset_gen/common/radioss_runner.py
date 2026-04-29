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

"""
Parallel OpenRadioss batch runner.

Executes the Starter, Engine, anim-to-VTK, and anim-to-D3PLOT steps over every
`run*/` subfolder of a dataset directory. Shared by bumper-beam and drop-test
flows — only the `input_base_name` and OpenRadioss install path differ.
"""

from __future__ import annotations

import concurrent.futures
import glob
import os
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class RunnerConfig:
    """Runtime configuration for the OpenRadioss batch runner."""

    openradioss_root: str
    dataset_dir: str
    input_base_name: str
    max_parallel_jobs: int = 2
    omp_num_threads: str = "8"
    debug_mode: bool = False


def build_radioss_env(openradioss_root: str, omp_num_threads: str) -> dict:
    """Build the env dict (LD_LIBRARY_PATH, OMP, RAD_CFG_PATH) for OpenRadioss."""
    env = os.environ.copy()
    env["OPENRADIOSS_PATH"] = openradioss_root
    env["RAD_CFG_PATH"] = os.path.join(openradioss_root, "hm_cfg_files")
    env["OMP_STACKSIZE"] = "400m"
    env["OMP_NUM_THREADS"] = omp_num_threads

    lib_paths = [
        os.path.join(openradioss_root, "extlib/hm_reader/linux64/"),
        os.path.join(openradioss_root, "extlib/h3d/lib/linux64/"),
    ]
    current_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{':'.join(lib_paths)}:{current_ld}"
    env["PYTHONPATH"] = env.get("PYTHONPATH", "")
    return env


def _run_d3plot_conversion(cwd: str, stem_name: str, env: dict, log_file) -> None:
    # Absolute stem path — vortex_radioss writes temp files relative to '/'
    # if given a relative path, so resolve it here.
    abs_stem_path = os.path.abspath(os.path.join(cwd, stem_name))
    python_cmd = (
        "from vortex_radioss.animtod3plot.Anim_to_D3plot import readAndConvert; "
        f"readAndConvert('{abs_stem_path}')"
    )
    subprocess.run(
        [sys.executable, "-c", python_cmd],
        cwd=cwd,
        env=env,
        stdout=log_file,
        stderr=log_file,
        check=True,
    )


def run_case(run_folder: str, cfg: RunnerConfig) -> bool:
    """Run Starter -> Engine -> VTK -> D3PLOT for one case folder."""
    starter_exe = os.path.join(cfg.openradioss_root, "exec/starter_linux64_gf")
    engine_exe = os.path.join(cfg.openradioss_root, "exec/engine_linux64_gf")
    to_vtk_exe = os.path.join(cfg.openradioss_root, "exec/anim_to_vtk_linux64_gf")

    starter_file = f"{cfg.input_base_name}_0000.rad"
    engine_file = f"{cfg.input_base_name}_0001.rad"

    case_id = os.path.basename(run_folder)
    log_prefix = f"[{case_id}]"
    print(f"{log_prefix} Starting processing in: {run_folder}")

    env = build_radioss_env(cfg.openradioss_root, cfg.omp_num_threads)
    cwd = run_folder

    starter_cmd = [starter_exe, "-i", starter_file, "-nt", cfg.omp_num_threads]
    try:
        with open(os.path.join(cwd, "starter.log"), "w") as log:
            subprocess.run(
                starter_cmd, cwd=cwd, env=env, stdout=log, stderr=log, check=True
            )
    except subprocess.CalledProcessError:
        print(f"{log_prefix} FAIL: Starter. See starter.log")
        return False

    engine_cmd = [engine_exe, "-i", engine_file]
    try:
        with open(os.path.join(cwd, "engine.log"), "w") as log:
            subprocess.run(
                engine_cmd, cwd=cwd, env=env, stdout=log, stderr=log, check=True
            )
        print(f"{log_prefix} Simulation Complete.")
    except subprocess.CalledProcessError:
        print(f"{log_prefix} FAIL: Engine. See engine.log")
        return False

    first_anim = os.path.join(cwd, f"{cfg.input_base_name}A001")
    if not os.path.exists(first_anim):
        print(
            f"{log_prefix} Warning: No animation files (A001) found. "
            "Skipping conversions."
        )
        return True

    try:
        anim_files = sorted(
            glob.glob(os.path.join(cwd, f"{cfg.input_base_name}A[0-9][0-9][0-9]"))
        )
        for a_file in anim_files:
            fname = os.path.basename(a_file)
            vtk_path = os.path.join(cwd, f"{fname}.vtk")
            with open(vtk_path, "w") as f_out:
                subprocess.run(
                    [to_vtk_exe, fname], cwd=cwd, env=env, stdout=f_out, check=True
                )
    except Exception as e:
        print(f"{log_prefix} Error in VTK conversion: {e}")

    try:
        with open(os.path.join(cwd, "d3plot_conv.log"), "w") as log:
            _run_d3plot_conversion(cwd, cfg.input_base_name, env, log)
        print(f"{log_prefix} D3PLOT Generated.")
    except Exception as e:
        print(f"{log_prefix} Error in D3PLOT conversion: {e} (See d3plot_conv.log)")

    return True


def run_batch(cfg: RunnerConfig) -> None:
    """Discover every `run*/` under `cfg.dataset_dir` and execute in parallel."""
    if not os.path.exists(cfg.dataset_dir):
        print(f"Error: Dataset directory '{cfg.dataset_dir}' not found.")
        sys.exit(1)

    all_runs = [
        f.path for f in os.scandir(cfg.dataset_dir) if f.is_dir() and "run" in f.name
    ]
    all_runs.sort()

    if not all_runs:
        print("No run folders found.")
        return

    if cfg.debug_mode:
        print(f"DEBUG MODE ON: Running only 1 case ({all_runs[0]})")
        run_case(all_runs[0], cfg)
        return

    print("--- STARTING BATCH EXECUTION ---")
    print(
        f"Cases: {len(all_runs)} | Jobs: {cfg.max_parallel_jobs} "
        f"| Threads/Job: {cfg.omp_num_threads}"
    )

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=cfg.max_parallel_jobs
    ) as executor:
        future_to_run = {
            executor.submit(run_case, folder, cfg): folder for folder in all_runs
        }
        for future in concurrent.futures.as_completed(future_to_run):
            folder = future_to_run[future]
            try:
                future.result()
            except Exception as e:
                print(f"Exception in {folder}: {e}")

    elapsed = time.time() - start_time
    print(f"BATCH COMPLETE. Total time: {elapsed:.2f} seconds.")
