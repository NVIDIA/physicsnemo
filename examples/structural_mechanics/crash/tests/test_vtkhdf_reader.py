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

"""Contract test for ``vtkhdf_reader.Reader`` (pure h5py/numpy, no torch/physicsnemo).

Validates that the reader returns the 4-tuple the crash datapipe expects. Runs against a
local copy of the bumper dataset; set ``BUMPER_DATA_DIR`` / ``BUMPER_MASTER_CSV`` /
``BUMPER_GLOBALS_JSON`` to point at it, otherwise it auto-discovers the layout shipped in
this workspace and skips if absent.

Run directly:   python tests/test_vtkhdf_reader.py
Or via pytest:  pytest tests/test_vtkhdf_reader.py
"""

import os
import sys

import numpy as np

try:
    import pytest
except ModuleNotFoundError:  # allow running standalone without pytest installed
    class _Skip(Exception):
        pass

    class _PytestShim:
        Skipped = _Skip

        @staticmethod
        def skip(reason):
            raise _Skip(reason)

        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def deco(fn):
                    fn._skip = bool(condition)
                    fn._skip_reason = reason
                    return fn

                return deco

    pytest = _PytestShim()

THIS_DIR = os.path.dirname(__file__)
CRASH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if CRASH_DIR not in sys.path:
    sys.path.insert(0, CRASH_DIR)

import vtkhdf_reader as R  # noqa: E402

# Workspace default: physicsnemo/examples/structural_mechanics/crash/ -> 4 levels up.
WORKSPACE = os.path.abspath(os.path.join(CRASH_DIR, "..", "..", "..", ".."))
DATA_DIR = os.environ.get("BUMPER_DATA_DIR", os.path.join(WORKSPACE, "simulations"))
MASTER_CSV = os.environ.get(
    "BUMPER_MASTER_CSV", os.path.join(WORKSPACE, "bumper_beam_master_with_split.csv")
)
GLOBALS_JSON = os.environ.get(
    "BUMPER_GLOBALS_JSON", os.path.join(WORKSPACE, "global_features.json")
)

N_STEPS = 101
TARGET_KEYS = ("effective_plastic_strain", "stress_vm")


pytestmark = pytest.mark.skipif(
    not R.find_sim_files(DATA_DIR),
    reason=f"No VTKHDF samples found under {DATA_DIR}",
)


def _load(num_samples=2, split=None):
    reader = R.Reader(master_csv=MASTER_CSV)
    gf = GLOBALS_JSON if os.path.isfile(GLOBALS_JSON) else None
    return reader(
        data_dir=DATA_DIR,
        num_samples=num_samples,
        split=split,
        global_features_filepath=gf,
    )


def test_reader_contract():
    srcs, dsts, point_data_all, global_features_all = _load(num_samples=2)
    assert len(srcs) == len(dsts) == len(point_data_all) == len(global_features_all)

    for src, dst, rec in zip(srcs, dsts, point_data_all):
        coords = rec["coords"]
        N = coords.shape[1]
        assert coords.ndim == 3 and coords.shape[0] == N_STEPS and coords.shape[2] == 3

        # edges: parallel int arrays, indices within [0, N)
        assert src.shape == dst.shape and src.ndim == 1
        assert int(src.min()) >= 0 and int(dst.max()) < N

        pd = rec["point_data"]
        for key in TARGET_KEYS:
            steps = [k for k in pd if k.startswith(f"{key}_t")]
            assert len(steps) == N_STEPS, f"{key}: {len(steps)} steps, expected {N_STEPS}"
            assert pd[f"{key}_t0"].shape == (N,)

        # geometry actually evolves over time (the bumper deforms)
        disp = np.linalg.norm(coords[-1] - coords[0], axis=-1)
        assert disp.max() > 0.0


def test_global_features_contract():
    if not os.path.isfile(GLOBALS_JSON):
        pytest.skip("global_features.json not built")
    *_, global_features_all = _load(num_samples=1)
    assert set(global_features_all[0]) == {
        "velocity_x",
        "thickness_scale",
        "rwall_origin_y",
    }


def test_split_filter():
    if not os.path.isfile(MASTER_CSV):
        pytest.skip("master CSV not present")
    split_map = R.load_split_map(MASTER_CSV)
    for split in ("train", "validation", "test"):
        files = R.find_sim_files(DATA_DIR, split=split, split_map=split_map)
        want = "val" if split == "validation" else split
        for path in files:
            run_id = os.path.splitext(os.path.basename(path))[0]
            assert split_map[run_id] == want


if __name__ == "__main__":
    if hasattr(pytest, "main"):
        raise SystemExit(pytest.main([__file__, "-v"]))
    # Standalone fallback (no pytest installed): run each test function directly.
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        if getattr(fn, "_skip", False):
            print(f"SKIP {fn.__name__}: {getattr(fn, '_skip_reason', '')}")
            continue
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except pytest.Skipped as e:
            print(f"SKIP {fn.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    raise SystemExit(1 if failures else 0)
