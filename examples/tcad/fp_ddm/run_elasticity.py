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

"""Run the numerical plane-stress baseline for FP-DDM."""

import argparse

from fpddm.elasticity_example import run_elasticity


def main() -> None:
    """Run the comparison and print its main accuracy metrics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/fp_ddm/elasticity")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    result = run_elasticity(
        args.output_dir,
        size=args.size,
        max_iterations=args.max_iterations,
        visualize=not args.no_plot,
    )
    print(f"Converged: {result.converged} ({len(result.metrics)} iterations)")
    print(f"Displacement relative error: {result.displacement_relative_error:.3e}")
    print(f"Stress relative error: {result.stress_relative_error:.3e}")
    if result.comparison_path is not None:
        print(f"Comparison plot: {result.comparison_path}")


if __name__ == "__main__":
    main()
