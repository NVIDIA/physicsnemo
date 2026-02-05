# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import numpy as np
from scipy.io import loadmat


def main(mat_file, npz_file):
    # Load the .mat file
    data = loadmat(mat_file)

    # Extract 'coeff' and 'sol'
    coeff = data["coeff"]
    sol = data["sol"]

    # Save to .npz file
    np.savez(npz_file, coeff=coeff, sol=sol)
    print(f"Saved 'coeff' and 'sol' to {npz_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_mat_to_npz.py input.mat")
        sys.exit(1)
    mat_file = sys.argv[1]
    npz_file = mat_file.replace(".mat", ".npz")
    main(mat_file, npz_file)
