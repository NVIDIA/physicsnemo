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

"""``physicsnemo`` merges portions installed in other directories into its ``__path__``."""

import subprocess
import sys
import textwrap


def test_extra_portion_on_sys_path_is_importable(tmp_path):
    """A ``physicsnemo/<sub>/`` tree elsewhere on sys.path becomes ``physicsnemo.<sub>``.

    This is how sibling distributions (and editable checkouts of them) contribute
    subpackages without shipping their own ``physicsnemo/__init__.py``.
    """
    portion = tmp_path / "physicsnemo" / "extra_portion_for_test"
    portion.mkdir(parents=True)
    (portion / "__init__.py").write_text("MARKER = 'merged'\n")

    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import physicsnemo
        from physicsnemo import Module  # anchor attributes still work
        assert any(p.startswith({str(tmp_path)!r}) for p in physicsnemo.__path__), physicsnemo.__path__
        from physicsnemo.extra_portion_for_test import MARKER
        assert MARKER == "merged"
        assert physicsnemo.__version__
        """
    )
    subprocess.run(  # noqa: S603 - interpreter and snippet are test constants
        [sys.executable, "-c", code], check=True
    )
