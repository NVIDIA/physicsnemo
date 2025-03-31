# PhysicsNeMo Contribution Guide

## Introduction

Welcome to Project PhysicsNeMo! We're excited you're here and want to contribute.
This documentation is intended for individuals and institutions interested in
contributing to PhysicsNeMo. PhysicsNeMo is an open-source project and, as such, its
success relies on its community of contributors willing to keep improving it.
Your contribution will be a valued addition to the code base; we simply ask
that you read this page and understand our contribution process, whether you
are a seasoned open-source contributor or whether you are a first-time
contributor.

### Communicate with Us

We are happy to talk with you about your needs for PhysicsNeMo and your ideas for
contributing to the project. One way to do this is to create an issue discussing
your thoughts. It might be that a very similar feature is under development or
already exists, so an issue is a great starting point. If you are looking for an
issue to resolve that will help, refer to the
[issue](https://github.com/NVIDIA/physicsnemo/issues) section.
If you are considering collaborating with NVIDIA PhysicsNeMo team to enhance PhysicsNeMo,
fill this [proposal form](https://forms.gle/fYsbZEtgRWJUQ3oQ9) and
we will get back to you.

## Contribute to PhysicsNeMo-Core

### Pull Requests

Developer workflow for code contributions is as follows:

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo)
the [upstream](https://github.com/NVIDIA/physicsnemo) PhysicsNeMo repository.

2. Git clone the forked repository and push changes to the personal fork.

3. Once the code changes are staged on the fork and ready for review, a
[Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR)
can be [requested](https://help.github.com/en/articles/creating-a-pull-request)
to merge the changes from a branch of the fork into a selected branch of upstream.

    - Exercise caution when selecting the source and target branches for the PR.
    - Ensure that you update the [`CHANGELOG.md`](CHANGELOG.md) to reflect your contributions.
    - Creation of a PR creation kicks off CI and a code review process.
    - Atleast one PhysicsNeMo engineer will be assigned for the review.

4. The PR will be accepted and the corresponding issue closed after adequate review and
testing has been completed. Note that every PR should correspond to an open issue and
should be linked on Github.

### Licensing Information

All source code files should start with this paragraph:

```bash
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
```

### Signing Your Work

- We require that all contributors "sign-off" on their commits. This certifies that the
contribution is your original work, or you have rights to submit it under the same
license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when
committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```text
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```text
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license 
    document, but changing it is not allowed.
  ```

  ```text
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to 
    submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge,
    is covered under an appropriate open source license and I have the right under that
    license to submit that work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am permitted to submit under a
    different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified
    (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and
    that a record of the contribution (including all personal information I submit with
    it, including my sign-off) is maintained indefinitely and may be redistributed
    consistent with this project or the open source license(s) involved.

  ```

### Pre-commit

For PhysicsNeMo development, [pre-commit](https://pre-commit.com/) is **required**.
This will not only help developers pass the CI pipeline, but also accelerate reviews.
Contributions that have not used pre-commit will *not be reviewed*.

To install `pre-commit` follow the below steps inside the PhysicsNeMo repository folder:

```bash
pip install pre-commit
pre-commit install
```

Once the above commands are executed, the pre-commit hooks will be activated and all
the commits will be checked for appropriate formatting.

### CI

To ensure quality of the code, your merge request will pass through several CI checks.
It is mandatory for your MRs to pass these pipelines to ensure a successful merge.
Please keep checking this document for the latest guidelines on pushing code. Currently,
The pipeline has following stages:

1. `format`
    *Pre-commit will check this for you!*
    Checks for formatting of your python code.
    Refer [black](https://black.readthedocs.io/en/stable/) for more information.
    If your MR fails this test, run `black <script-name>.py` on problematic scripts and
    black will take care of the rest.

2. `interrogate`
   *Pre-commit will check this for you!*
   Checks if the code being pushed is well documented. The goal is to make the
   documentation live inside code. Very few exceptions are made.
   Elements that are fine to have no documentation include `init-module`, `init-method`,
   `private` and `semiprivate` classes/functions and `dunder` methods. For definitions of
   these, refer [interrogate](https://interrogate.readthedocs.io/en/latest/). Meaning for
   some methods/functions is very explicit and exceptions for these are made. These
   include `forward`, `reset_parameters`, `extra_repr`, `MetaData`. If your MR fails this
   test, add the missing documentation. Take a look at the pipeline output for hints on
   which functions/classes need documentation.
   To test the documentation before making a commit, you can run the following during
   your development

    ```bash
    interrogate \
      --ignore-init-method \
      --ignore-init-module \
      --ignore-module \
      --ignore-private \
      --ignore-semiprivate \
      --ignore-magic \
      --fail-under 99 \
      --exclude '[setup.py]' \
      --ignore-regex forward \
      --ignore-regex reset_parameters \
      --ignore-regex extra_repr \
      --ignore-regex MetaData \
      -vv \
      --color \
      ./physicsnemo/
    ```

3. `lint`
    *Pre-commit will check this for you!*
    Linters will perform static analysis to check the style, complexity, errors and more.
    For markdown files `markdownlint` is used, its suggested to use the vscode,
    neovim or sublime [extensions](https://github.com/DavidAnson/markdownlint#related).
    PhysicsNeMo uses [Ruff](https://docs.astral.sh/ruff/) for linting of various types.
    Currently we use flake8/pycodestyle (`E`), Pyflakes (`F`), flake8-bandit (`S`),
    isort (`I`), and performance 'PERF' rules with the isort rules being fixable.

4. `license`
    *Pre-commit will check this for you!*
    Checks for correct license headers of all files.
    To run this locally use `make license`.
    See the Licensing Information section above for details about the license header required.

5. `pytest`
    Checks if the test scripts from the `test` folder run and produce desired outputs. It
    is imperative that your changes don't break the existing tests. If your MR fails this
    test, you will have to review your changes and fix the issues.
    To run pytest locally you can simply run `pytest` inside the `test` folder.

    While writing these tests, we encourage you to make use of the
    [`@nfs_data_or_fail`](https://github.com/NVIDIA/physicsnemo/blob/main/test/pytest_utils.py#L92)
    and the [`@import_of_fail`](https://github.com/NVIDIA/physicsnemo/blob/main/test/pytest_utils.py#L25)
    decorators to appropriately skip your tests for developers and users not having your
    test specific datasets and dependencies respectively. The CI has these datasets and
    dependencies so your tests will get executed during CI.
    This mechanism helps us provide a better developer and user experience
    when working with the unit tests.

6. `doctest`
    Checks if the examples in the docstrings run and produce desired outputs.
    It is highly recommended that you provide simple examples of your functions/classes
    in the code's docstring itself.
    Keep these examples simple and also add the expected outputs.
    Refer [doctest](https://docs.python.org/3/library/doctest.html) for more information.
    If your MR fails this test, check your changes and the docstrings.
    To run doctest locally, you can simply run `pytest --doctest-modules` inside the
    `physicsnemo` folder.

7. `coverage`
    Checks if your code additions have sufficient coverage.
    Refer [coverage](https://coverage.readthedocs.io/en/6.5.0/index.html#) for more details.
    If your MR fails this test, this means that you have not added enough tests to the `test`
    folder for your module/functions.
    Add extensive test scripts to cover different
    branches and lines of your additions.
    Aim for more than 80% code coverage.
    To test coverage locally, run the `get_coverage.sh` script from the `test` folder and
    check the coverage of the module that you added/edited.
