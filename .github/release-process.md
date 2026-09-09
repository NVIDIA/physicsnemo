# PhysicsNeMo RC release process

This runbook describes the maintainer workflow for creating a release-candidate
branch and returning its commits to `main` after the release. GitHub Actions
automates branch creation, version changes, and pull-request preparation. A
maintainer still performs the final rebase, resolves conflicts, reviews the
result, and selects **Rebase and merge**.

## Invariants

- Run both release workflows from the repository default branch, `main`.
- Never rebase, delete, or force-push the preserved RC branch.
- Perform history rewriting only on the disposable `-rc-rebaseN` branch.
- Use `git push --force-with-lease` after rebasing.
- Do not merge `main` into the disposable branch.
- Supply the next development version explicitly. Do not infer whether the
  next release is a major, minor, or patch release.
- Merge the final pull request with **Rebase and merge**, not squash.

## Required access

The maintainer running the workflows needs write access to the repository.

The workflows use the built-in `GITHUB_TOKEN` by default. The repository must
allow GitHub Actions to create pull requests. If organization policy prevents
that, configure a `RELEASE_AUTOMATION_TOKEN` repository or environment secret
containing a fine-grained token with permission to write contents, issues, and
pull requests. The workflow can be adapted to mint a short-lived GitHub App
installation token if that is preferred for long-term use.

The maintainer performing the manual rebase needs Git, push access to the
repository, and a clean local checkout.

## Start code freeze

1. Open the repository on GitHub.
2. Select **Actions**.
3. Select **Release: Start RC**.
4. Select **Run workflow**.
5. Select `main` in the branch dropdown.
6. Enter:

   - **Final release version**: for example, `2.3.0`.
   - **RC branch**: leave blank to use `2.3.0-rc`.
   - **Release date**: for example, `2026-08-15`.
   - **Dry run**: select this to validate without writing to GitHub.

7. Select **Run workflow**.
8. Open the completed run and follow the branch and tracking-issue links in
   its summary.

The workflow quietly verifies the existing development version before changing
it. For a `2.3.0` release, the expected package version on `main` is
`2.3.0a0`. This is an internal guardrail and is not another maintainer input.

For a new release, the workflow:

1. Captures the exact current `main` SHA.
2. Creates `2.3.0-rc` locally from that SHA.
3. Updates `physicsnemo/__init__.py` to `2.3.0`.
4. Finalizes the `2.3.0` changelog heading and release date.
5. Commits and pushes the RC branch.
6. Creates a release-tracking issue with the relevant SHAs and checklist.

If the requested branch already exists with the same version and changelog
date, rerunning the workflow treats it as the existing release state and
reuses the tracking issue. It never overwrites an inconsistent branch.

## Work during the RC period

Continue using the existing review and CI process. In the first version of this
automation, no additional GitHub policy determines whether an RC pull request
is a bug fix. Release maintainers retain that decision.

Do not manually update or rebase the RC branch as part of merge-back
preparation. The next workflow creates a separate disposable branch.

## Prepare the merge-back pull request

After the release is complete:

1. Open **Actions**.
2. Select **Release: Prepare RC Merge-back**.
3. Select **Run workflow**.
4. Select `main` in the branch dropdown.
5. Enter:

   - **RC branch**: for example, `2.3.0-rc`.
   - **Next development version**: for example, `2.4.0a0`.
   - **Dry run**: select this to validate without writing to GitHub.

6. Select **Run workflow**.
7. Follow the draft pull-request link in the workflow summary.

The workflow:

1. Reads the final release version from the RC branch.
2. Records the current RC and `main` SHAs.
3. Creates the next available branch, such as `2.3.0-rc-rebase1`.
4. Changes the package version to `2.4.0a0`.
5. Adds an empty `2.4.0` section to the top of `CHANGELOG.md`.
6. Pushes the disposable branch.
7. Opens a draft pull request against `main`.
8. Links the pull request from the release-tracking issue.

The workflow does not rebase either branch.

## Rebase the disposable branch

Use the exact branch name shown in the draft pull request. For example:

```bash
git fetch origin
git switch -c 2.3.0-rc-rebase1 \
  --track origin/2.3.0-rc-rebase1
git rebase origin/main
```

When Git stops for a conflict:

1. Run `git status`.
2. Resolve each listed file.
3. Stage the resolution:

   ```bash
   git add <resolved-files>
   ```

4. Continue:

   ```bash
   git rebase --continue
   ```

5. Repeat until the rebase completes.

For `CHANGELOG.md`:

- Keep released work under the final release section, such as `2.3.0`.
- Keep work added to `main` during the freeze under the next section, such as
  `2.4.0`.
- Remove duplicate entries created by conflict resolution.
- Preserve the requested next-development section at the top.

Pull request
[NVIDIA/physicsnemo#1673](https://github.com/NVIDIA/physicsnemo/pull/1673)
is an example of this reconciliation.

If the resolution is unclear, stop safely:

```bash
git rebase --abort
```

The preserved RC branch is unaffected. Ask another maintainer to review the
conflict before starting again.

After a successful rebase, update only the disposable branch:

```bash
git push --force-with-lease origin \
  HEAD:2.3.0-rc-rebase1
```

## Review and merge

Before merging:

1. Confirm the disposable branch was rebased onto the latest `main`.
2. Review the commit history and changelog reconciliation.
3. Confirm the package version is the requested next development version.
4. Confirm the original RC branch remains unchanged.
5. Run and pass the normal GitHub and Blossom CI.
6. Mark the draft pull request ready for review.
7. Obtain the normal approvals.
8. Select **Rebase and merge**.

GitHub rewrites commit SHAs during rebase-and-merge, but preserves the
individual non-empty commits.

## Post-merge verification

After merging:

1. Confirm `main` reports the requested next development version.
2. Confirm the next changelog section is first.
3. Confirm the final release section is still present.
4. Confirm the original RC branch still exists at the SHA recorded in the
   tracking issue and pull request.
5. Delete the disposable merge-back branch if desired.
6. Close the release-tracking issue.

## Recovery notes

- If **Release: Start RC** fails before the push, fix the reported input or
  repository state and rerun it.
- If it fails after the branch push, rerun it with the same inputs. A matching
  branch is reused; a mismatching branch is never overwritten.
- If merge-back preparation finds an existing open pull request for the same
  RC branch and next version, it returns that pull request rather than opening
  another.
- If `main` advances after the manual rebase, rebase the disposable branch
  again and push with `--force-with-lease`.
- If a rebase becomes confusing, use `git rebase --abort`. Do not attempt the
  operation on the preserved RC branch.
