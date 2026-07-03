# Nightly Cache Contract

This document is the authoritative reference for how the
`Nightly Github UV Workflow`
([.github/workflows/github-nightly-uv.yml](workflows/github-nightly-uv.yml))
and `Multi-GPU Github CI`
([.github/workflows/github-multigpu.yml](workflows/github-multigpu.yml))
publish caches and how downstream PR workflows consume them.  PR gating
relies on these contracts being honored on both sides; do not weaken
them without updating this document.

## Caches

### uv download cache (`~/.cache/uv`)

| Property | Value |
|---|---|
| Key | `<UV_CACHE_KEY_PREFIX>-latest` |
| Prefix encodes | container image + Python version + uv version |
| Suffix | literal `latest` (mutable slot, refreshed via delete-before-save) |
| Contents | every wheel uv has ever downloaded for this baseline; additive across lockfile changes |
| Invalidates when | container image, CUDA version, Python version, or uv version changes (prefix change → new slot) |
| Does **not** invalidate on | `uv.lock` or `pyproject.toml` changes |
| Restore semantics | **fail-open**; missing cache only costs download time, never correctness |
| Save semantics | nightly only, on cold-cache runs: delete the existing entry first, then save, then verify with `gh cache list` |

The uv download cache is purely a speed optimisation. Correctness comes
from three independent sources: a pinned CUDA container image, a pinned
`uv` version, and `uv sync --frozen` against the committed lockfile.

### JIT compilation cache (`/root/.cache/jit`)

| Property | Value |
|---|---|
| Key | `<JIT_CACHE_KEY_PREFIX>-latest` |
| Prefix encodes | container image + Python version |
| Suffix | literal `latest` (mutable slot, refreshed via delete-before-save) |
| Contents | compiled artifacts from warp (`warp/`), triton (`triton/`), and torch inductor (`inductor/`); additive across lockfile and kernel-source changes |
| Invalidates when | container image or Python version changes (prefix change → new slot) |
| Does **not** invalidate on | `uv.lock`, `pyproject.toml`, or kernel source changes (each compiler handles its own source-hash invalidation internally) |
| Restore semantics | **fail-open**; missing cache only costs compilation time, never correctness |
| Save semantics | nightly `testmon` job only, via the `replace-cache` action; PR workflows restore but never save |

The JIT compilation cache bundles all JIT compiler artifact directories
under a single umbrella path.  Each compiler writes to a subdirectory
controlled by its own environment variable:

- **Warp**: `WARP_CACHE_PATH` → `$JIT_CACHE_DIR/warp`
- **Triton**: `TRITON_CACHE_DIR` → `$JIT_CACHE_DIR/triton`
- **torch.compile / Inductor**: `TORCHINDUCTOR_CACHE_DIR` → `$JIT_CACHE_DIR/inductor`

The cache is additive and survives lockfile changes.  Correctness is
guaranteed by each compiler's built-in source-hash invalidation: Warp
hashes kernel source and recompiles changed kernels; Triton and
Inductor hash computational graphs.  Warp also namespaces its cache by
version, so upgrading warp simply adds new entries without invalidating
old ones.

To add a new JIT backend: create a subdirectory under `$JIT_CACHE_DIR`,
set the backend's cache-path env var in the test step, done.

### Testmon database cache (`.testmondata*`)

| Property | Value |
|---|---|
| Key | `<TESTMON_CACHE_KEY_PREFIX>-latest` |
| Prefix encodes | nightly identity (`testmon-nightly`) |
| Suffix | literal `latest` (mutable slot, refreshed via delete-before-save) |
| Contents | `.testmondata`, `.testmondata-shm`, `.testmondata-wal` -- testmon's per-test dependency graph and last-run signatures |
| Invalidates when | prefix is bumped (essentially never, by design) |
| Does **not** invalidate on | `uv.lock` or `pyproject.toml` changes -- testmon detects changed dependency hashes itself and re-runs only the affected tests |
| Restore semantics | **fail-open**; a miss only costs full-suite runtime, never correctness, and testmon handles stale DBs gracefully |
| Save semantics | nightly `testmon` job only, via the `replace-cache` action with `if: always()` so partial DBs from flaky runs still publish |

Historical note: the key was previously suffixed with
`hashFiles('uv.lock', 'pyproject.toml')`.  Because GitHub Actions
caches are immutable, two consecutive nightlies with an unchanged
lockfile (the common case) collided on the same key, and the second
save logged `Failed to save: Unable to reserve cache` only as a
*warning*.  The stale DB persisted for days, PRs restored it via the
prefix fallback, and testmon then invalidated everything because the
realized environment had drifted away from what the cached DB
recorded.  Switching to a `-latest` mutable slot via `replace-cache`
fixes the save bug, and the embedded verify step turns any future
silent save failure into a hard job failure.

#### Why a separate `ci-requirements.lock`

The cache fix above only addresses *saving* the DB; the DB is still
worthless to PRs if testmon's environment fingerprint at PR time
differs from the fingerprint stored at nightly time.  Testmon
computes that fingerprint from `importlib.metadata.distributions()`
over the active venv -- i.e. *everything* in
`.venv/lib/python3.12/site-packages`, not just the lockfile-pinned
closure.

`setup-uv-env` builds the venv in two layered steps:

1. `uv sync --frozen --group dev --extra <EXTRAS_TAG>` -- deterministic
   against `uv.lock`.
2. `uv pip install -r .github/ci-requirements.txt` -- adds CI-only
   test deps that have no home in pyproject extras (moto,
   scikit-image, numpy-stl, shapely, multi-storage-client, tensorstore,
   plus the PyG CUDA wheel swap).

Step 2 is *not* covered by `uv.lock`.  Several of the direct pins in
`ci-requirements.txt` are absent from `uv.lock` entirely, so their
transitive closure (`responses`, `xmltodict`, `jsonpath-ng`,
`lazy-loader`, `tifffile`, `pywavelets`, `imageio`,
`antlr4-python3-runtime`, ...) gets re-resolved fresh against PyPI on
every job.  A single transitive minor bump between the nightly that
publishes the testmon DB and the PR that consumes it changes the
sorted `name version` string testmon hashes, trips its
"packages installed have been changed" guard, and re-runs the entire
suite.

[`.github/ci-requirements.lock`](ci-requirements.lock) is a fully
pinned closure of `ci-requirements.txt` (direct + transitive), passed
to the layered install via `--constraint`.  It is generated by
[`.github/regen-ci-deps-lock.sh`](regen-ci-deps-lock.sh), and must be
regenerated and committed whenever a `==` pin in
`ci-requirements.txt` changes.

Two ways to run the regen:

1. **Standalone [`Regen CI-deps Lock`](workflows/regen-ci-deps-lock.yml)
   workflow** (workflow_dispatch).  Runs the regen on a CPU runner
   in ~5 min and uploads `.github/ci-requirements.lock` as an
   artifact for the maintainer to download and commit.  Requires
   the workflow file to be on the default branch (GitHub refuses
   workflow_dispatch on files that exist only on feature branches --
   both the UI dropdown and `gh workflow run --ref` enforce this).

2. **Local docker.**  See the header of
   [`.github/regen-ci-deps-lock.sh`](regen-ci-deps-lock.sh) for the
   `docker run …` invocation.  Useful when iterating on the script
   itself or when the standalone workflow is unavailable (e.g. a
   feature branch where the workflow file has not yet landed on
   the default branch).

### Coverage baseline cache (`.coverage*`)

| Property | Value |
|---|---|
| Key | `<COVERAGE_CACHE_KEY_PREFIX>-latest` |
| Prefix encodes | nightly identity (`coverage-nightly`) |
| Suffix | literal `latest` (mutable slot, refreshed via delete-before-save) |
| Contents | parallel-mode coverage shards (`.coverage.*`) produced by the nightly's full-suite pytest run, before `coverage combine` |
| Invalidates when | prefix is bumped |
| Does **not** invalidate on | `uv.lock` or `pyproject.toml` changes |
| Restore semantics | **fail-open**; PR coverage merges its own shards on top of the restored baseline |
| Save semantics | nightly `coverage` job only, via the `replace-cache` action |

Same immutable-key bug class as testmon; migrated to the `-latest`
slot for parity.

### Multi-GPU impact manifest caches

The multi-GPU workflow owns two independent, small JSON caches: one for
the dynamic stream and one for the static stream.

| Property | Value |
|---|---|
| Keys | `multigpu-impact-dynamic-v1-latest` and `multigpu-impact-static-v1-latest` |
| Prefix encodes | stream + manifest schema version |
| Suffix | literal `latest` (mutable slots, refreshed via delete-before-save) |
| Contents | source SHA, configured rank count, exact marker-test inventory, per-rank JUnit totals, and repository-relative files executed under the stream's dedicated parallel coverage run |
| Invalidates when | the manifest schema changes (version bump) |
| Restore semantics | exact-key hit after clearing the checked-out restore directory; **fail-open** on a miss, malformed schema, marker-inventory drift, unavailable baseline history, or age over 72 hours |
| Save semantics | successful scheduled runs only, plus explicitly approved default-branch dispatches; PR GPU jobs never write these slots |

These manifests answer a narrower question than the canonical coverage
baseline: whether a PR changed a file exercised by either multi-GPU
stream.  They are never combined into the normal coverage report.  The
nightly runs collect process/rank-safe parallel coverage with
[`test/coverage.multigpu.rc`](../test/coverage.multigpu.rc), then publish
only a validated JSON manifest.  Testmon is intentionally not used for
this purpose: the normal testmon baseline skips multi-GPU tests, dynamic
tests execute in subprocesses, and static tests execute from multiple
torchrun ranks.  Manifest publication requires exact JUnit rank files and
coverage from every configured static rank.  Known rank/topology skips are
allowlisted; missing dependencies, unavailable CUDA, and other unexpected
skips prevent publication rather than creating a partial dependency picture.

The PR selector also treats dependency files, shared CI setup actions,
the global pytest harness, new marker-bearing tests, newly added
production modules, Python helpers below marker-test directories,
non-Python package/test resources, and known distributed source prefixes
as conservative triggers.  API failures,
truncated file lists, stale manifests, and selector failures all run the
affected stream rather than silently skipping it.

The selector reads PR metadata before and after the paginated files API
request.  The PR head must remain stable and equal the mirrored checkout.
For a manifest older than the PR base, the selector verifies ancestry and
inspects the complete local manifest-to-base Git diff.  Relevant baseline
changes run the affected stream; unrelated drift (for example, README-only
changes) can still skip it.  Missing or shallow history fails open.  This
prevents a source push or an advanced default branch from turning an old
dependency picture into a false skip without making every post-nightly PR
consume both GPU pools.

## Reusable building blocks

### `replace-cache` action ([.github/actions/replace-cache/action.yml](actions/replace-cache/action.yml))

All mutable-slot caches above (uv, JIT, testmon, coverage, and the two
multi-GPU impact manifests) share the same delete-before-save recipe:
GitHub Actions cache slots are
immutable, so refreshing a `-latest` key requires deleting the
existing entry, calling `actions/cache/save`, and (because the save
silently no-ops on key collision) re-querying `gh cache list` to
confirm the slot now exists.  The `replace-cache` composite action
encapsulates that recipe:

```yaml
- name: Replace <some> cache
  if: <caller-supplied gate>
  uses: ./.github/actions/replace-cache
  with:
    path: <one or more paths>
    key:  <foo>-latest
    description: <human-readable label>
    github-token: ${{ secrets.GITHUB_TOKEN }}
```

The verify step is on by default (`verify: "true"`).  Disable only
when verification is genuinely undesirable; the default exists
because silent save failures are how stale slots persist for days.

## Why no `.venv` cache

A previous iteration of this pipeline also cached the realized `.venv`
keyed on the lockfile hash, with a "fail-on-cache-miss" exact-match
contract for PR consumers. It was dropped because:

- The pinned container + pinned uv + frozen lockfile already make `uv
  sync` deterministic; caching its output added a second correctness
  boundary no stronger than the first.
- The venv cache was responsible for most of the pipeline's complexity:
  two cache contracts, cross-job lockhash plumbing, fail-on-cache-miss
  restores, and a Contract 1 / Contract 2 branch at every consumer site.
- The cached `.venv` and the uv download cache together were pushing
  against GitHub Actions' 10 GB per-repo limit and would have needed
  separate slots per extras tag (cu12, cu13, ...), making eviction
  thrash likely.

Each job now does the same thing: restore the uv download cache
fail-open, then `uv sync --frozen --group dev --extra <tag>`. The sync
is fast because the warm uv cache already has every wheel locally.

## PR consumer contract

```yaml
- name: Setup uv environment from cache
  uses: ./.github/actions/setup-uv-env
  with:
    uv-cache-key-prefix: ${{ env.UV_CACHE_KEY_PREFIX }}
    uv-cache-key-suffix: "latest"
    extras: ${{ env.EXTRAS_TAG }}

- name: Use the env, read-only
  env:
    UV_FROZEN: "1"
    UV_NO_SYNC: "1"
  run: |
    .venv/bin/python -c "import torch; print(torch.__version__)"
    uv run --no-sync python -m pytest ...
```

Guarantees:

- `.venv` is always rebuilt from the committed lockfile; there is no
  "partial match" failure mode.
- If the PR touches `pyproject.toml` without regenerating `uv.lock`,
  `uv sync --frozen` fails loudly rather than silently producing a
  mismatched venv.
- `UV_FROZEN=1` and `UV_NO_SYNC=1` (plus `uv run --no-sync`) make it
  impossible for a downstream step to mutate the built venv.
- `physicsnemo` itself is installed editable, so PR source changes are
  picked up without rebuilding the venv.

The multi-GPU workflow follows the same environment contract and is
restore-only for the uv and JIT caches.  Its CPU selector separately
restores the two impact manifests.  Only successful nightly publisher
jobs can replace those manifest slots.

## Operational notes

- **Concurrency**: the nightly workflow declares
  `concurrency: nightly-github-uv` with `cancel-in-progress: false` so
  two overlapping runs cannot race on the static `-latest` uv cache key.
- **Multi-GPU concurrency**: `github-multigpu.yml` serializes runs per
  ref. Superseded `pull-request/*` pushes cancel in progress, while
  scheduled/default-branch publisher runs are never canceled.
- **Runner inventory**: both streams default to the confirmed
  `linux-amd64-gpu-h100-latest-2` pool.  Some static tests intentionally
  skip configurations that require four ranks.  Set both
  `MULTIGPU_STATIC_RUNNER` and `MULTIGPU_STATIC_NPROC` repository
  variables together when a four-GPU pool is available; the manifest
  records its rank count and JUnit skip total.  Jobs require visible GPU
  count to equal `NPROC`, and the selector rejects a cached manifest whose
  rank count or runner profile no longer matches the repository variables.
  Manifest `complete` is scoped to that recorded runner/rank profile; the
  default two-rank stream is not a claim that four-rank-only cases executed.
- **Operator overrides**: `ci:multi-gpu`, `ci:multi-gpu-dynamic`, and
  `ci:multi-gpu-static` force selection on the next mirror sync or rerun.
  Applying a label alone does not emit a push event; use the manual
  dispatch for an immediate run.
- **Save verification**: every mutable-slot save (uv download, JIT,
  testmon, coverage, multi-GPU impact manifests) goes through the
  `replace-cache` action, which re-queries `gh cache list` after
  `actions/cache/save` and fails the job if the slot is not visible.
  `cache/save` silently no-ops on key collision and only logs a warning
  on reservation failure; without verification a corrupted slot can
  persist for days.
- **Lockfile-mutation guard**: [.github/actions/setup-uv-env/action.yml](actions/setup-uv-env/action.yml)
  snapshots `sha256(uv.lock)` and `sha256(pyproject.toml)` before any uv
  command runs and compares them again at the end. Any drift (caused by
  a forgotten `--frozen`, a dropped `--extra`, etc.) trips this guard
  and fails the job with a pointed error message.
- **uv version pin**: `bootstrap-cudnn-ci` installs a pinned uv version
  via `https://astral.sh/uv/<version>/install.sh` and asserts the
  installed binary matches. The pin is what allows the uv version to
  appear in the cache key prefix without surprise invalidations.
- **PR workflows never save the uv cache.** Only the nightly mutates
  the `-latest` slot; PRs restore fail-open and any fresh wheels they
  download are simply not preserved until the next nightly.
- **PR workflows never save impact manifests.** GPU test jobs have
  read-only Actions permissions. Dedicated CPU publisher jobs receive
  `actions: write`, and only for successful scheduled or explicitly
  approved default-branch runs.

## Bumping any of the baseline values

If you change the container image, CUDA version, Python version, uv
version, or extras tag, you must update all three workflows:

1. The matching `env:` value at the top of
   [.github/workflows/github-nightly-uv.yml](workflows/github-nightly-uv.yml)
   [.github/workflows/github-pr.yml](workflows/github-pr.yml), and
   [.github/workflows/github-multigpu.yml](workflows/github-multigpu.yml).
2. The corresponding literals embedded in `UV_CACHE_KEY_PREFIX` and
   `JIT_CACHE_KEY_PREFIX` (GitHub Actions does not support env-to-env
   references within the same `env:` block, so these are kept in
   lockstep manually).

The first nightly run after a baseline bump will miss all caches, do a
full download/compilation, and republish under the new prefix.  Existing
PR workflows that pin to the old prefix will silently fall back to
cold-cache (slow but correct) until they are updated.
