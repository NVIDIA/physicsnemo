# Nightly Cache Contract

This document is the authoritative reference for how the
`Nightly Github UV Workflow`
([.github/workflows/github-nightly-uv.yml](workflows/github-nightly-uv.yml))
publishes caches and how downstream PR workflows consume them.  PR
gating relies on these contracts being honored on both sides; do not
weaken them without updating this document.

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
| Save semantics | nightly `testmon` job only, via delete-before-save; PR workflows restore but never save |

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

## Operational notes

- **Concurrency**: the nightly workflow declares
  `concurrency: nightly-github-uv` with `cancel-in-progress: false` so
  two overlapping runs cannot race on the static `-latest` uv cache key.
- **Save verification**: after `actions/cache/save@v4` writes the uv
  download cache slot, the workflow re-queries `gh cache list` to
  confirm the entry exists. `cache/save` silently no-ops on key
  collision; without verification a corrupted slot can persist for days.
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

## Bumping any of the baseline values

If you change the container image, CUDA version, Python version, uv
version, or extras tag, you must update both:

1. The matching `env:` value at the top of both
   [.github/workflows/github-nightly-uv.yml](workflows/github-nightly-uv.yml)
   and
   [.github/workflows/github-pr.yml](workflows/github-pr.yml).
2. The corresponding literals embedded in `UV_CACHE_KEY_PREFIX` and
   `JIT_CACHE_KEY_PREFIX` (GitHub Actions does not support env-to-env
   references within the same `env:` block, so these are kept in
   lockstep manually).

The first nightly run after a baseline bump will miss all caches, do a
full download/compilation, and republish under the new prefix.  Existing
PR workflows that pin to the old prefix will silently fall back to
cold-cache (slow but correct) until they are updated.
