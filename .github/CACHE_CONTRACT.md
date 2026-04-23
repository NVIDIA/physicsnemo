# Nightly UV Cache Contract

This document is the authoritative reference for how the
`Nightly Github UV Workflow`
([.github/workflows/github-nightly-uv.yml](workflows/github-nightly-uv.yml))
publishes Python environment caches and how downstream PR workflows must
consume them. PR gating relies on these contracts being honored on both
the producer and consumer side; do not weaken them without updating this
document.

## Two caches, two contracts

The pipeline maintains two strictly disjoint caches with different
invalidation rules. Conflating them is the historical bug class this
design exists to forbid.

### A. uv download cache (`~/.cache/uv`)

| Property | Value |
|---|---|
| Key | `<UV_CACHE_KEY_PREFIX>-latest` |
| Prefix encodes | container image + Python version + uv version |
| Suffix | literal `latest` (mutable slot, refreshed via delete-before-save) |
| Contents | every wheel uv has ever downloaded for this baseline; additive across lockfile changes |
| Invalidates when | container image, CUDA version, Python version, or uv version changes (prefix change → new slot) |
| Does **not** invalidate on | `uv.lock` or `pyproject.toml` changes |
| Restore semantics | **fail-open**; missing cache only costs download time, never correctness |
| Save semantics | always save (delete the existing entry first, then save, then verify with `gh cache list`) |

The uv download cache is purely a speed optimisation. Anything that
correctness depends on must come from the venv cache.

### B. venv cache (`.venv`)

| Property | Value |
|---|---|
| Key | `<VENV_CACHE_KEY_PREFIX>-<lockhash>` |
| Prefix encodes | container image + Python version + uv version + extras tag (e.g. `cu12`) |
| Suffix | `hashFiles('uv.lock', 'pyproject.toml')`, computed once per job and propagated via `needs.<job>.outputs.lockhash` |
| Contents | the fully realized `.venv` produced by `uv sync --frozen --group dev --extra <tag>` against the committed lockfile |
| Invalidates when | any prefix component changes, or the lockfile hash changes |
| Restore semantics | **exact-match only, no `restore-keys` fallback** |
| Save semantics | standard `actions/cache/save`. Same lockhash → same content → save no-ops, which is correct |

The extras tag (`cu12`, `cu13`, ...) is part of the prefix so cu12 and
cu13 builds never overwrite each other.

The lockhash includes both `uv.lock` and `pyproject.toml`. If a PR
touches `pyproject.toml` without regenerating `uv.lock`, the build fails
loudly during `uv sync --frozen` rather than silently producing a
mismatched venv.

## PR consumer contracts

PR workflows that gate on the nightly venv MUST implement one of two
exhaustive paths.

### Contract 1 — PR does not touch `pyproject.toml` or `uv.lock`

The PR's lockhash equals the hash that the most recent successful nightly
saved under.

```yaml
- name: Restore uv download cache (fail-open)
  uses: actions/cache/restore@v4
  with:
    path: ~/.cache/uv
    key: <UV_CACHE_KEY_PREFIX>-latest
    # NO fail-on-cache-miss. Missing uv cache is acceptable here.

- name: Restore venv cache (exact match, MUST hit)
  uses: actions/cache/restore@v4
  with:
    path: .venv
    key: <VENV_CACHE_KEY_PREFIX>-${{ hashFiles('uv.lock', 'pyproject.toml') }}
    fail-on-cache-miss: true
    # NO restore-keys. A partial match would silently degrade test
    # validity.

- name: Use the env, read-only
  env:
    UV_FROZEN: "1"
    UV_NO_SYNC: "1"
  run: |
    .venv/bin/python -c "import torch; print(torch.__version__)"
    uv run --no-sync python -m pytest ...
```

Guarantees:

- Either the venv is byte-identical to what nightly validated, or the
  job fails on cache miss.
- `UV_FROZEN=1` and `UV_NO_SYNC=1` (plus the `--no-sync` flags) make it
  impossible for any subsequent `uv run` to mutate the restored venv.
- `physicsnemo` itself is installed editable, so the PR's source code
  changes are picked up without rebuilding the venv.

### Contract 2 — PR updates `pyproject.toml` and/or `uv.lock`

The PR's lockhash is new; the venv cache misses by design.

```yaml
- name: Restore uv download cache (fail-open)
  uses: actions/cache/restore@v4
  with:
    path: ~/.cache/uv
    key: <UV_CACHE_KEY_PREFIX>-latest

- name: Restore venv cache (will miss; that is fine)
  id: venv-restore
  uses: actions/cache/restore@v4
  with:
    path: .venv
    key: <VENV_CACHE_KEY_PREFIX>-${{ hashFiles('uv.lock', 'pyproject.toml') }}
    # No fail-on-cache-miss; we expect to miss on lock-change PRs.

- name: Clean-build venv
  if: steps.venv-restore.outputs.cache-hit != 'true'
  env:
    UV_LINK_MODE: copy
    UV_FROZEN: "1"
    UV_NO_SYNC: "1"
  run: |
    rm -rf .venv
    uv sync --frozen --group dev --extra cu12
    # Optional: assert the lockfile was not mutated, e.g. with sha256sum
    # before/after. setup-uv-env does this automatically.
```

Guarantees:

- `rm -rf .venv` ensures no leftover state from a previous PR push or a
  partial restore-keys hit. (Restore-keys is forbidden anyway, but this
  is cheap insurance.)
- `--frozen` + `UV_FROZEN=1` ensure the resolver cannot rewrite
  `uv.lock` in CI. If the PR shipped a stale lock, the job fails fast
  with a clear error rather than papering over the mismatch.
- The uv download cache (restored fail-open) supplies most wheels, so
  the rebuild is fast even though the venv itself is fresh.

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

## Bumping any of the baseline values

If you change the container image, CUDA version, Python version, uv
version, or extras tag, you must update both:

1. The matching `env:` value at the top of
   [.github/workflows/github-nightly-uv.yml](workflows/github-nightly-uv.yml).
2. The corresponding literal embedded in `UV_CACHE_KEY_PREFIX` and
   `VENV_CACHE_KEY_PREFIX` (GitHub Actions does not support env-to-env
   references within the same `env:` block, so these are kept in
   lockstep manually).

The first nightly run after a baseline bump will miss both caches, do a
full rebuild, and republish under the new prefix. Existing PR workflows
that pin to the old prefix will hard-fail until they are updated, which
is the desired behaviour.
