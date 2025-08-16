#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/runberry/physicsnemo.git}"
REPO_DIR="${REPO_DIR:-$PWD/physicsnemo}"
BRANCH="${BRANCH:-exp/tp1}"

PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pipcache}"   # 加速 pip
PYTHON_BIN="${PYTHON_BIN:-python3}"               # 或 python

# 第一次才做 clone
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

# 避免以root執行的 safe.directory 警告
git config --global --add safe.directory "$REPO_DIR" 2>/dev/null || true

cd "$REPO_DIR"

# 取最新並切分支 
git fetch --prune origin

# 若本地已存在分支就切換 否則從遠端建立 沒有則新建
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git checkout "$BRANCH"
else
  if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
    git checkout -B "$BRANCH" "origin/$BRANCH"
  else
    git checkout -B "$BRANCH"
  fi
fi

# 避免產生merge commit
git pull --ff-only origin "$BRANCH" || true

# install requirements
export PIP_CACHE_DIR
cd dev
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.txt
cd ..

# make run
make run
