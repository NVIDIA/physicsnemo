#!/bin/bash

### Limit OpenBLAS threads to avoid hitting process limits and be a kind neighbor on shared cluster login nodes
export OPENBLAS_NUM_THREADS=1

uv run --no-sync mlflow ui --backend-store-uri sqlite:///output/mlflow.db --workers 1
