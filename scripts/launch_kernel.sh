#!/usr/bin/env bash
# Jupyter kernel entrypoint for Cursor/VS Code (fast, logged startup).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${TMPDIR:-/tmp}"
exec "$ROOT/.venv/bin/python" -u -m ipykernel_launcher "$@"
