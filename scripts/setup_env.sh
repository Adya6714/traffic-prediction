#!/usr/bin/env bash
# Recreate .venv (Homebrew Python 3.11) and register the Jupyter kernel.
# Do NOT use `python -m venv` under pyenv here — it can build a broken .venv
# (missing packages, wrong 3.11.x label in Cursor). Always run this script.
set -euo pipefail
cd "$(dirname "$0")/.."

PY311="${PY311:-/opt/homebrew/opt/python@3.11/bin/python3.11}"
if [[ ! -x "$PY311" ]]; then
  echo "Python 3.11 not found at $PY311"
  echo "Install: brew install python@3.11"
  exit 1
fi

echo "Creating .venv with $("$PY311" --version)"
rm -rf .venv
"$PY311" -m venv --copies .venv
.venv/bin/pip install --upgrade pip wheel
.venv/bin/pip install -r requirements.txt

mkdir -p .venv/etc/ipython
cat > .venv/etc/ipython/ipython_kernel_config.py <<'EOF'
import os
os.environ.setdefault("MPLBACKEND", "Agg")
EOF

.venv/bin/python -m ipykernel install --sys-prefix --name traffic-prediction --display-name "Traffic Prediction (.venv)"
.venv/bin/python -m ipykernel install --user --name traffic-prediction --display-name "Traffic Prediction (.venv)"

# Faster kernel startup (no debugpy / frozen_modules flags ipykernel adds by default).
write_kernel_json() {
  local dest="$1"
  mkdir -p "$(dirname "$dest")"
  cat > "$dest" <<EOF
{
  "argv": [
    "$(pwd)/scripts/launch_kernel.sh",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Traffic Prediction (.venv)",
  "language": "python",
  "env": {
    "MPLBACKEND": "Agg",
    "PYTHONUNBUFFERED": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TMPDIR": "/tmp"
  },
  "metadata": { "debugger": false }
}
EOF
}
chmod +x scripts/launch_kernel.sh
write_kernel_json ".venv/share/jupyter/kernels/traffic-prediction/kernel.json"
write_kernel_json "$HOME/Library/Jupyter/kernels/traffic-prediction/kernel.json"
rm -rf ".venv/share/jupyter/kernels/python3"

echo ""
echo "Done. Interpreter: $(pwd)/.venv/bin/python"
echo "In Cursor: Reload Window, then select that interpreter or kernel 'Traffic Prediction (.venv)'."
