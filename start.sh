#!/usr/bin/env bash
set -euo pipefail

# Determine script directory so we can run from anywhere.
ROOT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

echo "==> Using project dir: $ROOT_DIR"

if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtualenv"
  python3 -m venv "$VENV_DIR"
fi

echo "==> Activating virtualenv"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "==> Installing dependencies"
pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements.txt"

echo "==> Launching app"
exec python "$ROOT_DIR/app.py"
