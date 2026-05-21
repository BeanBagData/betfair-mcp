#!/usr/bin/env bash
# One-shot launcher for daily Betfair paper-bet automation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../paper_autobet.py" ]]; then
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -f "$SCRIPT_DIR/../app/paper_autobet.py" ]]; then
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/../app" && pwd)"
else
  echo "Could not locate paper_autobet.py relative to $SCRIPT_DIR" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p logs

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
elif [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ $# -eq 0 ]]; then
  set -- --run-due
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]] && [[ -x "/Library/Frameworks/Python.framework/Versions/Current/bin/python3" ]]; then
  PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/Current/bin/python3"
fi
if [[ -z "$PYTHON_BIN" ]] && [[ -x "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3" ]]; then
  PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
fi
if [[ -z "$PYTHON_BIN" ]] && command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
fi
if [[ -z "$PYTHON_BIN" ]] && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "No Python interpreter found" >&2
  exit 127
fi

exec "$PYTHON_BIN" "paper_autobet.py" "$@"
