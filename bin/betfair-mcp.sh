#!/usr/bin/env bash
# Launcher for Betfair MCP server — resolves project root relative to this script
# so Codex CLI can call it from any working directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate venv if present.
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
elif [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

# Load .env if present.
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

# Prefer `python` (set by an active venv) but fall back to `python3` on systems
# without a `python` symlink (e.g. bare macOS).
if command -v python >/dev/null 2>&1; then
  exec python "mcp_server.py"
else
  exec python3 "mcp_server.py"
fi
