#!/usr/bin/env bash
# Install the daily paper-bet LaunchAgent for the current macOS user.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LABEL="com.nickjordan.betfair-paper-autobet"
SOURCE_PLIST="$PROJECT_ROOT/launchd/$LABEL.plist"
DEST_DIR="$HOME/Library/LaunchAgents"
DEST_PLIST="$DEST_DIR/$LABEL.plist"
DOMAIN="gui/$(id -u)"
RUNTIME_ROOT="$HOME/Library/Application Support/BetfairMCP"
RUNTIME_APP="$RUNTIME_ROOT/app"
RUNTIME_BIN="$RUNTIME_ROOT/bin"
RUNTIME_STATE="$RUNTIME_ROOT/state"
RUNTIME_LOGS="$RUNTIME_ROOT/logs"

update_env_file() {
  local env_file="$1"
  mkdir -p "$(dirname "$env_file")"
  touch "$env_file"

  local tmp_file
  tmp_file="$(mktemp)"
  grep -Ev '^(PAPER_BET_LOG_PATH|BETTING_HISTORY_PATH|PAPER_AUTOBET_STATE_PATH|ORDERS_CSV_PATH)=' "$env_file" > "$tmp_file" || true
  mv "$tmp_file" "$env_file"

  {
    printf '\n# Shared state used by the daily LaunchAgent and MCP tools.\n'
    printf 'PAPER_BET_LOG_PATH=%q\n' "$RUNTIME_STATE/bet_log.json"
    printf 'BETTING_HISTORY_PATH=%q\n' "$RUNTIME_STATE/betting_history.json"
    printf 'PAPER_AUTOBET_STATE_PATH=%q\n' "$RUNTIME_STATE/paper_autobet_state.json"
    printf 'ORDERS_CSV_PATH=%q\n' "$RUNTIME_STATE/orders_agent.csv"
  } >> "$env_file"
}

mkdir -p "$DEST_DIR" "$RUNTIME_APP" "$RUNTIME_BIN" "$RUNTIME_STATE" "$RUNTIME_LOGS"
chmod +x "$PROJECT_ROOT/bin/paper-autobet-once.sh"
chmod +x "$PROJECT_ROOT/bin/install-paper-autobet-launchd.sh"

rsync -a --delete \
  --exclude '.git/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude 'logs/' \
  --exclude '*.log' \
  --exclude 'bet_log.json' \
  --exclude 'betting_history.json' \
  --exclude 'orders_agent.csv' \
  --exclude 'paper_autobet_state.json' \
  "$PROJECT_ROOT/" "$RUNTIME_APP/"
cp "$PROJECT_ROOT/bin/paper-autobet-once.sh" "$RUNTIME_BIN/paper-autobet-once.sh"
chmod +x "$RUNTIME_BIN/paper-autobet-once.sh"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  update_env_file "$PROJECT_ROOT/.env"
fi
if [[ -f "$RUNTIME_APP/.env" ]]; then
  update_env_file "$RUNTIME_APP/.env"
fi

cp "$SOURCE_PLIST" "$DEST_PLIST"

launchctl bootout "$DOMAIN" "$DEST_PLIST" >/dev/null 2>&1 || true
launchctl bootstrap "$DOMAIN" "$DEST_PLIST"
launchctl enable "$DOMAIN/$LABEL"
launchctl kickstart -k "$DOMAIN/$LABEL"

launchctl print "$DOMAIN/$LABEL"
