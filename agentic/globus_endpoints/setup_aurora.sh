#!/bin/bash
#
# One-time bootstrap of the nekRS-ML Globus Compute endpoint on an Aurora
# login node. Run this from any Aurora login node, as your user, after
# cloning nekRS-ML somewhere persistent (e.g. under $HOME or a project
# directory on /lus).
#
# What this script does:
#   1. Loads the Aurora frameworks module (provides Python 3)
#   2. Creates a dedicated venv at $VENV_PATH (default: $HOME/.local/nekrs-ml-agentic)
#   3. Installs globus-compute-endpoint and the local agentic package
#   4. Initialises an endpoint named ENDPOINT_NAME (default: nekrs-ml-aurora)
#      and copies our LocalProvider config into place
#   5. Prints next steps (auth, start, capture UUID)
#
# What it does NOT do:
#   - Start the endpoint (you do that manually so you can see the auth URL)
#   - Set up a systemd-user unit or cron-restart (mentioned at the end)
#
# Re-running this script is safe: it skips steps that are already done.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="${VENV_PATH:-$HOME/.local/nekrs-ml-agentic}"
ENDPOINT_NAME="${ENDPOINT_NAME:-nekrs-ml-aurora}"
FRAMEWORKS_MODULE="${FRAMEWORKS_MODULE:-frameworks}"

echo "[setup] REPO_ROOT      = $REPO_ROOT"
echo "[setup] VENV_PATH      = $VENV_PATH"
echo "[setup] ENDPOINT_NAME  = $ENDPOINT_NAME"
echo "[setup] FRAMEWORKS_MOD = $FRAMEWORKS_MODULE"
echo

# 1. Modules ----------------------------------------------------------------
echo "[setup] Loading $FRAMEWORKS_MODULE module ..."
module restore >/dev/null 2>&1 || true
module load "$FRAMEWORKS_MODULE"
module list 2>&1 | sed 's/^/    /'
echo

# 2. venv -------------------------------------------------------------------
if [ -d "$VENV_PATH" ]; then
  echo "[setup] venv already exists at $VENV_PATH, reusing"
else
  echo "[setup] Creating venv at $VENV_PATH ..."
  python -m venv --system-site-packages "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip

# 3. Install ----------------------------------------------------------------
echo "[setup] Installing globus-compute-endpoint and the agentic package ..."
pip install "globus-compute-endpoint>=2.27"
pip install -e "$REPO_ROOT/agentic"
echo

# 4. Endpoint config --------------------------------------------------------
CONFIG_DIR="$HOME/.globus_compute/$ENDPOINT_NAME"
if [ -d "$CONFIG_DIR" ]; then
  echo "[setup] Endpoint $ENDPOINT_NAME already configured at $CONFIG_DIR"
  echo "        (delete $CONFIG_DIR to re-initialise)"
else
  echo "[setup] Initialising endpoint $ENDPOINT_NAME ..."
  globus-compute-endpoint configure "$ENDPOINT_NAME"
fi

echo "[setup] Installing LocalProvider config (preserving existing as .bak) ..."
TARGET_CONFIG="$CONFIG_DIR/config.yaml"
SOURCE_CONFIG="$REPO_ROOT/agentic/globus_endpoints/aurora_config.yaml"
if [ -f "$TARGET_CONFIG" ] && ! cmp -s "$TARGET_CONFIG" "$SOURCE_CONFIG"; then
  cp "$TARGET_CONFIG" "$TARGET_CONFIG.bak.$(date +%s)"
fi
cp "$SOURCE_CONFIG" "$TARGET_CONFIG"
echo

# 5. Next steps -------------------------------------------------------------
cat <<EOF
[setup] Done with bootstrap. Next steps (run by hand so you can see the
        Globus auth URL and capture the UUID):

  source $VENV_PATH/bin/activate
  module load $FRAMEWORKS_MODULE
  globus-compute-endpoint start $ENDPOINT_NAME
  # follow the printed URL, authenticate with your ALCF identity
  globus-compute-endpoint list

  # Take the UUID for $ENDPOINT_NAME from that output. Then on your laptop:
  #
  #   pip install -e ./agentic
  #   python -m agentic.client.setup \\
  #       --uuid <paste UUID here> \\
  #       --repo-root $REPO_ROOT
  #
  # That writes endpoints.json, registers all functions, and pings the
  # endpoint as a smoke test. No JSON hand-editing required.

To keep the endpoint up across login-node reboots, wrap the start command
in a long-running session (tmux/screen) or write a systemd --user unit.
This script does neither so you can verify it works interactively first.
EOF
