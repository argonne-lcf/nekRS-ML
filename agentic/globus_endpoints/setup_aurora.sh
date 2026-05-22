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
# Default venv lives inside the repo to avoid filling up $HOME (which is quota-
# limited on Aurora). The "_env" prefix is already ignored by .gitignore.
# Override with VENV_PATH if you want it elsewhere.
VENV_PATH="${VENV_PATH:-$REPO_ROOT/_env-agentic}"
ENDPOINT_NAME="${ENDPOINT_NAME:-nekrs-ml-aurora}"
FRAMEWORKS_MODULE="${FRAMEWORKS_MODULE:-frameworks}"

echo "[setup] REPO_ROOT      = $REPO_ROOT"
echo "[setup] VENV_PATH      = $VENV_PATH"
echo "[setup] ENDPOINT_NAME  = $ENDPOINT_NAME"
echo "[setup] FRAMEWORKS_MOD = $FRAMEWORKS_MODULE"
echo

# 1. Modules ----------------------------------------------------------------
# Aurora's Lmod modulefiles reference shell-specific variables like
# ZSH_EVAL_CONTEXT that aren't set under bash; with `set -u` active that
# triggers "unbound variable". Relax `-u` just for the module calls so the
# rest of the script keeps catching typos.
echo "[setup] Loading $FRAMEWORKS_MODULE module ..."
set +u
module restore >/dev/null 2>&1 || true
module load "$FRAMEWORKS_MODULE"
module list 2>&1 | sed 's/^/    /'
set -u
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
# Detect the Python MAJOR.MINOR provided by the frameworks module so we can
# tell the user exactly which version to match on their laptop. Done outside
# the heredoc so awk doesn't have to fight backslash-escaping rules.
ENDPOINT_PY_VERSION="$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)"
if [ -z "$ENDPOINT_PY_VERSION" ]; then
  ENDPOINT_PY_VERSION="<check with: python --version>"
fi

cat <<EOF
[setup] Done with bootstrap. The endpoint is configured but NOT started --
        you start it manually so you can complete the Globus auth flow
        and capture the UUID.

        Your FIRST start must be foreground or in tmux so you can see and
        complete the browser auth URL. After that the token is cached
        and you can use --detach.

  # ---- A) Foreground (use this for first-time auth) ----
  module load $FRAMEWORKS_MODULE
  source $VENV_PATH/bin/activate
  globus-compute-endpoint start $ENDPOINT_NAME
  # follow the printed URL, authenticate, Ctrl-C to stop

  # ---- B) tmux (foreground inside a session that survives logout) ----
  tmux new -s gc-endpoint
  # ...then the same three commands as (A); detach with Ctrl-b d

  # ---- C) --detach (after first-time auth is done) ----
  module load $FRAMEWORKS_MODULE
  source $VENV_PATH/bin/activate
  globus-compute-endpoint start $ENDPOINT_NAME --detach
  # endpoint daemonises; logs in ~/.globus_compute/$ENDPOINT_NAME/EndpointLogs/
  # stop later with: globus-compute-endpoint stop $ENDPOINT_NAME

  # ---- Grab the UUID after starting ----
  globus-compute-endpoint list   # line for $ENDPOINT_NAME shows the UUID

  # ---- On your laptop ----
  # IMPORTANT: your laptop venv MUST use the same Python MAJOR.MINOR as the
  # frameworks module here (currently $ENDPOINT_PY_VERSION). Globus Compute
  # serializes functions between machines and version mismatch breaks unpickling.
  #
  #   python$ENDPOINT_PY_VERSION -m venv _env-agentic
  #   source _env-agentic/bin/activate
  #   pip install -e ./agentic
  #   python -m agentic.client.setup \\
  #       --uuid       <paste UUID here> \\
  #       --repo-root  $REPO_ROOT \\
  #       --nekrs-home <e.g., /home/$USER/.local/nekrs>
EOF
