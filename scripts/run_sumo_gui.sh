#!/usr/bin/env bash#!/usr/bin/env bash

# Helper to launch SUMO GUI on macOS (pkg builds often require XQuartz/X11).set -euo pipefail

# Usage:

#   ./scripts/run_sumo_gui.sh [path/to/config.sumocfg]# Run SUMO GUI for a given .sumocfg file on macOS.

# This SUMO build is linked against X11, so it requires XQuartz.

set -euo pipefail#

# Usage:

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"#   ./scripts/run_sumo_gui.sh [path/to/simulation.sumocfg]

DEFAULT_CFG="$ROOT_DIR/sumo/cfg/before_simulation.sumocfg"#

CFG="${1:-$DEFAULT_CFG}"# Defaults:

#   sumo/cfg/before_simulation.sumocfg

SUMO_BIN_DEFAULT="/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/bin"

if [[ -d "$SUMO_BIN_DEFAULT" ]]; thenSCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

  export SUMO_HOME="${SUMO_HOME:-$(cd "$SUMO_BIN_DEFAULT/.." && pwd)}"PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

  export PATH="$SUMO_BIN_DEFAULT:$PATH"

fiCFG_PATH="${1:-${PROJECT_ROOT}/sumo/cfg/before_simulation.sumocfg}"



if ! command -v sumo-gui >/dev/null 2>&1; then# SUMO framework install location from the macOS .pkg installer.

  echo "sumo-gui not found on PATH." >&2SUMO_FRAMEWORK="/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO"

  echo "If you installed SUMO via .pkg, ensure it's under:" >&2SUMO_GUI_BIN="${SUMO_FRAMEWORK}/bin/sumo-gui"

  echo "  $SUMO_BIN_DEFAULT" >&2SUMO_BIN="${SUMO_FRAMEWORK}/bin/sumo"

  exit 127

filog()  { printf "[sumo-gui-helper] %s\n" "$*"; }

warn() { printf "[sumo-gui-helper] WARN: %s\n" "$*" >&2; }

if [[ ! -f "$CFG" ]]; thenerr()  { printf "[sumo-gui-helper] ERROR: %s\n" "$*" >&2; }

  echo "SUMO config not found: $CFG" >&2

  echo "Generate it first (pipeline SUMO step) or pass an existing .sumocfg path." >&2die() {

  exit 2  err "$*"

fi  exit 1

}

# Try to ensure XQuartz is running (best effort).

if [[ "$(uname -s)" == "Darwin" ]]; thenif [[ ! -f "${CFG_PATH}" ]]; then

  open -g -a XQuartz || true  die "Config file not found: ${CFG_PATH}"

fifi



# Default DISPLAY for XQuartzif [[ ! -x "${SUMO_GUI_BIN}" ]]; then

export DISPLAY="${DISPLAY:-:0}"  die "SUMO GUI binary not found/executable at: ${SUMO_GUI_BIN}\nInstall SUMO via the .pkg, or update this script's SUMO_FRAMEWORK path."

fi

echo "Launching SUMO GUI…"

echo "  Config: $CFG"# Export SUMO_HOME so SUMO tools and XML validation behave better.

echo "  DISPLAY=$DISPLAY"export SUMO_HOME="${SUMO_FRAMEWORK}"



set +e# Ensure XQuartz is installed and running.

sumo-gui -c "$CFG"# SUMO-GUI uses X11 (libX11), so it needs an X server on macOS.

RC=$?if [[ ! -d "/Applications/Utilities/XQuartz.app" ]]; then

set -e  warn "XQuartz not found at /Applications/Utilities/XQuartz.app"

  warn "Install it from: https://www.xquartz.org"

if [[ $RC -ne 0 ]]; then  warn "Then log out/in (or reboot), open XQuartz once, and rerun this script."

  echo  warn "\nYou can still run headless simulation with:"

  echo "sumo-gui exited with code $RC. Common macOS fix:" >&2  warn "  ${SUMO_BIN} -c ${CFG_PATH} --no-step-log true"

  echo "  1) Install XQuartz: https://www.xquartz.org" >&2  exit 2

  echo "  2) Start XQuartz once, then log out/in (or reboot)" >&2fi

  echo "  3) In XQuartz Preferences → Security: enable 'Allow connections from network clients'" >&2

  echo "  4) Re-run this script" >&2# Try to start XQuartz if it isn't already.

  echo# (This will no-op if it's already running.)

  echo "Headless fallback:" >&2/usr/bin/open -gj "/Applications/Utilities/XQuartz.app" || true

  echo "  sumo -c \"$CFG\" --no-step-log true" >&2

  exit $RC# Give XQuartz a moment to come up.

fisleep 1


# DISPLAY handling:
# Many X11 apps default to :0.0; making it explicit helps.
# If the user already set DISPLAY, keep it.
if [[ -z "${DISPLAY:-}" ]]; then
  export DISPLAY=":0"
fi

log "Using SUMO_HOME=${SUMO_HOME}"
log "Using DISPLAY=${DISPLAY}"
log "Launching: ${SUMO_GUI_BIN} -c ${CFG_PATH}"

set +e
"${SUMO_GUI_BIN}" -c "${CFG_PATH}"
STATUS=$?
set -e

if [[ ${STATUS} -ne 0 ]]; then
  warn "SUMO-GUI exited with status ${STATUS}."
  warn "If you see 'unable to open display :0.0', XQuartz isn't accepting connections yet."
  warn "Try these in order:"
  warn "  1) Open XQuartz manually (Applications → Utilities → XQuartz)"
  warn "  2) In XQuartz: Settings → Security → enable 'Allow connections from network clients' (then restart XQuartz)"
  warn "  3) In a new terminal: export DISPLAY=:0  (or export DISPLAY=localhost:0)"
  warn "  4) Re-run this script"
  warn "\nHeadless fallback (always works):"
  warn "  ${SUMO_BIN} -c ${CFG_PATH} --no-step-log true"
  exit ${STATUS}
fi
