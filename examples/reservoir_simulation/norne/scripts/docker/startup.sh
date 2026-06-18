#!/usr/bin/env bash
# =============================================================================
# startup.sh — End-to-end environment bootstrap and pipeline launcher
# =============================================================================
#
# Run this once inside the ptyche container with:
#
#     source ./scripts/docker/startup.sh
#
# It will:
#   [1] Install PhysicsNeMo (if not already installed)
#   [2] Activate the physicsnemoenv virtual environment
#   [3] Install OPM (always, per workflow requirement)
#   [4] cd into src/
#   [5] Launch well_modeling.sh
#
# Any extra arguments are forwarded to well_modeling.sh, e.g.:
#     source ./scripts/docker/startup.sh --ranks 4 --config conf/MY_CONFIG.yaml
#
# Author: Clement Etienam <cetienam@nvidia.com>
# =============================================================================

# ── ANSI colour helpers ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
RESET='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${RESET}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
log_error() { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
log_step()  { echo -e "${MAGENTA}[STEP]${RESET}  $*"; }

log_section() {
    local title="$1"
    local width=70
    local line
    line=$(printf '%*s' "$width" '' | tr ' ' '─')
    echo
    echo -e "${BOLD}${CYAN}${line}${RESET}"
    printf "${BOLD}${CYAN}  %-$((width - 2))s${RESET}\n" "$title"
    echo -e "${BOLD}${CYAN}${line}${RESET}"
}

# ── Resolve project paths ────────────────────────────────────────────────────
# When sourced, BASH_SOURCE[0] still points to this file's path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INSTALL_PHYSICSNEMO="${SCRIPT_DIR}/install_phynemo.sh"
INSTALL_OPM="${SCRIPT_DIR}/opm_install.sh"
VENV_ACTIVATE="${PROJECT_ROOT}/physicsnemo_venv/bin/activate"
SRC_DIR="${PROJECT_ROOT}/src"
WELL_MODELING="${SRC_DIR}/well_modeling.sh"

PIPELINE_START=$(date +%s)
elapsed() {
    local secs=$(( $(date +%s) - PIPELINE_START ))
    printf '%02d:%02d:%02d' $(( secs/3600 )) $(( (secs%3600)/60 )) $(( secs%60 ))
}

# Helper: log an error and return non-zero WITHOUT killing the shell.
fail() {
    log_error "$1"
    return 1
}

# ── Banner ───────────────────────────────────────────────────────────────────
log_section "PhysicsNeMo Reservoir — Automated Startup"
log_info "Project root : ${PROJECT_ROOT}"
log_info "Started at   : $(date '+%Y-%m-%d %H:%M:%S')"
log_info "Hostname     : $(hostname)"

# ── Sanity checks ────────────────────────────────────────────────────────────
log_section "Validating Required Files"

for f in "$INSTALL_PHYSICSNEMO" "$INSTALL_OPM" "$WELL_MODELING"; do
    if [[ ! -f "$f" ]]; then
        fail "Required script not found: $f"
        return 1
    fi
    log_ok "Found: ${f#$PROJECT_ROOT/}"
done

# ── Step 1: Install PhysicsNeMo if missing ───────────────────────────────────
log_section "Step 1 — PhysicsNeMo Installation"
step_start=$(date +%s)

if [[ -f "$VENV_ACTIVATE" ]]; then
    log_ok "physicsnemoenv already exists — skipping installation."
else
    log_step "Running: ${INSTALL_PHYSICSNEMO}"
    if ! bash "${INSTALL_PHYSICSNEMO}"; then
        fail "PhysicsNeMo installation failed."
        return 1
    fi
    if [[ ! -f "$VENV_ACTIVATE" ]]; then
        fail "Installation completed but ${VENV_ACTIVATE} not found."
        return 1
    fi
    log_ok "PhysicsNeMo installed."
fi

step_secs=$(( $(date +%s) - step_start ))
log_ok "Step 1 completed in $(printf '%02d:%02d' $((step_secs/60)) $((step_secs%60)))"

# ── Step 2: Activate virtual environment ─────────────────────────────────────
log_section "Step 2 — Activating physicsnemoenv"

# shellcheck source=/dev/null
source "${VENV_ACTIVATE}"
log_ok "Virtual environment activated."
log_info "Python : $(command -v python)"
log_info "Version: $(python --version 2>&1)"

# ── Step 3: Install OPM (always) ─────────────────────────────────────────────
log_section "Step 3 — OPM Installation"
step_start=$(date +%s)

log_step "Running: ${INSTALL_OPM}"
if ! bash "${INSTALL_OPM}"; then
    fail "OPM installation failed."
    return 1
fi
log_ok "OPM installation finished."

step_secs=$(( $(date +%s) - step_start ))
log_ok "Step 3 completed in $(printf '%02d:%02d' $((step_secs/60)) $((step_secs%60)))"

# ── Step 4: Move into src/ and launch the pipeline ───────────────────────────
log_section "Step 4 — Launching Well Modeling Pipeline"

cd "${SRC_DIR}" || { fail "Could not cd into ${SRC_DIR}"; return 1; }
log_ok "Changed directory to: $(pwd)"

if [[ ! -x "$WELL_MODELING" ]]; then
    log_warn "well_modeling.sh is not executable — fixing permissions."
    chmod +x "$WELL_MODELING"
fi

log_step "Running: ./well_modeling.sh $*"
echo
./well_modeling.sh "$@"
WELL_MODELING_RC=$?

# ── Done ─────────────────────────────────────────────────────────────────────
log_section "Startup Complete"
if [[ $WELL_MODELING_RC -eq 0 ]]; then
    log_ok "All stages finished successfully."
else
    log_warn "well_modeling.sh exited with code ${WELL_MODELING_RC}."
fi
log_ok "Total elapsed: $(elapsed)"
log_info "Virtual environment is still active in this shell."
log_info "To deactivate, run: deactivate"