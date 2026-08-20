#!/usr/bin/env bash
# ======================================================================
# setup.sh — Environment setup for audio-restoration
#
# Usage:
#   bash setup.sh            # Full CPU install (recommended)
#   bash setup.sh --gpu      # Install GPU (CUDA) build of PyTorch
#   bash setup.sh --dev      # Also install dev tools (pytest, ruff, mypy)
#   bash setup.sh --skip-neural  # Skip PyTorch / neural models (core DSP only)
#   bash setup.sh --check    # Only inspect what is already installed
#   bash setup.sh --help
#
# Safe to re-run: the script is idempotent.
# ======================================================================

set -Eeuo pipefail

# ─── Script location ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_DIR="$SCRIPT_DIR/venv_audio"

# ─── Options ─────────────────────────────────────────────────────────
USE_GPU=0
DEV_INSTALL=0
CHECK_ONLY=0
SKIP_NEURAL=0

# ─── Logging helpers ─────────────────────────────────────────────────
log_info() { echo "[$(date +'%H:%M:%S')] INFO: $*"; }
log_warn() { echo "[$(date +'%H:%M:%S')] WARN: $*" >&2; }
log_error() { echo "[$(date +'%H:%M:%S')] ERROR: $*" >&2; }

# ─── Neural runtime check ────────────────────────────────────────────
# Returns 1 if any of the core neural toolkits cannot be imported.
neural_failed() {
    local -a missing=()
    for mod in torch demucs deepfilternet; do
        if ! venv_python -c "import $mod" &>/dev/null; then
            missing+=("$mod")
        fi
    done
    [[ ${#missing[@]} -gt 0 ]]
}

usage() {
    cat <<'EOF'
Usage: bash setup.sh [OPTIONS]

Options:
    --gpu           Install the GPU (CUDA) build of PyTorch instead of CPU.
    --dev           Also install development tools (pytest, ruff, mypy, build).
    --skip-neural   Skip PyTorch / neural model installation (core DSP only).
    --check         Only print what is already installed and exit.
    --help          Show this help message and exit.
EOF
    exit "${1:-0}"
}

# ─── Argument parsing ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)  USE_GPU=1; shift ;;
        --dev)  DEV_INSTALL=1; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        --skip-neural) SKIP_NEURAL=1; shift ;;
        --help) usage 0 ;;
        *)
            log_error "Unknown option: $1"
            usage 1
            ;;
    esac
done

# ─── Dependency checks ───────────────────────────────────────────────
check_dependencies() {
    local -a missing_deps=()
    local -a required=("python3" "git" "ffmpeg")

    for cmd in "${required[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            missing_deps+=("$cmd")
        fi
    done

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_warn "Missing commands (optional but recommended): ${missing_deps[*]}"
    fi

    if ! command -v "python3" &>/dev/null; then
        log_error "python3 is required to create the virtual environment."
        exit 1
    fi
}

# ─── Helpers ─────────────────────────────────────────────────────────
venv_python() { "$VENV_DIR/bin/python" "$@"; }
venv_pip()    { "$VENV_DIR/bin/python" -m pip "$@"; }

require_venv() {
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        log_error "Virtual environment not found at: $VENV_DIR"
        log_error "Run 'bash setup.sh' first to create it."
        exit 1
    fi
}

# ─── Main flow ───────────────────────────────────────────────────────
main() {
    check_dependencies
    log_info "Audio restoration environment setup"
    log_info "Script dir: $SCRIPT_DIR"

    # ── 1. Create / reuse the virtual environment ────────────────────
    if [[ -d "$VENV_DIR" ]]; then
        log_info "Virtual environment already exists: $VENV_DIR"
    else
        log_info "Creating virtual environment: $VENV_DIR"
        python3 -m venv "$VENV_DIR" || {
            log_error "Failed to create virtual environment."
            exit 1
        }
        # Bootstrap pip (needed on some distros, e.g. Arch Linux)
        venv_python -m ensurepip >/dev/null 2>&1 || true
    fi

    if [[ "$CHECK_ONLY" -eq 1 ]]; then
        log_info "Installed audio-related packages:"
        venv_pip list 2>/dev/null | grep -E \
            "librosa|torch|torchaudio|scipy|soundfile|numpy|noisereduce|demucs|deepfilter|audiosr|PyWavelets|matplotlib|pyloudnorm|pytest|ruff|mypy" \
            || echo "  (none found)"
        log_info "Neural runtime imports:"
        neural_failed && echo "  torch/demucs/deepfilternet: NOT all importable" \
            || echo "  torch / demucs / deepfilternet: OK"
        exit 0
    fi

    # ── 2. Upgrade build tools ───────────────────────────────────────
    log_info "Upgrading pip / setuptools / wheel"
    venv_pip install --upgrade pip setuptools wheel --quiet

    # ── 3a. Install the package + core dependencies (always required) ──
    log_info "Installing audio-restoration (core extras)"
    venv_pip install --quiet -e "." || {
        log_error "Core package install failed. Check the error above."
        exit 1
    }

    # ── 3b. Optional neural stack (torch, demucs, deepfilternet) ─────
    if [[ "$SKIP_NEURAL" -eq 1 ]]; then
        log_info "Skipping neural installation (--skip-neural)."
    else
        log_info "Installing neural extras"
        if [[ "$USE_GPU" -eq 1 ]]; then
            log_info "  PyTorch build: GPU (CUDA)"
            venv_pip install --quiet "torch" "torchaudio"
        else
            log_info "  PyTorch build: CPU (use --gpu for CUDA)"
            venv_pip install --quiet "torch" "torchaudio" \
                --index-url https://download.pytorch.org/whl/cpu || {
                    log_warn "CPU PyTorch install failed; trying default index."
                    venv_pip install --quiet "torch" "torchaudio"
                }
        fi

        venv_pip install --quiet -e ".[neural]" || log_warn \
            "Neural extras install failed (non-critical). Core DSP still works."

        if neural_failed; then
            log_warn "Some neural models could not be imported."
            log_warn "Rerun with '--skip-neural' to skip these on future runs."
        fi
    fi

    # ── 4. Dev tools ─────────────────────────────────────────────────
    if [[ "$DEV_INSTALL" -eq 1 ]]; then
        log_info "Installing development tools"
        venv_pip install --quiet -e ".[dev]"
    fi

    # ── 5. Optional extras ───────────────────────────────────────────
    PYTHON_MINOR="$(venv_python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$SKIP_NEURAL" -eq 1 ]]; then
        log_info "Skipping AudioSR (requires the neural stack)."
    else
        case "$PYTHON_MINOR" in
            3.8|3.9|3.10)
                log_info "Python $PYTHON_MINOR supports AudioSR — installing (optional)."
                venv_pip install --quiet audiosr || log_warn "AudioSR install failed (non-critical)."
                ;;
            *)
                log_info "Python $PYTHON_MINOR > 3.10 — AudioSR not supported; scipy resampler will be used."
                ;;
        esac
    fi

    # ── 6. Verification ──────────────────────────────────────────────
    log_info "Verifying installation"
    venv_python -c \
        "import audio_restoration; print('  audio_restoration', audio_restoration.__version__)" || {
            log_error "Import check failed."
            exit 1
        }
    if [[ "$SKIP_NEURAL" -eq 0 ]] && neural_failed; then
        log_warn "Neural imports failed — retry with '--skip-neural' for core-only install."
    fi

    log_info "Setup complete."
    log_info "Activate:  source \"$VENV_DIR/bin/activate\""
    log_info "Run:       audio-restore --help"
}

trap 'log_error "Setup aborted (line $LINENO)."' ERR
main "$@"
