#!/usr/bin/env bash
# setup.sh — Automated environment setup for Audio Restoration Pipeline
# ======================================================================
# Usage:
#   bash setup.sh           # Full install (CPU PyTorch)
#   bash setup.sh --gpu     # Install GPU (CUDA) version of PyTorch
#   bash setup.sh --check   # Check what is already installed

set -euo pipefail

VENV_DIR="$(dirname "$0")/venv_audio"
USE_GPU=0
CHECK_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --gpu)   USE_GPU=1 ;;
        --check) CHECK_ONLY=1 ;;
        --help)
            echo "Usage: bash setup.sh [--gpu] [--check]"
            exit 0 ;;
    esac
done

echo "============================================================"
echo "  Audio Restoration Pipeline — Environment Setup"
echo "============================================================"

# ── 1. Create virtual environment ────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[1/5] Creating virtual environment at $VENV_DIR …"
    python3 -m venv "$VENV_DIR"
    # Bootstrap pip inside the venv (needed on Arch Linux)
    "$VENV_DIR/bin/python" -m ensurepip 2>/dev/null || true
else
    echo "[1/5] Virtual environment already exists: $VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip3"

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo ""
    echo "=== Installed packages ==="
    "$PIP" list 2>/dev/null | grep -E \
        "librosa|torch|scipy|soundfile|numpy|noisereduce|demucs|deepfilter|audiosr|PyWavelets|matplotlib" \
        || echo "(none found)"
    exit 0
fi

# ── 2. Upgrade pip ────────────────────────────────────────────────────
echo "[2/5] Upgrading pip …"
"$PIP" install --upgrade pip setuptools wheel --quiet

# ── 3. Core audio & DSP packages ─────────────────────────────────────
echo "[3/5] Installing core audio & DSP packages …"
"$PIP" install \
    "librosa>=0.10.0" \
    "soundfile>=0.12.0" \
    "scipy>=1.10.0" \
    "numpy>=1.24.0" \
    "matplotlib>=3.7.0" \
    "PyWavelets>=1.4.0" \
    "noisereduce>=3.0.0" \
    --quiet

# ── 4. PyTorch ───────────────────────────────────────────────────────
echo "[4/5] Installing PyTorch + torchaudio …"
if [[ $USE_GPU -eq 1 ]]; then
    echo "      → GPU (CUDA) build"
    "$PIP" install torch torchaudio --quiet
else
    echo "      → CPU build (add --gpu flag for CUDA)"
    "$PIP" install torch torchaudio \
        --index-url https://download.pytorch.org/whl/cpu \
        --quiet
fi

# ── 5. Neural audio models ────────────────────────────────────────────
echo "[5/5] Installing DeepFilterNet and Demucs …"
"$PIP" install deepfilternet demucs --quiet

# ── Apply torchaudio compatibility patch ─────────────────────────────
echo ""
echo "Applying torchaudio compatibility patch for DeepFilterNet …"
BACKEND_DIR="$VENV_DIR/lib/python3.*/site-packages/torchaudio/backend"
# shellcheck disable=SC2086
mkdir -p $BACKEND_DIR 2>/dev/null || true
# shellcheck disable=SC2086
cat > $BACKEND_DIR/common.py << 'PYEOF'
"""Compatibility shim: provides AudioMetaData for deepfilternet with torchaudio >= 2.0"""
from dataclasses import dataclass

@dataclass
class AudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int
    encoding: str
PYEOF
echo "  Patch applied."

# ── Optional: AudioSR ─────────────────────────────────────────────────
PYTHON_VER=$("$VENV_DIR/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo ""
echo "── Optional: AudioSR (super-resolution) ───────────────────"
if python3 -c "import sys; assert sys.version_info <= (3, 10)" 2>/dev/null; then
    echo "  Python $PYTHON_VER detected — installing audiosr …"
    "$PIP" install audiosr --quiet && echo "  AudioSR installed." || echo "  AudioSR install failed (non-critical)."
else
    echo "  Python $PYTHON_VER detected — AudioSR requires Python <= 3.10."
    echo "  Skipping. scipy resampler will be used instead (still high quality)."
fi

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "Activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Run a restoration:"
echo "  python restore.py --input your_song.wav --output restored.wav"
echo "  python restore.py --help    # see all options"
echo ""
