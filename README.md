# Audio Restoration Pipeline

> A professional-grade, end-to-end audio restoration toolkit combining classical DSP algorithms with modern neural models. Designed to rescue degraded recordings — vinyl rips, cassette transfers, old radio broadcasts, and historical archives.

---

## Table of Contents

- [Features](#features)
- [Processing Pipeline](#processing-pipeline)
- [Requirements & Installation](#requirements--installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
  - [Input / Output](#input--output)
  - [Batch Processing](#batch-processing)
  - [Genre Presets](#genre-presets)
  - [Denoising](#denoising)
  - [Click / Pop / Crackle Removal](#click--pop--crackle-removal)
  - [Electrical Hum Removal](#electrical-hum-removal)
  - [Equalization (5-Band)](#equalization-5-band)
  - [Source Separation (Demucs)](#source-separation-demucs)
  - [Super-Resolution](#super-resolution)
  - [Multiband Compression](#multiband-compression)
  - [M/S Stereo Processing](#ms-stereo-processing)
  - [Wow & Flutter Correction](#wow--flutter-correction)
  - [Output Options](#output-options)
  - [Reporting](#reporting)
- [Usage Examples](#usage-examples)
- [Python API](#python-api)
- [Project Structure](#project-structure)
- [Known Limitations](#known-limitations)

---

## Features

| Category | Technology | Description |
|---|---|---|
| **Denoising** | HPSS + Wiener, noisereduce, DeepFilterNet, Wavelet | Multi-method noise reduction, from light hiss to heavy static |
| **Click removal** | Dual-window RMS + LPC interpolation | Eliminates vinyl clicks, pops, and crackle in the time domain |
| **Scratch repair** | Bidirectional decaying LPC crossfade | Repairs 30–200 ms needle-drag scratches |
| **Hum removal** | IIR notch filter cascade | Removes 50/60 Hz power-line hum and up to 5 harmonics |
| **Equalization** | Butterworth zero-phase, 5-band mastering EQ | Bass, mid, presence, treble, and air shelves; sub-sonic rumble filter |
| **Source separation** | Demucs `htdemucs_ft` | Splits audio into vocals/drums/bass/other before per-stem processing |
| **Super-resolution** | AudioSR / scipy sinc resampler | Recovers high-frequency content and upsamples to 48 kHz |
| **Multiband compression** | Linkwitz-Riley 3-band compressor | Independent control over low, mid, and high frequency dynamics |
| **M/S stereo processing** | Mid/Side matrix encode → process → decode | Controls stereo width; denoise the side channel independently |
| **Wow & flutter correction** | Phase-vocoder pitch tracking | Corrects slow pitch instability from worn turntable/cassette mechanics |
| **Loudness normalization** | BS.1770-4 / EBU R128 | Targets –14 LUFS (streaming) or –23 LUFS (broadcast) |
| **Format support** | libsndfile + ffmpeg | Input and output: WAV, FLAC, AIFF, MP3, M4A, OGG, OPUS, WMA, … |
| **Batch mode** | Recursive folder scan | Process an entire collection with one command |
| **Quality metrics** | SNR, THD, spectral centroid, LUFS | Before/after report + waveform/spectrogram comparison PNG |

---

## Processing Pipeline

Each restoration job passes through the following stages in order:

```
Input file
    │
    ▼
[Phase 1]  Load & Pre-process     — decode any format, stereo/mono handling, float32 normalisation
    │
    ▼
[Phase 1.3] Wow & Flutter          — (optional) pitch-drift correction via phase vocoder
    │
    ▼
[Phase 1.7] Click / Scratch Repair — (optional) time-domain LPC interpolation
    │
    ▼
[Phase 2]  Denoising              — HPSS+Wiener / noisereduce / DeepFilterNet / Wavelet
    │
    ▼
[Phase 3]  Source Separation      — (optional) Demucs htdemucs_ft stem split → per-stem denoise → remix
    │
    ▼
[Phase 4]  Super-Resolution       — (optional) AudioSR / scipy → 48 kHz
    │
    ▼
[Phase 5]  Equalization           — 5-band mastering EQ, hum notch filters, rumble HPF
    │
    ▼
[Phase 5.3] Multiband Compression — (optional) 3-band Linkwitz-Riley compressor
    │
    ▼
[Phase 5.7] M/S Stereo Processing — (optional) Mid/Side encode → process → decode
    │
    ▼
[Phase 6]  Peak Normalize + Limit + LUFS
    │
    ▼
Output file  +  metrics report  +  comparison plot PNG
```

---

## Requirements & Installation

### System requirements

- **Python** 3.10 or newer (Python 3.14 tested for core DSP; AudioSR requires ≤ 3.10)
- **ffmpeg** — required for encoding/decoding MP3, M4A, OGG, OPUS, WMA
  ```bash
  # Debian/Ubuntu
  sudo apt install ffmpeg
  # Arch Linux
  sudo pacman -S ffmpeg
  # macOS (Homebrew)
  brew install ffmpeg
  ```

### Automated setup (recommended)

```bash
# Clone or enter the project folder
cd ProcesamientoAudioIA

# Full install — CPU-only PyTorch (fastest to set up, no GPU required)
bash setup.sh

# GPU install — CUDA-enabled PyTorch (faster for neural models)
bash setup.sh --gpu

# Verify installed packages
bash setup.sh --check
```

The script creates a virtual environment at `venv_audio/` and installs all dependencies.

### Manual install

```bash
python3 -m venv venv_audio
source venv_audio/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# CPU-only PyTorch:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# or GPU:
pip install torch torchaudio
```

### Activate the environment

```bash
source venv_audio/bin/activate
# Now run restore.py directly
python restore.py --help
```

---

## Quick Start

```bash
source venv_audio/bin/activate

# Restore a vinyl rip — lossless 24-bit WAV output
python restore.py -i vinyl_rip.wav -o restored.wav

# Restore and export as FLAC (archival quality)
python restore.py -i old_recording.mp3 -o restored.flac

# Restore a cassette tape with click removal and hum filter
python restore.py -i cassette.wav -o clean.wav \
    --preset vinyl \
    --declick \
    --dehum 50
```

---

## CLI Reference

All options shown below are combined on a single `restore.py` invocation. Run `python restore.py --help` for the full inline help.

---

### Input / Output

| Argument | Type | Default | Description |
|---|---|---|---|
| `-i`, `--input FILE` | path | — | Input audio file. Accepts WAV, FLAC, AIFF, MP3, M4A, OGG, OPUS, WMA, … |
| `-o`, `--output FILE` | path | — | Output file. Format inferred from extension. Lossless: `.wav` `.flac` `.aiff`; Lossy: `.mp3` `.m4a` `.ogg` `.opus` |

---

### Batch Processing

| Argument | Default | Description |
|---|---|---|
| `--batch FOLDER` | — | Process every audio file in a folder recursively |
| `--output-dir FOLDER` | `<input>/restored/` | Output folder for `--batch` results |
| `--output-ext EXT` | `.wav` | Output extension for batch mode (e.g. `flac`, `mp3`) |
| `--output-suffix TEXT` | `""` | Suffix appended to each output filename stem (e.g. `_restored`) |

---

### Genre Presets

Apply a pre-tuned 5-band EQ curve for a specific genre. Individual `--bass`, `--mid`, etc. flags always override the preset.

```bash
python restore.py -i song.wav -o out.wav --preset vinyl
```

| Preset | Bass (80 Hz) | Mid (250 Hz) | Presence (3.5 kHz) | Treble (8 kHz) | Air (12 kHz) |
|---|---|---|---|---|---|
| `vinyl` | +2.5 dB | −1.5 dB | +2.5 dB | +2.0 dB | +3.5 dB |
| `jazz` | +1.0 dB | 0.0 dB | +1.0 dB | +1.5 dB | +2.0 dB |
| `classical` | +1.5 dB | −0.5 dB | +1.0 dB | +1.5 dB | +3.0 dB |
| `hiphop` | +5.0 dB | −2.0 dB | +1.5 dB | +1.5 dB | +2.5 dB |
| `metal` | +2.0 dB | −3.5 dB | +3.0 dB | +2.5 dB | +2.0 dB |
| `electronic` | +4.0 dB | −1.0 dB | +2.0 dB | +2.5 dB | +3.0 dB |
| `podcast` | 0.0 dB | −1.0 dB | +3.5 dB | +1.5 dB | +1.0 dB |
| `flat` | 0.0 dB | 0.0 dB | 0.0 dB | 0.0 dB | 0.0 dB |

---

### Denoising

| Argument | Type | Default | Description |
|---|---|---|---|
| `--denoise-method METHOD` | choice | `music` | Algorithm: `music` (HPSS+Wiener, best for music), `noisereduce` (spectral subtraction), `deepfilternet` (neural, speech only), `wavelet` (fast fallback), `auto` (tries all) |
| `--prop-decrease 0.0–1.0` | float | `0.85` | Reduction aggressiveness. 0 = no change, 1 = full suppression. **Music**: 0.6–0.8; **Speech**: 0.7–0.9 |
| `--stationary` | flag | off | Tell `noisereduce` the noise is constant (tape hiss, vinyl hiss). Disable for variable background noise |
| `--n-std-thresh THRESH` | float | `1.5` | Noise detection sensitivity for `noisereduce --stationary`. Lower = more aggressive removal. `1.5` = conservative · `1.0` = balanced · `0.5` = aggressive |
| `--denoise-passes N` | int | `1` | Sequential denoising passes. `2`–`3` for stubborn static |

---

### Click / Pop / Crackle Removal

Operates entirely in the time domain; does not affect continuous noise.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--declick` | flag | off | Enable click/pop/crackle removal |
| `--declick-threshold N` | float | `4.0` | Detection sensitivity (× local RMS). Lower = more aggressive. **Vinyl**: 4–6; **Heavy crackle**: 3–4 |
| `--declick-lpc-order N` | int | `32` | LPC interpolation order. Higher = better timbre modelling for complex signals |
| `--declick-max-ms MS` | float | `30.0` | Maximum duration (ms) treated as a short click and repaired with forward LPC. Longer events escalate to scratch repair |
| `--declick-max-scratch MS` | float | `200.0` | Maximum duration (ms) of needle-drag scratches repaired with bidirectional LPC crossfade. Events beyond this limit are left untouched |

**How it works:**  
The detector computes a fast 50 ms RMS and a slow 1000 ms reference RMS for every sample. Transients that exceed `threshold × fast_RMS` are flagged as clicks; sustained bursts where `fast_RMS > (threshold−1) × slow_RMS` are flagged as scratches. Clicks (≤ `max-ms`) are reconstructed with single-sided LPC forward prediction. Scratches (30–200 ms) are repaired with a bidirectional decaying LPC blend that fades in from both boundaries and meets in silence at the centre.

---

### Electrical Hum Removal

Removes power-line hum and its harmonics using a cascade of narrow IIR notch filters.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dehum HZ` | float | off | Hum fundamental frequency. `50` = Europe/Asia/Africa/Australia; `60` = Americas/Japan |
| `--dehum-harmonics N` | int | `5` | Number of harmonics to notch (including fundamental: 50, 100, 150, 200, 250 Hz) |
| `--dehum-q Q` | float | `35.0` | Notch filter Q factor. Higher = narrower notch, less music colouration |

---

### Equalization (5-Band)

Zero-phase Butterworth EQ applied after denoising. All gains are in dB.

| Argument | Frequency | Default (vinyl) | Description |
|---|---|---|---|
| `--bass DB` | 80 Hz (low shelf) | +2.5 | Restores warmth lost in vinyl transfer |
| `--mid DB` | 250 Hz (bell) | −1.5 | Cuts muddy/boxy resonance common in old recordings |
| `--presence DB` | 3500 Hz (bell) | +2.5 | Sharpens vocal articulation and instrument attack |
| `--treble DB` | 8000 Hz (high shelf) | +2.0 | Restores high-frequency clarity and definition |
| `--air DB` | 12000 Hz (high shelf) | +3.5 | Adds sparkle and open "air" of modern masters |
| `--no-rumble-filter` | < 30 Hz (HPF) | on | Disable the sub-sonic high-pass filter that removes turntable rumble |

Pass `0` to any band to disable it, or negative values to cut.

---

### Source Separation (Demucs)

Splits the audio into stems — vocals, drums, bass, other — processes each one, then remixes.

| Argument | Default | Description |
|---|---|---|
| `--separate` | off | Enable source separation via Demucs |
| `--demucs-model MODEL` | `htdemucs_ft` | Model: `htdemucs_ft` (best quality), `htdemucs`, `mdx_extra` |
| `--stem-denoise` | off | Denoise each stem independently before remixing. Gives the denoiser a simpler, per-instrument signal. Requires `--separate` |
| `--device auto\|cpu\|cuda` | `auto` | Processing device for neural models |

> **Note:** Demucs runs at approximately 4× real-time on CPU. A 3-minute song takes ~12 minutes without a GPU.

---

### Super-Resolution

Recovers high-frequency content lost in MP3 encoding or old recordings and upsamples to 48 kHz.

| Argument | Default | Description |
|---|---|---|
| `--super-resolution` | off | Enable super-resolution upsampling |
| `--output-sr HZ` | pipeline SR | Resample the final output to this sample rate (e.g. `44100`) |
| `--device auto\|cpu\|cuda` | `auto` | Processing device (shared with `--separate`) |

AudioSR (neural diffusion model) is used when available. Falls back to scipy's high-quality sinc resampler automatically.

---

### Multiband Compression

3-band Linkwitz-Riley compressor applied after EQ for balanced dynamics.

| Argument | Default | Description |
|---|---|---|
| `--multiband` | off | Enable the multiband compressor |
| `--mb-low-threshold DB` | `−20` | Low band threshold (dBFS) |
| `--mb-low-ratio R` | `2.5` | Low band compression ratio |
| `--mb-low-makeup DB` | `1.5` | Low band make-up gain |
| `--mb-mid-threshold DB` | `−18` | Mid band threshold (dBFS) |
| `--mb-mid-ratio R` | `3.0` | Mid band compression ratio |
| `--mb-mid-makeup DB` | `1.0` | Mid band make-up gain |
| `--mb-high-threshold DB` | `−16` | High band threshold (dBFS) |
| `--mb-high-ratio R` | `2.0` | High band compression ratio |
| `--mb-high-makeup DB` | `0.5` | High band make-up gain |

---

### M/S Stereo Processing

Encodes stereo to Mid/Side, processes each channel independently, then decodes back to stereo. No-op for mono files.

| Argument | Default | Description |
|---|---|---|
| `--ms` | off | Enable M/S stereo processing |
| `--ms-side-prop 0.0–1.0` | `0.6` | Noise reduction strength on the Side channel |
| `--ms-no-side-denoise` | on | Disable Side channel noise reduction (use when the stereo image is intact) |
| `--ms-width SCALE` | `1.0` | Stereo width multiplier applied to the Side channel. `< 1` = narrower, `> 1` = wider |
| `--ms-mid-presence DB` | `0.0` | Presence EQ boost/cut on the Mid channel (centred at `--ms-mid-presence-freq`) |
| `--ms-mid-presence-freq HZ` | `3500` | Centre frequency for Mid presence EQ |

---

### Wow & Flutter Correction

Corrects slow pitch instability caused by worn turntable or cassette mechanics using a phase-vocoder pitch tracker.

| Argument | Default | Description |
|---|---|---|
| `--wow-flutter` | off | Enable wow & flutter correction |
| `--wf-max-cents CENTS` | `100` | Maximum pitch correction in cents (100 cents = 1 semitone) |
| `--wf-smoothing MS` | `200` | Smoothing window for nominal pitch in ms. Longer = correct only slow wow; shorter = also correct fast flutter |
| `--wf-max-freq HZ` | `100` | Upper frequency of pitch fluctuations to correct. Values above this are assumed intentional |

---

### Output Options

| Argument | Default | Description |
|---|---|---|
| `--bitrate BITRATE` | format default | Bitrate for lossy output. MP3: `V0` (VBR) or `320k` (CBR); M4A: `256k`, `192k`; OGG: quality 0–10; OPUS: `192k` |
| `--no-vbr` | VBR on | Use CBR instead of VBR for MP3 output |
| `--bit-depth 16\|24\|32` | `24` | Bit depth for lossless output (.wav, .flac, .aiff) |
| `--no-normalize` | normalise on | Disable peak normalisation of the output |
| `--no-limit` | limit on | Disable the soft limiter |
| `--limit-threshold DBFS` | `−0.3` | Soft limiter threshold in dBFS |
| `--lufs-target LUFS` | `−14.0` | Target integrated loudness: `−14` = streaming · `−23` = broadcast |
| `--no-lufs` | — | Disable LUFS normalisation (use peak-only) |

---

### Reporting

| Argument | Default | Description |
|---|---|---|
| `--no-plot` | plot on | Do not save the before/after spectrogram comparison PNG |
| `--no-metrics` | metrics on | Do not print the quality metrics report |
| `--verbose` | off | Enable DEBUG-level logging |

---

## Usage Examples

### Vinyl Records

```bash
# Standard vinyl restoration — bright, warm, no crackle
python restore.py \
    -i vinyl_rip.wav \
    -o vinyl_restored.flac \
    --preset vinyl \
    --declick \
    --declick-threshold 4 \
    --dehum 50

# Heavily crackled 78 RPM record — aggressive click removal
python restore.py \
    -i old_78rpm.wav \
    -o clean_78rpm.wav \
    --preset vinyl \
    --declick \
    --declick-threshold 3 \
    --declick-max-scratch 200 \
    --denoise-method noisereduce \
    --prop-decrease 0.8 \
    --stationary \
    --n-std-thresh 1.0

# Vinyl with bad hum and rumble (European pressing)
python restore.py \
    -i noisy_vinyl.wav \
    -o clean_vinyl.wav \
    --preset vinyl \
    --dehum 50 \
    --dehum-harmonics 5 \
    --declick
```

### Cassette Tapes

```bash
# Standard cassette restoration — remove hiss and wow
python restore.py \
    -i cassette.wav \
    -o cassette_restored.wav \
    --denoise-method noisereduce \
    --prop-decrease 0.75 \
    --stationary \
    --n-std-thresh 1.0 \
    --wow-flutter \
    --preset flat \
    --treble 2 --presence 1.5

# Multi-pass for extreme tape degradation
python restore.py \
    -i degraded_tape.wav \
    -o tape_restored.wav \
    --denoise-method noisereduce \
    --prop-decrease 0.85 \
    --stationary \
    --n-std-thresh 0.8 \
    --denoise-passes 2 \
    --wow-flutter \
    --wf-smoothing 300 \
    --declick
```

### Streaming / Distribution

```bash
# Restore and master for streaming (–14 LUFS, VBR MP3)
python restore.py \
    -i recording.wav \
    -o master_streaming.mp3 \
    --preset vinyl \
    --declick \
    --multiband \
    --lufs-target -14

# Archival FLAC at 24-bit, no loudness processing
python restore.py \
    -i recording.wav \
    -o archive.flac \
    --preset flat \
    --declick \
    --no-lufs \
    --no-normalize \
    --no-limit \
    --bit-depth 24

# Broadcast WAV at –23 LUFS (EBU R128)
python restore.py \
    -i recording.wav \
    -o broadcast.wav \
    --preset flat \
    --lufs-target -23 \
    --no-plot
```

### Neural Models (GPU recommended)

```bash
# Full AI pipeline: separate → per-stem denoise → super-resolution
python restore.py \
    -i song.mp3 \
    -o song_ai_restored.wav \
    --preset vinyl \
    --separate \
    --stem-denoise \
    --super-resolution \
    --device cuda

# Source separation only (no super-resolution)
python restore.py \
    -i song.wav \
    -o song_separated.wav \
    --separate \
    --demucs-model htdemucs_ft \
    --device auto
```

### Stereo Width Control

```bash
# Widen the stereo image of a mono-ish old recording
python restore.py \
    -i narrow_stereo.wav \
    -o wider.wav \
    --ms \
    --ms-width 1.4 \
    --ms-no-side-denoise

# Narrow an overly wide recording and clean up the side channel
python restore.py \
    -i wide_noisy.wav \
    -o narrowed.wav \
    --ms \
    --ms-width 0.7 \
    --ms-side-prop 0.6
```

### Podcasts and Speech

```bash
# Restore a spoken-word recording with presence boost and hum removal
python restore.py \
    -i interview.wav \
    -o interview_clean.wav \
    --preset podcast \
    --dehum 60 \
    --denoise-method noisereduce \
    --prop-decrease 0.9 \
    --stationary

# Speech with DeepFilterNet (best neural quality, speech only)
python restore.py \
    -i speech.wav \
    -o speech_clean.wav \
    --denoise-method deepfilternet \
    --preset podcast \
    --no-lufs
```

### Batch Processing

```bash
# Restore an entire folder of vinyl rips → FLAC
python restore.py \
    --batch ./Vinyl_Rips/ \
    --output-dir ./Restored_FLAC/ \
    --output-ext flac \
    --preset vinyl \
    --declick \
    --dehum 50

# Batch with a filename suffix, keeping original format
python restore.py \
    --batch ./Cassettes/ \
    --output-suffix _restored \
    --denoise-method noisereduce \
    --prop-decrease 0.75 \
    --stationary \
    --wow-flutter
```

### Manual EQ Override

```bash
# Override specific bands on top of a preset
python restore.py \
    -i song.wav \
    -o out.wav \
    --preset vinyl \
    --bass 4.0 \
    --air 0

# Flat output with only a mid cut (remove boxiness)
python restore.py \
    -i recording.wav \
    -o out.wav \
    --preset flat \
    --mid -2 \
    --presence 1
```

### Output Format Examples

```bash
# WAV — PCM 24-bit lossless (default)
python restore.py -i song.mp3 -o song.wav

# FLAC — lossless compression, archival quality
python restore.py -i song.mp3 -o song.flac

# MP3 — VBR V0 (~245 kbps), highest variable-bitrate quality
python restore.py -i song.wav -o song.mp3

# MP3 — 320 kbps CBR
python restore.py -i song.wav -o song_320.mp3 --bitrate 320k --no-vbr

# M4A / AAC — streaming-friendly
python restore.py -i song.wav -o song.m4a --bitrate 256k

# OGG Vorbis — open format, high quality
python restore.py -i song.wav -o song.ogg

# OPUS — low-bitrate with excellent quality
python restore.py -i song.wav -o song.opus --bitrate 192k

# AIFF — Apple lossless interchange format
python restore.py -i song.wav -o song.aiff --bit-depth 24
```

---

## Python API

The pipeline can be used programmatically from any Python script or notebook.

```python
from restoration_pipeline import RestorationPipeline, PipelineConfig

# Build a configuration object
config = PipelineConfig(
    denoise_method="noisereduce",
    denoise_stationary=True,
    denoise_prop_decrease=0.75,
    denoise_passes=2,

    enable_declicker=True,
    declicker_threshold=4.0,
    declicker_max_click_ms=30.0,
    declicker_max_scratch_ms=200.0,

    dehum_freq=50.0,          # European 50 Hz hum
    dehum_harmonics=5,

    bass_gain_db=2.5,
    mid_gain_db=-1.5,
    presence_gain_db=2.5,
    treble_gain_db=2.0,
    air_gain_db=3.5,

    lufs_target=-14.0,        # Streaming standard
    output_bit_depth=24,
)

# Instantiate once — heavy models are lazy-loaded on first use
pipeline = RestorationPipeline(config)

# Restore a single file
metrics = pipeline.restore("vinyl_rip.wav", "restored.flac")
print(metrics)
```

### Working with individual modules

Each processing stage is also available as a standalone class:

```python
import numpy as np
import soundfile as sf
from declicker import Declicker
from denoiser import Denoiser
from equalizer import AudioEqualizer

audio, sr = sf.read("recording.wav")

# Remove clicks and scratches
dc = Declicker(threshold=4.0, max_click_ms=30.0, max_scratch_ms=200.0)
audio = dc.process(audio, sr)

# Denoise
dn = Denoiser(method="noisereduce", prop_decrease=0.75, stationary=True)
audio = dn.process(audio, sr)

# Equalise
eq = AudioEqualizer(bass_gain_db=2.5, treble_gain_db=2.0, air_gain_db=3.5)
audio = eq.process(audio, sr)

sf.write("output.wav", audio, sr, subtype="PCM_24")
```

---

## Project Structure

```
ProcesamientoAudioIA/
├── restore.py                 # CLI entry point
├── restoration_pipeline.py   # Pipeline orchestrator + PipelineConfig
│
├── Pre-processing.py          # Phase 1  — AudioPreprocessor (load, decode, normalise)
├── declicker.py               # Phase 1.7 — Declicker (click/pop/scratch removal)
├── denoiser.py                # Phase 2  — Denoiser (multi-method noise reduction)
├── source_separator.py        # Phase 3  — SourceSeparator (Demucs)
├── super_resolution.py        # Phase 4  — SuperResolution (AudioSR / scipy)
├── equalizer.py               # Phase 5  — AudioEqualizer (5-band Butterworth EQ)
├── multiband_compressor.py    # Phase 5.3 — MultibandCompressor (Linkwitz-Riley)
├── ms_processor.py            # Phase 5.7 — MSProcessor (Mid/Side stereo)
├── wow_flutter.py             # Phase 1.3 — WowFlutterCorrector (pitch stabiliser)
├── format_handler.py          # FormatHandler (libsndfile + ffmpeg encode/decode)
├── metrics.py                 # QualityMetrics (SNR, THD, LUFS, spectral centroid)
│
├── requirements.txt           # Python dependencies
├── setup.sh                   # Automated environment setup script
└── Canciones/                 # Working directory for input/output audio files
```

---

## Known Limitations

| Limitation | Detail |
|---|---|
| **AudioSR requires Python ≤ 3.10** | The AudioSR super-resolution model is incompatible with Python 3.11+. The pipeline falls back to scipy sinc resampling automatically |
| **DeepFilterNet is speech-only** | Applying `--denoise-method deepfilternet` to music will suppress musical content. Use `music` or `noisereduce` for music |
| **Demucs CPU speed** | `htdemucs_ft` runs at ~4× real-time on CPU. A 4-minute song takes ~16 minutes without a GPU |
| **Scratch repair length limit** | Events longer than `--declick-max-scratch` (default 200 ms) are left untouched to avoid replacing too much original content |
| **Mono super-resolution** | AudioSR processes mono audio; stereo files are split, upsampled per channel, and recombined |
| **M/S processing is stereo-only** | `--ms` flags are silently ignored for mono input files |
| **LUFS measurement** | Requires a minimum signal duration (~2 s) for a reliable integrated loudness reading; very short clips fall back to peak normalisation |
