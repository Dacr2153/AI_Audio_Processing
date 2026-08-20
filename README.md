# Audio Restoration Pipeline

> A professional-grade, end-to-end audio restoration toolkit combining classical
> DSP algorithms with modern neural models. Designed to rescue degraded
> recordings — vinyl rips, cassette transfers, old radio broadcasts, and
> historical archives.

**Key features**

- Full restoration chain: dehum → declick → denoise → source separation
  (Demucs) → super-resolution (AudioSR/scipy) → mastering EQ → multiband
  compression → M/S stereo processing → LUFS loudness normalisation.
- **True stereo end-to-end** — every DSP stage operates channel-wise, so
  stereo material is never down-mixed.
- Multiple denoising engines: HPSS+Wiener (`music`), spectral subtraction
  (`noisereduce`), neural DeepFilterNet, and wavelet BayesShrink, with an
  `auto` fallback chain.
- Extensible, tested, and typed: packaged as `audio_restoration`
  (Python >= 3.10), with pytest, ruff and mypy wired into CI.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command-Line Interface](#command-line-interface)
- [Processing Pipeline](#processing-pipeline)
- [Genre Presets](#genre-presets)
- [Denoising](#denoising)
- [Developer Guide](#developer-guide)
- [Project Layout](#project-layout)
- [Optional Dependencies](#optional-dependencies)

---

## Installation

Python **>= 3.10** is required. Two options:

### 1. Automated setup script (recommended)

```bash
bash setup.sh          # CPU PyTorch
bash setup.sh --gpu    # CUDA PyTorch
bash setup.sh --dev    # also install pytest/ruff/mypy/build
```

The script creates `venv_audio/`, installs the package and its dependencies,
and is **idempotent** — safe to re-run. Use `bash setup.sh --check` to see what
is installed.

### 2. Manual

```bash
python -m venv venv_audio
source venv_audio/bin/activate
pip install -e .
# neural models (optional, heavy):
pip install -e ".[neural]"
```

After installation the command is `audio-restore` (also available via
`python -m audio_restoration`).

---

## Quick Start

Process a single file:

```bash
audio-restore -i old_recording.mp3 -o restored.flac
```

Vinyl-optimised run (dehum + declick + denoise + EQ preset + plot):

```bash
audio-restore -i vinyl_rip.wav -o restored.wav --preset vinyl \
              --dehum 50 --declick
```

Process an entire folder:

```bash
audio-restore --batch ./archives --output-dir ./restored --output-ext flac \
              --output-suffix _restored
```

Lossy output (needs ffmpeg):

```bash
audio-restore -i track.wav -o track.mp3 --bitrate 320k
```

Use in Python:

```python
from audio_restoration import PipelineConfig, RestorationPipeline

cfg = PipelineConfig(
    denoise_method="auto",
    enable_declicker=True,
    dehum_freq=50.0,
    enable_ms=True,
)
pipeline = RestorationPipeline(cfg)
pipeline.restore("input.wav", "output.flac")
```

---

## Command-Line Interface

```
audio-restore [-h] [--version] [-i FILE] [-o FILE] [--batch FOLDER]
              [--output-dir FOLDER] [--output-ext EXT] [--output-suffix SUFFIX]
              [--preset GENRE]
              # Denoising
              [--denoise-method METHOD] [--prop-decrease 0-1] [--stationary]
              [--n-std-thresh THRESH] [--denoise-passes N]
              # Source separation
              [--separate] [--demucs-model MODEL] [--stem-denoise]
              # Super-resolution
              [--super-resolution] [--output-sr HZ]
              # Equalization
              [--bass DB] [--mid DB] [--presence DB] [--treble DB] [--air DB]
              [--no-rumble-filter]
              # Output
              [--bitrate BITRATE] [--no-vbr] [--bit-depth BITS]
              [--no-normalize] [--no-limit] [--limit-threshold DBFS]
              [--lufs-target LUFS] [--no-lufs]
              # Dehum / Declick / Multiband / M-S / Wow & flutter
              [--dehum HZ] [--dehum-harmonics N] [--dehum-q Q]
              [--declick] [--declick-threshold N] [--declick-lpc-order N]
              [--declick-max-ms MS] [--declick-max-scratch MS]
              [--multiband] [--mb-{low,mid,high}-{threshold,ratio,makeup} ...]
              [--ms] [--ms-side-prop F] [--ms-no-side-denoise] [--ms-width S]
              [--ms-mid-presence DB] [--ms-mid-presence-freq HZ]
              [--wow-flutter] [--wf-max-cents C] [--wf-smoothing MS]
              [--wf-max-freq HZ]
              # Reporting / misc
              [--no-plot] [--no-metrics] [--metrics-csv PATH]
              [--device DEVICE] [--verbose]
```

Reliable input/output formats:

| Category  | Extensions                                   |
|-----------|----------------------------------------------|
| Lossless  | `.wav` `.flac` `.aiff` (16/24/32-bit)        |
| Lossy     | `.mp3` `.m4a` `.aac` `.ogg` `.opus` `.wma`   |
| Any input| all of the above plus anything ffmpeg decodes |

---

## Processing Pipeline

| Phase | Stage                    | Module                              | Enabled by |
|-------|--------------------------|-------------------------------------|------------|
| 1     | Load & pre-process       | `io.preprocessing`                  | always     |
| 1.3   | Wow & flutter correction | `dsp.wow_flutter`                   | `--wow-flutter` |
| 1.5   | Electrical hum removal   | `dsp.dehum`                         | `--dehum HZ` |
| 1.7   | Click / pop / crackle    | `dsp.declicker`                     | `--declick` |
| 2     | Denoising                | `dsp.denoiser`                      | always     |
| 3     | Source separation        | `neural.source_separation` (Demucs) | `--separate` |
| 4     | Super-resolution         | `neural.super_resolution`           | `--super-resolution` |
| 5     | EQ & mastering           | `dsp.equalizer`                     | always     |
| 5.3   | Multiband compression    | `dsp.multiband_compressor`          | `--multiband` |
| 5.7   | M/S stereo processing    | `dsp.ms_processor`                  | `--ms` |
| 5.5   | LUFS loudness            | (pyloudnorm)                        | default on, `--no-lufs` |
| 6     | Metrics & report         | `reporting.metrics`                 | `--no-plot`/`--no-metrics` |

All DSP stages preserve the channel count: mono `(N,)` stays mono and stereo
`(N, 2)` stays stereo (M/S processing is a no-op for mono).

---

## Genre Presets

`--preset` applies an EQ curve tuned for a genre; individual band flags override it.

| Preset      | Bass  | Mid   | Presence | Treble | Air  |
|-------------|-------|-------|----------|--------|------|
| `vinyl`     | +2.5  | −1.5  | +2.5     | +2.0   | +3.5 |
| `jazz`      | +1.0  | 0.0   | +1.0     | +1.5   | +2.0 |
| `classical` | +1.5  | −0.5  | +1.0     | +1.5   | +3.0 |
| `hiphop`    | +5.0  | −2.0  | +1.5     | +1.5   | +2.5 |
| `metal`     | +2.0  | −3.5  | +3.0     | +2.5   | +2.0 |
| `electronic`| +4.0  | −1.0  | +2.0     | +2.5   | +3.0 |
| `podcast`   | 0.0   | −1.0  | +3.5     | +1.5   | +1.0 |
| `flat`      | 0.0   | 0.0   | 0.0      | 0.0    | 0.0  |
| *(default)* | +2.5  | −1.5  | +2.5     | +2.0   | +3.5 |

> `bass` = low-shelf @ 80 Hz · `mid` = peaking @ 250 Hz · `presence` = peaking
> @ 3.5 kHz · `treble` = high-shelf @ 8 kHz · `air` = high-shelf @ 12 kHz.
> A 30 Hz high-pass rumble filter runs before the EQ and can be disabled with
> `--no-rumble-filter`.

---

## Denoising

`--denoise-method` selects the engine:

| Method          | Description                                              | Good for            |
|-----------------|----------------------------------------------------------|---------------------|
| `music` (default)| HPSS separation + Wiener filter (statistical minimum tracking) | vinyl, full music |
| `noisereduce`   | Spectral subtraction (`--stationary` for constant hiss)  | speech, tape hiss   |
| `deepfilternet` | Neural RNN model (48 kHz) — **speech only**              | dialogue, broadcast |
| `wavelet`       | BayesShrink soft thresholding (fast fallback)            | quick/light clean   |
| `auto`          | `music` → `noisereduce` → `wavelet` (first available)    | general purpose     |

Tune with `--prop-decrease` (0.6–0.8 for music), `--n-std-thresh` (lower =
more aggressive for stationary noise), and `--denoise-passes` (2–3 for
stubborn static).

---

## Developer Guide

### Set up a dev environment

```bash
bash setup.sh --dev
source venv_audio/bin/activate
```

### Lint, type-check, test

```bash
ruff check src tests          # lint
ruff format src tests         # format
mypy src/audio_restoration    # type-check
pytest                        # run tests
pytest --cov=audio_restoration # coverage (gate: 60%)
```

All four are wired into the pre-commit hooks (`.pre-commit-config.yaml`) and
the CI workflow (`.github/workflows/ci.yml`).

---

## Project Layout

```
src/audio_restoration/
├── __init__.py          # public API + version
├── __main__.py          # python -m audio_restoration
├── cli.py               # audio-restore entry point
├── config.py            # typed config dataclasses + genre presets
├── constants.py         # global DSP constants
├── exceptions.py        # exception hierarchy
├── pipeline.py          # RestorationPipeline orchestrator
├── dsp/
│   ├── audio_utils.py         # channel-aware helpers (process_channels…)
│   ├── biquad.py              # Audio EQ Cookbook filters
│   ├── declicker.py           # LPC click/scratch repair
│   ├── dehum.py               # 50/60 Hz notch removal
│   ├── denoiser.py            # music / noisereduce / deepfilternet / wavelet
│   ├── equalizer.py           # 5-band mastering EQ + rumble filter
│   ├── ms_processor.py        # Mid/Side stereo processing
│   ├── multiband_compressor.py# 3-band Linkwitz-Riley compressor
│   └── wow_flutter.py         # pitch-instability correction
├── io/
│   ├── format_handler.py      # multi-format read/write (soundfile + ffmpeg)
│   └── preprocessing.py       # load / trim / resample / normalise
├── neural/
│   ├── devices.py             # device resolution (auto/cpu/cuda)
│   ├── source_separation.py   # Demucs stems
│   └── super_resolution.py    # AudioSR (CLI) / scipy upsampling
└── reporting/
    ├── batch_report.py        # per-file CSV summaries (pandas, optional)
    └── metrics.py             # SNR / PSNR / RMS / centroid / HF ratio + plots
tests/                         # 78 pytest tests (unit + small integration)
setup.sh                       # defensive, idempotent environment setup
pyproject.toml                 # packaging, extras, tool config (single source)
```

### Backwards compatibility

`PipelineConfig` still accepts the historical flat keyword arguments
(`denoise_method=`, `bass_gain_db=`, `lufs_target=`, …), mapping them onto the
grouped dataclasses internally:

```python
cfg = PipelineConfig(denoise_method="auto", bass_gain_db=3.0)  # still works
```

---

## Optional Dependencies

| Extra        | Provides                          | Requirements        |
|--------------|-----------------------------------|---------------------|
| (none)       | Core DSP pipeline                 | none                |
| `neural`     | torch, DeepFilterNet, Demucs      | —                   |
| `audiosr`    | diffusion super-resolution        | Python <= 3.10 only |
| `dev`        | pytest, ruff, mypy, build, pandas | —                   |
| ffmpeg       | lossy I/O + broader decode support| system binary       |

Super-resolution gracefully falls back to a high-quality scipy polyphase
resampler when AudioSR is unavailable.

---

## License

MIT. Original work — see the git history for authorship details.