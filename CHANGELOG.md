# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- TUI (Textual sidecar) foundation: bilingual EN/ES shell, custom dark/light
  themes, sidebar navigation and shared state/profiles/history (Fase 1).
- TUI Fase 2: interactive Home screen with quick-action buttons and a real
  screen registry (per-section panels replace the static placeholder).
- TUI Fase 3: single-file restoration screen with file pickers (DirectoryTree
  modal), pipeline run via async worker, and inline quality-metrics report.
- TUI Fase 4: batch restoration screen with input/output folder pickers,
  extension/suffix/workers options, DataTable progress per file and summary.
- TUI Fase 5: profiles management screen with list/load/delete/save
  pipeline-configuration presets.
- TUI Fase 6: history screen showing past restorations with timestamps, file
  paths and key metrics, plus clear action.
- TUI Fase 7: about screen with version, neural-dependency status (DeepFilterNet,
  Demucs, AudioSR) and license information.

### Changed
- Added `LICENSE` (MIT), raised minimum test coverage to 80 % and added a
  `package_build` CI job that builds and inspects the wheel/sdist.
- Version is now single-sourced from `src/audio_restoration/_version.py`.

## [2.0.0] - 2026

### Added
- Installable `src/` layout package `audio-restoration` with `audio-restore` CLI.
- Stereo preservation end to end (loading, DSP, neural stages and encoding).
- RESTful pipeline phases: denoise, declick, dehum, EQ, multiband compression,
  M/S processing, wow/flutter, source separation (Demucs), super-resolution,
  loudness normalization and quality metrics reporting.
- Optional neural extras (`torch`/`demucs`/`deepfilternet`, `audiosr`) with
  graceful fallbacks when unavailable.
- `setup.sh` installer (idempotent), CI workflow, pre-commit hooks.

### Changed
- Migrated from flat legacy modules to an installable package under `src/`.
- `numpy` is pinned to `<2.5` to stay compatible with `librosa`/`numba`.

### Removed
- Legacy top-level scripts replaced by the packaged module.

## [1.0.0] - Legacy

- Pre-migration functionality (flat scripts, restored audio from tape/vinyl).