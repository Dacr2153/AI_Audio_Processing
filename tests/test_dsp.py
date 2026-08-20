"""Tests for the DSP processing stages (denoiser, declicker, EQ, dehum…)."""

from __future__ import annotations

import numpy as np

from audio_restoration.config import MSConfig
from audio_restoration.dsp import (
    AudioEqualizer,
    Declicker,
    Dehummer,
    Denoiser,
    MSProcessor,
    MultibandCompressor,
    WowFlutterCorrector,
)


def test_denoiser_preserves_length_and_shape(noisy_signal, sample_rate):
    denoiser = Denoiser(method="wavelet")
    out = denoiser.denoise(noisy_signal, sample_rate)
    assert out.shape == noisy_signal.shape
    assert out.dtype == noisy_signal.dtype


def test_denoiser_wavelet_removes_noise(noisy_signal, sample_rate):
    out = Denoiser(method="wavelet").denoise(noisy_signal, sample_rate)
    # The denoised result should be quieter than the noisy input.
    assert _rms_db(out) < _rms_db(noisy_signal)


def test_denoiser_stereo_preserves_channels(stereo_signal, sample_rate):
    out = Denoiser(method="wavelet").denoise(stereo_signal, sample_rate)
    assert out.ndim == 2
    assert out.shape == stereo_signal.shape


def test_declicker_passthrough_on_clean(mono_signal, sample_rate):
    out = Declicker().process(mono_signal, sample_rate)
    assert out.shape == mono_signal.shape
    assert np.allclose(out, mono_signal, atol=1e-4)  # clean audio is untouched


def test_declicker_removes_click(mono_signal, sample_rate):
    signal = mono_signal.copy()
    signal[1000:1005] = 2.0  # impulse well above local RMS
    out = Declicker(threshold=4.0).process(signal, sample_rate)
    repaired = out[995:1010]
    assert np.max(np.abs(repaired)) < 1.5


def test_equalizer_flat_gain_passthrough(mono_signal, sample_rate):
    eq = AudioEqualizer(
        bass_gain_db=0.0,
        mid_gain_db=0.0,
        presence_gain_db=0.0,
        treble_gain_db=0.0,
        air_gain_db=0.0,
        rumble_filter=False,
    )
    out = eq.process(mono_signal, sample_rate, normalize=False, limit=False)
    assert out.shape == mono_signal.shape
    # DC-steady filter should not explode amplitude
    assert np.max(np.abs(out)) < 1.5


def test_equalizer_stereo_shape(stereo_signal, sample_rate):
    eq = AudioEqualizer()
    out = eq.process(stereo_signal, sample_rate)
    assert out.shape == stereo_signal.shape


def test_rumble_filter_attenuates_low_frequencies(sample_rate):
    t = np.arange(sample_rate) / sample_rate
    low_freq = 0.1 * np.sin(2 * np.pi * 15.0 * t).astype(np.float32)  # sub-30 Hz rumble
    eq = AudioEqualizer(rumble_filter=True)
    out = eq.process(low_freq, sample_rate, normalize=False, limit=False)
    assert _rms_db(out) < _rms_db(low_freq)  # rumble reduced


def test_dehummer_removes_hum_at_fundamental(sample_rate):
    t = np.arange(sample_rate) / sample_rate
    hum = 0.5 * np.sin(2 * np.pi * 50.0 * t).astype(np.float32)
    out = Dehummer(freq=50.0, harmonics=1).process(hum, sample_rate)
    # filtfilt adds edge transients; measure steady-state middle of the signal.
    mid = slice(len(out) // 4, 3 * len(out) // 4)
    assert _rms_db(out[mid]) < _rms_db(hum[mid]) - 20.0  # deep notch attenuation


def test_dehummer_stereo(sample_rate):
    t = np.arange(sample_rate) / sample_rate
    hum = 0.3 * np.sin(2 * np.pi * 50.0 * t)
    stereo = np.column_stack([hum, hum]).astype(np.float32)
    out = Dehummer(freq=50.0).process(stereo, sample_rate)
    assert out.shape == stereo.shape


def test_declicker_and_wow_do_not_crash(mono_signal, sample_rate):
    wfc = WowFlutterCorrector(max_deviation_cents=20.0)
    out = wfc.process(mono_signal, sample_rate)
    assert out.shape == mono_signal.shape
    assert np.all(np.isfinite(out))


def test_multiband_compressor_passthrough_shape(stereo_signal, sample_rate):
    mbc = MultibandCompressor()
    out = mbc.process(stereo_signal, sample_rate)
    assert out.shape == stereo_signal.shape
    assert np.all(np.isfinite(out))


def test_ms_processor_reduces_width_when_narrowed(stereo_signal, sample_rate):
    proc = MSProcessor(MSConfig(side_gain=0.1, side_denoise=False))
    out = proc.process(stereo_signal, sample_rate)
    assert out.shape == stereo_signal.shape
    # A narrower stereo image → the two channels become more similar.
    corr_before = _corr(stereo_signal[:, 0], stereo_signal[:, 1])
    corr_after = _corr(out[:, 0], out[:, 1])
    assert corr_after > corr_before


def test_ms_processor_is_noop_for_mono(mono_signal, sample_rate):
    proc = MSProcessor(MSConfig())
    out = proc.process(mono_signal, sample_rate)
    assert np.array_equal(out, mono_signal)


def _rms_db(x):
    x = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.mean(x**2))
    if rms < 1e-12:
        return -120.0
    return 20.0 * np.log10(rms)


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(a * b) / denom)
