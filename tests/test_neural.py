"""Tests for the optional neural stages (device helpers, Demucs, AudioSR).

None of these tests require torch / demucs / audiosr to be installed: the
heavy dependencies are mocked so the test suite stays CPU-only and fast.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from audio_restoration.exceptions import NeuralModelUnavailableError
from audio_restoration.neural.devices import resolve_device, seed_all
from audio_restoration.neural.source_separation import SourceSeparator
from audio_restoration.neural.super_resolution import SuperResolution

# ---------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_auto_without_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            assert resolve_device("auto") == "cpu"

    def test_auto_with_torch_cpu(self):
        torch_mock = Mock()
        torch_mock.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": torch_mock}):
            assert resolve_device("auto") == "cpu"

    def test_auto_with_cuda(self):
        torch_mock = Mock()
        torch_mock.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": torch_mock}):
            assert resolve_device("auto") == "cuda"

    def test_explicit_cpu(self):
        assert resolve_device("cpu") == "cpu"

    def test_cuda_without_torch_falls_back(self):
        with patch.dict("sys.modules", {"torch": None}):
            assert resolve_device("cuda") == "cpu"

    def test_cuda_unavailable_falls_back(self):
        torch_mock = Mock()
        torch_mock.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": torch_mock}):
            assert resolve_device("cuda") == "cpu"

    def test_cuda_available(self):
        torch_mock = Mock()
        torch_mock.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": torch_mock}):
            assert resolve_device("cuda") == "cuda"


class TestSeedAll:
    def test_seeds_numpy(self):
        with patch.dict("sys.modules", {"torch": None}):
            seed_all(42)
        first = np.random.default_rng(42).integers(0, 100, 5)
        second = np.random.default_rng(42).integers(0, 100, 5)
        assert np.array_equal(first, second)

    def test_with_torch(self):
        torch_mock = Mock()
        torch_mock.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": torch_mock}):
            seed_all(7)
        torch_mock.manual_seed.assert_called_once_with(7)
        torch_mock.cuda.manual_seed_all.assert_called_once_with(7)


# ---------------------------------------------------------------------------
# SourceSeparator
# ---------------------------------------------------------------------------


@pytest.fixture
def stems_dirs(tmp_path) -> list[str]:
    """Create a fake Demucs output tree and return its stem WAV paths."""
    import soundfile as sf

    model_dir = tmp_path / "htdemucs_ft" / "song"
    model_dir.mkdir(parents=True)
    signal = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    paths = {}
    for stem in ("drums", "bass", "other", "vocals"):
        p = model_dir / f"{stem}.wav"
        sf.write(p, signal, 44_100, subtype="PCM_16")
        paths[stem] = p
    return list(paths.values())


def test_separator_invalid_model():
    with pytest.raises(ValueError):
        SourceSeparator(model="bad_model")


def test_separator_unavailable_raises():
    with patch("audio_restoration.neural.source_separation._DEMUCS_AVAILABLE", False):
        sep = SourceSeparator()
        assert not sep.is_available
        with pytest.raises(NeuralModelUnavailableError):
            sep.separate("x.wav")


def test_separate_from_array_writes_temp_then_cleans(tmp_path):
    sep = SourceSeparator()
    audio = np.zeros((100, 2), dtype=np.float32)
    with patch.object(sep, "separate", return_value={"vocals": audio}) as mock_sep:
        result = sep.separate_from_array(audio, 22050)
    assert result == {"vocals": audio}
    mock_sep.assert_called_once()
    temp_file = mock_sep.call_args.args[0]
    import os

    assert not os.path.exists(temp_file)


def test_separate_missing_input_raises(tmp_path):
    with patch("audio_restoration.neural.source_separation._DEMUCS_AVAILABLE", True):
        sep = SourceSeparator()
        with pytest.raises(FileNotFoundError):
            sep.separate(str(tmp_path / "nope.wav"))


def test_separate_end_to_end_mock(tmp_path, wav_file):

    import soundfile as sf

    with patch("audio_restoration.neural.source_separation._DEMUCS_AVAILABLE", True):
        sep = SourceSeparator()

        # Build a fake Demucs output tree mirroring what _load_stems expects:
        # output_dir/<model>/<track_name>/*.wav
        demucs_root = tmp_path / "htdemucs_ft" / "stereo"
        demucs_root.mkdir(parents=True)
        signal = np.linspace(-1, 1, 20, dtype=np.float32)
        for stem in ("drums", "bass", "other", "vocals"):
            sf.write(demucs_root / f"{stem}.wav", signal, 44_100, subtype="PCM_16")

        with patch("audio_restoration.neural.source_separation.demucs.separate.main"):
            result = sep.separate(wav_file, output_dir=str(tmp_path))
    assert set(result) == {"drums", "bass", "other", "vocals"}


def test_separate_keeps_temp_cleanup_on_error(tmp_path, wav_file):
    """A failure inside Demucs must still remove the temp output dir."""
    with patch("audio_restoration.neural.source_separation._DEMUCS_AVAILABLE", True):
        sep = SourceSeparator()
        with (
            patch(
                "audio_restoration.neural.source_separation.demucs.separate.main",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(RuntimeError),
        ):
            sep.separate(wav_file)


def test_get_instrumental_excludes_vocals(tmp_path, wav_file):
    sep = SourceSeparator()
    stems = {
        name: np.zeros((50, 2), dtype=np.float32)
        for name in ("drums", "bass", "other", "vocals")
    }
    with patch.object(sep, "separate", return_value=stems):
        mixed, sr = sep.get_instrumental(wav_file)
    assert sr == 44_100
    assert mixed.shape == (50, 2)


def test_get_instrumental_all_excluded_raises(tmp_path, wav_file):
    sep = SourceSeparator()
    stems = {
        name: np.zeros((50, 2), dtype=np.float32)
        for name in ("drums", "bass", "other", "vocals")
    }
    with patch.object(sep, "separate", return_value=stems), pytest.raises(ValueError):
        sep.get_instrumental(wav_file, stems_to_exclude=list(stems))


def test_save_stems(tmp_path):
    sep = SourceSeparator()
    stems = {"vocals": np.zeros((10, 2), dtype=np.float32)}
    sep.save_stems(stems, str(tmp_path))
    assert (tmp_path / "vocals.wav").is_file()


def test_load_stems_missing_dir_raises(tmp_path):
    sep = SourceSeparator()
    with pytest.raises(RuntimeError):
        sep._load_stems("/virtual/song.wav", str(tmp_path))


def test_run_demucs_builds_args_and_calls(tmp_path):
    sep = SourceSeparator(segment=7)
    main_mock = Mock()
    with patch(
        "audio_restoration.neural.source_separation.demucs.separate.main", main_mock
    ):
        sep._run_demucs("/in.wav", "/out")
    args = main_mock.call_args.args[0]
    assert "--segment" in args and "7" in args


# ---------------------------------------------------------------------------
# SuperResolution
# ---------------------------------------------------------------------------


def test_super_resolution_no_op_when_already_target():
    sr_mod = SuperResolution(target_sr=22050, device="cpu")
    audio = np.zeros((100, 2), dtype=np.float32)
    out, sr = sr_mod.upsample(audio, 22050)
    assert sr == 22050
    assert np.array_equal(out, audio)


def test_upsample_scipy_fallback():
    sr_mod = SuperResolution(target_sr=44100, device="cpu")
    with patch.object(sr_mod, "_audiosr_available", False):
        audio = np.zeros((100, 2), dtype=np.float32)
        out, sr = sr_mod.upsample(audio, 22050)
    assert sr == 44100
    assert out.shape[0] == 200
    assert out.shape[1] == 2


def test_upsample_audiosr_success_mono():
    sr_mod = SuperResolution(target_sr=44100, device="cpu")
    with (
        patch.object(sr_mod, "_audiosr_available", True),
        patch.object(
            sr_mod, "_audiosr_channel", return_value=np.zeros(400, dtype=np.float32)
        ) as mock_ch,
    ):
        out, sr = sr_mod.upsample(np.zeros((200,), dtype=np.float32), 22050)
    assert sr == 44100
    assert out.shape == (400,)
    mock_ch.assert_called_once()


def test_upsample_audiosr_fallback_on_failure():
    sr_mod = SuperResolution(target_sr=44100, device="cpu")
    with (
        patch.object(sr_mod, "_audiosr_available", True),
        patch.object(sr_mod, "_upsample_audiosr", side_effect=RuntimeError("boom")),
    ):
        out, sr = sr_mod.upsample(np.zeros((200, 2), dtype=np.float32), 22050)
    assert sr == 44100
    assert out.shape == (400, 2)


def test_audiosr_channel_processes_via_cli(tmp_path):
    import os

    import soundfile as real_sf

    sr_mod = SuperResolution(target_sr=44100, device="cpu")

    def fake_run(*args, **kwargs):
        cmd = args[0]
        output_dir = cmd[cmd.index("-s") + 1]
        os.makedirs(output_dir, exist_ok=True)
        real_sf.write(
            os.path.join(output_dir, "input_audiosr.wav"),
            np.zeros(1000, dtype=np.float32),
            44100,
            subtype="PCM_16",
        )
        return Mock(returncode=0, stderr="")

    with (
        patch("subprocess.run", side_effect=fake_run) as mock_run,
        patch("audio_restoration.neural.super_resolution.sf.read") as mock_read,
    ):
        mock_read.return_value = (np.linspace(-1, 1, 1000, dtype=np.float32), 44100)
        out = sr_mod._audiosr_channel(np.zeros(500, dtype=np.float32), 22050)
    assert out.shape == (1000,)
    mock_run.assert_called_once()


def test_audiosr_channel_nonzero_returncode_raises(tmp_path):
    sr_mod = SuperResolution(target_sr=44100, device="cpu")
    fake_result = Mock(returncode=1, stderr="boom")
    with (
        patch("subprocess.run", return_value=fake_result),
        patch("audio_restoration.neural.super_resolution.sf.write"),
        pytest.raises(RuntimeError),
    ):
        sr_mod._audiosr_channel(np.zeros(500, dtype=np.float32), 22050)


def test_audiosr_channel_no_output_raises(tmp_path):
    sr_mod = SuperResolution(target_sr=44100, device="cpu")
    fake_result = Mock(returncode=0, stderr="")
    with (
        patch("subprocess.run", return_value=fake_result),
        patch("audio_restoration.neural.super_resolution.sf.write"),
        pytest.raises(RuntimeError),
    ):
        sr_mod._audiosr_channel(np.zeros(500, dtype=np.float32), 22050)


def test_check_audiosr_detects_cli(monkeypatch):
    monkeypatch.setattr(
        "audio_restoration.neural.super_resolution.shutil.which",
        lambda _cmd: "/usr/bin/audiosr",
    )
    assert SuperResolution._check_audiosr() is True


def test_check_audiosr_not_found(monkeypatch):
    monkeypatch.setattr(
        "audio_restoration.neural.super_resolution.shutil.which", lambda _cmd: None
    )
    assert SuperResolution._check_audiosr() is False
