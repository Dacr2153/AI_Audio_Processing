"""Typed configuration objects for the restoration pipeline.

Configuration is split into small, focused dataclasses (one per processing
domain). :class:`PipelineConfig` composes them and — for backwards
compatibility — still accepts the historical flat keyword arguments used by
earlier versions of the library (e.g. ``denoise_method=``, ``bass_gain_db=``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constants import DEEPFILTER_SR, DEMUCS_MODELS

# ---------------------------------------------------------------------------
# Genre EQ presets (gains in dB)
# ---------------------------------------------------------------------------

#: One 5-band EQ curve per genre. Individual band flags override these.
GENRE_PRESETS: dict[str, dict[str, float]] = {
    "vinyl": {
        "bass_gain_db": 2.5,
        "mid_gain_db": -1.5,
        "presence_gain_db": 2.5,
        "treble_gain_db": 2.0,
        "air_gain_db": 3.5,
    },
    "jazz": {
        "bass_gain_db": 1.0,
        "mid_gain_db": 0.0,
        "presence_gain_db": 1.0,
        "treble_gain_db": 1.5,
        "air_gain_db": 2.0,
    },
    "classical": {
        "bass_gain_db": 1.5,
        "mid_gain_db": -0.5,
        "presence_gain_db": 1.0,
        "treble_gain_db": 1.5,
        "air_gain_db": 3.0,
    },
    "hiphop": {
        "bass_gain_db": 5.0,
        "mid_gain_db": -2.0,
        "presence_gain_db": 1.5,
        "treble_gain_db": 1.5,
        "air_gain_db": 2.5,
    },
    "metal": {
        "bass_gain_db": 2.0,
        "mid_gain_db": -3.5,
        "presence_gain_db": 3.0,
        "treble_gain_db": 2.5,
        "air_gain_db": 2.0,
    },
    "electronic": {
        "bass_gain_db": 4.0,
        "mid_gain_db": -1.0,
        "presence_gain_db": 2.0,
        "treble_gain_db": 2.5,
        "air_gain_db": 3.0,
    },
    "podcast": {
        "bass_gain_db": 0.0,
        "mid_gain_db": -1.0,
        "presence_gain_db": 3.5,
        "treble_gain_db": 1.5,
        "air_gain_db": 1.0,
    },
    "flat": {
        "bass_gain_db": 0.0,
        "mid_gain_db": 0.0,
        "presence_gain_db": 0.0,
        "treble_gain_db": 0.0,
        "air_gain_db": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Per-domain configuration
# ---------------------------------------------------------------------------


@dataclass
class DenoiseConfig:
    """Denoising algorithm settings."""

    method: str = "music"
    """Denoiser method: 'auto', 'music', 'noisereduce', 'deepfilternet', 'wavelet'."""

    stationary: bool = False
    """For noisereduce only: assume stationary noise (constant hiss)."""

    prop_decrease: float = 0.85
    """Noise reduction aggressiveness (0–1)."""

    n_std_thresh: float = 1.5
    """For noisereduce stationary: std-devs above noise mean treated as signal."""

    passes: int = 1
    """Number of sequential denoising passes."""


@dataclass
class DeclickConfig:
    """Click / pop / scratch removal settings."""

    enabled: bool = False
    threshold: float = 4.0
    margin_ms: float = 5.0
    lpc_order: int = 32
    max_click_ms: float = 30.0
    max_scratch_ms: float = 200.0


@dataclass
class DehumConfig:
    """Electrical hum (50/60 Hz) notch-filter settings."""

    freq: float | None = None
    """Hum fundamental: 50 (Europe/Asia) or 60 (Americas). None = disabled."""

    harmonics: int = 5
    """Number of harmonics to notch (including the fundamental)."""

    q: float = 35.0
    """Notch filter Q. Higher = narrower notch, less musical colouration."""


@dataclass
class EQConfig:
    """5-band mastering EQ + rumble filter settings (gains in dB)."""

    bass_gain_db: float = 2.5
    mid_gain_db: float = -1.5
    presence_gain_db: float = 2.5
    treble_gain_db: float = 2.0
    air_gain_db: float = 3.5
    rumble_filter: bool = True
    """Sub-sonic HPF @ 30 Hz to remove turntable rumble."""


@dataclass
class MultibandConfig:
    """3-band Linkwitz-Riley compressor settings."""

    enabled: bool = False
    xover_low: float = 250.0
    xover_high: float = 4000.0

    low_threshold_db: float = -20.0
    low_ratio: float = 2.5
    low_makeup_db: float = 1.5

    mid_threshold_db: float = -18.0
    mid_ratio: float = 3.0
    mid_makeup_db: float = 1.0

    high_threshold_db: float = -16.0
    high_ratio: float = 2.0
    high_makeup_db: float = 0.5


@dataclass
class MSConfig:
    """Mid/Side stereo processing settings."""

    enabled: bool = False
    side_denoise: bool = False
    """Apply noise reduction to the Side channel."""

    side_prop_decrease: float = 0.6
    """Noise reduction aggressiveness on the Side channel (0–1)."""

    side_gain: float = 1.0
    """Side channel amplitude scale: <1 narrower, >1 wider."""

    mid_presence_db: float = 0.0
    """Presence EQ boost/cut on the Mid channel (0 = disabled)."""

    mid_presence_freq: float = 3500.0
    """Centre frequency for the Mid presence EQ in Hz."""


@dataclass
class WowFlutterConfig:
    """Wow & flutter pitch-correction settings."""

    enabled: bool = False
    frame_ms: float = 50.0
    hop_ms: float = 10.0
    max_cents: float = 100.0
    """Maximum pitch correction in cents (100 = 1 semitone)."""

    smoothing_ms: float = 200.0
    max_freq_hz: float = 100.0
    """Upper bound of fluctuations to correct in Hz."""


@dataclass
class SourceSeparationConfig:
    """Demucs source separation settings."""

    enabled: bool = False
    model: str = "htdemucs_ft"
    device: str = "auto"
    stem_denoise: bool = False
    """Denoise each stem independently before remixing."""


@dataclass
class SuperResolutionConfig:
    """Super-resolution settings."""

    enabled: bool = False
    target_sr: int = DEEPFILTER_SR
    device: str = "auto"


@dataclass
class LoudnessConfig:
    """LUFS loudness normalisation settings."""

    target_lufs: float | None = -14.0
    """Target integrated loudness in LUFS. None = disabled."""


@dataclass
class OutputConfig:
    """Output encoding / mastering settings."""

    sample_rate: int | None = None
    bitrate: str | None = None
    bit_depth: int = 24
    vbr: bool = True
    normalize: bool = True
    limit: bool = True
    limit_threshold_db: float = -0.3


@dataclass
class ReportConfig:
    """Metrics / reporting settings."""

    print_metrics: bool = True
    save_comparison_plot: bool = True
    metrics_summary_csv: str | None = None
    """Write a batched per-file metrics summary CSV to this path."""


# ---------------------------------------------------------------------------
# Composite pipeline configuration
# ---------------------------------------------------------------------------

#: Maps legacy flat keyword arguments to (sub-config attribute, field name).
#: Kept so the documented ``PipelineConfig(**kwargs)`` API keeps working.
_LEGACY_FIELD_MAP: dict[str, tuple[str, str]] = {
    # Denoising
    "denoise_method": ("denoise", "method"),
    "denoise_stationary": ("denoise", "stationary"),
    "denoise_prop_decrease": ("denoise", "prop_decrease"),
    "denoise_n_std_thresh": ("denoise", "n_std_thresh"),
    "denoise_passes": ("denoise", "passes"),
    # Declicking
    "enable_declicker": ("declick", "enabled"),
    "declicker_threshold": ("declick", "threshold"),
    "declicker_margin_ms": ("declick", "margin_ms"),
    "declicker_lpc_order": ("declick", "lpc_order"),
    "declicker_max_click_ms": ("declick", "max_click_ms"),
    "declicker_max_scratch_ms": ("declick", "max_scratch_ms"),
    # Dehum
    "dehum_freq": ("dehum", "freq"),
    "dehum_harmonics": ("dehum", "harmonics"),
    "dehum_q": ("dehum", "q"),
    # EQ
    "bass_gain_db": ("eq", "bass_gain_db"),
    "mid_gain_db": ("eq", "mid_gain_db"),
    "presence_gain_db": ("eq", "presence_gain_db"),
    "treble_gain_db": ("eq", "treble_gain_db"),
    "air_gain_db": ("eq", "air_gain_db"),
    "eq_rumble_filter": ("eq", "rumble_filter"),
    # Multiband
    "enable_multiband": ("multiband", "enabled"),
    "eq_crossover_low": ("multiband", "xover_low"),
    "eq_crossover_high": ("multiband", "xover_high"),
    "multiband_low_threshold_db": ("multiband", "low_threshold_db"),
    "multiband_low_ratio": ("multiband", "low_ratio"),
    "multiband_low_makeup_db": ("multiband", "low_makeup_db"),
    "multiband_mid_threshold_db": ("multiband", "mid_threshold_db"),
    "multiband_mid_ratio": ("multiband", "mid_ratio"),
    "multiband_mid_makeup_db": ("multiband", "mid_makeup_db"),
    "multiband_high_threshold_db": ("multiband", "high_threshold_db"),
    "multiband_high_ratio": ("multiband", "high_ratio"),
    "multiband_high_makeup_db": ("multiband", "high_makeup_db"),
    # M/S
    "enable_ms": ("ms", "enabled"),
    "ms_side_denoise": ("ms", "side_denoise"),
    "ms_side_prop_decrease": ("ms", "side_prop_decrease"),
    "ms_side_gain": ("ms", "side_gain"),
    "ms_mid_presence_db": ("ms", "mid_presence_db"),
    "ms_mid_presence_freq": ("ms", "mid_presence_freq"),
    # Wow & flutter
    "enable_wow_flutter": ("wow_flutter", "enabled"),
    "wow_flutter_frame_ms": ("wow_flutter", "frame_ms"),
    "wow_flutter_hop_ms": ("wow_flutter", "hop_ms"),
    "wow_flutter_max_cents": ("wow_flutter", "max_cents"),
    "wow_flutter_smoothing_ms": ("wow_flutter", "smoothing_ms"),
    "wow_flutter_max_freq_hz": ("wow_flutter", "max_freq_hz"),
    # Source separation
    "enable_source_separation": ("separate", "enabled"),
    "demucs_model": ("separate", "model"),
    "demucs_device": ("separate", "device"),
    "stem_denoise": ("separate", "stem_denoise"),
    # Super-resolution
    "enable_super_resolution": ("sr", "enabled"),
    "sr_target_sr": ("sr", "target_sr"),
    "super_resolution_device": ("sr", "device"),
    # Loudness
    "lufs_target": ("loudness", "target_lufs"),
    # Output
    "output_sample_rate": ("output", "sample_rate"),
    "output_bitrate": ("output", "bitrate"),
    "output_bit_depth": ("output", "bit_depth"),
    "output_vbr": ("output", "vbr"),
    "normalize_output": ("output", "normalize"),
    "limit_output": ("output", "limit"),
    "limit_threshold_db": ("output", "limit_threshold_db"),
    # Reporting
    "save_comparison_plot": ("report", "save_comparison_plot"),
    "print_metrics": ("report", "print_metrics"),
    "metrics_summary_csv": ("report", "metrics_summary_csv"),
}

#: Rough constant used to detect "still at default" for genre application.
_DEFAULT_EQ = EQConfig()


@dataclass
class PipelineConfig:
    """Complete configuration for :class:`audio_restoration.pipeline.RestorationPipeline`.

    Each processing domain is grouped into a dedicated dataclass. The
    constructor also accepts the historical flat keyword arguments (see
    ``_LEGACY_FIELD_MAP``) so existing callers keep working unchanged::

        config = PipelineConfig(
            denoise_method="auto",
            bass_gain_db=2.0,
            enable_wow_flutter=True,
        )
    """

    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    declick: DeclickConfig = field(default_factory=DeclickConfig)
    dehum: DehumConfig = field(default_factory=DehumConfig)
    eq: EQConfig = field(default_factory=EQConfig)
    multiband: MultibandConfig = field(default_factory=MultibandConfig)
    ms: MSConfig = field(default_factory=MSConfig)
    wow_flutter: WowFlutterConfig = field(default_factory=WowFlutterConfig)
    separate: SourceSeparationConfig = field(default_factory=SourceSeparationConfig)
    sr: SuperResolutionConfig = field(default_factory=SuperResolutionConfig)
    loudness: LoudnessConfig = field(default_factory=LoudnessConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    genre: str | None = None
    """Genre preset used to derive EQ defaults (see GENRE_PRESETS)."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        denoise: DenoiseConfig | None = None,
        declick: DeclickConfig | None = None,
        dehum: DehumConfig | None = None,
        eq: EQConfig | None = None,
        multiband: MultibandConfig | None = None,
        ms: MSConfig | None = None,
        wow_flutter: WowFlutterConfig | None = None,
        separate: SourceSeparationConfig | None = None,
        sr: SuperResolutionConfig | None = None,
        loudness: LoudnessConfig | None = None,
        output: OutputConfig | None = None,
        report: ReportConfig | None = None,
        genre: str | None = None,
        **legacy: object,
    ) -> None:
        self.denoise = denoise or DenoiseConfig()
        self.declick = declick or DeclickConfig()
        self.dehum = dehum or DehumConfig()
        self.eq = eq or EQConfig()
        self.multiband = multiband or MultibandConfig()
        self.ms = ms or MSConfig()
        self.wow_flutter = wow_flutter or WowFlutterConfig()
        self.separate = separate or SourceSeparationConfig()
        self.sr = sr or SuperResolutionConfig()
        self.loudness = loudness or LoudnessConfig()
        self.output = output or OutputConfig()
        self.report = report or ReportConfig()
        self.genre = genre

        self._from_legacy(legacy)
        self.apply_genre(genre)
        self.validate()

    def _from_legacy(self, legacy: dict[str, object]) -> None:
        configs: dict[str, object] = {
            "denoise": self.denoise,
            "declick": self.declick,
            "dehum": self.dehum,
            "eq": self.eq,
            "multiband": self.multiband,
            "ms": self.ms,
            "wow_flutter": self.wow_flutter,
            "separate": self.separate,
            "sr": self.sr,
            "loudness": self.loudness,
            "output": self.output,
            "report": self.report,
        }
        for key, value in legacy.items():
            mapping = _LEGACY_FIELD_MAP.get(key)
            if mapping is None:
                raise TypeError(f"Unknown PipelineConfig option: {key!r}")
            group, field_name = mapping
            setattr(configs[group], field_name, value)

    def apply_genre(self, genre: str | None) -> None:
        """Apply a genre preset to EQ values still at their built-in defaults.

        Explicit values already set elsewhere are never overwritten.
        """
        self.genre = genre
        preset = GENRE_PRESETS.get(genre) if genre else None
        if not preset:
            return
        for key, value in preset.items():
            if getattr(self.eq, key) == getattr(_DEFAULT_EQ, key):
                setattr(self.eq, key, value)

    def validate(self) -> None:
        """Cross-field validation. Raises :class:`ValidationError` on bad input."""
        from .exceptions import ValidationError

        denoisers = {"auto", "music", "noisereduce", "deepfilternet", "wavelet"}
        if self.denoise.method not in denoisers:
            raise ValidationError(
                f"denoise.method must be one of {sorted(denoisers)}, got {self.denoise.method!r}"
            )
        if not 0.0 <= self.denoise.prop_decrease <= 1.0:
            raise ValidationError(
                f"denoise.prop_decrease must be in [0, 1], got {self.denoise.prop_decrease}"
            )
        if self.denoise.passes < 1:
            raise ValidationError("denoise.passes must be >= 1")
        if self.separate.model not in DEMUCS_MODELS:
            raise ValidationError(
                f"separate.model must be one of {DEMUCS_MODELS}, got {self.separate.model!r}"
            )
        if self.declick.threshold <= 0:
            raise ValidationError("declick.threshold must be > 0")
        if self.multiband.xover_low >= self.multiband.xover_high:
            raise ValidationError("multiband.xover_low must be < multiband.xover_high")
