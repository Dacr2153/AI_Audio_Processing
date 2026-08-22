"""About screen.

Shows version, neural-dependency status, license and repository info.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from audio_restoration import __version__
from audio_restoration.dsp.denoiser import deepfilternet_available
from audio_restoration.neural.source_separation import SourceSeparator
from audio_restoration.neural.super_resolution import SuperResolution

from .. import i18n
from .base import TuiScreen


class AboutScreen(TuiScreen):
    """Version, dependency status and licence information."""

    TITLE_KEY = "nav.about"

    DEFAULT_CSS = """
    AboutScreen .section-label {
        text-style: bold;
        color: $text-secondary;
        height: 1;
        margin-bottom: 0;
    }
    AboutScreen .version-panel {
        background: $surface;
        border: solid $border;
        padding: 1 2;
        margin-bottom: 1;
        height: auto;
    }
    AboutScreen .deps-panel {
        background: $surface;
        border: solid $border;
        padding: 0 1;
        margin-bottom: 1;
        height: auto;
    }
    AboutScreen .deps-panel Static {
        height: 1;
    }
    AboutScreen .license-panel {
        background: $surface;
        border: solid $border;
        padding: 0 1;
        margin-bottom: 1;
        height: auto;
    }
    AboutScreen .repo-panel {
        background: $surface;
        border: solid $border;
        padding: 0 1;
        height: auto;
    }
    """

    def form(self) -> ComposeResult:
        with Vertical(classes="version-panel"):
            yield Static(
                f"{i18n.t('about.version')}: {__version__}",
                id="about-version",
            )
        with Vertical(classes="deps-panel"):
            yield Static(i18n.t("about.deps"), classes="section-label")
            yield Static(self._dep_line("DeepFilterNet", deepfilternet_available()),
                         id="about-deepfilter")
            yield Static(self._dep_line("Demucs", SourceSeparator().is_available),
                         id="about-demucs")
            yield Static(self._dep_line("AudioSR", SuperResolution().is_audiosr_available),
                         id="about-audiosr")
        with Vertical(classes="license-panel"):
            yield Static(i18n.t("about.license"), id="about-license")
        with Vertical(classes="repo-panel"):
            yield Static("https://github.com/Dacr2153/AI_Audio_Processing",
                         id="about-repo")

    @staticmethod
    def _dep_line(name: str, ok: bool) -> str:
        status = i18n.t("about.neural_ok") if ok else i18n.t("about.neural_missing")
        return f"  {name}: {status}"

    def refresh_labels(self) -> None:
        super().refresh_labels()
        self.query_one("#about-version", Static).update(
            f"{i18n.t('about.version')}: {__version__}"
        )
        self.query_one("#about-deepfilter", Static).update(
            self._dep_line("DeepFilterNet", deepfilternet_available())
        )
        self.query_one("#about-demucs", Static).update(
            self._dep_line("Demucs", SourceSeparator().is_available)
        )
        self.query_one("#about-audiosr", Static).update(
            self._dep_line("AudioSR", SuperResolution().is_audiosr_available)
        )
        self.query_one("#about-license", Static).update(i18n.t("about.license"))
