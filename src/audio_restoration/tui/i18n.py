"""Lightweight bilingual (EN/ES) internationalisation for the TUI.

The package ships its CLI/docs in English; the TUI adds a runtime language
switcher (``L`` key) between English and Spanish without pulling in a heavy
i18n dependency.  Fallback is always English.
"""

from __future__ import annotations

from typing import Literal

Language = Literal["en", "es"]

#: English → Spanish string table.  Only the subset used by the TUI.
_TRANSLATIONS: dict[str, dict[str, str]] = {
    "es": {
        # Generic
        "generic.cancel": "Cancelar",
        "generic.confirm": "Confirmar",
        "generic.select_first": "Selecciona un elemento primero",
        "generic.select_input": "Seleccionar archivo de entrada",
        "generic.select_output": "Seleccionar archivo de salida",
        "generic.pick_file": "Elegir Archivo",
        "error.unsupported_format": "Formato no compatible",
        # Navigation
        "aria.sidebar": "Navegación",
        "nav.home": "Inicio",
        "nav.single": "Archivo único",
        "nav.batch": "Lote",
        "nav.profiles": "Perfiles",
        "nav.history": "Historial",
        "nav.about": "Acerca de",
        # Header / footer
        "header.subtitle": "Restauración profesional de audio",
        # Home / welcome
        "welcome.title": "¡Bienvenido a la restauración de audio!",
        "welcome.intro": (
            "Restauración profesional de audio: DSP clásico + modelos neurales."
        ),
        "welcome.single": "Restaurar un único archivo",
        "welcome.batch": "Restaurar una carpeta completa",
        "welcome.presets": "Perfiles guardados",
        "welcome.history": "Ver el historial de trabajos",
        "welcome.about": "Información del programa",
        "welcome.tip": "Pulsa L para cambiar de idioma (EN/ES) y D para el tema.",
        # Dashboard
        "dashboard.quick_actions": "Acciones rápidas",
        "dashboard.recent": "Actividad reciente",
        "dashboard.recent_empty": "Aún no hay trabajos. Empieza restaurando un archivo.",
        "dashboard.active_profile": "Perfil activo",
        "dashboard.profile_none": "Usando configuración por defecto",
        "dashboard.status_restored": "Restaurado",
        "dashboard.status_processing": "Procesando",
        "dashboard.status_failed": "Fallido",
        # Error handling
        "error.not_found": "No se encontró el archivo",
        "error.batch_empty": "No hay archivos de audio en la carpeta",
        "error.pipeline_failed": "El procesamiento falló",
        "ok.saved": "Guardado",
        # Processing
        "proc.title": "Procesando…",
        "proc.phase": "Fase",
        "proc.elapsed": "Tiempo transcurrido",
        "proc.done": "¡Completado!",
        "proc.cancel_hint": "Pulsa Ctrl+S para detener.",
        # Results
        "results.title": "Resultados",
        "results.hint": "Procesa un archivo para ver las métricas aquí.",
        "results.metric": "Métrica",
        "results.original": "Original",
        "results.restored": "Restaurado",
        "results.snr": "SNR vs original (dB)",
        "results.psnr": "PSNR (dB)",
        "results.original_rms": "RMS original (dB)",
        "results.restored_rms": "RMS restaurado (dB)",
        "results.save_plot": "Guardar gráfica PNG",
        "results.saved_plot": "Gráfica guardada en",
        # Batch
        "batch.title": "Procesamiento en lote",
        "batch.folder": "Carpeta de entrada",
        "batch.output_dir": "Carpeta de salida",
        "batch.ext": "Extensión de salida",
        "batch.suffix": "Sufijo de salida",
        "batch.workers": "Trabajadores en paralelo",
        "batch.start": "Iniciar lote",
        "batch.pending": "Pendiente",
        "batch.processing": "Procesando",
        "batch.ok": "OK",
        "batch.failed": "Fallido",
        "batch.complete": "Lote completado",
        "batch.summary": "Resumen",
        "batch.succeeded": "éxitos",
        "batch.failures": "fallos",
        # Config
        "config.title": "Configuración",
        "config.preset": "Preset de género",
        "config.save_profile": "Guardar perfil",
        "config.profile_name": "Nombre del perfil",
        "config.run_single": "Restaurar archivo",
        "config.run_batch": "Restaurar carpeta",
        "config.group.denoise": "Denoising",
        "config.group.declick": "Declicking",
        "config.group.dehum": "Dehum (50 Hz)",
        "config.group.eq": "Ecualizador",
        "config.group.multiband": "Compresor multibanda",
        "config.group.ms": "M/S estéreo",
        "config.group.wow": "Wow & flutter",
        "config.group.separation": "Separación de fuentes",
        "config.group.super_res": "Super-resolución",
        "config.group.output": "Salida",
        "config.group.report": "Informe",
        # Profiles
        "profiles.title": "Perfiles",
        "profiles.none": "No hay perfiles guardados todavía.",
        "profiles.load": "Cargar",
        "profiles.delete": "Eliminar",
        "profiles.empty_name": "Introduce un nombre de perfil.",
        "profiles.saved": "Perfil guardado:",
        # History
        "history.title": "Historial",
        "history.none": "Aún no hay trabajos en el historial.",
        "history.rerun": "Re-ejecutar",
        "history.clear": "Vaciar histórico",
        # About
        "about.title": "Acerca de audio-restore",
        "about.version": "Versión",
        "about.deps": "Dependencias neurales",
        "about.neural_ok": "Disponibles",
        "about.neural_missing": "No disponibles",
        "about.license": "Licencia MIT",
        # Help overlay
        "help.title": "Atajos de teclado",
        "help.switch": "Cambiar pantalla (tab / shift+tab)",
        "help.process": "Procesar / iniciar",
        "help.quit": "Salir",
        "help.lang": "Idioma (EN/ES)",
        "help.theme": "Alternar tema claro/oscuro",
        "help.help": "Mostrar/ocultar ayuda",
        "help.stop": "Detener el procesamiento",
        "help.enter": "Confirmar",
        # Language selector
        "lang.label": "Idioma",
        "lang.en": "English",
        "lang.es": "Español",
        # input/output
        "io.input": "Archivo de entrada",
        "io.output": "Archivo de salida",
        "io.browse": "Explorar…",
        # generic
        "generic.browse_input": "Seleccionar archivo de entrada",
        "generic.select_dir": "Seleccionar carpeta de entrada",
        "generic.output_dir": "Seleccionar carpeta de salida",
    },
}

#: Defaults for every key (English).
_DEFAULTS: dict[str, str] = {
    "generic.cancel": "Cancel",
    "generic.confirm": "Confirm",
    "generic.select_first": "Select an item first",
    "generic.select_input": "Select input file",
    "generic.select_output": "Select output file",
    "generic.pick_file": "Pick File",
    "error.unsupported_format": "Unsupported format",
    "aria.sidebar": "Navigation",
    "nav.home": "Home",
    "nav.single": "Single file",
    "nav.batch": "Batch",
    "nav.profiles": "Profiles",
    "nav.history": "History",
    "nav.about": "About",
    "header.subtitle": "Professional audio restoration",
    "welcome.title": "Welcome to audio restoration!",
    "welcome.intro": (
        "Professional audio restoration: classic DSP + neural models."
    ),
    "welcome.single": "Restore a single file",
    "welcome.batch": "Restore a whole folder",
    "welcome.presets": "Saved profiles",
    "welcome.history": "View job history",
    "welcome.about": "About this program",
    "welcome.tip": "Press L to switch language (EN/ES) and D to toggle theme.",
    # Dashboard
    "dashboard.quick_actions": "Quick Actions",
    "dashboard.recent": "Recent Activity",
    "dashboard.recent_empty": "No restoration jobs yet. Start by restoring a file.",
    "dashboard.active_profile": "Active Profile",
    "dashboard.profile_none": "Using default configuration",
    "dashboard.status_restored": "Restored",
    "dashboard.status_processing": "Processing",
    "dashboard.status_failed": "Failed",
    "error.not_found": "File not found",
    "error.batch_empty": "No audio files found in the folder",
    "error.pipeline_failed": "Processing failed",
    "ok.saved": "Saved",
    "proc.title": "Processing…",
    "proc.phase": "Phase",
    "proc.elapsed": "Elapsed",
    "proc.done": "Completed!",
    "proc.cancel_hint": "Press Ctrl+S to stop.",
    "results.title": "Results",
    "results.hint": "Process a file to see metrics here.",
    "results.metric": "Metric",
    "results.original": "Original",
    "results.restored": "Restored",
    "results.snr": "SNR vs original (dB)",
    "results.psnr": "PSNR (dB)",
    "results.original_rms": "Original RMS (dB)",
    "results.restored_rms": "Restored RMS (dB)",
    "results.save_plot": "Save PNG plot",
    "results.saved_plot": "Plot saved to",
    "batch.title": "Batch processing",
    "batch.folder": "Input folder",
    "batch.output_dir": "Output folder",
    "batch.ext": "Output extension",
    "batch.suffix": "Output suffix",
    "batch.workers": "Parallel workers",
    "batch.start": "Start batch",
    "batch.pending": "Pending",
    "batch.processing": "Processing",
    "batch.ok": "OK",
    "batch.failed": "Failed",
    "batch.complete": "Batch complete",
    "batch.summary": "Summary",
    "batch.succeeded": "succeeded",
    "batch.failures": "failed",
    "config.title": "Configuration",
    "config.preset": "Genre preset",
    "config.save_profile": "Save profile",
    "config.profile_name": "Profile name",
    "config.run_single": "Restore file",
    "config.run_batch": "Restore folder",
    "config.group.denoise": "Denoising",
    "config.group.declick": "Declicking",
    "config.group.dehum": "Dehum (50 Hz)",
    "config.group.eq": "Equalizer",
    "config.group.multiband": "Multiband compressor",
    "config.group.ms": "M/S stereo",
    "config.group.wow": "Wow & flutter",
    "config.group.separation": "Source separation",
    "config.group.super_res": "Super-resolution",
    "config.group.output": "Output",
    "config.group.report": "Report",
    "profiles.title": "Profiles",
    "profiles.none": "No profiles saved yet.",
    "profiles.load": "Load",
    "profiles.delete": "Delete",
    "profiles.empty_name": "Enter a profile name.",
    "profiles.saved": "Profile saved:",
    "history.title": "History",
    "history.none": "No jobs in history yet.",
    "history.rerun": "Re-run",
    "history.clear": "Clear history",
    "about.title": "About audio-restore",
    "about.version": "Version",
    "about.deps": "Neural dependencies",
    "about.neural_ok": "Available",
    "about.neural_missing": "Not available",
    "about.license": "MIT License",
    "help.title": "Keyboard shortcuts",
    "help.switch": "Switch screen (tab / shift+tab)",
    "help.process": "Process / start",
    "help.quit": "Quit",
    "help.lang": "Language (EN/ES)",
    "help.theme": "Toggle dark/light theme",
    "help.help": "Show/hide help",
    "help.stop": "Stop processing",
    "help.enter": "Confirm",
    "lang.label": "Language",
    "lang.en": "English",
    "lang.es": "Español",
    "io.input": "Input file",
    "io.output": "Output file",
    "io.browse": "Browse…",
    "generic.browse_input": "Select input file",
    "generic.select_dir": "Select input folder",
    "generic.output_dir": "Select output folder",
}

_CURRENT: Language = "en"


def get_language() -> Language:
    """Return the active language code."""
    return _CURRENT


def set_language(lang: Language) -> None:
    """Switch the active language."""
    global _CURRENT
    if lang not in ("en", "es"):
        raise ValueError(f"Unsupported language: {lang!r}")
    _CURRENT = lang


def toggle_language() -> Language:
    """Flip between English and Spanish; returns the new language."""
    set_language("es" if _CURRENT == "en" else "en")
    return _CURRENT


def t(key: str, **kwargs: object) -> str:
    """Translate *key* into the active language, interpolating ``{kw}``."""
    table = _TRANSLATIONS.get(_CURRENT, {})
    text = table.get(key, _DEFAULTS.get(key, key))
    if kwargs:
        text = text.format(**kwargs)
    return text