"""Declicker — impulse-noise removal for vinyl and cassette recordings.

Vinyl surface noise (clicks, crackles, pops) consists of extremely short
transients (0.5–5 ms) far louder than the music beneath.  Spectral denoisers
handle *continuous* noise but smear impulse transients; this module works in
the time domain:

1. **Detection** — a sample is flagged when |x| > ``threshold`` × fast-RMS,
   and sustained bursts when fast-RMS > (``threshold`` − 1) × slow-RMS.
2. **Repair** — short clicks are filled with LPC forward prediction (Levinson-
   Durbin on the autocorrelation of clean context); 30–200 ms needle drags use
   a bidirectional decaying-LPC crossfade that meets in silence at the centre.
3. **Gain matching** — reconstructed regions are scaled to the surrounding RMS.

Clicks of type a: continuous hiss → Denoiser; hum → Dehum; clipping distortion
spans many cycles → LPC breaks down, do NOT use.
"""

from __future__ import annotations

import logging

import numpy as np

from . import audio_utils

logger = logging.getLogger(__name__)


class Declicker:
    """Impulse-noise (click / pop / crackle) remover for vinyl and cassette audio.

    Usage::

        declicker = Declicker(threshold=6.0)
        clean_audio = declicker.process(audio, sample_rate)
    """

    def __init__(
        self,
        threshold: float = 4.0,
        window_ms: float = 200.0,
        margin_ms: float = 5.0,
        lpc_order: int = 32,
        max_click_ms: float = 30.0,
        max_scratch_ms: float = 200.0,
    ):
        self.threshold = float(threshold)
        self.window_ms = float(window_ms)
        self.margin_ms = float(margin_ms)
        self.lpc_order = int(lpc_order)
        self.max_click_ms = float(max_click_ms)
        self.max_scratch_ms = float(max_scratch_ms)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Detect and remove click/pop artefacts (mono or stereo, channel-wise).

        Args:
            audio: Float32 array with shape ``(N,)`` or ``(N, 2)``.
            sample_rate: Sample rate in Hz.

        Returns:
            Declicked audio with the same shape and dtype as the input.
        """
        audio = np.asarray(audio, dtype=np.float32)
        return audio_utils.process_channels(audio, self._process_channel, sample_rate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_channel(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        y = audio.astype(np.float64)
        mask = self._detect_clicks(y, sample_rate)

        if not np.any(mask):
            logger.debug("Declicker: no clicks detected.")
            return audio.astype(np.float32)

        regions = self._mask_to_regions(mask)
        max_click_samples = int(self.max_click_ms * sample_rate / 1000)
        max_scratch_samples = int(self.max_scratch_ms * sample_rate / 1000)
        n_repaired = 0

        for start, end in regions:
            length = end - start
            if length <= max_click_samples:
                y = self._repair_region(y, start, end, sample_rate)
                n_repaired += 1
            elif length <= max_scratch_samples:
                y = self._repair_scratch(y, start, end, sample_rate)
                n_repaired += 1
            # Events longer than max_scratch_ms are skipped (musical transients).

        logger.info(
            "Declicker: detected %d regions, repaired %d (threshold=%.1f, "
            "click<=%.0fms, scratch<=%.0fms).",
            len(regions),
            n_repaired,
            self.threshold,
            self.max_click_ms,
            self.max_scratch_ms,
        )
        return y.astype(np.float32)

    def _detect_clicks(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a boolean mask indicating click / scratch samples."""
        n = len(audio)
        margin = max(1, int(self.margin_ms * sample_rate / 1000))

        sq = audio.copy()
        sq **= 2

        # Fast centered RMS (50 ms).
        win_fast = max(1, int(50.0 * sample_rate / 1000))
        kernel_fast = np.ones(win_fast) / win_fast
        rms_fast = np.sqrt(np.convolve(sq, kernel_fast, mode="same") + 1e-20)

        # Slow causal RMS (1000 ms) via cumsum — O(n).
        win_slow = max(1, int(1000.0 * sample_rate / 1000))
        first_sq = float(sq[: min(win_slow, n)].mean()) + 1e-10
        pad_slow = np.full(win_slow - 1, first_sq, dtype=np.float64)
        sq_padded = np.concatenate([pad_slow, sq.astype(np.float64)])
        cs = np.concatenate([[0.0], np.cumsum(sq_padded)])
        rms_slow = np.sqrt((cs[win_slow : win_slow + n] - cs[:n]) / win_slow + 1e-20)

        # Click mask (explicit out= avoids in-place refcount corruption of rms_fast).
        click_thresh = np.empty_like(rms_fast)
        np.multiply(self.threshold, rms_fast, out=click_thresh)
        mask_click = np.abs(audio) > click_thresh

        # Scratch onset mask (threshold reduced by 1 to catch sustained energy).
        onset_thresh = max(1.5, self.threshold - 1.0)
        mask_scratch = rms_fast > onset_thresh * rms_slow

        mask = mask_click | mask_scratch

        # Margin dilation.
        if margin > 0 and np.any(mask):
            dilated = np.convolve(
                mask.astype(np.float32), np.ones(2 * margin + 1), mode="same"
            )
            mask = dilated > 0

        return mask

    @staticmethod
    def _mask_to_regions(mask: np.ndarray) -> list[tuple[int, int]]:
        """Convert a boolean mask into a list of (start, end) index pairs."""
        regions: list[tuple[int, int]] = []
        in_region = False
        start = 0
        for i, val in enumerate(mask):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                regions.append((start, i))
                in_region = False
        if in_region:
            regions.append((start, len(mask)))
        return regions

    def _repair_region(
        self,
        audio: np.ndarray,
        start: int,
        end: int,
        sample_rate: int,
    ) -> np.ndarray:
        """Replace audio[start:end] using LPC forward prediction (linear fallback)."""
        gap = end - start
        ctx_len = max(self.lpc_order * 4, int(5.0 * sample_rate / 1000))

        ctx_start = max(0, start - ctx_len)
        context = audio[ctx_start:start]

        if len(context) < self.lpc_order + 2:
            audio = audio.copy()
            left = audio[start - 1] if start > 0 else 0.0
            right = audio[end] if end < len(audio) else 0.0
            audio[start:end] = np.linspace(left, right, gap)
            return audio

        a = self._lpc(context, self.lpc_order)

        audio = audio.copy()
        buf = list(context[-(self.lpc_order) :])
        for i in range(gap):
            pred = -sum(a[k + 1] * buf[-(k + 1)] for k in range(self.lpc_order))
            buf.append(pred)
            audio[start + i] = pred

        # Gain-match to surrounding RMS.
        ctx_rms = np.sqrt(np.mean(context[-min(512, len(context)) :] ** 2) + 1e-20)
        rep_rms = np.sqrt(np.mean(audio[start:end] ** 2) + 1e-20)
        if rep_rms > 1e-10:
            scale = min(ctx_rms / rep_rms, 2.0)
            audio[start:end] *= scale

        return audio

    def _repair_scratch(
        self,
        audio: np.ndarray,
        start: int,
        end: int,
        sample_rate: int,
    ) -> np.ndarray:
        """Repair 30–200 ms scratches with bidirectional decaying LPC."""
        gap = end - start
        order = min(self.lpc_order, 8)
        ctx_len = max(order * 6, int(30.0 * sample_rate / 1000))
        audio = audio.copy()

        decay_fwd = np.exp(-np.arange(gap, dtype=np.float64) * 3.0 / max(gap, 1))
        decay_bwd = decay_fwd[::-1].copy()

        # --- Forward decaying prediction (from before the scratch) ---
        ctx_start = max(0, start - ctx_len)
        ctx_fwd = audio[ctx_start:start].astype(np.float64)

        if len(ctx_fwd) >= order + 2:
            a_fwd = self._lpc(ctx_fwd, order)
            buf = list(ctx_fwd[-order:])
            fwd = np.empty(gap, dtype=np.float64)
            for i in range(gap):
                pred = float(-np.dot(a_fwd[1:], buf[-order:][::-1]))
                pred = float(np.clip(pred, -2.0, 2.0))
                buf.append(pred)
                fwd[i] = pred * decay_fwd[i]
        else:
            last = float(audio[start - 1]) if start > 0 else 0.0
            fwd = last * decay_fwd

        # --- Backward decaying prediction (from after the scratch) ---
        bwd_skip = max(0, int(5.0 * sample_rate / 1000))
        ctx_end = min(len(audio), end + bwd_skip + ctx_len)
        ctx_bwd = audio[min(len(audio), end + bwd_skip) : ctx_end].astype(np.float64)[
            ::-1
        ]

        if len(ctx_bwd) >= order + 2:
            a_bwd = self._lpc(ctx_bwd, order)
            buf = list(ctx_bwd[-order:])
            bwd_rev = np.empty(gap, dtype=np.float64)
            for i in range(gap):
                pred = float(-np.dot(a_bwd[1:], buf[-order:][::-1]))
                pred = float(np.clip(pred, -2.0, 2.0))
                buf.append(pred)
                bwd_rev[i] = pred * decay_fwd[i]
            bwd = bwd_rev[::-1]
        else:
            first = float(audio[end]) if end < len(audio) else 0.0
            bwd = first * decay_bwd

        blended = np.clip(fwd + bwd, -1.0, 1.0)
        audio[start:end] = blended.astype(audio.dtype)
        return audio

    @staticmethod
    def _lpc(signal: np.ndarray, order: int) -> np.ndarray:
        """Levinson-Durbin LPC coefficients; returns a where a[0] = 1."""
        r = np.array(
            [np.dot(signal[k:], signal[: len(signal) - k]) for k in range(order + 1)]
        )

        if r[0] < 1e-20:
            a = np.zeros(order + 1)
            a[0] = 1.0
            return a

        a = np.zeros(order + 1)
        a[0] = 1.0
        error = r[0]

        for i in range(1, order + 1):
            lam = -np.dot(a[1:i], r[i - 1 : 0 : -1]) - r[i]
            lam /= error
            a_new = a.copy()
            for j in range(1, i + 1):
                a_new[j] = a[j] + lam * a[i - j]
            a_new[i] = lam
            a = a_new
            error *= 1.0 - lam**2
            if error < 1e-20:
                break

        return a
