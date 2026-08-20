"""
Declicker / Depop — Impulse noise removal for vinyl and cassette recordings.

Vinyl surface noise (clicks, crackles, pops) consists of extremely short
transient bursts (typically 0.5–5 ms) that are orders of magnitude louder
than the underlying music signal.  Spectral-domain denoisers (Wiener filters,
noisereduce) handle *continuous* noise well but fail on impulse noise because
they average the transient across many frames, smearing it and distorting the
music around it.

Algorithm (time-domain click detection + interpolation):
─────────────────────────────────────────────────────────
1. **Local RMS baseline** — Compute a short-time RMS of the signal using a
   sliding window.  This follows the music's dynamic envelope without being
   thrown off by individual clicks.

2. **Click detection** — A sample is flagged as a click if its absolute value
   exceeds `threshold` × local_RMS.  Flagged regions are extended slightly
   (±`margin_samples`) to capture click tails.

3. **AR interpolation** — Each flagged region is replaced using Linear
   Predictive Coding (LPC).  Coefficients are estimated from the clean audio
   immediately *before* the click, and the gap is filled by forward prediction.
   This preserves the local timbre and phase of the music.

4. **Gain matching** — After interpolation, the reconstructed segment's RMS is
   matched to the surrounding audio to avoid energy discontinuities.

Why LPC (not simple cubic interpolation)?
  Cubic splines work for very short gaps (< 2 ms) but fail on pops (5-20 ms)
  because they do not model the resonant structure of the instrument.
  LPC essentially fits an all-pole filter to the local signal, capturing the
  formant/harmonic structure — the interpolated samples sound natural even
  for longer gaps.

Suitable for:
  - Vinyl records: surface dust, scratches → clicks < 5 ms
  - Cassette tape: dropout glitches
  - Old shellac / 78 rpm: heavy crackle

NOT suitable for:
  - Continuous hiss/noise → use Denoiser (denoiser.py)
  - Electrical hum → use Dehum (_apply_dehum in restoration_pipeline.py)
  - Clipping distortion → clips span many cycles, LPC breaks down
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.signal import lfilter

logger = logging.getLogger(__name__)


class Declicker:
    """
    Impulse-noise (click / pop / crackle) remover for vinyl and cassette audio.

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
        """
        Args:
            threshold:      A sample is a click if |x| > threshold × local_RMS.
                            Lower = more aggressive (may remove musical transients).
                            Recommended: 5–8 for vinyl, 4–6 for heavy crackle.
            window_ms:      Length of the sliding RMS window in milliseconds.
                            Should be long enough to cover one note onset (≥ 10 ms).
            margin_ms:      Extra padding added around each detected click region (ms).
            lpc_order:      Number of LPC coefficients used for AR interpolation.
                            Higher = better models complex timbres (voices, piano).
            max_click_ms:   Regions shorter than this use standard LPC forward
                            prediction (good for clicks ≤ 30 ms).
            max_scratch_ms: Regions between max_click_ms and this limit use
                            bidirectional crossfade interpolation — forward LPC
                            from before the scratch blended with backward LPC from
                            after it.  This handles needle drags (50–200 ms).
                            Regions longer than max_scratch_ms are skipped
                            (treated as musical transients).
        """
        self.threshold      = float(threshold)
        self.window_ms      = float(window_ms)
        self.margin_ms      = float(margin_ms)
        self.lpc_order      = int(lpc_order)
        self.max_click_ms   = float(max_click_ms)
        self.max_scratch_ms = float(max_scratch_ms)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Detects and removes click/pop artifacts from audio.

        Args:
            audio:       Input audio (float32 or float64, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            Declicked audio (same shape and dtype as input).
        """
        audio = audio.astype(np.float64)
        mask  = self._detect_clicks(audio, sample_rate)

        if not np.any(mask):
            logger.debug("Declicker: no clicks detected.")
            return audio.astype(np.float32)

        # Find contiguous click regions
        regions = self._mask_to_regions(mask)
        max_click_samples   = int(self.max_click_ms   * sample_rate / 1000)
        max_scratch_samples = int(self.max_scratch_ms * sample_rate / 1000)
        n_repaired  = 0

        for start, end in regions:
            length = end - start
            if length <= max_click_samples:
                # Short click/pop: standard LPC forward prediction
                audio = self._repair_region(audio, start, end, sample_rate)
                n_repaired += 1
            elif length <= max_scratch_samples:
                # Long scratch/drag: bidirectional crossfade interpolation
                audio = self._repair_scratch(audio, start, end, sample_rate)
                n_repaired += 1
            # else: too long — skip (musical transient or unrecoverable)

        logger.info(
            "Declicker: detected %d regions, repaired %d (threshold=%.1f, "
            "click≤%.0fms, scratch≤%.0fms).",
            len(regions), n_repaired, self.threshold,
            self.max_click_ms, self.max_scratch_ms,
        )
        return audio.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_clicks(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Returns a boolean mask where True indicates a click or scratch sample.

        Two-pass strategy:

        Click detection (short transients)
          • Fast centered RMS (50 ms) — adapts quickly to local dynamics.
          • A sample is flagged if |x| > threshold × fast_rms.

        Scratch detection (sustained noise bursts)
          • Fast centered RMS (50 ms) — captures current energy.
          • Slow causal RMS (1 000 ms, padded with initial-window energy) —
            represents the long-term background BEFORE the current point.
            Because it only looks backward, a 120-200 ms scratch cannot inflate
            its own reference more than ~12-20 % of the full window, keeping
            the detection threshold anchored to the pre-scratch music level.
          • A region is flagged when fast_rms > (threshold − 1) × slow_rms.
            Using threshold − 1 (floor 1.5) catches sustained elevated energy,
            not just sharp peaks.

        Both masks are combined, then dilated by margin_ms.
        """
        n      = len(audio)
        margin = max(1, int(self.margin_ms * sample_rate / 1000))

        sq = audio.copy()
        sq **= 2

        # ── Fast centered RMS (50 ms) ───────────────────────────────────────
        # Small kernel → np.convolve is fine.
        win_fast    = max(1, int(50.0 * sample_rate / 1000))
        kernel_fast = np.ones(win_fast) / win_fast
        rms_fast    = np.sqrt(np.convolve(sq, kernel_fast, mode="same") + 1e-20)

        # ── Slow causal RMS (1 000 ms) via cumsum — O(n), no large kernel ──
        # Pad with the initial-window energy so the causal warm-up period does
        # not create a near-zero floor that flags all initial content.
        win_slow  = max(1, int(1000.0 * sample_rate / 1000))
        first_sq  = float(sq[:min(win_slow, n)].mean()) + 1e-10
        pad_slow  = np.full(win_slow - 1, first_sq, dtype=np.float64)
        sq_padded = np.concatenate([pad_slow, sq.astype(np.float64)])
        cs        = np.concatenate([[0.0], np.cumsum(sq_padded)])
        rms_slow  = np.sqrt((cs[win_slow : win_slow + n] - cs[:n]) / win_slow + 1e-20)

        # ── Click mask ──────────────────────────────────────────────────────
        # IMPORTANT: use np.multiply with explicit out= to prevent NumPy's
        # refcount=1 in-place scalar-multiplication from corrupting rms_fast
        # before it is reused for mask_scratch.
        click_thresh = np.empty_like(rms_fast)
        np.multiply(self.threshold, rms_fast, out=click_thresh)
        mask_click   = np.abs(audio) > click_thresh

        # ── Scratch onset mask ──────────────────────────────────────────────
        # Threshold reduced by 1 to detect sustained energy, not just spikes.
        onset_thresh = max(1.5, self.threshold - 1.0)
        mask_scratch = rms_fast > onset_thresh * rms_slow

        mask = mask_click | mask_scratch

        # ── Margin dilation (both sides) ────────────────────────────────────
        if margin > 0 and np.any(mask):
            mask_f     = mask.astype(np.float32)
            dil_kernel = np.ones(2 * margin + 1)
            dilated    = np.convolve(mask_f, dil_kernel, mode="same")
            mask       = dilated > 0

        return mask

    @staticmethod
    def _mask_to_regions(mask: np.ndarray) -> list:
        """Converts a boolean mask into a list of (start, end) index pairs."""
        regions = []
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
        """
        Replaces audio[start:end] using LPC forward prediction estimated from
        the clean audio immediately before the click.

        Falls back to linear interpolation if the LPC context is too short.
        """
        gap = end - start
        ctx_len = max(self.lpc_order * 4, int(5.0 * sample_rate / 1000))  # ≥ 5 ms

        # Context: clean audio just before the click
        ctx_start = max(0, start - ctx_len)
        context = audio[ctx_start:start]

        if len(context) < self.lpc_order + 2:
            # Not enough context — use linear interpolation
            audio = audio.copy()
            left  = audio[start - 1] if start > 0 else 0.0
            right = audio[end]       if end < len(audio) else 0.0
            audio[start:end] = np.linspace(left, right, gap)
            return audio

        # Estimate LPC coefficients from context
        a = self._lpc(context, self.lpc_order)

        # Forward-predict into the gap
        audio = audio.copy()
        buf = list(context[-(self.lpc_order):])
        for i in range(gap):
            # y[n] = -a[1]*y[n-1] - a[2]*y[n-2] - ... (a[0]=1 always)
            pred = -sum(a[k + 1] * buf[-(k + 1)] for k in range(self.lpc_order))
            buf.append(pred)
            audio[start + i] = pred

        # Gain-match repaired region to surrounding RMS to avoid energy jumps
        ctx_rms = np.sqrt(np.mean(context[-min(512, len(context)):]**2) + 1e-20)
        rep_rms = np.sqrt(np.mean(audio[start:end]**2) + 1e-20)
        if rep_rms > 1e-10:
            scale = min(ctx_rms / rep_rms, 2.0)  # cap at 2× to avoid explosions
            audio[start:end] *= scale

        return audio

    def _repair_scratch(
        self,
        audio: np.ndarray,
        start: int,
        end: int,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Repairs a long scratch/needle-drag (max_click_ms … max_scratch_ms) using
        **bidirectional decaying LPC**:

          1. Forward LPC from clean audio *before* the scratch, with an exponential
             decay applied *during* each prediction step so the signal naturally
             fades to silence by the midpoint (prevents divergence on long gaps).
          2. Backward LPC from clean audio *after* the scratch (reversed), also
             decaying — then un-reversed, so it rises from silence toward the end.
          3. The two decaying tails are summed: each contributes where it is
             strong (near its own boundary) and is near-zero in the middle,
             creating a smooth "dropout" that the ear accepts far better than a
             loud scratch.

        A low LPC order (≤ 8) is forced here for numerical stability over the
        potentially thousands of samples in the gap.
        """
        gap     = end - start
        order   = min(self.lpc_order, 8)           # low order → stable over long gaps
        ctx_len = max(order * 6, int(30.0 * sample_rate / 1000))  # ≥ 30 ms context
        audio   = audio.copy()

        # Exponential decay so each tail reaches ≈ e^-3 ≈ 5% by the midpoint
        decay_fwd = np.exp(-np.arange(gap, dtype=np.float64) * 3.0 / max(gap, 1))
        decay_bwd = decay_fwd[::-1].copy()

        # --- Forward decaying prediction (from before the scratch) ---
        ctx_start = max(0, start - ctx_len)
        ctx_fwd   = audio[ctx_start:start].astype(np.float64)

        if len(ctx_fwd) >= order + 2:
            a_fwd = self._lpc(ctx_fwd, order)
            buf   = list(ctx_fwd[-order:])
            fwd   = np.empty(gap, dtype=np.float64)
            for i in range(gap):
                pred = float(-np.dot(a_fwd[1:], buf[-order:][::-1]))
                pred = float(np.clip(pred, -2.0, 2.0))
                buf.append(pred)
                fwd[i] = pred * decay_fwd[i]   # decay applied to stored value
        else:
            last = float(audio[start - 1]) if start > 0 else 0.0
            fwd  = last * decay_fwd

        # --- Backward decaying prediction (from after the scratch) ---
        # Skip a small safety window past the detection boundary to avoid
        # initialising the LPC from scratch-tail samples that the detector
        # may have missed (detection boundary can lag by a few ms because the
        # causal rms_slow is elevated by the scratch's own energy).
        bwd_skip = max(0, int(5.0 * sample_rate / 1000))  # 5 ms
        ctx_end  = min(len(audio), end + bwd_skip + ctx_len)
        ctx_bwd  = audio[min(len(audio), end + bwd_skip):ctx_end].astype(np.float64)[::-1]

        if len(ctx_bwd) >= order + 2:
            a_bwd   = self._lpc(ctx_bwd, order)
            buf     = list(ctx_bwd[-order:])
            bwd_rev = np.empty(gap, dtype=np.float64)
            for i in range(gap):
                pred = float(-np.dot(a_bwd[1:], buf[-order:][::-1]))
                pred = float(np.clip(pred, -2.0, 2.0))
                buf.append(pred)
                bwd_rev[i] = pred * decay_fwd[i]   # also decays away from edge
            bwd = bwd_rev[::-1]    # un-reverse → rises from silence toward end
        else:
            first = float(audio[end]) if end < len(audio) else 0.0
            bwd   = first * decay_bwd

        # --- Sum: each tail is already envelope-shaped; no extra crossfade weight needed ---
        blended = fwd + bwd
        blended = np.clip(blended, -1.0, 1.0)   # hard safety clamp

        audio[start:end] = blended.astype(audio.dtype)
        return audio

    @staticmethod
    def _lpc(signal: np.ndarray, order: int) -> np.ndarray:
        """
        Estimates LPC (Linear Predictive Coding) coefficients using the
        Levinson-Durbin algorithm on the autocorrelation of the signal.

        Returns a coefficient array `a` of length (order + 1) where a[0] = 1.
        """
        # Autocorrelation
        r = np.array([np.dot(signal[k:], signal[:len(signal) - k])
                      for k in range(order + 1)])

        if r[0] < 1e-20:
            a = np.zeros(order + 1)
            a[0] = 1.0
            return a

        # Levinson-Durbin recursion
        a     = np.zeros(order + 1)
        a[0]  = 1.0
        error = r[0]

        for i in range(1, order + 1):
            # Reflection coefficient
            lam = -np.dot(a[1:i], r[i - 1:0:-1]) - r[i]
            lam /= error
            # Update coefficients
            a_new = a.copy()
            for j in range(1, i + 1):
                a_new[j] = a[j] + lam * a[i - j]
            a_new[i] = lam
            a = a_new
            error *= 1.0 - lam ** 2
            if error < 1e-20:
                break

        return a
