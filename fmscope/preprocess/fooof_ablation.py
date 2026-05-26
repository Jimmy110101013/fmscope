"""FOOOF-based aperiodic / periodic component removal.

This is the recipe the paper uses to disentangle the contribution of
the 1/f aperiodic component vs the narrow-band oscillatory peaks
when probing what an FM encodes.

Per channel × per recording:
    1. Welch PSD over all epochs concatenated.
    2. FOOOF fit (fixed aperiodic mode).
    3. Build aperiodic amplitude envelope and Gaussian peak power.
    4. Apply ablation in the frequency domain (preserve phase), then
       inverse FFT back to time domain.

Three ablation modes:
    - ``aperiodic_removed`` : divide spectrum by the 1/f envelope
    - ``periodic_removed``  : subtract Gaussian-peak power
    - ``both_removed``      : compose both

For paper Tab 3's "1/f role" column, the relevant comparison is
``original.label_BA - aperiodic_removed.label_BA`` and the same
substituting ``subject_BA``.

Dependencies
------------
:pep:`fooof>=1.1` (PyPI ``fooof``) — added to fmscope's optional
dependency group ``[fms]``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


Mode = Literal["aperiodic_removed", "periodic_removed", "both_removed"]


# Paper-locked defaults (see scripts/analysis/fooof_ablation.py).
_DEFAULTS = {
    "fit_range": (1.0, 45.0),
    "welch_nperseg": 512,
    "peak_width_limits": (1.0, 12.0),
    "max_n_peaks": 6,
    "min_peak_height": 0.1,
}


def _fit_one_channel_recording(
    all_epochs_one_ch: np.ndarray, sfreq: float,
    *, fit_range, welch_nperseg, peak_width_limits, max_n_peaks, min_peak_height,
) -> dict | None:
    """One FOOOF fit per (channel, recording) — uses PSD averaged over epochs."""
    from fooof import FOOOF
    from scipy import signal as sig

    psds = []
    for x in all_epochs_one_ch:
        freqs, psd = sig.welch(x, fs=sfreq, nperseg=welch_nperseg,
                               noverlap=welch_nperseg // 2)
        psds.append(psd)
    psd_mean = np.mean(psds, axis=0)
    mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    if np.any(psd_mean[mask] <= 0) or not np.all(np.isfinite(psd_mean[mask])):
        return None
    fm = FOOOF(
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        aperiodic_mode="fixed",
        verbose=False,
    )
    try:
        fm.fit(freqs[mask], psd_mean[mask], fit_range)
    except Exception:
        return None
    if not fm.has_model:
        return None
    b = float(fm.aperiodic_params_[0])
    chi = float(fm.aperiodic_params_[1])
    peaks = (fm.peak_params_.astype(np.float32)
             if fm.n_peaks_ > 0 else np.zeros((0, 3), np.float32))
    return {"b": b, "chi": chi, "peaks": peaks, "r2": float(fm.r_squared_)}


def _build_aperiodic_amp(n_fft: int, sfreq: float, b: float, chi: float,
                         fit_range_low: float) -> np.ndarray:
    """Amplitude envelope from FOOOF aperiodic parameters."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)
    safe = np.maximum(freqs, fit_range_low)
    return np.sqrt(10.0 ** (b - chi * np.log10(safe))).astype(np.float32)


def _reconstruct(x: np.ndarray, fit: dict, sfreq: float,
                 mode: Mode, fit_range_low: float) -> np.ndarray:
    """Apply one ablation mode in the FFT domain."""
    X = np.fft.rfft(x)
    n_fft = len(x)
    amp_aper = _build_aperiodic_amp(n_fft, sfreq, fit["b"], fit["chi"],
                                    fit_range_low)
    amp_aper_safe = np.maximum(amp_aper, 1e-8)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sfreq)

    if mode in ("periodic_removed", "both_removed"):
        # Subtract Gaussian-peak power, then re-amplitude.
        power_orig = np.abs(X) ** 2 / n_fft
        power_peaks = np.zeros_like(power_orig)
        for (cf, amp, bw) in fit["peaks"]:
            log_gauss = amp * np.exp(-((freqs - cf) ** 2) / (2.0 * (bw / 2.0) ** 2))
            delta = (amp_aper ** 2) * (10.0 ** log_gauss - 1.0)
            power_peaks += delta
        power_flat = np.maximum(power_orig - power_peaks, 1e-16)
        amp_flat = np.sqrt(power_flat * n_fft)
        phase = np.angle(X)
        X = amp_flat * (np.cos(phase) + 1j * np.sin(phase))

    if mode in ("aperiodic_removed", "both_removed"):
        X = X / amp_aper_safe

    return np.fft.irfft(X, n=n_fft).astype(np.float32)


def fooof_ablate(
    epochs: np.ndarray,
    sfreq: float,
    *,
    mode: Mode = "aperiodic_removed",
    fit_range: tuple[float, float] = _DEFAULTS["fit_range"],
    welch_nperseg: int = _DEFAULTS["welch_nperseg"],
    peak_width_limits: tuple[float, float] = _DEFAULTS["peak_width_limits"],
    max_n_peaks: int = _DEFAULTS["max_n_peaks"],
    min_peak_height: float = _DEFAULTS["min_peak_height"],
) -> np.ndarray:
    """Apply FOOOF-based aperiodic/periodic ablation to EEG epochs.

    The fit is computed **once per (channel, recording)** on the PSD
    averaged over all epochs of that recording, then applied epoch by
    epoch — matching the paper's recipe.

    Parameters
    ----------
    epochs : np.ndarray, shape (n_epochs, n_channels, n_samples)
        Time-domain EEG epochs (one recording's worth).
    sfreq : float
        Sample rate in Hz (typically 200 for FMScope-compatible data).
    mode : {"aperiodic_removed", "periodic_removed", "both_removed"}

    Returns
    -------
    np.ndarray, shape (n_epochs, n_channels, n_samples)
        Ablated time-domain epochs, same dtype as input.

    Notes
    -----
    - Channels where the FOOOF fit fails (non-positive PSD, no peaks,
      diverged) are passed through unchanged — caller should inspect
      ``.r2`` if surfaced; here we silently keep the original channel.
    - For full coverage of the paper's `fooof_ablation/` dataset
      bundle, run all three modes and stack the outputs.
    """
    epochs = np.asarray(epochs, dtype=np.float32)
    assert epochs.ndim == 3, f"epochs must be 3-D (n_ep, C, T), got {epochs.shape}"
    n_ep, n_ch, n_t = epochs.shape

    fit_kw = dict(fit_range=fit_range, welch_nperseg=welch_nperseg,
                  peak_width_limits=peak_width_limits, max_n_peaks=max_n_peaks,
                  min_peak_height=min_peak_height)

    out = np.empty_like(epochs)
    for c in range(n_ch):
        fit = _fit_one_channel_recording(epochs[:, c, :], sfreq, **fit_kw)
        if fit is None:
            out[:, c, :] = epochs[:, c, :]
            continue
        for e in range(n_ep):
            out[e, c, :] = _reconstruct(epochs[e, c, :], fit, sfreq,
                                        mode, fit_range[0])
    return out
