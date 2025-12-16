import numpy as np
from scipy.signal import welch

def compute_erp(trials: np.ndarray, stim_onset: int, fs: float):
    """
    Baseline-correct each trial using pre-stimulus mean, then average across trials.
    Returns:
        erp (1D), t_ms (1D)
    """
    baseline = trials[:, :stim_onset].mean(axis=1, keepdims=True)
    trials_bc = trials - baseline
    erp = trials_bc.mean(axis=0)

    t_ms = np.arange(erp.shape[0]) / fs * 1000.0
    return erp, t_ms

def peak_amp_latency(erp: np.ndarray, fs: float, stim_onset: int, win_start_ms: float, win_end_ms: float):
    """
    Find minimum peak within [win_start_ms, win_end_ms] window (in ms),
    and report its amplitude and latency relative to stimulus onset.
    """
    start_idx = int(win_start_ms / 1000.0 * fs)
    end_idx = int(win_end_ms / 1000.0 * fs)

    segment = erp[start_idx:end_idx]
    local_idx = int(np.argmin(segment))
    peak_amp = float(segment[local_idx])

    peak_idx = start_idx + local_idx
    latency_ms = (peak_idx - stim_onset) / fs * 1000.0
    return peak_amp, float(latency_ms)

def erp_psd(erp: np.ndarray, fs: float, max_freq: float = 200.0, nperseg: int = 512):
    """
    Welch PSD for ERP; returns frequencies and PSD up to max_freq.
    """
    freqs, psd = welch(erp, fs=fs, nperseg=nperseg)
    mask = freqs <= max_freq
    return freqs[mask], psd[mask]
