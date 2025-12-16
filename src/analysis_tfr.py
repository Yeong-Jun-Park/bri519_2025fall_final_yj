import numpy as np
from scipy.signal import spectrogram

def compute_mean_spectrogram(trials: np.ndarray, fs: float, nperseg: int = 256, noverlap: int = 200, max_freq: float = 100.0):
    """
    Compute trial-averaged spectrogram power.
    Returns:
        f_sel, t, S_avg (power)
    """
    n_trials = trials.shape[0]
    f, t, S = spectrogram(trials[0], fs=fs, nperseg=nperseg, noverlap=noverlap)
    S_sum = (np.abs(S) ** 2)

    for i in range(1, n_trials):
        _, _, Si = spectrogram(trials[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        S_sum += (np.abs(Si) ** 2)

    S_avg = S_sum / n_trials

    mask = f <= max_freq
    return f[mask], t, S_avg[mask, :]

def extract_band_power(S: np.ndarray, freqs: np.ndarray, band: tuple):
    """
    Average power within a frequency band over time.
    """
    f_low, f_high = band
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return np.zeros(S.shape[1])
    return S[mask, :].mean(axis=0)
