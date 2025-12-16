import numpy as np
from scipy.signal import butter, filtfilt

def design_lowpass_butter(order: int, cutoff_hz: float, nyquist: float):
    """
    Design Butterworth low-pass filter.
    """
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    return b, a

def apply_filter_trials(trials: np.ndarray, b, a):
    """
    Apply filtfilt to each trial. Input shape: (n_trials, dataSamples)
    Returns np.ndarray with same shape.
    """
    filtered = []
    for i in range(trials.shape[0]):
        filtered.append(filtfilt(b, a, trials[i]))
    return np.array(filtered)

def filter_sessions(lfp_low_list, lfp_high_list, b, a):
    """
    Filter low/high trials for each session.
    Returns:
        low_filtered_list, high_filtered_list (list of np.ndarray)
    """
    low_filt, high_filt = [], []
    for s in range(len(lfp_low_list)):
        low_filt.append(apply_filter_trials(lfp_low_list[s], b, a))
        high_filt.append(apply_filter_trials(lfp_high_list[s], b, a))
    return low_filt, high_filt
