import numpy as np

def compute_baseline_metrics(lfp_raw: np.ndarray, stim_onset: int):
    """
    Compute baseline RMS and peak-to-peak (P2P) over ALL trials across sessions.

    Returns:
        rms_mean, rms_std, p2p_mean, p2p_std
    """
    whole = lfp_raw.reshape(-1, lfp_raw.shape[-1])
    baseline = whole[:, :stim_onset]

    baseline_rms = np.sqrt(np.mean(baseline**2, axis=1))
    baseline_p2p = baseline.max(axis=1) - baseline.min(axis=1)

    return (baseline_rms.mean(), baseline_rms.std(),
            baseline_p2p.mean(), baseline_p2p.std())

def build_signal_mask(lfp_raw: np.ndarray, stim_onset: int,
                      rms_mean: float, rms_std: float,
                      p2p_mean: float, p2p_std: float,
                      z: float = 3.0):
    """
    Build boolean mask (numSessions, numTrials) where True means 'keep trial'.
    A trial is rejected if baseline RMS or P2P exceeds mean + z*std.
    """
    num_sessions, num_trials, _ = lfp_raw.shape
    mask = np.zeros((num_sessions, num_trials), dtype=bool)

    rms_thr = rms_mean + z * rms_std
    p2p_thr = p2p_mean + z * p2p_std

    for s in range(num_sessions):
        for t in range(num_trials):
            trial = lfp_raw[s, t]
            base = trial[:stim_onset]
            rms = np.sqrt(np.mean(base**2))
            p2p = base.max() - base.min()
            keep = (rms <= rms_thr) and (p2p <= p2p_thr)
            mask[s, t] = keep

    return mask

def apply_mask_per_session(lfp_raw: np.ndarray, tone_vals: np.ndarray, mask: np.ndarray):
    """
    Apply session-wise mask to LFP and tone arrays.
    Returns:
        lfp_clean: list of np.ndarray, each shape (n_kept, dataSamples)
        tone_clean: list of np.ndarray, each shape (n_kept,)
    """
    num_sessions = lfp_raw.shape[0]
    lfp_clean, tone_clean = [], []

    for s in range(num_sessions):
        lfp_clean.append(lfp_raw[s][mask[s]])
        tone_clean.append(tone_vals[s][mask[s]])

    return lfp_clean, tone_clean

def split_by_tone(lfp_clean_session: np.ndarray, tone_clean_session: np.ndarray):
    """
    Split one session's cleaned trials into low/high tone based on unique tone values.
    Returns:
        lfp_low, lfp_high, low_tone, high_tone
    """
    uniq = np.unique(tone_clean_session)
    low_tone = uniq.min()
    high_tone = uniq.max()

    low_mask = (tone_clean_session == low_tone)
    high_mask = (tone_clean_session == high_tone)

    return (lfp_clean_session[low_mask],
            lfp_clean_session[high_mask],
            low_tone, high_tone)

def split_all_sessions_by_tone(lfp_clean_list, tone_clean_list):
    """
    Split cleaned trials into low/high for each session.
    Returns:
        lfp_low_list, lfp_high_list (each list of np.ndarray)
        tone_info_list: list of dicts with low_tone/high_tone/counts
    """
    lfp_low_list, lfp_high_list = [], []
    tone_info_list = []

    for s in range(len(lfp_clean_list)):
        lfp_s = lfp_clean_list[s]
        tone_s = tone_clean_list[s]
        l_low, l_high, low_tone, high_tone = split_by_tone(lfp_s, tone_s)

        lfp_low_list.append(l_low)
        lfp_high_list.append(l_high)
        tone_info_list.append({
            "session": s + 1,
            "low_tone": float(low_tone),
            "high_tone": float(high_tone),
            "n_low": int(l_low.shape[0]),
            "n_high": int(l_high.shape[0]),
        })

    return lfp_low_list, lfp_high_list, tone_info_list
