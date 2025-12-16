import numpy as np
import scipy.io as io

def load_mouse_lfp_mat(mat_path: str):
    """
    Load 'mouseLFP.mat' and extract session-wise LFP trials and tone labels.

    Returns:
        lfp_raw: np.ndarray, shape (numSessions, numTrials, dataSamples)
        tone_vals: np.ndarray, shape (numSessions, numTrials)
        meta: dict with numSessions, numTrials, dataSamples
    """
    DATA = io.loadmat(mat_path)["DATA"]

    num_sessions = DATA.shape[0]
    num_trials = DATA[0, 0].shape[0]
    data_samples = DATA[0, 0].shape[1]

    lfp_raw = []
    tone_vals = []

    for i in range(num_sessions):
        lfp_raw.append(DATA[i, 0])          # (numTrials, dataSamples)
        tone_vals.append(DATA[i, 4][:, 0])  # (numTrials,)

    lfp_raw = np.array(lfp_raw)
    tone_vals = np.array(tone_vals)

    meta = {
        "numSessions": num_sessions,
        "numTrials": num_trials,
        "dataSamples": data_samples,
    }
    return lfp_raw, tone_vals, meta
