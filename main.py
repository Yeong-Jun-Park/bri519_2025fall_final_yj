import os
from src.pipeline import run_pipeline

def main():
    # initial parameters
    fs = 1e4
    nyquist = fs / 2
    stim_onset = 1000   
    cutoff_frequency = 1e3
    max_freq = 200        
    win_start_ms = 100
    win_end_ms = 250

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    mat_path = os.path.join(BASE_DIR, "mouseLFP.mat")
    out_dir = os.path.join(BASE_DIR, "results")

    run_pipeline(
        mat_path=mat_path,
        out_dir=out_dir,
        fs=fs,
        nyquist=nyquist,
        stim_onset=stim_onset,
        cutoff_frequency=cutoff_frequency,
        max_freq=max_freq,
        win_start_ms=win_start_ms,
        win_end_ms=win_end_ms,
    )

if __name__ == "__main__":
    main()
