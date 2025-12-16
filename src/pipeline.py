import numpy as np
import matplotlib.pyplot as plt

from src.loader import *
from src.outliers import *
from src.filtering import *
from src.analysis_erp import *
from src.analysis_tfr import *
from src.saving import *

def run_pipeline(
    mat_path: str,
    out_dir: str,
    fs: float,
    nyquist: float,
    stim_onset: int,
    cutoff_frequency: float,
    max_freq: float,
    win_start_ms: float,
    win_end_ms: float,
):
    ensure_dir(out_dir)

    # 1) Load
    lfp_raw, tone_vals, meta = load_mouse_lfp_mat(mat_path)

    # Save raw
    save_npy(out_dir, "raw_data.npy", lfp_raw)

    # 2) Outlier rejection (baseline RMS & P2P)
    rms_mean, rms_std, p2p_mean, p2p_std = compute_baseline_metrics(lfp_raw, stim_onset)
    mask = build_signal_mask(lfp_raw, stim_onset, rms_mean, rms_std, p2p_mean, p2p_std, z=3.0)

    lfp_clean_list, tone_clean_list = apply_mask_per_session(lfp_raw, tone_vals, mask)
    lfp_low_list, lfp_high_list, tone_info = split_all_sessions_by_tone(lfp_clean_list, tone_clean_list)

    # Report counts (text)
    lines = []
    for s in range(meta["numSessions"]):
        before = meta["numTrials"]
        after = int(lfp_clean_list[s].shape[0])
        lines.append(f"Session {s+1}: {before} -> {after} trials after outlier rejection")
    save_text(out_dir, "outlier_rejection_summary.txt", lines)

    # 3) Filtering
    b, a = design_lowpass_butter(order=10, cutoff_hz=cutoff_frequency, nyquist=nyquist)
    low_filt_list, high_filt_list = filter_sessions(lfp_low_list, lfp_high_list, b, a)

    save_npy(out_dir, "low_filtered_data.npy", np.array(low_filt_list, dtype=object))
    save_npy(out_dir, "high_filtered_data.npy", np.array(high_filt_list, dtype=object))

    # 4) Method 1 (ERP + PSD) per session
    erp_low_all = []
    erp_high_all = []
    t_ms = None

    peak_lines = []
    for s in range(meta["numSessions"]):
        erp_low, t_ms = compute_erp(low_filt_list[s], stim_onset, fs)
        erp_high, _ = compute_erp(high_filt_list[s], stim_onset, fs)
        erp_low_all.append(erp_low)
        erp_high_all.append(erp_high)

        amp_l, lat_l = peak_amp_latency(erp_low, fs, stim_onset, win_start_ms, win_end_ms)
        amp_h, lat_h = peak_amp_latency(erp_high, fs, stim_onset, win_start_ms, win_end_ms)
        peak_lines.append(f"Session {s+1} Low: peak_amp={amp_l:.4f}, latency_ms={lat_l:.1f}")
        peak_lines.append(f"Session {s+1} High: peak_amp={amp_h:.4f}, latency_ms={lat_h:.1f}")

    save_text(out_dir, "Idea1_time_domain_results.txt", peak_lines)

    # Plot ERP per session
    plt.figure(figsize=(12, 8))
    for s in range(meta["numSessions"]):
        plt.subplot(2, 2, s + 1)
        plt.plot(t_ms, erp_low_all[s], label="Low tone")
        plt.plot(t_ms, erp_high_all[s], label="High tone")
        plt.axvline(stim_onset / fs * 1000.0, linestyle="--", alpha=0.5)
        plt.title(f"Session {s+1} ERP")
        plt.xlabel("Time (ms)")
        plt.ylabel("LFP (a.u.)")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Idea1_erp_per_session.png")
    plt.close()

    # Plot PSD per session
    plt.figure(figsize=(12, 8))
    for s in range(meta["numSessions"]):
        f_l, psd_l = erp_psd(erp_low_all[s], fs, max_freq=max_freq)
        f_h, psd_h = erp_psd(erp_high_all[s], fs, max_freq=max_freq)
        plt.subplot(2, 2, s + 1)
        plt.semilogy(f_l, psd_l, label="Low tone")
        plt.semilogy(f_h, psd_h, label="High tone")
        plt.title(f"Session {s+1} ERP Power Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (PSD)")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Idea1_frequency_domain_results.png")
    plt.close()

    # 5) Method 2 (TFR + band power) per session
    bands = {
        "theta (4-8 Hz)": (4, 8),
        "beta (13-30 Hz)": (13, 30),
        "gamma (30-80 Hz)": (30, 80),
    }

    for s in range(meta["numSessions"]):
        f_l, t_l, S_l = compute_mean_spectrogram(low_filt_list[s], fs, max_freq=100)
        f_h, t_h, S_h = compute_mean_spectrogram(high_filt_list[s], fs, max_freq=100)

        S_l_db = 10 * np.log10(S_l + 1e-12)
        S_h_db = 10 * np.log10(S_h + 1e-12)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(t_l * 1000.0, f_l, S_l_db, shading="gouraud")
        plt.axvline(stim_onset / fs * 1000.0, color="w", linestyle="--", alpha=0.7)
        plt.title(f"Session {s+1} - Low tone TFR")
        plt.xlabel("Time (ms)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Power (dB)")

        plt.subplot(1, 2, 2)
        plt.pcolormesh(t_h * 1000.0, f_h, S_h_db, shading="gouraud")
        plt.axvline(stim_onset / fs * 1000.0, color="w", linestyle="--", alpha=0.7)
        plt.title(f"Session {s+1} - High tone TFR")
        plt.xlabel("Time (ms)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Power (dB)")

        plt.tight_layout()
        plt.savefig(f"{out_dir}/Idea2_time_domain_session_{s+1}_tfr.png")
        plt.close()

        # Band power time courses
        plt.figure(figsize=(9, 5))
        for band_name, band_range in bands.items():
            bp_low = extract_band_power(S_l, f_l, band_range)
            bp_high = extract_band_power(S_h, f_h, band_range)
            plt.plot(t_l * 1000.0, bp_low, label=f"Low - {band_name}")
            plt.plot(t_h * 1000.0, bp_high, linestyle="--", label=f"High - {band_name}")

        plt.axvline(stim_onset / fs * 1000.0, linestyle="--", alpha=0.5)
        plt.title(f"Session {s+1} Band Power Time Course")
        plt.xlabel("Time (ms)")
        plt.ylabel("Band Power (a.u.)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/Idea2_frequency_domain_session_{s+1}_results.png")
        plt.close()

    # 6) Combined across sessions (optional but matches your midterm)
    lfp_low_all = np.concatenate(low_filt_list, axis=0)
    lfp_high_all = np.concatenate(high_filt_list, axis=0)
    save_npy(out_dir, "lfp_low_all.npy", lfp_low_all)
    save_npy(out_dir, "lfp_high_all.npy", lfp_high_all)

    print(f"Done. Results saved to: {out_dir}")
