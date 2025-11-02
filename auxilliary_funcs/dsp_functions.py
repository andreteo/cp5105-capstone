import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from auxilliary_funcs.iq_functions import load_iq_auto
from scipy.signal import medfilt, savgol_filter, find_peaks


def compute_fft(iq_data, fs):
    """
    Calculate the range FFT for a single channel (i or q).
    :param iq_data: 1D array of received samples from the single channel.
    :return: FFT array.
    """
    N = len(iq_data)
    blackman = np.blackman(N)

    # Mean removal to eliminate DC offset
    iq_data_nodc = iq_data - np.mean(iq_data)
    iq_data_windowed = iq_data_nodc * blackman
    iq_data_fft = np.fft.fftshift(np.fft.fft(iq_data_windowed, n=N))
    iq_data_mag = np.abs(iq_data_fft)

    # Normalize to maximum power
    iq_data_mag /= np.max(iq_data_mag)
    magnitude_dB = 20 * np.log10(np.abs(iq_data_mag) + 1e-12)

    iq_data_freq_axis = np.fft.fftshift(
        np.fft.fftfreq(N, d=1/fs)) / 1e6  # Convert Hz to MHz

    return iq_data_freq_axis, magnitude_dB


def q_demod(x, fs=None, gain=None, remove_dc=True):
    """
    Quadrature demod via 1-sample phase difference.

    x : complex baseband 1-D
    fs: if provided and gain is None → output in Hz
    gain: scale override (e.g., fs/(2π)/deviation)
    remove_dc: subtract mean

    Returns: y, length len(x)-1
    """
    x = np.ravel(x).astype(np.complex64, copy=False)
    if x.size < 2:
        return np.empty(0, dtype=np.float32)

    dphi = np.angle(x[1:] * np.conj(x[:-1]))  # rad/sample

    if gain is not None:
        y = gain * dphi
    elif fs is not None:
        y = (fs / (2.0*np.pi)) * dphi        # Hz
    else:
        y = dphi                              # rad/sample

    if remove_dc and y.size:
        y = y - np.mean(y)
    return y.astype(np.float32, copy=False)


def resample_chain(y, downs, ups=None, window=("kaiser", 8.6)):
    """
    Decimate in safe stages using resample_poly.
    """
    z = np.asarray(y, float)
    if ups is None:
        ups = [1] * len(list(downs))
    scale = 1.0
    for up, down in zip(ups, downs):
        z = signal.resample_poly(z, up=up, down=down, window=window)
        scale *= down / up
    return z, scale


def bandpass(y, fs, f_lo, f_hi, order=4, zero_phase=True):
    """Butterworth band-pass (SOS)"""
    sos = signal.butter(order, [f_lo, f_hi], btype="band", fs=fs, output="sos")
    y = np.asarray(y, dtype=float)

    return signal.sosfiltfilt(sos, y) if zero_phase else signal.sosfilt(sos, y)


def extract_breath(x, fs_raw, decim1=100, decim2=100, f_band=(0.05, 0.5), remove_cfo=True):
    """
    IQ → instantaneous frequency (Hz) → decimate → bandlimit to breathing band.
    Returns (breath, fs_out).
    """
    f_inst = q_demod(x, fs=fs_raw, gain=None, remove_dc=False)  # Hz
    if remove_cfo and f_inst.size:
        f_inst -= np.median(f_inst)

    y1, _ = resample_chain(f_inst, downs=[decim1])     # e.g., 2e6 → 20 kHz
    y2, _ = resample_chain(y1,    downs=[decim2])      # e.g., 20 k → 200 Hz
    fs_out = fs_raw / (decim1*decim2)

    breath = bandpass(y2 - np.median(y2), fs_out,
                      f_band[0], f_band[1], order=4, zero_phase=True)
    return breath, float(fs_out)


def analyze_spectra_peaks(
    iq_stems,
    ma_win=1001,          # moving-average window
    edge_trim=5000,       # bins trimmed on each side before peak-pick
    peak_prom=2.0,        # dB prominence for find_peaks
    min_width_khz=30.0,   # min lobe width (converted to bins)
    exclude_width_mhz=0.15,  # exclude ±this MHz around peak for NF
    ref_index=0,          # which entry is the reference (e.g., 10 cm)
    make_plots=True,      # plot spectra + summary charts
    title="Frequency Response (Averaged)"
):
    """
    iq_stems: list of paths WITHOUT extension ('.bin' & '.json' must exist).
              Example: "sessions/10cm_away/.../pluto_10cm_2000000Sps_int16_60s"
    Returns:
        tbl (pd.DataFrame): results table with Peak, NF, SNR, etc.
    """

    rows = []

    if make_plots:
        plt.figure(figsize=(12, 7))

    for idx, stem in enumerate(iq_stems):
        stem = Path(stem)
        meta_path = stem.with_suffix(".json")
        bin_path = stem.with_suffix(".bin")

        # Read metadata
        meta = json.loads(meta_path.read_text())
        fs_hz = float(meta["sample_rate_hz"])
        fc_mhz = float(meta["center_frequency_hz"]) / 1e6

        # Distance/label from folder like ".../10cm_away/..."
        try:
            dist_label = stem.parents[2].name.replace("_away", "")
        except Exception:
            dist_label = stem.stem

        # Load complex IQ (your helper returns complex array)
        x = load_iq_auto(bin_path)

        # FFT via your helper: frequency axis (MHz) centered at 0
        f_mhz_centered, mag_db = compute_fft(x, fs_hz)
        f_abs_mhz = f_mhz_centered + fc_mhz

        # Keep positive frequencies
        pos_mask = (f_mhz_centered >= 0)
        f_half = f_abs_mhz[pos_mask]
        y_half = mag_db[pos_mask].astype(float)
        y_half[~np.isfinite(y_half)] = np.nan

        # Moving-average smoothing (keep odd window)
        if len(y_half) > 5:
            win = min(ma_win, len(y_half) - (len(y_half)+1) % 2)
            win = max(9, win if win % 2 == 1 else win-1)
        else:
            win = 9
        kernel = np.ones(win)/win
        y_smooth = np.convolve(y_half, kernel, mode="same")

        # Trim edges before peak search
        if len(y_smooth) > 2*edge_trim:
            f_roi = f_half[edge_trim:-edge_trim]
            y_roi = y_smooth[edge_trim:-edge_trim]
        else:
            f_roi = f_half.copy()
            y_roi = y_smooth.copy()

        # Median filter + Savitzky–Golay (preserve lobe shape)
        if len(y_roi) >= 9:
            y_clean = medfilt(y_roi, kernel_size=7)
            sg_win = min((len(y_roi) - len(y_roi)+1) % 2, 201)
            sg_win = max(9, sg_win if sg_win % 2 == 1 else sg_win-1)
            y_pk = savgol_filter(y_clean, sg_win, 2)
        else:
            y_pk = y_roi.copy()

        # Peak width in bins from df
        if len(f_roi) > 1:
            df_mhz = float(np.median(np.diff(f_roi)))
        else:
            df_mhz = 0.001
        width_bins = max(1, int((min_width_khz/1e3) / max(df_mhz, 1e-9)))

        peaks, props = find_peaks(y_pk, prominence=peak_prom, width=width_bins)
        if peaks.size == 0:
            j = int(np.nanargmax(y_pk))  # fallback
        else:
            j = peaks[np.argmax(props["prominences"])]

        fpk_mhz = float(f_roi[j])
        dBpk = float(y_roi[j])

        # Noise floor: 10th percentile excluding ±exclude_width_mhz around peak
        exclude = (np.abs(f_roi - fpk_mhz) < exclude_width_mhz)
        if np.any(~exclude):
            noise_floor_db = float(np.nanpercentile(y_roi[~exclude], 10))
        else:
            noise_floor_db = float(np.nanpercentile(y_roi, 10))
        snr_db = dBpk - noise_floor_db

        rows.append({
            "Distance": dist_label,
            "Peak dB": dBpk,
            "Peak Freq (MHz)": fpk_mhz,
            "Noise Floor (dB)": noise_floor_db,
            "SNR (dB)": snr_db,
            "_f_roi_len": len(f_roi)  # debug/helpful
        })

        if make_plots:
            (line,) = plt.plot(f_roi, y_roi, lw=1.6,
                               label=f"{dist_label} | SNR {snr_db:.1f} dB")
            c = line.get_color()
            plt.scatter([fpk_mhz], [dBpk], color=c, s=30, marker='x', zorder=5)
            plt.axhline(noise_floor_db, color=c, ls="--", lw=0.8, alpha=0.5)
            # annotate peak
            x0, x1 = f_roi[0], f_roi[-1]
            span = x1-x0
            dx, ha = (0.03*span, "left") if fpk_mhz <= (x1 -
                                                        0.08*span) else (-0.03*span, "right")
            plt.annotate(
                f"{dist_label}\nPeak {dBpk:.1f} dB",
                xy=(fpk_mhz, dBpk), xytext=(fpk_mhz+dx, dBpk+3),
                textcoords="data", ha=ha, va="bottom", color=c,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=c),
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec=c, alpha=0.7),
                fontsize=8
            )

    # Build table
    tbl = pd.DataFrame(rows)

    # sort by numeric distance if possible (handles "10cm", "1m", etc.)
    def _dist_key(s):
        s = str(s)
        if s.endswith("cm"):
            try:
                return float(s.replace("cm", ""))
            except:
                return np.inf
        if s.endswith("m"):
            try:
                return float(s.replace("m", ""))*100.0
            except:
                return np.inf
        return np.inf

    if "Distance" in tbl.columns:
        tbl = tbl.sort_values(by="Distance", key=lambda col: col.map(
            _dist_key)).reset_index(drop=True)

    # Reference deltas
    if len(tbl) and 0 <= ref_index < len(tbl):
        ref_peak = float(tbl.loc[ref_index, "Peak dB"])
        ref_nf = float(tbl.loc[ref_index, "Noise Floor (dB)"])
        tbl["ΔPeak vs ref (dB)"] = tbl["Peak dB"] - ref_peak
        tbl["ΔNF vs ref (dB)"] = tbl["Noise Floor (dB)"] - ref_nf

    if make_plots:
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Relative Gain (dB)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.show()

        # Summary plots
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].bar(tbl["Distance"], tbl["SNR (dB)"], color="steelblue")
        ax[0].set_title("SNR across distances")
        ax[0].set_ylabel("SNR (dB)")
        ax[0].grid(axis="y", alpha=0.3)

        if "ΔPeak vs ref (dB)" in tbl.columns:
            ax[1].plot(tbl["Distance"], tbl["ΔPeak vs ref (dB)"],
                       marker="o", label="Peak rel. to ref")
        if "ΔNF vs ref (dB)" in tbl.columns:
            ax[1].plot(tbl["Distance"], tbl["ΔNF vs ref (dB)"],
                       marker="s", label="Noise floor rel. to ref")
        ax[1].axhline(0, color="gray", lw=0.8)
        ax[1].set_title("Relative Power Change vs Reference")
        ax[1].set_ylabel("Δ Power (dB)")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()

        fig.suptitle(
            "Signal & Noise Power Comparison (your peak/NF method)", fontsize=13)
        plt.tight_layout()
        plt.show()

    return tbl


def _largest_odd_leq(n, cap):
    n = min(n, cap)

    return n if n % 2 == 1 else n - 1


def spectrum_metrics(iq, fs, fc_hz, ma_win=1001, edge_trim=5000, peak_prom=2.0, min_width_khz=30.0, exclude_width_mhz=0.15):
    """
    Compute peak dB, noise floor, and SNR from a capture using:
      - moving-average smoothing
      - median + Savitzky–Golay for peak pick
      - noise floor = 10th percentile outside ±exclude_width_mhz
    """
    fc_mhz = fc_hz/1e6
    f_mhz, mag_db = compute_fft(iq, fs)
    f_abs_mhz = f_mhz + fc_mhz

    pos = (f_mhz >= 0)
    f_half = f_abs_mhz[pos]
    y_half = mag_db[pos].astype(float)
    y_half[~np.isfinite(y_half)] = np.nan

    # moving-average smoothing
    if len(y_half) > 5:
        win = _largest_odd_leq(max(9, ma_win), len(
            y_half)-1 if (len(y_half)-1) > 9 else 9)
    else:
        win = 9
    kernel = np.ones(win, float)/win
    y_smooth = np.convolve(y_half, kernel, mode="same")

    # Trim edges
    if len(y_smooth) > 2*edge_trim:
        f_roi = f_half[edge_trim:-edge_trim]
        y_roi = y_smooth[edge_trim:-edge_trim]
    else:
        f_roi, y_roi = f_half, y_smooth

    # median + Savitzky–Golay (keep lobe shape, remove needles)
    if len(y_roi) >= 9:
        y_med = signal.medfilt(y_roi, kernel_size=7)
        sg_win = _largest_odd_leq(min(201, len(y_roi)), len(y_roi))
        sg_win = max(9, sg_win)
        y_pk = signal.savgol_filter(y_med, sg_win, polyorder=2)
    else:
        y_pk = y_roi

    # peak width bins from MIN_WIDTH_KHZ and df
    if len(f_roi) > 1:
        df_mhz = float(np.median(np.diff(f_roi)))
    else:
        df_mhz = 1e-3
    width_bins = max(1, int((min_width_khz/1e3) / max(df_mhz, 1e-9)))

    peaks, props = signal.find_peaks(
        y_pk, prominence=peak_prom, width=width_bins)
    if peaks.size == 0:
        j = int(np.nanargmax(y_pk))
    else:
        j = peaks[np.argmax(props["prominences"])]

    fpk_mhz = float(f_roi[j])
    dBpk = float(y_roi[j])

    exclude = (np.abs(f_roi - fpk_mhz) < exclude_width_mhz)
    nf_db = float(np.nanpercentile(y_roi[~exclude], 10)) if np.any(
        ~exclude) else float(np.nanpercentile(y_roi, 10))

    return {
        "peak_db": dBpk,
        "peak_freq_mhz": fpk_mhz,
        "noise_floor_db": nf_db,
        "snr_db": dBpk - nf_db,
    }


def peak_nf_from_fft(f_center_mhz, mag_db, fc_mhz, ma_win=1001, edge_trim=5000, prom_db=2.0, min_width_khz=30.0, exclude_width_mhz=0.15):
    """
    Given an FFT (centered freq axis in MHz and mag in dB), compute:
    - peak frequency (MHz, absolute)
    - peak level (dB)
    - noise floor (dB) via 10th percentile excluding ±exclude_width around peak
    - SNR (dB)
    Also returns the trimmed ROI arrays for plotting (f_roi, y_roi).

    Returns a dict with keys:
      fpk_mhz, peak_db, noise_floor_db, snr_db, f_roi, y_roi
    """
    f_abs_mhz = f_center_mhz + fc_mhz

    # use positive half
    mask = (f_center_mhz >= 0)
    f_half = f_abs_mhz[mask]
    y_half = np.asarray(mag_db[mask], dtype=float)
    y_half[~np.isfinite(y_half)] = np.nan

    # moving average smoothing
    if len(y_half) > 5:
        win = min(ma_win, len(y_half) - (len(y_half)+1) % 2)
        win = max(8, win if win % 2 == 1 else win-1)
        kernel = np.ones(win)/win
        y_smooth = np.convolve(y_half, kernel, mode="same")
    else:
        y_smooth = y_half

    # Trim edges for robust peak picking
    if len(y_smooth) > 2*edge_trim:
        f_roi = f_half[edge_trim:-edge_trim]
        y_roi = y_smooth[edge_trim:-edge_trim]
    else:
        f_roi = f_half.copy()
        y_roi = y_smooth.copy()

    # Clean needles, preserve lobe
    if len(y_roi) >= 9:
        y_clean = medfilt(y_roi, kernel_size=7)
        sg_win = min(len(y_roi) - (len(y_roi)+1) % 2, 201)  # odd ≤ 201
        sg_win = max(9, sg_win if sg_win % 2 == 1 else sg_win-1)
        y_pk = savgol_filter(y_clean, sg_win, 2)
    else:
        y_pk = y_roi

    # Peak finding with min width in bins
    df_mhz = float(np.median(np.diff(f_roi))) if len(f_roi) > 1 else 0.001
    width_bins = max(1, int((min_width_khz/1e3) / max(df_mhz, 1e-9)))

    peaks, props = find_peaks(y_pk, prominence=prom_db, width=width_bins)
    if peaks.size == 0:
        j = int(np.nanargmax(y_pk))  # fallback
    else:
        j = peaks[np.argmax(props["prominences"])]

    fpk_mhz = float(f_roi[j])
    peak_db = float(y_roi[j])

    # Noise floor via 10th percentile excluding main lobe ±exclude_width_mhz
    exclude = (np.abs(f_roi - fpk_mhz) < exclude_width_mhz)
    if np.any(~exclude):
        nf_db = float(np.nanpercentile(y_roi[~exclude], 10))
    else:
        nf_db = float(np.nanpercentile(y_roi, 10))

    snr_db = peak_db - nf_db

    return dict(
        fpk_mhz=fpk_mhz,
        peak_db=peak_db,
        noise_floor_db=nf_db,
        snr_db=snr_db,
        f_roi=f_roi,
        y_roi=y_roi,
    )

def carrier_snr_from_iq(x, fs_hz, fc_hz, **kwargs):
    """
    Convenience wrapper: compute FFT → call peak_nf_from_fft.
    kwargs are forwarded to peak_nf_from_fft (ma_win, edge_trim, etc).
    """
    from .dsp_functions import compute_fft  # local import to avoid cycles
    f_center_mhz, mag_db = compute_fft(x, fs_hz)  # centered MHz axis
    return peak_nf_from_fft(f_center_mhz, mag_db, fc_hz/1e6, **kwargs)