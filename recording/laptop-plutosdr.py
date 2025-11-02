# Logs PlutoSDR I/Q as interleaved int16 with UTC timestamps + metadata and per-chunk timing.
import sys
sys.path.append("..")

import argparse, json, os, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sdr.sdr_config import load_config
from sdr.sdr_init import initialize_pluto_sdr
from auxilliary_funcs.iq_functions import interleave_iq
from auxilliary_funcs.dsp_functions import compute_fft

def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def main():
    ap = argparse.ArgumentParser(description="Record PlutoSDR I/Q to raw int16.")
    ap.add_argument("--duration", type=float, default=None, help="Seconds to record (override config)")
    ap.add_argument("--session", type=str, default=None, help="Optional session ID (else UTC now)")
    ap.add_argument("--dist", type=str, default="NA", help="Optional label (e.g., 30cm)")
    ap.add_argument("--outdir", type=str, default="sessions", help="Root output directory")
    ap.add_argument("--dtype", type=str, default="int16", choices=["int16"], help="On-disk dtype (int16 recommended)")
    args = ap.parse_args()

    config   = load_config()
    sdr_cfg  = config["sdr_config"]
    rec_cfg  = config["record_config"]

    fs       = int(sdr_cfg["sample_rate"])
    fc       = float(sdr_cfg["center_frequency"])
    seconds  = float(args.duration if args.duration is not None else rec_cfg["duration_sec"])
    buf_size = int(sdr_cfg["rx_buffer_size"])

    # Init SDR
    sdr = initialize_pluto_sdr(sdr_cfg)
    sdr.tx_destroy_buffer()

    # Session paths
    session_id = args.session or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(args.outdir) / f"session_{session_id}" / "sdr"
    root.mkdir(parents=True, exist_ok=True)

    stem = f"pluto_{args.dist}_{fs}Sps_{args.dtype}_{int(seconds)}s"
    bin_path   = root / f"{stem}.bin"
    meta_path  = root / f"{stem}.json"
    png_path   = root / f"{stem}.png"
    tlog_path  = root / f"{stem}.chunks.ndjson"   # per-chunk timestamps for fine alignment

    print(f"[sdr] Session: {session_id}")
    print(f"[sdr] Writing: {bin_path}")

    # Capture plan
    total_complex = int(fs * seconds)
    # On-disk format: interleaved int16 IQ with full-scale scaling
    scale = 32767.0
    bytes_per_complex = 4  # 2*int16

    written = 0
    preview = None

    t0_epoch = time.time()
    t0_iso   = utc_now_iso()

    with open(bin_path, "wb") as f, open(tlog_path, "w") as flog:
        print(f"[sdr] Recording {seconds}s @ {fs:,} S/s… Ctrl-C to stop.")
        try:
            while written < total_complex:
                block_start_epoch = time.time()
                x = sdr.rx()  # complex float block, e.g., complex64
                if x is None or len(x) == 0:
                    print("[sdr] WARN: empty read; continuing")
                    continue

                n = len(x)
                if written + n > total_complex:
                    n = total_complex - written
                    x = x[:n]

                # Save first good block for preview
                if preview is None:
                    preview = x.copy()

                if not np.isfinite(x).all():
                    print("[sdr] WARN: non-finite samples detected.")

                I = np.real(x)
                Q = np.imag(x)
                iq_interleaved = interleave_iq(I, Q)
                f.write(iq_interleaved.tobytes())

                # Per-chunk timing log (NDJSON)
                rec = {
                    "chunk_index": int(written // buf_size),
                    "sample_start": int(written),
                    "sample_count": int(n),
                    "epoch_start_s": float(block_start_epoch),  # already there
                    "epoch_end_s": float(time.time()),          # add this
                    "chunk_duration_s": float(n / fs),          # and this
                    "iso_start_utc": utc_now_iso()
                }
                flog.write(json.dumps(rec) + "\n")

                written += n
        except KeyboardInterrupt:
            print("[sdr] Interrupted by user.")

    dt = time.time() - t0_epoch
    sz_mb = os.path.getsize(bin_path) / (1024*1024)
    print(f"[sdr] Wrote {written:,} complex samples in {dt:.2f}s ({sz_mb:.2f} MB)")

    # Metadata JSON
    meta = {
        "session_id": session_id,
        "label": args.dist,
        "center_frequency_hz": fc,
        "sample_rate_hz": fs,
        "duration_s_target": seconds,
        "duration_s_recorded": written / fs,
        "rx_buffer_size": buf_size,
        "dtype_on_disk": "int16_interleaved_IQ",
        "scale_int16": int(scale),
        "start_epoch_s": t0_epoch,
        "start_iso_utc": t0_iso,
        "complex_samples": int(written),
        "bytes_per_complex": int(bytes_per_complex),
        "binary_filename": bin_path.name,
        "chunk_timing_log": tlog_path.name
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[sdr] Meta saved: {meta_path}")

    # Quick FFT preview
    if preview is not None:
        try:
            f_axis_mhz, mag_db = compute_fft(preview, fs)
            fc_mhz = fc / 1e6
            plt.figure(figsize=(10, 5))
            plt.plot(f_axis_mhz + fc_mhz, mag_db, lw=1)
            plt.title("PlutoSDR Spectrum Preview")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Relative Gain (dB)")
            plt.xlim(fc_mhz - (fs/2)/1e6, fc_mhz + (fs/2)/1e6)
            plt.ylim(-140, 10)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close()
            print(f"[sdr] Preview saved: {png_path}")
        except Exception as e:
            print(f"[sdr] Preview failed (non-fatal): {e}")

if __name__ == "__main__":
    main()