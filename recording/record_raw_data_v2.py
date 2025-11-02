import sys, os, json, time
sys.path.append("..")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sdr.sdr_config import load_config
from sdr.sdr_init import initialize_pluto_sdr
from auxilliary_funcs.iq_functions import interleave_iq
from auxilliary_funcs.dsp_functions import compute_fft


# ---------- paths (dated output dir) ----------
from datetime import datetime

today_str = datetime.now().strftime("%Y-%m-%d")   # e.g., 2025-10-05
out_dir = Path(f"raw_{today_str}")
out_dir.mkdir(parents=True, exist_ok=True)




# ---------- config & SDR ----------
config = load_config()
sdr_cfg = config["sdr_config"]
rec_cfg = config["record_config"]

fs        = int(sdr_cfg["sample_rate"])
fc        = float(sdr_cfg["center_frequency"])
seconds   = int(rec_cfg["duration_sec"])
buf_size  = int(sdr_cfg["rx_buffer_size"])
dtype     = getattr(np, rec_cfg["datatype"])   # e.g., np.float32, np.int16

sdr = initialize_pluto_sdr(sdr_cfg)
sdr.tx_destroy_buffer()                        # ensure TX is idle

# ---------- paths ----------
dist_tag   = "diffrooms_slowattackgain"  # <- set your tag here
out_dir = Path(f"raw_{today_str}")
out_dir.mkdir(parents=True, exist_ok=True)
stem       = f"plutosdr-{dist_tag}-{fs}-Hz-np{dtype.__name__}-{seconds}s"
bin_path   = out_dir / f"{stem}.bin"
meta_path  = out_dir / f"{stem}.json"
png_path   = out_dir / f"{stem}.png"

# ---------- capture plan ----------
total_complex_samples = fs * seconds
bytes_per_real    = np.dtype(dtype).itemsize
bytes_per_complex = 2 * bytes_per_real
expected_bytes    = total_complex_samples * bytes_per_complex

print(f"Target: {seconds}s @ {fs:,} S/s → {total_complex_samples:,} complex samples "
      f"({expected_bytes/1024/1024:.2f} MB as interleaved {dtype.__name__}).")

# ---------- record, writing chunks directly ----------
written = 0
preview_chunk = None

t0 = time.time()

try:
    with open(bin_path, "wb") as f:
        while written < total_complex_samples:
            # Read one SDR buffer (complex64 from Pluto typically)
            x = sdr.rx()
            if x is None or len(x) == 0:
                print("WARN: empty read; continuing")
                continue

            # Trim final chunk if we’d exceed target
            n = len(x)
            if written + n > total_complex_samples:
                n = total_complex_samples - written
                x = x[:n]

            # Optional quick sanity checks for bad data
            if not np.isfinite(x).all():
                print("WARN: non-finite samples detected (NaN/Inf).")

            # Interleave to desired on-disk dtype (I0,Q0,I1,Q1, …)
            # Faster than building big arrays: write chunk by chunk.
            i = x.real.astype(dtype, copy=False)
            q = x.imag.astype(dtype, copy=False)
            iq_interleaved = interleave_iq(i, q)          # returns 1-D array [I0,Q0,I1,Q1,...]
            f.write(iq_interleaved.tobytes())

            # keep a small preview buffer for FFT
            preview_chunk = x if preview_chunk is None else preview_chunk

            written += n

    os.sync() if hasattr(os, "sync") else None
finally:
    pass  # place for sdr cleanup if you have one

dt = time.time() - t0
mb_written = os.path.getsize(bin_path) / (1024*1024)

print(f"Wrote {written:,} complex samples in {dt:.2f}s -> {mb_written:.2f} MB")
if mb_written*1024*1024 != expected_bytes:
    print(f"NOTE: file size differs from theoretical ({expected_bytes/1024/1024:.2f} MB). "
          "This is ok if you intentionally trimmed the last chunk.")

# ---------- save metadata sidecar ----------
meta = {
    "center_frequency_hz": fc,
    "sample_rate_hz": fs,
    "duration_s": seconds,
    "rx_buffer_size": buf_size,
    "dtype": f"np.{dtype.__name__}",
    "layout": "interleaved_IQ",
    "complex_samples": int(written),
    "bytes_per_real": int(bytes_per_real),
    "bytes_per_complex": int(bytes_per_complex),
    "filename": bin_path.name,
}
meta_path.write_text(json.dumps(meta, indent=2))
print(f"Saved metadata: {meta_path}")

# ---------- quick FFT preview (on the preview chunk only) ----------
try:
    f_axis_mhz, mag_db = compute_fft(preview_chunk, fs)
    fc_mhz = fc / 1e6                       # <<< use /1e6 (not *10e-7)
    plt.figure(figsize=(10, 6))
    plt.plot(f_axis_mhz + fc_mhz, mag_db)
    plt.title("PlutoSDR Spectrum Preview")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Relative Gain (dB)")
    plt.xlim(fc_mhz - (fs/2)/1e6, fc_mhz + (fs/2)/1e6)
    plt.ylim(-140, 10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    print(f"Saved preview plot: {png_path}")
except Exception as e:
    print(f"Preview FFT failed (non-fatal): {e}")
