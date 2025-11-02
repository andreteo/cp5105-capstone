import sys
sys.path.append("..")
from pathlib import Path
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sdr.sdr_config import load_config
from sdr.sdr_init import initialize_pluto_sdr
from auxilliary_funcs.iq_functions import interleave_iq
from auxilliary_funcs.dsp_functions import compute_fft

config = load_config()
sdr_config = config["sdr_config"]
record_config = config["record_config"]

sample_rate = int(sdr_config["sample_rate"])
duration_sec = int(record_config["duration_sec"])

sdr = initialize_pluto_sdr(sdr_config)

# Number of segments to collect
buf_size = sdr_config["rx_buffer_size"]
total_samples = int(sample_rate * duration_sec)
num_chunks = total_samples // buf_size

# Start PltutoSDR RX
sdr.tx_destroy_buffer()  # Ensure TX is idle

start = time.time()


rx_all = []
datatype = getattr(np, record_config["datatype"])
dtype_str = f"np{datatype.__name__}"

Path("raw").mkdir(exist_ok=True)

bin_filename = Path(
    os.getcwd()) / "raw" / f"plutosdr-300cm-{int(sample_rate)}-MHz-{dtype_str}-{duration_sec}secs_withbreathing.bin"
img_filename = str(bin_filename)[:-4] + '.png'

print(f"Capturing {num_chunks} buffers of size {buf_size}...")

for _ in range(num_chunks):
    rx_chunk = sdr.rx()
    rx_all.append(rx_chunk)

rx_all = np.concatenate(rx_all)

# Perform range FFT
iq_data_freq_axis, magnitude_dB = compute_fft(
    rx_all, sample_rate)

center_freq_mhz = sdr_config["center_frequency"] * 10e-7
iq_data_freq_absolute = iq_data_freq_axis + center_freq_mhz

print(f"Captured {len(rx_all)} samples.")

rx_i = datatype(rx_all.real)
rx_q = datatype(rx_all.imag)

iq_interleaved = interleave_iq(rx_i, rx_q)

with open(bin_filename, "wb") as f:
    iq_interleaved.tofile(f)

# Plotting the Range FFT
plt.figure(figsize=(10, 6))
plt.plot(iq_data_freq_absolute, magnitude_dB)
plt.title("PlutoSDR Spectrum Analyzer")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Relative Gain (dB)")
plt.grid(True)
plt.xlim(center_freq_mhz - (sample_rate/2e6),
          center_freq_mhz + (sample_rate/2e6))
plt.ylim(-140, 10)
plt.tight_layout()

# plt.show()

# === Save instead of show ===
plt.savefig(img_filename)
plt.close()

print(
    f"Saved raw binary to: {bin_filename} ({os.path.getsize(bin_filename)/1024:.2f} KB)")
