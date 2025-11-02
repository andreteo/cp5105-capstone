import numpy as np
import os


def load_iq_auto(path):
    size = os.path.getsize(path)

    for dtype in (np.float32, np.float64, np.int16):
        if size % np.dtype(dtype).itemsize != 0:
            continue
        raw = np.fromfile(path, dtype=dtype)
        if raw.size % 2:  # must be even for I/Q pairs
            continue
        I, Q = raw.reshape(-1, 2).T

        # Normalize int16 if needed
        if dtype == np.int16:
            x = (I.astype(np.float32) + 1j*Q.astype(np.float32)) / 32768.0
        else:
            x = I.astype(np.float32) + 1j*Q.astype(np.float32)

        # reject if mostly NaN/Inf or denormals
        bad = np.mean(~np.isfinite(np.real(x))) + \
            np.mean(~np.isfinite(np.imag(x)))
        if bad < 0.01 and np.nanmax(np.abs(x)) > 1e-6:
            return x  # looks sane

    raise ValueError("Could not determine dtype; file may be corrupted.")



def interleave_iq(i_data, q_data, dtype=np.int16):
    iq = np.empty((i_data.size + q_data.size), dtype=dtype)
    iq[0::2] = i_data
    iq[1::2] = q_data

    return iq
