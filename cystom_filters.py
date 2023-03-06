# -*- coding: utf-8 -*-
import numpy as np
from librosa import stft, amplitude_to_db

def hz_to_log(frequencies):
    frequencies = np.asanyarray(frequencies)
    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1024  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def log_to_hz(mels, *, htk=False):

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1024  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def log_filter(
    sr: float = 22050,
    n_fft: int = 1024,
    n_modes: int = 128,
    fmin: float = 0.0,
    fmax: float = None,
    htk: bool = False,
    base: float = 2,
    norm="slaney",
    dtype=np.float32,
):

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_modes = int(n_modes)
    weights = np.zeros((n_modes, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    log_f = log_to_hz(
        np.linspace(start=hz_to_log(fmin), stop=hz_to_log(fmax), num=n_modes + 2)
    )
    # print(log_f)
    # log_f[0] = 0
    # print(log_f)
    fdiff = np.diff(log_f)
    ramps = np.subtract.outer(log_f, fftfreqs)
    # print(ramps)
    for i in range(n_modes):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # if isinstance(norm, str) and norm == "slaney":
    enorm = 2.0 / (log_f[2 : n_modes + 2] - log_f[:n_modes])
    weights *= enorm[:, np.newaxis]
    return weights


def logspec(y, sr=22500, n_fft=1024, part='real'):
    S_complex = stft(y, n_fft=1024, hop_length=512)
    # print(S_raw.shape)
    log_f = log_filter()
    if part == 'real':
        return np.einsum(
            "...ft,mf->...mt",
            amplitude_to_db(np.abs(S_complex.real)),
            log_f,
            optimize=True,
        )
    if part == 'phase':
        return np.einsum(
            "...ft,mf->...mt", np.angle(S_complex.imag), log_f, optimize=True
        )
