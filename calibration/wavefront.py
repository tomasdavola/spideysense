#!/usr/bin/env python3
"""Is the array seeing a real plane wave, or just reverb?

    python calibration/wavefront.py <true_bearing_deg> [speaker_device]

Transitivity (tau_AB + tau_BC == tau_AC) must hold for any single plane wave,
regardless of whether the assumed mic geometry is right. It cannot hold for a
diffuse/reverberant field. That separates a signal problem from a code problem.
"""
import sys, itertools
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import spideysense as s


def pink(secs=10, fs=44100):
    """Pink noise (1/f power), peak 0.5, for the probe speaker. Generated, not shipped: it's 10 s of
    filtered white noise, so no WAV needs to live in the repo."""
    X = np.fft.rfft(np.random.default_rng(0).standard_normal(int(secs * fs)))
    X[1:] /= np.sqrt(np.arange(1, len(X)))
    X[0] = 0
    x = np.fft.irfft(X)
    return (0.5 * x / np.abs(x).max()).astype(np.float32)

LAB = "SWNE"


def gcc(a, b, fs, mx=32):
    n = 1 << int(np.ceil(np.log2(len(a) * 2)))
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    f = np.fft.rfftfreq(n, 1 / fs)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    R[(f < s.BAND[0]) | (f > s.BAND[1])] = 0
    c = np.fft.irfft(R, n)
    c = np.concatenate([c[-mx:], c[:mx + 1]])
    k = int(np.argmax(c))
    pk = c[k]
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-20)
    return (k - mx) / fs * 1e6, pk


def main():
    az = float(sys.argv[1])
    dev = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    sd.play(pink(), 44100, device=dev, loop=True)
    try:
        X = np.concatenate([b for b, _ in zip(s.capture(), range(6))])
    finally:
        sd.stop()

    u = np.array([np.sin(np.radians(az)), np.cos(np.radians(az))])
    lag, peak = {}, {}
    print(f"source at {az:.0f} deg   {len(X)/s.FS:.1f}s   "
          f"level {20*np.log10(np.sqrt((X**2).mean())+1e-12):.1f} dBFS\n")
    print("pair     measured   predicted     err    corr_peak")
    for i, j in s.PAIRS:
        d, p = gcc(X[:, i], X[:, j], s.FS)
        pred = (s.MICS[j] - s.MICS[i]) @ u / s.C * 1e6
        lag[(i, j)] = d
        peak[(i, j)] = p
        print(f"{LAB[i]}-{LAB[j]}   {d:+8.1f}   {pred:+8.1f}  {d-pred:+7.1f}     {p:.3f}")

    get = lambda i, j: lag[(i, j)] if (i, j) in lag else -lag[(j, i)]
    print("\ntriangle   tau_ij+tau_jk-tau_ik   (0 = consistent plane wave)")
    res = []
    for i, j, k in itertools.combinations(range(4), 3):
        r = get(i, j) + get(j, k) - get(i, k)
        res.append(r)
        print(f"{LAB[i]}{LAB[j]}{LAB[k]}          {r:+8.1f} us")

    rms = np.sqrt(np.mean(np.square(res)))
    span = max(abs((s.MICS[a] - s.MICS[b]) @ u / s.C * 1e6) for a, b in s.PAIRS)
    print(f"\ntransitivity rms {rms:.1f} us   (aperture {span:.0f} us, "
          f"1 sample = {1e6/s.FS:.1f} us)")
    print(f"mean corr peak   {np.mean(list(peak.values())):.3f}")
    if rms < 15:
        print("=> COHERENT plane wave. Delays are trustworthy; compare the err column.")
    elif rms < 40:
        print("=> PARTIALLY coherent. Direct path present but reverb is competing.")
    else:
        print("=> INCOHERENT. No single wavefront - reverb dominated, bearings meaningless.")


if __name__ == "__main__":
    main()
