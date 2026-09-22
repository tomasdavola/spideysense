#!/usr/bin/env python3
"""Acoustic direction map from a 4-mic diamond array (SRP-PHAT).  Run under .venv."""
import sys, os, json, pathlib, itertools
import numpy as np

# --- config -----------------------------------------------------------------
DEVICE  = os.environ.get("SPIDEY_DEVICE", "ZOOM AMS-44")   # 4-channel input, matched by name
                                                           # (indices renumber when devices come and go)
FS      = 44100
C       = 343.0
BAND    = (300, 1350)    # 1382 Hz is the S-N spatial-alias limit; above it, phase wraps
NFFT    = 16384          # 372 ms
NGRID   = 360

# Mic positions, metres, centred, +y=North +x=East.  ch1=S ch2=W ch3=N ch4=E
# A nominal diamond of ARRAY_M across until calibration/mics.json exists. Measure yours with
# `calibration/calibrate.py solve --write`: a tape measure is not good enough (3 mm is 25 deg of
# phase at 8 kHz), and the file survives re-mounting the mics.
ARRAY_M = float(os.environ.get("SPIDEY_ARRAY_M", 0.12))
MICS_FILE = pathlib.Path(__file__).parent / "calibration" / "mics.json"
MICS = (np.array(json.loads(MICS_FILE.read_text())) if MICS_FILE.exists() else
        ARRAY_M / 2 * np.array([[0, -1], [-1, 0], [0, 1], [1, 0]], float))
PAIRS = list(itertools.combinations(range(4), 2))


def capture(n=NFFT):
    """Yield n-frame blocks of 4-channel audio from the interface.

    sounddevice/PortAudio, not ffmpeg: ffmpeg's avfoundation input buffers a single
    frame and silently drops ~7% of audio whenever its read loop is late (measured
    0.93 on a click train; PortAudio measured 0.9996). Run under .venv.
    """
    import queue
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("needs sounddevice: run with .venv/bin/python")
    q = queue.Queue()
    stream = sd.InputStream(device=DEVICE, samplerate=FS, channels=4, dtype="float32",
                            blocksize=2048, callback=lambda x, f, t, st: q.put(x.copy()))
    buf = np.zeros((0, 4), np.float32)
    with stream:
        while True:
            while len(buf) < n:
                buf = np.concatenate([buf, q.get()])
            yield buf[:n].astype(float)
            buf = buf[n:]


def _steering():
    """Precompute phase ramps for every bearing x mic-pair. Runs once."""
    f = np.fft.rfftfreq(NFFT, 1 / FS)
    keep = (f >= BAND[0]) & (f <= BAND[1])
    f = f[keep]
    th = np.linspace(0, 2 * np.pi, NGRID, endpoint=False)
    u = np.stack([np.sin(th), np.cos(th)], 1)      # az 0=North(+y), 90=East(+x)
    steer = [np.exp(2j * np.pi * np.outer(f, (MICS[j] - MICS[i]) @ u.T / C)).T
             for i, j in PAIRS]
    return keep, th, steer


KEEP, THETA, STEER = _steering()


def srp_phat(x):
    """Steered-response power over azimuth. x: (NFFT, 4) -> (NGRID,)"""
    X = np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)[KEEP]
    power = np.zeros(NGRID)
    for k, (i, j) in enumerate(PAIRS):
        g = X[:, i] * np.conj(X[:, j])
        power += np.real(STEER[k] @ (g / (np.abs(g) + 1e-12)))
    return power / len(PAIRS)


def render(power, level):
    peak = np.degrees(THETA[np.argmax(power)])
    prominence = (power.max() - power.mean()) / (power.std() + 1e-12)
    norm = (power - power.min()) / (power.max() - power.min() + 1e-12)
    rows = []
    for i in range(0, NGRID, NGRID // 36):
        a = i * 360 // NGRID
        tag = {0: "N", 90: "E", 180: "S", 270: "W"}.get(a, "")
        rows.append(f"{a:3d}{tag:1} |{'#' * int(norm[i] * 40)}")
    print("\033[2J\033[H", end="")
    print(f"  bearing {peak:5.1f}   prominence {prominence:4.2f}"
          f"   level {level:6.1f} dBFS\n")
    print("\n".join(rows), flush=True)


def selftest():
    """Synthesise sources at known bearings and check we recover them."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for true_az in [0, 30, 45, 90, 135, 180, 250, 315]:
        u = np.array([np.sin(np.radians(true_az)), np.cos(np.radians(true_az))])
        s = rng.standard_normal(NFFT * 3)
        x = np.stack([np.interp(np.arange(NFFT) + NFFT + (MICS[k] @ u) / C * FS,
                                np.arange(len(s)), s) for k in range(4)], 1)
        est = np.degrees(THETA[np.argmax(srp_phat(x))])
        err = abs((est - true_az + 180) % 360 - 180)
        worst = max(worst, err)
        print(f"  true {true_az:5.1f}   est {est:6.1f}   err {err:5.2f}")
    assert worst < 2.0, f"selftest FAILED: worst error {worst:.2f} deg"
    print(f"OK - worst error {worst:.2f} deg")


def check():
    """Verify the 4 inputs are genuinely independent (catches MONO/link switches)."""
    blk = np.concatenate([b for b, _ in zip(capture(NFFT), range(6))])
    print(f"{len(blk) / FS:.1f}s captured\n")
    print("pair   corr   null_depth   verdict")
    bad = False
    for i, j in PAIRS:
        a, b = blk[:, i], blk[:, j]
        corr = np.corrcoef(a, b)[0, 1]
        null = 20 * np.log10(np.sqrt(((a - b) ** 2).mean()) /
                             (np.sqrt((a ** 2).mean()) + 1e-12) + 1e-12)
        dup = null < -25
        bad |= dup
        print(f"{i+1}-{j+1}  {corr:6.3f}   {null:7.1f} dB   "
              f"{'*** DUPLICATE - MONO IS ON ***' if dup else 'independent'}")
    print("\nFAIL - disable MONO on the interface" if bad else "\nOK - all 4 inputs independent")


def probe(expected, secs=12):
    """Log bearings for a source at a KNOWN bearing, then diagnose any error."""
    expected = float(expected)
    print(f"source should be at {expected:.0f} deg - make broadband noise "
          f"(hiss/clap/pink noise) for {secs}s\n")
    az, pr = [], []
    for blk in capture():
        p = srp_phat(blk)
        prom = (p.max() - p.mean()) / (p.std() + 1e-12)
        a = np.degrees(THETA[np.argmax(p)])
        az.append(a); pr.append(prom)
        print(f"  {a:6.1f} deg   prominence {prom:4.2f}"
              f"   {'<-- weak, make more noise' if prom < 2.0 else ''}")
        if len(az) * NFFT / FS >= secs:
            break

    strong = [a for a, q in zip(az, pr) if q >= 2.0] or az
    z = np.mean(np.exp(1j * np.radians(strong)))
    m, R = np.degrees(np.angle(z)) % 360, abs(z)
    wrap = lambda d: (d + 180) % 360 - 180
    print(f"\nmeasured {m:.1f} deg   R={R:.3f}   ({len(strong)}/{len(az)} strong frames)")
    if R < 0.7:
        print("INCONCLUSIVE - bearings too scattered, need a louder source")
        return
    rot, mir = wrap(m - expected), wrap(m + expected)
    if abs(rot) < 15:
        print(f"CORRECT - off by {rot:+.1f} deg")
    elif abs(mir) < 15:
        print(f"MIRRORED - handedness is flipped (residual {mir:+.1f} deg)")
    elif abs(abs(rot) - 180) < 15:
        print("180 OFF - TDOA sign error, or mirrored+rotated")
    else:
        print(f"ROTATED by {rot:+.1f} deg")
    print("re-run at a different bearing to confirm which")


def live():
    for blk in capture():
        render(srp_phat(blk), 20 * np.log10(np.sqrt((blk ** 2).mean()) + 1e-12))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "map"
    {"selftest": selftest, "check": check, "map": live,
     "probe": probe}[cmd](*sys.argv[2:])
