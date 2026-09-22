#!/usr/bin/env python3
"""Acoustic self-calibration: measure the mic geometry with sound, not a ruler.

    python calibration/calibrate.py add 0         speaker roughly north  -> capture, append
    python calibration/calibrate.py add 45        ... repeat at 6-10 spots around the array
    python calibration/calibrate.py list          show what's been collected
    python calibration/calibrate.py solve         fit mic positions, print MICS
    python calibration/calibrate.py solve --write   ... and save them to calibration/mics.json
    python calibration/calibrate.py reset         start over
    python calibration/calibrate.py selftest      prove the solver on synthetic data

The delay between two mics peaks when the source is in line with them, and that
peak delay times the speed of sound IS their separation. Sweeping a source round
the array therefore measures every pairwise distance directly. The nominal angle
you give each spot is only a starting guess - the fit solves for the true source
directions too, so speaker placement does not need to be precise.

Spots whose delays are not mutually consistent (transitivity) are reverb-
dominated and are rejected automatically, so straying into the dead zone is safe.
"""
import sys, json, itertools, pathlib
import numpy as np
from scipy.optimize import least_squares

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import spideysense as s
from wavefront import gcc, pink

CAL = ROOT / "calibration" / "calib.json"     # your spots; not in git
LAB = "SWNE"
MAX_TRANS_US = 15.0       # transitivity rms above this = not a single wavefront


def load():
    return json.loads(CAL.read_text()) if CAL.exists() else []


def transitivity(lags):
    get = lambda i, j: lags[s.PAIRS.index((i, j))] if (i, j) in s.PAIRS \
        else -lags[s.PAIRS.index((j, i))]
    r = [get(i, j) + get(j, k) - get(i, k)
         for i, j, k in itertools.combinations(range(4), 3)]
    return float(np.sqrt(np.mean(np.square(r))))


def speak(text):
    """Audible cue via macOS TTS on the default output. Blocks until spoken."""
    import subprocess
    subprocess.run(["say", text])


def add(nominal, device=2, voice=False):
    import sounddevice as sd
    sd.play(pink(), s.FS, device=device, loop=True)
    try:
        X = np.concatenate([b for b, _ in zip(s.capture(), range(6))])
    finally:
        sd.stop()
    lags, peaks = zip(*(gcc(X[:, i], X[:, j], s.FS) for i, j in s.PAIRS))
    lags = [float(v) for v in lags]
    t = transitivity(lags)
    lvl = float(20 * np.log10(np.sqrt((X ** 2).mean()) + 1e-12))
    ok = t < MAX_TRANS_US
    print(f"spot @ {nominal:.0f} deg   level {lvl:.1f} dBFS   "
          f"transitivity {t:.1f} us   corr {np.mean(peaks):.3f}")
    print("   " + "  ".join(f"{LAB[i]}{LAB[j]} {l:+6.1f}" for (i, j), l in zip(s.PAIRS, lags)))
    if not ok:
        print(f"   REJECTED - not a clean wavefront (>{MAX_TRANS_US:.0f} us). "
              "Reverb-dominated; move the speaker.")
        if voice:
            speak("rejected")
        return False
    data = load()
    for k, prev in enumerate(data):
        if np.sqrt(np.mean((np.array(lags) - prev["lags"]) ** 2)) < 5.0:
            print(f"   REJECTED - same position as spot {k} ({prev['nominal']:.0f} deg). "
                  "Speaker didn't move.")
            if voice:
                speak("didn't move")
            return False
    data.append({"nominal": nominal, "lags": lags, "trans": t, "level": lvl})
    CAL.write_text(json.dumps(data, indent=1))
    print(f"   accepted -> {len(data)} spots collected")
    if voice:
        speak("accepted")
    return True


def sweep(angles, delay=2.0, device=2):
    """Walk the circle hands-free: it speaks the next angle, waits, captures,
    speaks the verdict. Silence = move, pink noise = hold still."""
    import time
    speak("starting. move to " + f"{angles[0]:.0f}")
    for k, a in enumerate(angles):
        print(f"\n[{k+1}/{len(angles)}] move speaker to {a:.0f} deg ...", flush=True)
        if k:
            speak(f"move to {a:.0f}")
        time.sleep(delay)
        speak("hold")
        add(a, device, voice=True)
    n = len(load())
    print(f"\nsweep done - {n} spots collected. Run: solve")
    speak(f"done. {n} spots.")


def fit(data, c=s.C):
    K = len(data)
    lags = np.array([d["lags"] for d in data])
    nom = np.radians([d["nominal"] for d in data])
    x0 = np.concatenate([s.MICS.ravel(), nom])

    def resid(x):
        P, th = x[:8].reshape(4, 2), x[8:]
        u = np.stack([np.sin(th), np.cos(th)], 1)               # K x 2
        pred = np.stack([(P[j] - P[i]) @ u.T for i, j in s.PAIRS], 1) / c * 1e6
        gauge = [P[:, 0].mean() * 1e5, P[:, 1].mean() * 1e5,      # centroid at origin
                 P[2, 0] * 1e5]                                    # N mic on +y axis
        return np.concatenate([(pred - lags).ravel(), gauge])

    r = least_squares(resid, x0, x_scale="jac")
    P, th = r.x[:8].reshape(4, 2), r.x[8:]

    # rotate whole solution so the fitted angles agree with the nominal compass on average
    off = np.angle(np.mean(np.exp(1j * (th - nom))))
    ca, sa = np.cos(off), np.sin(off)
    P = P @ np.array([[ca, -sa], [sa, ca]]).T
    th = th - off
    res = resid(r.x)[:-3].reshape(K, 6)
    return P, np.degrees(th) % 360, res


def solve(write=False):
    data = load()
    if len(data) < 3:
        sys.exit(f"need at least 3 spots, have {len(data)}")
    P, th, res = fit(data)
    rms = np.sqrt((res ** 2).mean())
    print(f"{len(data)} spots, {res.size} delays -> 8 coords + {len(data)} angles\n")
    print(f"fit residual rms {rms:.1f} us   "
          f"(measurement floor ~{np.mean([d['trans'] for d in data]):.1f} us)")
    if rms > 3 * max(1.0, np.mean([d["trans"] for d in data])):
        print("   WARNING: residual well above measurement noise - "
              "a rigid 4-mic array does not fully explain these delays")

    print("\nspot   nominal   fitted   placement_err   spot_rms_us")
    for d, a, r in zip(data, th, res):
        e = (a - d["nominal"] + 180) % 360 - 180
        print(f"       {d['nominal']:6.0f}   {a:6.1f}      {e:+6.1f}        {np.sqrt((r**2).mean()):5.1f}")

    print("\nmic   x_in     y_in      (+y = N, +x = E)")
    for k in range(4):
        print(f" {k+1}{LAB[k]}  {P[k,0]/0.0254:+7.3f}  {P[k,1]/0.0254:+7.3f}")

    print("\npair   acoustic_in   model_in   ruler_in")
    ruler = {"SW": 1.85, "SN": 3.0, "SE": 1.8, "WN": 1.85, "WE": 2.8, "NE": 1.8}
    for i, j in s.PAIRS:
        d = np.linalg.norm(P[i] - P[j]) / 0.0254
        m = np.linalg.norm(s.MICS[i] - s.MICS[j]) / 0.0254
        print(f"{LAB[i]}-{LAB[j]}    {d:6.3f}       {m:6.3f}     {ruler[LAB[i]+LAB[j]]:5.2f}")

    lim = min(s.C / (2 * np.linalg.norm(P[i] - P[j])) for i, j in s.PAIRS)
    print(f"\nspatial-alias limit {lim:.0f} Hz   (BAND upper is {s.BAND[1]})")
    if P[3, 0] < 0:
        print("WARNING: E mic fitted at negative x - handedness flipped, check initial guess")

    mics = [[round(float(P[k, 0]), 5), round(float(P[k, 1]), 5)] for k in range(4)]
    print("\nMICS =", mics)
    if write:
        s.MICS_FILE.write_text(json.dumps(mics))
        print(f"\nwritten to {s.MICS_FILE} - spideysense.py loads it on start")


def selftest():
    """Synthesise delays from a known geometry with noise; solver must recover it."""
    rng = np.random.default_rng(0)
    true = np.array([[0.002, -0.038], [-0.030, 0.003], [0.0, 0.036], [0.033, -0.001]])
    true -= true.mean(0)
    data = []
    for nom in [0, 40, 95, 130, 200, 250, 300, 340]:
        az = np.radians(nom + rng.normal(0, 5))                # sloppy placement
        u = np.array([np.sin(az), np.cos(az)])
        lags = [(true[j] - true[i]) @ u / s.C * 1e6 + rng.normal(0, 1.0)
                for i, j in s.PAIRS]
        data.append({"nominal": nom, "lags": lags, "trans": 1.0, "level": -40})
    P, th, res = fit(data)
    # compare pairwise distances (invariant to the rotation gauge)
    dt = [np.linalg.norm(true[i] - true[j]) for i, j in s.PAIRS]
    df = [np.linalg.norm(P[i] - P[j]) for i, j in s.PAIRS]
    worst = max(abs(a - b) for a, b in zip(dt, df)) * 1000
    print(f"residual rms {np.sqrt((res**2).mean()):.2f} us   "
          f"worst pair-distance error {worst:.2f} mm")
    assert worst < 1.0, "selftest FAILED"
    print("OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else "list"
    if cmd == "add":
        add(float(a[1]), int(a[2]) if len(a) > 2 else 2)
    elif cmd == "sweep":
        rest, delay = a[1:], 2.0
        if "--delay" in rest:
            k = rest.index("--delay"); delay = float(rest[k + 1]); del rest[k:k + 2]
        sweep([float(v) for v in rest], delay)
    elif cmd == "solve":
        solve("--write" in a)
    elif cmd == "list":
        for d in load():
            print(f"  {d['nominal']:6.0f} deg   trans {d['trans']:4.1f} us   {d['level']:.1f} dBFS")
        print(f"{len(load())} spots")
    elif cmd == "reset":
        CAL.unlink(missing_ok=True); print("cleared")
    elif cmd == "selftest":
        selftest()
