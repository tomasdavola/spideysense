#!/usr/bin/env python3
"""Listen to individual mics on the 4-channel array.

    hear.py              live meter, all 4 channels
    hear.py 3            hear ch3, meter on all four
    hear.py 0            hear all four summed
    hear.py 3 -g 20      +20 dB (mics idle near -45 dBFS; you need the boost)
    hear.py -l           list devices
    hear.py 3 -r 5       record 5s -> ch3.wav

USE EARPHONES. Monitoring an array through speakers is a feedback loop.

Input and output must share ONE duplex stream. Piping between two devices
(interface at 44100, headphones at 48000) has no common clock, so it plays
about a second of buffer and then silently starves.
"""
import sys, os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

DEVICE = os.environ.get("SPIDEY_DEVICE", "ZOOM AMS-44")   # by name: indices renumber when devices come and go
MICS, RATE = 4, 48000
NAMES = ["S", "W", "N", "E"]   # ch1..ch4


def listdev():
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] or d["max_output_channels"]:
            print(f"  [{i}] {d['name'][:38]:<38} "
                  f"in{d['max_input_channels']} out{d['max_output_channels']} "
                  f"{d['default_samplerate']:.0f}Hz")
    print(f"\n  default in/out: {sd.default.device}")


def meter(peak, db=None):
    print("\033[H\033[J  ch  level                                    peak\n")
    for i in range(MICS):
        bar = "#" * int(max(0, (db[i] + 60) / 60 * 34))
        print(f"  {i+1} {NAMES[i]} {db[i]:6.1f} |{bar:<34}| {peak[i]:6.1f}")
    sys.stdout.flush()


def run(ch, gain_db):
    g = 10.0 ** (gain_db / 20.0)
    peak = np.full(MICS, -99.0)
    state = {"db": np.full(MICS, -99.0)}

    def cb(indata, outdata, frames, t, status):
        x = np.asarray(indata, np.float32)
        state["db"] = 20 * np.log10(np.sqrt((x ** 2).mean(0)) + 1e-9)
        y = x.mean(axis=1) if ch == 0 else x[:, ch - 1]
        outdata[:] = np.clip(y * g, -1.0, 1.0)[:, None]

    what = "all summed" if ch == 0 else f"ch{ch} ({NAMES[ch - 1]})"
    print(f"  hearing {what} at {gain_db:+.0f} dB - USE EARPHONES, ctrl-C to stop")
    with sd.Stream(device=(DEVICE, None), samplerate=RATE,
                   channels=(MICS, 1), dtype="float32",
                   latency="low", callback=cb):
        try:
            while True:
                sd.sleep(200)
                np.maximum(peak - 1.0, state["db"], out=peak)
                meter(peak, state["db"])
        except KeyboardInterrupt:
            print("\n  stopped")


def just_meter():
    peak = np.full(MICS, -99.0)
    print("  level meter - ctrl-C to stop")
    with sd.InputStream(device=DEVICE, samplerate=RATE, channels=MICS,
                        dtype="float32") as s:
        try:
            while True:
                x, _ = s.read(int(RATE * 0.15))
                db = 20 * np.log10(np.sqrt((np.asarray(x) ** 2).mean(0)) + 1e-9)
                np.maximum(peak - 1.0, db, out=peak)
                meter(peak, db)
        except KeyboardInterrupt:
            print("\n  stopped")


def record(ch, secs):
    print(f"  recording {secs}s ...")
    x = sd.rec(int(RATE * secs), samplerate=RATE, channels=MICS,
               dtype="float32", device=DEVICE)
    sd.wait()
    name = f"ch{ch}.wav" if ch else "all4.wav"
    wav.write(name, RATE, x[:, ch - 1] if ch else x)
    db = 20 * np.log10(np.sqrt((x ** 2).mean(0)) + 1e-9)
    print(f"  {name}  {secs}s  levels {np.round(db, 1)} dBFS")


if __name__ == "__main__":
    a = sys.argv[1:]
    ch = int(a[0]) if a and a[0].lstrip("-").isdigit() else None
    arg = lambda f, d: float(a[a.index(f) + 1]) if f in a else d
    if "-l" in a:
        listdev()
    elif "-r" in a:
        record(ch, int(arg("-r", 5)))
    elif ch is not None:
        run(ch, arg("-g", 12.0))
    else:
        just_meter()
