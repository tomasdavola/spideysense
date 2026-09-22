#!/usr/bin/env python3
"""Room transcript.   .venv/bin/python listen.py   (posts utterances to webui.py on :8765)

Hears the whole room (all four mics summed), cuts it into utterances on silence, runs
faster-whisper locally, and posts {t0, t1, text}. webui.py attaches the bearing the
tracker saw during [t0, t1] and checks for your name.

    listen.py              the array, live
    listen.py --mic        the laptop mic instead (no array plugged in)
    listen.py --wav f.wav  play a file through the same pipeline at real-time speed
"""
import os, sys, time, json, queue, threading, urllib.request
import numpy as np
from scipy.signal import resample_poly

WEBUI   = "http://127.0.0.1:8765/utterance"
PING    = "http://127.0.0.1:8765/ping"       # heartbeat so the page can show the transcript is alive
MODEL   = os.environ.get("SPIDEY_WHISPER", "small.en")  # ~0.45 s per 3 s utterance; base.en is 3x faster but mangles names
DEVICE  = os.environ.get("SPIDEY_DEVICE", "ZOOM AMS-44")   # the 4-channel array, by name
FS_IN   = 44100
FS      = 16000              # whisper's rate
FRAME   = 0.03               # VAD frame, seconds
MIN_SPEECH, END_SIL, MAX_UTT = 0.4, 0.55, 9.0   # seconds
NOISE_UP, NOISE_DOWN = 6.0, 0.4   # floor rises slowly, drops fast (seconds)
ONSET_DB  = 9                # speech = this much above the floor
JUNK = {"you", "thank you", "thanks for watching", "bye", "so", "the", "um", "uh", "hmm", "mm", "oh", ""}   # near-silence

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def audio_source():
    """Yield (t, chunk, rate): mono float32 at the source's native rate, t = wall time of chunk[0].
    Resampling happens once per utterance, not per chunk (chunk-edge filter artefacts)."""
    import sounddevice as sd
    if "--wav" in sys.argv:
        import scipy.io.wavfile as wav
        fs, x = wav.read(sys.argv[sys.argv.index("--wav") + 1])
        x = x.astype(np.float32) / (32768 if x.dtype == np.int16 else 1)
        x = x.mean(1) if x.ndim == 2 else x
        n, t0 = int(fs * 0.1), time.time()
        for i in range(0, len(x), n):
            time.sleep(max(0, t0 + i / fs - time.time()))
            yield t0 + i / fs, x[i:i + n], fs
        return
    dev, ch = DEVICE, 4
    if "--mic" in sys.argv or not any(DEVICE in d["name"] for d in sd.query_devices()):
        dev, ch = None, 1
        log("array not found, using the default input" if "--mic" not in sys.argv else "using the default input")
    q = queue.Queue()
    cb = lambda x, f, t, st: q.put((time.time() - f / FS_IN, x.copy()))
    try:
        stream = sd.InputStream(device=dev, samplerate=FS_IN, channels=ch, dtype="float32", blocksize=4410, callback=cb)
        stream.start()
    except Exception as e:                      # array busy or unplugged: keep the transcript alive
        log(f"could not open {dev}: {e} - falling back to the default input")
        stream = sd.InputStream(device=None, samplerate=FS_IN, channels=1, dtype="float32", blocksize=4410, callback=cb)
    with stream:
        while True:
            t, x = q.get()
            yield t, x.mean(1), FS_IN


def utterances():
    """Energy VAD on 30 ms frames against an adaptive noise floor."""
    floor, buf, speech, t_start, since_voice = -65.0, [], 0.0, None, 0.0
    pend, t_pend = np.zeros(0, np.float32), 0.0          # t_pend: wall time of pend[0]
    for t, x, rate in audio_source():
        n = int(rate * FRAME)
        to16k = lambda a: (resample_poly(a, FS, rate) if rate != FS else a).astype(np.float32)
        if not len(pend):
            t_pend = t
        pend = np.concatenate([pend, x])
        while len(pend) >= n:
            fr, pend, t_fr = pend[:n], pend[n:], t_pend
            t_pend += FRAME
            db = 20 * np.log10(np.sqrt((fr ** 2).mean()) + 1e-9)
            voiced = db > floor + ONSET_DB
            # floor: drops fast, rises slowly, and barely at all while someone is talking
            floor += (FRAME / (30.0 if voiced else NOISE_UP if db > floor else NOISE_DOWN)) * (db - floor)
            if voiced:
                if t_start is None:
                    t_start, buf, speech = t_fr, [], 0.0
                speech += FRAME
                since_voice = 0.0
            elif t_start is not None:
                since_voice += FRAME
            if t_start is not None:
                buf.append(fr)
                dur = len(buf) * FRAME
                if (since_voice >= END_SIL and speech >= MIN_SPEECH) or dur >= MAX_UTT:
                    yield t_start, t_fr - since_voice, to16k(np.concatenate(buf))
                    t_start, buf = None, []
                elif since_voice >= END_SIL:            # too short to be words: drop it
                    t_start, buf = None, []
    if t_start is not None and speech >= MIN_SPEECH:    # --wav ended mid-utterance
        yield t_start, t_pend, to16k(np.concatenate(buf))


def heartbeat():
    while True:
        try:
            urllib.request.urlopen(PING, timeout=1)
        except Exception:
            pass
        time.sleep(2)


def main():
    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    log(f"listening with {MODEL} -> {WEBUI}")
    threading.Thread(target=heartbeat, daemon=True).start()
    for t0, t1, x in utterances():
        x = x * (0.5 / max(np.abs(x).max(), 1e-3))            # the array idles around -45 dBFS
        segs, info = model.transcribe(x, beam_size=1, language="en", vad_filter=False,
                                      condition_on_previous_text=False)
        segs = [s for s in segs if s.no_speech_prob < 0.6]
        text = " ".join(s.text.strip() for s in segs).strip()
        if text.strip(".!?, ").lower() in JUNK:
            continue
        log(f"{t1 - t0:4.1f}s  {text}")
        try:
            urllib.request.urlopen(urllib.request.Request(
                WEBUI, json.dumps({"t0": t0, "t1": t1, "text": text}).encode(),
                {"Content-Type": "application/json"}), timeout=2)
        except Exception as e:
            log(f"  (webui not reachable: {e})")


if __name__ == "__main__":
    main()
