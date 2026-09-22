#!/usr/bin/env python3
"""Listen in one direction.   .venv/bin/python beam.py   (control on :8766, or click the map)

Four mics can't make a narrow beam, but they can put nulls on up to three other
sources. MVDR keeps the chosen bearing at unity gain and drives everything else it
can hear as low as it can; DS (delay-and-sum) is the baseline to A/B against.

    .venv/bin/python beam.py            run: array in -> headphones out
    .venv/bin/python beam.py selftest   two synthetic sources, assert the null works

Control (GET):  /steer?az=270   /mode?m=off|ds|mvdr|mask   /gain?db=20   /load?x=0.5   /nn?on=1
                /rec?on=1 ... /rec?on=0  -> rec/<stamp>_out.wav (what you hear) + _in.wav (4-ch raw)
                /cfg?width=50&ramp=40&floor=0.1&smooth=0.6&post=0.5&dsmooth=0.1   (mask tuning, live)
                /follow?on=1    (steer where the hat looks)   /track?on=1  (steer at the map's talker)   /state
"""
import sys, os, json, threading, time, queue, subprocess, pathlib, urllib.request, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import sounddevice as sd
import spideysense as s

DEVICE_IN, DEVICE_OUT = s.DEVICE, None          # 4-ch input by name (SPIDEY_DEVICE); None = default output
FS      = int(os.environ.get("BEAM_FS", 48000))   # 48k = DeepFilterNet's rate; map runs 44.1k in
                                                   # its own process, CoreAudio serves both (verified)
N       = int(os.environ.get("BEAM_N", 1024))   # 23 ms frames (2048 = 46 ms, sparser)
HOP     = N // 2
TAU     = 0.5          # covariance memory, seconds
LOAD    = 0.5          # diagonal loading; raise if the target gets quieter in MVDR (/load?x=)
GAIN_DB = 20.0
BAND    = (80, 8000)   # MVDR only inside; DS outside
# "mask" mode: per-bin direction test on top of MVDR. Speech is sparse in time-
# frequency, so keeping only bins that arrive from the target's direction separates
# far past the 3-null limit of a linear beamformer. Nonlinear -> some artefacts.
# Tuned on real recordings of a person talking (near, far, with a second talker), not on
# loudspeaker playback: a person in a room scatters far more direction votes than a speaker
# aimed at the array, so the window is wide and the votes are smoothed. A tighter setting
# (width 25, ramp 20, floor 0.04, dsmooth 0) separates two loudspeakers ~1.5 dB better but
# chops 3-7% of a real voice's frames.
MASK_BAND   = (200, 2000)   # bins judged by a 36-way direction search; above, target-vs-
                            #   tracked-interferer (stays valid past the alias limit)
MASK_WIDTH  = 50            # deg either side of target at full gain ...
MASK_RAMP   = 40            # ... then ramps down to the floor over this many more degrees
MASK_FLOOR  = 0.10          # gain for rejected bins (-20 dB); higher = gentler, more leakage
MASK_SMOOTH = 0.6           # per-hop mask update rate; faster tracked speech onsets better
MASK_POST   = 0.5           # 2-beam Wiener post-filter weight; 0 = off, >1 starts eating target
MASK_DSMOOTH = 0.1          # s: running average of each bin's spatial covariance before its
                            # direction vote. Reverb decorrelates over time, the direct path
                            # doesn't: -2.5 dB solo damage AND +1-2 dB separation (measured)
# Competition gate. A real voice in this room scatters ~60% of its bin votes away from its
# bearing, so "votes far from the target" can't tell "alone" from "someone else talking".
# But scatter is DIFFUSE and a competitor is CONCENTRATED: the share of far votes that pile
# onto one direction is 0.00-0.12 for a lone talker and 0.3-0.6 with a second one (measured).
# Below GATE_LO the mask relaxes to pass-through; above GATE_HI it engages fully.
GATE_LO, GATE_HI = 0.10, 0.30
GATE_SMOOTH = 0.12          # per-hop update of the competition estimate (~100 ms)
GATE_HPF    = 250           # Hz: below this the array can't tell who's who, so while a competitor
                            # is present these bins take the floor instead of leaking. 0 = off
GATE_REST   = 0.6           # floor when no competitor is detected. 1 = pass-through (solo -3.5 dB, readers
                            # 6 dB apart); 0.6 keeps a light mask (solo -4 dB, readers 9 dB); measured
# "nn": DeepFilterNet3 after the mask, via tools/df-stream (stdin->stdout, 480-sample hops).
# It repairs the mask's musical noise; measured: it wants a
# GENTLER mask and then does the suppression itself: at equal separation, ~3 dB closer to
# the dry voice. So enabling it also relaxes the floor.
NN_BIN      = pathlib.Path(__file__).parent / "tools" / "df-stream" / "target" / "release" / "df-stream"
NN_FLOOR    = 0.3           # mask floor while the NN is on
NN_CUSHION  = 3             # hops of output buffered before playback starts (absorbs pipe jitter)
NN_PEAK     = 0.3           # the model gets a signal normalised to this peak (-10 dBFS): below about
                            # -45 dBFS it treats everything as noise and outputs silence (measured)
NN_ATTEN    = 8.0           # dB cap on what the model may remove. Per 4 dB of cap, ~1 dB more separation
                            # and ~1 dB more solo-voice loss; above 12 it starts cutting whole frames of
                            # a real voice (unlimited: 10%). 8 = 1% cut. Restart beam.py to change.
PORT    = 8766
MAP_URL = "http://127.0.0.1:8765"   # webui.py, for follow-gaze

M    = 4
FREQ = np.fft.rfftfreq(N, 1 / FS)
NB   = len(FREQ)
INBAND = (FREQ >= BAND[0]) & (FREQ <= BAND[1])
WIN  = np.sqrt(np.hanning(N + 1)[:-1])          # sqrt-Hann in and out: COLA at 50%

MODES = ("off", "ds", "mvdr", "mask")


class NN:
    """DeepFilterNet3 in a subprocess; two threads keep pipe I/O out of the audio callback."""
    def __init__(self):
        self.p = subprocess.Popen([str(NN_BIN)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, bufsize=0,
                                  env={**os.environ, "DF_ATTEN_LIM": str(NN_ATTEN)})
        self.inq, self.ring, self.lock = queue.Queue(), np.zeros(0, np.float32), threading.Lock()
        self.primed, self.underruns, self.waited = False, 0, 0
        threading.Thread(target=self._writer, daemon=True).start()
        threading.Thread(target=self._reader, daemon=True).start()

    def _writer(self):
        while True:
            self.p.stdin.write(self.inq.get().astype("<f4").tobytes())

    def _reader(self):
        need, acc = 480 * 4, b""
        while (b := self.p.stdout.read(need - len(acc))):      # unbuffered pipe: partial reads
            acc += b
            if len(acc) == need:
                with self.lock:
                    self.ring = np.concatenate([self.ring, np.frombuffer(acc, "<f4")])
                acc = b""

    def __call__(self, x):
        """Push a hop in, pull a hop out (delayed by the cushion + model lookahead)."""
        self.inq.put(x)
        with self.lock:
            if not self.primed:
                self.primed = len(self.ring) >= NN_CUSHION * len(x)
                self.waited += 1
                if self.waited > 100:                    # >1 s and nothing back: it's dead, say so
                    self.underruns += 1
                return np.zeros_like(x)
            if len(self.ring) < len(x):
                self.underruns += 1
                return np.zeros_like(x)
            if len(self.ring) > 3 * NN_CUSHION * len(x):     # latency crept up: skip ahead
                self.ring = self.ring[-NN_CUSHION * len(x):]
            y, self.ring = self.ring[:len(x)], self.ring[len(x):]
        return y

    def alive(self):
        return self.p.poll() is None
ctl = {"az": 270.0, "mode": "mvdr", "gain_db": GAIN_DB, "follow": False, "load": LOAD,
       "width": MASK_WIDTH, "floor": MASK_FLOOR, "smooth": MASK_SMOOTH,
       "ramp": MASK_RAMP, "post": MASK_POST, "nn": False, "nn_underruns": 0, "nn_out_db": -99.0,
       "track": False,     # steer at whoever the map is tracking (mutually exclusive with follow)
       "dsmooth": MASK_DSMOOTH, "gate": 1.0, "comp": 0.0, "hpf": GATE_HPF,
       "rec": False, "rec_secs": 0.0,
       "underruns": 0, "running": False, "out_db": -99.0}   # out_db: pre-gain, ~1 s average


def steering(az):
    """d[f, k] = exp(+j2πf p_k·u / c): a source at az arrives as X ∝ d."""
    u = np.array([np.sin(np.radians(az)), np.cos(np.radians(az))])
    return np.exp(2j * np.pi * np.outer(FREQ, (s.MICS @ u) / s.C))


class Beamformer:
    def __init__(self):
        self.R = np.tile(np.eye(M, dtype=complex), (NB, 1, 1))
        self.alpha = HOP / (TAU * FS)
        self.inbuf, self.outbuf = np.zeros((N, M)), np.zeros(N)
        self.az_w = self.mode_w = None
        self.k = 0
        self.w = self.weights()
        self.grid = np.arange(0, 360, 10)
        self.D = np.stack([steering(a) for a in self.grid])        # (36, NB, 4)
        self.inmask = (FREQ >= MASK_BAND[0]) & (FREQ <= MASK_BAND[1])
        self.mask = np.ones(NB)
        self.hist = np.zeros(len(self.grid))       # where low-band energy keeps landing
        self.interf = np.array([], dtype=int)      # tracked interferer grid indices
        self.wI = None                             # MVDR weights toward the top interferer
        self.Rf = np.zeros((NB, M, M), complex)    # fast per-bin covariance for DOA voting
        self.comp = 0.0                            # smoothed competition estimate, 0..1
        self.nn = None                             # started on first /nn?on=1
        self.rec_in, self.rec_out = [], []         # /rec?on=1: raw 4-ch input + what you hear

    def weights(self):
        az, mode = ctl["az"], ctl["mode"]
        d = self.d = steering(az)
        self.az_w, self.mode_w = az, mode
        if mode not in ("mvdr", "mask"):
            return d / M
        tr = np.trace(self.R, axis1=1, axis2=2).real / M
        RL = self.R + (ctl["load"] * tr)[:, None, None] * np.eye(M)
        Rd = np.linalg.solve(RL, d[:, :, None])[:, :, 0]
        w = Rd / np.einsum("fk,fk->f", d.conj(), Rd)[:, None]
        w[~INBAND] = d[~INBAND] / M
        return w

    def spectrum(self, frame):
        return np.fft.rfft(frame * WIN[:, None], axis=0)

    def step(self, X):
        """One hop: update covariance, refresh weights when needed, beamform."""
        self.R = (1 - self.alpha) * self.R + self.alpha * np.einsum("fi,fj->fij", X, X.conj())
        self.k += 1
        if (ctl["az"], ctl["mode"]) != (self.az_w, self.mode_w) or self.k % 4 == 0:
            self.w = self.weights()
        if ctl["mode"] == "off":
            return X[:, 0]
        if ctl["mode"] == "mask":
            self.update_mask(X)
        return self.apply(X)

    def apply(self, X):
        """Current weights (and mask) applied to X with no state update. Linear in X,
        so a mixture's output decomposes exactly into per-source parts (see sep_eval)."""
        if ctl["mode"] == "off":
            return X[:, 0]
        Y = np.einsum("fk,fk->f", self.w.conj(), X)
        return Y * self.mask if ctl["mode"] == "mask" else Y

    def update_mask(self, X):
        """Per bin: is this energy coming from the target's direction?

        In MASK_BAND a full 36-direction search decides. Outside it that search is
        ambiguous (spatial aliasing), so the bin is judged target-vs-tracked-
        interferer instead: two known steering vectors stay separable there."""
        az, W, FL = ctl["az"], ctl["width"], ctl["floor"]
        if ctl["dsmooth"] > 0:
            a = HOP / (ctl["dsmooth"] * FS)
            self.Rf = (1 - a) * self.Rf + a * np.einsum("fi,fj->fij", X, X.conj())
            resp = np.einsum("afk,fkl,afl->af", self.D.conj(), self.Rf, self.D).real   # (36, NB)
        else:
            resp = np.abs(np.einsum("afk,fk->af", self.D.conj(), X)) ** 2   # (36, NB)
        off = np.abs((self.grid - az + 180) % 360 - 180)                  # grid angle vs target
        win = np.argmax(resp[:, self.inmask], axis=0)
        self.hist *= 0.97
        np.add.at(self.hist, win, 1.0)
        cand = np.where((off > W) & (self.hist > 0.2 * self.hist.max()))[0]
        self.interf = cand[np.argsort(-self.hist[cand])][:2]
        win = np.argmax(resp, axis=0)
        doa_off = off[win]
        near = doa_off <= W
        if ctl["gate"]:
            far = self.inmask & (doa_off > W)
            share = 0.0
            if far.sum() >= 5:
                hist = np.bincount(win[far], minlength=len(self.grid))
                k = int(np.argmax(hist))
                dk = np.abs((self.grid - self.grid[k] + 180) % 360 - 180)
                share = hist[dk <= 25].sum() / self.inmask.sum()   # concentrated far votes, per in-band bin
            self.comp += GATE_SMOOTH * (share - self.comp)
            ctl["comp"] = round(float(self.comp), 3)
            c = np.clip((self.comp - GATE_LO) / (GATE_HI - GATE_LO), 0, 1)
            FL = FL + (GATE_REST - FL) * (1 - c)  # no competitor -> floor rises toward GATE_REST
        m = FL + (1 - FL) * np.clip(1 - (doa_off - W) / max(ctl["ramp"], 1e-9), 0, 1)
        if ctl["gate"] and ctl["hpf"] > 0:
            m[FREQ < ctl["hpf"]] = FL + (1 - FL) * (1 - c)   # low bins: floor while competing
        ob = ~self.inmask
        if len(self.interf):
            rt = np.abs(np.einsum("fk,fk->f", self.d.conj(), X)) ** 2
            ri = resp[self.interf].max(axis=0)
            soft = np.clip((rt / (rt + ri + 1e-20) - 0.5) * 2, 0, 1)
            m[ob] = FL + (1 - FL) * soft[ob]
        else:
            m[ob] = FL + (1 - FL) * near[self.inmask].mean()
        if ctl["post"] > 0 and len(self.interf):     # 2-beam Wiener post-filter
            if self.k % 4 == 0 or self.wI is None:
                dI = self.D[self.interf[0]]
                tr = np.trace(self.R, axis1=1, axis2=2).real / M
                RL = self.R + (ctl["load"] * tr)[:, None, None] * np.eye(M)
                Rd = np.linalg.solve(RL, dI[:, :, None])[:, :, 0]
                self.wI = Rd / np.einsum("fk,fk->f", dI.conj(), Rd)[:, None]
            yt = np.abs(np.einsum("fk,fk->f", self.w.conj(), X)) ** 2
            yi = np.abs(np.einsum("fk,fk->f", self.wI.conj(), X)) ** 2
            m = m * np.maximum(yt / (yt + ctl["post"] * yi + 1e-20), FL)
        self.mask += ctl["smooth"] * (m - self.mask)
        return self.mask

    def callback(self, indata, outdata, frames, t, status):
        if status and self.k > 100:            # ignore priming flags in the first ~1 s
            ctl["underruns"] += 1
        self.inbuf = np.roll(self.inbuf, -HOP, axis=0)
        self.inbuf[-HOP:] = indata
        y = np.fft.irfft(self.step(self.spectrum(self.inbuf)), N) * WIN
        self.outbuf = np.roll(self.outbuf, -HOP)
        self.outbuf[-HOP:] = 0
        self.outbuf += y
        out = self.outbuf[:HOP]
        p = float((out ** 2).mean())
        self.pow = getattr(self, "pow", p) * 0.97 + p * 0.03       # ~1 s at 86 hops/s
        ctl["out_db"] = float(10 * np.log10(self.pow + 1e-12))
        if ctl["nn"] and self.nn is not None:
            # level-normalise into the model and back out (what the offline bench does)
            self.pk = max(getattr(self, "pk", 1e-3) * 0.999, float(np.abs(out).max()), 1e-4)
            out = self.nn(out / self.pk * NN_PEAK) * self.pk / NN_PEAK
            q = float((out ** 2).mean())
            self.npow = getattr(self, "npow", q) * 0.97 + q * 0.03
            ctl["nn_out_db"] = float(10 * np.log10(self.npow + 1e-12))
            ctl["nn_underruns"] = self.nn.underruns
            ctl["nn_ring"] = len(self.nn.ring) // HOP      # hops buffered; 0 = starving
        out = np.clip(out * 10 ** (ctl["gain_db"] / 20), -1, 1)
        outdata[:] = out[:, None]
        if ctl["rec"]:
            self.rec_in.append(np.asarray(indata, np.float32).copy())
            self.rec_out.append(out.astype(np.float32))
            ctl["rec_secs"] = len(self.rec_out) * HOP / FS


def follow_map():
    """Steer from webui.py's stream: the hat's gaze (follow) or the map's tracked talker
    (track). A fixed beam loses a walking talker - 29% of frames cut with the NN on;
    following the tracker cuts 1% (measured on a real walking talker)."""
    while True:
        try:
            with urllib.request.urlopen(MAP_URL + "/stream", timeout=5) as r:
                ev = None
                for line in r:
                    if line.startswith(b"event:"):
                        ev = line[6:].strip()
                    elif line.startswith(b"data:"):
                        d = json.loads(line[5:])
                        if ev == b"hat" and ctl["follow"] and d.get("gaze") is not None:
                            ctl["az"] = float(d.get("gaze_target", d["gaze"]))   # parallax-corrected by webui
                        elif ev is None and ctl["track"] and d.get("active") and "az" in d:
                            ctl["az"] = float(d["az"])
                        ev = None
        except Exception:
            time.sleep(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path, _, qs = self.path.partition("?")
        q = {k: v[0] for k, v in urllib.parse.parse_qs(qs).items()}
        if path == "/steer":
            ctl["az"] = float(q["az"]) % 360
        elif path == "/mode" and q.get("m") in MODES:
            ctl["mode"] = q["m"]
        elif path == "/gain":
            ctl["gain_db"] = float(q["db"])
        elif path == "/follow":
            ctl["follow"] = q.get("on", "1") in ("1", "true")
            if ctl["follow"]:
                ctl["track"] = False
        elif path == "/track":
            ctl["track"] = q.get("on", "1") in ("1", "true")
            if ctl["track"]:
                ctl["follow"] = False
        elif path == "/load":
            ctl["load"] = float(q["x"])
        elif path == "/rec":
            on = q.get("on", "1") in ("1", "true")
            if on and not ctl["rec"]:
                BF.rec_in, BF.rec_out = [], []
                ctl["rec"] = True
            elif not on and ctl["rec"]:
                ctl["rec"] = False
                import scipy.io.wavfile as wav
                d = pathlib.Path(__file__).parent / "rec"; d.mkdir(exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                wav.write(d / f"{stamp}_out.wav", FS, np.concatenate(BF.rec_out))
                wav.write(d / f"{stamp}_in.wav", FS, np.concatenate(BF.rec_in))
                ctl["rec_file"] = f"rec/{stamp}_out.wav (+ _in.wav, 4-ch raw)"
        elif path == "/nn":
            on = q.get("on", "1") in ("1", "true")
            if on and FS != 48000:
                ctl["nn_error"] = f"NN needs 48 kHz, beam.py is at {FS}"
            elif on and not NN_BIN.exists():
                ctl["nn_error"] = "tools/df-stream not built (cargo build --release)"
            else:
                ctl.pop("nn_error", None)
                if on and BF.nn is None:
                    BF.nn = NN()
                ctl["nn"] = on
                ctl["floor"] = NN_FLOOR if on else MASK_FLOOR
        elif path == "/cfg":                       # /cfg?width=35&floor=0.1&smooth=0.3
            ctl.update({k: float(v) for k, v in q.items() if k in ("width", "floor", "smooth", "ramp", "post", "dsmooth", "gate", "hpf")})
        elif path != "/state":
            self.send_error(404)
            return
        body = json.dumps(ctl).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


BF = None


def run():
    global BF
    bf = BF = Beamformer()
    out = DEVICE_OUT if DEVICE_OUT is not None else sd.default.device[1]
    name = sd.query_devices(out)["name"]
    if "speaker" in name.lower() and "--speakers-ok" not in sys.argv:
        sys.exit(f"default output is '{name}' - that feeds the mics back into themselves.\n"
                 "Connect headphones (or pass --speakers-ok if you really mean it).")
    och = min(2, sd.query_devices(out)["max_output_channels"])
    threading.Thread(target=follow_map, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sd.Stream(device=(DEVICE_IN, out), samplerate=FS, blocksize=HOP,
                   channels=(M, och), dtype="float32", latency="low",
                   callback=bf.callback):
        ctl["running"] = True
        print(f"beam -> {name}   control http://127.0.0.1:{PORT}/state   (ctrl-c to stop)",
              flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print()


def selftest():
    """Two broadband sources through the real geometry (+1 us jitter). Steer at A.
    MVDR must cut B >= 15 dB below a single mic, beat DS by >= 8 dB, keep A within 1 dB."""
    rng = np.random.default_rng(0)
    T = FS * 4
    def render(az, sig, jitter):
        u = np.array([np.sin(np.radians(az)), np.cos(np.radians(az))])
        f = np.fft.rfftfreq(T, 1 / FS); S = np.fft.rfft(sig)
        tau = (s.MICS @ u) / s.C + jitter
        return np.fft.irfft(S[:, None] * np.exp(2j * np.pi * np.outer(f, tau)), T, axis=0)
    jit = rng.normal(0, 1e-6, M)
    A = render(270, rng.standard_normal(T), jit)         # target
    B = render(45,  rng.standard_normal(T), jit)         # interferer
    ctl.update(az=270.0, mode="mvdr", load=0.1)         # light loading: test the math, not room tuning
    bf = Beamformer()
    for i in range(0, T - N, HOP):                       # adapt on the mixture
        bf.step(bf.spectrum(A[i:i+N] + B[i:i+N]))
    w_mvdr, w_ds = bf.w, steering(270) / M
    w_mic = np.zeros((NB, M)); w_mic[:, 0] = 1           # "no beamformer": one mic
    def power(w, x):
        return sum(np.sum(np.abs(np.einsum("fk,fk->f", w.conj(), bf.spectrum(x[i:i+N]))[INBAND]) ** 2)
                   for i in range(0, T - N, HOP))
    rej_mic = 10 * np.log10(power(w_mic, B) / power(w_mvdr, B))
    rej_ds  = 10 * np.log10(power(w_ds, B)  / power(w_mvdr, B))
    tgt     = 10 * np.log10(power(w_mvdr, A) / power(w_ds, A))
    print(f"interferer at 45, steered 270:  MVDR cuts it {rej_mic:.1f} dB below one mic, "
          f"{rej_ds:.1f} dB below DS;  target {tgt:+.2f} dB")
    assert rej_mic >= 15 and rej_ds >= 8 and abs(tgt) <= 1.0, "selftest FAILED"
    ctl["mode"] = "mask"
    def through(x):
        bf2 = Beamformer(); bf2.R, bf2.w = bf.R, bf.w
        return sum(np.sum(np.abs(bf2.step(bf2.spectrum(x[i:i+N]))[INBAND]) ** 2)
                   for i in range(0, T - N, HOP))
    sep = 10 * np.log10(through(A) / through(B))
    print(f"mask mode: target passes {sep:.1f} dB louder than interferer (MVDR alone: "
          f"{10*np.log10(power(w_mvdr, A) / power(w_mvdr, B)):.1f} dB)")
    assert sep >= 20, "selftest FAILED (mask)"
    print("OK")


if __name__ == "__main__":
    selftest() if "selftest" in sys.argv else run()
