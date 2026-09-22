#!/usr/bin/env python3
"""Live room view in the browser.   .venv/bin/python webui/webui.py   ->  http://127.0.0.1:8765
   webui/webui.py --sim    no hardware: two synthetic talkers, a sweeping gaze, a smoke alarm at 20 s

Runs SRP-PHAT continuously (~21 frames/s, 372 ms window) and streams to the page over SSE:
  message  audio frame: smoothed power ring, tracked bearing, secondary sources
  hat      gaze bearing (whatever POSTs /gaze: a head tracker, a phone compass, a webcam)
  beam     beam.py control state (steer/mode/hazard), on change
  utt      a transcribed utterance with its bearing (posted by listen.py)
  alert    name-called / hazard events
"""
import json, os, re, threading, http.server, webbrowser, pathlib, sys, time, difflib
import urllib.request, urllib.parse
from collections import deque
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import spideysense as s

PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8765
WEB  = pathlib.Path(__file__).parent                    # index.html read per request: edit and reload

BEAM_URL  = "http://127.0.0.1:8766"    # beam.py control API; steer/mode/gain/follow are proxied
# Gaze comes from outside: any process POSTs {"yaw": deg} to /gaze as often as it likes (see gaze/).
# yaw is the wearer's head heading in degrees, any zero, increasing clockwise (set YAW_SIGN = -1 if
# not). Re-zero (R on the page) takes the yaw arriving at that moment as North.
YAW_SIGN  = 1       # -1 if the arrow turns the wrong way
GAZE_STALE = 2.0    # s without a /gaze post -> "no gaze"; the last gaze is kept (a dropped tracker doesn't mean the head moved)
# The wearer is not at the array. Their gaze ray starts from where they sit, so "what am I
# looking at" is a parallax problem: intersect the gaze ray with each map source's ray
# from the array; the first source the gaze crosses in front of both is the target, and
# its ARRAY bearing is what the beam steers at. No distances assumed - the crossing gives
# them. With nothing on the line of sight, assume the look lands LOOK_M along the ray.
HAT_POS   = (0.0, -3.0)  # metres, (+x east, +y north): where the wearer sits relative to the array
LOOK_M    = 3.0          # m: only used when no source is on the line of sight (looked away)
PARALLEL_DEG = 8         # gaze within this of a source's bearing counts as looking at it (far)

HOP        = s.NFFT // 8    # 46 ms -> ~21 frames/s; the 372 ms window still sets the resolution
MAP_SMOOTH = 0.35           # per-frame update of the displayed ring (~100 ms memory); fades ~3 s in silence
TRACK_GLIDE = 0.5           # how fast the bearing follows a peak that stays within the same source
SNAP_DEG   = 40             # a peak further than this is a different source ...
SNAP_FRAMES = 4             # ... and must persist this many frames (~190 ms) before we jump to it ...
SNAP_MARGIN = 6             # ... at this many dB above the activity threshold. Measured on real recordings:
                            # word gaps and reverb tails made the old tracker wander 30-150 deg.
ACTIVE_DB  = 5              # room is "active" when the hop level is this far above the slow noise floor
ACTIVE_HOLD = 0.3           # ... and stays active this long after (inter-word gaps). Measured on real recordings:
                            # ambient sits 0-1.3 dB over its floor, speech hops 8-23 dB. SRP prominence
                            # does NOT separate them (ambient 1.63 vs near voice 1.58), so it isn't used.
SRC_MIN    = 0.85           # secondary peaks above this fraction of the max are shown as sources ...
SRC_SEP    = 50             # ... if at least this far from a stronger one. A lone source's sidelobes
                            # reach ~0.8 with 4 mics; two real talkers of similar level both reach >= 0.95.

# Hazard break-in: an alarm (loud AND tonal AND sustained) switches the beam OFF so you hear the room.
HAZ_BAND   = (800, 5000)    # where sirens / smoke alarms / beepers live. Don't extend it downward: speech
                            #     fundamentals live at 250-800 Hz and make every vowel look tonal.
HAZ_LOUD   = 8              # dB above the slow room floor (the phone alarm across the table was +8..12)
HAZ_TOP    = 0.05           # the "tonal" test: this fraction of HAZ_BAND's bins (~19) ...
HAZ_TONAL  = 0.8            # ... must hold this share of its energy. A pure tone -> ~1.0; a 4-tone chord
                            #     like a phone alarm -> ~0.8; speech vowels reach
                            #     0.8-0.95 too, only briefly - the streak below is what tells them apart.
HAZ_STABLE = 30             # bins (~320 Hz) the peak may move per 46 ms hop: tolerates a warbling alarm
HAZ_MISS   = 0.5            # score lost per non-hit hop (hits gain 1) ...
HAZ_TRIG   = 24             # ... and the score that fires: ~1.1 s of a steady alarm, ~5 s of a 50%-duty
                            #     beeper. Human voices peak at 14, a TTS voice through a
                            #     loudspeaker at 20; the phone alarm reaches 26 within 2 s.
BANG_DB    = 25             # a single 46 ms hop this far above the floor, jumping >15 dB in, with >= 30% of
                            # its energy above 4 kHz = a bang (shot, glass, clap). Speech onsets jump as hard
                            # but carry ~0% above 4 kHz on these mics (real recordings); a dull door thud is missed.
BANG_ON    = False          # off by default: applause and dropped pens are bangs too. /haz?bang=1 or the button
ALARM_ON   = True           # the tonal alarm detector. /haz?alarm=0 or the button; Test alarm still works
HAZ_HOLD   = 8              # seconds the beam stays off after the last hit
HAZ_COOL   = 10             # seconds before it can fire again
# Ceiling: a siren sweeping faster than ~1.5 kHz/s is missed; held synth notes in music will fire.

ME_FILE = ROOT / "me.txt"          # the wearer's name: set on the page, or SPIDEY_NAME=Sam,Sammy
# Danger words in the transcript break in too. STRONG ones always; WEAK ones only when shouted
# (a "!" or a one/two-word utterance) so "that talk was fire" stays harmless.
DANGER_STRONG = ("gunshot", "gunshots", "shooter", "shooting", "evacuate", "call 911", "get out", "emergency", "there's a fire", "fire alarm", "somebody help")
DANGER_WEAK   = ("fire", "gun", "help", "bomb", "police")

MY_NAME = os.environ.get("SPIDEY_NAME") or (ME_FILE.read_text().strip() if ME_FILE.exists() else "")   # "" = no name alerts until set
NAME_OFF_GAZE = 45          # only "someone said your name" when they're this far from where you look
DUCK_DB, DUCK_SECS = 12, 3  # how much and how long the beam ducks under an alert

latest, seq = {}, 0
hat, hat_seq = {"connected": False}, 0
gaze_zero = [None]                         # yaw that counts as North; None until the first post or re-zero
gaze_at = [0.0]                            # time of the last /gaze post
beam, beam_seq = {"running": False}, 0
events, ev_seq = deque(maxlen=1000), 0     # (seq, kind, payload): utterances, alerts
track_log = deque(maxlen=2000)             # (t, az, active, facing) so utterances get a bearing + were-you-facing-it
hazard = None                              # {"t", "kind", "hold"} while the beam is forced off
muted = False                              # M on the page: ignore utterances while you narrate
listen_at = 0.0                            # last heartbeat from listen.py
cond = threading.Condition()


def wrap(d):
    return (d + 180) % 360 - 180


def push(kind, payload):
    global ev_seq
    with cond:
        events.append((ev_seq + 1, kind, payload))
        ev_seq += 1
        cond.notify_all()


SIM = "--sim" in sys.argv
sim_alarm_until = 0.0                      # /sim/alarm: 3 s of smoke-alarm beeps in --sim
sim_beam = {"running": True, "mode": "mask", "az": 300.0, "gain_db": 20.0, "follow": False, "nn": False,
            "width": 25, "underruns": 0, "out_db": -60.0}


def beam_get(path):
    try:
        with urllib.request.urlopen(BEAM_URL + path, timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        if not SIM:
            return None
        p, _, qs = path.partition("?")
        q = {k: v[0] for k, v in urllib.parse.parse_qs(qs).items()}
        if p == "/steer": sim_beam["az"] = float(q["az"]) % 360
        if p == "/mode": sim_beam["mode"] = q["m"]
        if p == "/gain": sim_beam["gain_db"] = float(q["db"])
        if p == "/follow": sim_beam["follow"] = q.get("on") == "1"
        if p == "/nn": sim_beam["nn"] = q.get("on") == "1"
        if sim_beam["follow"] and hat.get("gaze") is not None:
            sim_beam["az"] = hat.get("gaze_target", hat["gaze"])
        if latest:
            off = abs(wrap(latest["az"] - sim_beam["az"]))
            sim_beam["out_db"] = latest["level"] - (0 if off < 30 else min(25, (off - 30) * 0.5))
        return dict(sim_beam)


def beam_worker():
    """Poll beam.py's /state ~4x/s so the page shows the true target, and hazards land fast."""
    global beam, beam_seq
    while True:
        b = beam_get("/state") or {"running": False}
        if hazard:
            b["hazard"] = hazard
        if b != beam:
            with cond:
                beam, beam_seq = b, beam_seq + 1
                cond.notify_all()
        time.sleep(0.25)


def set_gaze(d):
    """A /gaze post: {"yaw": deg} (the head heading), optional "connected": false to say the tracker lost it."""
    global hat, hat_seq
    with cond:
        if d.get("connected", True) and d.get("yaw") is not None:
            if gaze_zero[0] is None:
                gaze_zero[0] = float(d["yaw"])
            gaze = (YAW_SIGN * (float(d["yaw"]) - gaze_zero[0])) % 360
            gaze_at[0] = time.time()
            hat = with_target({"connected": True, "gaze": gaze, "raw": float(d["yaw"])})
        else:
            hat = with_target({"connected": False, "gaze": hat.get("gaze"), "raw": hat.get("raw")})
        hat_seq += 1
        cond.notify_all()


def rezero():
    """Face North and call this: the yaw arriving now becomes North."""
    global hat, hat_seq
    with cond:
        if hat.get("raw") is not None:
            gaze_zero[0] = hat["raw"]
            hat = with_target({**hat, "gaze": 0.0})
            hat_seq += 1
            cond.notify_all()
        print(f"re-zero ({'hat connected' if hat.get('connected') else 'NO HAT: nothing to zero'})", flush=True)


def hat_worker():
    """Mark the gaze stale when nothing has posted for GAZE_STALE; in --sim, sweep it between the talkers."""
    global hat, hat_seq
    k = 0
    if SIM:
        gaze_zero[0] = 0.0                            # the sim posts compass bearings, not raw yaw
    while True:
        if SIM:
            k = (k + 1) % 300
            set_gaze({"yaw": float((180 + 120 * np.cos(2 * np.pi * k / 300)) % 360) if k % 150 <= 60
                             else (300.0 if k < 150 else 60.0)})
            time.sleep(0.04)
            continue
        if hat.get("connected") and time.time() - gaze_at[0] > GAZE_STALE:
            set_gaze({"connected": False})
        time.sleep(0.25)


def gaze_target(g, srcs):
    """Array bearing of what a wearer at HAT_POS looking along compass bearing g is looking at.

    Each source is a ray t*u(phi) from the array; the gaze is P + s*u(g). They cross where
      t = (-Px cos g + Py sin g) / sin(g - phi),   s = (Py sin phi - Px cos phi) / sin(g - phi)
    and the crossing is real iff t > 0 and s > 0. Nearly parallel with the same heading is
    a far source on the line. Returns (bearing, hit_source_or_None, distance_from_array)."""
    px, py = HAT_POS
    gr = np.radians(g)
    best = None
    for src in srcs:
        phi = np.radians(src["az"])
        if abs(wrap(g - src["az"])) <= PARALLEL_DEG:
            cand = (LOOK_M * 2, src, LOOK_M * 2)                   # same heading: on the line, far
        else:
            det = np.sin(gr - phi)
            if abs(det) < 1e-6:          # collinear, opposite heading: a source between the wearer and
                continue                 # the array or beyond both; prefer the same-heading "beyond" case
            t = (-px * np.cos(gr) + py * np.sin(gr)) / det
            s_ = (py * np.sin(phi) - px * np.cos(phi)) / det
            if t <= 0 or s_ <= 0:
                continue
            cand = (s_, src, t)
        if best is None or cand[0] < best[0]:                       # first thing along the line of sight
            best = cand
    if best:
        return float(best[1]["az"]), best[1], float(best[2])
    qx, qy = px + LOOK_M * np.sin(gr), py + LOOK_M * np.cos(gr)   # looked away: a point LOOK_M out
    return float(np.degrees(np.arctan2(qx, qy)) % 360), None, float(np.hypot(qx, qy))


def with_target(h):
    """Add gaze_target / gaze_hit / gaze_t / me to a gaze dict (in place, returns it)."""
    if h.get("gaze") is not None:
        az, hit, t = gaze_target(h["gaze"], latest.get("sources", []))
        h.update(gaze_target=round(az, 1), gaze_hit=(hit["az"] if hit else None), gaze_t=round(t, 2))
    h["me"] = HAT_POS
    return h


def sources(n):
    """Local maxima of the smoothed ring, strongest first, at least SRC_SEP apart."""
    idx = np.where((n > np.roll(n, 1)) & (n >= np.roll(n, -1)) & (n > SRC_MIN))[0]
    out = []
    for i in idx[np.argsort(-n[idx])]:
        if all(abs(wrap(i - o["az"])) >= SRC_SEP for o in out):
            out.append({"az": int(i), "str": round(float(n[i]), 2)})
        if len(out) == 3:
            break
    return out


def sim_capture(n):
    """--sim: two talkers taking turns through the real array geometry, no hardware.
    Speech-like = pink-ish noise under a 4 Hz syllable envelope; bearings drift a little."""
    rng = np.random.default_rng()
    A, B = 300.0, 60.0
    t, until, who = 0.0, 0.0, -1                         # who: 0 = A, 1 = B, 2 = both, -1 = a pause
    while True:
        if t >= until:                                   # next turn, with a short pause between turns
            who = rng.choice([0, 1, 2], p=[0.45, 0.45, 0.1]) if who == -1 else -1
            until = t + (rng.uniform(1.5, 4.5) if who != -1 else rng.uniform(0.3, 0.9))
        tt = t + np.arange(n) / s.FS
        out = np.zeros((n, 4))
        for src, az0 in ((0, A), (1, B)):
            if who != src and who != 2 or t + n / s.FS > until - 0.3 and rng.random() < 0.02:
                continue
            az = az0 + 4 * np.sin(2 * np.pi * 0.1 * t + src)
            env = 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * (3.5 + src) * tt) * 3, -1, 1)
            w = rng.standard_normal(n + 64)
            w = np.convolve(w, np.ones(6) / 6, "same")[:n] * env        # rough pink tilt
            u = np.array([np.sin(np.radians(az)), np.cos(np.radians(az))])
            for k in range(4):
                out[:, k] += np.interp(np.arange(n) + (s.MICS[k] @ u) / s.C * s.FS,
                                       np.arange(n), w) * 0.05
        out += rng.standard_normal((n, 4)) * 0.002              # sensor noise
        if (20 <= t < 24 or time.time() < sim_alarm_until) and t % 1.0 < 0.5:   # smoke-alarm beep, 3.1 kHz
            out += (0.3 * np.sin(2 * np.pi * 3100 * tt))[:, None]
        t += n / s.FS
        time.sleep(n / s.FS)
        yield out


def worker():
    buf = np.zeros((s.NFFT, 4))
    ring = np.zeros(s.NGRID)
    st = {"track": 270.0, "pending": None, "pend_n": 0, "floor": None, "last_active": 0.0}
    hist = deque(maxlen=32)                      # ~1.5 s of tracked bearings
    while True:
        try:
            for hop in (sim_capture if SIM else s.capture)(HOP):
                _frame(hop, buf, ring, HAZ, hist, st)
        except Exception as e:                      # device unplugged / not found: keep trying
            print(f"audio: {e} - retrying in 2 s (or run with --sim)", flush=True)
            time.sleep(2)


def _frame(hop, buf, ring, haz, hist, st):
    """One 46 ms hop: update the buffer, hazard detector, ring, tracker; publish a frame."""
    global latest, seq
    buf[:-HOP] = buf[HOP:]; buf[-HOP:] = hop
    haz.step(buf)
    t = time.time()
    db = float(20 * np.log10(np.sqrt((hop ** 2).mean()) + 1e-9))
    if st["floor"] is None:
        st["floor"] = db
    loud = db - st["floor"] > ACTIVE_DB
    # floor drops fast, rises slowly, and barely at all under speech (a long monologue must not fade out)
    st["floor"] += (0.046 / (40.0 if loud else 8.0 if db > st["floor"] else 0.6)) * (db - st["floor"])
    if loud:
        st["last_active"] = t
    active = t - st["last_active"] < ACTIVE_HOLD
    p = s.srp_phat(buf)
    prom = float((p.max() - p.mean()) / (p.std() + 1e-12))
    n = (p - p.min()) / (p.max() - p.min() + 1e-12)
    loud = loud and hazard is None              # an alarm is outside the SRP band: don't chase noise under it
    if loud:                                    # the ring follows sound; in silence it holds its shape and fades
        ring += MAP_SMOOTH * (n - ring)
    else:
        ring *= 0.985
    if loud:                                    # the bearing only moves on real energy, never on a reverb tail
        cand = float(np.degrees(s.THETA[np.argmax(ring)]))
        d = wrap(cand - st["track"])
        if abs(d) < SNAP_DEG:
            st["track"] = (st["track"] + TRACK_GLIDE * d) % 360
            st["pending"] = None
        elif db - st["floor"] > ACTIVE_DB + SNAP_MARGIN:   # a new source: must be loud AND persist
            if st["pending"] is not None and abs(wrap(cand - st["pending"])) < SNAP_DEG:
                st["pend_n"] += 1
            else:
                st["pending"], st["pend_n"] = cand, 1
            if st["pend_n"] >= SNAP_FRAMES:
                st["track"], st["pending"] = cand, None
        hist.append(st["track"])
    track = st["track"]
    track_log.append((t, track, active, facing()))
    frame = {
        "t": t,
        "az": round(track, 1),
        "active": active,
        "prom": round(prom, 2),
        "R": round(float(abs(np.mean(np.exp(1j * np.radians(hist))))), 3) if hist else 0.0,
        "level": round(db, 1),
        "over": round(db - st["floor"], 1),
        "power": np.round(ring, 2).tolist(),
        "sources": sources(ring) if active else [],
        "listen": t - listen_at < 5,
        "haz": dict(haz.last, score=round(haz.score, 1), trig=HAZ_TRIG, bang=BANG_ON, alarm=ALARM_ON),
    }
    with cond:
        latest, seq = frame, seq + 1
        cond.notify_all()


class Hazard:
    """Alarm detector on the raw room mix. Loud + tonal + sustained -> beam off for a while."""
    def __init__(self):
        self.floor, self.score, self.last_hit, self.fired_at, self.prev_mode = -60.0, 0.0, 0.0, -1e9, None
        n = 4096
        self.n, self.win = n, np.hanning(n)
        f = np.fft.rfftfreq(n, 1 / s.FS)
        self.band = (f >= HAZ_BAND[0]) & (f <= HAZ_BAND[1])
        self.k, self.prev_pk = max(3, int(self.band.sum() * HAZ_TOP)), -99
        self.last, self.prev_hop_db = {}, -99.0
        self.hf = np.fft.rfftfreq(HOP, 1 / s.FS) > 4000
        self.hwin = np.hanning(HOP)

    def fire(self, kind, db=None, what=None):
        """Beam off, room red, whisper; step() restores the mode HAZ_HOLD after the last hit."""
        global hazard
        now = time.time()
        self.fired_at = self.last_hit = now
        b = beam_get("/state") or {}
        self.prev_mode = b.get("mode") if b.get("running") else None
        if self.prev_mode and self.prev_mode != "off":
            beam_get("/mode?m=off")
        hazard = {"t": now, "kind": kind, "hold": HAZ_HOLD, "db": db, "what": what or kind}
        alert("hazard", None, what or kind, sub=kind)

    def step(self, buf):
        """buf: the worker's (NFFT, 4) buffer; only its last 93 ms are examined."""
        global hazard
        x = buf[-self.n:].mean(1)
        db = 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)
        # floor: drops fast, rises slowly, barely under loud sound (a long conversation must not
        # raise the bar an alarm has to clear)
        self.floor += (0.046 / (40.0 if db > self.floor + HAZ_LOUD else 5.0 if db > self.floor else 0.5)) * (db - self.floor)
        P = np.abs(np.fft.rfft(x * self.win)[self.band]) ** 2
        pk = int(np.argmax(P))
        tonal = np.sort(P)[-self.k:].sum() / (P.sum() + 1e-20)
        hit = db > self.floor + HAZ_LOUD and tonal > HAZ_TONAL and abs(pk - self.prev_pk) <= HAZ_STABLE
        self.last = {"over": round(float(db - self.floor), 1), "tonal": round(float(tonal), 2),
                     "moved": int(abs(pk - self.prev_pk)), "hit": bool(hit)}
        self.prev_pk = pk
        # bang: one hop that jumps out of the floor, broadband (a shot, a slam, glass) - if enabled
        hop = buf[-HOP:].mean(1)
        hop_db = 20 * np.log10(np.sqrt((hop ** 2).mean()) + 1e-9)
        bang = hop_db > self.floor + BANG_DB and hop_db - self.prev_hop_db > 15
        if bang:                                                    # only then pay for the spectrum
            Ph = np.abs(np.fft.rfft(hop * self.hwin)) ** 2
            bang = Ph[self.hf].sum() / (Ph.sum() + 1e-20) >= 0.3
        self.prev_hop_db = hop_db
        now = time.time()
        if bang and BANG_ON and now - self.fired_at > HAZ_COOL and hazard is None:
            self.fire("bang", round(float(hop_db - self.floor), 1), "loud bang")
            return
        self.score = min(2.0 * HAZ_TRIG, self.score + 1) if hit else max(0.0, self.score - HAZ_MISS)
        if hit:
            self.last_hit = now
        if ALARM_ON and self.score >= HAZ_TRIG and now - self.fired_at > HAZ_COOL and hazard is None:
            self.fire("alarm", round(float(db - self.floor), 1), "alarm")
        elif hazard and now - self.last_hit > HAZ_HOLD:
            hazard = None
            if self.prev_mode and self.prev_mode != "off":
                beam_get(f"/mode?m={self.prev_mode}")
            self.prev_mode = None


def cmean(az):
    z = np.mean(np.exp(1j * np.radians(az)))
    return float(np.degrees(np.angle(z)) % 360), float(abs(z))


HAZ = Hazard()                             # the live detector; /hazard fires it by hand


def bearing_between(t0, t1):
    """Where the sound came from during [t0, t1] (circular mean while active), and where the
    wearer was facing meanwhile (None if there was neither a gaze source nor a beam)."""
    rows = [r for r in track_log if t0 <= r[0] <= t1]
    az = [r[1] for r in rows if r[2]] or [r[1] for r in track_log if t0 - 1 <= r[0] <= t1 + 1] or [latest.get("az", 0)]
    fac = [r[3] for r in rows if r[3] is not None]
    (a, R) = cmean(az)
    return a, R, (cmean(fac)[0] if fac else facing())


def says_my_name(text):
    words = [w.strip(".,!?;:'\"").lower() for w in text.split()]
    targets = {n.strip().lower() for n in MY_NAME.split(",") if n.strip()}
    return any(difflib.SequenceMatcher(None, w, n).ratio() >= 0.8 for w in words for n in targets)


last_alert = {}


def alert(kind, az, text, sub=None):
    """Show it and duck the beam for a moment. 6 s cooldown per kind."""
    now = time.time()
    if now - last_alert.get(kind, 0) < 6:
        return
    last_alert[kind] = now
    push("alert", {"t": now, "kind": kind, "az": az, "text": text, "sub": sub})
    threading.Thread(target=_duck, daemon=True).start()


duck = {"n": 0, "base": None, "lock": threading.Lock()}   # overlapping alerts share one duck


def _duck():
    with duck["lock"]:
        if duck["n"] == 0:
            b = beam_get("/state") or {}
            duck["base"] = b.get("gain_db") if b.get("running") else None
            if duck["base"] is not None:
                beam_get(f"/gain?db={duck['base'] - DUCK_DB}")
        duck["n"] += 1
    time.sleep(DUCK_SECS)
    with duck["lock"]:
        duck["n"] -= 1
        if duck["n"] == 0 and duck["base"] is not None:
            beam_get(f"/gain?db={duck['base']}")


def facing():
    """Where the wearer's attention is: the gaze, else the beam target, else None."""
    if hat.get("gaze") is not None:
        return hat.get("gaze_target", hat["gaze"])
    if beam.get("running") and beam.get("mode") != "off":
        return beam["az"]
    return None


def utts(secs=None):
    since = time.time() - secs if secs else 0
    return [p for (_, k, p) in events if k == "utt" and p["t"] >= since]


def danger(text):
    """The danger phrase in this line, if it should break in. See DANGER_STRONG / DANGER_WEAK."""
    t = " " + re.sub(r"[^a-z0-9' ]+", " ", text.lower()) + " "
    for w in DANGER_STRONG:
        if f" {w} " in t:
            return w
    shouted = "!" in text or len(text.split()) <= 2
    for w in DANGER_WEAK:
        if shouted and f" {w} " in t:
            return w
    return None


def on_utterance(u):
    """listen.py posts {t0, t1, text}; we attach a bearing, check for our name and for danger words."""
    az, R, g = bearing_between(u["t0"], u["t1"])   # g: where you were facing while they spoke
    u = {"t": u["t1"], "az": round(az, 1), "R": round(R, 2), "text": u["text"].strip(),
         "in_beam": None if g is None else abs(wrap(az - g)) <= max(30, beam.get("width", 45)),   # None: no hat, no beam
         "off_gaze": bool(g is not None and abs(wrap(az - g)) > NAME_OFF_GAZE)}
    u["name"] = says_my_name(u["text"]) and (g is None or u["off_gaze"])
    d = danger(u["text"])
    u["danger"] = d
    push("utt", u)
    if u["name"]:
        alert("name", u["az"], u["text"])
    if d and hazard is None and time.time() - HAZ.fired_at > HAZ_COOL:
        HAZ.fire("shout", None, f"someone said “{d}”")


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        d = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/utterance":
            if not muted:
                on_utterance(d)
            return self._json({"ok": True})
        if self.path == "/gaze":                   # {"yaw": 123.4}  from whatever tracks the head
            set_gaze(d)
            return self._json({"ok": True, "gaze": hat.get("gaze")})
        if self.path == "/alert":                  # manual trigger, for testing
            alert(d.get("kind", "name"), d.get("az"), d.get("text", ""))
            return self._json({"ok": True})
        self.send_error(404)

    def do_GET(self):
        path = self.path.partition("?")[0]
        if path in ("/steer", "/mode", "/gain", "/follow", "/load", "/cfg", "/nn", "/rec"):   # forward to beam.py
            b = beam_get(self.path)
            return self._json(b or {"running": False}, 200 if b else 503)
        if path == "/rezero":
            rezero()
            self.send_response(204)
            self.end_headers()
            return
        q = {k: v[0] for k, v in urllib.parse.parse_qs(self.path.partition("?")[2]).items()}
        if path == "/recent":                      # raw transcript
            return self._json(utts(float(q.get("secs", 30))))
        if path == "/sim/alarm" and SIM:           # trigger the sim's alarm without a real one
            global sim_alarm_until
            sim_alarm_until = time.time() + 3
            return self._json({"ok": True})
        if path == "/ping":                        # listen.py heartbeat
            global listen_at
            listen_at = time.time()
            return self._json({"ok": True})
        if path == "/me":                          # your name (comma-separated aliases ok)
            global MY_NAME
            MY_NAME = q.get("name", MY_NAME).strip()[:60] or MY_NAME
            ME_FILE.write_text(MY_NAME)
            push("me", MY_NAME)
            return self._json({"me": MY_NAME})
        if path == "/haz":                         # tune the detector live: /haz?loud=9&tonal=0.9&stable=30&trig=20
            global BANG_ON, ALARM_ON
            for k in ("loud", "tonal", "stable", "trig", "miss"):
                if k in q:
                    globals()["HAZ_" + k.upper()] = float(q[k])
            if "bang" in q:
                BANG_ON = q["bang"] in ("1", "true")
            if "alarm" in q:
                ALARM_ON = q["alarm"] in ("1", "true")
            return self._json({**{k: globals()["HAZ_" + k.upper()] for k in ("loud", "tonal", "stable", "trig", "miss")}, "bang": BANG_ON, "alarm": ALARM_ON})
        if path == "/hazard":                      # test the break-in without a real alarm
            HAZ.fire("test", None, "test alarm")
            return self._json({"hazard": hazard})
        if path == "/mute":
            global muted
            muted = q.get("on", "1") == "1"
            push("mute", muted)
            return self._json({"muted": muted})
        if path.startswith("/static/"):             # d3 + fonts vendored: works offline
            f = WEB / "static" / pathlib.Path(path).name
            if not f.is_file():
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", {"js": "text/javascript", "css": "text/css", "woff2": "font/woff2"}[f.suffix[1:]])
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            return self.wfile.write(f.read_bytes())
        if path != "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write((WEB / "index.html").read_bytes())
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        last = hlast = blast = -1
        with cond:                                     # new client: state + the recent transcript
            elast = ev_seq
            out = f"event: mute\ndata: {json.dumps(muted)}\n\nevent: me\ndata: {json.dumps(MY_NAME)}\n\n"
            out += "".join(f"event: utt\ndata: {json.dumps(p)}\n\n" for (_, k, p) in list(events)[-40:] if k == "utt")
        self.wfile.write(out.encode())
        try:
            while True:
                with cond:
                    cond.wait_for(lambda: seq != last or hat_seq != hlast or beam_seq != blast or ev_seq != elast)
                    out = b""
                    if beam_seq != blast:
                        blast = beam_seq
                        out += f"event: beam\ndata: {json.dumps(beam)}\n\n".encode()
                    if seq != last:
                        last = seq
                        out += f"data: {json.dumps(latest)}\n\n".encode()
                    if hat_seq != hlast:
                        hlast = hat_seq
                        out += f"event: hat\ndata: {json.dumps(hat)}\n\n".encode()
                    if ev_seq != elast:
                        for (i, kind, payload) in events:
                            if i > elast:
                                out += f"event: {kind}\ndata: {json.dumps(payload)}\n\n".encode()
                        elast = ev_seq
                self.wfile.write(out)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=hat_worker, daemon=True).start()
    threading.Thread(target=beam_worker, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}"
    print(f"spideysense -> {url}   listening for '{MY_NAME}'   (ctrl-c to stop)")
    if "--no-open" not in sys.argv:
        webbrowser.open(url)
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
