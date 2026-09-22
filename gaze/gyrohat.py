#!/usr/bin/env python3
"""Feed webui.py the head heading from the gyro-hat.   python3 gaze/gyrohat.py --hat 192.168.4.1

Reference gaze source. Any tracker works the same way: POST {"yaw": degrees} to
http://127.0.0.1:8765/gaze whenever you have a new heading (clockwise, any zero;
the page's Re-zero takes the current yaw as North). Post {"connected": false} if
you lose the sensor; webui.py also marks it lost after 2 s of silence.

The gyro-hat (ESP32-S3 + LSM6DSOX) hosts its own WiFi and streams one JSON packet
per sample over UDP to whoever last said HELLO. It integrates yaw itself.
"""
import argparse, json, socket, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--hat", default="192.168.4.1")
ap.add_argument("--port", type=int, default=5005)
ap.add_argument("--webui", default="http://127.0.0.1:8765")
a = ap.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", a.port))
sock.settimeout(1.0)
hello = 0.0
while True:
    if time.time() - hello > 1:                       # the hat streams to whoever said HELLO last
        sock.sendto(b"HELLO", (a.hat, a.port))
        hello = time.time()
    try:
        data, _ = sock.recvfrom(512)
        d = json.loads(data)
    except (socket.timeout, ValueError):
        continue
    if "yaw" not in d:
        continue
    try:
        urllib.request.urlopen(urllib.request.Request(a.webui + "/gaze", json.dumps({"yaw": d["yaw"]}).encode(),
                                                      {"Content-Type": "application/json"}), timeout=1)
    except Exception:
        pass
