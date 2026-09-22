#!/bin/sh
# ./run.sh        start beam + room view + transcript (gaze: start gaze/gyrohat.py or your own)
# ./run.sh sim    room view + transcript, no hardware
# ./run.sh stop
# ./run.sh check  selftests: geometry, beamformer
cd "$(dirname "$0")" && mkdir -p logs
PY=.venv/bin/python
case "$1" in
  stop) pkill -f "beam.py|webui.py|listen.py"; echo stopped; exit ;;
  check) $PY spideysense.py selftest && $PY beam.py selftest; exit ;;
  sim)  $PY webui/webui.py --sim > logs/webui.log 2>&1 & sleep 2
        $PY listen.py --mic > logs/listen.log 2>&1 & ;;
  *)    $PY beam.py  > logs/beam.log  2>&1 & sleep 2
        $PY webui/webui.py > logs/webui.log 2>&1 & sleep 2
        $PY listen.py > logs/listen.log 2>&1 & ;;
esac
sleep 1; echo "room view: http://127.0.0.1:8765   logs/*.log   ./run.sh stop"
