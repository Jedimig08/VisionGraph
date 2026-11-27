#!/data/data/com.termux/files/usr/bin/bash
set -eu

CONFIG_DIR="$HOME/.config/visiongraph"
CONFIG_FILE="$CONFIG_DIR/termux_service.conf"
PID_FILE="$CONFIG_DIR/termux_web.pid"
LOG_FILE="$CONFIG_DIR/termux_web.log"
APP_DIR=${VISIONGRAPH_DIR:-$HOME/VisionGraph}

load_config() {
    WIDTH=600
    HEIGHT=400
    FPS=2
    HOST=127.0.0.1
    PORT=5000
    if [ -f "$CONFIG_FILE" ]; then
        . "$CONFIG_FILE"
    fi
}

running_pid() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
        fi
    fi
}

start_sshd() {
    if pgrep -x sshd >/dev/null 2>&1; then
        echo "sshd already running on port 8022"
    else
        sshd
        echo "sshd started on port 8022"
    fi
}

start_viewer() {
    load_config
    pid=$(running_pid || true)
    if [ -n "$pid" ]; then
        echo "VisionGraph viewer already running (PID $pid) at http://$HOST:$PORT"
        return
    fi
    mkdir -p "$CONFIG_DIR"
    cd "$APP_DIR"
    setsid python3 termux_web.py --width "$WIDTH" --height "$HEIGHT" --fps "$FPS" --host "$HOST" --port "$PORT" >>"$LOG_FILE" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$PID_FILE"
    ready=0
    attempt=0
    while [ "$attempt" -lt 40 ]; do
        if curl -fsS -m 1 "http://$HOST:$PORT/api/status" >/dev/null 2>&1; then
            ready=1
            break
        fi
        attempt=$((attempt + 1))
        sleep 0.5
    done
    if [ "$ready" -eq 1 ]; then
        echo "VisionGraph viewer is running at http://$HOST:$PORT (PID $pid)"
    else
        kill "$pid" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "VisionGraph viewer failed to start; see $LOG_FILE" >&2
        return 1
    fi
}

stop_viewer() {
    pid=$(running_pid || true)
    if [ -z "$pid" ]; then
        echo "VisionGraph viewer is not running"
        rm -f "$PID_FILE"
        return
    fi
    kill "$pid"
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid"
    fi
    rm -f "$PID_FILE"
    echo "VisionGraph viewer stopped"
}

status() {
    load_config
    pid=$(running_pid || true)
    if [ -n "$pid" ]; then
        echo "VisionGraph viewer is running at http://$HOST:$PORT (PID $pid)"
    else
        echo "VisionGraph viewer is not running"
    fi
    if curl -fsS -m 1 "http://$HOST:$PORT/api/status" >/dev/null 2>&1; then
        echo "HTTP endpoint: OK at http://$HOST:$PORT"
    else
        echo "HTTP endpoint: not responding at http://$HOST:$PORT"
    fi
    if pgrep -x sshd >/dev/null 2>&1; then
        echo "sshd is running on port 8022"
    else
        echo "sshd is not running"
    fi
    echo "Stop with: visiongraph-service stop"
}

case "${1:-status}" in
    start)
        start_sshd
        start_viewer
        echo "Stop with: visiongraph-service stop"
        ;;
    stop) stop_viewer ;;
    restart)
        stop_viewer
        start_viewer
        ;;
    status) status ;;
    *)
        echo "Usage: visiongraph-service {start|stop|restart|status}" >&2
        exit 2
        ;;
esac
