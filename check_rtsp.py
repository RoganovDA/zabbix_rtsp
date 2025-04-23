#!/usr/bin/env python3
import cv2
import sys
import os
import time
import argparse
import json
import subprocess
import numpy as np
from datetime import datetime

LOGFILE = "/var/log/check_rtsp.log"
MAX_LOG_SIZE_MB = 15  # Максимальный размер лога в МБ

def log(msg):
    if os.path.exists(LOGFILE) and os.path.getsize(LOGFILE) > MAX_LOG_SIZE_MB * 1024 * 1024:
        with open(LOGFILE, "w") as f:
            f.write(f"[{datetime.now().isoformat()}] Log rotated due to size\n")

    with open(LOGFILE, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid RTSP check via OpenCV + GStreamer + FFprobe")
    p.add_argument("login")
    p.add_argument("password")
    p.add_argument("ip")
    p.add_argument("port")
    p.add_argument("path")
    p.add_argument("--timeout", "-t", type=int, default=5)
    return p.parse_args()

def capture_rtsp(cap, duration=3.0):
    start_time = time.time()
    deadline = start_time + duration
    frames = 0
    sizes = []
    brightness = []
    change_levels = []
    prev_gray = None
    width = height = None

    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frames += 1
        sizes.append(frame.nbytes)
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(np.mean(gray))

        if prev_gray is not None:
            delta = np.mean(np.abs(gray.astype("int16") - prev_gray.astype("int16")))
            change_levels.append(delta)

        prev_gray = gray

    cap.release()

    if frames == 0:
        return "no_frames", 0, None, None, 0.0, 0.0, 0.0, "Connected, but no valid frames"

    avg_kb = round(sum(sizes) / len(sizes) / 1024, 2)
    avg_brightness = round(np.mean(brightness), 2)
    change_level = round(np.mean(change_levels), 2) if change_levels else 0.0
    fps = round(frames / duration, 2)

    return "ok", frames, avg_kb, (width, height), avg_brightness, change_level, fps, ""

def try_opencv_backends(url, timeout, use_gstreamer=False):
    backend_name = "OpenCV"
    if use_gstreamer:
        backend_name = "GStreamer"
        url = f"rtspsrc location={url} latency=0 ! decodebin ! videoconvert ! appsink"
        cap = cv2.VideoCapture(url, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        return "error", 0, None, None, 0.0, 0.0, 0.0, f"{backend_name} failed to open stream"

    status, frames, avg_kb, res, brightness, change_level, fps, note = capture_rtsp(cap)
    return status, frames, avg_kb, res, brightness, change_level, fps, note

def fallback_ffprobe(url, timeout):
    cmd = [
        "ffprobe", "-v", "error",
        "-rtsp_transport", "tcp",
        "-timeout", str(int(timeout * 1e6)),
        "-i", url,
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name",
        "-of", "json"
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+2)
        info = json.loads(p.stdout)["streams"][0]
        return {
            "codec": info.get("codec_name"),
            "width": info.get("width"),
            "height": info.get("height")
        }
    except Exception as e:
        log(f"FFprobe fallback failed: {e}")
        return None

def main():
    args = parse_args()
    base_url = f"rtsp://{args.login}:{args.password}@{args.ip}:{args.port}/{args.path}"
    log(f"Checking: {base_url}")

    # 1. Попытка через OpenCV
    latency_start = time.time()
    status, frames, avg_kb, res, brightness, change_level, fps, note = try_opencv_backends(base_url, args.timeout)
    latency_ms = int((time.time() - latency_start) * 1000)

    # 2. Попытка через GStreamer (если OpenCV не открылся)
    if status == "error":
        log("Trying GStreamer fallback...")
        latency_start = time.time()
        status, frames, avg_kb, res, brightness, change_level, fps, note = try_opencv_backends(base_url, args.timeout, use_gstreamer=True)
        latency_ms = int((time.time() - latency_start) * 1000)

    result = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "url": base_url,
        "status": status,
        "latency_ms": latency_ms,
        "frames_read": frames,
        "avg_frame_size_kb": avg_kb,
        "width": res[0] if res else None,
        "height": res[1] if res else None,
        "avg_brightness": brightness,
        "frame_change_level": change_level,
        "real_fps": fps,
        "note": note
    }

    if status == "error":
        meta = fallback_ffprobe(base_url, args.timeout)
        if meta:
            result["status"] = "no_frames"
            result["codec"] = meta.get("codec_name")
            result["width"] = meta.get("width")
            result["height"] = meta.get("height")
            result["note"] += " | Metadata via ffprobe"

    log(f"Result: {result}")
    print(json.dumps(result))
    sys.exit(1 if result["status"] == "error" else 0)

if __name__ == "__main__":
    main()
