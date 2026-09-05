#!/usr/bin/env python3
"""Polished local browser dashboard for the live FFEM ROS 2 demo."""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import rclpy
from PIL import Image
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as RosImage, PointCloud2
from std_msgs.msg import Float32MultiArray

from ffem.ros2.pointcloud2_codec import decode_pointcloud2

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "outputs" / "dashboard_state.json"
FINEST = 0.05


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {
            "frame": 0,
            "points": 0,
            "moving": 0,
            "active_cells": 0,
            "uniform_cells": 0,
            "memory_reduction_pct": 0.0,
            "total_ms": 0.0,
            "map_ms": 0.0,
            "topology_changes": 0,
            "tracks": 0,
            "classes": [0] * 7,
            "semantic": [],
            "semantic_labels": [],
            "adaptive": [],
            "adaptive_levels": [],
            "moving_points": [],
            "updated": 0.0,
            "camera_rgb_jpeg": "",
            "camera_timestamp": 0.0,
        }

    def write(self):
        with self.lock:
            STATE.parent.mkdir(parents=True, exist_ok=True)
            tmp = STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, separators=(",", ":")))
            tmp.replace(STATE)


class DashboardNode(Node):
    def __init__(self, state):
        super().__init__("ffem_dashboard")
        self.s = state
        self.last_sem = np.empty((0, 3), np.float32)
        self.last_map = np.empty((0, 3), np.float32)
        self.last_move = np.empty((0, 3), np.float32)
        self.last_labels = np.empty((0,), np.int32)
        self.last_levels = np.empty((0,), np.int32)
        self.metrics = [0.0] * 7
        self.seq = 0

        qos = QoSProfile(
            depth=5,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(PointCloud2, "/ffem_mapper/map/elevation", self.map_cb, qos)
        self.create_subscription(PointCloud2, "/ffem_mapper/map/semantic", self.sem_cb, qos)
        self.create_subscription(PointCloud2, "/ffem_mapper/map/moving_points", self.move_cb, qos)
        self.create_subscription(Float32MultiArray, "/ffem_mapper/metrics", self.metrics_cb, 5)
        self.create_subscription(RosImage, "/carla/hero/rgb/image", self.camera_cb, qos)
        self.get_logger().info("Dashboard subscriptions ready")

    @staticmethod
    def pts(msg):
        d = decode_pointcloud2(msg, remove_invalid=True)
        pts = np.asarray(d.points, dtype=np.float32).reshape(-1, 3)
        intensity = None if d.intensity is None else np.rint(d.intensity).astype(np.int32).reshape(-1)
        return pts, intensity

    @staticmethod
    def down(a, n=3000):
        return a if len(a) <= n else a[np.linspace(0, len(a) - 1, n, dtype=int)]

    @staticmethod
    def image_to_rgb(msg: RosImage) -> np.ndarray | None:
        h, w = int(msg.height), int(msg.width)
        if h <= 0 or w <= 0:
            return None
        enc = msg.encoding.lower()
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if enc in {"rgb8", "8uc3"}:
            return raw.reshape(h, msg.step)[:, : w * 3].reshape(h, w, 3).copy()
        if enc in {"bgr8"}:
            bgr = raw.reshape(h, msg.step)[:, : w * 3].reshape(h, w, 3)
            return bgr[:, :, ::-1].copy()
        if enc in {"rgba8"}:
            rgba = raw.reshape(h, msg.step)[:, : w * 4].reshape(h, w, 4)
            return rgba[:, :, :3].copy()
        if enc in {"bgra8", "8uc4"}:
            bgra = raw.reshape(h, msg.step)[:, : w * 4].reshape(h, w, 4)
            return bgra[:, :, [2, 1, 0]].copy()
        if enc in {"mono8", "8uc1"}:
            g = raw.reshape(h, msg.step)[:, :w]
            return np.repeat(g[:, :, None], 3, axis=2).copy()
        # Best-effort fallback for common 4-byte image streams.
        channels = max(1, int(msg.step // max(w, 1)))
        if channels >= 4 and raw.size >= h * msg.step:
            x = raw.reshape(h, msg.step)[:, : w * channels].reshape(h, w, channels)
            return x[:, :, [2, 1, 0]].copy()
        if channels >= 3 and raw.size >= h * msg.step:
            x = raw.reshape(h, msg.step)[:, : w * channels].reshape(h, w, channels)
            return x[:, :, :3].copy()
        return None

    @staticmethod
    def jpeg_b64(rgb: np.ndarray, width=640) -> str:
        img = Image.fromarray(rgb, mode="RGB")
        if img.width > width:
            h = max(1, round(img.height * width / img.width))
            img = img.resize((width, h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def camera_cb(self, msg):
        rgb = self.image_to_rgb(msg)
        if rgb is None:
            return
        encoded = self.jpeg_b64(rgb)
        with self.s.lock:
            self.s.data["camera_rgb_jpeg"] = encoded
            self.s.data["camera_timestamp"] = time.time()
        self.s.write()

    def sem_cb(self, msg):
        self.last_sem, self.last_labels = self.pts(msg)
        if self.last_labels is None:
            self.last_labels = np.zeros(len(self.last_sem), np.int32)

    def map_cb(self, msg):
        self.last_map, self.last_levels = self.pts(msg)
        if self.last_levels is None:
            self.last_levels = np.zeros(len(self.last_map), np.int32)
        self.publish_state()

    def move_cb(self, msg):
        self.last_move, _ = self.pts(msg)

    def metrics_cb(self, msg):
        self.metrics = (list(msg.data) + [0.0] * 7)[:7]
        self.publish_state()

    def publish_state(self):
        if len(self.last_sem):
            q = np.floor(self.last_sem[:, :2] / FINEST).astype(np.int64)
            uniform = int(len(np.unique(q, axis=0)))
        else:
            uniform = 0
        adaptive = int(len(self.last_map))
        reduction = 100.0 * (1.0 - adaptive / uniform) if uniform else 0.0
        with self.s.lock:
            self.seq += 1
            self.s.data.update(
                {
                    "frame": int(self.seq),
                    "points": int(self.metrics[4]),
                    "moving": int(self.metrics[5]),
                    "active_cells": adaptive,
                    "uniform_cells": uniform,
                    "memory_reduction_pct": float(reduction),
                    "total_ms": float(self.metrics[0]),
                    "map_ms": float(self.metrics[1]),
                    "topology_changes": int(self.metrics[3]),
                    "tracks": int(self.metrics[6]),
                    "classes": np.bincount(self.last_labels, minlength=7).tolist() if len(self.last_labels) else [0] * 7,
                    "semantic": self.down(self.last_sem).tolist(),
                    "semantic_labels": self.down(self.last_labels).tolist(),
                    "adaptive": self.down(self.last_map).tolist(),
                    "adaptive_levels": self.down(self.last_levels).tolist(),
                    "moving_points": self.down(self.last_move).tolist(),
                    "updated": time.time(),
                }
            )
        self.s.write()


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/outputs/dashboard_state.json":
            try:
                data = (ROOT / "outputs" / "dashboard_state.json").read_bytes()
            except FileNotFoundError:
                data = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    rclpy.init()
    state = State()
    state.write()
    node = DashboardNode(state)
    import functools
    web = ThreadingHTTPServer((args.host, args.port), functools.partial(Handler, directory=str(ROOT / "dashboard")))
    threading.Thread(target=web.serve_forever, daemon=True).start()
    print(f"Dashboard: http://{args.host}:{args.port}/")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        web.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
