#!/usr/bin/env python3
"""
Hardware metrics sidecar for the Inference Overdrive monitor.

Samples the GPU (nvidia-smi), CPU, RAM, disk and the LLM backend process
(llama-server or vllm) in a
background thread and serves the latest snapshot as JSON. Requests never wait
on nvidia-smi. Inside the container nginx exposes it on the llama.cpp port at
/monitor/hw (see docker/nginx.conf), so it binds to localhost by default.

Stdlib + psutil only.

Usage:
  python3 hw-metrics.py [--host 127.0.0.1] [--port 8083] [--interval 1.0]
"""

import argparse
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil

# (nvidia-smi query field, JSON key, parser)
GPU_FIELDS = [
    # index follows PCI bus order; backends get CUDA_DEVICE_ORDER=PCI_BUS_ID so CUDA ids match it
    ("index", "index", int),
    ("uuid", "uuid", str),
    ("pci.bus_id", "pci_bus", str),
    ("name", "name", str),
    ("driver_version", "driver", str),
    ("pstate", "pstate", str),
    ("utilization.gpu", "util", float),
    ("utilization.memory", "mem_util", float),
    ("temperature.gpu", "temp", float),
    ("power.draw", "power", float),
    ("power.limit", "power_limit", float),
    ("fan.speed", "fan", float),
    ("clocks.current.graphics", "clock_gpu", float),
    ("clocks.max.graphics", "clock_gpu_max", float),
    ("clocks.current.memory", "clock_mem", float),
    ("clocks.max.memory", "clock_mem_max", float),
    ("memory.used", "vram_used", float),
    ("memory.total", "vram_total", float),
    ("pcie.link.gen.current", "pcie_gen", float),
    ("pcie.link.width.current", "pcie_width", float),
]


def _parse(value, conv):
    value = value.strip()
    if not value or "N/A" in value or "Not Supported" in value:
        return None
    try:
        return conv(value)
    except ValueError:
        return None


class Sampler:
    def __init__(self, interval, disk_path, proc_name):
        self.interval = interval
        self.disk_path = disk_path
        self.proc_names = {n.strip() for n in proc_name.split(",") if n.strip()}
        self.gpu_ok = True
        self.proc = None
        self.lock = threading.Lock()
        self.snapshot = {"ts": time.time()}
        psutil.cpu_percent(percpu=True)  # prime: the next call reports usage since now

    def gpus(self):
        if not self.gpu_ok:
            return []
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=" + ",".join(f for f, _, _ in GPU_FIELDS),
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except FileNotFoundError:
            self.gpu_ok = False  # no nvidia-smi in this container; stop trying
            return []
        except subprocess.TimeoutExpired:
            return []
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            values = line.split(",")
            if len(values) != len(GPU_FIELDS):
                continue
            gpus.append({key: _parse(v, conv) for (_, key, conv), v in zip(GPU_FIELDS, values)})
        return gpus

    def llm_proc(self):
        """The running LLM backend process: the first match from the comma-separated --proc names."""
        if self.proc is not None and self.proc.is_running() and self.proc.name() in self.proc_names:
            return self.proc
        self.proc = None
        for p in psutil.process_iter(["name"]):
            if p.info["name"] in self.proc_names:
                self.proc = p
                p.cpu_percent(None)  # prime
                break
        return self.proc

    def sample(self):
        per_core = psutil.cpu_percent(percpu=True)
        freq = psutil.cpu_freq()
        mem = psutil.virtual_memory()
        snap = {
            "ts": time.time(),
            "gpus": self.gpus(),
            "cpu": {
                "percent": round(sum(per_core) / len(per_core), 1) if per_core else None,
                "per_core": per_core,
                "freq": round(freq.current) if freq else None,
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "load": [round(x, 2) for x in os.getloadavg()],
            },
            "ram": {
                "used": mem.used,
                "total": mem.total,
                "percent": mem.percent,
            },
        }
        try:
            disk = psutil.disk_usage(self.disk_path)
            snap["disk"] = {"path": self.disk_path, "used": disk.used, "total": disk.total, "percent": disk.percent}
        except OSError:
            snap["disk"] = None

        proc = self.llm_proc()
        if proc is not None:
            try:
                with proc.oneshot():
                    snap["proc"] = {
                        "pid": proc.pid,
                        "rss": proc.memory_info().rss,
                        "cpu": proc.cpu_percent(None),
                        "threads": proc.num_threads(),
                        "uptime": time.time() - proc.create_time(),
                    }
            except psutil.Error:
                self.proc = None
                snap["proc"] = None
        else:
            snap["proc"] = None
        return snap

    def run(self):
        while True:
            started = time.monotonic()
            try:
                snap = self.sample()
                with self.lock:
                    self.snapshot = snap
            except Exception as e:  # keep sampling; surface the error to the UI
                with self.lock:
                    self.snapshot = {**self.snapshot, "error": repr(e)}
            time.sleep(max(0.0, self.interval - (time.monotonic() - started)))

    def latest(self):
        with self.lock:
            return self.snapshot


class Handler(BaseHTTPRequestHandler):
    sampler = None

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/hw", "/monitor/hw"):
            body = self.sampler.latest()
        elif path == "/health":
            body = {"status": "ok"}
        else:
            self.send_error(404)
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Hardware metrics sidecar for the Inference Overdrive monitor")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1; nginx fronts it)")
    parser.add_argument("--port", type=int, default=8083, help="port (default: 8083)")
    parser.add_argument("--interval", type=float, default=1.0, help="sample interval in seconds (default: 1.0)")
    parser.add_argument("--disk", default="/models" if os.path.isdir("/models") else "/", help="path for disk usage")
    parser.add_argument("--proc", default="llama-server,vllm",
                        help="LLM backend process name(s) to report RSS/CPU for, comma-separated")
    args = parser.parse_args()

    sampler = Sampler(args.interval, args.disk, args.proc)
    threading.Thread(target=sampler.run, daemon=True).start()
    Handler.sampler = sampler

    gpu = sampler.gpus()
    print(f"hw-metrics on {args.host}:{args.port} (GPU: {gpu[0]['name'] if gpu else 'not available'})", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
