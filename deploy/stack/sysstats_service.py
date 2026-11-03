#!/usr/bin/env python3
"""Tiny host-metrics service for the demo overlay: CPU, memory and Arc GPU utilisation.

GPU busy time comes from i915 DRM fdinfo of this user's processes (no root needed).
Served as JSON on 127.0.0.1:$SYSSTATS_PORT (default 8793), behind Caddy at /sys-stats.
"""
import glob
import json
import os
import platform
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def _load_stack_env():
    path = os.path.join(os.path.expanduser("~"), "work", "stack.env")
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), os.path.expandvars(v.strip()))


def detect_gpu_pdev():
    """Prefer a discrete Intel GPU; otherwise the only (integrated) one at 00:02.0."""
    found = []
    for dev in glob.glob("/sys/class/drm/card[0-9]*/device"):
        try:
            if os.path.basename(os.readlink(dev + "/driver")) not in ("i915", "xe"):
                continue
        except OSError:
            continue
        found.append(os.path.basename(os.path.realpath(dev)))
    discrete = [p for p in found if not p.endswith(":00:02.0")]
    return (sorted(discrete) or sorted(found) or [""])[0]


_load_stack_env()
GPU_PDEV = os.environ.get("SYSSTATS_GPU_PDEV") or detect_gpu_pdev()
PERIOD = 1.0
HISTORY = 60


def read_cpu():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + v[4]
    return sum(v), idle


def read_mem():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, rest = line.split(":", 1)
            info[k] = int(rest.split()[0]) * 1024
    total = info["MemTotal"]
    used = total - info["MemAvailable"]
    return total, used


def read_gpu():
    """Return ({engine: cumulative busy ns summed over clients}, vram_bytes)."""
    busy = {}
    vram = 0
    seen = set()
    for fd in glob.glob("/proc/[0-9]*/fdinfo/*"):
        try:
            with open(fd) as f:
                text = f.read()
        except OSError:
            continue
        if "drm-driver:\ti915" not in text or GPU_PDEV not in text:
            continue
        m = re.search(r"drm-client-id:\s*(\d+)", text)
        key = (fd.split("/")[2], m.group(1) if m else fd)
        if key in seen:
            continue
        seen.add(key)
        for name, val in re.findall(r"drm-engine-([a-z-]+):\s*(\d+) ns", text):
            busy[name] = busy.get(name, 0) + int(val)
        m = re.search(r"drm-resident-local0:\s*(\d+) KiB", text) or re.search(r"drm-resident-system0:\s*(\d+) KiB", text)
        if m:
            vram += int(m.group(1)) * 1024
    return busy, vram


def _read(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def _count_cpus(spec):
    n = 0
    for part in spec.split(","):
        if part:
            lo, _, hi = part.partition("-")
            n += int(hi or lo) - int(lo) + 1
    return n


def _pci_name(pdev):
    try:
        out = subprocess.run(["lspci", "-s", pdev, "-mm"], capture_output=True, text=True, timeout=3).stdout
        parts = re.findall(r'"([^"]*)"', out)
        name = parts[2] if len(parts) > 2 else ""
    except (OSError, subprocess.SubprocessError):
        name = ""
    m = re.search(r"\[(.+)\]", name)
    bracket = m.group(1) if m else name
    return re.sub(r"^Intel(\(R\))?( Corporation)?\s*", "", bracket) or "Intel GPU"


def collect_info():
    cpu_model = "CPU"
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    cpu_model = re.sub(r"\((R|TM)\)", "", cpu_model).replace("  ", " ")
    threads = os.cpu_count() or 0
    p_cores = _count_cpus(_read("/sys/devices/cpu_core/cpus"))
    e_cores = _count_cpus(_read("/sys/devices/cpu_atom/cpus"))
    fmax = _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    cpu_spec = [f"{threads} 线程" + (f" ({p_cores // 2}P + {e_cores}E 核)" if p_cores and e_cores else "")]
    extra = []
    if fmax.isdigit():
        extra.append(f"最高 {int(fmax) / 1e6:.1f} GHz")
    l3 = _read("/sys/devices/system/cpu/cpu0/cache/index3/size")
    if l3:
        kb = int(l3[:-1]) * (1024 if l3.endswith("M") else 1)
        extra.append(f"L3 {kb / 1024:.0f} MB")
    if extra:
        cpu_spec.append(" · ".join(extra))

    pdevs = []
    for dev in glob.glob("/sys/class/drm/card[0-9]*/device"):
        try:
            if os.path.basename(os.readlink(dev + "/driver")) not in ("i915", "xe"):
                continue
        except OSError:
            continue
        pdevs.append(os.path.basename(os.path.realpath(dev)))
    pdevs = sorted(set(pdevs))
    integrated = [p for p in pdevs if p.endswith(":00:02.0")]
    discrete = [p for p in pdevs if p not in integrated]
    ordered = integrated + discrete  # OpenVINO enumerates the iGPU first
    gpus = []
    for i, pdev in enumerate(ordered):
        tag = "GPU" if len(ordered) == 1 else f"GPU.{i}"
        is_discrete = pdev in discrete
        gpus.append({
            "tag": tag, "pdev": pdev, "name": "Intel " + _pci_name(pdev), "discrete": is_discrete,
            "spec": ["独立显卡 · 独立显存" if is_discrete else "核显 · 共享系统内存"],
        })
    mem_total = 0
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal"):
            mem_total = int(line.split()[1]) * 1024
    os_name = ""
    for line in _read("/etc/os-release").splitlines():
        if line.startswith("PRETTY_NAME="):
            os_name = line.split("=", 1)[1].strip('"')
    vendor = _read("/sys/class/dmi/id/sys_vendor")
    product = _read("/sys/class/dmi/id/product_name") or platform.node()
    ov_devices = ["CPU"] + [g["tag"] for g in gpus]
    return {
        "host": {
            "name": product,
            "meta": [
                f"{vendor + ' · ' if vendor else ''}{os_name or platform.system()} · Linux {platform.release()}",
                f"{mem_total / 2**30:.0f} GB 内存 · OpenVINO 可用 {' / '.join(ov_devices)}",
            ],
        },
        "cpu": {"name": cpu_model, "spec": cpu_spec},
        "gpus": gpus,
    }


_INFO = None


class Sampler(threading.Thread):
    daemon = True

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.history = []
        self.latest = {}

    def run(self):
        prev_cpu = read_cpu()
        prev_gpu, _ = read_gpu()
        prev_t = time.monotonic()
        while True:
            time.sleep(PERIOD)
            now = time.monotonic()
            cpu = read_cpu()
            d_total, d_idle = cpu[0] - prev_cpu[0], cpu[1] - prev_cpu[1]
            cpu_pct = 100.0 * (1 - d_idle / d_total) if d_total > 0 else 0.0
            gpu, vram = read_gpu()
            wall = (now - prev_t) * 1e9
            engines = {}
            for name in gpu:
                delta = gpu[name] - prev_gpu.get(name, gpu[name])
                engines[name] = max(0.0, min(100.0, 100.0 * delta / wall)) if wall > 0 else 0.0
            gpu_pct = max(engines.values()) if engines else 0.0
            total, used = read_mem()
            sample = {
                "t": time.time(),
                "cpu": round(cpu_pct, 1),
                "gpu": round(gpu_pct, 1),
                "mem": round(100.0 * used / total, 1),
                "memUsedGb": round(used / 2**30, 1),
                "memTotalGb": round(total / 2**30, 1),
                "vramGb": round(vram / 2**30, 1),
                "gpuEngines": {k: round(v, 1) for k, v in engines.items()},
            }
            with self.lock:
                self.latest = sample
                self.history.append(sample)
                del self.history[:-HISTORY]
            prev_cpu, prev_gpu, prev_t = cpu, gpu, now


SAMPLER = Sampler()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path.endswith("/sys-stats") or path == "/stats":
            with SAMPLER.lock:
                body = {"latest": SAMPLER.latest, "history": list(SAMPLER.history)}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path.endswith("/sys-stats/info"):
            global _INFO
            if _INFO is None:
                _INFO = collect_info()
            data = json.dumps(_INFO, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


def main():
    SAMPLER.start()
    port = int(os.environ.get("SYSSTATS_PORT", "8793"))
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
