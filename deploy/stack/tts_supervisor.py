#!/usr/bin/env python3
"""TTS engine supervisor for the speech-to-speech demo.

Why this exists
---------------
Every TTS backend is a *startup* flag of the s2s process, and the two families do
not fit in RAM together on this 15.6 GB box:

    qwen3        ~6.1 GB RSS (GGML, in-process, streams)
    CosyVoice2/3 ~4-6 GB RSS (separate venv/process, renders a whole reply)

So "pick a TTS engine in the web page" means "tear the pipeline down and bring it
back with different flags". That is what this service does: it owns the s2s
process and, for the CosyVoice engines, the OpenAI-compatible bridge on :8791.

HTTP API (proxied by Caddy at /tts-admin/*):

    GET  /tts         current state + engine catalogue + per-engine voices
    POST /tts         {"engine": "qwen3" | "cosyvoice2" | "cosyvoice3"}
                      -> 202 immediately, then poll GET /tts
    GET  /health      liveness

Switch order matters for memory: the pipeline is stopped *first* (releasing the
6 GB Qwen3 weights), then the bridge is (re)started, then the pipeline comes back
against the new backend.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOME = Path.home()
WORK = HOME / "work"
STATE_DIR = WORK / "state"


def _load_stack_env() -> None:
    """Fill os.environ from stack.env (real environment wins)."""
    path = WORK / "stack.env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), os.path.expandvars(value.strip()))


_load_stack_env()
ACCEL_DEVICE = os.environ.get("ACCEL_DEVICE", "GPU.1")
STT_DEVICE = os.environ.get("STT_DEVICE", ACCEL_DEVICE)
LOG_DIR = WORK / "logs"
ENGINE_STATE_FILE = STATE_DIR / "tts_engine"
ACTIVE_S2S_FILE = STATE_DIR / "active_engine.s2s"
ACTIVE_BRIDGE_FILE = STATE_DIR / "active_engine.bridge"

S2S_PORT = 8765
BRIDGE_PORT = 8791
S2S_LOG = LOG_DIR / "serve_gpu_restart.log"
BRIDGE_LOG = LOG_DIR / "cosyvoice_tts.log"
LAUNCHER = WORK / "start_s2s_gpu.sh"
BRIDGE_SCRIPT = WORK / "cosyvoice_tts_service.py"
BRIDGE_PYTHON = WORK / "cosyvoice/.venv/bin/python"
# CosyVoice3 runs as OpenVINO IR on the Arc A770 (GPU.1) from its own venv; the
# PyTorch/CPU path stays as the automatic fallback when the IR is not present.
OV_BRIDGE_PYTHON = WORK / "cv3ov/venv/bin/python"
OV_MODEL_DIR = WORK / "cv3ov/Fun-CosyVoice3-0.5B-2512-ov"

QWEN3_VOICES = [
    "Vivian",
    "Aiden",
    "Ryan",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Serena",
    "Sohee",
    "Uncle_Fu",
]

ENGINES: dict[str, dict] = {
    "qwen3": {
        "label": "Qwen3-TTS 1.7B",
        "kind": "inproc",
        "transport": "流式",
        "detail": "CPU / GGML Q4_K_M，逐句流式；首音约 2–3 s",
        "voices": QWEN3_VOICES,
    },
    "cosyvoice2": {
        "label": "CosyVoice2-0.5B",
        "kind": "bridge",
        "model": "cosyvoice2",
        "transport": "整段",
        "detail": "CPU，整段渲染后播放；短句约 7 s，三句约 22 s",
        "voices": ["vivian"],
    },
    "cosyvoice3": {
        "label": "CosyVoice3-0.5B",
        "kind": "bridge",
        "model": "cosyvoice3",
        "transport": "逐句",
        "detail": "Arc A770 (OpenVINO)，逐句合成，快于实时",
        "voices": ["vivian"],
    },
}

LOGGER = logging.getLogger("tts_supervisor")


# ── process helpers ─────────────────────────────────────────────────────────


def pid_on_port(port: int) -> int | None:
    """Return the PID listening on `port` (loopback or wildcard), if any."""
    try:
        out = subprocess.run(
            ["ss", "-ltnp"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        local = fields[3]
        if not local.endswith(f":{port}"):
            continue
        match = re.search(r"pid=(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def wait_port_free(port: int, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pid_on_port(port) is None:
            return True
        time.sleep(0.25)
    return pid_on_port(port) is None


def stop_port(port: int, timeout_s: float = 20.0) -> None:
    pid = pid_on_port(port)
    if pid is None:
        return
    LOGGER.info("stopping pid %d on :%d", pid, port)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pid_on_port(port) is None:
            return
        time.sleep(0.25)
    LOGGER.warning("pid %d on :%d ignored SIGTERM, killing", pid, port)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    wait_port_free(port, 10.0)


def http_get_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None


def tcp_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def recent_log_tail(path: Path, offset: int) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def log_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


# ── supervisor ──────────────────────────────────────────────────────────────


class Supervisor:
    def __init__(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._switch_lock = threading.Lock()
        self.state = "ready"
        self.progress = ""
        self.error: str | None = None
        self.engine = self._read_state_file()
        self.bridge_model: str | None = None
        self._reconcile_async()

    # -- persisted choice --

    @staticmethod
    def _read_state_file() -> str:
        try:
            value = ENGINE_STATE_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            return "qwen3"
        return value if value in ENGINES else "qwen3"

    @staticmethod
    def _write_state_file(engine: str) -> None:
        tmp = ENGINE_STATE_FILE.with_suffix(".tmp")
        tmp.write_text(engine + "\n", encoding="utf-8")
        tmp.replace(ENGINE_STATE_FILE)

    @staticmethod
    def _read_text(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return None

    # -- status --

    def snapshot(self) -> dict:
        with self._lock:
            state, progress, error, engine = (
                self.state,
                self.progress,
                self.error,
                self.engine,
            )
            bridge = self.bridge_model
        s2s_running = tcp_open(S2S_PORT)
        # The STT backend is picked by the pipeline launcher (S2S_STT=auto) and can
        # fall back to whisper, so report what is actually answering.
        asr = http_get_json("http://127.0.0.1:8792/health", timeout=1.0)
        stt = (
            {"id": "qwen3asr", "model": "Qwen3-ASR-0.6B", "device": asr.get("device", "CPU"), "dir": asr.get("dir", "")}
            if asr and asr.get("ok")
            else {"id": "whisper", "model": "whisper-small", "device": STT_DEVICE, "dir": ""}
        )
        models = http_get_json("http://127.0.0.1:8000/v3/models", timeout=1.0)
        llm_id = ""
        if models and models.get("data"):
            llm_id = str(models["data"][0].get("id", ""))
        return {
            "llm": {"id": llm_id, "device": os.environ.get("LLM_DEVICE", ACCEL_DEVICE)},
            "stt": stt,
            "active": engine,
            "state": state,
            "progress": progress,
            "error": error,
            "s2s_running": s2s_running,
            "bridge": {
                "port": BRIDGE_PORT,
                "model": bridge,
                "running": tcp_open(BRIDGE_PORT),
            },
            "engines": [
                {
                    "id": engine_id,
                    "label": meta["label"],
                    "kind": meta["kind"],
                    "transport": meta["transport"],
                    "detail": meta["detail"],
                    "voices": meta["voices"],
                }
                for engine_id, meta in ENGINES.items()
            ],
        }

    def _set(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    # -- boot --

    def _reconcile_async(self) -> None:
        """Adopt a pipeline that is already running, otherwise bring it up."""

        def run() -> None:
            desired = self._read_state_file()
            running = self._read_text(ACTIVE_S2S_FILE)
            if tcp_open(S2S_PORT) and running == desired:
                bridge = self._read_text(ACTIVE_BRIDGE_FILE)
                self._set(engine=desired, state="ready", progress="", bridge_model=bridge)
                LOGGER.info("adopted running pipeline: engine=%s bridge=%s", desired, bridge)
                return
            LOGGER.info("reconciling: running=%s desired=%s", running, desired)
            self._do_switch(desired)

        threading.Thread(target=run, name="reconcile", daemon=True).start()

    # -- switching --

    def request_switch(self, engine: str) -> tuple[int, dict]:
        if engine not in ENGINES:
            return 400, {"error": f"unknown engine {engine!r}", "engines": sorted(ENGINES)}
        with self._lock:
            if self.state == "switching":
                return 409, {"error": "a switch is already in progress", "progress": self.progress}
            if engine == self.engine and tcp_open(S2S_PORT):
                return 200, {"ok": True, "state": "ready", "active": engine}
            self.state = "switching"
            self.progress = "准备切换…"
            self.error = None
        threading.Thread(target=self._do_switch, args=(engine,), name="switch", daemon=True).start()
        return 202, {"ok": True, "state": "switching", "active": engine}

    def _do_switch(self, engine: str) -> None:
        if not self._switch_lock.acquire(blocking=False):
            return
        try:
            meta = ENGINES[engine]
            self._set(engine=engine, state="switching", error=None)
            started = time.monotonic()

            self._set(progress="停止语音管线…")
            stop_port(S2S_PORT, 20.0)
            # Also drop the stale marker so a half-finished switch is detectable.
            try:
                ACTIVE_S2S_FILE.unlink()
            except OSError:
                pass

            if meta["kind"] == "bridge":
                self._set(progress=f"启动 {meta['label']} 合成服务…")
                bridge_model = self._ensure_bridge(meta["model"])
                self._set(bridge_model=bridge_model)
            else:
                self._set(progress="释放 CosyVoice 服务…")
                stop_port(BRIDGE_PORT, 20.0)
                try:
                    ACTIVE_BRIDGE_FILE.unlink()
                except OSError:
                    pass
                self._set(bridge_model=None)

            self._set(progress=f"加载语音管线（{meta['label']}）…")
            self._start_s2s(engine)

            self._write_state_file(engine)
            elapsed = time.monotonic() - started
            self._set(state="ready", progress="", engine=engine)
            LOGGER.info("switch to %s complete in %.1fs", engine, elapsed)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("switch to %s failed", engine)
            self._set(state="error", error=f"{type(exc).__name__}: {exc}", progress="")
        finally:
            self._switch_lock.release()

    # -- bridge --

    def _ensure_bridge(self, model: str) -> str:
        health = http_get_json(f"http://127.0.0.1:{BRIDGE_PORT}/health")
        if health and health.get("model") == model:
            LOGGER.info("bridge already serving %s", model)
            self._write_active_bridge(model)
            return model
        stop_port(BRIDGE_PORT, 20.0)
        use_ov = model == "cosyvoice3" and OV_MODEL_DIR.is_dir() and OV_BRIDGE_PYTHON.exists()
        bridge_cmd = [
            str(OV_BRIDGE_PYTHON if use_ov else BRIDGE_PYTHON), str(BRIDGE_SCRIPT),
            "--model", model, "--port", str(BRIDGE_PORT), "--threads", "8",
        ] + (["--backend", "ov", "--ov-device", ACCEL_DEVICE] if use_ov else [])
        LOGGER.info("starting bridge %s (%s)", model, f"openvino {ACCEL_DEVICE}" if use_ov else "torch cpu")
        bridge_offset = log_size(BRIDGE_LOG)
        with BRIDGE_LOG.open("a", encoding="utf-8") as log:
            log.write(f"\n=== supervisor: start {model} {time.strftime('%F %T')} ===\n")
            log.flush()
            subprocess.Popen(
                bridge_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                cwd=str(WORK),
            )
        deadline = time.monotonic() + 300.0
        while time.monotonic() < deadline:
            health = http_get_json(f"http://127.0.0.1:{BRIDGE_PORT}/health")
            if health and health.get("model") == model:
                self._write_active_bridge(model)
                LOGGER.info("bridge %s ready (load %.1fs)", model, health.get("load_s", -1))
                return model
            if "Traceback (most recent call last)" in recent_log_tail(BRIDGE_LOG, bridge_offset) and "Error" in recent_log_tail(BRIDGE_LOG, bridge_offset).split("Traceback")[-1][-600:] and not tcp_open(BRIDGE_PORT):
                raise RuntimeError("CosyVoice bridge crashed; see logs/cosyvoice_tts.log")
            time.sleep(1.0)
        raise TimeoutError(f"CosyVoice bridge {model} did not become healthy")

    @staticmethod
    def _write_active_bridge(model: str) -> None:
        try:
            ACTIVE_BRIDGE_FILE.write_text(model + "\n", encoding="utf-8")
        except OSError:
            pass

    # -- s2s --

    def _start_s2s(self, engine: str) -> None:
        offset = log_size(S2S_LOG)
        env = os.environ.copy()
        env["S2S_TTS_ENGINE"] = engine
        env["S2S_PORT"] = str(S2S_PORT)
        env.pop("PYTHONPATH", None)
        # The launcher may not be executable on all copies; call bash explicitly.
        with S2S_LOG.open("a", encoding="utf-8") as log:
            log.write(f"\n=== supervisor: engine={engine} {time.strftime('%F %T')} ===\n")
            log.flush()
            subprocess.Popen(
                ["bash", str(LAUNCHER)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                cwd=str(WORK / "speech-to-speech"),
            )

        marker = "Qwen3-TTS model loaded" if engine == "qwen3" else "Uvicorn running"
        deadline = time.monotonic() + 300.0
        while time.monotonic() < deadline:
            tail = recent_log_tail(S2S_LOG, offset)
            if "Traceback (most recent call last)" in tail:
                raise RuntimeError(f"pipeline crashed while loading {engine}; see logs/serve_gpu_restart.log")
            if marker in tail and tcp_open(S2S_PORT):
                LOGGER.info("pipeline up with %s", engine)
                return
            time.sleep(1.0)
        raise TimeoutError(f"pipeline did not become ready for {engine}")


SUPERVISOR: Supervisor


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.rstrip("/") or "/"
        if path in ("/health", "/tts-admin/health"):
            self._json(200, {"ok": True})
        elif path in ("/tts", "/tts-admin/tts"):
            self._json(200, SUPERVISOR.snapshot())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/") or "/"
        if path not in ("/tts", "/tts-admin/tts"):
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, ValueError) as exc:
            self._json(400, {"error": f"bad json: {exc}"})
            return
        engine = str(body.get("engine") or "").strip()
        status, payload = SUPERVISOR.request_switch(engine)
        self._json(status, payload)


def main() -> None:
    global SUPERVISOR
    port = int(os.environ.get("TTS_SUPERVISOR_PORT", "8790"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "tts_supervisor.log", encoding="utf-8"),
        ],
    )
    for path in (LAUNCHER, BRIDGE_SCRIPT, BRIDGE_PYTHON):
        if not path.exists():
            raise SystemExit(f"missing required file: {path}")
    SUPERVISOR = Supervisor()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    LOGGER.info("tts supervisor on http://127.0.0.1:%d (engine=%s)", port, SUPERVISOR.engine)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
