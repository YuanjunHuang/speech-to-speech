#!/usr/bin/env python3
"""OpenAI-compatible /v1/audio/transcriptions server for Qwen3-ASR-0.6B (OpenVINO).

Run it from ~/work/q3asr/venv (transformers 5.13 + the Optimum Intel Qwen3-ASR branch).
The speech-to-speech pipeline talks to it through its stock `--stt openai` handler, so
nothing on the pipeline side knows this is not OVMS/Whisper:

    --stt openai --openai_stt_base_url http://127.0.0.1:8792/v1 \
        --openai_stt_model qwen3-asr-0.6b-int8 --openai_stt_language zh

Notes
-----
* Requests are serialised: one OpenVINO model, one `generate()` at a time. The VAD
  sends a *progressive* request while the user is still talking and a *final* one when
  the turn ends; the handler cancels stale ones client-side.
* Near-silent audio returns an empty transcript without running the model (the
  handler's warm-up request is one second of zeros, and LLM-style ASR models love to
  hallucinate text on silence).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import threading
import time
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("qwen3_asr")
HERE = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"
SAMPLE_RATE = 16000
# RMS below this is treated as silence (full scale = 1.0).
SILENCE_RMS = 2e-4

LANGUAGE_NAMES = {
    "zh": "Chinese", "zh-cn": "Chinese", "zh-tw": "Chinese", "chinese": "Chinese", "cmn": "Chinese",
    "en": "English", "english": "English",
    "ja": "Japanese", "japanese": "Japanese",
    "ko": "Korean", "korean": "Korean",
}


def _language_hint(value: str | None) -> str | None:
    if not value:
        return None
    return LANGUAGE_NAMES.get(value.strip().lower())


def _decode_wav(data: bytes) -> np.ndarray:
    import librosa
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32, copy=False)


class Engine:
    def __init__(self, model_dir: Path, device: str, threads: int) -> None:
        from optimum.intel import OVModelForSpeechSeq2Seq
        from transformers import AutoProcessor

        started = time.perf_counter()
        LOGGER.info("loading %s on %s", model_dir, device)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        ov_config = {"INFERENCE_NUM_THREADS": threads} if threads > 0 and device == "CPU" else {}
        self.model = OVModelForSpeechSeq2Seq.from_pretrained(str(model_dir), device=device, ov_config=ov_config)
        self.device = device
        self.model_dir = model_dir
        self.lock = threading.Lock()
        self.load_s = time.perf_counter() - started
        # One real inference so the first user turn does not pay graph compilation.
        self.transcribe(np.sin(np.linspace(0, 440 * 2 * np.pi, SAMPLE_RATE)).astype(np.float32) * 0.1, None)
        LOGGER.info("ready in %.1fs", time.perf_counter() - started)

    def transcribe(self, audio: np.ndarray, language: str | None) -> tuple[str, str]:
        if audio.size == 0 or float(np.sqrt(np.mean(audio**2))) < SILENCE_RMS:
            return "", ""
        with self.lock:
            kwargs = {"language": language} if language else {}
            inputs = self.processor.apply_transcription_request(audio=audio, sampling_rate=SAMPLE_RATE, **kwargs)
            inputs = inputs.to(self.model.device)
            out = self.model.generate(**inputs, max_new_tokens=160)[:, inputs["input_ids"].shape[1]:]
            parsed = self.processor.decode(out, return_format="parsed")[0]
        return parsed.get("transcription", "").strip(), parsed.get("language", "") or ""


ENGINE: Engine


def parse_multipart(headers, body: bytes) -> tuple[dict[str, str], bytes | None]:
    """Return (text fields, file bytes) from a multipart/form-data body."""
    raw = b"Content-Type: " + headers.get("Content-Type", "").encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    message = BytesParser(policy=policy.default).parsebytes(raw)
    fields: dict[str, str] = {}
    file_bytes: bytes | None = None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        if part.get_filename() is not None or name == "file":
            file_bytes = part.get_payload(decode=True)
        else:
            fields[name] = part.get_content().strip() if part.get_content_type().startswith("text/") else (
                part.get_payload(decode=True) or b""
            ).decode("utf-8", "replace").strip()
    return fields, file_bytes


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        LOGGER.debug("%s - %s", self.address_string(), fmt % args)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            self._json(200, {"ok": True, "model": "qwen3-asr-0.6b", "device": ENGINE.device,
                             "dir": ENGINE.model_dir.name, "load_s": round(ENGINE.load_s, 1)})
        elif self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{"id": "qwen3-asr-0.6b-int8", "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in ("/v1/audio/transcriptions", "/audio/transcriptions"):
            self._json(404, {"error": "not found"})
            return
        started = time.perf_counter()
        try:
            length = int(self.headers.get("Content-Length") or 0)
            fields, file_bytes = parse_multipart(self.headers, self.rfile.read(length))
            if not file_bytes:
                self._json(400, {"error": "file is required"})
                return
            audio = _decode_wav(file_bytes)
            text, detected = ENGINE.transcribe(audio, _language_hint(fields.get("language")))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("transcription failed")
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        wall = time.perf_counter() - started
        LOGGER.info("audio=%.2fs wall=%.2fs lang=%s text=%r", len(audio) / SAMPLE_RATE, wall, detected, text[:40])
        if fields.get("response_format") == "text":
            body = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(200, {"text": text, "language": detected})


def main() -> None:
    global ENGINE
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(HERE / "Qwen3-ASR-0.6B-hf-ov-int8"))
    parser.add_argument("--device", default="CPU", help="OpenVINO device (CPU, GPU.0 = iGPU, GPU.1 = A770)")
    parser.add_argument("--threads", type=int, default=0, help="CPU inference threads (0 = OpenVINO default)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    ENGINE = Engine(Path(args.model_dir), args.device, args.threads)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    LOGGER.info("serving on http://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
