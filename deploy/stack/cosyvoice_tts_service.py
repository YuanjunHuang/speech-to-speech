"""OpenAI-compatible TTS front-end for CosyVoice2 / CosyVoice3 (CPU).

Why a separate service: the speech-to-speech pipeline already speaks the OpenAI
`/v1/audio/speech` protocol (`--tts openai`), including a per-utterance
`instructions` field. CosyVoice, however, lives in its own venv (torch 2.3 +
Matcha-TTS) while the pipeline runs torch 2.11, so instead of forking the
dependency stack the pipeline calls this worker over HTTP.

Design notes:
  * One model resident at a time. The caller (supervisor) restarts the process to
    switch v2 <-> v3, which is the only way to guarantee the previous weights are
    actually returned to the OS on a 15.6 GB box.
  * `/v1/audio/speech` returns the whole utterance in one response. The pipeline
    then hands the browser one burst per sentence, which is exactly the
    "render the sentence, then play it" mode chosen for these slow CPU backends.
  * `instructions` carries the pipeline's canonical style word (开心/难过/…).
    CosyVoice documents its own phrasing and needs the `<|endofprompt|>` marker,
    so the mapping happens here rather than in the shared pipeline.

Run (from the CosyVoice venv):
  ~/work/cosyvoice/.venv/bin/python ~/work/cosyvoice_tts_service.py \
      --model cosyvoice2 --port 8791
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import logging
import os
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(os.path.expanduser("~/work/cosyvoice/CosyVoice"))
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / "third_party/Matcha-TTS"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

LOGGER = logging.getLogger("cosyvoice_tts")

MODEL_DIRS = {
    "cosyvoice2": REPO / "pretrained_models/CosyVoice2-0.5B",
    "cosyvoice3": REPO / "pretrained_models/Fun-CosyVoice3-0.5B-2512",
}

OV_HELPER_DIR = Path.home() / "work/cv3ov"
# Prefer the INT8-weight LLM IR: same accuracy as FP16 on 60 sentences (CER 0.057 vs
# 0.056 zero_shot, 0.038 vs 0.040 instruct2) but ~1.6x the LLM tokens/s on CPU.
_OV_INT8 = OV_HELPER_DIR / "Fun-CosyVoice3-0.5B-2512-ov-int8"
OV_MODEL_DIR = Path(
    os.environ.get("OV_MODEL_DIR")
    or (_OV_INT8 if _OV_INT8.is_dir() else OV_HELPER_DIR / "Fun-CosyVoice3-0.5B-2512-ov")
)

# CosyVoice's own documented emotion wording. v2 (CosyVoice 2 paper §2.6 + the
# official demo) accepts a bare question; v3's published instruct set requires the
# "You are a helpful assistant." prefix and reads anything else as speech, which
# leaks the instruction into the audio. Unknown styles fall through verbatim, so
# the pipeline vocabulary can grow without touching this file.
STYLE_TEXT_V2 = {
    "开心": "你能用高兴的情感说吗？",
    "难过": "你能用悲伤的情感说吗？",
    "生气": "你能用愤怒的情感说吗？",
    "平静": "你能用冷静的情感说吗？",
    "耳语": "用压低声音、悄悄耳语的语气说这句话",
    # CosyVoice 2 paper Table 1 lists these four under Emotion.
    "惊讶": "你能用惊讶的情感说吗？",
    "恐惧": "你能用恐惧的情感说吗？",
    "厌恶": "你能用厌恶的情感说吗？",
    "严肃": "你能用严肃的情感说吗？",
}
STYLE_TEXT_V3 = {
    "开心": "请非常开心地说一句话",
    "难过": "请非常伤心地说一句话",
    "生气": "请非常生气地说一句话",
    "平静": "请用平静、自然的语气说一句话",
    "耳语": "请用非常轻柔的声音说一句话",
    # Wordings picked by ear from the /emotion_v3/ audition (惊讶 C, 恐惧 C, 厌恶 A, 严肃 C).
    "惊讶": "请带着震惊的情绪说一句话",
    "恐惧": "请用颤抖、惊恐的声音说一句话",
    "厌恶": "请非常厌恶地说一句话",
    "严肃": "请用沉稳、庄重的语气说一句话",
}
V3_PREFIX = "You are a helpful assistant. "
NEUTRAL_TEXT = {
    "cosyvoice2": "用自然的语气说这句话",
    "cosyvoice3": "请用自然、平静的语气说一句话",
}
ENDOF = "<|endofprompt|>"

# voice name -> reference wav (timbre only; instruct2 needs no transcript)
VOICES = {
    "vivian": REPO / "asset/zero_shot_prompt.wav",
    "default": REPO / "asset/zero_shot_prompt.wav",
    "zero_shot_prompt": REPO / "asset/zero_shot_prompt.wav",
    "cross_lingual_prompt": REPO / "asset/cross_lingual_prompt.wav",
}


# Transcript of each reference wav. zero_shot needs it; instruct2 does not.
VOICE_PROMPT_TEXT = {
    "zero_shot_prompt.wav": "希望你以后能够做的比我还好呦。",
}
# Styles delivered by plain zero-shot cloning (the LLM also sees the reference
# speech tokens, which instruct2 discards): judged more natural for Chinese in
# the /zh_natural/ audition. Everything else needs instruct2 for emotion control.
ZERO_SHOT_STYLES = {None, "", "平静"}


def _instruct(model_name: str, style: str | None) -> str:
    """Map a canonical style word to the phrasing this CosyVoice version expects."""
    if model_name == "cosyvoice3":
        table = STYLE_TEXT_V3
        prefix = V3_PREFIX
        neutral = NEUTRAL_TEXT["cosyvoice3"]
    else:
        table = STYLE_TEXT_V2
        prefix = ""
        neutral = NEUTRAL_TEXT["cosyvoice2"]
    if style and style.startswith("@"):
        # Audition hook: "@<phrase>" is used verbatim, so wordings can be compared
        # without touching the style table.
        body = style[1:]
    else:
        body = table.get(style, f"用{style}的语气说这句话") if style else neutral
    return prefix + body + ENDOF


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm)
    return buf.getvalue()


def _pcm_bytes(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class Synth:
    """Holds the single loaded model and serialises inference."""

    def __init__(self, model_name: str, threads: int, backend: str = "torch", ov_device: str = "GPU.1", llm_device: str = "CPU") -> None:
        if model_name not in MODEL_DIRS:
            raise SystemExit(f"unknown model {model_name!r}; expected {sorted(MODEL_DIRS)}")
        self.model_name = model_name
        self.model_dir = MODEL_DIRS[model_name]
        if not self.model_dir.is_dir():
            raise SystemExit(f"model dir missing: {self.model_dir}")

        torch.set_num_threads(threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("TRANSFORMERS_NO_TORCHAO", "1")

        self.backend = backend
        self.devices = None
        started = time.perf_counter()
        if backend == "ov":
            # OpenVINO IR of CosyVoice3 (official openvino_notebooks conversion).
            # Flow-matching + HiFT on the Arc GPU, LLM on the CPU. The LLM must NOT
            # run on the GPU: in 6-sentence checks about 1/3 of sentences came back
            # mispronounced (e.g. "周末打算"->"中模打算") on GPU.1 at fp16 *and* fp32,
            # versus 0/18 on CPU, at the same speed (~0.6x realtime).
            if model_name != "cosyvoice3":
                raise SystemExit("the OpenVINO backend only exists for cosyvoice3")
            if not OV_MODEL_DIR.is_dir():
                raise SystemExit(f"OpenVINO model dir missing: {OV_MODEL_DIR}")
            sys.path.insert(0, str(OV_HELPER_DIR))
            from ov_cosyvoice_helper import OVCosyVoice3

            self.devices = {"llm": llm_device, "flow": ov_device, "hift": ov_device, "frontend": "CPU"}
            LOGGER.info("loading OVCosyVoice3 from %s on %s", OV_MODEL_DIR, self.devices)
            self.model = OVCosyVoice3(
                model_dir=str(OV_MODEL_DIR),
                device="CPU",
                llm_device=llm_device,
                flow_device=ov_device,
                hift_device=ov_device,
                frontend_device="CPU",
                hift_input_len=1000,
            )
            hop = int(os.environ.get("COSY_HOP", "25"))
            if hop != 25:
                # tokens per streamed chunk (25 tokens = 1 s of audio)
                self.model.model.token_hop_len = hop
        else:
            from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice3

            cls = CosyVoice2 if model_name == "cosyvoice2" else CosyVoice3
            LOGGER.info("loading %s from %s", cls.__name__, self.model_dir)
            if model_name == "cosyvoice2":
                # CosyVoice2 accepts load_jit; CosyVoice3 dropped that kwarg.
                self.model = cls(str(self.model_dir), load_jit=False, load_trt=False, fp16=False)
            else:
                self.model = cls(str(self.model_dir), load_trt=False, load_vllm=False, fp16=False)
        self.load_s = time.perf_counter() - started
        self.sample_rate = int(self.model.sample_rate)
        self.lock = threading.Lock()
        LOGGER.info("loaded in %.1fs (sample_rate=%d)", self.load_s, self.sample_rate)

    def _begin(self, text: str, style: str | None, voice: str, stream: bool):
        """Pick reference/instruct/mode and start the model generator."""
        ref = VOICES.get((voice or "").strip().lower(), VOICES["default"])
        instruct = _instruct(self.model_name, style)
        prompt_text = VOICE_PROMPT_TEXT.get(ref.name)
        use_zero_shot = (
            self.model_name == "cosyvoice3" and style in ZERO_SHOT_STYLES and prompt_text is not None
        )
        if use_zero_shot:
            results = self.model.inference_zero_shot(
                text, V3_PREFIX.strip() + ENDOF + prompt_text, str(ref), stream=stream, speed=1.0
            )
        else:
            results = self.model.inference_instruct2(text, instruct, str(ref), stream=stream, speed=1.0)
        meta = {
            "mode": "zero_shot" if use_zero_shot else "instruct2",
            "instruct": None if use_zero_shot else instruct,
            "voice_ref": ref.name,
            "backend": self.backend,
            "stream": stream,
        }
        return results, meta

    def synthesize(self, text: str, style: str | None, voice: str) -> tuple[np.ndarray, dict]:
        started = time.perf_counter()
        with self.lock:
            results, meta = self._begin(text, style, voice, stream=False)
            chunks = [out["tts_speech"] for out in results]
            audio = torch.cat(chunks, dim=1).squeeze(0).to(torch.float32).cpu().numpy()
        wall = time.perf_counter() - started
        audio_s = len(audio) / self.sample_rate
        stats = {
            "audio_s": round(audio_s, 2),
            "wall_s": round(wall, 2),
            # conventional RTF here is wall/audio: <1 is faster than realtime
            "rtf": round(wall / audio_s, 2) if audio_s > 0 else None,
            **meta,
        }
        return audio, stats

    def synthesize_stream(self, text: str, style: str | None, voice: str, stats: dict):
        """Yield float32 numpy chunks as soon as the model produces them.

        `stats` is filled in as we go (first-chunk latency, totals) so the caller can
        log it after the stream ends, however it ends.
        """
        started = time.perf_counter()
        total = 0
        with self.lock:
            results, meta = self._begin(text, style, voice, stream=True)
            stats.update(meta)
            try:
                for out in results:
                    chunk = out["tts_speech"].squeeze(0).to(torch.float32).cpu().numpy()
                    if chunk.size == 0:
                        continue
                    if "first_chunk_s" not in stats:
                        stats["first_chunk_s"] = round(time.perf_counter() - started, 2)
                    total += len(chunk)
                    yield chunk
            finally:
                wall = time.perf_counter() - started
                audio_s = total / self.sample_rate
                stats.update(
                    audio_s=round(audio_s, 2),
                    wall_s=round(wall, 2),
                    rtf=round(wall / audio_s, 2) if audio_s > 0 else None,
                )


def _wav_stream_header(sample_rate: int) -> bytes:
    """RIFF/WAVE header for a stream of unknown length (sizes = 0xFFFFFFFF)."""
    import struct

    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


class Handler(BaseHTTPRequestHandler):
    synth: Synth  # set by main()
    stream_enabled = False  # set by main()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # quieter, structured
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, str(value))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{"id": self.synth.model_name, "object": "model"}]})
        elif self.path.rstrip("/") in ("/health", "/v1/health"):
            self._json(
                200,
                {
                    "ok": True,
                    "model": self.synth.model_name,
                    "sample_rate": self.synth.sample_rate,
                    "load_s": round(self.synth.load_s, 1),
                    "backend": self.synth.backend,
                    "stream": Handler.stream_enabled,
                    "devices": self.synth.devices,
                    "voices": sorted({k for k in VOICES}),
                    "styles": sorted(STYLE_TEXT_V3 if self.synth.model_name == "cosyvoice3" else STYLE_TEXT_V2),
                },
            )
            return
        self._json(404, {"error": "not found"})

    def _write_chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def _stream_response(self, text: str, style: str | None, voice: str, fmt: str) -> None:
        """Send audio as it is generated (HTTP chunked, WAV of unknown length).

        Headers are held back until the first audio chunk exists, so a failure
        before any audio is still reported as a clean HTTP 500.
        """
        stats: dict = {}
        started = time.perf_counter()
        gen = self.synth.synthesize_stream(text, style, voice, stats)
        sent_header = False
        try:
            for chunk in gen:
                payload = _pcm_bytes(chunk)
                if not sent_header:
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/pcm" if fmt == "pcm" else "audio/wav")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    if fmt == "wav":
                        payload = _wav_stream_header(self.synth.sample_rate) + payload
                    sent_header = True
                self._write_chunk(payload)
            if not sent_header:
                self._json(500, {"error": "no audio produced"})
            else:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("client went away mid-stream text=%r", text[:24])
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("streaming synthesis failed")
            if not sent_header:
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            else:
                self.close_connection = True
        finally:
            gen.close()
            LOGGER.info(
                "synth text=%r style=%r %s total=%.2fs", text[:24], style, stats, time.perf_counter() - started
            )

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in ("/v1/audio/speech", "/audio/speech"):
            self._json(404, {"error": "not found"})
            return
        try:
            body = self._read_json()
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"error": f"bad json: {exc}"})
            return

        text = (body.get("input") or "").strip()
        if not text:
            self._json(400, {"error": "input is required"})
            return
        voice = body.get("voice") or "default"
        style = body.get("instructions") or None
        fmt = (body.get("response_format") or "wav").lower()
        if fmt not in ("wav", "pcm"):
            self._json(400, {"error": f"unsupported response_format {fmt!r}"})
            return
        if body.get("model") and body["model"] != self.synth.model_name:
            LOGGER.info("request model=%r ignored; serving %s", body["model"], self.synth.model_name)

        if self.stream_enabled:
            self._stream_response(text, style, voice, fmt)
            return

        started = time.perf_counter()
        try:
            audio, stats = self.synth.synthesize(text, style, voice)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("synthesis failed")
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        total_s = time.perf_counter() - started

        if fmt == "pcm":
            self._send(200, _pcm_bytes(audio), "audio/pcm", {"X-Audio-Seconds": f"{stats['audio_s']:.2f}"})
        else:
            self._send(200, _wav_bytes(audio, self.synth.sample_rate), "audio/wav",
                       {"X-Audio-Seconds": f"{stats['audio_s']:.2f}"})
        LOGGER.info(
            "synth text=%r style=%r %s total=%.2fs", text[:24], style, stats, total_s
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cosyvoice2", choices=sorted(MODEL_DIRS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--backend", default="torch", choices=["torch", "ov"])
    parser.add_argument("--ov-device", default="GPU.1", help="OpenVINO device for flow/HiFT (never GPU.0)")
    parser.add_argument("--llm-device", default="CPU", help="OpenVINO device for the speech-token LLM")
    parser.add_argument("--stream", action="store_true", help="stream audio chunks as generated (OpenVINO backend only)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    if not VOICES["default"].is_file():
        raise SystemExit(f"reference wav missing: {VOICES['default']}")

    Handler.synth = Synth(args.model, args.threads, args.backend, args.ov_device, args.llm_device)
    Handler.stream_enabled = bool(args.stream and args.backend == "ov")
    gc.collect()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    LOGGER.info("%s serving on http://%s:%d", args.model, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
