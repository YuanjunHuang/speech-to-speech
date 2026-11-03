#!/usr/bin/env bash
# Chinese voice pipeline: LLM + STT on Arc A770 (OVMS), TTS = selectable engine.
#
#   S2S_TTS_ENGINE=qwen3        (default) Qwen3-TTS CustomVoice, GGML on CPU,
#                               in-process, streams ~0.5 s bursts
#   S2S_TTS_ENGINE=cosyvoice2   CosyVoice2-0.5B via the local OpenAI-compatible
#   S2S_TTS_ENGINE=cosyvoice3   CosyVoice3-0.5B bridge on :8791 (separate venv)
#
# The CosyVoice engines are CPU-only and ~2-3x slower than realtime, so their
# bridge returns whole utterances; the browser then gets one burst per sentence
# ("render, then play") instead of a starving stream. See §23.
#
# --stream_batch_sentences 1 makes TTS start after the first sentence.
# --enable_style_tags turns on the emotion channel: the LLM prefixes a sentence
# with [style:开心]; the tag is stripped from the transcript and the emotion is
# handed to the TTS stage (qwen3: per-utterance instruct; cosyvoice: the bridge
# maps the canonical word to CosyVoice's own instruct2 phrasing).
#
# The PyPI `qwentts-cpp-python` wheel is a CUDA build and this host has no NVIDIA
# GPU, so the official pure-CPU wheel staged in ~/tmp/qwentts_cpu/site is injected
# via PYTHONPATH/LD_LIBRARY_PATH: it takes precedence over site-packages and
# leaves the venv's CUDA build untouched. (Only used by the qwen3 engine.)
set -euo pipefail
cd "$HOME/work/speech-to-speech"

S2S_TTS_ENGINE=${S2S_TTS_ENGINE:-qwen3}
S2S_PORT=${S2S_PORT:-8765}

# LLM served by OVMS (OpenAI chat-completions). Override to A/B another model.
# End-of-turn grace: after Smart Turn calls the turn complete the pipeline still waits
# this long for you to resume before answering. Default is 800 ms; 400 ms saves ~0.4 s
# per turn at the cost of answering sooner when you pause mid-sentence.
SPECULATIVE_REOPEN_MS=${SPECULATIVE_REOPEN_MS:-400}

LLM_URL=${LLM_URL:-http://127.0.0.1:8000/v3}
# MiniCPM-V 4.5 (INT4, group-size quantised IR) replaced Qwen3-8B INT4, which had replaced Qwen3.5-4B INT8.
# Rollback: LLM_MODEL=qwen3-8b-int4-ov and serve ~/work/ovms/models/qwen3-8b-int4-ov on :8000.
LLM_MODEL=${LLM_MODEL:-minicpm-v-4_5-int4-g-ov}

# STT backend (auto | whisper | qwen3asr). whisper = OVMS whisper-small-int8 on the A770 (:8001); qwen3asr = the
# Qwen3-ASR-0.6B INT8 OpenVINO service (qwen3_asr_service.py, CPU, :8792): ~1/3 the
# character errors on our Chinese test set for ~0.2 s more latency.
S2S_STT=${S2S_STT:-auto}
Q3ASR_DIR="$HOME/work/q3asr"
if [ "$S2S_STT" = auto ]; then
  # Prefer Qwen3-ASR; start its service if it is not up yet, and fall back to whisper
  # rather than leaving the pipeline without a working STT.
  if ! curl -fsS -m 2 http://127.0.0.1:8792/health >/dev/null 2>&1 \
     && [ -x "$Q3ASR_DIR/venv/bin/python" ] && [ -d "$Q3ASR_DIR/Qwen3-ASR-0.6B-hf-ov-int8" ]; then
    echo "starting Qwen3-ASR service on :8792 ..." >&2
    ( cd "$Q3ASR_DIR" && HF_HUB_OFFLINE=1 setsid nohup venv/bin/python qwen3_asr_service.py --device CPU \
        >> "$HOME/work/logs/qwen3_asr.log" 2>&1 < /dev/null & ) 
    for _ in $(seq 1 90); do
      curl -fsS -m 2 http://127.0.0.1:8792/health >/dev/null 2>&1 && break
      sleep 1
    done
  fi
  if curl -fsS -m 2 http://127.0.0.1:8792/health >/dev/null 2>&1; then S2S_STT=qwen3asr; else
    echo "Qwen3-ASR unavailable; falling back to whisper" >&2; S2S_STT=whisper
  fi
fi
case "$S2S_STT" in
  qwen3asr) STT_URL=${STT_URL:-http://127.0.0.1:8792/v1}; STT_MODEL=${STT_MODEL:-qwen3-asr-0.6b-int8} ;;
  whisper) STT_URL=${STT_URL:-http://127.0.0.1:8001/v3}; STT_MODEL=${STT_MODEL:-whisper-small-int8-ov} ;;
  *) echo "unknown S2S_STT=$S2S_STT (auto|whisper|qwen3asr)" >&2; exit 1 ;;
esac
COSYVOICE_URL=${COSYVOICE_URL:-http://127.0.0.1:8791/v1}
QWENTTS_CPU="$HOME/tmp/qwentts_cpu/site"
QWEN3_TTS_QUANT=${QWEN3_TTS_QUANT:-Q4_K_M}
QWEN3_TTS_SIZE=${QWEN3_TTS_SIZE:-1.7b}
QWEN3_TTS_MODEL=${QWEN3_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}

# qwen3 streams well, so hand the TTS stage one sentence at a time (per-sentence
# TTFA and per-sentence emotion). CosyVoice on CPU is 2-4x slower than realtime
# and pays a fixed ~4-5 s prompt-encoding cost per HTTP request, so batch the
# whole reply into a single request: one overhead instead of N, and the browser
# receives one complete burst that cannot starve.
# cosyvoice3 runs on the A770 (OpenVINO) at ~0.5x realtime but pays ~1.2 s fixed
# cost per request, so 1-2 sentence batches still starve on 3-sentence replies
# (measured); 3 sentences per request was SMOOTH on every probe turn.
case "$S2S_TTS_ENGINE" in
  qwen3) STREAM_BATCH_SENTENCES=${STREAM_BATCH_SENTENCES:-1} ;;
  cosyvoice3)
    STREAM_BATCH_SENTENCES=${STREAM_BATCH_SENTENCES:-3}
    # First batch = one sentence, but at least 26 characters (~5 s of speech): the
    # opener plays while the rest renders. FP16 LLM needed 26 chars to avoid gaps
    # (TTFA 5.7-7.4 s); with the INT8 LLM 18 chars was gap-free on 8/8 turns (3.0-5.0 s).
    STREAM_BATCH_SCHEDULE=${STREAM_BATCH_SCHEDULE-1}
    STREAM_BATCH_FIRST_MIN_CHARS=${STREAM_BATCH_FIRST_MIN_CHARS-18}
    ;;
  *) STREAM_BATCH_SENTENCES=${STREAM_BATCH_SENTENCES:-99} ;;
esac

TTS_ARGS=()
if [ "$S2S_TTS_ENGINE" = "qwen3" ]; then
  GGUF_DIR=$(ls -d "$HOME/.cache/huggingface/hub/models--Serveurperso--Qwen3-TTS-GGUF/snapshots/"*/ 2>/dev/null | head -1)
  if [ -z "$GGUF_DIR" ]; then
    echo "no cached Qwen3-TTS GGUF snapshot found" >&2
    exit 1
  fi
  TALKER_GGUF="$GGUF_DIR/qwen-talker-${QWEN3_TTS_SIZE}-customvoice-${QWEN3_TTS_QUANT}.gguf"
  CODEC_GGUF="$GGUF_DIR/qwen-tokenizer-12hz-${QWEN3_TTS_QUANT}.gguf"
  for f in "$TALKER_GGUF" "$CODEC_GGUF"; do
    if [ ! -f "$f" ]; then
      echo "missing GGUF: $f (run ~/work/dl_gguf_quants.py or pick another QWEN3_TTS_QUANT)" >&2
      exit 1
    fi
  done
  TTS_ARGS+=(
    --tts qwen3
    --qwen3_tts_backend ggml
    --qwen3_tts_device auto
    --qwen3_tts_speaker vivian
    --qwen3_tts_model_name "$QWEN3_TTS_MODEL"
    --qwen3_tts_ggml_quantization "$QWEN3_TTS_QUANT"
    --qwen3_tts_gguf_talker_path "$TALKER_GGUF"
    --qwen3_tts_gguf_codec_path "$CODEC_GGUF"
  )
elif [ "$S2S_TTS_ENGINE" = "cosyvoice2" ] || [ "$S2S_TTS_ENGINE" = "cosyvoice3" ]; then
  if ! curl -fsS -m 3 "${COSYVOICE_URL%/v1}/health" >/dev/null 2>&1; then
    echo "CosyVoice bridge not ready at ${COSYVOICE_URL%/v1}/health (start it first)" >&2
    exit 1
  fi
  TTS_ARGS+=(
    --tts openai
    --openai_tts_base_url "$COSYVOICE_URL"
    --openai_tts_model "$S2S_TTS_ENGINE"
    --openai_tts_voice vivian
    --openai_tts_response_format wav
    --openai_tts_sample_rate 24000
    --openai_tts_style_instructions
    --openai_tts_timeout 300
  )
else
  echo "unknown S2S_TTS_ENGINE=$S2S_TTS_ENGINE (qwen3|cosyvoice2|cosyvoice3)" >&2
  exit 1
fi

# Read by the supervisor (tts_supervisor.py) so it can adopt a pipeline that is
# already up instead of restarting it at boot.
mkdir -p "$HOME/work/state"
printf '%s\n' "$S2S_TTS_ENGINE" > "$HOME/work/state/active_engine.s2s"

exec env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  https_proxy=http://child-prc.intel.com:913 http_proxy=http://child-prc.intel.com:913 \
  no_proxy=localhost,127.0.0.1 \
  PYTHONPATH="$QWENTTS_CPU" \
  LD_LIBRARY_PATH="$QWENTTS_CPU/qwentts_cpp_python.libs" \
  .venv/bin/speech-to-speech serve --log_level info --host 0.0.0.0 --port "$S2S_PORT" \
  --llm_backend chat-completions \
  --model_name "$LLM_MODEL" \
  --responses_api_base_url "$LLM_URL" \
  --responses_api_api_key none \
  --responses_api_stream \
  --speculative_reopen_ms "$SPECULATIVE_REOPEN_MS" \
  --enable_style_tags \
  --stream_batch_sentences "$STREAM_BATCH_SENTENCES" \
  ${STREAM_BATCH_SCHEDULE:+--stream_batch_schedule "$STREAM_BATCH_SCHEDULE"} \
  ${STREAM_BATCH_FIRST_MIN_CHARS:+--stream_batch_first_min_chars "$STREAM_BATCH_FIRST_MIN_CHARS"} \
  --torch_num_threads 8 \
  "${TTS_ARGS[@]}" \
  --stt openai \
  --openai_stt_base_url "$STT_URL" \
  --openai_stt_model "$STT_MODEL" \
  --openai_stt_language zh
