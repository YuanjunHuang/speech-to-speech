#!/usr/bin/env bash
# Restore the full dgpu-nuc production stack after the login session was recycled.
# Order: OVMS-LLM(:8000) + OVMS-STT(:8001) -> s2s(:8765) -> demo(:7860) -> Caddy(:8080/:8443)
# s2s :8765 runs the Qwen3-TTS CustomVoice stack: the former A/B instance (:8766)
# was retired in §22 and its /qwen3tts/* Caddy route removed, because the box only
# fits one Qwen3-TTS copy. This script only (re)starts processes.
set -uo pipefail
HOME_DIR=$HOME
# The NUC has changed address before; use whatever it has now for the pinned URLs below.
NUC_IP=${NUC_IP:-$(hostname -I | awk "{print \$1}")}
LOG=$HOME_DIR/work/logs
# machine-specific settings (devices, model names/dirs); see stack.env
set -a; [ -f "$HOME_DIR/work/stack.env" ] && . "$HOME_DIR/work/stack.env"; set +a
LLM_MODEL=${LLM_MODEL:-minicpm-v-4_5-int4-g-ov}
LLM_DEVICE=${LLM_DEVICE:-GPU.1}; STT_DEVICE=${STT_DEVICE:-GPU.1}
OVMS_MODELS_DIR=${OVMS_MODELS_DIR:-$HOME_DIR/work/ovms/models}
OVMS_STT_MODEL=${OVMS_STT_MODEL:-whisper-small-int8-ov}
OVMS_STT_DIR=${OVMS_STT_DIR:-$HOME_DIR/work/ovms/models_stt}
# OVMS keeps its device in each model's graph.pbtxt; point it at the configured one.
set_graph_device() {  # file device
  [ -f "$1" ] && sed -i -E "s/^(\s*(target_)?device:\s*)\"[^\"]*\"/\1\"$2\"/" "$1"
}
if [ "${OVMS_MODE:-native}" != docker ]; then
  set_graph_device "$OVMS_MODELS_DIR/$LLM_MODEL/graph.pbtxt" "$LLM_DEVICE"
  set_graph_device "$OVMS_STT_DIR/$OVMS_STT_MODEL/graph.pbtxt" "$STT_DEVICE"
fi
OVMS=$HOME_DIR/work/ovms/ovms/bin/ovms
OVMS_LIB=$HOME_DIR/work/ovms/ovms/lib
OVMS_PY=$HOME_DIR/work/ovms_ov_only
mkdir -p "$LOG"

echo "== 0. try to make the user manager survive session churn =="
loginctl enable-linger "$USER" 2>&1 || echo "   (enable-linger not permitted; will rely on keeping sessions open)"

if [ "${OVMS_MODE:-native}" = docker ]; then
  echo "== 1-2. OVMS in Docker (llm :8000, stt :8001) =="
  bash "$HOME_DIR/work/ovms_docker.sh" up all
else
  echo "== 1. OVMS LLM :8000 ($LLM_MODEL on $LLM_DEVICE) =="
  env LD_LIBRARY_PATH="$OVMS_LIB" PYTHONPATH="$OVMS_PY" \
    setsid nohup "$OVMS" \
      --model_path "$OVMS_MODELS_DIR/$LLM_MODEL" \
      --model_name "$LLM_MODEL" \
      --rest_port 8000 --rest_bind_address 127.0.0.1 \
      >"$LOG/ovms_llm.log" 2>&1 </dev/null &
  disown 2>/dev/null || true

  echo "== 2. OVMS STT :8001 =="
  env LD_LIBRARY_PATH="$OVMS_LIB" PYTHONPATH="$OVMS_PY" \
    setsid nohup "$OVMS" \
      --model_path "$OVMS_STT_DIR/$OVMS_STT_MODEL" \
      --model_name "$OVMS_STT_MODEL" \
      --rest_port 8001 --rest_bind_address 127.0.0.1 \
      >"$LOG/ovms_stt.log" 2>&1 </dev/null &
  disown 2>/dev/null || true
fi

echo "   waiting for OVMS models to become AVAILABLE ..."
for i in $(seq 1 60); do
  ok_llm=$(curl -s -o /dev/null -m 3 -w "%{http_code}" http://127.0.0.1:8000/v3/models 2>/dev/null)
  ok_stt=$(curl -s -o /dev/null -m 3 -w "%{http_code}" http://127.0.0.1:8001/v3/models 2>/dev/null)
  if [ "$ok_llm" = "200" ] && [ "$ok_stt" = "200" ]; then echo "   both OVMS up after ${i}0s-ish"; break; fi
  sleep 5
done

echo "== 3. demo frontend :7860 =="
cd "$HOME_DIR/work/speech-to-speech/demo"
env SPEECH_TO_SPEECH_URL=ws://$NUC_IP:8765/v1/realtime \
  setsid nohup "$HOME_DIR/work/demo-env/.venv/bin/uvicorn" server:app \
    --host 127.0.0.1 --port 7860 \
    >"$LOG/demo_frontend.log" 2>&1 </dev/null &
disown 2>/dev/null || true

echo "== 4. Caddy :8080 / :8443 =="
cd "$HOME_DIR/caddy-server"
setsid nohup ./caddy run --config ./Caddyfile >"$LOG/caddy.log" 2>&1 </dev/null &
disown 2>/dev/null || true

echo "== 5. TTS supervisor + speech-to-speech :8765 =="
# tts_supervisor.py owns :8765 and restores the engine last picked in the demo's
# settings panel (state/tts_engine, default qwen3). For a CosyVoice choice it
# also starts the OpenAI-compatible bridge on :8791 first. The pipeline log is
# still logs/serve_gpu_restart.log.
# Rollback to the previous Kokoro stack: bash "$HOME_DIR/work/start_s2s_gpu_kokoro.sh"
# TTS weights default to Q4_K_M: BF16 ran at 0.47x realtime on this CPU, which made
# playback starve (choppy). Override with QWEN3_TTS_QUANT=BF16|Q8_0 if ever needed.
if curl -fsS -m 2 http://127.0.0.1:8790/health >/dev/null 2>&1; then
  echo "   supervisor already running; leaving the pipeline state alone"
else
  setsid nohup python3 "$HOME_DIR/work/tts_supervisor.py" >"$LOG/tts_supervisor.stdout.log" 2>&1 </dev/null &
  disown 2>/dev/null || true
fi

# host CPU/GPU/memory metrics for the demo overlay
if ! curl -fsS -m 2 http://127.0.0.1:8793/stats >/dev/null 2>&1; then
  setsid nohup python3 "$HOME_DIR/work/sysstats_service.py" >"$LOG/sysstats.log" 2>&1 </dev/null &
  disown 2>/dev/null || true
fi

echo "   waiting for the s2s backend ..."
for i in $(seq 1 60); do
  sleep 5
  a=$(curl -s -o /dev/null -m 3 -w "%{http_code}" http://127.0.0.1:8765/docs 2>/dev/null)
  if echo "$a" | grep -qE "200|404"; then
    echo "   s2s responding"; break
  fi
done

echo
echo "== verification =="
for u in http://127.0.0.1:8000/v3/models http://127.0.0.1:8001/v3/models \
         http://127.0.0.1:7860/ http://127.0.0.1:8080/ \
         http://$NUC_IP:8080/ http://$NUC_IP:8080/emotion_prod/ \
         http://$NUC_IP:8080/emotion_ab/ http://$NUC_IP:8080/qwen3tts_audition/; do
  printf "  %-52s %s\n" "$u" "$(curl -s -o /dev/null -m 8 -w "%{http_code}" "$u" 2>/dev/null)"
done
echo
echo "== listening ports =="
ss -ltn 2>/dev/null | grep -E ":8080|:8443|:7860|:8000|:8001|:8765" | awk '{print "  "$4}' | sort -u
echo
echo "== s2s tail =="
grep -aE "Uvicorn running|Realtime API starting|Qwen3-TTS model loaded|Qwen3-TTS TTFA|Error|error" "$LOG/serve_gpu_restart.log" 2>/dev/null | tail -4
