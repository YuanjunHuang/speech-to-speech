#!/usr/bin/env bash
# OVMS in Docker, following "Ubuntu24.04 + OVMS AI Model 安装测试指南 v1":
#   image openvino/model_server:<ver>-gpu, --device /dev/dri, --group-add <render gid>,
#   model dir mounted read-only at /opt/ml, flags passed on the command line.
# Services:  llm  -> :8000  (/v3/chat/completions)   stt -> :8001  (/v3/audio/transcriptions)
# Usage:     ovms_docker.sh up|down|restart|status|logs [llm|stt|all]      DRY_RUN=1 prints the commands.
# Settings come from ~/work/stack.env (LLM_MODEL, LLM_DEVICE, STT_DEVICE, OVMS_IMAGE, ...).
set -uo pipefail
W=$HOME/work
[ -f "$W/stack.env" ] && { set -a; . "$W/stack.env"; set +a; }
OVMS_IMAGE=${OVMS_IMAGE:-openvino/model_server:2026.4.0-gpu}
LLM_MODEL=${LLM_MODEL:-minicpm-v-4_5-int4-g-ov}
LLM_DEVICE=${LLM_DEVICE:-GPU}; STT_DEVICE=${STT_DEVICE:-$LLM_DEVICE}
OVMS_MODELS_DIR=${OVMS_MODELS_DIR:-$W/ovms/models}
OVMS_STT_MODEL=${OVMS_STT_MODEL:-whisper-small-int8-ov}
OVMS_STT_DIR=${OVMS_STT_DIR:-$W/ovms/models_stt}
# Pipeline type is auto-detected: an IR with a vision encoder (MiniCPM-V, Qwen3-VL) must use VLM_CB,
# OVMS refuses LM_CB for it; plain text IRs use LM_CB. Override with LLM_PIPELINE.
if [ -z "${LLM_PIPELINE:-}" ]; then
  if [ -f "$OVMS_MODELS_DIR/$LLM_MODEL/openvino_vision_embeddings_model.xml" ]; then LLM_PIPELINE=VLM_CB; else LLM_PIPELINE=LM_CB; fi
fi
LLM_PREFIX_CACHING=${LLM_PREFIX_CACHING:-false}   # true needs intel-igc-cm inside the image (guide 3.3); without it the GPU kernel build crashes
LLM_EXTRA_ARGS=${LLM_EXTRA_ARGS:-}               # e.g. "--tool_parser qwen3coder --reasoning_parser qwen3" for Qwen3
IGC_DEB=${IGC_DEB:-$W/ovms-docker/pkgs/intel-igc-cm_1.0.392-1~u24.04_amd64.deb}
ONE=${2:-all}; CMD=${1:-status}
RUN() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*" >&2; else "$@"; fi; }
render_gid() { stat -c "%g" /dev/dri/render* 2>/dev/null | head -1; }

# A graph.pbtxt left by a native OVMS run pins its device and would override --target_device,
# so mount a hard-linked copy of the IR without it (no extra disk space on the same filesystem).
prepare_model() {  # src name -> echoes mount dir
  local src=$1 name=$2 dst="$W/ovms-docker/models/$2"
  if [ ! -d "$dst" ]; then
    [ -n "${DRY_RUN:-}" ] || mkdir -p "$(dirname "$dst")"
    RUN cp -al "$src" "$dst" 2>/dev/null || RUN cp -a "$src" "$dst"
  fi
  RUN rm -f "$dst/graph.pbtxt"
  echo "$dst"
}

up_llm() {
  local dir; dir=$(prepare_model "$OVMS_MODELS_DIR/$LLM_MODEL" "$LLM_MODEL")
  RUN docker rm -f ovms-llm >/dev/null 2>&1
  # shellcheck disable=SC2086
  RUN docker run -d --name ovms-llm --restart unless-stopped -p 127.0.0.1:8000:8000 \
    --device /dev/dri --group-add "$(render_gid)" -v "$dir:/opt/ml:ro" "$OVMS_IMAGE" \
    --rest_port 8000 --model_path /opt/ml --model_name "$LLM_MODEL" \
    --task text_generation --target_device "$LLM_DEVICE" --pipeline_type "$LLM_PIPELINE" \
    --enable_prefix_caching "$LLM_PREFIX_CACHING" $LLM_EXTRA_ARGS
  # guide 3.3: intel-igc-cm into the container, then restart (needed for prefix caching on GPU)
  if [ "$LLM_PREFIX_CACHING" = true ] && [ -f "$IGC_DEB" ]; then
    RUN docker cp "$IGC_DEB" ovms-llm:/igc-cm.deb
    RUN docker exec -u 0 ovms-llm apt install -y /igc-cm.deb >/dev/null
    RUN docker restart ovms-llm
  elif [ "$LLM_PREFIX_CACHING" = true ]; then
    echo "!! $IGC_DEB missing: prefix caching will crash the GPU kernel build; set LLM_PREFIX_CACHING=false" >&2
  fi
}
up_stt() {
  local dir; dir=$(prepare_model "$OVMS_STT_DIR/$OVMS_STT_MODEL" "$OVMS_STT_MODEL")
  RUN docker rm -f ovms-stt >/dev/null 2>&1
  RUN docker run -d --name ovms-stt --restart unless-stopped -p 127.0.0.1:8001:8001 \
    --device /dev/dri --group-add "$(render_gid)" -v "$dir:/opt/ml:ro" "$OVMS_IMAGE" \
    --rest_port 8001 --model_path /opt/ml --model_name "$OVMS_STT_MODEL" \
    --task speech2text --target_device "$STT_DEVICE"
}
each() { case "$ONE" in llm) "$1_llm" ;; stt) "$1_stt" ;; all) "$1_llm"; "$1_stt" ;; *) echo "service: llm|stt|all" >&2; exit 2 ;; esac; }
down_llm() { RUN docker rm -f ovms-llm; }
down_stt() { RUN docker rm -f ovms-stt; }
status_one() {  # container port
  local code; code=$(curl -s -o /dev/null -m 3 -w "%{http_code}" "http://127.0.0.1:$2/v3/models" 2>/dev/null)
  printf "%-10s %-12s http %s\n" "$1" "$(docker inspect -f '{{.State.Status}}' "$1" 2>/dev/null || echo absent)" "$code"
}
status_llm() { status_one ovms-llm 8000; }
status_stt() { status_one ovms-stt 8001; }
case "$CMD" in
  up) each up ;;
  down) each down ;;
  restart) each down; each up ;;
  status) each status ;;
  logs) docker logs --tail 80 "ovms-${ONE/all/llm}" ;;
  *) echo "usage: $0 up|down|restart|status|logs [llm|stt|all]" >&2; exit 2 ;;
esac
