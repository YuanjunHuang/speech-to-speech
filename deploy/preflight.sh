#!/usr/bin/env bash
# Read-only checks to run before starting the stack on a new machine.
W=$HOME/work; ok=0; bad=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  ok   $1"; ok=$((ok+1)); else echo "  FAIL $1"; bad=$((bad+1)); fi; }
[ -f "$W/stack.env" ] && { set -a; . "$W/stack.env"; set +a; }
LLM_MODEL=${LLM_MODEL:-minicpm-v-4_5-int4-g-ov}
echo "== hardware"
echo "  cpu: $(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2}')   ram: $(free -g | awk '/Mem/{print $2" GB"}')   free disk in ~: $(df -h ~ | awk 'NR==2{print $4}')"
chk "user is in the render group" "id -nG | grep -qw render"
chk "/dev/dri/renderD* present" "ls /dev/dri/renderD*"
chk "level-zero / compute runtime installed" "ldconfig -p | grep -qE 'libze_intel_gpu|libigdrcl'"
echo "== OpenVINO devices (from the cv3ov env)"
"$W/cv3ov/venv/bin/python" -c "import openvino as ov; c=ov.Core(); [print('  ', d, '-', c.get_property(d,'FULL_DEVICE_NAME')) for d in c.available_devices]" 2>&1 | head -6
echo "  configured: ACCEL_DEVICE=${ACCEL_DEVICE:-?} LLM_DEVICE=${LLM_DEVICE:-?} STT_DEVICE=${STT_DEVICE:-?}  (must appear in the list above)"
echo "== files"
if [ "${OVMS_MODE:-native}" = docker ]; then
  chk "docker usable (user in docker group)" "docker info"
  chk "OVMS image ${OVMS_IMAGE:-openvino/model_server:2026.4.0-gpu}" "docker image inspect ${OVMS_IMAGE:-openvino/model_server:2026.4.0-gpu}"
else
  chk "OVMS binary" "[ -x $W/ovms/ovms/bin/ovms ]"
fi
chk "LLM IR $LLM_MODEL" "[ -f ${OVMS_MODELS_DIR:-$W/ovms/models}/$LLM_MODEL/graph.pbtxt ]"
chk "whisper IR" "[ -d ${OVMS_STT_DIR:-$W/ovms/models_stt}/${OVMS_STT_MODEL:-whisper-small-int8-ov} ]"
chk "Qwen3-ASR INT8" "[ -d $W/q3asr/Qwen3-ASR-0.6B-hf-ov-int8 ]"
chk "CosyVoice3 OV IR (flow/HiFT)" "[ -f $W/cv3ov/Fun-CosyVoice3-0.5B-2512-ov/openvino_hift_model.xml ]"
chk "CosyVoice3 OV IR (int8 LLM)" "[ -f $W/cv3ov/Fun-CosyVoice3-0.5B-2512-ov-int8/openvino_model.xml ]"
chk "CosyVoice repo + prompt asset" "[ -f $W/cosyvoice/CosyVoice/asset/zero_shot_prompt.wav ]"
chk "smart-turn model" "ls $HOME/.cache/huggingface/hub/models--pipecat-ai--smart-turn-v3"
chk "caddy binary" "[ -x $HOME/caddy-server/caddy ]"
echo "== python envs"
for e in speech-to-speech/.venv cv3ov/venv q3asr/venv demo-env/.venv; do chk "$e" "[ -x $W/$e/bin/python ]"; done
echo "== ports free (stack not yet running)"
for p in 8000 8001 8765 8790 8791 8792 8793 7860 8080 8443; do chk "port $p free" "! ss -ltn | grep -q ':$p '"; done
echo; echo "$ok ok, $bad failed (failed port checks just mean the stack is already running)"
