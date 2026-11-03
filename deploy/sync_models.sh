#!/usr/bin/env bash
# Copy the model files and prebuilt runtimes of the S2S stack from a working host
# (default alias: dgpu-nuc) to this machine. Run it ON THE NEW MACHINE (pull mode):
#     ./sync_models.sh dgpu-nuc            # everything needed for the CosyVoice3 + MiniCPM-V stack
#     ./sync_models.sh -n dgpu-nuc         # dry run: list what would be copied
#     ./sync_models.sh --llm qwen3-8b-int4-ov dgpu-nuc
# Options: --no-ovms-bin (use Docker/own OVMS)  --with-qwen3tts (+9.7 GB GGUF)  --with-torch (+11 GB CosyVoice2/300M)
# Paths mirror the source layout under ~/work, ~/.cache and ~/caddy-server, which the launchers expect.
set -euo pipefail
LLM=minicpm-v-4_5-int4-g-ov; DRY=""; OVMS_BIN=1; QWEN3TTS=0; TORCH=0
while [ $# -gt 1 ]; do
  case "$1" in
    -n|--dry-run) DRY="-n" ;;
    --llm) LLM=$2; shift ;;
    --no-ovms-bin) OVMS_BIN=0 ;;
    --with-qwen3tts) QWEN3TTS=1 ;;
    --with-torch) TORCH=1 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
SRC=${1:?usage: sync_models.sh [options] SOURCE_HOST}
R=(rsync -a --info=progress2 $DRY)
pull() {  # path relative to $HOME on the source; optional extra rsync args
  local rel=$1; shift
  echo ">> $rel"
  [ -n "$DRY" ] || mkdir -p "$HOME/$(dirname "$rel")"
  "${R[@]}" "$@" "$SRC:$rel" "$HOME/$(dirname "$rel")/"
}
W=work
[ $OVMS_BIN = 1 ] && { pull $W/ovms/ovms; pull $W/ovms_ov_only; }
pull $W/ovms/models/$LLM
pull $W/ovms/models_stt/whisper-small-int8-ov
# Qwen3-ASR (INT8 IR + the HF processor/tokenizer, the service runs with HF_HUB_OFFLINE=1)
pull $W/q3asr/Qwen3-ASR-0.6B-hf-ov-int8
pull .cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B-hf
# VAD / turn detection
pull .cache/huggingface/hub/models--pipecat-ai--smart-turn-v3
pull .cache/torch/hub/snakers4_silero-vad_master
# CosyVoice3 OpenVINO IRs (flow/HiFT/embeddings from -ov, LLM from -ov-int8)
pull $W/cv3ov/Fun-CosyVoice3-0.5B-2512-ov
pull $W/cv3ov/Fun-CosyVoice3-0.5B-2512-ov-int8
# CosyVoice code + prompt assets; PyTorch weights only for the v3 frontend unless --with-torch
EX=(--exclude '/pretrained_models/CosyVoice2-0.5B' --exclude '/pretrained_models/CosyVoice-300M-Instruct' --exclude '.git')
[ $TORCH = 1 ] && EX=(--exclude '.git')
pull $W/cosyvoice/CosyVoice "${EX[@]}"
if [ $QWEN3TTS = 1 ]; then
  pull .cache/huggingface/hub/models--Serveurperso--Qwen3-TTS-GGUF
  pull tmp/qwentts_cpu
fi
echo "done. Next: ./setup_host.sh"
