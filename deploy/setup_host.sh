#!/usr/bin/env bash
# One-time host setup after sync_models.sh: Python envs, stack scripts, front end, Caddy.
# Run from the repo checkout (deploy/ is next to demo/ and src/). Needs: uv, rsync, a Caddy binary.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
W=$HOME/work; mkdir -p "$W/logs" "$W/state"
IP=${NUC_IP:-$(hostname -I | awk '{print $1}')}
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }

echo "== stack scripts -> $W"
for f in start_s2s_gpu.sh restore_nuc_stack.sh ovms_docker.sh tts_supervisor.py cosyvoice_tts_service.py sysstats_service.py; do
  install -m 0755 "$HERE/stack/$f" "$W/$f"
done
[ -f "$W/stack.env" ] || cp "$HERE/stack/stack.env.example" "$W/stack.env"
install -m 0644 "$HERE/stack/qwen3_asr_service.py" "$W/q3asr/qwen3_asr_service.py"
install -m 0644 "$HERE/stack/ov_cosyvoice_helper.py" "$W/cv3ov/ov_cosyvoice_helper.py"

echo "== python envs (uv)"
mkenv() {  # path python requirements [extra uv args]
  local path=$1 py=$2 req=$3; shift 3
  [ -x "$path/bin/python" ] || uv venv --python "$py" "$path"
  uv pip install --python "$path/bin/python" -r "$HERE/stack/requirements/$req" \
      --extra-index-url https://download.pytorch.org/whl/cpu "$@"
}
mkenv "$W/speech-to-speech/.venv" 3.11 speech-to-speech_venv.txt
uv pip install --python "$W/speech-to-speech/.venv/bin/python" -e "$REPO"
mkenv "$W/cv3ov/venv"             3.10 cv3ov_venv.txt
mkenv "$W/q3asr/venv"             3.11 q3asr_venv.txt
mkenv "$W/demo-env/.venv"         3.11 demo-env_venv.txt
[ "${WITH_TORCH:-0}" = 1 ] && mkenv "$W/cosyvoice/.venv" 3.10 cosyvoice_venv.txt

echo "== front end"
[ -e "$W/speech-to-speech" ] || ln -s "$REPO" "$W/speech-to-speech"
mkdir -p "$HOME/caddy-server/www"
rsync -a --exclude node_modules --exclude tests "$REPO/demo/" "$HOME/caddy-server/www/"
sed "s/__IP__/$IP/" "$HERE/stack/Caddyfile.template" > "$HOME/caddy-server/Caddyfile"
[ -x "$HOME/caddy-server/caddy" ] || echo "!! put a caddy binary at ~/caddy-server/caddy (https://caddyserver.com/download)"
echo "Edit $W/stack.env (devices!), run ./preflight.sh, then: bash $W/restore_nuc_stack.sh"
