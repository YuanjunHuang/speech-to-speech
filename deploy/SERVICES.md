# 服务清单与接口约定

所有模型服务都只监听 `127.0.0.1`，由 s2s 管线通过 OpenAI 兼容接口调用；浏览器只经 Caddy 访问。

| 服务 | 端口 | 提供方 | 接口 | 健康检查 |
| --- | --- | --- | --- | --- |
| LLM（MiniCPM-V 4.5 等） | 8000 | OVMS（native 或 Docker `ovms-llm`） | `POST /v3/chat/completions`（流式） | `GET /v3/models` |
| STT 备用（whisper-small） | 8001 | OVMS（`ovms-stt`，`--task speech2text`） | `POST /v3/audio/transcriptions` | `GET /v3/models` |
| STT 主用（Qwen3-ASR INT8） | 8792 | `q3asr/qwen3_asr_service.py` | `POST /v1/audio/transcriptions` | `GET /health`、`GET /v1/models` |
| TTS（CosyVoice 桥） | 8791 | `cosyvoice_tts_service.py` | `POST /v1/audio/speech`（`style` 字段传情绪） | `GET /health`、`GET /v1/models` |
| 引擎切换 | 8790 | `tts_supervisor.py` | `GET/POST /tts`（Caddy: `/tts-admin/`） | `GET /health` |
| 占用统计 | 8793 | `sysstats_service.py` | `GET /sys-stats`、`/sys-stats/info`（Caddy 同名路径） | 同上 |
| s2s 管线 | 8765 | `speech-to-speech` | WebSocket `/v1/realtime`，`GET /v1/pool` | `GET /v1/pool` |
| 前端 | 7860 / 8080 / 8443 | uvicorn / Caddy | 静态页 + `/api/*` | — |

STT 和 TTS 的地址、模型名由 `start_s2s_gpu.sh` 传给管线（`--openai_stt_*`、`--openai_tts_*`），要换成别的 OpenAI 兼容后端只改这里。

## 快速自检

```bash
curl -s localhost:8000/v3/models | head -c 200      # LLM
curl -s localhost:8001/v3/models | head -c 200      # whisper（OVMS）
curl -s localhost:8792/v1/models                    # Qwen3-ASR
curl -s localhost:8791/v1/models                    # CosyVoice 桥
curl -s localhost:8790/tts | head -c 200            # 当前引擎与各模块设备
```
