# 迁移到新机器（含只有核显的机器）

这个目录把"在另一台机器上复现当前 S2S 栈"需要的东西集中起来。现有栈的布局是
`~/work`（脚本、模型、虚拟环境）、`~/caddy-server`（前端和 HTTPS 反代），迁移后保持同样布局，启动脚本不用改路径。

```
deploy/
  sync_models.sh      在新机器上运行，从旧机器拉取模型和预构建运行时（约 27 GB）
  setup_host.sh       安装脚本、建 Python 环境、部署前端、生成 Caddyfile
  preflight.sh        只读检查：驱动、OpenVINO 设备、文件、端口
  stack/              启动栈用的脚本、stack.env 示例、各环境的依赖清单、Caddyfile 模板
```

## 步骤

1. **系统准备**（新机器）：Intel GPU 计算运行时（`intel-opencl-icd`、`libze-intel-gpu1`、`level-zero`），
   用户加入 `render` 组后重新登录；安装 `uv`、`rsync`、`git`；放好 Caddy 二进制到 `~/caddy-server/caddy`。
2. **获取代码**：`git clone https://github.com/YuanjunHuang/speech-to-speech.git`，进入仓库。
3. **拉取模型**：`deploy/sync_models.sh -n dgpu-nuc` 先看清单，确认后去掉 `-n` 正式拉取。
   - 新机器要能 SSH 到旧机器（别名或 `user@ip`）。
   - `--no-ovms-bin`：不拷 OVMS 二进制（打算用 Docker 的 OVMS 镜像时）。
   - `--with-qwen3tts`：额外拷 Qwen3-TTS 的 GGUF 与 CPU 轮子（+约 10 GB，不用 Qwen3-TTS 引擎可省）。
   - `--with-torch`：额外拷 CosyVoice2/300M 的 PyTorch 权重（+约 11 GB，仅 cosyvoice2 引擎需要）。
   - `--llm 名称`：换 LLM（默认 `minicpm-v-4_5-int4-g-ov`）。
4. **安装环境**：`deploy/setup_host.sh`。虚拟环境不能直接拷贝（里面是绝对路径），所以按 `stack/requirements/`
   里记录的版本用 `uv` 重建。它还会把前端同步到 `~/caddy-server/www`，并按本机 IP 生成 Caddyfile。
5. **改设备配置** `~/work/stack.env`（见下节），然后 `deploy/preflight.sh`，全部 ok 再继续。
6. **启动**：`bash ~/work/restore_nuc_stack.sh`，浏览器打开 `https://<本机IP>:8443/`
   （首次要信任 Caddy 的内部根证书，或者用 SSH 隧道到 `localhost`，否则浏览器不给麦克风权限）。

## 只有核显的机器：要改什么

`stack.env` 里的三个设备名改成核显的 OpenVINO 名字（`preflight.sh` 会列出，通常是 `GPU`）：

```
ACCEL_DEVICE=GPU    # CosyVoice3 的 flow / HiFT
LLM_DEVICE=GPU      # OVMS 的 LLM
STT_DEVICE=GPU      # whisper 备用 STT（默认的 Qwen3-ASR 本来就在 CPU）
```

启动时 `restore_nuc_stack.sh` 会把它们写进各模型的 `graph.pbtxt`，前端的"推理设备"面板和占用看板也会自动识别核显。

核显没有独立显存，模型权重占用系统内存，建议 **至少 32 GB**。当前的 MiniCPM-V 4.5（约 8B、INT4）在 A770 上占约 6 GB 显存，
在核显上读写的是内存，速度取决于内存带宽，**首字延迟和生成速度都会比现在慢**，需要实测后再定：

| 如果… | 可以尝试 |
| --- | --- |
| LLM 首字延迟太高 | 换更小的 LLM（`--llm` 拉取，或自行转换 Qwen3-4B / 1.7B 的 INT4 IR），改 `stack.env` 的 `LLM_MODEL` |
| 核显上 LLM 比 CPU 还慢 | `LLM_DEVICE=CPU` |
| CosyVoice3 的 flow/HiFT 在核显上慢或发音异常 | `ACCEL_DEVICE=CPU`（之前在 A770 上排查过：发音问题来自采样，不是精度/设备） |
| 内存吃紧 | 不要同时保留 whisper OVMS 实例（默认 Qwen3-ASR 已够用） |

延迟用仓库外的 `e2e_latency.py`、`tts_lead_probe.py` 这类探针测（见 `HugginFace-S2S/openvino_dgpu_plan.md`），
以"说完话到听见第一声"的端到端时间为准。

## OVMS 的两种运行方式

部署规范是"AI 模型尽量通过 OVMS 提供服务"，并参照《Ubuntu24.04 + OVMS AI Model 安装测试指南 v1》（Docker 形态）。
`stack.env` 里的 `OVMS_MODE` 选择方式：

| `OVMS_MODE` | 说明 |
| --- | --- |
| `native`（NUC 现状） | 直接跑 `~/work/ovms` 里的二进制，设备写在各模型的 `graph.pbtxt` |
| `docker`（新机器推荐） | 按指南用 `openvino/model_server:2026.4.0-gpu` 镜像，`stack/ovms_docker.sh` 管理 |

`ovms_docker.sh up|down|restart|status|logs [llm|stt|all]` 起两个容器：`ovms-llm`（:8000）和 `ovms-stt`（:8001），
与指南一致：`--device /dev/dri`、`--group-add <render 组>`、模型目录只读挂载到 `/opt/ml`、参数写在命令行，
设备用 `--target_device`（取自 `LLM_DEVICE` / `STT_DEVICE`）。`DRY_RUN=1` 只打印命令。
`restore_nuc_stack.sh` 在 `OVMS_MODE=docker` 时会自动调用它。

和指南示例不同的地方（都是有意的）：

- 管线类型自动判断：带视觉编码器的 IR（MiniCPM-V、Qwen3-VL）必须用 `VLM_CB`，OVMS 会拒绝对它用 `LM_CB`；纯文本 IR 用 `LM_CB`。
- 前缀缓存默认开，需要指南 3.3 的 `intel-igc-cm` 包：把 deb 放到 `~/work/ovms-docker/pkgs/`，脚本会自动装进容器并重启；
  没有这个包就设 `LLM_PREFIX_CACHING=false`（否则 GPU 内核编译崩溃）。NUC 实测：关闭时 TTFT 约 0.9 秒，开启后约 0.3 秒，与原生一致，工具调用正常。
- `--tool_parser qwen3coder --reasoning_parser qwen3` 只适用于 Qwen3 系列，通过 `LLM_EXTRA_ARGS` 按需填写。
  MiniCPM-V 4.5 没有对应的解析器，换成 docker 后要确认工具调用仍可用。
- 之前 native 运行留下的 `graph.pbtxt` 会固定设备并盖过 `--target_device`，所以脚本挂载的是一份**不含它的硬链接副本**
  （`~/work/ovms-docker/models/`，同一文件系统下不占额外空间）。
- 指南要求的内核 6.17 + `xe` 驱动 + 新版计算运行时针对新一代 Intel 平台；NUC（A770、`i915`、内核 6.8）不要照搬，保持 `native`。

## 不走 OVMS 的服务

这两类模型 OVMS 没有对应管线，仍作为独立服务运行，但对外同样是 OpenAI 兼容接口，管线侧只认地址，见 `SERVICES.md`：

- STT 主用 Qwen3-ASR（`:8792`，`/v1/audio/transcriptions`）。
- TTS：CosyVoice3 / CosyVoice2 桥（`:8791`，`/v1/audio/speech`）与 Qwen3-TTS。

## 已知限制

- 这套迁移流程在旧机器上做过只读检查和 `rsync -n` 演练，`ovms_docker.sh` 只做过 `DRY_RUN`，
  **还没有在真正的新机器上、也没有真正起过 OVMS 容器**。
- `stack/requirements/` 来自当前 NUC 的环境快照；PyTorch 走 CPU 轮子索引，若装出来版本不同请对照该文件。
- `qwentts-cpp-python`（Qwen3-TTS 引擎）是单独准备的 CPU 轮子，默认的 CosyVoice3 引擎不用它。
- 占用看板的 GPU 利用率读的是当前用户进程的 DRM fdinfo，用 root 或别的用户跑的模型不会计入。
