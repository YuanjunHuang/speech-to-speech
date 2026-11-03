// @ts-check
/**
 * Minimal voice conversation app, talking to a Hugging Face speech-to-speech
 * backend over **WebSocket** or **WebRTC**.
 *
 * Click the orb -> we ask for the mic, connect (WS dial, or SDP handshake via
 * the /api/calls proxy), push session.update + mic audio, play back the TTS
 * audio. The orb visually reflects the live state (idle, connecting,
 * listening, user-speaking, processing, ai-speaking).
 *
 * One adapter drives the official Agents SDK RealtimeSession over either stock
 * transport. WebRTC is offered only in env-pinned direct mode (the /api/calls
 * proxy forwards exclusively to SPEECH_TO_SPEECH_URL); LB mode and user-typed
 * URLs stay on WebSocket.
 *
 * @typedef {"idle" | "connecting" | "queued" | "your-turn" | "listening" | "user-speaking" | "processing" | "ai-speaking" | "error"} AppState
 * @typedef {S2sRealtimeClient} RealtimeClient
 */

import { S2sRealtimeClient, normalizePlaybackBufferMs } from "./s2s-realtime-client.js?v=audio-24k-v5";
import { $, truncateError, errorMessage, DEBUG } from "./ui/dom.js";
import { ChatView } from "./ui/chat.js?v=bubble-hold-v1";
import { Account } from "./ui/account.js";
import { initSystemInfo, setStageModule, setTtsModule, getAccelBadge, whenHostInfoLoaded } from "./ui/system-info.js?v=sysinfo-v10";
import { initSysMonitor } from "./ui/sys-monitor.js?v=sysmon-v2";
import { initStylePicker } from "./ui/style-picker.js?v=style-v3";
import { createTtsEnginePicker } from "./tts-engine.js?v=engine-v2";

const DEFAULT_VOICE = "Vivian";
const DEFAULT_INSTRUCTIONS = "You are a friendly voice assistant.";

const STORAGE_KEYS = {
  // Direct s2s server URL, used only when the deploy has no LOAD_BALANCER_URL
  // (in LB mode the browser never learns the LB address — it POSTs /api/session).
  directUrl: "s2s.ws.directUrl",
  voice: "s2s.ws.voice",
  instructions: "s2s.ws.instructions",
  tools: "s2s.ws.tools",
  searchKey: "s2s.ws.searchKey",
  noiseGate: "s2s.ws.noiseGate",
  // ".v2": the default moved 0 -> 500 ms when the backend became Qwen3-TTS,
  // whose audio arrives in ~0.5 s bursts; a stored 0 would keep the old choppy
  // start. New key, so existing browsers pick up the new default once.
  playbackBufferMs: "s2s.ws.playbackBufferMs.v2",
  localVad: "s2s.audio.localVad",
  // "ws" | "webrtc". Not under the historical "s2s.ws." prefix — it selects
  // between the transports rather than configuring the WS one.
  transport: "s2s.transport",
  audioInputId: "s2s.audio.inputId",
  audioOutputId: "s2s.audio.outputId",
};

// ── Noise gate ──────────────────────────────────────────────────────────────
// The Settings cursor sets the gate's open threshold in dBFS. Its leftmost
// position is an OFF detent (gate disabled, pure passthrough); the rest of the
// travel is the active threshold. The cursor shares the meter's dB axis, so the
// handle sits on the level bar — raise it until room noise stops lighting it up.
// The slider range IS the shared axis: the live meter fill and the threshold
// thumb both map across [GATE_OFF_DB, GATE_MAX_DB], so the thumb sits exactly
// where the gate cuts on the same scale as the level bar.
const GATE_OFF_DB = -66; // slider minimum = off / bottom of the meter axis
const GATE_MAX_DB = -3; // slider maximum = most aggressive / top of the meter axis
const GATE_DEFAULT_DB = -50; // first-run default: a gentle gate, enabled

/** @param {number} thresholdDb @returns {import("./s2s-realtime-client.js").NoiseGate} */
function gateParams(thresholdDb) {
  return { enabled: thresholdDb > GATE_OFF_DB, thresholdDb };
}

// ── Tools ─────────────────────────────────────────────────────────────────
// Function tools we declare to the backend. The model decides when to call
// one; the executor below runs it and returns the result (see runTool).
/** @type {Record<string, import("./s2s-realtime-client.js").ToolDef>} */
const TOOL_DEFS = {
  web_search: {
    type: "function",
    name: "web_search",
    description:
      "Search the web for current or factual information you don't already know " +
      "(news, prices, facts, documentation). Returns the top results with titles, " +
      "snippets and URLs.",
    parameters: {
      type: "object",
      properties: { query: { type: "string", description: "The search query." } },
      required: ["query"],
    },
  },
  get_weather: {
    type: "function",
    name: "get_weather",
    description:
      "Get the real-time weather and a short forecast for a city. Always use this for any " +
      "weather question instead of answering from memory. City may be in Chinese or English.",
    parameters: {
      type: "object",
      properties: {
        city: { type: "string", description: "City name, e.g. 北京 or Shanghai." },
        days: { type: "integer", description: "Forecast days (1-7). Default 3." },
      },
      required: ["city"],
    },
  },
  camera_snapshot: {
    type: "function",
    name: "camera_snapshot",
    description:
      "Capture the current frame from the user's webcam so you can see what they " +
      "are showing you. Use it whenever the user refers to something visual or " +
      "asks you to look.",
    parameters: { type: "object", properties: {}, required: [] },
  },
};

/** Longest edge of the snapshot sent to the VLM, in px (keeps payload sane). */
const SNAPSHOT_MAX_EDGE = 768;
const SNAPSHOT_QUALITY = 0.7;
// Over WebRTC the snapshot travels inside ONE data-channel message, and SCTP
// messages above the negotiated max (64 KiB on the aiortc side) fail to send.
// Budget for the data-URL portion, leaving headroom for the JSON envelope.
const SNAPSHOT_DC_BUDGET_CHARS = 60_000;
// Re-encode ladder walked until the frame fits the budget: quality first
// (cheap wins), then resolution. The last rung is ~15–25 KB for any content,
// so a frame that "fits" is deterministic, not content-dependent luck.
const SNAPSHOT_LADDER = /** @type {[number, number][]} */ ([
  [SNAPSHOT_MAX_EDGE, SNAPSHOT_QUALITY],
  [SNAPSHOT_MAX_EDGE, 0.5],
  [640, 0.45],
  [512, 0.4],
  [448, 0.35],
  [384, 0.3],
]);

function loadSettings() {
  return {
    directUrl: localStorage.getItem(STORAGE_KEYS.directUrl) || "",
    voice: localStorage.getItem(STORAGE_KEYS.voice) || DEFAULT_VOICE,
    instructions: localStorage.getItem(STORAGE_KEYS.instructions) || DEFAULT_INSTRUCTIONS,
    noiseGate: loadGateThreshold(),
    playbackBufferMs: normalizePlaybackBufferMs(localStorage.getItem(STORAGE_KEYS.playbackBufferMs)),
    localVad: localStorage.getItem(STORAGE_KEYS.localVad) !== "off",
    // Default WebSocket: the proven path stays the first-run experience.
    transport: localStorage.getItem(STORAGE_KEYS.transport) === "webrtc" ? "webrtc" : "ws",
    audioInputId: localStorage.getItem(STORAGE_KEYS.audioInputId) || "",
    audioOutputId: localStorage.getItem(STORAGE_KEYS.audioOutputId) || "",
  };
}

/** Stored gate threshold (dBFS), clamped to the slider range. Defaults to a
 * gentle enabled gate (GATE_DEFAULT_DB) when the user hasn't set one yet. */
function loadGateThreshold() {
  const stored = localStorage.getItem(STORAGE_KEYS.noiseGate);
  // getItem returns null when unset, and Number(null) === 0 (finite!), so guard
  // the missing/empty case explicitly before coercing — otherwise the default
  // never fires and 0 clamps to the slider max.
  if (stored === null || stored === "") return GATE_DEFAULT_DB;
  const raw = Number(stored);
  if (!Number.isFinite(raw)) return GATE_DEFAULT_DB;
  return Math.min(GATE_MAX_DB, Math.max(GATE_OFF_DB, Math.round(raw)));
}

/** @param {ReturnType<typeof loadSettings>} s */
function saveSettings(s) {
  localStorage.setItem(STORAGE_KEYS.directUrl, s.directUrl);
  localStorage.setItem(STORAGE_KEYS.voice, s.voice);
  localStorage.setItem(STORAGE_KEYS.instructions, s.instructions);
  localStorage.setItem(STORAGE_KEYS.noiseGate, String(s.noiseGate));
  localStorage.setItem(STORAGE_KEYS.playbackBufferMs, String(s.playbackBufferMs));
  localStorage.setItem(STORAGE_KEYS.localVad, s.localVad ? "on" : "off");
  localStorage.setItem(STORAGE_KEYS.transport, s.transport);
  localStorage.setItem(STORAGE_KEYS.audioInputId, s.audioInputId || "");
  localStorage.setItem(STORAGE_KEYS.audioOutputId, s.audioOutputId || "");
}

/** @returns {{ web_search: boolean, get_weather: boolean, camera_snapshot: boolean }} */
function loadTools() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEYS.tools) || "{}");
    // Web search defaults ON (it still only activates when a key exists).
    // Camera defaults OFF: showing a live self-view and silently requesting
    // camera permission on every page load is surprising, and the active LLM
    // backend may not even be a vision-language model that can use the frame.
    // The user can still switch it on from the Tools panel.
    return {
      web_search: raw.web_search ?? true,
      get_weather: raw.get_weather ?? true,
      camera_snapshot: raw.camera_snapshot ?? false,
    };
  } catch {
    return { web_search: true, get_weather: true, camera_snapshot: false };
  }
}

function saveTools() {
  localStorage.setItem(STORAGE_KEYS.tools, JSON.stringify(toolsEnabled));
}

/** @type {Record<AppState, { caption: string; disabled: boolean }>} */
const STATE_VIEWS = {
  idle:            { caption: "点击开始",      disabled: false },
  connecting:      { caption: "连接中",        disabled: true  },
  queued:          { caption: "正在为你寻找空位…", disabled: true },
  "your-turn":     { caption: "轮到你了!🎉", disabled: true  },
  listening:       { caption: "",              disabled: false },
  "user-speaking": { caption: "",              disabled: false },
  processing:      { caption: "",              disabled: false },
  "ai-speaking":   { caption: "",              disabled: false },
  error:           { caption: "点击重试",      disabled: false },
};

/** @type {Record<AppState, string>} */
const STATE_CLASS = {
  idle: "state-idle",
  connecting: "state-connecting",
  queued: "state-queued",
  "your-turn": "state-your-turn",
  listening: "state-listening",
  "user-speaking": "state-user-speaking",
  processing: "state-processing",
  "ai-speaking": "state-ai-speaking",
  error: "state-error",
};

/** @type {ReadonlySet<AppState>} */
const LIVE_STATES = new Set(["listening", "user-speaking", "processing", "ai-speaking"]);

/** @type {HTMLButtonElement} */
const circleBtn = $("#main-circle");
/** @type {HTMLParagraphElement} */
const circleCaption = $("#circle-caption");
/** @type {HTMLParagraphElement} */
const circleSubcaption = $("#circle-subcaption");
/** @type {HTMLElement} */
const orbWrap = $(".orb-wrap");
/** @type {HTMLButtonElement} */
const micBtn = $("#mic-btn");
/** @type {HTMLButtonElement} */
const stopBtn = $("#stop-btn");
/** @type {HTMLElement} */
const queueActions = $("#queue-actions");
/** @type {HTMLButtonElement} */
const joinQueueBtn = $("#join-queue-btn");
/** @type {HTMLButtonElement} */
const leaveQueueBtn = $("#leave-queue-btn");

/** @type {HTMLButtonElement} */
const settingsBtn = $("#settings-btn");
/** @type {HTMLDialogElement} */
const settingsModal = $("#settings-modal");

/** @type {HTMLButtonElement} */
const aboutBtn = $("#about-btn");
/** @type {HTMLDialogElement} */
const aboutModal = $("#about-modal");
/** @type {HTMLButtonElement} */
const aboutClose = $("#about-close");

/** @type {HTMLButtonElement} */
const toolsBtn = $("#tools-btn");
/** @type {HTMLDialogElement} */
const toolsModal = $("#tools-modal");
/** @type {HTMLButtonElement} */
const toolsClose = $("#tools-close");
/** @type {HTMLInputElement} */
const toolWebSwitch = $("#tool-web");
/** @type {HTMLInputElement} */
const toolCamSwitch = $("#tool-cam");
const toolWeatherSwitch = $("#tool-weather");
/** @type {HTMLElement} */
const toolWebRow = $("#tool-web-row");
/** @type {HTMLElement} */
const toolWebHint = $("#tool-web-hint");
/** @type {HTMLElement} */
const toolCamHint = $("#tool-cam-hint");
/** @type {HTMLInputElement} */
const searchKeyInput = $("#search-key");
/** @type {HTMLElement} */
const camPip = $("#cam-pip");
/** @type {HTMLVideoElement} */
const camVideo = $("#cam-video");

/** @type {HTMLInputElement} */
const inputLbUrl = $("#lb-url");
/** @type {HTMLElement} */
const connField = $("#conn-field");
/** @type {HTMLElement} */
const connHint = $("#conn-hint");
/** @type {HTMLElement} */
const transportField = $("#transport-field");
/** @type {HTMLSelectElement} */
const inputTransport = $("#transport");
/** @type {HTMLElement} */
const transportHint = $("#transport-hint");
/** @type {HTMLElement} */
const gateField = $("#gate-field");
/** @type {HTMLElement} */
const playbackBufferField = $("#playback-buffer-field");
/** @type {HTMLInputElement} */
const inputPlaybackBuffer = $("#playback-buffer-ms");
/** @type {HTMLSelectElement} */
const inputLocalVad = $("#local-vad");
/** @type {HTMLSelectElement} */
const inputVoice = $("#voice");
/** @type {HTMLSelectElement} */
const inputAudioInput = $("#audio-input");
/** @type {HTMLSelectElement} */
const inputAudioOutput = $("#audio-output");
/** @type {HTMLElement} */
const audioOutputHint = $("#audio-output-hint");
const audioTestBtn = $("#audio-test");
const audioTestStatus = $("#audio-test-status");

// Plays a short two-tone beep through a fresh AudioContext on the chosen output,
// entirely client-side. It separates "the browser/speaker is silent" from "the
// pipeline sent no audio" without needing a microphone or a server round trip.
audioTestBtn.addEventListener("click", async () => {
  const say = (text, kind) => {
    audioTestStatus.hidden = !text;
    audioTestStatus.textContent = text;
    audioTestStatus.dataset.kind = kind;
  };
  try {
    const Ctx = window.AudioContext || /** @type {any} */ (window).webkitAudioContext;
    const ctx = new Ctx();
    await ctx.resume();
    const sink = inputAudioOutput.value || "";
    let sinkNote = "系统默认设备";
    if (typeof /** @type {any} */ (ctx).setSinkId === "function" && sink) {
      await /** @type {any} */ (ctx).setSinkId(sink);
      sinkNote = inputAudioOutput.selectedOptions[0]?.textContent || sink;
    }
    const gain = ctx.createGain();
    gain.gain.value = 0.2;
    gain.connect(ctx.destination);
    [660, 880].forEach((freq, i) => {
      const osc = ctx.createOscillator();
      osc.frequency.value = freq;
      osc.connect(gain);
      osc.start(ctx.currentTime + i * 0.35);
      osc.stop(ctx.currentTime + i * 0.35 + 0.3);
    });
    say(`已向「${sinkNote}」播放提示音（AudioContext: ${ctx.state}, ${ctx.sampleRate} Hz）。听到了吗？`, "ok");
    setTimeout(() => void ctx.close().catch(() => {}), 1500);
  } catch (err) {
    say(`播放失败：${err instanceof Error ? err.message : err}`, "error");
  }
});
/** @type {HTMLTextAreaElement} */
const inputInstructions = $("#instructions");
/** @type {HTMLInputElement} */
const inputNoiseGate = $("#noise-gate");
/** @type {HTMLElement} */
const gateValue = $("#gate-value");
/** @type {HTMLElement} */
const gateMeterFill = $("#gate-meter-fill");
/** @type {HTMLElement} */
const micGate = $("#mic-gate");
const mgaArc = /** @type {SVGSVGElement} */ (document.querySelector("#mic-gate-arc"));
const mgaTrack = /** @type {SVGPathElement} */ (document.querySelector("#mga-track"));
const mgaFill = /** @type {SVGPathElement} */ (document.querySelector("#mga-fill"));
const mgaHit = /** @type {SVGPathElement} */ (document.querySelector("#mga-hit"));
const mgaHandle = /** @type {SVGCircleElement} */ (document.querySelector("#mga-handle"));
/** @type {HTMLButtonElement} */
const restartBtn = $("#restart-conversation");
/** @type {HTMLElement} */
const restartHint = $("#restart-hint");
const settingsForm = /** @type {HTMLFormElement} */ (settingsModal.querySelector("form"));

/** @type {AppState} */
let currentState = "idle";
let settings = loadSettings();

// ── Connection target ────────────────────────────────────────────────────────
// Three modes, decided by the deploy via /api/config:
//   • SPEECH_TO_SPEECH_URL set -> direct mode pinned by the deploy: the browser
//     connects straight to that URL, shown read-only in Settings. Overrides the
//     load balancer entirely.
//   • LOAD_BALANCER_URL set  -> original flow: POST the same-origin /api/session
//     proxy (the server forwards to the LB; the LB address is never sent here).
//   • neither (allowDirect)  -> the user sets a speech-to-speech server URL and
//     the browser connects to it directly (no load balancer, no /session).
let lbMode = false;
// Fail open: direct entry is allowed unless /api/config reports an LB URL. This
// way a missing/unreachable config (e.g. static hosting) leaves the field
// usable rather than locked.
let allowDirect = true;
// Deploy-pinned s2s URL (SPEECH_TO_SPEECH_URL). Non-empty -> locked direct
// mode: the field displays it read-only and the saved user URL is untouched.
let pinnedUrl = "";
// Whether the deploy offers the WebRTC transport (/api/config `rtc`; true
// exactly when the URL is env-pinned, since /api/calls only forwards there).
let rtcAvailable = false;
/** @type {RTCIceServer[]} STUN/TURN servers for the browser peer connection
 * (deploy-provided via RTC_ICE_SERVERS; empty -> host candidates only). */
let iceServers = [];
// Optional hidden user prompt supplied by the deployment. When non-empty, the
// client asks the model to greet once after the initial session configuration.
let startupGreeting = "";
// Transport of the LIVE (or starting) conversation — as opposed to
// `settings.transport`, which is what the NEXT one will use. Drives the
// camera-snapshot size budget while a call is running.
/** @type {"ws" | "webrtc"} */
let activeTransport = "ws";

// ── Tool state ──────────────────────────────────────────────────────────────
let toolsEnabled = loadTools();
// Whether the server holds a Serper key (learned from /api/config on load).
let serverSearchKey = false;
// A user-supplied key (fallback when the deploy has none). localStorage only.
let userSearchKey = localStorage.getItem(STORAGE_KEYS.searchKey) || "";
/** @type {MediaStream | null} */
let cameraStream = null;

/** Search is usable if the server has a key or the user supplied one. */
function searchAvailable() {
  return serverSearchKey || !!userSearchKey;
}

/** Tool definitions for the currently-enabled (and usable) tools. */
function activeToolDefs() {
  const defs = [];
  if (toolsEnabled.web_search && searchAvailable()) defs.push(TOOL_DEFS.web_search);
  if (toolsEnabled.get_weather) defs.push(TOOL_DEFS.get_weather);
  if (toolsEnabled.camera_snapshot) defs.push(TOOL_DEFS.camera_snapshot);
  return defs;
}

/** Push the active tool set to a live session so toggles apply mid-call. */
function pushToolsToSession() {
  if (!client || !LIVE_STATES.has(currentState)) return;
  client.setTools(activeToolDefs());
}

// ── Chat view ───────────────────────────────────────────────────────────────
// Owns the history panel, the ephemeral bubbles, and all transcript/tool
// streaming state. The client's events are forwarded to its on* methods.
let userAudioReplaying = false;
const chat = new ChatView({
  onUserAudioPlaybackChange(playing) {
    userAudioReplaying = playing;
    syncMicMuteState();
  },
});

// ── Account / limiter ─────────────────────────────────────────────────────
// Login chip + daily-limit modal (inert unless the deploy is in LB mode). The
// server meters conversation time; the client just heartbeats a live session
// and tears down when the server reports the budget is spent.
const account = new Account();

// ── System info rail ──────────────────────────────────────────────────────
// Renders the inference-host hardware inventory and the stage→device map into
// the left rail, and wires the topbar toggle / drawer affordances.
initSystemInfo();

// ── Voice style picker ────────────────────────────────────────────────────
// 情绪固定：所选情绪作为 Realtime session 的扩展字段下发，后端优先采用它
// （见 LLM/lm_output_processor.py::_session_style）；"自动" 则由模型自己打标签。
// `client` is declared *before* the picker initialises on purpose: initStylePicker
// calls back once with the stored pin, and that callback reads `client`. Declaring
// it below would leave `client` in the temporal dead zone, throw a ReferenceError,
// and abort this whole module — which silently kills the start button.
/** @type {RealtimeClient | null} */
let client = null;
// The pin is also re-applied on every (re)connect, so a session always carries it.
const stylePin = initStylePicker((style) => {
  if (client) client.setStyle(style);
});

// ── TTS engine picker ─────────────────────────────────────────────────────
// Qwen3-TTS vs the CosyVoice bridge. The backend is a startup flag, so a switch
// restarts the pipeline through tts_supervisor.py (:8790 via Caddy /tts-admin).
// We do NOT auto-reconnect afterwards: resuming an AudioContext outside a user
// gesture can leave it suspended, so the user starts the next call themselves.
// The STT backend is chosen by the pipeline launcher and reported by the supervisor
// (it can fall back to whisper), so the rail must follow it rather than be hardcoded.
// The LLM is whatever OVMS :8000 serves; known models get a tidy label, anything
// else is shown by its served id rather than hidden behind a stale name.
const LLM_PANEL = {
  "minicpm-v-4_5-int4-g-ov": "MiniCPM-V 4.5 · INT4",
  "minicpm-v-4_5-int4-cw-ov": "MiniCPM-V 4.5 · INT4 (cw)",
  "qwen3-8b-int4-ov": "Qwen3-8B · INT4",
  "qwen3.5-4b-int8-ov": "Qwen3.5-4B · INT8",
};

const STT_PANEL = {
  qwen3asr: { name: "Qwen3-ASR-0.6B · INT8", job: "语音转文字", detail: "OpenVINO · CPU", device: "cpu" },
  whisper: { name: "whisper-small · INT8", job: "语音转文字", detail: "OVMS :8001", device: "gpu1" },
};

const TTS_PANEL = {
  qwen3: { name: "Qwen3-TTS · 1.7B CustomVoice", job: "语音合成 · 情绪 instruct", detail: "GGML", device: "cpu" },
  cosyvoice2: { name: "CosyVoice2 · 0.5B", job: "语音合成 · 整段渲染", detail: "PyTorch · HTTP bridge", device: "cpu" },
  cosyvoice3: { name: "CosyVoice3 · 0.5B", job: "语音合成 · 逐句", detail: "LLM·CPU · flow/HiFT·A770", device: "gpu1" },
};

/** "CPU" stays on the CPU row; any GPU.* name means the accelerator row. */
const deviceKey = (/** @type {string} */ d) => (/^cpu$/i.test(d) ? "cpu" : "gpu1");

const ttsEngine = createTtsEnginePicker({
  select: /** @type {HTMLSelectElement} */ ($("#tts-engine")),
  hint: /** @type {HTMLElement} */ ($("#tts-engine-hint")),
  status: /** @type {HTMLElement} */ ($("#tts-engine-status")),
  voiceSelect: inputVoice,
  onSwitchStart: async () => {
    if (client) await teardown();
  },
  onActive: (engine, state) => {
    const info = TTS_PANEL[engine];
    if (info) {
      setTtsModule({
        ...info,
        detail: info.detail.replace("flow/HiFT·A770", `flow/HiFT·${getAccelBadge()}`),
      });
    }
    const sttPanel = state?.stt && STT_PANEL[state.stt.id];
    if (sttPanel) {
      const dev = state.stt.device;
      setStageModule("STT", dev ? { ...sttPanel, device: deviceKey(dev) } : sttPanel);
    }
    const llmId = state?.llm?.id;
    if (llmId) {
      const dev = state.llm.device;
      setStageModule("LLM", { name: LLM_PANEL[llmId] || llmId, ...(dev ? { device: deviceKey(dev) } : {}) });
    }
  },
});
void ttsEngine.refresh();
void whenHostInfoLoaded().then(() => ttsEngine.refresh());
initSysMonitor({ select: /** @type {HTMLSelectElement} */ ($("#sys-monitor")) });
let limiterOn = false;
let heartbeatTimer = 0;
let trackedSessionId = "";
let trackedTier = "";
// The waiting-queue ticket id while we're in line (else ""). Used to leave the
// queue on teardown / tab-close so we don't hold a phantom place.
let queuedTicketId = "";

/** @type {MediaStream | null} */
let micStream = null;
let micMuted = false;

/** Apply both the user's mute choice and the temporary replay guard. */
function syncMicMuteState() {
  const muted = micMuted || userAudioReplaying;
  for (const track of micStream?.getAudioTracks() ?? []) {
    track.enabled = !muted;
  }
  client?.setMuted(muted);
}

/** @param {AppState} next */
function setState(next) {
  currentState = next;
  const view = STATE_VIEWS[next];
  circleBtn.disabled = view.disabled;
  circleBtn.className = `circle ${STATE_CLASS[next]}`;
  if (next !== "error") setCaption(view.caption);

  const live = LIVE_STATES.has(next);
  orbWrap.classList.toggle("live", live);
  micBtn.setAttribute("aria-hidden", live ? "false" : "true");
  stopBtn.setAttribute("aria-hidden", live ? "false" : "true");
  micBtn.tabIndex = live ? 0 : -1;
  stopBtn.tabIndex = live ? 0 : -1;

  // Queue affordances: "Leave queue" whenever we're in line; "Join now" only once
  // it's our turn (a slot is held for us). Both live under #queue-actions.
  const yourTurn = next === "your-turn";
  const inLine = next === "queued" || yourTurn;
  queueActions.hidden = !inLine;
  joinQueueBtn.hidden = !yourTurn;
  joinQueueBtn.tabIndex = yourTurn ? 0 : -1;
  leaveQueueBtn.hidden = !inLine;
  leaveQueueBtn.tabIndex = inLine ? 0 : -1;
  if (!yourTurn) stopJoinCountdown();

  // Warm reassurance under the terse position, only while waiting in line.
  if (next === "queued") {
    circleSubcaption.textContent =
      "抱歉,我们被挤爆啦!🤗 当前所有坑位都满了,已经为你保留了一个位置,马上就好。";
    circleSubcaption.hidden = false;
  } else {
    circleSubcaption.hidden = true;
  }

  updateRestartAvailability();
}

function updateRestartAvailability() {
  // Restart works from any settled state — it tears down a live call (if any)
  // and reconnects with the current settings. Only block while mid-connect or
  // while waiting in the queue (restarting from there would just re-queue).
  restartBtn.disabled =
    currentState === "connecting" || currentState === "queued" || currentState === "your-turn";
  restartHint.hidden = false;
  restartHint.textContent = LIVE_STATES.has(currentState)
    ? "使用以上设置立即重新连接。"
    : "使用以上设置开始一次对话。";
}

/**
 * @param {string} text
 * @param {"" | "error" | "muted"} [kind]
 */
function setCaption(text, kind = "") {
  const trimmed = text.trim();
  circleCaption.textContent = trimmed;
  circleCaption.className = `circle-caption${kind ? ` ${kind}` : ""}${trimmed ? "" : " empty"}`;
}

function openSettings() {
  syncConnectionUi();
  inputVoice.value = settings.voice;
  inputPlaybackBuffer.value = String(settings.playbackBufferMs);
  inputLocalVad.value = settings.localVad ? "on" : "off";
  inputInstructions.value = settings.instructions;
  syncGateUi();
  updateRestartAvailability();
  void refreshAudioDeviceLists();
  // The stack can be switched from another tab (or by a restore), so re-read
  // the supervisor every time the dialog opens.
  void ttsEngine.refresh();
  settingsModal.showModal();
}

/** dB position (clamped to the slider axis) as a 0..1 fraction of the track.
 * @param {number} db */
function dbToFraction(db) {
  const clamped = Math.min(GATE_MAX_DB, Math.max(GATE_OFF_DB, db));
  return (clamped - GATE_OFF_DB) / (GATE_MAX_DB - GATE_OFF_DB);
}

/** @param {number} f @returns {number} dB at a 0..1 position on the gate axis. */
function fractionToDb(f) {
  const clamped = Math.min(1, Math.max(0, f));
  return Math.round(GATE_OFF_DB + clamped * (GATE_MAX_DB - GATE_OFF_DB));
}

// ── Radial gate arc (around the mic button, live during a call) ─────────────
// A 270° arc with the gap facing the orb (right). Fraction 0 (=Off) sits at the
// bottom-ish start; 1 (=max) at the top-ish end. The level fill and the
// threshold handle ride this same axis, mirroring the Settings widget.
const ARC_R = 40;
// A ~200° arc centred on the left (180°) so the wide gap faces the orb (right).
const ARC_SPAN_DEG = 200;
const ARC_START_DEG = 180 - ARC_SPAN_DEG / 2; // lower-left start; Off end

/** Point at fraction f (0..1) and radius r, in the 0..100 viewBox.
 * @param {number} f @param {number} [r] */
function arcPoint(f, r = ARC_R) {
  const deg = ARC_START_DEG + f * ARC_SPAN_DEG;
  const rad = (deg * Math.PI) / 180;
  return { x: 50 + r * Math.cos(rad), y: 50 + r * Math.sin(rad) };
}

/** SVG path `d` for the full 0..1 arc (clockwise). */
function fullArcD() {
  const a = arcPoint(0);
  const b = arcPoint(1);
  const largeArc = ARC_SPAN_DEG > 180 ? 1 : 0;
  return `M ${a.x} ${a.y} A ${ARC_R} ${ARC_R} 0 ${largeArc} 1 ${b.x} ${b.y}`;
}

/** One-time geometry: track, fill (dash-revealed) and the transparent hit band. */
function initGateArc() {
  const d = fullArcD();
  mgaTrack.setAttribute("d", d);
  mgaFill.setAttribute("d", d);
  mgaHit.setAttribute("d", d);
  // pathLength 100 lets us reveal the fill by fraction via dashoffset.
  mgaFill.setAttribute("pathLength", "100");
  mgaFill.style.strokeDasharray = "100 100";
  mgaFill.style.strokeDashoffset = "100"; // empty until levels arrive
  renderGateHandle();
}

/** Place the threshold bead on the arc at the stored threshold; flag off state. */
function renderGateHandle() {
  const off = settings.noiseGate <= GATE_OFF_DB;
  const p = arcPoint(dbToFraction(settings.noiseGate));
  mgaHandle.setAttribute("cx", String(p.x));
  mgaHandle.setAttribute("cy", String(p.y));
  micGate.classList.toggle("gate-off", off);
}

/** Paint a 0..1 live level onto the arc fill (and the Settings meter if open).
 * Brightens the tick when the level crosses the threshold — i.e. the gate is
 * actually open — but only when gating is enabled.
 * @param {number} rms */
function paintInputLevel(rms) {
  const db = rms > 0 ? 20 * Math.log10(rms) : GATE_OFF_DB;
  const f = dbToFraction(db);
  mgaFill.style.strokeDashoffset = String(100 * (1 - f));
  if (settingsModal.open) gateMeterFill.style.width = `${f * 100}%`;
  const enabled = settings.noiseGate > GATE_OFF_DB;
  micGate.classList.toggle("gate-open", enabled && f >= dbToFraction(settings.noiseGate));
}

/** The single place that commits a new gate threshold: updates both controls,
 * persists, and applies live to the running session.
 * @param {number} db */
function setGateThreshold(db) {
  settings.noiseGate = Math.min(GATE_MAX_DB, Math.max(GATE_OFF_DB, Math.round(db)));
  const off = settings.noiseGate <= GATE_OFF_DB;
  inputNoiseGate.value = String(settings.noiseGate);
  gateValue.textContent = off ? "Off" : `${settings.noiseGate} dB`;
  renderGateHandle();
  localStorage.setItem(STORAGE_KEYS.noiseGate, String(settings.noiseGate));
  if (client && LIVE_STATES.has(currentState)) {
    client.setNoiseGate(gateParams(settings.noiseGate));
  }
}

/** Reflect the stored gate threshold into the slider, label and arc handle. */
function syncGateUi() {
  inputNoiseGate.value = String(settings.noiseGate);
  const off = settings.noiseGate <= GATE_OFF_DB;
  gateValue.textContent = off ? "Off" : `${settings.noiseGate} dB`;
  renderGateHandle();
}

// Drag along the arc band to set the threshold (a tap on the glyph still mutes).
let gateDragging = false;
/** @param {PointerEvent} e */
function gatePointerToDb(e) {
  const rect = mgaArc.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  let deg = (Math.atan2(e.clientY - cy, e.clientX - cx) * 180) / Math.PI;
  if (deg < 0) deg += 360;
  // Map the on-arc angle to a fraction; angles in the right-side gap fall
  // outside [0,1] and fractionToDb clamps them to the nearest end (just-below
  // start -> Off, just-past end -> max).
  const f = (deg - ARC_START_DEG) / ARC_SPAN_DEG;
  return fractionToDb(f);
}
mgaHit.addEventListener("pointerdown", (e) => {
  gateDragging = true;
  mgaHit.setPointerCapture(e.pointerId);
  setGateThreshold(gatePointerToDb(e));
});
mgaHit.addEventListener("pointermove", (e) => {
  if (gateDragging) setGateThreshold(gatePointerToDb(e));
});
const endGateDrag = (/** @type {PointerEvent} */ e) => {
  if (!gateDragging) return;
  gateDragging = false;
  try { mgaHit.releasePointerCapture(e.pointerId); } catch {}
};
mgaHit.addEventListener("pointerup", endGateDrag);
mgaHit.addEventListener("pointercancel", endGateDrag);

settingsBtn.addEventListener("click", openSettings);

// About panel: native <dialog>, Esc closes for free; also close on the X and
// on a click in the backdrop (a click whose target is the dialog itself).
aboutBtn.addEventListener("click", () => aboutModal.showModal());
// Mobile twin of the (i), living in the right-hand control cluster.
$("#about-btn-m").addEventListener("click", () => aboutModal.showModal());
aboutClose.addEventListener("click", () => aboutModal.close());
aboutModal.addEventListener("click", (e) => {
  if (e.target === aboutModal) aboutModal.close();
});

// ── Tools panel ───────────────────────────────────────────────────────────

/** Reflect the current tool state into the panel controls. */
function syncToolsUi() {
  const avail = searchAvailable();
  toolWebSwitch.checked = toolsEnabled.web_search && avail;
  toolWebSwitch.disabled = !avail;
  toolWebRow.classList.toggle("disabled", !avail);
  toolCamSwitch.checked = toolsEnabled.camera_snapshot;
  toolWeatherSwitch.checked = toolsEnabled.get_weather;

  if (serverSearchKey) {
    // Key lives server-side: show it as configured, never expose it.
    searchKeyInput.value = "";
    searchKeyInput.placeholder = "••••••••  · 由服务器提供";
    searchKeyInput.disabled = true;
    toolWebHint.textContent = "已就绪。搜索由服务器端完成(未配置密钥时使用 Bing 免费搜索)。";
  } else {
    searchKeyInput.disabled = false;
    searchKeyInput.value = userSearchKey;
    searchKeyInput.placeholder = "粘贴一个 Serper 密钥以启用网络搜索";
    toolWebHint.textContent = userSearchKey
      ? "正在使用你的密钥——仅保存在本浏览器中。"
      : "服务器未配置密钥。添加你自己的 Serper 密钥以启用网络搜索。";
  }
}

toolsBtn.addEventListener("click", () => { syncToolsUi(); toolsModal.showModal(); });
toolsClose.addEventListener("click", () => toolsModal.close());
toolsModal.addEventListener("click", (e) => {
  if (e.target === toolsModal) toolsModal.close();
});

toolWebSwitch.addEventListener("change", () => {
  if (toolWebSwitch.checked && !searchAvailable()) {
    toolWebSwitch.checked = false; // guard: can't enable without a key
    return;
  }
  toolsEnabled.web_search = toolWebSwitch.checked;
  saveTools();
  pushToolsToSession();
});

toolWeatherSwitch.addEventListener("change", () => {
  toolsEnabled.get_weather = toolWeatherSwitch.checked;
  saveTools();
  pushToolsToSession();
});

toolCamSwitch.addEventListener("change", async () => {
  if (toolCamSwitch.checked) {
    try {
      // Flipping the switch always re-requests the camera, so a permission that
      // was only dismissed earlier is asked again here.
      await enableCamera();
    } catch (err) {
      toolCamSwitch.checked = false;
      const denied = err instanceof Error && (err.name === "NotAllowedError" || err.name === "SecurityError");
      toolCamHint.textContent = denied
        ? "摄像头被阻止了。请在浏览器地址栏的摄像头图标处允许访问,它会自动打开。"
        : `摄像头不可用${err instanceof Error ? `:${err.message}` : ""}`;
      return;
    }
    toolsEnabled.camera_snapshot = true;
    toolCamHint.textContent = "摄像头已开启。助手需要时可以拍一张照片看看。";
  } else {
    disableCamera();
    toolsEnabled.camera_snapshot = false;
    toolCamHint.textContent = "让助手通过你的摄像头“看到”画面。";
  }
  saveTools();
  pushToolsToSession();
});

searchKeyInput.addEventListener("input", () => {
  if (serverSearchKey) return;
  userSearchKey = searchKeyInput.value.trim();
  if (userSearchKey) localStorage.setItem(STORAGE_KEYS.searchKey, userSearchKey);
  else localStorage.removeItem(STORAGE_KEYS.searchKey);

  const avail = searchAvailable();
  toolWebSwitch.disabled = !avail;
  toolWebRow.classList.toggle("disabled", !avail);
  // Losing the key disables a previously-enabled tool.
  if (!avail && toolsEnabled.web_search) {
    toolsEnabled.web_search = false;
    toolWebSwitch.checked = false;
    saveTools();
    pushToolsToSession();
  }
  toolWebHint.textContent = userSearchKey
    ? "正在使用你的密钥——仅保存在本浏览器中。"
    : "服务器未配置密钥。添加你自己的 Serper 密钥以启用网络搜索。";
});

// ── Camera ──────────────────────────────────────────────────────────────────

async function enableCamera() {
  if (cameraStream) return;
  assertCaptureContext();
  cameraStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false,
  });
  camVideo.srcObject = cameraStream;
  try { await camVideo.play(); } catch { /* autoplay quirks; muted video is fine */ }
  camPip.classList.add("visible");
  camPip.setAttribute("aria-hidden", "false");
  // Lets the footer reflow to the bottom-right (and hide on mobile) while the
  // webcam preview occupies the bottom of the stage.
  document.body.classList.add("cam-on");
}

function disableCamera() {
  if (cameraStream) {
    for (const t of cameraStream.getTracks()) t.stop();
    cameraStream = null;
  }
  camVideo.srcObject = null;
  camPip.classList.remove("visible");
  camPip.setAttribute("aria-hidden", "true");
  document.body.classList.remove("cam-on");
}

/** Starts the webcam only when the camera tool is explicitly enabled (it
 *  defaults off — see loadTools). If the user declines the permission,
 *  switch the tool off and reflect it in the UI rather than nagging. */
async function autoStartCamera() {
  if (!toolsEnabled.camera_snapshot || cameraStream) return;
  try {
    await enableCamera();
  } catch (err) {
    console.warn("[main] camera auto-start declined/failed:", err);
    toolsEnabled.camera_snapshot = false;
    saveTools();
    syncToolsUi();
  }
}

/** Track the browser's camera permission so a later re-grant (e.g. the user
 *  unblocks it from the address bar after a denial) turns the camera back on
 *  without another toggle, and a revoke turns it off. Best-effort: the
 *  Permissions API doesn't support "camera" everywhere (e.g. Safari). */
async function watchCameraPermission() {
  try {
    const status = await navigator.permissions?.query?.({ name: /** @type {any} */ ("camera") });
    if (!status) return;
    status.addEventListener("change", () => {
      if (status.state === "granted") {
        if (!toolsEnabled.camera_snapshot) { toolsEnabled.camera_snapshot = true; saveTools(); }
        void autoStartCamera();
        syncToolsUi();
      } else if (status.state === "denied") {
        disableCamera();
        if (toolsEnabled.camera_snapshot) { toolsEnabled.camera_snapshot = false; saveTools(); }
        syncToolsUi();
      }
    });
  } catch {
    // Permissions API unavailable for "camera" — the toggle still re-asks.
  }
}

/**
 * Grab the current webcam frame as a downscaled JPEG data URL. The preview is
 * mirrored in CSS for a natural self-view, but we draw the raw (un-mirrored)
 * video here so the model sees the scene in its true orientation.
 *
 * Over WebRTC the frame must fit one data-channel message, so it's re-encoded
 * down the SNAPSHOT_LADDER until it's under SNAPSHOT_DC_BUDGET_CHARS; over
 * WebSocket the first (full-quality) rung is used as before.
 * @returns {string | null}
 */
function captureSnapshot() {
  if (!cameraStream || !camVideo.videoWidth) return null;
  const vw = camVideo.videoWidth;
  const vh = camVideo.videoHeight;

  /** @param {number} maxEdge @param {number} quality @returns {string | null} */
  const encode = (maxEdge, quality) => {
    const scale = Math.min(1, maxEdge / Math.max(vw, vh));
    const w = Math.max(1, Math.round(vw * scale));
    const h = Math.max(1, Math.round(vh * scale));
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(camVideo, 0, 0, w, h);
    return canvas.toDataURL("image/jpeg", quality);
  };

  if (activeTransport !== "webrtc") {
    return encode(SNAPSHOT_MAX_EDGE, SNAPSHOT_QUALITY);
  }
  let dataUrl = null;
  for (const [edge, quality] of SNAPSHOT_LADDER) {
    dataUrl = encode(edge, quality);
    if (!dataUrl) return null;
    if (dataUrl.length <= SNAPSHOT_DC_BUDGET_CHARS) return dataUrl;
  }
  // Even the last rung overflowed (shouldn't happen in practice) — send it
  // anyway; the client logs the failed send rather than killing the session.
  console.warn(`[tool] snapshot exceeds the data-channel budget after the full ladder (${dataUrl?.length} chars)`);
  return dataUrl;
}

/** Brief shutter flash on the preview so the user sees a snapshot was taken. */
function flashPreview() {
  camPip.classList.remove("flash");
  void camPip.offsetWidth; // reflow so the animation restarts
  camPip.classList.add("flash");
}

// ── Tool executor ─────────────────────────────────────────────────────────
// Runs the function the model called and returns the result. RealtimeSession
// owns function output ordering and the follow-up response. Errors come back as
// tool output too, so the model can recover gracefully instead of stalling.

/**
 * Run the function the model called. The Agents SDK preserves call order and
 * submits the returned value to the session.
 * @param {string} name @param {string} argsJson @param {string} callId
 * @returns {Promise<{ output: string, image?: string }>}
 */
async function runTool(name, argsJson, callId) {
  if (!client) return { output: "" };
  let args = /** @type {Record<string, unknown>} */ ({});
  try { args = JSON.parse(argsJson || "{}"); } catch { /* keep {} */ }

  if (DEBUG) console.debug(`[tool] run name=${name} callId=${JSON.stringify(callId)} args=${argsJson}`);
  if (!callId) console.warn("[tool] empty call_id — the backend didn't tag the call, can't return a function_call_output");

  /** @type {{ output: string, image?: string }} */
  let result = { output: "" };
  try {
    if (name === "web_search") {
      const query = typeof args.query === "string" ? args.query : "";
      result.output = await execWebSearch(query);
    } else if (name === "get_weather") {
      result.output = await execWeather(typeof args.city === "string" ? args.city : "", Number(args.days) || 3);
    } else if (name === "camera_snapshot") {
      const dataUrl = captureSnapshot();
      if (dataUrl) {
        if (DEBUG) console.debug(`[tool] camera_snapshot captured frame (${dataUrl.length} chars), sending image + output`);
        result = { output: "已从摄像头拍摄照片并作为图片附上。", image: dataUrl };
        flashPreview();
      } else {
        console.warn("[tool] camera_snapshot: no frame — camera off or not ready");
        result.output = "摄像头当前不可用。";
      }
    } else {
      result.output = `未知工具:${name}`;
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    result.output = `工具执行失败:${msg}`;
  }
  return result;
}

/** @param {string} city @param {number} days @returns {Promise<string>} */
async function execWeather(city, days) {
  if (!city) return "未提供城市。";
  const res = await fetch(`api/weather?city=${encodeURIComponent(city)}&days=${days}`);
  if (res.status === 404) return `没有找到城市"${city}"。`;
  if (!res.ok) throw new Error(`天气查询出错(${res.status})`);
  const w = await res.json();
  const c = w.current;
  const lines = [`实时天气(Open-Meteo),${w.place},当地时间 ${w.time}:`,
    `现在:${c.weather},${c.temp}°C(体感 ${c.feels_like}°C),湿度 ${c.humidity}%,风速 ${c.wind_kmh} km/h。`];
  for (const d of w.forecast) lines.push(`${d.date}:${d.weather},${d.min}~${d.max}°C,降水概率 ${d.rain_prob}%`);
  return lines.join("\n");
}

/** @param {string} query @returns {Promise<string>} */
async function execWebSearch(query) {
  if (!query) return "未提供查询内容。";
  /** @type {Record<string, string>} */
  const body = { query };
  // Only send a user key when there's no server key (server prefers its own).
  if (!serverSearchKey && userSearchKey) body.key = userSearchKey;
  // On the Space, the server key only serves searches from a live session.
  if (trackedSessionId) body.sessionId = trackedSessionId;

  const res = await fetch("api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = String(res.status);
    try { const j = await res.json(); if (j.detail) detail = j.detail; } catch {}
    throw new Error(`搜索出错(${detail})`);
  }
  const json = await res.json();
  // Date-stamp the header so the model treats these as fresh realtime facts
  // rather than its (older) training knowledge.
  const today = new Date().toISOString().slice(0, 10);
  /** @type {string[]} */
  const lines = [`网络搜索结果,日期 ${today}:`];
  if (json.answer) lines.push(`答案:${json.answer}`);
  for (const r of json.results || []) {
    lines.push(`- ${r.title}:${r.snippet}(${r.url})`);
  }
  return lines.length > 1 ? lines.join("\n") : `${lines[0]}\n未找到结果。`;
}

/** Learn server config (search key + connection target), then refresh the UI. */
async function fetchConfig() {
  try {
    const res = await fetch("api/config");
    if (res.ok) {
      const json = await res.json();
      serverSearchKey = !!json.search;
      lbMode = !!json.lb;
      // Lock to LB mode only when the deploy reports a load balancer.
      allowDirect = json.allowDirect ?? !lbMode;
      // Deploy-pinned direct URL (overrides the LB server-side already).
      pinnedUrl = (json.s2sUrl || "").trim();
      // WebRTC transport: offered only when the deploy pins the URL (the
      // /api/calls proxy refuses to forward anywhere else).
      rtcAvailable = !!json.rtc;
      iceServers = Array.isArray(json.iceServers) ? json.iceServers : [];
      startupGreeting = typeof json.startupGreeting === "string"
        ? json.startupGreeting.trim()
        : "";
      // The conversation-time limiter rides on the LB being present.
      limiterOn = lbMode;
    }
    // Non-OK response: leave the fail-open default (allowDirect = true).
  } catch {
    // Config endpoint unreachable (e.g. static hosting): keep direct entry.
  }
  if (DEBUG) console.debug(`[ui] config: allowDirect=${allowDirect} lbMode=${lbMode}`);
  // Login chip + remaining-budget (no-op / hidden when the limiter is off).
  void account.refresh();
  syncToolsUi();
  syncConnectionUi();
}

/**
 * Resolve where to connect, per the deploy's mode:
 *   • LB mode  -> `{ sessionUrl }`, the client POSTs the same-origin /api/session
 *     proxy and the server forwards to the LB (its address stays server-side).
 *   • direct   -> `{ directUrl }`, connect straight to the s2s WebSocket.
 * Throws a user-facing error if direct mode is on but no URL was entered.
 * @returns {{ sessionUrl: string } | { directUrl: string }}
 */
function connectionTarget() {
  if (!allowDirect) {
    return { sessionUrl: "api/session" };
  }
  const directUrl = buildDirectWsUrl(pinnedUrl || settings.directUrl);
  if (!directUrl) {
    throw new Error("请在设置中填写 speech-to-speech 服务器地址。");
  }
  return { directUrl };
}

/**
 * Normalise a user-typed server address into a realtime WebSocket URL.
 * Accepts bare hosts (`localhost:8080`), http(s) URLs, or ws(s) URLs, and adds
 * the `/v1/realtime` path when none is given. A full connect URL (with path
 * and/or query) is preserved as-is.
 * @param {string} raw @returns {string}
 */
function buildDirectWsUrl(raw) {
  let s = (raw || "").trim();
  if (!s) return "";
  if (!/^wss?:\/\//i.test(s)) {
    if (/^https?:\/\//i.test(s)) {
      s = s.replace(/^http/i, "ws"); // http→ws, https→wss
    } else {
      const isLocal = /^(localhost|127\.0\.0\.1|\[::1\])(:|\/|$)/i.test(s);
      s = (isLocal ? "ws://" : "wss://") + s;
    }
  }
  try {
    const u = new URL(s);
    if (u.pathname === "" || u.pathname === "/") u.pathname = "/v1/realtime";
    // An https page may only open wss:// (mixed content), and this deploy puts
    // Caddy in front of the s2s socket on the same host, so a pinned ws:// URL
    // for this machine is re-homed onto the page's own origin. The .crt for the
    // LAN IP comes from Caddy's local CA; without trust in it only http works.
    if (
      location.protocol === "https:" &&
      u.protocol === "ws:" &&
      u.hostname === location.hostname
    ) {
      u.protocol = "wss:";
      u.host = location.host;
    }
    return u.toString();
  } catch {
    return s;
  }
}

/** Create + resume an AudioContext synchronously (must run inside the user
 *  gesture so iOS lets it start). Returns null if construction fails. */
function createResumedAudioContext() {
  try {
    const Ctx = window.AudioContext || /** @type {any} */ (window).webkitAudioContext;
    const ctx = new Ctx({ latencyHint: "interactive" });
    if (ctx.state === "suspended") void ctx.resume().catch(() => {});
    return /** @type {AudioContext} */ (ctx);
  } catch (err) {
    console.warn("[main] AudioContext init failed:", err);
    return null;
  }
}

/** Read the editable settings out of the form. The URL field is only honoured
 *  in free direct mode — in LB mode it's hidden, and when the deploy pins a
 *  URL it's read-only, so the user's saved URL survives either way. The
 *  transport select is only honoured while selectable, so a saved "webrtc"
 *  survives a visit to a deploy that can't offer it. */
function readSettingsFromForm() {
  return {
    directUrl: allowDirect && !pinnedUrl ? inputLbUrl.value.trim() : settings.directUrl,
    voice: inputVoice.value || DEFAULT_VOICE,
    instructions: inputInstructions.value.trim() || DEFAULT_INSTRUCTIONS,
    noiseGate: readGateThreshold(),
    playbackBufferMs: normalizePlaybackBufferMs(inputPlaybackBuffer.value),
    localVad: inputLocalVad.value !== "off",
    transport: /** @type {"ws" | "webrtc"} */ (
      transportSelectable()
        ? (inputTransport.value === "webrtc" ? "webrtc" : "ws")
        : settings.transport
    ),
    audioInputId: inputAudioInput.value || "",
    audioOutputId: inputAudioOutput.value || "",
  };
}

/** Gate threshold (dBFS) currently shown on the slider, clamped to range. */
function readGateThreshold() {
  const v = Math.round(Number(inputNoiseGate.value));
  if (!Number.isFinite(v)) return GATE_OFF_DB;
  return Math.min(GATE_MAX_DB, Math.max(GATE_OFF_DB, v));
}

/** Whether the transport picker is live: env-pinned direct mode only. The
 *  /api/calls proxy forwards exclusively to SPEECH_TO_SPEECH_URL, so without
 *  the pin there is nowhere safe to send a WebRTC offer. */
function transportSelectable() {
  return allowDirect && !!pinnedUrl && rtcAvailable;
}

/** The transport the next conversation will actually use. */
function effectiveTransport() {
  return transportSelectable() && settings.transport === "webrtc" ? "webrtc" : "ws";
}

/** Reflect transport availability + selection into Settings, and hide the
 *  noise gate when WebRTC is picked (the gate lives in the WS capture
 *  worklet; the WebRTC mic path sends the raw track). */
function syncTransportUi() {
  // Hidden in LB mode (nothing to choose); visible-but-locked in un-pinned
  // direct mode so the option is discoverable along with what unlocks it.
  transportField.hidden = !allowDirect;
  const selectable = transportSelectable();
  inputTransport.disabled = !selectable;
  inputTransport.value = selectable && settings.transport === "webrtc" ? "webrtc" : "ws";
  transportHint.textContent = selectable
    ? "音频传输到服务器的方式。下一次对话生效。"
    : "WebRTC 需要部署方固定一个服务器地址(SPEECH_TO_SPEECH_URL)。";
  gateField.hidden = effectiveTransport() === "webrtc";
  playbackBufferField.hidden = effectiveTransport() === "webrtc";
}

/** Adapt the connection field to the mode learned from /api/config. */
function syncConnectionUi() {
  syncTransportUi();
  if (pinnedUrl) {
    // Deploy-pinned URL: show it, but locked — the deployment owns it.
    connField.hidden = false;
    inputLbUrl.value = pinnedUrl;
    inputLbUrl.readOnly = true;
    connHint.classList.remove("error");
    connHint.textContent = "本次部署已固定 speech-to-speech 服务器地址。";
  } else if (allowDirect) {
    // Direct mode: the user sets their own s2s server URL.
    connField.hidden = false;
    inputLbUrl.value = settings.directUrl;
    inputLbUrl.readOnly = false;
    inputLbUrl.placeholder = "http://localhost:端口号";
    connHint.classList.remove("error");
    connHint.textContent =
      "你的 speech-to-speech 服务器地址,例如 http://localhost:8080(应用会自动加上 /v1/realtime)。";
  } else {
    // LB mode: the load balancer URL is deployment-owned — hide it entirely so
    // its address is never exposed in Settings.
    connField.hidden = true;
  }
}

/** True when the user must supply a server URL before connecting (direct mode
 *  with nothing set). */
function missingServerUrl() {
  return allowDirect && !pinnedUrl && !buildDirectWsUrl(settings.directUrl);
}

/** Open Settings and point the user at the empty server-URL field. */
function promptServerUrl() {
  if (settingsModal.open) syncConnectionUi();
  else openSettings();
  connHint.textContent = "请先设置 speech-to-speech 服务器地址才能开始。";
  connHint.classList.add("error");
  inputLbUrl.focus();
}

settingsForm.addEventListener("submit", (event) => {
  const submitter = /** @type {HTMLButtonElement | null} */ ((/** @type {SubmitEvent} */ (event)).submitter);
  if (submitter?.value !== "save") return;

  settings = readSettingsFromForm();
  saveSettings(settings);

  // Voice + instructions can apply to a live session without reconnecting; a
  // changed connection URL only takes effect on the next restart. Speaker
  // output can switch live when the browser supports AudioContext.setSinkId;
  // mic device changes need a Restart (new getUserMedia stream).
  if (client && LIVE_STATES.has(currentState)) {
    client.updateSession({ voice: settings.voice, instructions: settings.instructions });
    if (typeof client.setAudioOutputDevice === "function") {
      void client.setAudioOutputDevice(settings.audioOutputId);
    }
  }
});

// The noise gate applies live (worklet param), so tune it without a restart:
// update the label/marker, persist, and push straight to the running client.
inputNoiseGate.addEventListener("input", () => {
  setGateThreshold(readGateThreshold());
});

// Transport persists on change (like the gate) and takes effect on the next
// conversation; the gate field previews its WS-only availability right away.
inputTransport.addEventListener("change", () => {
  if (!transportSelectable()) return;
  settings.transport = inputTransport.value === "webrtc" ? "webrtc" : "ws";
  localStorage.setItem(STORAGE_KEYS.transport, settings.transport);
  syncTransportUi();
});

restartBtn.addEventListener("click", async () => {
  if (currentState === "connecting") return; // a connect is already underway
  settings = readSettingsFromForm();
  saveSettings(settings);
  if (missingServerUrl()) { promptServerUrl(); return; } // keep settings open
  settingsModal.close();
  // Grab the AudioContext NOW, inside the click gesture — teardown() awaits, and
  // creating it afterwards would fall outside the gesture (silent on iOS).
  const audioContext = createResumedAudioContext();
  try {
    if (client) await teardown();
    await doStart(audioContext);
  } catch (err) {
    await handleStartError(err);
  }
});

circleBtn.addEventListener("click", async () => {
  try {
    if (currentState === "idle" || currentState === "error") {
      if (missingServerUrl()) { promptServerUrl(); return; }
      await doStart();
    }
  } catch (err) {
    await handleStartError(err);
  }
});

/** A failed start is either the daily limit (show the modal, return to idle) or
 *  a real fault (surface it). doStart already closed any orphan AudioContext.
 *  @param {any} err */
async function handleStartError(err) {
  if (err && err.code === "login-required") {
    await teardown();
    setState("error");
    setCaption("请重新登录以继续。", "error");
    account.showLoginRequired(err.loginUrl);
    return;
  }
  if (err && err.code === "limit") {
    await teardown();
    account.showLimit(err.tier);
    return;
  }
  // The user left the queue (close() aborted the wait): teardown already reset
  // the UI to idle, so there's nothing to report.
  if (err && err.code === "aborted") return;
  // The whole waiting line is full: a warm, reassuring modal rather than an error.
  if (err && err.code === "queue-full") {
    await teardown();
    account.showBusy();
    return;
  }
  // Our place lapsed (ticket reaped, or the join window ran out). Recoverable, not
  // a fault: land on the retry state with a kind, plain-language reason.
  if (err && (err.code === "queue-expired" || err.code === "join-expired")) {
    await teardown();
    setState("error");
    setCaption(
      err.code === "join-expired"
        ? "你的名额已过期,点击重新加入。"
        : "等待时间太久了,点击重新加入。",
      "error",
    );
    return;
  }
  await onFatalError(err);
}

micBtn.addEventListener("click", () => {
  if (!micStream || !client) return;
  micMuted = !micMuted;
  syncMicMuteState();
  micBtn.classList.toggle("muted", micMuted);
  micBtn.setAttribute("aria-label", micMuted ? "取消静音" : "静音");
  micBtn.title = micMuted ? "取消静音" : "静音";
});

stopBtn.addEventListener("click", async () => {
  await teardown();
});

// "Leave queue": tear down the pending connect (aborts the poll wait) and drop
// our place in line. Same teardown path as stopping a live call.
leaveQueueBtn.addEventListener("click", async () => {
  await teardown();
});

// "Join now": accept the held slot. The click is a user gesture, so the client
// re-resumes the AudioContext here (iOS) before dialing.
joinQueueBtn.addEventListener("click", () => {
  stopJoinCountdown();
  if (client) client.join();
});

const MIC_CONSTRAINTS_BASE = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
};

/** @returns {MediaStreamConstraints} */
function micConstraints() {
  /** @type {MediaTrackConstraints} */
  const audio = { ...MIC_CONSTRAINTS_BASE };
  if (settings.audioInputId) {
    // ideal (not exact): if the saved device was unplugged, fall back quietly.
    audio.deviceId = { ideal: settings.audioInputId };
  }
  return { audio };
}

/** True when Web Audio can route playback to a chosen output device. */
function supportsAudioOutputSelection() {
  const Ctx = window.AudioContext || /** @type {any} */ (window).webkitAudioContext;
  return typeof Ctx?.prototype?.setSinkId === "function";
}

/**
 * Rebuild the mic/speaker <select>s from enumerateDevices. Labels are blank
 * until mic permission has been granted at least once.
 */
async function refreshAudioDeviceLists() {
  const canPickOutput = supportsAudioOutputSelection();
  inputAudioOutput.disabled = !canPickOutput;
  audioOutputHint.textContent = canPickOutput
    ? "助手语音的播放设备。连接中也可以实时切换。"
    : "选择扬声器需要支持 AudioContext.setSinkId 的浏览器(Chrome/Edge)。";

  /** @type {MediaDeviceInfo[]} */
  let devices = [];
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch (err) {
    console.warn("[main] enumerateDevices failed:", err);
  }

  const inputs = devices.filter((d) => d.kind === "audioinput");
  const outputs = devices.filter((d) => d.kind === "audiooutput");
  const labelsReady = devices.some((d) => d.label);

  fillDeviceSelect(inputAudioInput, inputs, settings.audioInputId, "麦克风");
  fillDeviceSelect(inputAudioOutput, outputs, settings.audioOutputId, "扬声器");

  if (!labelsReady) {
    // Permission unlocks real device names; keep it quiet — user can tap Start
    // or we unlock when they already connected once this session.
    const hint = inputAudioInput.parentElement?.querySelector("small");
    if (hint) {
      hint.textContent =
        "允许麦克风访问(点击一次“开始”)后可查看设备名称。麦克风更改在“重新连接”后生效。";
    }
  } else {
    const hint = inputAudioInput.parentElement?.querySelector("small");
    if (hint) hint.textContent = "下一次对话(或点击“重新连接”)后生效。";
  }
}

/**
 * @param {HTMLSelectElement} select
 * @param {MediaDeviceInfo[]} devices
 * @param {string} selectedId
 * @param {string} fallbackLabel
 */
function fillDeviceSelect(select, devices, selectedId, fallbackLabel) {
  const prev = selectedId || select.value || "";
  select.replaceChildren();
  const def = document.createElement("option");
  def.value = "";
  def.textContent = "系统默认";
  select.appendChild(def);
  devices.forEach((d, i) => {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    opt.textContent = d.label || `${fallbackLabel} ${i + 1}`;
    select.appendChild(opt);
  });
  // Keep a saved id even if it isn't currently listed (unplugged); browser
  // will fall back via ideal constraints / setSinkId errors.
  if (prev && ![...select.options].some((o) => o.value === prev)) {
    const missing = document.createElement("option");
    missing.value = prev;
    missing.textContent = `${fallbackLabel}(已保存,未找到)`;
    select.appendChild(missing);
  }
  select.value = prev;
  if (select.value !== prev) select.value = "";
}

if (navigator.mediaDevices?.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => {
    if (settingsModal.open) void refreshAudioDeviceLists();
  });
}

/** getUserMedia exists only in a secure context: https://, or http://localhost.
 *  A page opened at http://<LAN IP> has no navigator.mediaDevices at all, so the
 *  browser's own TypeError ("undefined is not an object") is replaced with
 *  something the user can act on. */
function assertCaptureContext() {
  if (navigator.mediaDevices?.getUserMedia) return;
  throw new Error(
    "当前地址不是安全上下文（浏览器只在 https 或 localhost 下开放麦克风）。" +
      `请改用 https://${location.hostname}:8443/ 打开（首次需信任 Caddy 自签证书），` +
      `或在本机执行 ssh -L 8080:127.0.0.1:8080 dgpu-nuc 后访问 http://localhost:8080/。`,
  );
}

/** Prompt for mic permission up front, then immediately release the tracks so no
 *  recording indicator lingers during a queue wait. Throws a friendly error if the
 *  user denies. */
async function primeMicPermission() {
  assertCaptureContext();
  try {
    const s = await navigator.mediaDevices.getUserMedia(micConstraints());
    for (const track of s.getTracks()) track.stop();
  } catch (err) {
    throw new Error(
      `麦克风访问被拒绝${err instanceof Error ? `:${err.message}` : ""}`,
    );
  }
}

/** Acquire the live capture stream once a slot is granted. Permission was primed
 *  in the tap gesture, so this is silent. Stored module-side for mute + teardown. */
async function acquireMicStream() {
  assertCaptureContext();
  micStream = await navigator.mediaDevices.getUserMedia(micConstraints());
  return micStream;
}

/** @param {number} position Update the queued caption ("You're #N in line"). */
function onQueuePosition(position) {
  const n = Number(position) || 0;
  setCaption(n > 0 ? `排在第 ${n} 位` : "正在为你寻找空位…", "muted");
}

// ── "Your turn" join countdown ──────────────────────────────────────────────
// While a slot is held for us, show how long is left to accept it. The client's
// join gate expires just before the load balancer reclaims the slot.
let joinCountdownTimer = 0;

/** @param {number} sec */
function startJoinCountdown(sec) {
  stopJoinCountdown();
  let left = Math.max(0, Math.floor(sec));
  const paint = () => {
    joinQueueBtn.textContent = left > 0 ? `立即加入(${left} 秒)` : "立即加入";
  };
  paint();
  joinCountdownTimer = window.setInterval(() => {
    left -= 1;
    if (left <= 0) {
      stopJoinCountdown();
      joinQueueBtn.textContent = "立即加入";
      return;
    }
    paint();
  }, 1000);
}

function stopJoinCountdown() {
  if (joinCountdownTimer) {
    clearInterval(joinCountdownTimer);
    joinCountdownTimer = 0;
  }
}

/**
 * Start a conversation. Pass a pre-created AudioContext when the caller already
 * made one inside the tap/click gesture (required on iOS); otherwise one is
 * created here, which is still inside the gesture for a direct orb tap.
 * @param {AudioContext | null} [audioContext]
 */
async function doStart(audioContext = null) {
  const transport = effectiveTransport();
  // Resolve the target before touching mic/audio so a misconfiguration (e.g.
  // direct mode with no URL) fails fast with a clear message. Over WebRTC the
  // browser never dials the s2s server itself — the offer goes to the
  // same-origin /api/calls proxy — so there is no target to resolve.
  const target = transport === "webrtc" ? null : connectionTarget();
  activeTransport = transport;
  // The radial gate arc (threshold handle around the mic button) is a WS
  // feature; over WebRTC only the mute button remains.
  document.body.classList.toggle("rtc-live", transport === "webrtc");

  chat.clear();
  chat.reset();
  setState("connecting");
  setCaption("正在请求麦克风权限…", "muted");

  // Create + resume the AudioContext SYNCHRONOUSLY, still inside the gesture.
  // iOS Safari only starts an AudioContext from a user gesture; if we waited
  // until after the getUserMedia / session-creation awaits below, it would stay
  // suspended and the whole pipeline would be silent.
  if (!audioContext) audioContext = createResumedAudioContext();

  // Prime the mic permission now (get the prompt out of the way up front), then
  // release it. The real capture stream is acquired only once a slot is granted
  // (see acquireMicStream), so the mic 'in use' indicator never lights while we
  // sit in the queue. Permission persists, so the later acquire is silent.
  try {
    await primeMicPermission();
  } catch (err) {
    if (audioContext) void audioContext.close().catch(() => {});
    throw err;
  }

  // The webcam is started on arrival (autoStartCamera), so nothing to do here;
  // a still-pending grant just means the snapshot tool isn't ready yet.

  const common = {
    localVad: settings.localVad,
    voice: settings.voice,
    instructions: settings.instructions,
    startupGreeting,
    acquireMic: acquireMicStream,
    tools: activeToolDefs(),
    audioOutputId: settings.audioOutputId || "",
    executeTool: async ({ name, arguments: args, callId }) => {
      chat.onToolCall(name);
      const result = await runTool(name, args, callId);
      if (client === c) chat.onToolResult(name, args, result.output, result.image);
      return result;
    },
    ...(audioContext ? { audioContext } : {}),
  };
  const c = target === null
    ? new S2sRealtimeClient({
        transport: "webrtc",
        callsUrl: "api/calls",
        iceServers,
        ...common,
      })
    : new S2sRealtimeClient({
        transport: "websocket",
        ...target,
        noiseGate: gateParams(settings.noiseGate),
        playbackBufferMs: settings.playbackBufferMs,
        ...common,
      });
  client = c;
  c.setMuted(micMuted || userAudioReplaying);
  c.setStyle(stylePin);
  c.setInterruptible(!String(ttsEngine.engine || "").startsWith("cosyvoice"));

  c.addEventListener("queue", (e) => {
    const { position, queueId } = /** @type {CustomEvent<{ position: number; queueId: string }>} */ (e).detail;
    if (queueId) queuedTicketId = queueId;
    onQueuePosition(position);
  });

  c.addEventListener("ready-to-join", (e) => {
    const { info, expiresSec } = /** @type {CustomEvent<{ info: import("./s2s-realtime-client.js").SessionInfo; expiresSec: number }>} */ (e).detail;
    // A slot is held for us. We're out of the queue now, so drop the ticket ref.
    // Track the granted session id already so that leaving (or letting the timer
    // lapse) refunds the budget the server reserved at claim, even before we dial.
    queuedTicketId = "";
    if (info?.sessionId) {
      trackedSessionId = info.sessionId;
      trackedTier = info.tier || "anon";
      // Heartbeat while the user decides to join, or the server reaps the
      // granted session as abandoned.
      if (info.heartbeatSec) startHeartbeat(info.heartbeatSec, !!info.limited);
    }
    startJoinCountdown(expiresSec);
  });

  c.addEventListener("status", (e) => {
    const detail = /** @type {CustomEvent<{ status: string }>} */ (e).detail;
    onClientStatus(detail.status);
    if (detail.status === "ai-speaking") chat.onAssistantActivity();
  });
  c.addEventListener("transcript", (e) => {
    const d = /** @type {CustomEvent<{ role: "user" | "assistant"; text: string; partial: boolean; itemId?: string; responseId?: string }>} */ (e).detail;
    chat.onTranscript(d);
  });
  c.addEventListener("user-turn-started", (e) => {
    const detail = /** @type {CustomEvent<{ itemId?: string }>} */ (e).detail;
    chat.onUserTurnStarted(detail);
  });
  c.addEventListener("user-turn-stopped", (e) => {
    const detail = /** @type {CustomEvent<{ itemId?: string }>} */ (e).detail;
    chat.onUserTurnStopped(detail);
  });
  c.addEventListener("user-audio", (e) => {
    const detail = /** @type {CustomEvent<{ itemId?: string; audio: Blob; durationMs?: number; truncated?: boolean }>} */ (e).detail;
    chat.onUserAudio(detail);
  });

  c.addEventListener("response-finished", (e) => {
    const detail = /** @type {CustomEvent<{ responseId: string; status: string; audible?: boolean; transcript?: string; latency?: import("./turn-latency.js").TurnLatency | null; clientTiming?: import("./client-latency.js").ClientLatencySummary | null }>} */ (e).detail;
    chat.onResponseFinished(detail);
  });
  c.addEventListener("response-audio-finished", (e) => {
    // The reply's buffered audio finished playing — the caption bubble has been
    // read out in full and may now fade.
    const detail = /** @type {CustomEvent<{ responseId: string }>} */ (e).detail;
    chat.onAssistantAudioFinished(detail.responseId);
  });
  c.addEventListener("error", (e) => {
    const detail = /** @type {CustomEvent<{ error: unknown }>} */ (e).detail;
    void onFatalError(detail.error);
  });
  c.addEventListener("server-error", (e) => {
    // Non-fatal: the backend reported an error mid-session. Log it, keep the
    // socket and the conversation alive (the model can recover on its own).
    const detail = /** @type {CustomEvent<{ error: unknown }>} */ (e).detail;
    const msg = errorMessage(detail.error);
    console.warn("[main] server error (non-fatal):", msg);
  });
  c.addEventListener("session", (e) => {
    const info = /** @type {CustomEvent<{ info: import("./s2s-realtime-client.js").SessionInfo }>} */ (e).detail.info;
    console.log("[ws] session created:", info.sessionId);
    // A slot was granted — we're out of the queue; drop the ticket reference so
    // teardown doesn't try to leave a line we already left.
    queuedTicketId = "";
    // Heartbeat every granted session: it shows the server the conversation is
    // live (web search requires that), and for a metered tier (anon / free) it
    // extends the reservation and tells us when the daily budget runs out.
    if (info.sessionId && info.heartbeatSec) {
      trackedSessionId = info.sessionId;
      trackedTier = info.tier || "anon";
      startHeartbeat(info.heartbeatSec, !!info.limited);
    }
  });
  c.addEventListener("input-level", (e) => {
    const { rms } = /** @type {CustomEvent<{ rms: number }>} */ (e).detail;
    paintInputLevel(rms);
  });

  try {
    // After a disconnect (or an engine switch) the single pipeline slot is
    // recycled for a few seconds; dialing in that window is rejected with
    // "All N session slots are in use". Retry briefly instead of surfacing a
    // connection error for something that clears itself.
    for (let attempt = 0; ; attempt++) {
      try {
        await c.connect();
        break;
      } catch (err) {
        const busy = /slot|in use|1008|busy/i.test(String(err && err.message || err));
        if (!busy || attempt >= 8) throw err;
        setCaption("语音管线正在回收，稍候重试…", "muted");
        await new Promise((r) => setTimeout(r, 1500));
      }
    }
  } catch (err) {
    // The grant can be refused (402 → limit) or the dial can fail. In LB mode
    // the AudioContext hasn't been adopted by the client yet (the session POST
    // runs first), so close the one we created here to avoid leaking it.
    if (audioContext) void audioContext.close().catch(() => {});
    throw err;
  }
}

// ── Conversation-time heartbeat ─────────────────────────────────────────────

/** Ping the server every `sec` seconds so it knows the session is live and can
 *  meter it; when it reports a metered session's daily budget is spent, cut the
 *  call and show the limit modal.
 *  @param {number} sec @param {boolean} limited */
function startHeartbeat(sec, limited) {
  stopHeartbeat();
  heartbeatTimer = window.setInterval(async () => {
    if (!trackedSessionId) return;
    try {
      const res = await fetch("api/session/heartbeat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: trackedSessionId }),
        keepalive: true,
      });
      const json = await res.json().catch(() => ({}));
      if (json.expired && limited) await onLimitReached();
    } catch (err) {
      // A transient network blip shouldn't kill the call; the next tick retries.
      if (DEBUG) console.debug("[ui] heartbeat failed:", err);
    }
  }, Math.max(1, sec) * 1000);
}

function stopHeartbeat() {
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
    heartbeatTimer = 0;
  }
}

/** The server cut the live session: tear down and explain why. */
async function onLimitReached() {
  const tier = trackedTier;
  stopHeartbeat();
  await teardown();
  account.showLimit(tier);
}

/** Tell the server a session ended so it reconciles + refunds the unused chunk.
 *  Uses sendBeacon so it still fires when the tab is closing. */
function endTrackedSession() {
  if (!trackedSessionId) return;
  const body = JSON.stringify({ sessionId: trackedSessionId });
  try {
    const blob = new Blob([body], { type: "application/json" });
    if (!navigator.sendBeacon("api/session/end", blob)) {
      void fetch("api/session/end", {
        method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive: true,
      }).catch(() => {});
    }
  } catch {
    // Best-effort; the server sweep reaps the session anyway.
  }
  trackedSessionId = "";
  trackedTier = "";
}

/** Leave the waiting queue so the LB frees our place. sendBeacon so it still
 *  fires on tab close; the LB also reaps the ticket on TTL as a backstop. */
function endQueueTicket() {
  if (!queuedTicketId) return;
  const body = JSON.stringify({ queueId: queuedTicketId });
  try {
    const blob = new Blob([body], { type: "application/json" });
    if (!navigator.sendBeacon("api/queue/end", blob)) {
      void fetch("api/queue/end", {
        method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive: true,
      }).catch(() => {});
    }
  } catch {
    // Best-effort; the LB reaps the ticket on TTL anyway.
  }
  queuedTicketId = "";
}

/** @param {string} status */
function onClientStatus(status) {
  switch (status) {
    case "creating-session":
    case "connecting":
      setState("connecting");
      break;
    case "queued":
      setState("queued");
      break;
    case "your-turn":
      setState("your-turn");
      break;
    case "connected":
      setState("listening");
      break;
    case "user-speaking":
      setState("user-speaking");
      break;
    case "processing":
      setState("processing");
      break;
    case "ai-speaking":
      setState("ai-speaking");
      break;
    case "closed":
      // teardown() will move us to idle
      break;
    case "error":
      setState("error");
      break;
  }
}

async function teardown() {
  stopHeartbeat();
  stopJoinCountdown();
  endTrackedSession();
  endQueueTicket();
  chat.reset({ dismiss: true });
  if (client) {
    try {
      await client.close();
    } catch (err) {
      console.warn("[main] error closing client:", err);
    }
    client = null;
  }
  if (micStream) {
    for (const track of micStream.getTracks()) track.stop();
    micStream = null;
  }
  // The webcam is independent of the call lifecycle (it runs while the user is
  // on the page), so we leave it on here — only the camera toggle stops it.
  micMuted = false;
  micBtn.classList.remove("muted");
  document.body.classList.remove("rtc-live");
  setState("idle");
  // Refresh the chip's remaining-today after the budget moved.
  if (limiterOn) void account.refresh();
}

/** @param {unknown} err */
async function onFatalError(err) {
  console.error("[main] fatal:", err);
  const message = errorMessage(err);
  try {
    await teardown();
  } catch (teardownError) {
    console.warn("[main] error during fatal teardown:", teardownError);
  } finally {
    setState("error");
    setCaption(truncateError(message), "error");
  }
}

setState("idle");
chat.renderEmptyState();
initGateArc();
void fetchConfig();
// Start the webcam as soon as the user lands (camera tool defaults on), and
// react to later permission changes (re-grant after a denial re-enables it).
void autoStartCamera();
void watchCameraPermission();

// Reconcile a live session if the tab is closed/hidden mid-call (no teardown).
window.addEventListener("pagehide", () => { endTrackedSession(); endQueueTicket(); });

requestAnimationFrame(() => {
  document.body.classList.remove("booting");
});
