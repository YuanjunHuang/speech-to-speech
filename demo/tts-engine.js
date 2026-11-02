/**
 * TTS engine picker.
 *
 * The speech-to-speech backend selects its TTS at *startup*, and the two
 * families do not fit in RAM together, so "pick CosyVoice" means "restart the
 * pipeline with different flags". That job belongs to tts_supervisor.py on
 * :8790, proxied by Caddy at /tts-admin/*.
 *
 * This module:
 *   1. renders the engine catalog reported by the supervisor,
 *   2. POSTs the chosen engine and polls until the pipeline is back up,
 *   3. keeps the voice list valid for the active engine (Qwen3 has nine
 *      speakers; CosyVoice has one reference timbre),
 *   4. never auto-starts audio: after a switch the user starts the next
 *      conversation with a real click, so the AudioContext stays inside a
 *      user gesture (a resumed-after-60s context can stay suspended).
 */

const ADMIN_URL = "/tts-admin/tts";
const POLL_MS = 1500;
const SWITCH_TIMEOUT_MS = 6 * 60 * 1000;
// Per-engine voice memory, so switching qwen3 -> cosyvoice2 -> qwen3 comes back
// to the speaker that was chosen before.
const VOICE_KEY_PREFIX = "s2s.ws.voice.";


/**
 * @typedef {{id: string, label: string, kind: string, transport: string,
 *            detail: string, voices: string[]}} EngineInfo
 * @typedef {{active: string, state: string, progress: string, error: string|null,
 *            s2s_running: boolean, engines: EngineInfo[]}} SupervisorState
 */

/**
 * @param {{
 *   select: HTMLSelectElement,
 *   hint: HTMLElement,
 *   status: HTMLElement,
 *   voiceSelect: HTMLSelectElement,
 *   onSwitchStart?: () => void | Promise<void>,
 *   onReady?: (engine: string) => void,
 *   onActive?: (engine: string, state: SupervisorState) => void,
 * }} opts
 */
export function createTtsEnginePicker(opts) {
  const { select, hint, status, voiceSelect, onSwitchStart, onReady, onActive } = opts;
  /** @type {Map<string, EngineInfo>} */
  let catalog = new Map();
  let active = "";
  let busy = false;
  let released = false;
  let pollTimer = 0;
  /** Resolves once the current switch reaches ready/error (used by tests). */
  let settle = null;
  let settled = Promise.resolve();
  const beginSwitch = () => {
    settled = new Promise((resolve) => {
      settle = resolve;
    });
  };

  /** @param {string} text @param {"busy"|"ok"|"error"|"info"} [kind] */
  function setStatus(text, kind = "info") {
    if (!text) {
      status.hidden = true;
      status.textContent = "";
      return;
    }
    status.hidden = false;
    status.textContent = text;
    status.dataset.kind = kind;
  }

  function setBusy(value) {
    busy = value;
    select.disabled = value;
    voiceSelect.disabled = value;
  }

  /** Replace the voice options for `engine`, restoring the remembered pick. */
  function applyVoices(engine) {
    const info = catalog.get(engine);
    const voices = info?.voices?.length ? info.voices : ["Vivian"];
    const remembered = localStorage.getItem(VOICE_KEY_PREFIX + engine);
    const fallback = voices.includes("vivian") ? "vivian" : voices[0];
    const chosen = remembered && voices.includes(remembered) ? remembered : fallback;
    voiceSelect.replaceChildren(
      ...voices.map((v) => {
        const option = document.createElement("option");
        option.value = v;
        option.textContent = engine.startsWith("cosyvoice") ? `${v}（参考音）` : v;
        return option;
      }),
    );
    voiceSelect.value = chosen;
  }

  function renderCatalog(nextActive) {
    const previous = select.value;
    select.replaceChildren(
      ...Array.from(catalog.values()).map((info) => {
        const option = document.createElement("option");
        option.value = info.id;
        option.textContent = `${info.label}（${info.transport}）`;
        return option;
      }),
    );
    const target = catalog.has(nextActive) ? nextActive : previous;
    if (target) select.value = target;
    const info = catalog.get(select.value);
    if (info) hint.textContent = `${info.detail}。切换会重启语音管线，当前对话会中断。`;
    applyVoices(select.value);
  }

  /** @returns {Promise<SupervisorState>} */
  async function fetchState() {
    const res = await fetch(ADMIN_URL, { cache: "no-store", headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return /** @type {SupervisorState} */ (await res.json());
  }

  function adopt(data) {
    catalog = new Map(data.engines.map((info) => [info.id, info]));
    active = data.active;
    renderCatalog(data.active);
  }

  /** Load (or reload) the catalog. Safe to call when Settings opens. */
  async function refresh() {
    if (busy) return;
    try {
      const data = await fetchState();
      released = false;
      select.disabled = false;
      adopt(data);
      if (data.state === "switching") {
        setStatus(data.progress || "正在切换…", "busy");
        setBusy(true);
        schedulePoll(select.value);
      } else if (data.state === "error") {
        setStatus(`切换失败：${data.error || "未知错误"}`, "error");
      } else {
        setStatus("");
        onActive?.(data.active, data);
      }
    } catch (err) {
      released = !isLocalAdminReachable();
      select.disabled = true;
      voiceSelect.disabled = true;
      hint.textContent = released
        ? "引擎切换需要通过 Caddy（:8080 或 :8443）访问本页。"
        : "无法连接引擎管理服务（tts_supervisor.py）。";
      setStatus(released ? "" : `管理服务不可用：${err instanceof Error ? err.message : err}`, "error");
    }
  }

  function isLocalAdminReachable() {
    // The page can be opened straight from the demo server (:7860) or from a
    // file:// copy, where /tts-admin is not proxied. Keep the control disabled
    // but do not shout at the user in that case.
    return !["7860", ""].includes(location.port) && location.protocol.startsWith("http");
  }

  /** @param {string} engine */
  function schedulePoll(engine) {
    window.clearTimeout(pollTimer);
    const deadline = Date.now() + SWITCH_TIMEOUT_MS;
    const tick = async () => {
      try {
        const data = await fetchState();
        adopt(data);
        if (data.state === "switching") {
          setStatus(data.progress || "正在切换…", "busy");
          if (Date.now() > deadline) {
            setStatus("切换超时，请查看 tts_supervisor.log。", "error");
            setBusy(false);
            settle?.(data);
            return;
          }
          pollTimer = window.setTimeout(tick, POLL_MS);
          return;
        }
        if (data.state === "error") {
          setStatus(`切换失败：${data.error || "未知错误"}`, "error");
          setBusy(false);
          select.value = active;
          settle?.(data);
          return;
        }
        // ready
        active = data.active;
        setBusy(false);
        select.value = data.active;
        const info = catalog.get(data.active);
        if (info) hint.textContent = `${info.detail}。`;
        applyVoices(data.active);
        setStatus(`已切换为 ${info?.label || data.active}，点击“开始/重新连接”继续对话。`, "ok");
        onActive?.(data.active, data);
        onReady?.(data.active);
        settle?.(data);
      } catch (err) {
        if (Date.now() > deadline) {
          setStatus(`切换未完成：${err instanceof Error ? err.message : err}`, "error");
          setBusy(false);
          settle?.({ state: "error" });
          return;
        }
        setStatus(`查询切换进度失败：${err instanceof Error ? err.message : err}`, "error");
        pollTimer = window.setTimeout(tick, POLL_MS * 2);
      }
    };
    pollTimer = window.setTimeout(tick, POLL_MS);
  }

  select.addEventListener("change", async () => {
    const engine = select.value;
    if (!engine || engine === active) return;
    const info = catalog.get(engine);
    const label = info?.label || engine;
    const ok = window.confirm(
      `切换到 ${label}？\n\n` +
        "TTS 后端是启动参数，切换需要重启语音管线（约 30–90 秒），当前对话会中断。",
    );
    if (!ok) {
      select.value = active;
      return;
    }

    setBusy(true);
    setStatus(`正在切换到 ${label}…`, "busy");
    beginSwitch();
    try {
      await onSwitchStart?.();
      const res = await fetch(ADMIN_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok && res.status !== 202) {
        throw new Error(body.error || `HTTP ${res.status}`);
      }
      schedulePoll(engine);
    } catch (err) {
      setBusy(false);
      select.value = active;
      setStatus(`切换失败：${err instanceof Error ? err.message : err}`, "error");
    }
  });

  voiceSelect.addEventListener("change", () => {
    const engine = select.value;
    if (engine) localStorage.setItem(VOICE_KEY_PREFIX + engine, voiceSelect.value);
  });

  return {
    refresh,
    /** Resolves after the in-flight switch settles (ready or error). */
    whenSettled: () => settled,
    get engine() {
      return active;
    },
    get busy() {
      return busy;
    },
  };
}
