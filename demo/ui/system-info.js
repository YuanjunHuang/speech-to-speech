// @ts-check
/**
 * SystemInfo — renders the "推理设备" rail: the inference host's CPU / GPU / NPU
 * inventory plus a stage→device map for the S2S pipeline (which module runs on
 * which accelerator, and with which model).
 *
 * The host's hardware and the running model lineup cannot be discovered from
 * the browser (they live on the backend machine), so they are declared here as
 * data and rendered declaratively. Update `SYSTEM_INFO` when the backend host
 * or the served models change. Values below are for the dgpu-nuc demo host
 * (10.239.9.50), read back from `lscpu` / `lspci` and the OVMS model configs.
 *
 * Layout: a static left column on wide viewports; a slide-in drawer (toggled
 * from the topbar) on narrow ones. See the `.system-info` rules in style.css.
 */

import { $, escHtml } from "./dom.js";

/**
 * @typedef {"cpu" | "gpu0" | "gpu1" | "npu"} DeviceKey
 *
 * @typedef {object} DeviceInfo
 * @property {DeviceKey} key      selector used by `modules[].device`
 * @property {string} tag         rail tag shown on the card (e.g. "GPU.1")
 * @property {string} badge       compact label shown next to a pipeline stage
 * @property {string} name        product name
 * @property {string[]} spec      short spec lines, one fact per line
 * @property {string} [note]      what this device is used for
 *
 * @typedef {object} ModuleInfo
 * @property {string} stage       short stage tag (VAD / STT / LLM / TTS …)
 * @property {string} job         what the stage does, in Chinese
 * @property {string} name        model + quantization
 * @property {DeviceKey} device   where it runs
 * @property {string} [detail]    extra hint (e.g. the serving port)
 *
 * @typedef {object} SystemInfoData
 * @property {{name: string, meta: string[]}} host
 * @property {DeviceInfo[]} devices
 * @property {ModuleInfo[]} modules
 * @property {string[]} notes
 */

/** @type {SystemInfoData} */
export const SYSTEM_INFO = {
  host: {
    name: "Intel NUC 12 Extreme",
    meta: ["NUC12DCMi7 · Ubuntu / Linux 6.8.0-41", "16 GB DDR4 · OpenVINO 可用 CPU / GPU.0 / GPU.1"],
  },
  devices: [
    {
      key: "cpu",
      tag: "CPU",
      badge: "CPU",
      name: "Intel Core i7-12700",
      spec: ["第 12 代 · 12 核 20 线程 (8P + 4E)", "最高 4.9 GHz · L3 25 MB"],
      note: "VAD · 断句 · TTS",
    },
    {
      key: "gpu0",
      tag: "GPU.0",
      badge: "UHD 770",
      name: "Intel UHD Graphics 770",
      spec: ["核显 · Alder Lake-S GT1 · 32 EU · 共享系统内存"],
      note: "空闲",
    },
    {
      key: "gpu1",
      tag: "GPU.1",
      badge: "Arc A770",
      name: "Intel Arc A770",
      spec: ["独立显卡 · Alchemist (DG2) · 16 GB GDDR6"],
      note: "STT · LLM",
    },
    {
      key: "npu",
      tag: "NPU",
      badge: "NPU",
      name: "无独立 NPU",
      spec: ["第 12 代酷睿平台未集成 NPU"],
      note: "未启用",
    },
  ],
  modules: [
    { stage: "VAD", job: "语音活动检测", name: "silero-vad v5", device: "cpu" },
    { stage: "断句", job: "语义端点判断", name: "smart-turn v3.2", device: "cpu" },
    {
      stage: "STT",
      job: "语音转文字",
      name: "whisper-small · INT8",
      device: "gpu1",
      detail: "OVMS :8001",
    },
    {
      stage: "LLM",
      job: "生成回复 · 情绪标签",
      name: "MiniCPM-V 4.5 · INT4",
      device: "gpu1",
      detail: "OVMS :8000",
    },
    {
      stage: "TTS",
      job: "语音合成 · 情绪 instruct",
      name: "Qwen3-TTS · 1.7B CustomVoice",
      device: "cpu",
      detail: "GGML · CPU",
    },
  ],
  notes: [
    "推理链路：麦克风 → VAD → STT → LLM → TTS → 扬声器。STT 与 LLM 在 Arc A770 (GPU.1) 上加速，其余环节运行于 CPU。",
    "情绪：LLM 在文本里打 [style:…] 标签，管线剥离后把情绪交给 TTS 逐句 instruct；左侧可手动固定情绪。",
  ],
};

/** Rail data currently shown (static template until the host reports itself). */
let currentInfo = SYSTEM_INFO;

/**
 * Key of the device that runs the accelerated stages: the discrete GPU, else the
 * iGPU, else the CPU. Modules written against the NUC layout use "gpu1" for it.
 */
let accelKey = "gpu1";

/** @param {string} [key] */
function resolveKey(key) {
  return key === "gpu1" ? accelKey : key;
}

/** Badge text of the accelerator, for stage details ("flow/HiFT · Arc A770"). */
export function getAccelBadge() {
  const d = currentInfo.devices.find((x) => x.key === accelKey);
  return d ? d.badge : "GPU";
}

/**
 * Turn the host report of sysstats_service.py (/sys-stats/info) into rail data.
 * Pure; the static SYSTEM_INFO supplies the pipeline modules and notes.
 * @param {any} host
 * @param {SystemInfoData} [base]
 * @returns {{info: SystemInfoData, accel: string}}
 */
export function deriveSystemInfo(host, base = SYSTEM_INFO) {
  /** @type {DeviceInfo[]} */
  const devices = [
    {
      key: "cpu",
      tag: "CPU",
      badge: "CPU",
      name: host.cpu?.name || "CPU",
      spec: host.cpu?.spec || [],
      note: "VAD · 断句 · TTS",
    },
  ];
  for (const g of host.gpus || []) {
    const key = g.discrete ? "gpu1" : "gpu0";
    if (devices.some((d) => d.key === key)) continue;
    devices.push({
      key,
      tag: g.tag,
      badge: g.discrete ? String(g.name).replace(/^Intel\s+/, "") : "iGPU",
      name: g.name,
      spec: g.spec || [],
      note: "",
    });
  }
  const accel = devices.some((d) => d.key === "gpu1") ? "gpu1" : devices.some((d) => d.key === "gpu0") ? "gpu0" : "cpu";
  const accelDev = /** @type {DeviceInfo} */ (devices.find((d) => d.key === accel));
  for (const d of devices) {
    if (d.key === accel && accel !== "cpu") d.note = "STT · LLM";
    else if (d.key !== "cpu") d.note = "空闲";
  }
  const modules = base.modules.map((m) => ({ ...m, device: m.device === "gpu1" ? accel : m.device }));
  const notes = base.notes.slice();
  notes[0] = accel === "cpu"
    ? "推理链路：麦克风 → VAD → STT → LLM → TTS → 扬声器。本机没有可用的 GPU，所有环节运行于 CPU。"
    : `推理链路：麦克风 → VAD → STT → LLM → TTS → 扬声器。STT 与 LLM 在 ${accelDev.name} (${accelDev.tag}) 上加速，其余环节运行于 CPU。`;
  return { info: { host: host.host || base.host, devices, modules, notes }, accel };
}

/**
 * Build the rail's inner HTML. Pure (no DOM access) so it can be unit tested.
 * @param {SystemInfoData} [info]
 * @returns {string}
 */
export function buildSystemInfoHTML(info = SYSTEM_INFO) {
  const esc = escHtml;
  const badgeFor = (key) => {
    const device = info.devices.find((d) => d.key === key);
    return device ? device.badge : String(key);
  };

  const hostMeta = info.host.meta.map((line) => `<span class="sysinfo-host-meta">${esc(line)}</span>`).join("");

  const devices = info.devices
    .map((d) => {
      const spec = d.spec.map((line) => `<span class="sysinfo-device-spec">${esc(line)}</span>`).join("");
      return `
        <li class="sysinfo-device" data-device="${esc(d.key)}">
          <div class="sysinfo-device-head">
            <span class="sysinfo-device-tag">${esc(d.tag)}</span>
            ${d.note ? `<span class="sysinfo-device-note">${esc(d.note)}</span>` : ""}
          </div>
          <span class="sysinfo-device-name">${esc(d.name)}</span>
          ${spec}
        </li>`;
    })
    .join("");

  const modules = info.modules
    .map(
      (m) => `
        <li class="sysinfo-module" data-device="${esc(m.device)}" data-stage="${esc(m.stage)}">
          <span class="sysinfo-stage">${esc(m.stage)}</span>
          <div class="sysinfo-module-text">
            <span class="sysinfo-module-name">${esc(m.name)}</span>
            <span class="sysinfo-module-job">${esc(m.job)}${m.detail ? ` · ${esc(m.detail)}` : ""}</span>
          </div>
          <span class="sysinfo-badge" data-device="${esc(m.device)}">${esc(badgeFor(m.device))}</span>
        </li>`,
    )
    .join("");

  const notes = info.notes.map((n) => `<li>${esc(n)}</li>`).join("");

  return `
    <div class="sysinfo-host">
      <span class="sysinfo-host-name">${esc(info.host.name)}</span>
      ${hostMeta}
    </div>

    <section class="sysinfo-section">
      <h3 class="sysinfo-section-title">硬件设备</h3>
      <ul class="sysinfo-devices">${devices}</ul>
    </section>

    <section class="sysinfo-section">
      <h3 class="sysinfo-section-title">S2S 推理链路</h3>
      <ul class="sysinfo-modules">${modules}</ul>
    </section>

    <ul class="sysinfo-notes">${notes}</ul>`;
}

const COLLAPSE_KEY = "s2s.devicePanel.collapsed";
const DESKTOP_QUERY = "(min-width: 1080px)";

/**
 * Rewrite the TTS row after the engine picker learns which backend is live.
 * The rail is otherwise static, but the TTS stage is user-selectable, so a
 * stale "Qwen3-TTS" next to a CosyVoice call would be a lie.
 *
 * @param {{name?: string, job?: string, detail?: string, device?: string}} info
 */
export function setTtsModule(info) {
  setStageModule("TTS", info);
}

/**
 * Same as setTtsModule for any pipeline stage ("STT", "LLM", "TTS"…).
 *
 * @param {string} stage
 * @param {{name?: string, job?: string, detail?: string, device?: string}} info
 */
export function setStageModule(stage, info) {
  const row = document.querySelector(`.sysinfo-module[data-stage="${stage}"]`);
  if (!row) return;
  if (info.name) {
    const el = row.querySelector(".sysinfo-module-name");
    if (el) el.textContent = info.name;
  }
  if (info.job || info.detail) {
    const el = row.querySelector(".sysinfo-module-job");
    if (el) el.textContent = [info.job, info.detail].filter(Boolean).join(" · ");
  }
  if (info.device) {
    const key = resolveKey(info.device);
    const device = currentInfo.devices.find((d) => d.key === key);
    row.dataset.device = key;
    const badge = row.querySelector(".sysinfo-badge");
    if (badge && device) {
      badge.dataset.device = key;
      badge.textContent = device.badge;
    }
  }
}

/** Resolves once the live host report has been applied (or failed → template stays). */
let hostLoaded = Promise.resolve();
export function whenHostInfoLoaded() {
  return hostLoaded;
}

/** @param {HTMLElement} body */
async function loadHostInfo(body) {
  try {
    const res = await fetch("sys-stats/info", { cache: "no-store" });
    if (!res.ok) return;
    const { info, accel } = deriveSystemInfo(await res.json());
    currentInfo = info;
    accelKey = accel;
    body.innerHTML = buildSystemInfoHTML(info);
  } catch {
    /* direct :7860 or backend without the stats service: keep the template */
  }
}

/**
 * Render the rail and wire the topbar toggle / drawer close affordances.
 * Idempotent: safe to call once at startup.
 */
export function initSystemInfo() {
  const body = $("#system-info-body");
  const toggle = $("#device-btn");
  const closeBtn = document.querySelector("#system-info-close");
  const backdrop = document.querySelector("#system-info-backdrop");

  body.innerHTML = buildSystemInfoHTML(SYSTEM_INFO);
  hostLoaded = loadHostInfo(body);

  const desktop = window.matchMedia(DESKTOP_QUERY);
  const readCollapsed = () => {
    try {
      return localStorage.getItem(COLLAPSE_KEY) === "1";
    } catch {
      return false;
    }
  };

  const syncAria = () => {
    const expanded = desktop.matches
      ? !document.body.classList.contains("device-collapsed")
      : document.body.classList.contains("device-open");
    toggle.setAttribute("aria-expanded", expanded ? "true" : "false");
  };

  const resetForViewport = () => {
    if (desktop.matches) {
      document.body.classList.remove("device-open");
      document.body.classList.toggle("device-collapsed", readCollapsed());
    } else {
      document.body.classList.remove("device-collapsed", "device-open");
    }
    syncAria();
  };

  toggle.addEventListener("click", () => {
    if (desktop.matches) {
      const collapsed = document.body.classList.toggle("device-collapsed");
      try {
        localStorage.setItem(COLLAPSE_KEY, collapsed ? "1" : "0");
      } catch {
        /* storage unavailable (private mode) — state just isn't persisted */
      }
    } else {
      document.body.classList.toggle("device-open");
    }
    syncAria();
  });

  const closeDrawer = () => {
    document.body.classList.remove("device-open");
    syncAria();
  };
  closeBtn?.addEventListener("click", closeDrawer);
  backdrop?.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && document.body.classList.contains("device-open")) {
      closeDrawer();
    }
  });
  desktop.addEventListener("change", resetForViewport);

  resetForViewport();
}
