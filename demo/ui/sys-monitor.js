// @ts-check
/**
 * SysMonitor — floating, top-centre overlay with live CPU / GPU / memory
 * utilisation of the host running the S2S pipeline, drawn as sparklines.
 * Data comes from sysstats_service.py via the same-origin /sys-stats route.
 * Visibility is a persisted setting (settings panel) and the overlay has its own ✕.
 */

const KEY = "s2s.ui.sysMonitor";
const POS_KEY = "s2s.ui.sysMonitor.pos";
const POLL_MS = 1000;
const ROWS = [
  { key: "cpu", label: "CPU", color: "#22d3ee" },
  { key: "gpu", label: "GPU", color: "#8b7dff" },
  { key: "mem", label: "内存", color: "#34d399" },
];
const W = 120;
const H = 34;

/** @param {{ select: HTMLSelectElement }} opts */
export function initSysMonitor({ select }) {
  let enabled = localStorage.getItem(KEY) !== "off";
  /** @type {number | null} */
  let timer = null;
  let unavailable = false;

  const root = document.createElement("div");
  root.className = "sys-monitor";
  root.setAttribute("role", "status");
  root.setAttribute("aria-label", "设备实时占用");
  root.hidden = true;
  root.innerHTML = ROWS.map((r) => `
    <div class="sm-row" data-key="${r.key}">
      <div class="sm-head"><span class="sm-label">${r.label}</span><span class="sm-val">--</span></div>
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
        <path class="sm-area" fill="${r.color}" fill-opacity="0.18"></path>
        <path class="sm-line" fill="none" stroke="${r.color}" stroke-width="1.6" vector-effect="non-scaling-stroke"></path>
      </svg>
      <div class="sm-sub"></div>
    </div>`).join("") + `<button type="button" class="sm-close" aria-label="关闭占用面板" title="关闭">×</button>`;
  document.body.append(root);

  /** @param {number[]} ys */
  function paths(ys) {
    if (!ys.length) return ["", ""];
    const step = W / 59;
    const x0 = W - (ys.length - 1) * step;
    const pts = ys.map((v, i) => [x0 + i * step, H - 1 - (Math.min(100, Math.max(0, v)) / 100) * (H - 3)]);
    const line = "M" + pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("L");
    const area = `${line}L${pts[pts.length - 1][0].toFixed(1)},${H}L${pts[0][0].toFixed(1)},${H}Z`;
    return [line, area];
  }

  /** @param {any} data */
  function render(data) {
    const hist = data.history || [];
    const cur = data.latest || {};
    for (const r of ROWS) {
      const row = root.querySelector(`[data-key="${r.key}"]`);
      if (!row) continue;
      const [line, area] = paths(hist.map((/** @type {any} */ s) => s[r.key]));
      row.querySelector(".sm-line")?.setAttribute("d", line);
      row.querySelector(".sm-area")?.setAttribute("d", area);
      const v = cur[r.key];
      const val = row.querySelector(".sm-val");
      if (val) val.textContent = typeof v === "number" ? `${v.toFixed(0)}%` : "--";
      const sub = row.querySelector(".sm-sub");
      if (sub) {
        sub.textContent = r.key === "mem" && cur.memTotalGb ? `${cur.memUsedGb} / ${cur.memTotalGb} GB`
          : r.key === "gpu" && cur.vramGb !== undefined ? `显存 ${cur.vramGb} GB`
          : "";
      }
    }
  }

  async function poll() {
    try {
      const res = await fetch("sys-stats", { cache: "no-store" });
      if (!res.ok) throw new Error(String(res.status));
      const data = await res.json();
      unavailable = false;
      render(data);
      root.hidden = !enabled;
    } catch {
      unavailable = true;
      root.hidden = true;
    }
  }

  function apply() {
    select.value = enabled ? "on" : "off";
    root.hidden = !enabled || unavailable;
    if (enabled && timer === null) {
      void poll();
      timer = window.setInterval(() => { if (!document.hidden) void poll(); }, POLL_MS);
    } else if (!enabled && timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }

  /** @param {boolean} on */
  function set(on) {
    enabled = on;
    localStorage.setItem(KEY, on ? "on" : "off");
    apply();
  }

  // Drag anywhere on the panel (except ✕); position persists, double-click resets to top-centre.
  /** @param {number} x @param {number} y */
  function place(x, y) {
    const maxX = Math.max(0, window.innerWidth - root.offsetWidth);
    const maxY = Math.max(0, window.innerHeight - root.offsetHeight);
    root.style.transform = "none";
    root.style.left = `${Math.min(maxX, Math.max(0, x))}px`;
    root.style.top = `${Math.min(maxY, Math.max(0, y))}px`;
  }
  function restorePos() {
    try {
      const pos = JSON.parse(localStorage.getItem(POS_KEY) || "null");
      if (pos && Number.isFinite(pos.x) && Number.isFinite(pos.y)) place(pos.x, pos.y);
    } catch { /* ignore */ }
  }
  /** @type {{ dx: number, dy: number } | null} */
  let drag = null;
  root.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || /** @type {HTMLElement} */ (e.target).closest(".sm-close")) return;
    const b = root.getBoundingClientRect();
    drag = { dx: e.clientX - b.left, dy: e.clientY - b.top };
    root.setPointerCapture(e.pointerId);
    root.classList.add("dragging");
    e.preventDefault();
  });
  root.addEventListener("pointermove", (e) => { if (drag) place(e.clientX - drag.dx, e.clientY - drag.dy); });
  const endDrag = () => {
    if (!drag) return;
    drag = null;
    root.classList.remove("dragging");
    const b = root.getBoundingClientRect();
    localStorage.setItem(POS_KEY, JSON.stringify({ x: Math.round(b.left), y: Math.round(b.top) }));
  };
  root.addEventListener("pointerup", endDrag);
  root.addEventListener("pointercancel", endDrag);
  root.addEventListener("dblclick", () => {
    localStorage.removeItem(POS_KEY);
    root.style.left = root.style.top = root.style.transform = "";
  });
  window.addEventListener("resize", () => {
    if (root.style.transform === "none") {
      const b = root.getBoundingClientRect();
      place(b.left, b.top);
    }
  });
  root.hidden = false;
  restorePos();
  root.hidden = true;

  select.addEventListener("change", () => set(select.value !== "off"));
  root.querySelector(".sm-close")?.addEventListener("click", () => set(false));
  apply();
}
