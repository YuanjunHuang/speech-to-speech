// Unit tests for the TTS engine picker (tts-engine.js).
// The module talks to the DOM and to /tts-admin, so the test supplies a tiny
// fake element pair, a stubbed fetch queue and an immediate timer. What matters
// here is the contract the UI relies on: the catalog comes from the supervisor,
// the voice list always matches the active engine, and a switch is only
// reported ready after the supervisor says so.

import test from "node:test";
import assert from "node:assert/strict";

class FakeEl {
  constructor() {
    this.value = "";
    this.textContent = "";
    this.hidden = false;
    this.disabled = false;
    this.dataset = {};
    this.children = [];
    this.listeners = new Map();
  }
  addEventListener(type, fn) {
    const list = this.listeners.get(type) ?? [];
    list.push(fn);
    this.listeners.set(type, list);
  }
  replaceChildren(...kids) {
    this.children = kids;
    if (kids.length && !kids.some((kid) => kid.value === this.value)) this.value = kids[0].value;
  }
  async emit(type) {
    for (const fn of this.listeners.get(type) ?? []) await fn();
  }
}

const CATALOG = [
  { id: "qwen3", label: "Qwen3-TTS 1.7B", kind: "inproc", transport: "流式", detail: "流式合成", voices: ["Vivian", "Aiden"] },
  { id: "cosyvoice2", label: "CosyVoice2-0.5B", kind: "bridge", transport: "整段", detail: "整段合成", voices: ["vivian"] },
  { id: "cosyvoice3", label: "CosyVoice3-0.5B", kind: "bridge", transport: "整段", detail: "整段合成", voices: ["vivian"] },
];

const state = (overrides = {}) => ({
  active: "qwen3",
  state: "ready",
  progress: "",
  error: null,
  s2s_running: true,
  engines: CATALOG,
  ...overrides,
});

function installGlobals(responses) {
  const store = new Map();
  const posts = [];
  globalThis.location = { protocol: "http:", port: "8080" };
  globalThis.localStorage = {
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => store.set(key, String(value)),
  };
  globalThis.document = {
    createElement: () => ({ value: "", textContent: "" }),
  };
  globalThis.window = {
    setTimeout: (fn) => {
      queueMicrotask(fn);
      return 0;
    },
    clearTimeout: () => {},
    confirm: () => true,
  };
  globalThis.fetch = async (url, init) => {
    if (init?.method === "POST") {
      posts.push(JSON.parse(init.body));
      return { ok: true, status: 202, json: async () => ({ ok: true, state: "switching" }) };
    }
    const next = responses.shift();
    if (!next) throw new Error("unexpected GET");
    return { ok: true, status: 200, json: async () => next };
  };
  return { store, posts };
}

test("refresh renders the catalog, activates the reported engine and matches voices", async () => {
  installGlobals([state({ active: "cosyvoice2" })]);
  const { createTtsEnginePicker } = await import("../tts-engine.js");
  const select = new FakeEl();
  const hint = new FakeEl();
  const status = new FakeEl();
  const voiceSelect = new FakeEl();

  const picker = createTtsEnginePicker({ select, hint, status, voiceSelect });
  await picker.refresh();

  assert.deepEqual(select.children.map((o) => o.value), ["qwen3", "cosyvoice2", "cosyvoice3"]);
  assert.equal(select.value, "cosyvoice2");
  assert.deepEqual(voiceSelect.children.map((o) => o.value), ["vivian"]);
  assert.equal(voiceSelect.value, "vivian");
  assert.match(hint.textContent, /整段合成/);
  assert.equal(status.hidden, true);
});

test("a switch posts the engine, shows progress and calls back only when ready", async () => {
  const responses = [
    state(), // initial refresh
    { ...state(), state: "switching", progress: "停止语音管线…" },
    { ...state(), state: "switching", progress: "加载语音管线…" },
    state({ active: "cosyvoice3" }), // ready
  ];
  const { posts } = installGlobals(responses);
  const { createTtsEnginePicker } = await import("../tts-engine.js");
  const select = new FakeEl();
  const hint = new FakeEl();
  const status = new FakeEl();
  const voiceSelect = new FakeEl();
  const events = [];

  const picker = createTtsEnginePicker({
    select,
    hint,
    status,
    voiceSelect,
    onSwitchStart: async () => events.push("stop"),
    onReady: (engine) => events.push(`ready:${engine}`),
  });
  await picker.refresh();
  select.value = "cosyvoice3";
  await select.emit("change");
  await picker.whenSettled();

  assert.deepEqual(posts, [{ engine: "cosyvoice3" }]);
  assert.deepEqual(events, ["stop", "ready:cosyvoice3"]);
  assert.equal(select.value, "cosyvoice3");
  assert.equal(select.disabled, false);
  assert.deepEqual(voiceSelect.children.map((o) => o.value), ["vivian"]);
  assert.match(status.textContent, /已切换为 CosyVoice3-0\.5B/);
  assert.equal(status.dataset.kind, "ok");
  assert.equal(voiceSelect.disabled, false);
});

test("a failed switch restores the previous engine and surfaces the error", async () => {
  const responses = [
    state(),
    { ...state(), state: "error", error: "RuntimeError: bridge crashed" },
  ];
  installGlobals(responses);
  const { createTtsEnginePicker } = await import("../tts-engine.js");
  const select = new FakeEl();
  const status = new FakeEl();
  const picker = createTtsEnginePicker({
    select,
    hint: new FakeEl(),
    status,
    voiceSelect: new FakeEl(),
  });
  await picker.refresh();
  select.value = "cosyvoice2";
  await select.emit("change");
  await picker.whenSettled();

  assert.equal(select.value, "qwen3", "failed switch must revert the dropdown");
  assert.equal(select.disabled, false);
  assert.equal(status.dataset.kind, "error");
  assert.match(status.textContent, /bridge crashed/);
});
