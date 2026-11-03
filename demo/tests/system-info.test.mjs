// Unit tests for the left-rail system info panel (ui/system-info.js).
// The module is DOM-free at import time (all document access lives inside
// initSystemInfo), so we can exercise the data + HTML builder under node:test.

import test from "node:test";
import assert from "node:assert/strict";

import { SYSTEM_INFO, buildSystemInfoHTML, deriveSystemInfo } from "../ui/system-info.js";

test("every pipeline module maps to a declared device", () => {
  const keys = new Set(SYSTEM_INFO.devices.map((d) => d.key));
  for (const module of SYSTEM_INFO.modules) {
    assert.ok(keys.has(module.device), `unknown device "${module.device}" for stage ${module.stage}`);
  }
});

test("hardware inventory covers CPU, both GPUs and the NPU slot", () => {
  assert.deepEqual(
    SYSTEM_INFO.devices.map((d) => d.key).sort(),
    ["cpu", "gpu0", "gpu1", "npu"],
  );
  for (const device of SYSTEM_INFO.devices) {
    assert.ok(device.badge, `${device.key} is missing a badge label`);
    assert.ok(device.tag && device.name, `${device.key} is missing card text`);
    // Each card renders its specs one fact per line, so the list must be well formed.
    assert.ok(Array.isArray(device.spec) && device.spec.length > 0, `${device.key} needs spec lines`);
    for (const line of device.spec) assert.equal(typeof line, "string");
  }
});

test("builder labels every module with its device badge and model", () => {
  const html = buildSystemInfoHTML();
  for (const module of SYSTEM_INFO.modules) {
    assert.ok(html.includes(module.name), `missing model: ${module.name}`);
  }
  for (const device of SYSTEM_INFO.devices) {
    for (const line of device.spec) assert.ok(html.includes(line), `missing spec line: ${line}`);
  }
  // The Arc GPU is the headline accelerator; both its card and badge must show.
  assert.ok(html.includes("Intel Arc A770"));
  assert.ok(html.includes('data-device="gpu1"'));
  assert.ok(html.includes(SYSTEM_INFO.host.name));
});

test("builder escapes untrusted model/host strings", () => {
  const html = buildSystemInfoHTML({
    host: { name: "<img src=x onerror=alert(1)>", meta: ["m <x>"] },
    devices: [{ key: "cpu", tag: "CPU", badge: "CPU", name: "n", spec: ["s <b>"] }],
    modules: [{ stage: "S", job: "j", name: "mod", device: "cpu" }],
    notes: ["a < b & c"],
  });
  assert.ok(!html.includes("<img src=x"));
  assert.ok(html.includes("&lt;img src=x"));
  assert.ok(html.includes("m &lt;x&gt;"));
  assert.ok(html.includes("s &lt;b&gt;"));
  assert.ok(html.includes("a &lt; b &amp; c"));
});

test("deriveSystemInfo: iGPU-only host makes the iGPU the accelerator", () => {
  const { info, accel } = deriveSystemInfo({
    host: { name: "Mind", meta: ["x"] },
    cpu: { name: "i5-1340P", spec: ["16 线程"] },
    gpus: [{ tag: "GPU", name: "Intel Iris Xe Graphics", discrete: false, spec: ["核显"] }],
  });
  assert.equal(accel, "gpu0");
  assert.deepEqual(info.devices.map((d) => d.key), ["cpu", "gpu0"]);
  for (const m of info.modules) assert.ok(info.devices.some((d) => d.key === m.device));
  assert.ok(info.modules.find((m) => m.stage === "LLM").device === "gpu0");
  assert.ok(buildSystemInfoHTML(info).includes("Iris Xe"));
});

test("deriveSystemInfo: discrete GPU wins and the iGPU is idle", () => {
  const { info, accel } = deriveSystemInfo({
    cpu: { name: "i7", spec: [] },
    gpus: [
      { tag: "GPU.0", name: "Intel AlderLake-S GT1", discrete: false, spec: [] },
      { tag: "GPU.1", name: "Intel Arc A770", discrete: true, spec: [] },
    ],
  });
  assert.equal(accel, "gpu1");
  assert.equal(info.devices.find((d) => d.key === "gpu0").note, "空闲");
  assert.equal(info.devices.find((d) => d.key === "gpu1").badge, "Arc A770");
});

test("deriveSystemInfo: no GPU falls back to CPU", () => {
  const { info, accel } = deriveSystemInfo({ cpu: { name: "x", spec: [] }, gpus: [] });
  assert.equal(accel, "cpu");
  assert.ok(info.modules.every((m) => m.device === "cpu"));
});
