// Unit tests for the left-rail system info panel (ui/system-info.js).
// The module is DOM-free at import time (all document access lives inside
// initSystemInfo), so we can exercise the data + HTML builder under node:test.

import test from "node:test";
import assert from "node:assert/strict";

import { SYSTEM_INFO, buildSystemInfoHTML } from "../ui/system-info.js";

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
