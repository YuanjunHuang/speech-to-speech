// Unit tests for the voice-style picker (ui/style-picker.js).
// The module is DOM-free at import time (all document access lives inside
// initStylePicker), so the vocabulary + HTML builder run under node:test.

import test from "node:test";
import assert from "node:assert/strict";

import { AUTO_STYLE, STYLE_OPTIONS, buildStylePickerHTML } from "../ui/style-picker.js";

// Mirrors pipeline/style.py::CANONICAL_STYLES. A style the backend cannot map
// would be silently ignored, so the two lists must not drift.
const BACKEND_STYLES = ["开心", "难过", "生气", "平静", "耳语", "惊讶", "恐惧", "厌恶", "严肃"];

test("offers auto plus exactly the backend vocabulary, auto first", () => {
  assert.equal(STYLE_OPTIONS[0].value, AUTO_STYLE);
  assert.equal(STYLE_OPTIONS[0].label, "自动");
  assert.deepEqual(
    STYLE_OPTIONS.slice(1).map((option) => option.value),
    BACKEND_STYLES,
  );
});

test("every style has a distinct, non-empty label", () => {
  const labels = STYLE_OPTIONS.map((option) => option.label);
  for (const option of STYLE_OPTIONS) assert.ok(option.label, `missing label for "${option.value}"`);
  assert.equal(new Set(labels).size, labels.length);
});

test("auto is preselected by default", () => {
  const html = buildStylePickerHTML();
  assert.ok(html.includes('value="" checked'), "auto should be checked when no pin is given");
  assert.equal((html.match(/checked/g) || []).length, 1, "exactly one chip may be checked");
});

test("the current pin is the checked chip", () => {
  const html = buildStylePickerHTML("难过");
  assert.ok(html.includes('value="难过" checked'));
  assert.ok(!html.includes('value="" checked'));
  assert.equal((html.match(/checked/g) || []).length, 1);
});

test("renders one radio per option inside a labelled group", () => {
  const html = buildStylePickerHTML();
  assert.equal((html.match(/type="radio"/g) || []).length, STYLE_OPTIONS.length);
  assert.ok(html.includes('name="style-pin"'));
  assert.ok(html.includes("语音情绪"));
});

test("escapes option values so markup cannot break out of the attribute", () => {
  const html = buildStylePickerHTML("", [{ value: '"><img src=x>', label: "<b>pwn</b>" }]);
  assert.ok(!html.includes("<img src=x"));
  assert.ok(html.includes("&lt;img src=x&gt;"));
  assert.ok(!html.includes("<b>pwn</b>"));
  assert.ok(html.includes("&lt;b&gt;pwn&lt;/b&gt;"));
});
