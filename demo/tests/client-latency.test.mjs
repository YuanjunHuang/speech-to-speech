import assert from "node:assert/strict";
import test from "node:test";

import { ClientLatencyMeter } from "../client-latency.js";
import { S2sRealtimeClient } from "../s2s-realtime-client.js";

/** Deterministic clock so the meter's performance.now() reads are reproducible. */
function useFixedClock(start = 1_000) {
  const original = globalThis.performance;
  let now = start;
  globalThis.performance = { now: () => now };
  return {
    at: (value) => { now = value; },
    advance: (ms) => { now += ms; },
    restore: () => { globalThis.performance = original; },
  };
}

const round = (value) => (value === null ? null : Math.round(value));

test("attributes audio to sentences in order and maps playback on the sample timeline", () => {
  const clock = useFixedClock();
  try {
    const meter = new ClientLatencyMeter();
    meter.begin("r1");

    clock.at(1_250); meter.noteText("r1", "第一句。");          // +0.250s
    clock.at(1_400); meter.noteAudioChunk("r1", 6_000);        // sentence 0 starts (0.25s of audio)
    clock.at(1_650); meter.noteAudioChunk("r1", 6_000);        // sentence 0 continuation: no re-claim
    clock.at(1_660); meter.notePlaybackStart("r1");            // +0.660s
    clock.at(2_044); meter.noteText("r1", "第二句。");          // +1.044s
    clock.at(2_964); meter.noteAudioChunk("r1", 12_000);       // sentence 1 starts after 0.5s of audio
    clock.at(3_500); meter.noteAudioChunk("r1", 12_000);

    const summary = meter.summary("r1");
    assert.equal(round(summary.audioFirstAtMs), 400);
    assert.equal(round(summary.playbackStartMs), 660);
    assert.equal(summary.sentences.length, 2);

    const [first, second] = summary.sentences;
    assert.equal(round(first.textAtMs), 250);
    assert.equal(round(first.audioAtMs), 400);
    assert.equal(round(first.playbackAtMs), 660); // first release
    assert.equal(round(first.textToAudioMs), 150);
    assert.equal(round(first.textToPlaybackMs), 410);

    // Sentence 1's audio is preceded by sentence 0's 0.5s, so it plays 0.5s
    // after playback started — earlier than its chunk arrival would suggest.
    assert.equal(round(second.textAtMs), 1_044);
    assert.equal(round(second.audioAtMs), 1_964);
    assert.equal(round(second.playbackAtMs), 1_160);
    assert.equal(round(second.textToAudioMs), 920);
    assert.equal(round(second.textToPlaybackMs), 116);
  } finally {
    clock.restore();
  }
});

test("ignores events from other responses and reports unknown audio as null", () => {
  const clock = useFixedClock();
  try {
    const meter = new ClientLatencyMeter();
    meter.begin("r1");
    clock.at(1_100); meter.noteText("other", "忽略");
    clock.at(1_200); meter.noteAudioChunk("other", 24_000);
    clock.at(1_300); meter.notePlaybackStart("other");
    clock.at(1_400); meter.noteText("r1", "只有文字。");

    const summary = meter.summary("r1");
    assert.equal(summary.sentences.length, 1);
    assert.equal(round(summary.sentences[0].textAtMs), 400);
    assert.equal(summary.sentences[0].audioAtMs, null);
    assert.equal(summary.sentences[0].playbackAtMs, null);
    assert.equal(summary.sentences[0].textToAudioMs, null);
    assert.equal(summary.sentences[0].textToPlaybackMs, null);
    assert.equal(summary.audioFirstAtMs, null);
  } finally {
    clock.restore();
  }
});

test("summary is unavailable for a response with no text and after a new response begins", () => {
  const clock = useFixedClock();
  try {
    const meter = new ClientLatencyMeter();
    meter.begin("r1");
    clock.at(1_100); meter.noteAudioChunk("r1", 24_000);
    assert.equal(meter.summary("r1"), null); // no sentence to describe
    meter.begin("r2");
    assert.equal(meter.summary("r1"), null); // r1 is no longer active
  } finally {
    clock.restore();
  }
});

for (const transport of ["websocket", "webrtc"]) {
  test(`${transport} attaches browser timings to response-finished`, () => {
    const clock = useFixedClock();
    try {
      const client = new S2sRealtimeClient({ transport, directUrl: "ws://unused" });
      const finished = [];
      client.addEventListener("response-finished", (event) => finished.push(event.detail));

      clock.at(2_000);
      client._onTransportEvent({ type: "response.created", response: { id: "r1", status: "in_progress" } });
      clock.at(2_250);
      client._onTransportEvent({
        type: "response.output_audio_transcript.delta", response_id: "r1", delta: "你好。",
      });
      clock.at(3_000);
      client._onTransportEvent({ type: "response.done", response: { id: "r1", status: "completed" } });

      assert.equal(finished.length, 1);
      const timing = finished[0].clientTiming;
      assert.equal(timing.sentences.length, 1);
      assert.equal(round(timing.sentences[0].textAtMs), 250);
      assert.equal(timing.sentences[0].textToPlaybackMs, null); // no audio was delivered
    } finally {
      clock.restore();
    }
  });
}
