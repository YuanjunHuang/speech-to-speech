// @ts-check
/**
 * Client-side conversation latency meter.
 *
 * The server already reports per-response timings (see `turn-latency.js`), but
 * those stop at "first audio produced on the server" and never see the network
 * hop or the browser's playback buffer. This meter measures what the user
 * actually experiences, using only the events the browser already receives:
 *
 *   - assistant text deltas  (`response.output_audio_transcript.delta`)
 *   - assistant audio chunks (`response.output_audio.delta` samples)
 *   - the moment playback really starts (the worklet's first released batch)
 *
 * Per sentence it reports `文字首字 -> 实际出声`. The text-to-audio pairing is
 * reliable because the pipeline is serialized: sentence N's text is emitted
 * before sentence N's audio, and sentence N+1's text is emitted only after
 * sentence N has been fully synthesized. So the first audio chunk that arrives
 * after a sentence's text (claimed in order) is that sentence's own audio.
 *
 * Per-sentence playback time uses the sample timeline: playback starts at
 * `playbackStartAt` and runs at the pipeline rate, so a sentence whose first
 * chunk is preceded by `k` samples of audio starts playing `k / rate` later.
 * This assumes gap-free playback, which holds while the server keeps ahead.
 */

/** Pipeline/client audio sample rate (matches AUDIO_SAMPLE_RATE). */
export const CLIENT_LATENCY_SAMPLE_RATE = 24_000;

/**
 * @typedef {Object} SentenceTiming
 * @property {number} index
 * @property {string} text
 * @property {number|null} textAtMs Assistant text delta arrival, relative to response start.
 * @property {number|null} audioAtMs First audio chunk received for this sentence, relative to response start.
 * @property {number|null} playbackAtMs Estimated start of audible playback for this sentence.
 * @property {number|null} textToAudioMs textAtMs -> audioAtMs (network + server TTS).
 * @property {number|null} textToPlaybackMs textAtMs -> playbackAtMs (what the user hears).
 */

/**
 * @typedef {Object} ClientLatencySummary
 * @property {string} responseId
 * @property {number|null} audioFirstAtMs
 * @property {number|null} playbackStartMs
 * @property {SentenceTiming|null} firstSentence
 * @property {SentenceTiming[]} sentences
 */

export class ClientLatencyMeter {
  constructor() {
    /** @type {string} */
    this.responseId = "";
    /** @type {number} */
    this._t0 = 0;
    /** @type {{index: number, text: string, tText: number, tAudio: number|null, samplesBeforeFirst: number}[]} */
    this._sentences = [];
    this._claimCursor = 0;
    this._samplesDelivered = 0;
    /** @type {number|null} */
    this._audioFirstAt = null;
    /** @type {number|null} */
    this._playbackStartAt = null;
  }

  /** Start tracking a new response. @param {string} responseId */
  begin(responseId) {
    this.responseId = responseId || "";
    this._t0 = performance.now();
    this._sentences = [];
    this._claimCursor = 0;
    this._samplesDelivered = 0;
    this._audioFirstAt = null;
    this._playbackStartAt = null;
  }

  /** @param {string} responseId */
  isActive(responseId) {
    return Boolean(responseId) && responseId === this.responseId;
  }

  /** A new assistant sentence's text arrived. @param {string} responseId @param {string} text */
  noteText(responseId, text) {
    if (!this.isActive(responseId) || !text) return;
    this._sentences.push({
      index: this._sentences.length,
      text,
      tText: performance.now(),
      tAudio: null,
      samplesBeforeFirst: 0,
    });
  }

  /**
   * One assistant audio chunk was decoded. Claim it for the next sentence that
   * is still waiting for audio, then advance the sample timeline.
   * @param {string} responseId @param {number} sampleCount
   */
  noteAudioChunk(responseId, sampleCount) {
    if (!this.isActive(responseId) || !(sampleCount > 0)) return;
    const now = performance.now();
    if (this._audioFirstAt === null) this._audioFirstAt = now;
    const sentence = this._sentences[this._claimCursor];
    if (sentence && sentence.tAudio === null) {
      sentence.tAudio = now;
      sentence.samplesBeforeFirst = this._samplesDelivered;
      this._claimCursor += 1;
    }
    this._samplesDelivered += sampleCount;
  }

  /** Audible playback began (first batch released to the worklet). @param {string} responseId */
  notePlaybackStart(responseId) {
    if (!this.isActive(responseId) || this._playbackStartAt !== null) return;
    this._playbackStartAt = performance.now();
  }

  /** @param {number|null} at @returns {number|null} */
  _relative(at) {
    return at === null ? null : Math.max(0, at - this._t0);
  }

  /**
   * Freeze the measurements for a finished response.
   * @param {string} responseId
   * @returns {ClientLatencySummary|null}
   */
  summary(responseId) {
    if (!this.isActive(responseId) || this._sentences.length === 0) return null;
    /** @type {SentenceTiming[]} */
    const sentences = this._sentences.map((sentence) => {
      const textAtMs = this._relative(sentence.tText);
      const audioAtMs = this._relative(sentence.tAudio);
      let playbackAt = null;
      if (this._playbackStartAt !== null && sentence.tAudio !== null) {
        playbackAt = this._playbackStartAt + (sentence.samplesBeforeFirst / CLIENT_LATENCY_SAMPLE_RATE) * 1000;
      }
      const playbackAtMs = this._relative(playbackAt);
      return {
        index: sentence.index,
        text: sentence.text,
        textAtMs,
        audioAtMs,
        playbackAtMs,
        textToAudioMs: textAtMs !== null && audioAtMs !== null ? audioAtMs - textAtMs : null,
        textToPlaybackMs: textAtMs !== null && playbackAtMs !== null ? playbackAtMs - textAtMs : null,
      };
    });
    return {
      responseId,
      audioFirstAtMs: this._relative(this._audioFirstAt),
      playbackStartMs: this._relative(this._playbackStartAt),
      firstSentence: sentences[0] ?? null,
      sentences,
    };
  }
}
