// @ts-check
/**
 * StylePicker — the "语音情绪" control in the left rail.
 *
 * The pipeline normally carries an emotion as an inline style tag in the
 * assistant text (`[style:开心]`), which the model has to emit on its own.
 * A client can also pin one explicitly: `session.update` accepts extra fields,
 * so we send `session: { style }` and the LM output processor reads it off the
 * Realtime session (`LLM/lm_output_processor.py::_session_style`). A pinned
 * style outranks the model's own reading, which makes the delivery
 * deterministic even with a model that ignores the tag protocol.
 *
 * "自动" (the default) sends no pin and leaves the tag protocol in charge.
 */

import { escHtml } from "./dom.js";

/**
 * Canonical vocabulary, in the order it is offered. The stored values must stay
 * in sync with `pipeline/style.py::CANONICAL_STYLES` plus the "" (auto) case.
 * @type {{value: string, label: string}[]}
 */
export const STYLE_OPTIONS = [
  { value: "", label: "自动" },
  { value: "开心", label: "开心" },
  { value: "难过", label: "难过" },
  { value: "生气", label: "生气" },
  { value: "平静", label: "平静" },
  { value: "耳语", label: "耳语" },
  { value: "惊讶", label: "惊讶" },
  { value: "恐惧", label: "恐惧" },
  { value: "厌恶", label: "厌恶" },
  { value: "严肃", label: "严肃" },
];

/** Sentinel for "let the model decide"; also what an unset pin resolves to. */
export const AUTO_STYLE = "";

const STORAGE_KEY = "s2s.stylePin";

/**
 * Build the picker's markup. Pure (no DOM access) so it can be unit tested.
 * @param {string} [current] the pinned style, or "" for auto
 * @param {{value: string, label: string}[]} [options]
 * @returns {string}
 */
export function buildStylePickerHTML(current = AUTO_STYLE, options = STYLE_OPTIONS) {
  const chips = options
    .map((option) => {
      const checked = option.value === current ? " checked" : "";
      return `<label class="style-chip">
          <input type="radio" name="style-pin" value="${escHtml(option.value)}"${checked} />
          <span>${escHtml(option.label)}</span>
        </label>`;
    })
    .join("\n        ");

  return `<fieldset class="style-picker">
        <legend class="style-picker-legend">语音情绪</legend>
        <div class="style-picker-options">
        ${chips}
        </div>
        <p class="style-picker-hint">自动：由模型按对话内容挑语气。选具体情绪后，之后的回复固定用它。</p>
      </fieldset>`;
}

/**
 * The remembered pin, ignoring anything no longer in the vocabulary.
 * @returns {string}
 */
export function readStoredStyle() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    const known = STYLE_OPTIONS.some((option) => option.value === stored);
    return known && stored !== null ? stored : AUTO_STYLE;
  } catch {
    return AUTO_STYLE;
  }
}

/**
 * Render the picker into #style-picker and report changes.
 * @param {(style: string) => void} onChange called once with the initial value, then on every change
 * @returns {string} the effective style at init time
 */
export function initStylePicker(onChange) {
  const current = readStoredStyle();
  // Tolerate a stale index.html with no rail slot: the picker is optional chrome,
  // and throwing here would abort the whole module (start button included). The
  // remembered pin is still returned, so a connect-time apply keeps working.
  const slot = document.querySelector("#style-picker");
  if (!(slot instanceof HTMLElement)) return current;

  slot.innerHTML = buildStylePickerHTML(current);

  const radios = /** @type {NodeListOf<HTMLInputElement>} */ (
    slot.querySelectorAll('input[name="style-pin"]')
  );
  for (const radio of radios) {
    radio.addEventListener("change", () => {
      if (!radio.checked) return;
      try {
        localStorage.setItem(STORAGE_KEY, radio.value);
      } catch {
        /* storage unavailable (private mode) — the pin just isn't remembered */
      }
      onChange(radio.value);
    });
  }

  onChange(current);
  return current;
}
