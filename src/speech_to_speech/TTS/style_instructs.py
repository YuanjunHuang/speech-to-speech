"""Map a canonical spoken style onto an instruct prompt for instruction-following TTS.

The style channel itself is backend-agnostic: it only carries a bare value such
as ``开心``. A backend that accepts a natural-language instruction turns that
value into one here.

The phrasings for the canonical styles are the ones verified in the Qwen3-TTS
CustomVoice audition (same model, same sentence, prosody measured against a
neutral baseline), so they are known to move the delivery rather than being
plausible-sounding guesses.
"""

from __future__ import annotations

import os

from speech_to_speech.pipeline.style import normalize_style

STYLE_INSTRUCTS: dict[str, str] = {
    "开心": "用非常兴奋、开心的语气说这句话，语调上扬，语速稍快，充满活力。",
    "难过": "用低落、难过的语气说这句话，语速缓慢，声音低沉无力。",
    "生气": "用生气、愤怒的语气说这句话，语调加重，语速偏快，情绪激动。",
    "平静": "用平静、温柔的语气说这句话，语速稍慢，声音柔和。",
    "耳语": "用压低声音、悄悄耳语的语气说这句话，气声较多。",
    # Added with the CosyVoice3 vocabulary; these Qwen3 phrasings follow the same
    # pattern as the audited ones above but have not been auditioned on Qwen3.
    "惊讶": "用惊讶、震惊的语气说这句话，语调上扬，语速稍快。",
    "恐惧": "用害怕、惊恐的语气说这句话，声音微微颤抖，语速偏快。",
    "厌恶": "用厌恶、嫌弃的语气说这句话，语调下沉，语速缓慢。",
    "严肃": "用严肃、庄重的语气说这句话，语速平稳，声音沉稳。",
}


def instruct_for_style(style: str | None) -> str | None:
    """Build an instruct prompt for *style*, or ``None`` when there is none to build.

    Known canonical styles reuse the verified phrasings. Long-tail styles fall
    back to the same ``用…的语气说这句话`` template instead of being dropped, so an
    unexpected adjective still shifts the delivery rather than doing nothing.
    """
    if not style or not style.strip():
        return None
    canonical = normalize_style(style)
    if canonical is not None:
        # QWEN3_CALM_INSTRUCT=0 sends no instruct for the neutral style, so the speaker's own
        # delivery is used (calm is the default reply style, so this is the common case).
        if canonical == "平静" and os.environ.get("QWEN3_CALM_INSTRUCT", "1") == "0":
            return None
        return STYLE_INSTRUCTS[canonical]
    return f"用{style.strip()}的语气说这句话。"
