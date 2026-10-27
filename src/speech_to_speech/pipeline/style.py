"""Spoken-style channel: a model-agnostic emotion tag carried in assistant text.

The language model may prefix a spoken sentence with an inline style tag::

    [style:开心]今天天气真不错，我们下午出去走走吧。

:class:`~speech_to_speech.LLM.lm_output_processor.LMOutputProcessor` strips the
tag before the text reaches the client or the TTS stage and forwards the bare
style value on :class:`~speech_to_speech.pipeline.messages.TTSInput`. Each TTS
backend maps that value onto its own knob, so the language model never needs to
know which backend is configured or how it expresses emotion.

The tag is deliberately bracket-shaped:

* ``remove_unspeechable`` keeps ``[``, ``]`` and ``:``, so a tag survives the
  per-token streaming filter intact and can be parsed once a sentence exists;
* brackets are punctuation rather than something a spoken sentence contains on
  its own, so a false positive is unlikely.
"""

from __future__ import annotations

import re

# ``[style:value]`` or ``[风格:value]`` with optional inner whitespace and an
# optional full-width colon (Chinese text routinely renders ``:`` as ``：``).
# The Chinese spelling is accepted because small models sometimes mirror the
# instruction in their own words.
STYLE_TAG_PATTERN = re.compile(
    r"\[\s*(?:style|风格)\s*[:：]\s*(?P<style>[^\[\]<>]{1,48}?)\s*\]",
    re.IGNORECASE,
)

# Canonical vocabulary. Backends map these onto their own knob; the language
# model is told to stay inside this list.
CANONICAL_STYLES: tuple[str, ...] = ("开心", "难过", "生气", "平静", "耳语", "惊讶", "恐惧", "厌恶", "严肃")

# What a reply sounds like when nothing in the conversation calls for emotion. Also the
# fallback for a tag outside the vocabulary.
DEFAULT_STYLE = "平静"

# One Latin alias per canonical style, used to advertise the vocabulary to the
# model and to document the examples in the prompt hint.
STYLE_EXAMPLES: dict[str, str] = {
    "开心": "happy",
    "难过": "sad",
    "生气": "angry",
    "平静": "calm",
    "耳语": "whisper",
    "惊讶": "surprised",
    "恐惧": "fearful",
    "厌恶": "disgusted",
    "严肃": "serious",
}

# Accepted spellings (Chinese synonyms and English words) mapped to canonical
# styles. Lookup is case-insensitive and whitespace-insensitive.
STYLE_ALIASES: dict[str, str] = {
    # 开心
    "开心": "开心",
    "高兴": "开心",
    "快乐": "开心",
    "愉快": "开心",
    "兴奋": "开心",
    "激动": "开心",
    "happy": "开心",
    "joy": "开心",
    "joyful": "开心",
    "excited": "开心",
    "cheerful": "开心",
    # 难过
    "难过": "难过",
    "悲伤": "难过",
    "伤心": "难过",
    "低落": "难过",
    "沮丧": "难过",
    "sad": "难过",
    "sadness": "难过",
    "unhappy": "难过",
    "sorrow": "难过",
    # 生气
    "生气": "生气",
    "愤怒": "生气",
    "恼火": "生气",
    "发怒": "生气",
    "angry": "生气",
    "anger": "生气",
    "mad": "生气",
    # 平静
    "平静": "平静",
    "温柔": "平静",
    "冷静": "平静",
    "轻声": "平静",
    "柔和": "平静",
    "放松": "平静",
    "calm": "平静",
    "gentle": "平静",
    "relaxed": "平静",
    "soft": "平静",
    # 耳语
    "耳语": "耳语",
    "悄悄": "耳语",
    "低声": "耳语",
    "whisper": "耳语",
    "whispering": "耳语",
    # 惊讶
    "惊讶": "惊讶",
    "吃惊": "惊讶",
    "震惊": "惊讶",
    "惊奇": "惊讶",
    "惊喜": "惊讶",
    "surprised": "惊讶",
    "surprise": "惊讶",
    "shocked": "惊讶",
    # 恐惧
    "恐惧": "恐惧",
    "害怕": "恐惧",
    "惊恐": "恐惧",
    "恐慌": "恐惧",
    "紧张": "恐惧",
    "fearful": "恐惧",
    "afraid": "恐惧",
    "scared": "恐惧",
    "fear": "恐惧",
    # 厌恶
    "厌恶": "厌恶",
    "恶心": "厌恶",
    "嫌弃": "厌恶",
    "反感": "厌恶",
    "disgusted": "厌恶",
    "disgust": "厌恶",
    # 严肃
    "严肃": "严肃",
    "认真": "严肃",
    "庄重": "严肃",
    "郑重": "严肃",
    "serious": "严肃",
    "solemn": "严肃",
}


def is_question(text: str) -> bool:
    """Whether the sentence is a question (ends in a full- or half-width question mark)."""
    return text.rstrip().rstrip("\"'”’）)」』 ").endswith(("？", "?"))


def _style_words_pattern() -> str:
    # Longest first so "whispering" is not cut short by "whisper".
    words = sorted((re.escape(w) for w in STYLE_ALIASES), key=len, reverse=True)
    return "|".join(words)


# A small model sometimes writes the style as a stage direction inside the reply instead of the
# [style:..] tag -- "耳语：上海是…", "（耳语）你好", "[whisper] hi" -- and the TTS then reads the word
# aloud *and* misses the style. Only an explicit delimiter qualifies (brackets, or a colon at the
# very start), so ordinary prose that merely mentions the word ("耳语是一种说话方式") is untouched.
_OPEN = "（(【\\[「『*＊"
_CLOSE = "）)】\\]」』*＊"
_LEADING_LABEL = re.compile(
    rf"^\s*(?:[{_OPEN}]\s*(?P<bracketed>{_style_words_pattern()})\s*[{_CLOSE}]\s*[：:，,]?\s*"
    rf"|(?P<bare>{_style_words_pattern()})\s*[：:]\s*)",
    re.IGNORECASE,
)
_INLINE_LABEL = re.compile(
    rf"[{_OPEN}]\s*(?P<word>{_style_words_pattern()})\s*[{_CLOSE}]",
    re.IGNORECASE,
)


def _strip_style_labels(text: str) -> tuple[str, str | None]:
    """Remove stage-direction style labels; return the cleaned text and the last style word."""
    style: str | None = None
    match = _LEADING_LABEL.match(text)
    if match:
        style = match.group("bracketed") or match.group("bare")
        text = text[match.end():]
    for inline in _INLINE_LABEL.finditer(text):
        style = inline.group("word")
    text = _INLINE_LABEL.sub("", text)
    return text, style


def strip_style_tags(text: str) -> str:
    """Remove every style tag (and stray style labels), keeping the surrounding text untouched."""
    text = STYLE_TAG_PATTERN.sub("", text)
    return _strip_style_labels(text)[0]


def extract_style_tags(text: str) -> tuple[str, str | None]:
    """Split *text* into ``(text without style tags, last style value or None)``.

    The last tag wins so a later sentence can retune the style mid-response.
    Whitespace left behind by a tag at the very start is trimmed, since the
    stale leading space would otherwise reach text-only clients verbatim.
    A style written as a stage direction instead of a tag ("耳语：…") is stripped
    too, and used as the style when the text has no proper tag.
    """
    matches = list(STYLE_TAG_PATTERN.finditer(text))
    style = matches[-1].group("style").strip() if matches else None
    cleaned = STYLE_TAG_PATTERN.sub("", text)
    if matches and matches[0].start() == 0:
        cleaned = cleaned.lstrip()
    cleaned, label_style = _strip_style_labels(cleaned)
    if label_style is not None:
        cleaned = cleaned.lstrip()
        style = style or label_style
    return cleaned, (style or None)


def normalize_style(raw: str | None) -> str | None:
    """Map an accepted spelling to a canonical style, or ``None`` if unknown."""
    if not raw:
        return None
    key = re.sub(r"\s+", " ", raw).strip().strip("\"'“”").strip().lower()
    if not key:
        return None
    return STYLE_ALIASES.get(key) or STYLE_ALIASES.get(key.replace(" ", ""))


def style_prompt_section() -> str:
    """Describe the tag and its vocabulary for the voice system prompt.

    Kept as a standalone ``##`` section so callers opt into it: only a TTS
    backend that honours styles should pay for the prompt words.

    The worked examples carry most of the weight. Measured through OVMS with
    Qwen3.5-4B, a rule phrased as a permission ("you may tag a sentence") is
    ignored on every turn, while patterned examples are followed on every turn
    and also pick the right emotion -- including the angry case that the plain
    imperative phrasings get wrong.
    """
    vocabulary = ", ".join(CANONICAL_STYLES)
    default = DEFAULT_STYLE
    return (
        "## Voice Style\n"
        f"- Start every spoken reply with one style tag, e.g. [style:{default}].\n"
        f"- Allowed styles (use these exact words only): {vocabulary}.\n"
        f"- The default is {default}. Greetings, questions, commands, facts, explanations, recommendations, "
        f"offers and small talk are always {default}: a warm, friendly or polite tone is still {default}. "
        "Use another style only when the user clearly expresses that emotion or the news itself calls for it "
        "(celebrating good news, consoling bad news).\n"
        "- A follow-up question or offer is always " + default + ", even after an emotional sentence.\n"
        "- If the user asks you to speak a certain way (悄悄地, 小声, 严肃地, 开心地...), choose the matching style "
        "in the tag and then just say the content. Never write a style word such as 耳语 or 平静 in the reply itself.\n"
        "- Examples:\n"
        f"  user: 现在几点了？ -> [style:{default}]现在是下午三点。\n"
        f"  user: 你好。 -> [style:{default}]你好，有什么可以帮你的吗？\n"
        f"  user: 帮我查一下明天的天气。 -> [style:{default}]好的，我这就帮你查。\n"
        f"  user: 你能做什么？ -> [style:{default}]我可以陪你聊天，也能帮你查资料。\n"
        f"  user: 帮我推荐几个适合旅游的地方。 -> [style:{default}]好的，我推荐九寨沟和云南，你更喜欢哪种风景呢？\n"
        f"  user: 周末我想去爬山。 -> [style:{default}]好主意，爬山既能锻炼身体又能看风景，你打算去哪座山？\n"
        "  user: 悄悄地对我说一句话。 -> [style:耳语]好的，我悄悄告诉你，你今天很棒。\n"
        "  user: 我升职了！ -> [style:开心]太好了，恭喜你！\n"
        "  user: 我的猫走了。 -> [style:难过]真让人难过，请节哀。\n"
        "  user: 你怎么这么笨！ -> [style:生气]抱歉让你着急了，我再说一遍。\n"
        "  user: 天哪，我中奖了！ -> [style:惊讶]真的吗？太不可思议了！\n"
        "  user: 外面有奇怪的声音，我好怕。 -> [style:恐惧]别慌，我们先确认一下门窗。\n"
        "  user: 这东西太恶心了。 -> [style:厌恶]确实让人不舒服，快扔掉吧。\n"
        "  user: 请严肃回答，这药有副作用吗？ -> [style:严肃]有，请务必遵医嘱使用。\n"
        "- The tag is removed before playback; never say it aloud."
    )
