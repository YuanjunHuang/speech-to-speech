"""Tests for the model-agnostic spoken-style channel."""

from queue import Queue
from threading import Event

import pytest
from openai.types.realtime import RealtimeSessionCreateRequest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.language_model import BaseLanguageModelHandler
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt
from speech_to_speech.pipeline.events import AssistantOutputEvent
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    EndOfResponse,
    LLMResponseChunk,
    TTSInput,
)
from speech_to_speech.pipeline.style import (
    CANONICAL_STYLES,
    extract_style_tags,
    normalize_style,
    strip_style_tags,
    style_prompt_section,
)
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler
from speech_to_speech.TTS.style_instructs import instruct_for_style

# ── Tag parsing ───────────────────────────────────────────────────────


def test_extract_style_tags_splits_leading_tag():
    assert extract_style_tags("[style:开心]你好。") == ("你好。", "开心")


def test_extract_style_tags_accepts_full_width_colon_and_spacing():
    assert extract_style_tags("[style：开心]你好。") == ("你好。", "开心")
    assert extract_style_tags("[ style : 开心 ] 你好。") == ("你好。", "开心")
    assert extract_style_tags("[STYLE:happy]你好。") == ("你好。", "happy")


def test_extract_style_tags_accepts_the_chinese_spelling():
    # Small models sometimes mirror the instruction in their own words.
    assert extract_style_tags("[风格:平静]你好。") == ("你好。", "平静")
    assert extract_style_tags("[风格：难过]你好。") == ("你好。", "难过")


def test_extract_style_tags_without_tag_is_a_passthrough():
    assert extract_style_tags("你好。") == ("你好。", None)


def test_extract_style_tags_keeps_the_last_tag():
    assert extract_style_tags("[style:开心]你好。[style:难过]再见。") == ("你好。再见。", "难过")


def test_extract_style_tags_only_trims_whitespace_left_by_a_leading_tag():
    # A mid-sentence tag keeps the surrounding spacing.
    assert extract_style_tags("你好。[style:开心] 再见。") == ("你好。 再见。", "开心")


def test_extract_style_tags_ignores_other_brackets():
    assert extract_style_tags("[1] 你好。") == ("[1] 你好。", None)
    assert extract_style_tags("参见 [style guide] 一节") == ("参见 [style guide] 一节", None)


def test_extract_style_tags_leaves_an_unterminated_tag_alone():
    assert extract_style_tags("[style:开心") == ("[style:开心", None)


def test_extract_style_tags_drops_a_malformed_tag_without_a_style():
    assert extract_style_tags("[style:  ]你好。") == ("你好。", None)


def test_strip_style_tags_is_idempotent():
    once = strip_style_tags("[style:开心]你好。[style:难过]")
    assert once == "你好。"
    assert strip_style_tags(once) == once


# ── Vocabulary ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("开心", "开心"),
        ("高兴", "开心"),
        ("happy", "开心"),
        ("HAPPY", "开心"),
        (" whisper ", "耳语"),
        ("calm", "平静"),
        ("悲伤", "难过"),
        ("angry", "生气"),
    ],
)
def test_normalize_style_maps_aliases_to_canonical_styles(raw, expected):
    assert normalize_style(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "这个词不在表里"])
def test_normalize_style_rejects_unknown_values(raw):
    assert normalize_style(raw) is None


def test_style_prompt_section_advertises_every_canonical_style():
    section = style_prompt_section()
    assert section.startswith("## Voice Style")
    assert "[style:开心]" in section
    for style in CANONICAL_STYLES:
        assert style in section
    # Worked examples, not prose, are what make a small model follow the protocol.
    assert section.count("->") >= 3


def test_style_prompt_defaults_to_calm_and_shows_neutral_examples_first():
    section = style_prompt_section()
    assert "The default is 平静" in section
    examples = [line for line in section.splitlines() if "->" in line]
    # A small model copies the example pattern: the opening ones must be neutral,
    # otherwise greetings and plain questions get tagged as emotions.
    assert all("[style:平静]" in line for line in examples[:4])
    assert any("[style:惊讶]" in line for line in examples)


def test_voice_prompt_only_advertises_styles_when_enabled():
    default = build_voice_system_prompt("Be concise.")
    enabled = build_voice_system_prompt("Be concise.", enable_style_tags=True)

    # The default voice rules keep their word budget; the style section is opt-in.
    assert "[style:" not in default
    assert "## Voice Style" in enabled
    for style in CANONICAL_STYLES:
        assert style in enabled


def test_instruct_for_style_uses_the_verified_phrasing():
    assert instruct_for_style("开心") == "用非常兴奋、开心的语气说这句话，语调上扬，语速稍快，充满活力。"
    # An alias resolves to the same verified phrasing as its canonical style.
    assert instruct_for_style("happy") == instruct_for_style("开心")


def test_instruct_for_style_templates_an_unknown_style():
    assert instruct_for_style("疲惫") == "用疲惫的语气说这句话。"


@pytest.mark.parametrize("style", [None, "", "   "])
def test_instruct_for_style_returns_none_without_a_style(style):
    assert instruct_for_style(style) is None


# ── TTSInput plumbing ─────────────────────────────────────────────────


def test_tts_input_style_defaults_to_none():
    assert TTSInput(text="你好。").style is None


# ── LMOutputProcessor: strip, forward, stick ──────────────────────────


def _processor() -> LMOutputProcessor:
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup()
    return processor


def test_processor_strips_the_tag_from_client_text_and_forwards_the_style():
    outputs = list(_processor().process(LLMResponseChunk(text="[style:开心]你好。", response_key="r1")))

    assert [type(output) for output in outputs] == [AssistantOutputEvent, TTSInput]
    event, tts_input = outputs
    assert event.parts == [AssistantTextPart(text="你好。")]
    assert event.text == "你好。"
    assert tts_input.text == "你好。"
    assert tts_input.style == "开心"


def test_processor_falls_back_to_calm_for_a_tag_outside_the_vocabulary():
    outputs = list(_processor().process(LLMResponseChunk(text="[style:抱歉]对不起。", response_key="r1")))

    event, tts_input = outputs
    assert event.text == "对不起。"
    assert tts_input.style == "平静"


def test_processor_style_sticks_for_the_rest_of_the_response():
    processor = _processor()
    first = list(processor.process(LLMResponseChunk(text="[style:难过]你好。", response_key="r1")))
    second = list(processor.process(LLMResponseChunk(text="再见。", response_key="r1")))

    assert first[1].style == "难过"
    assert second[1].text == "再见。"
    assert second[1].style == "难过"


def test_processor_inherited_emotion_does_not_carry_onto_a_follow_up_question():
    processor = _processor()
    first = list(processor.process(LLMResponseChunk(text="[style:难过]真让人难过，请节哀。", response_key="r1")))
    question = list(processor.process(LLMResponseChunk(text="要不要聊聊具体的情况？", response_key="r1")))
    statement = list(processor.process(LLMResponseChunk(text="我会一直陪着你。", response_key="r1")))

    assert first[1].style == "难过"
    assert question[1].style == "平静"
    # Only questions revert; the emotion still applies to the next statement.
    assert statement[1].style == "难过"


def test_processor_keeps_a_style_the_model_put_on_the_question_itself():
    processor = _processor()
    list(processor.process(LLMResponseChunk(text="[style:开心]太好了！", response_key="r1")))
    question = list(processor.process(LLMResponseChunk(text="[style:惊讶]真的吗？", response_key="r1")))

    assert question[1].style == "惊讶"


def test_processor_client_pin_still_applies_to_questions():
    outputs = list(_processor().process(_chunk("你想去哪座山？", session_style="开心")))

    assert outputs[1].style == "开心"


def test_is_question_handles_full_and_half_width_marks_and_closing_quotes():
    from speech_to_speech.pipeline.style import is_question

    assert is_question("你想去哪里？")
    assert is_question("Where to?")
    assert is_question("你说“真的吗？”")
    assert not is_question("我陪你去！")
    assert not is_question("好的。")


def test_processor_style_is_cleared_between_responses():
    processor = _processor()
    list(processor.process(LLMResponseChunk(text="[style:生气]你好。", response_key="r1")))
    list(processor.process(EndOfResponse(response_key="r1")))

    outputs = list(processor.process(LLMResponseChunk(text="你好。", response_key="r2")))
    assert outputs[1].style is None


def test_processor_leaves_untagged_text_untouched_without_a_style():
    outputs = list(_processor().process(LLMResponseChunk(text="你好。", response_key="r1")))
    assert outputs[0].parts == [AssistantTextPart(text="你好。")]
    assert outputs[1].style is None


def _chunk(text: str, *, session_style: str | None = None) -> LLMResponseChunk:
    chunk = LLMResponseChunk(text=text, response_key="r1")
    if session_style is not None:
        chunk.runtime_config = _runtime_config(session_style)
    return chunk


def _runtime_config(session_style: str) -> RuntimeConfig:
    config = RuntimeConfig()
    config.session = RealtimeSessionCreateRequest(type="realtime", style=session_style)
    return config


def test_processor_uses_a_client_pinned_session_style():
    outputs = list(_processor().process(_chunk("你好。", session_style="开心")))

    assert outputs[0].parts == [AssistantTextPart(text="你好。")]
    assert outputs[1].text == "你好。"
    assert outputs[1].style == "开心"


def test_processor_prefers_the_client_pin_over_the_model_tag_but_still_strips_it():
    outputs = list(_processor().process(_chunk("[style:难过]你好。", session_style="开心")))

    assert outputs[0].parts == [AssistantTextPart(text="你好。")]
    assert outputs[1].style == "开心"


def test_processor_accepts_a_pinned_session_alias():
    outputs = list(_processor().process(_chunk("你好。", session_style="happy")))
    assert outputs[1].style == "开心"


def test_processor_ignores_an_unknown_pinned_session_style():
    outputs = list(_processor().process(_chunk("[style:难过]你好。", session_style="auto")))
    assert outputs[1].style == "难过"


# ── History keeps only what was voiced ────────────────────────────────


def test_history_items_keep_only_the_voiced_text():
    items = BaseLanguageModelHandler._ordered_output_items(
        [
            AssistantTextPart(text="[style:开心]你好。"),
            AssistantTextPart(text="[style:难过]再见。"),
        ],
        wants_audio=True,
    )
    assert [item.content[0].text for item in items] == ["你好。 再见。"]


def test_openai_compatible_history_items_keep_only_the_voiced_text():
    from openai.types.realtime.realtime_conversation_item_assistant_message import Content

    from speech_to_speech.LLM.base_openai_compatible_language_model import AssistantMessage

    message = AssistantMessage(content=[Content(type="output_text", text="[style:开心]你好。")])
    item = message.to_chat_item()
    assert item.content[0].text == "你好。"


# ── Qwen3-TTS handler wiring ──────────────────────────────────────────


def _handler(**attrs) -> Qwen3TTSHandler:
    handler = object.__new__(Qwen3TTSHandler)
    handler.queue_in = Queue()
    handler.queue_out = Queue()
    handler.stop_event = Event()
    for key, value in attrs.items():
        setattr(handler, key, value)
    return handler


def test_handler_prefers_the_per_utterance_style_over_the_configured_instruct():
    handler = _handler(instruct="用固定的语气说。")
    assert handler._instruct_for_style(None) == "用固定的语气说。"
    assert handler._instruct_for_style("开心") == instruct_for_style("开心")


def test_handler_falls_back_to_the_configured_instruct_without_one():
    handler = _handler(instruct=None)
    assert handler._instruct_for_style("开心") == instruct_for_style("开心")
    assert handler._instruct_for_style(None) is None


def test_llm_request_history_gets_the_default_style_tag_without_touching_stored_history():
    from speech_to_speech.LLM.chat_completions_language_model import _with_default_style_tags

    original = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "你好。"},
        {"role": "assistant", "content": "你好，有什么可以帮你的吗？"},
        {"role": "assistant", "content": "[style:开心]太好了！"},
        {"role": "assistant", "content": "  "},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
    ]
    tagged = _with_default_style_tags(original)

    assert tagged[2]["content"] == "[style:平静]你好，有什么可以帮你的吗？"
    assert tagged[3]["content"] == "[style:开心]太好了！"  # an existing tag is kept
    assert tagged[1]["content"] == "你好。"  # only assistant text is touched
    assert tagged[4]["content"] == "  " and tagged[5]["content"] is None
    assert original[2]["content"] == "你好，有什么可以帮你的吗？"  # the input list is not mutated


@pytest.mark.parametrize(
    ("raw", "cleaned", "style"),
    [
        ("耳语：上海是一个繁华的大都市。", "上海是一个繁华的大都市。", "耳语"),
        ("（耳语）你好，我悄悄告诉你。", "你好，我悄悄告诉你。", "耳语"),
        ("[耳语] 你好", "你好", "耳语"),
        ("Whisper: hello there", "hello there", "Whisper"),
        ("这是（耳语）一个测试", "这是一个测试", "耳语"),
        ("[style:开心]耳语：你好", "你好", "开心"),  # a proper tag wins over a stray label
    ],
)
def test_a_style_written_as_a_stage_direction_is_stripped_and_used(raw, cleaned, style):
    assert extract_style_tags(raw) == (cleaned, style)


@pytest.mark.parametrize("text", ["耳语是一种说话方式。", "他说：耳语很好听。", "我们可以耳语交流。", "你好，今天过得怎么样？"])
def test_ordinary_prose_that_mentions_a_style_word_is_untouched(text):
    assert extract_style_tags(text) == (text, None)
