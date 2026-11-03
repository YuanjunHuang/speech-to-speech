from speech_to_speech.LLM.tool_intent import force_tool_for_intent

WX = {"type": "function", "function": {"name": "get_weather"}}
SEARCH = {"type": "function", "function": {"name": "web_search"}}


def user(text):
    return [{"role": "system", "content": "s"}, {"role": "user", "content": text}]


def test_weather_question_forces_only_the_weather_tool():
    out = force_tool_for_intent(user("北京明天天气怎么样？"), {"tools": [SEARCH, WX], "tool_choice": "auto"})
    assert out["tools"] == [WX] and out["tool_choice"] == "required"


def test_small_talk_is_untouched():
    kw = {"tools": [WX]}
    assert force_tool_for_intent(user("你好呀"), kw) is kw


def test_not_forced_after_a_tool_result():
    msgs = user("北京天气") + [{"role": "tool", "content": "晴"}]
    kw = {"tools": [WX]}
    assert force_tool_for_intent(msgs, kw) is kw


def test_explicit_tool_choice_and_missing_tool_respected():
    kw = {"tools": [WX], "tool_choice": "none"}
    assert force_tool_for_intent(user("天气"), kw) is kw
    kw2 = {"tools": [SEARCH]}
    assert force_tool_for_intent(user("天气"), kw2) is kw2
    assert force_tool_for_intent(user("天气"), {}) == {}


def test_list_content_parts():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "今天会下雨吗"}]}]
    assert force_tool_for_intent(msgs, {"tools": [WX]})["tool_choice"] == "required"


def test_search_intent_forces_web_search_but_not_small_talk():
    ws = {"type": "function", "function": {"name": "web_search"}}
    assert force_tool_for_intent(user("今天有什么新闻"), {"tools": [ws]})["tool_choice"] == "required"
    assert force_tool_for_intent(user("查一下特斯拉股价"), {"tools": [ws]})["tools"] == [ws]
    kw = {"tools": [ws]}
    assert force_tool_for_intent(user("你好呀"), kw) is kw
