"""
LLM Output Processor

Intercepts LLM output to:
1. Preserve text and tool events in model order
2. Forward clean text to TTS in the same queue
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from queue import Queue
from uuid import uuid4

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline import language_detection
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    AssistantToolCallReadyEvent,
    PipelineEvent,
    ResponseFailedEvent,
    ResponseGenerationDoneEvent,
    TokenUsageEvent,
)
from speech_to_speech.pipeline.handler_types import LLMOut, TTSIn
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    AssistantToolCallPart,
    EndOfResponse,
    LLMResponseChunk,
    TokenUsage,
    TTSInput,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.style import DEFAULT_STYLE, extract_style_tags, is_question, normalize_style
from speech_to_speech.pipeline.transcript_logging import transcript_for_log
from speech_to_speech.utils.utils import response_wants_audio

logger = logging.getLogger(__name__)


class LMOutputProcessor(BaseHandler[LLMOut, TTSIn | PipelineEvent]):
    """
    Places ordered output events and their TTS inputs on one queue.

    Input: :class:`LLMResponseChunk`, :class:`TokenUsage`, or :class:`EndOfResponse` from LLM
    Output: assistant and usage events, :class:`TTSInput`, or :class:`EndOfResponse`
    on the same ordered path through TTS.
    """

    def setup(
        self,
        speculative_turns: SpeculativeTurnTracker | None = None,
        text_output_queue: Queue[PipelineEvent] | None = None,
        detect_llm_output_language: bool = False,
    ) -> None:
        self.speculative_turns = speculative_turns
        self.text_output_queue = text_output_queue
        self.detect_llm_output_language = detect_llm_output_language
        self._language_detector = language_detection.warm_language_detector() if detect_llm_output_language else None
        self._response_key: str | None = None
        self._tool_call_ids: list[str] = []
        self._output_sequence = 0
        self._detected_assistant_language: str | None = None
        self._tts_runtime_config: RuntimeConfig | None = None
        self._response_selected_language: str | None = None
        self._response_tts_language: str | None = None
        self._response_language_resolved = False
        self._assistant_language_probe = ""
        self._response_style: str | None = None
        self._voiced_style: str | None = None

    def _start_response(self, response_key: str | None) -> str:
        key = response_key or self._response_key or uuid4().hex
        self._response_key = key
        return key

    def _reset_response(self) -> None:
        self._response_key = None
        self._tool_call_ids = []
        self._output_sequence = 0
        self._detected_assistant_language = None
        self._tts_runtime_config = None
        self._response_selected_language = None
        self._response_tts_language = None
        self._response_language_resolved = False
        self._assistant_language_probe = ""
        self._response_style = None
        self._voiced_style = None

    @staticmethod
    def _session_style(config: RuntimeConfig | None) -> str | None:
        """Read a client-pinned spoken style off the Realtime session, if any.

        ``session.update`` carries a full session object whose request type
        accepts extra fields, so a client can pin the delivery without the
        language model having a say::

            {"type": "session.update", "session": {"style": "开心"}}

        Anything unrecognised (including ``"auto"``) counts as unpinned, which
        leaves the model's own inline tags in charge.
        """
        if config is None:
            return None
        raw = getattr(config.session, "style", None)
        if not isinstance(raw, str):
            return None
        return normalize_style(raw)

    def _observe_assistant_language(self, text: str) -> None:
        if self._detected_assistant_language is not None:
            return
        # A few short streamed parts can make one detectable sentence. Keep only
        # a bounded window for this response.
        self._assistant_language_probe = (self._assistant_language_probe + " " + text).strip()[-256:]
        # The assistant can answer outside Parakeet's recognition languages.
        if self._language_detector is None:
            self._language_detector = language_detection.warm_language_detector()
        try:
            self._detected_assistant_language = language_detection.detect_language_from_text(
                self._assistant_language_probe,
                self._language_detector,
                minimum_confidence_gap=language_detection.MIN_ASSISTANT_CONFIDENCE_GAP,
                allow_short_cjk=True,
            )
        except Exception:
            logger.exception("Assistant language detection failed; using prior assistant language")

    def _notify_generation_done(
        self,
        lm_output: EndOfResponse,
        response_key: str | None,
        *,
        succeeded: bool,
    ) -> None:
        if self.text_output_queue is None:
            return
        self.text_output_queue.put(
            ResponseGenerationDoneEvent(
                response_key=response_key,
                call_ids=list(self._tool_call_ids),
                history=lm_output.history,
                succeeded=succeeded,
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                cancel_generation=lm_output.cancel_generation,
            )
        )

    def _turn_output_allowed(self, turn_id: str | None, turn_revision: int | None) -> bool:
        if self.speculative_turns is None:
            return True
        return self.speculative_turns.wait_for_gate(turn_id, turn_revision)

    def process(self, lm_output: LLMOut) -> Iterator[TTSIn | PipelineEvent]:
        """
        Forward response events and audio inputs in their original order.

        Yields:
            Response events, :class:`TTSInput`, or :class:`EndOfResponse`
        """
        if isinstance(lm_output, TokenUsage):
            usage_response_key = self._start_response(lm_output.response_key)
            usage_event = TokenUsageEvent(
                input_tokens=lm_output.input_tokens or 0,
                output_tokens=lm_output.output_tokens or 0,
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                cancel_generation=lm_output.cancel_generation,
                response_key=usage_response_key,
            )
            yield usage_event
            return

        if isinstance(lm_output, EndOfResponse):
            response_key = lm_output.response_key or self._response_key
            if not self._turn_output_allowed(
                lm_output.turn_id,
                lm_output.turn_revision,
            ):
                logger.debug(
                    "Dropping stale end-of-response for turn=%s rev=%s",
                    lm_output.turn_id,
                    lm_output.turn_revision,
                )
                self._notify_generation_done(lm_output, response_key, succeeded=False)
                self._reset_response()
                if response_key is not None:
                    # Bypass downstream speculative gates with a lifecycle-only
                    # terminal. The router uses its key to cancel an opened stale
                    # response or clear only that queued response.
                    yield EndOfResponse(
                        cancel_generation=lm_output.cancel_generation,
                        response_key=response_key,
                        cleanup_only=True,
                    )
                return
            succeeded = lm_output.error is None and lm_output.status == "completed"
            self._notify_generation_done(lm_output, response_key, succeeded=succeeded)
            if succeeded and self._detected_assistant_language and self._tts_runtime_config is not None:
                self._tts_runtime_config.last_assistant_language = self._detected_assistant_language
            if lm_output.error:
                yield ResponseFailedEvent(
                    history=lm_output.history,
                    message=lm_output.error,
                    turn_id=lm_output.turn_id,
                    turn_revision=lm_output.turn_revision,
                    cancel_generation=lm_output.cancel_generation,
                    response_key=response_key,
                )
            else:
                yield AssistantResponseDoneEvent(
                    history=lm_output.history,
                    status=lm_output.status,
                    reason=lm_output.reason,
                    response_key=response_key,
                    input_tool_call_ids=lm_output.input_tool_call_ids if succeeded else [],
                    turn_id=lm_output.turn_id,
                    turn_revision=lm_output.turn_revision,
                    cancel_generation=lm_output.cancel_generation,
                )
            yield EndOfResponse(
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                cancel_generation=lm_output.cancel_generation,
                response_key=response_key,
            )
            self._reset_response()
            return

        if not isinstance(lm_output, LLMResponseChunk):
            logger.warning("LMOutputProcessor received unexpected type: %s", type(lm_output))
            return

        if not self._turn_output_allowed(
            lm_output.turn_id,
            lm_output.turn_revision,
        ):
            logger.debug("Dropping stale LLM chunk for turn=%s rev=%s", lm_output.turn_id, lm_output.turn_revision)
            return

        logger.debug("LM processor: parts=%s", transcript_for_log(lm_output.parts))

        response_key = self._start_response(lm_output.response_key)
        config = lm_output.runtime_config

        for part in lm_output.parts:
            output_sequence = self._output_sequence
            self._output_sequence += 1
            if isinstance(part, AssistantToolCallPart) and part.tool.call_id not in self._tool_call_ids:
                self._tool_call_ids.append(part.tool.call_id)
                if self.text_output_queue is not None:
                    self.text_output_queue.put(
                        AssistantToolCallReadyEvent(
                            history=lm_output.history,
                            part=part,
                            output_sequence=output_sequence,
                            turn_id=lm_output.turn_id,
                            turn_revision=lm_output.turn_revision,
                            cancel_generation=lm_output.cancel_generation,
                            response_key=response_key,
                        )
                    )
            if isinstance(part, AssistantTextPart):
                # The style channel rides in the text, so split it out here:
                # everything downstream (client transcript, history via the LLM
                # layer, TTS) sees the clean text, and TTS additionally gets the
                # bare style value. The style sticks for the rest of the
                # response so only the first sentence needs a tag.
                cleaned_text, tagged_style = extract_style_tags(part.text)
                style_before = self._response_style
                pinned_style = self._session_style(config)
                if pinned_style is not None:
                    # A client-pinned style is an explicit request, so it
                    # outranks whatever mood the model read into the turn.
                    self._response_style = pinned_style
                elif tagged_style is not None:
                    # A small model sometimes invents a tag ("抱歉", "温柔"). Passing
                    # it through verbatim made the TTS act out the word, so anything
                    # outside the vocabulary falls back to the neutral default.
                    canonical = normalize_style(tagged_style)
                    if canonical is None:
                        logger.info("Unknown style tag %r from the model; using %s", tagged_style, DEFAULT_STYLE)
                    self._response_style = canonical or DEFAULT_STYLE
                voiced_style = self._response_style
                if (
                    pinned_style is None
                    and tagged_style is None
                    and voiced_style not in (None, DEFAULT_STYLE)
                    and is_question(cleaned_text)
                ):
                    # The emotion came from an earlier sentence of this reply. A follow-up
                    # question ("要不要我帮你看看？") in the same heightened voice sounds
                    # wrong, so questions fall back to the neutral default. A tag on the
                    # question itself, or a client-pinned style, is still honoured.
                    voiced_style = DEFAULT_STYLE
                self._voiced_style = voiced_style
                logger.info(
                    "Style: tag=%r pinned=%r inherited=%r -> using %r for %r",
                    tagged_style, pinned_style, style_before, voiced_style, cleaned_text[:16],
                )
                if cleaned_text != part.text:
                    part = AssistantTextPart(text=cleaned_text)
            event = AssistantOutputEvent(
                history=lm_output.history,
                parts=[part],
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                cancel_generation=lm_output.cancel_generation,
                response_key=response_key,
                output_sequence=output_sequence,
            )
            yield event
            if (
                not isinstance(part, AssistantTextPart)
                or not part.text.strip()
                or not response_wants_audio(lm_output.response)
            ):
                continue
            logger.debug("Forwarding to TTS: %s", transcript_for_log(part.text))
            if self.detect_llm_output_language or config is not None:
                self._observe_assistant_language(part.text)
            self._tts_runtime_config = config
            language_code = lm_output.language_code
            if self.detect_llm_output_language:
                language_code = self._detected_assistant_language
            if not self._response_language_resolved:
                self._response_selected_language = (
                    lm_output.selected_language
                    if "selected_language" in lm_output.model_fields_set
                    else config.selected_language
                    if config is not None
                    else None
                )
                selected = self._response_selected_language
                if selected == "auto" or (selected is None and self.detect_llm_output_language):
                    self._response_tts_language = self._detected_assistant_language or (
                        config.last_assistant_language if config is not None else None
                    )
                elif selected is not None:
                    # STT keeps the named selection. TTS follows a confident
                    # detection of this response's first spoken text, and
                    # otherwise the selection. TTS keeps the selection for a
                    # detected language it cannot speak.
                    detected = self._detected_assistant_language if self.detect_llm_output_language else None
                    self._response_tts_language = detected or selected
                self._response_language_resolved = True
            selected = self._response_selected_language
            yield TTSInput(
                text=part.text,
                style=self._voiced_style,
                language_code=language_code,
                selected_language=selected,
                tts_language_code=self._response_tts_language if selected is not None else language_code,
                response_assistant_language_code=(
                    self._response_tts_language if selected is None and self.detect_llm_output_language else None
                ),
                runtime_config=lm_output.runtime_config,
                response=lm_output.response,
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                speech_stopped_at_s=lm_output.speech_stopped_at_s,
                cancel_generation=lm_output.cancel_generation,
                response_key=response_key,
                prefetch_transaction=lm_output.prefetch_transaction,
            )

    def on_session_end(self) -> None:
        self._reset_response()
