from dataclasses import dataclass, field


@dataclass
class LanguageModelBaseArguments:
    model_name: str = field(
        default="Qwen/Qwen3.5-4B",
        metadata={"help": "The pretrained language model to use."},
    )
    user_role: str = field(
        default="user",
        metadata={"help": "Role assigned to the user in the chat context. Default is 'user'."},
    )
    init_chat_role: str = field(
        default="system",
        metadata={"help": "Initial role for setting up the chat context. Default is 'system'."},
    )
    init_chat_prompt: str = field(
        default="You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words.",
        metadata={"help": "The initial chat prompt to establish context for the language model."},
    )
    chat_size: int = field(
        default=30,
        metadata={"help": "Number of interactions assistant-user to keep for the chat."},
    )
    stream_batch_sentences: int = field(
        default=3,
        metadata={
            "help": "Number of sentences to accumulate before yielding a batch during streaming. "
            "Set to 1 for sentence-by-sentence streaming. Default is 3."
        },
    )
    stream_batch_schedule: str = field(
        default="",
        metadata={
            "help": "Comma-separated sentence counts for the first batches of a response, e.g. '1,2' sends the "
            "first sentence alone, then the next two together, then falls back to stream_batch_sentences. "
            "A short first batch lowers time-to-first-audio for slow TTS backends. Empty = no schedule."
        },
    )
    stream_batch_first_min_chars: int = field(
        default=0,
        metadata={
            "help": "Minimum characters in the first batch of a response. A very short opening sentence ('好的。') "
            "plays out before the next batch is rendered, leaving an audible gap, so keep accumulating "
            "sentences until this many characters are collected. 0 disables the check."
        },
    )
    enable_lang_prompt: bool = field(
        default=False,
        metadata={
            "help": "When True, add a language instruction to the system prompt for the current turn "
            "(e.g. 'Please reply to my message in French.'). Default is False."
        },
    )
    enable_style_tags: bool = field(
        default=False,
        metadata={
            "help": "When True, teach the model the inline [style:...] tag that steers TTS delivery "
            "(e.g. happy, sad, angry, calm, whisper). Only meaningful with a TTS backend that honours "
            "styles, such as Qwen3-TTS custom voice. Default is False."
        },
    )
    compact_history: bool = field(
        default=True,
        metadata={
            "help": "When True, summarize older turns in the background once the chat exceeds chat_size, "
            "instead of synchronously evicting them. Adds an extra LLM call per compaction. Default is True."
        },
    )
