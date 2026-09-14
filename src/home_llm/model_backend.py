"""Shared types used by model backends and user interfaces."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

type MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a model conversation."""

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(
                f"Invalid message role {self.role!r}; "
                "expected 'system', 'user', or 'assistant'."
            )

        if not self.content:
            raise ValueError("Message content must not be empty.")


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A fragment of generated assistant text."""

    text: str


@dataclass(frozen=True, slots=True)
class GenerationComplete:
    """Final generation statistics returned by a backend."""

    prompt_tokens: int
    generated_tokens: int
    prompt_duration_ns: int
    generation_duration_ns: int
    total_duration_ns: int
    load_duration_ns: int = 0

    @property
    def prompt_tokens_per_second(self) -> float:
        duration_seconds = self.prompt_duration_ns / 1_000_000_000
        if duration_seconds == 0:
            return 0.0
        return self.prompt_tokens / duration_seconds

    @property
    def generation_tokens_per_second(self) -> float:
        duration_seconds = self.generation_duration_ns / 1_000_000_000
        if duration_seconds == 0:
            return 0.0
        return self.generated_tokens / duration_seconds


type StreamEvent = TextChunk | GenerationComplete


class ModelBackend(Protocol):
    """Interface implemented by every model runtime backend."""

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
    ) -> Iterator[StreamEvent]:
        """Generate streaming events for a conversation."""
