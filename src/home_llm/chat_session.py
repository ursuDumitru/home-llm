"""Conversation state and model-backend coordination."""

from collections.abc import Iterator

from home_llm.model_backend import (
    ChatMessage,
    GenerationComplete,
    ModelBackend,
    StreamEvent,
    TextChunk,
)


class ChatSession:
    """Maintain one in-memory conversation with a model backend."""

    def __init__(
        self,
        backend: ModelBackend,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self._backend = backend
        self._system_message = self._create_system_message(system_prompt)
        self._messages: list[ChatMessage] = []
        self.clear()

    @property
    def messages(self) -> tuple[ChatMessage, ...]:
        """Return an immutable snapshot of the conversation."""

        return tuple(self._messages)

    def clear(self) -> None:
        """Remove conversation history while preserving the system prompt."""

        self._messages.clear()

        if self._system_message is not None:
            self._messages.append(self._system_message)

    def send(self, user_text: str) -> Iterator[StreamEvent]:
        """Send one user turn and stream the resulting backend events."""

        if not user_text.strip():
            raise ValueError("The user prompt must not be empty.")

        user_message = ChatMessage(role="user", content=user_text)
        self._messages.append(user_message)

        assistant_fragments: list[str] = []
        completed = False

        try:
            for event in self._backend.stream_chat(tuple(self._messages)):
                if isinstance(event, TextChunk):
                    assistant_fragments.append(event.text)
                    yield event
                    continue

                if isinstance(event, GenerationComplete):
                    if not assistant_fragments:
                        raise RuntimeError(
                            "The model completed without returning assistant text."
                        )

                    assistant_message = ChatMessage(
                        role="assistant",
                        content="".join(assistant_fragments),
                    )
                    self._messages.append(assistant_message)
                    completed = True

                    yield event
                    return

                raise TypeError(
                    "The model backend returned an unsupported event: "
                    f"{type(event).__name__}."
                )

            raise RuntimeError(
                "The model backend stopped without completing the response."
            )

        finally:
            if not completed and self._messages and self._messages[-1] is user_message:
                # Do not leave a failed or cancelled turn in the history.
                self._messages.pop()

    @staticmethod
    def _create_system_message(
        system_prompt: str | None,
    ) -> ChatMessage | None:
        if system_prompt is None:
            return None

        if not system_prompt.strip():
            raise ValueError("The system prompt must not be empty.")

        return ChatMessage(role="system", content=system_prompt)
