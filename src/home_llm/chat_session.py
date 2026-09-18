"""Conversation state and model-backend coordination."""

from collections.abc import Callable, Iterator

from home_llm.context_budget import ContextBudget, ContextSelection
from home_llm.model_backend import (
    ChatMessage,
    GenerationComplete,
    ModelBackend,
    StreamEvent,
    TextChunk,
)


# Question - is the context always loaded (like in RAM) or is it saved locally on disk ? during runtime
class ChatSession:
    """Maintain one in-memory conversation with a model backend."""

    def __init__(
        self,
        backend: ModelBackend,
        *,
        system_prompt: str | None = None,
        context_budget: ContextBudget | None = None,
    ) -> None:
        self._backend = backend
        self._system_message = self._create_system_message(system_prompt)
        self._messages: list[ChatMessage] = (
            [] if self._system_message is None else [self._system_message]
        )
        self._context_start = 0
        self._context_budget = context_budget
        self._last_context_selection: ContextSelection | None = None

    @classmethod
    def from_messages(
        cls,
        backend: ModelBackend,
        messages: tuple[ChatMessage, ...],
        *,
        context_start: int = 0,
        context_budget: ContextBudget | None = None,
    ) -> ChatSession:
        """Build a session from complete turns, retaining an optional system prompt."""
        messages = tuple(messages)
        for message in messages:
            if (
                not isinstance(message, ChatMessage)
                or not isinstance(message.content, str)
                or not message.content.strip()
            ):
                raise ValueError("Restored history must contain nonempty ChatMessages.")

        start = 1 if messages and messages[0].role == "system" else 0
        turns = messages[start:]
        if len(turns) % 2 or any(
            message.role != ("user" if index % 2 == 0 else "assistant")
            for index, message in enumerate(turns)
        ):
            raise ValueError(
                "Restored history must contain complete user/assistant turns."
            )

        if type(context_start) is not int or not 0 <= context_start <= len(turns) // 2:
            raise ValueError(
                "context_start must be a completed-turn count within the transcript."
            )
        session = cls(
            backend,
            system_prompt=messages[0].content if start else None,
            context_budget=context_budget,
        )
        session._messages = list(messages)
        session._context_start = context_start
        return session

    @property
    def messages(self) -> tuple[ChatMessage, ...]:
        """Return the full transcript, including turns excluded from requests."""
        return tuple(self._messages)

    @property
    def context_start(self) -> int:
        """Return the number of complete turns excluded by explicit clearing."""
        return self._context_start

    @property
    def context_budget(self) -> ContextBudget | None:
        """Return request limits; None keeps legacy unbudgeted API behavior."""
        return self._context_budget

    @property
    def last_context_selection(self) -> ContextSelection | None:
        """Return the last attempted request selection, not proof of model retention."""
        return self._last_context_selection

    @property
    def context_messages(self) -> tuple[ChatMessage, ...]:
        """Return the system prompt and history eligible for model requests."""
        system = () if self._system_message is None else (self._system_message,)
        start = len(system) + 2 * self._context_start
        return (*system, *self._messages[start:])

    def clear(self) -> None:
        """Exclude existing turns from future requests without deleting the transcript."""
        system_count = int(self._system_message is not None)
        turn_messages = len(self._messages) - system_count
        if turn_messages % 2:
            raise ValueError("Cannot clear context while a generation is in progress.")
        self._context_start = turn_messages // 2
        self._last_context_selection = None

    def send(
        self,
        user_text: str,
        *,
        on_context_selected: Callable[[ContextSelection], None] | None = None,
    ) -> Iterator[StreamEvent]:
        """Send one user turn and stream the resulting backend events."""

        if not user_text.strip():
            raise ValueError("The user prompt must not be empty.")

        user_message = ChatMessage(role="user", content=user_text)
        self._last_context_selection = None
        request_messages = (*self.context_messages, user_message)
        if self._context_budget is not None:
            selection = self._context_budget.select(self.context_messages, user_message)
            request_messages = selection.messages
            self._last_context_selection = selection
            if on_context_selected is not None:
                on_context_selected(selection)

        self._messages.append(user_message)

        assistant_fragments: list[str] = []
        completed = False

        try:
            for event in self._backend.stream_chat(request_messages):
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
