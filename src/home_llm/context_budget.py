"""Approximate, backend-independent selection of complete recent chat turns."""

from dataclasses import dataclass

from home_llm.model_backend import ChatMessage


def estimate_message_tokens(message: ChatMessage) -> int:
    """Estimate UTF-8 text at three bytes/token plus eight framing tokens.

    This heuristic is not a tokenizer or an upper bound. Different languages,
    tokenizers, and runtime templates can require more or fewer actual tokens.
    """
    return (len(message.content.encode("utf-8")) + 2) // 3 + 8


@dataclass(frozen=True, slots=True)
class ContextSelection:
    """Messages selected for one request and estimated input usage."""

    messages: tuple[ChatMessage, ...]
    estimated_input_tokens: int
    input_budget: int
    selected_turns: int
    omitted_turns: int


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Reserve generated output and extra template/safety space before selection."""

    context_length: int
    max_output_tokens: int
    safety_margin: int = 128

    def __post_init__(self) -> None:
        for name in ("context_length", "max_output_tokens", "safety_margin"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_output_tokens >= self.context_length:
            raise ValueError("max_output_tokens must be smaller than context_length.")

    @property
    def input_budget(self) -> int:
        """Return input allowance after output and template/safety reserves."""
        return self.context_length - self.max_output_tokens - self.safety_margin

    def select(
        self,
        history: tuple[ChatMessage, ...],
        current_prompt: ChatMessage,
    ) -> ContextSelection:
        """Keep required messages and the largest fitting suffix of complete turns.

        Input is eligible history (optional system message, then complete turns)
        and one new user prompt. Output never mutates that history. Stop at the
        first turn that does not fit rather than skipping over it to older turns.
        """
        if current_prompt.role != "user" or not current_prompt.content.strip():
            raise ValueError("The current prompt must be a nonempty user message.")
        system_count = int(bool(history) and history[0].role == "system")
        system = history[:system_count]
        turns = history[system_count:]
        if len(turns) % 2 or any(
            message.role != ("user" if index % 2 == 0 else "assistant")
            for index, message in enumerate(turns)
        ):
            raise ValueError(
                "Context history must contain complete user/assistant turns."
            )

        estimated_tokens = sum(
            estimate_message_tokens(message) for message in (*system, current_prompt)
        )
        if estimated_tokens > self.input_budget:
            raise ValueError(
                "The current prompt and system message need an estimated "
                f"{estimated_tokens} input tokens, but only {self.input_budget} remain "
                f"after reserving {self.max_output_tokens} output tokens and "
                f"{self.safety_margin} template/safety tokens. Shorten the prompt or "
                "system message, reduce the output limit, or increase context length "
                "within your model and hardware limits. No request was sent."
            )

        first_selected = len(turns)
        for index in range(len(turns) - 2, -1, -2):
            turn_cost = sum(
                estimate_message_tokens(message) for message in turns[index : index + 2]
            )
            if estimated_tokens + turn_cost > self.input_budget:
                break
            estimated_tokens += turn_cost
            first_selected = index

        return ContextSelection(
            messages=(*system, *turns[first_selected:], current_prompt),
            estimated_input_tokens=estimated_tokens,
            input_budget=self.input_budget,
            selected_turns=(len(turns) - first_selected) // 2,
            omitted_turns=first_selected // 2,
        )
