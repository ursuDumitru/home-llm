"""Terminal rendering for the interactive HomeLLM interface."""

import os
import sys
from typing import TextIO

from rich.console import Console
from rich.text import Text

from home_llm.model_backend import (
    GenerationComplete,
    StreamEvent,
    TextChunk,
)


class ConsoleRenderer:
    """Render model events without knowing which backend produced them."""

    _SEPARATOR = "─" * 72

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        use_color: bool | None = None,
    ) -> None:
        output_stream = stream if stream is not None else sys.stdout

        if use_color is None:
            use_color = output_stream.isatty() and "NO_COLOR" not in os.environ

        self._console = Console(
            file=output_stream,
            color_system="auto" if use_color else None,
            force_terminal=use_color,
            no_color=not use_color,
            highlight=False,
            markup=False,
        )

    def show_prompt(self) -> None:
        """Display the prompt before the CLI reads user input."""

        prompt = Text("PROMPT > ", style="bold cyan")
        self._console.print(prompt, end="")

    def begin_answer(self) -> None:
        """Print the heading displayed before streamed model output."""

        self._console.print()
        self._console.print(Text("ANSWER", style="bold green"))

    def render_event(self, event: StreamEvent) -> None:
        """Render one backend-independent streaming event."""

        if isinstance(event, TextChunk):
            # Model text is treated as plain text, not Rich markup.
            self._console.print(
                event.text,
                end="",
                markup=False,
                highlight=False,
                soft_wrap=True,
            )
            return

        if isinstance(event, GenerationComplete):
            self._render_completion(event)
            return

        raise TypeError(f"Unsupported stream event type: {type(event).__name__}")

    def show_error(self, message: str) -> None:
        """Display an actionable error without terminating the process."""

        error_text = Text()
        error_text.append("ERROR", style="bold red")
        error_text.append(f": {message}")

        self._console.print()
        self._console.print(error_text)

    def show_separator(self) -> None:
        """Display a visual boundary between conversation turns."""

        self._console.print(Text(self._SEPARATOR, style="dim"))

    def show_welcome(self, model_name: str) -> None:
        """Display initial session information."""

        self._console.print(Text("HomeLLM", style="bold cyan"))
        self._console.print(
            f"Model: {model_name}",
            markup=False,
            highlight=False,
        )
        self._console.print(
            "Type /help to see available commands.",
            markup=False,
            highlight=False,
        )
        self.show_separator()

    def show_info(self, message: str) -> None:
        """Display non-error information from the CLI."""

        self._console.print(
            message,
            markup=False,
            highlight=False,
        )

    def show_goodbye(self) -> None:
        """Display the normal session termination message."""

        self._console.print()
        self._console.print(Text("Session ended.", style="dim"))

    def _render_completion(
        self,
        completion: GenerationComplete,
    ) -> None:
        total_seconds = completion.total_duration_ns / 1_000_000_000

        statistics = (
            f"[prompt: "
            f"{completion.prompt_tokens_per_second:.1f} tok/s"
            f" | generation: "
            f"{completion.generation_tokens_per_second:.1f} tok/s"
            f" | output: {completion.generated_tokens} tokens"
            f" | total: {total_seconds:.2f} s]"
        )

        self._console.print()
        self._console.print()
        self._console.print(Text(statistics, style="dim"), soft_wrap=True)
        self.show_separator()
