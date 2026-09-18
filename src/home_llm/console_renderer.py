"""Terminal rendering for the interactive HomeLLM interface."""

import os
import sys
from typing import TextIO

from rich.console import Console
from rich.table import Table
from rich.text import Text

from home_llm.chat_session import ChatSession
from home_llm.context_budget import ContextSelection
from home_llm.model_backend import (
    GenerationComplete,
    StreamEvent,
    TextChunk,
)
from home_llm.model_selection import ModelStatus
from home_llm.session_store import SavedSession


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

    def show_models(self, rows: tuple[ModelStatus, ...], active_id: str) -> None:
        """Render configuration flags separately from live runtime state."""
        table = Table(title="Configured models")
        for heading in (
            "Active",
            "ID",
            "Runtime model",
            "Enabled",
            "Installed",
            "Loaded",
        ):
            table.add_column(heading)

        def status(value: bool | None) -> str:
            return "unknown" if value is None else ("yes" if value else "no")

        for row in rows:
            profile = row.profile
            table.add_row(
                "*" if profile.id == active_id else "",
                Text(profile.id),
                Text(profile.model_name),
                status(profile.enabled),
                status(row.installed),
                status(row.loaded),
            )
        self._console.print(table)
        self.show_info("Runtime status is a snapshot; loaded does not mean GPU-only.")

    def show_context_selection(self, selection: ContextSelection) -> None:
        """Notify the user before generation when older eligible turns are omitted."""
        if selection.omitted_turns:
            self.show_info(
                f"Context notice: omitted {selection.omitted_turns} older turns from "
                f"this request; kept {selection.selected_turns} recent turns. "
                "Omitted turns remain in the full transcript but are not sent to the model. "
                f"Estimated input: {selection.estimated_input_tokens}/"
                f"{selection.input_budget} tokens (not an exact tokenizer count)."
            )

    def show_context(self, session: ChatSession) -> None:
        """Distinguish eligible history from the last attempted request's selection."""
        budget = session.context_budget
        if budget is None:
            self.show_info(
                "Automatic context selection is not configured for this session."
            )
        else:
            self.show_info(
                f"Configured context: {budget.context_length} tokens\n"
                f"Output limit: {budget.max_output_tokens} tokens (reserved)\n"
                f"Template/safety reserve: {budget.safety_margin} tokens\n"
                f"Estimated input budget: {budget.input_budget} tokens\n"
                "Estimator: 3 UTF-8 bytes/token plus 8 framing tokens/message.\n"
                "Estimates are not a guarantee of fitting the model's actual context."
            )
        turns = sum(message.role == "user" for message in session.messages)
        self.show_info(
            f"Full transcript: {turns} turns\n"
            f"Excluded by /clear: {session.context_start} turns\n"
            f"Eligible history: {turns - session.context_start} turns\n"
            "The system prompt and current prompt are retained. Selection keeps "
            "complete recent turns; older turns may be omitted without deleting them."
        )
        selected = session.last_context_selection
        if selected is None:
            self.show_info(
                "No request selection available: new/cleared session or rejected prompt."
            )
        else:
            self.show_info(
                f"Last attempted request: {selected.selected_turns} history turns selected, "
                f"{selected.omitted_turns} omitted by budget; "
                f"estimated input {selected.estimated_input_tokens} tokens.\n"
                "This describes the prepared selection, not confirmed model retention."
            )
        self.show_info("The next selection depends on the next prompt's size.")

    def show_sessions(self, sessions: tuple[SavedSession, ...]) -> None:
        """Show full IDs for copying, without interpreting titles as markup."""
        if not sessions:
            self.show_info("No saved conversations.")
            return

        for session in sessions:
            turns = sum(message.role == "user" for message in session.messages)
            self._console.print(
                Text(f"{session.session_id}  {session.title or 'Untitled'}"),
                soft_wrap=True,
            )
            self.show_info(
                f"  {session.profile.id} | {turns} turns | updated {session.updated_at}"
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
