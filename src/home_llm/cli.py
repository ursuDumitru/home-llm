"""Interactive command-line interface for local model inference."""

import argparse
from collections.abc import Callable, Sequence

from home_llm.chat_session import ChatSession
from home_llm.console_renderer import ConsoleRenderer
from home_llm.model_backend import ModelBackend
from home_llm.ollama_backend import OllamaBackend

DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_MAX_OUTPUT_TOKENS = 512

HELP_TEXT = """Available commands:
  /help   Show this command list.
  /clear  Clear conversation history.
  /model  Show the active model.
  /exit   End the session."""


def positive_integer(value: str) -> int:
    """Parse a command-line integer that must be greater than zero."""

    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected an integer, received {value!r}."
        ) from error

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            f"Expected a value greater than zero, received {parsed_value}."
        )

    return parsed_value


def local_model_name(value: str) -> str:
    """Reject empty names and explicit Ollama cloud model identifiers."""

    model_name = value.strip()

    if not model_name:
        raise argparse.ArgumentTypeError("The model name must not be empty.")

    if model_name.casefold().endswith("-cloud"):
        raise argparse.ArgumentTypeError(
            "Cloud model identifiers are disabled for this local-only CLI."
        )

    return model_name


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="home-llm",
        description="Run an interactive chat with a local Ollama model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=local_model_name,
        default=DEFAULT_MODEL,
        help="Installed local Ollama model to use.",
    )
    parser.add_argument(
        "--context-length",
        type=positive_integer,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Maximum conversation context in tokens.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=positive_integer,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum tokens generated for one answer.",
    )

    return parser


def run_cli(
    backend: ModelBackend,
    renderer: ConsoleRenderer,
    model_name: str,
    *,
    input_function: Callable[[str], str] = input,
) -> int:
    """Run an interactive conversation until the user exits."""

    session = ChatSession(backend)
    renderer.show_welcome(model_name)

    while True:
        try:
            renderer.show_prompt()
            user_text = input_function("")
        except EOFError, KeyboardInterrupt:
            renderer.show_goodbye()
            return 0

        command = user_text.strip()

        if not command:
            continue

        if command.casefold() == "/exit":
            renderer.show_goodbye()
            return 0

        if command.casefold() == "/help":
            renderer.show_info(HELP_TEXT)
            continue

        if command.casefold() == "/clear":
            session.clear()
            renderer.show_info("Conversation history cleared.")
            continue

        if command.casefold() == "/model":
            renderer.show_info(f"Active model: {model_name}")
            continue

        if command.startswith("/"):
            renderer.show_error(
                f"Unknown command {command!r}. Type /help for commands."
            )
            continue

        renderer.begin_answer()

        try:
            for event in session.send(user_text):
                renderer.render_event(event)
        except KeyboardInterrupt:
            renderer.show_error("Generation cancelled.")
        except (RuntimeError, ValueError) as error:
            renderer.show_error(str(error))


def main(argv: Sequence[str] | None = None) -> int:
    """Parse settings, create production components, and start the CLI."""

    arguments = build_parser().parse_args(argv)

    backend = OllamaBackend(
        model=arguments.model,
        context_length=arguments.context_length,
        max_output_tokens=arguments.max_output_tokens,
    )
    renderer = ConsoleRenderer()

    return run_cli(
        backend=backend,
        renderer=renderer,
        model_name=arguments.model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
