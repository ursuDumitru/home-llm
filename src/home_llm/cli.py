"""Interactive command-line interface for local model inference."""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from home_llm.backend_factory import create_backend
from home_llm.chat_session import ChatSession
from home_llm.console_renderer import ConsoleRenderer
from home_llm.model_backend import ModelBackend
from home_llm.model_registry import ModelProfile, ModelRegistry, load_model_registry
from home_llm.model_selection import ModelSelection
from home_llm.session_store import SessionStore

DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_MAX_OUTPUT_TOKENS = 512

HELP_TEXT = """Available commands:
  /help                    Show this command list.
  /save [title]            Save this conversation; optionally set its title.
  /sessions                List saved conversation IDs.
  /load <id> [--discard]   Resume a saved conversation with its saved settings.
  /new [--discard]         Start an empty chat on the current model.
  /clear                   Reset request context; retain the full transcript.
  /context                 Show context budget and last request selection.
  /models                  List enabled, installed, and loaded status.
  /model                   Show the active profile and effective settings.
  /model <id>              Switch profiles in an empty conversation.
  /enable <id>             Enable a profile and save the registry.
  /disable <id>            Disable a non-active, non-default profile and save.
  /exit [--discard]        End the session.
Unsaved work requires /save or explicit --discard before replacement.
Switching models uses registry settings; loading a chat uses saved settings.
Startup setting overrides apply only to the initial selection."""


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
    """Create arguments whose omitted generation settings come from JSON."""

    parser = argparse.ArgumentParser(
        prog="home-llm",
        description="Run an interactive chat with a configured local model.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("config/models.json"),
        help="Registry path, relative to the working directory (default: config/models.json).",
    )
    parser.add_argument(
        "--model",
        type=local_model_name,
        default=None,
        help="Profile ID or unique configured runtime name; defaults to the registry selection.",
    )
    parser.add_argument(
        "--context-length",
        type=positive_integer,
        default=None,
        help="Override the selected profile's context length for this session.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=positive_integer,
        default=None,
        help="Override the selected profile's output limit for this session.",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path("conversations"),
        help="Saved chat directory (default: conversations, relative to working directory).",
    )
    return parser


def resolve_profile(registry: ModelRegistry, selector: str | None) -> ModelProfile:
    """Prefer profile IDs; accept a runtime name only when it is unambiguous."""

    if selector is None or any(profile.id == selector for profile in registry.models):
        return registry.select(selector)

    matches = [profile for profile in registry.models if profile.model_name == selector]
    if len(matches) == 1:
        return registry.select(matches[0].id)
    if len(matches) > 1:
        ids = ", ".join(profile.id for profile in matches)
        raise ValueError(
            f"Runtime name {selector!r} is ambiguous; use a profile ID: {ids}."
        )
    raise ValueError(
        f"Model {selector!r} is not configured. Add a profile to the registry "
        "or choose an existing profile ID."
    )


def run_cli(
    backend: ModelBackend,
    renderer: ConsoleRenderer,
    model_name: str,
    *,
    input_function: Callable[[str], str] = input,
    selection: ModelSelection | None = None,
) -> int:
    """Run interactive chat with explicit persistence and unsaved-work guards."""
    session = selection.session if selection is not None else ChatSession(backend)
    renderer.show_welcome(model_name)

    while True:
        if selection is not None:
            session = selection.session
        try:
            renderer.show_prompt()
            user_text = input_function("")
        except KeyboardInterrupt:
            if selection is not None and selection.has_unsaved_changes:
                renderer.show_error("Unsaved changes. Use /save or /exit --discard.")
                continue
            renderer.show_goodbye()
            return 0
        except EOFError:
            if selection is not None and selection.has_unsaved_changes:
                renderer.show_error("Input closed; unsaved changes were not saved.")
                return 1
            renderer.show_goodbye()
            return 0

        command = user_text.strip()
        if not command:
            continue

        if command.startswith("/"):
            parts = command.split()
            verb, arguments = parts[0].casefold(), parts[1:]
            try:
                if verb in {"/save", "/sessions", "/load", "/new"}:
                    if selection is None:
                        raise ValueError("Session commands require a model registry.")

                    if verb == "/save":
                        saved = selection.save(
                            " ".join(arguments) if arguments else None
                        )
                        renderer.show_info(f"Saved session: {saved.session_id}")

                    elif verb == "/sessions":
                        if arguments:
                            raise ValueError("Usage: /sessions (no arguments)")
                        rows, errors = selection.store.list_sessions()
                        renderer.show_sessions(rows)
                        for error in errors:
                            renderer.show_error(error)

                    elif verb == "/load":
                        if not arguments or (
                            len(arguments) != 1
                            and (len(arguments) != 2 or arguments[1] != "--discard")
                        ):
                            raise ValueError("Usage: /load <session-id> [--discard]")
                        renderer.show_info(
                            "Checking saved chat and model availability."
                        )
                        differences = selection.load(
                            arguments[0], discard=len(arguments) == 2
                        )
                        for difference in differences:
                            renderer.show_info(f"Using saved setting: {difference}")
                        renderer.show_info(
                            f"Loaded session {arguments[0]} "
                            f"({len(selection.session.messages)} messages)."
                        )
                        renderer.show_info(
                            f"Active model: {profile_label(selection.profile)}"
                        )
                        renderer.show_info(profile_settings(selection.profile))

                    else:
                        if arguments not in ([], ["--discard"]):
                            raise ValueError("Usage: /new [--discard]")
                        selection.new(discard=bool(arguments))
                        renderer.show_info(
                            "New conversation started; no system prompt carried over."
                        )
                    continue

                if verb == "/model":
                    if len(arguments) > 1:
                        raise ValueError("Usage: /model or /model <profile-id>")
                    if arguments:
                        if selection is None:
                            raise ValueError(
                                "Model switching requires a model registry."
                            )
                        renderer.show_info(
                            "Checking model selection; loading may take a moment."
                        )
                        selection.select(arguments[0])
                    if selection is not None:
                        model_name = profile_label(selection.profile)
                    renderer.show_info(f"Active model: {model_name}")
                    if selection is not None:
                        renderer.show_info(profile_settings(selection.profile))
                    continue

                if verb in {"/enable", "/disable"}:
                    if len(arguments) != 1:
                        raise ValueError(f"Usage: {verb} <profile-id>")
                    if selection is None:
                        raise ValueError("Registry updates require a model registry.")
                    enabled = verb == "/enable"
                    selection.set_enabled(arguments[0], enabled=enabled)
                    state = "enabled" if enabled else "disabled"
                    renderer.show_info(
                        f"Profile {arguments[0]!r} is {state} in the saved registry. "
                        "Model installation and loading are unchanged."
                    )
                    continue

                if verb not in {"/help", "/clear", "/context", "/models", "/exit"}:
                    raise ValueError(
                        f"Unknown command {command!r}. Type /help for commands."
                    )
                if verb == "/exit":
                    if arguments not in ([], ["--discard"]):
                        raise ValueError(f"Usage: {verb} [--discard]")
                elif arguments:
                    raise ValueError(f"Usage: {verb} (no arguments)")

                if verb == "/exit":
                    if selection is not None:
                        selection.require_saved(discard=bool(arguments))
                    renderer.show_goodbye()
                    return 0

                if verb == "/help":
                    renderer.show_info(HELP_TEXT)
                elif verb == "/clear":
                    if selection is None:
                        session.clear()
                    else:
                        selection.clear()
                    renderer.show_info(
                        "Request context cleared; full transcript retained. "
                        "Use /save to persist the boundary."
                    )
                elif verb == "/context":
                    if selection is None:
                        raise ValueError("Context limits require a model registry.")
                    renderer.show_context(session)
                elif verb == "/models":
                    if selection is None:
                        raise ValueError("Model listing requires a model registry.")
                    rows, errors = selection.list_models()
                    renderer.show_models(rows, selection.profile.id)
                    for error in errors:
                        renderer.show_error(error)

            except KeyboardInterrupt:
                renderer.show_error(
                    "Command interrupted; check /model and /sessions before retrying."
                )
            except (RuntimeError, ValueError) as error:
                renderer.show_error(str(error))
            continue

        renderer.begin_answer()
        try:
            for event in session.send(
                user_text, on_context_selected=renderer.show_context_selection
            ):
                renderer.render_event(event)
        except KeyboardInterrupt:
            renderer.show_error("Generation cancelled.")
        except (RuntimeError, ValueError) as error:
            renderer.show_error(str(error))


def profile_label(profile: ModelProfile) -> str:
    """Describe the selected profile without asking the language model."""
    return f"{profile.display_name} [{profile.id}] ({profile.model_name})"


def profile_settings(profile: ModelProfile) -> str:
    """Describe effective settings; unspecified sampling values use model defaults."""
    return (
        f"Settings: context={profile.context_length}, "
        f"output limit={profile.max_output_tokens}, "
        f"temperature={profile.temperature}, seed={profile.seed} "
        "(None means model default)."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve a profile, apply explicit overrides, and start the CLI."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        registry = load_model_registry(arguments.registry)
        profile = resolve_profile(registry, arguments.model)
        overrides: dict[str, int] = {}
        if arguments.context_length is not None:
            overrides["context_length"] = arguments.context_length
        if arguments.max_output_tokens is not None:
            overrides["max_output_tokens"] = arguments.max_output_tokens

        # replace() validates the effective settings without modifying the registry.
        profile = replace(profile, **overrides)
        backend = create_backend(profile, ollama_base_url=registry.ollama_base_url)
    except ValueError as error:
        parser.error(str(error))

    renderer = ConsoleRenderer()
    renderer.show_info(profile_settings(profile))
    return run_cli(
        backend=backend,
        renderer=renderer,
        model_name=profile_label(profile),
        selection=ModelSelection(
            registry,
            profile,
            backend,
            registry_path=arguments.registry,
            store=SessionStore(arguments.sessions_dir),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
