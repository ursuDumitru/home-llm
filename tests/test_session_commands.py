"""Session commands and safe resumption without a live model."""

import json
import unittest
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from home_llm.chat_session import ChatSession
from home_llm.cli import main, run_cli
from home_llm.console_renderer import ConsoleRenderer
from home_llm.model_backend import ChatMessage, GenerationComplete, TextChunk
from home_llm.model_registry import ModelProfile, ModelRegistry
from home_llm.model_selection import ModelSelection
from home_llm.session_store import SessionStore, create_session


def fake_backend() -> Mock:
    """Return a recording backend with a complete deterministic response."""
    backend = Mock()
    backend.stream_chat.side_effect = lambda messages: iter(
        (
            TextChunk("Answer"),
            GenerationComplete(1, 1, 1, 1, 1),
        )
    )
    return backend


class SessionCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = SessionStore(self.root / "chats")
        self.profile = ModelProfile(
            "first", "First", "ollama", "example:first", True, 4096, 128, None, None
        )
        self.other = replace(self.profile, id="second", model_name="example:second")
        self.registry = ModelRegistry(
            "first", "http://127.0.0.1:11434", (self.profile, self.other)
        )
        self.backend = fake_backend()
        self.selection = ModelSelection(
            self.registry, self.profile, self.backend, store=self.store
        )
        self.selection.runtime = Mock()
        self.saved = self.store.save(
            create_session(
                replace(self.profile, max_output_tokens=32, temperature=0.5, seed=7),
                (
                    ChatMessage("system", "Be concise."),
                    ChatMessage("user", "Remember blue."),
                    ChatMessage("assistant", "Blue."),
                ),
                title="Original",
            )
        )

    def test_resume_uses_saved_settings_and_supplies_history_to_next_turn(self) -> None:
        resumed = fake_backend()
        with patch(
            "home_llm.model_selection.create_backend", return_value=resumed
        ) as factory:
            differences = self.selection.load(self.saved.session_id)

        factory.assert_called_once_with(
            self.saved.profile, ollama_base_url=self.registry.ollama_base_url
        )
        self.selection.runtime.prepare.assert_called_once_with(self.saved.profile)
        self.assertTrue(any("max_output_tokens" in item for item in differences))
        self.assertFalse(self.selection.has_unsaved_changes)

        list(self.selection.session.send("Which color?"))
        self.assertEqual(
            resumed.stream_chat.call_args.args[0],
            (*self.saved.messages, ChatMessage("user", "Which color?")),
        )
        self.assertTrue(self.selection.has_unsaved_changes)
        updated = self.selection.save()
        self.assertEqual(updated.session_id, self.saved.session_id)
        self.assertFalse(self.selection.has_unsaved_changes)

    def test_unsaved_work_blocks_new_load_and_exit(self) -> None:
        list(self.selection.session.send("Keep me"))
        original = self.selection.session
        messages = original.messages
        operations = (
            self.selection.new,
            lambda: self.selection.load(self.saved.session_id),
            self.selection.require_saved,
        )
        for operation in operations:
            with self.assertRaisesRegex(ValueError, "Unsaved changes"):
                operation()
            self.assertIs(self.selection.session, original)
            self.assertEqual(original.messages, messages)
        self.selection.runtime.prepare.assert_not_called()

    def test_explicit_discard_starts_fresh_without_changing_saved_file(self) -> None:
        self.selection.session = ChatSession(self.backend, system_prompt="Private rule")
        list(self.selection.session.send("Discard me"))
        with patch(
            "home_llm.model_selection.create_backend", return_value=self.backend
        ):
            self.selection.new(discard=True)

        self.assertEqual(self.selection.session.messages, ())
        self.assertFalse(self.selection.has_unsaved_changes)
        self.assertEqual(self.store.load(self.saved.session_id), self.saved)

    def test_failed_load_preserves_history_even_when_discard_was_allowed(self) -> None:
        list(self.selection.session.send("Keep until loading succeeds"))
        original = self.selection.session
        messages = original.messages
        for error in (
            ValueError("not installed"),
            RuntimeError("load failed"),
            KeyboardInterrupt(),
        ):
            with self.subTest(error=type(error).__name__):
                self.selection.runtime.prepare.side_effect = error
                with self.assertRaises(type(error)):
                    self.selection.load(self.saved.session_id, discard=True)
                self.assertIs(self.selection.session, original)
                self.assertEqual(original.messages, messages)
                self.assertEqual(self.selection.profile, self.profile)

    def test_missing_disabled_and_remapped_models_are_rejected(self) -> None:
        variants = (
            replace(self.registry, models=(self.other,)),
            replace(
                self.registry, models=(replace(self.profile, enabled=False), self.other)
            ),
            replace(
                self.registry,
                models=(
                    replace(self.profile, model_name="different:model"),
                    self.other,
                ),
            ),
        )
        for registry in variants:
            with self.subTest(registry=registry):
                self.selection.registry = registry
                with self.assertRaises(ValueError):
                    self.selection.load(self.saved.session_id)
                self.assertEqual(self.selection.session.messages, ())
        self.selection.runtime.prepare.assert_not_called()

    def test_failed_save_remains_dirty_and_retry_keeps_identity(self) -> None:
        list(self.selection.session.send("First"))
        saved = self.selection.save()
        list(self.selection.session.send("Second"))
        with patch.object(self.store, "save", side_effect=RuntimeError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                self.selection.save()

        self.assertTrue(self.selection.has_unsaved_changes)
        self.assertEqual(self.store.load(saved.session_id), saved)
        self.assertEqual(self.selection.save().session_id, saved.session_id)

    def test_clear_preserves_transcript_identity_and_marks_boundary_unsaved(
        self,
    ) -> None:
        self.selection.load(self.saved.session_id)
        self.selection.clear()

        self.assertEqual(self.selection.session.messages, self.saved.messages)
        self.assertEqual(
            self.selection.session.context_messages, self.saved.messages[:1]
        )
        self.assertTrue(self.selection.has_unsaved_changes)
        self.assertEqual(self.store.load(self.saved.session_id), self.saved)

        cleared = self.selection.save()

        self.assertEqual(cleared.session_id, self.saved.session_id)
        self.assertEqual(cleared.messages, self.saved.messages)
        self.assertEqual(cleared.context_start, 1)
        self.assertFalse(self.selection.has_unsaved_changes)

    def test_cli_save_list_new_load_and_exit_guard(self) -> None:
        inputs = iter(
            (
                "Hi",
                "/new",
                "/save A new title",
                "/sessions",
                "/new",
                f"/load {self.saved.session_id}",
                "Which color?",
                "/exit",
                "/save",
                "/exit",
            )
        )
        output = StringIO()
        resumed = fake_backend()
        with patch("home_llm.model_selection.create_backend", return_value=resumed):
            result = run_cli(
                self.backend,
                ConsoleRenderer(stream=output, use_color=False),
                "First",
                selection=self.selection,
                input_function=lambda prompt: next(inputs),
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            resumed.stream_chat.call_args.args[0],
            (
                *self.saved.messages,
                ChatMessage("user", "Which color?"),
            ),
        )
        for expected in (
            "Unsaved changes",
            "Saved session:",
            "A new title",
            "Using saved setting:",
        ):
            self.assertIn(expected, output.getvalue())
        self.assertEqual(len(self.store.list_sessions()[0]), 2)

    def test_idle_ctrl_c_guards_changes_and_eof_reports_unsaved_exit(self) -> None:
        output = StringIO()
        inputs = Mock(side_effect=["Hi", KeyboardInterrupt(), "/save", "/exit"])
        self.assertEqual(
            run_cli(
                self.backend,
                ConsoleRenderer(stream=output, use_color=False),
                "First",
                selection=self.selection,
                input_function=inputs,
            ),
            0,
        )
        self.assertIn("Unsaved changes", output.getvalue())
        list(self.selection.session.send("Not saved"))

        self.assertEqual(
            run_cli(
                self.backend,
                ConsoleRenderer(stream=output, use_color=False),
                "First",
                selection=self.selection,
                input_function=Mock(side_effect=EOFError),
            ),
            1,
        )
        self.assertIn("unsaved changes were not saved", output.getvalue())

    def test_restore_api_rejects_incomplete_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete"):
            ChatSession.from_messages(
                self.backend, (ChatMessage("user", "Unanswered"),)
            )

    def test_startup_uses_custom_store_and_load_checks_fresh_registry(self) -> None:
        registry_path = self.root / "models.json"
        document = {"schema_version": 1, **asdict(self.registry)}
        registry_path.write_text(json.dumps(document), encoding="utf-8")

        with (
            patch("home_llm.cli.ConsoleRenderer"),
            patch("home_llm.cli.run_cli", return_value=0) as run,
        ):
            main(
                [
                    "--registry",
                    str(registry_path),
                    "--sessions-dir",
                    str(self.store.directory),
                ]
            )

        selection = run.call_args.kwargs["selection"]
        self.assertEqual(selection.store.directory, self.store.directory)
        document["default_model"] = "second"
        document["models"][0]["enabled"] = False
        registry_path.write_text(json.dumps(document), encoding="utf-8")
        selection.runtime = Mock()

        with self.assertRaisesRegex(ValueError, "disabled"):
            selection.load(self.saved.session_id)
        selection.runtime.prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
