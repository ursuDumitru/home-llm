"""Transcript preservation and persisted request-context boundaries."""

import json
import unittest
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from home_llm.chat_session import ChatSession
from home_llm.cli import run_cli
from home_llm.console_renderer import ConsoleRenderer
from home_llm.model_backend import ChatMessage, GenerationComplete, TextChunk
from home_llm.model_registry import ModelProfile, ModelRegistry
from home_llm.model_selection import ModelSelection
from home_llm.session_store import SessionStore, create_session


class ContextBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = SessionStore(Path(directory.name) / "chats")
        self.profile = ModelProfile(
            "local", "Local", "ollama", "example:small", True, 4096, 64, None, None
        )
        self.backend = Mock()
        self.backend.stream_chat.side_effect = lambda messages: iter(
            (
                TextChunk("Answer"),
                GenerationComplete(1, 1, 1, 1, 1),
            )
        )
        self.messages = (
            ChatMessage("system", "Be concise."),
            ChatMessage("user", "Old secret"),
            ChatMessage("assistant", "Old answer"),
        )

    def test_clear_preserves_transcript_but_omits_old_turns_from_request(self) -> None:
        session = ChatSession.from_messages(self.backend, self.messages)
        session.clear()
        self.assertEqual(session.messages, self.messages)
        self.assertEqual(session.context_messages, self.messages[:1])

        list(session.send("New question"))

        self.assertEqual(
            self.backend.stream_chat.call_args.args[0],
            (self.messages[0], ChatMessage("user", "New question")),
        )
        self.assertEqual(session.messages[:3], self.messages)
        self.assertEqual(len(session.messages), 5)
        self.assertEqual(session.context_start, 1)

        session.clear()
        session.clear()
        self.assertEqual(session.context_start, 2)
        self.assertEqual(len(session.messages), 5)

    def test_save_reload_and_resume_keep_the_boundary(self) -> None:
        saved = self.store.save(
            create_session(self.profile, self.messages, context_start=1)
        )
        registry = ModelRegistry("local", "http://127.0.0.1:11434", (self.profile,))
        selection = ModelSelection(
            registry, self.profile, self.backend, store=self.store
        )
        selection.runtime = Mock()
        with patch(
            "home_llm.model_selection.create_backend", return_value=self.backend
        ):
            selection.load(saved.session_id)

        self.assertFalse(selection.has_unsaved_changes)
        list(selection.session.send("After restart"))
        self.assertEqual(
            self.backend.stream_chat.call_args.args[0],
            (self.messages[0], ChatMessage("user", "After restart")),
        )

        updated = selection.save()
        self.assertEqual(updated.session_id, saved.session_id)
        self.assertEqual(updated.context_start, 1)
        self.assertEqual(updated.messages[:3], self.messages)

    def test_invalid_boundaries_are_rejected_in_memory_and_on_disk(self) -> None:
        saved = self.store.save(create_session(self.profile, self.messages))
        path = self.store.directory / f"{saved.session_id}.json"
        document = json.loads(path.read_text(encoding="utf-8"))

        for invalid in (-1, 2, True, 0.5, "1", None):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "context_start"):
                    ChatSession.from_messages(
                        self.backend, self.messages, context_start=invalid
                    )
                with self.assertRaisesRegex(ValueError, "context_start"):
                    self.store.save(replace(saved, context_start=invalid))
                path.write_text(
                    json.dumps(document | {"context_start": invalid}), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "context_start"):
                    self.store.load(saved.session_id)

    def test_version_one_loads_without_rewriting_and_upgrades_on_save(self) -> None:
        saved = self.store.save(create_session(self.profile, self.messages))
        path = self.store.directory / f"{saved.session_id}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["schema_version"] = 1
        del document["context_start"]
        legacy = json.dumps(document)
        path.write_text(legacy, encoding="utf-8")

        loaded = self.store.load(saved.session_id)

        self.assertEqual(loaded.context_start, 0)
        self.assertEqual(loaded.messages, self.messages)
        self.assertEqual(path.read_text(encoding="utf-8"), legacy)

        self.store.save(loaded)
        upgraded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(upgraded["schema_version"], 2)
        self.assertEqual(upgraded["context_start"], 0)

    def test_failed_and_cancelled_turns_keep_transcript_and_boundary(self) -> None:
        session = ChatSession.from_messages(
            self.backend, self.messages, context_start=1
        )
        for error in (RuntimeError("failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):

                def fail(messages, error=error):
                    yield TextChunk("Partial")
                    raise error

                self.backend.stream_chat.side_effect = fail
                with self.assertRaises(type(error)):
                    list(session.send("Failed question"))
                self.assertEqual(session.messages, self.messages)
                self.assertEqual(session.context_start, 1)
                self.assertEqual(session.context_messages, self.messages[:1])

    def test_cli_clear_is_nondestructive_and_context_reports_eligible_turns(
        self,
    ) -> None:
        registry = ModelRegistry("local", "http://127.0.0.1:11434", (self.profile,))
        selection = ModelSelection(
            registry, self.profile, self.backend, store=self.store
        )
        selection.session = ChatSession.from_messages(self.backend, self.messages)
        inputs = iter(("/clear", "/context", "/new", "/save", "/exit"))
        output = StringIO()

        result = run_cli(
            self.backend,
            ConsoleRenderer(stream=output, use_color=False),
            "Local",
            selection=selection,
            input_function=lambda prompt: next(inputs),
        )

        self.assertEqual(result, 0)
        for expected in (
            "full transcript retained",
            "Excluded by /clear: 1",
            "Eligible history: 0",
            "Unsaved changes",
        ):
            self.assertIn(expected, output.getvalue())

        saved = self.store.list_sessions()[0][0]
        self.assertEqual(saved.messages, self.messages)
        self.assertEqual(saved.context_start, 1)
        self.backend.stream_chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
