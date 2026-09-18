"""Conversation persistence checks without Ollama, GPU use, or private data."""

import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from home_llm.model_backend import ChatMessage
from home_llm.model_registry import ModelProfile
from home_llm.session_store import SessionStore, create_session


class SessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = SessionStore(self.root / "conversations")
        self.profile = ModelProfile(
            "local",
            "Local model",
            "ollama",
            "example:small",
            True,
            4096,
            64,
            0.25,
            7,
        )
        self.messages = (
            ChatMessage("system", "Answer concisely."),
            ChatMessage("user", "Salut!"),
            ChatMessage("assistant", "Bună!"),
        )
        self.session = create_session(
            self.profile, self.messages, title="../A title is not a filename"
        )

    def test_round_trip_preserves_history_settings_and_safe_filename(self) -> None:
        saved = self.store.save(self.session)
        loaded = self.store.load(saved.session_id)

        self.assertEqual(loaded, saved)
        self.assertEqual(loaded.messages, self.messages)
        self.assertEqual(loaded.profile, self.profile)
        self.assertEqual(loaded.created_at, self.session.created_at)
        self.assertEqual(
            [path.name for path in self.store.directory.iterdir()],
            [f"{saved.session_id}.json"],
        )

        document = json.loads(
            (self.store.directory / f"{saved.session_id}.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("enabled", document["profile"])

    def test_updates_keep_identity_and_reject_stale_snapshot(self) -> None:
        saved = self.store.save(self.session)
        messages = (
            *saved.messages,
            ChatMessage("user", "Another question"),
            ChatMessage("assistant", "Another answer"),
        )

        # Fixed later time makes the stale-write check deterministic.
        with patch("home_llm.session_store.datetime") as clock:
            clock.now.return_value = datetime(2100, 1, 1, tzinfo=UTC)
            clock.fromisoformat.side_effect = datetime.fromisoformat
            updated = self.store.save(replace(saved, messages=messages))

        self.assertEqual(updated.session_id, saved.session_id)
        self.assertEqual(updated.created_at, saved.created_at)
        self.assertEqual(self.store.load(saved.session_id).messages, messages)
        with self.assertRaisesRegex(ValueError, "changed on disk"):
            self.store.save(saved)

    def test_incomplete_and_misordered_turns_are_rejected_before_writing(self) -> None:
        variants = (
            (ChatMessage("user", "Unanswered"),),
            (ChatMessage("assistant", "First"), ChatMessage("user", "Wrong order")),
            (ChatMessage("user", "Question"), ChatMessage("system", "Misplaced")),
        )
        for messages in variants:
            with self.subTest(messages=messages):
                with self.assertRaises(ValueError):
                    self.store.save(replace(self.session, messages=messages))
        self.assertFalse(self.store.directory.exists())

    def test_unsafe_ids_and_symlink_files_are_rejected(self) -> None:
        for session_id in ("../outside", "/tmp/outside", "abc", "A" * 32):
            with self.subTest(session_id=session_id):
                with self.assertRaises(ValueError):
                    self.store.load(session_id)
                with self.assertRaises(ValueError):
                    self.store.save(replace(self.session, session_id=session_id))

        saved = self.store.save(self.session)
        alias_id = "a" * 32
        alias = self.store.directory / f"{alias_id}.json"
        alias.symlink_to(self.store.directory / f"{saved.session_id}.json")

        with self.assertRaisesRegex(ValueError, "symbolic link"):
            self.store.load(alias_id)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            self.store.save(replace(saved, session_id=alias_id))

    def test_bad_schema_fields_settings_and_messages_are_rejected(self) -> None:
        saved = self.store.save(self.session)
        path = self.store.directory / f"{saved.session_id}.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        variants = (
            {"schema_version": 3},
            {"schema_version": True},
            {"session_id": "b" * 32},
            {"created_at": "not-a-date"},
            {"updated_at": "2000-01-01T00:00:00+00:00"},
            {"updated_at": "2100-01-01T00:00:00"},
            {"profile": dict(original["profile"], context_length=0)},
            {"profile": dict(original["profile"], enabled=True)},
            {"messages": [{"role": "user", "content": 123}]},
            {"messages": [{"role": "tool", "content": "Unsupported"}]},
            {"extra": "unknown field"},
        )
        for changes in variants:
            with self.subTest(changes=changes):
                path.write_text(json.dumps(original | changes), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.store.load(saved.session_id)

        for raw in ("{broken", '{"schema_version":1,"schema_version":1}'):
            with self.subTest(raw=raw):
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.store.load(saved.session_id)

    def test_failed_write_or_replace_preserves_previous_file(self) -> None:
        saved = self.store.save(self.session)
        path = self.store.directory / f"{saved.session_id}.json"
        original = path.read_bytes()

        for operation in ("fsync", "replace"):
            with self.subTest(operation=operation):
                with patch(
                    f"home_llm.session_store.os.{operation}",
                    side_effect=OSError("simulated failure"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "Cannot save"):
                        self.store.save(replace(saved, title="Unsaved title"))
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(set(self.store.directory.iterdir()), {path})

    def test_listing_reports_bad_files_and_keeps_valid_sessions(self) -> None:
        self.assertEqual(self.store.list_sessions(), ((), ()))
        saved = self.store.save(self.session)
        bad_path = self.store.directory / f"{'b' * 32}.json"
        bad_path.write_text("{broken", encoding="utf-8")

        sessions, errors = self.store.list_sessions()

        self.assertEqual(sessions, (saved,))
        self.assertEqual(len(errors), 1)
        self.assertIn(bad_path.name, errors[0])

    def test_missing_file_is_actionable(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Cannot read session"):
            self.store.load("a" * 32)

    def test_empty_snapshot_is_valid_and_creation_has_no_disk_side_effect(self) -> None:
        session = create_session(self.profile, (), title=None)

        self.assertEqual(session.messages, ())
        self.assertNotEqual(session.session_id, self.session.session_id)
        self.assertFalse(self.store.directory.exists())


if __name__ == "__main__":
    unittest.main()
