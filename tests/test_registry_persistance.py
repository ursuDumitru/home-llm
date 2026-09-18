"""Registry writes and CLI commands, using only temporary files and fake backends."""

import json
import unittest
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from home_llm.cli import main, run_cli
from home_llm.console_renderer import ConsoleRenderer
from home_llm.model_registry import load_model_registry
from home_llm.model_selection import ModelSelection


class RegistryPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "models.json"
        profile = {
            "id": "first",
            "display_name": "First",
            "backend": "ollama",
            "model_name": "example:first",
            "enabled": True,
            "context_length": 4096,
            "max_output_tokens": 512,
            "temperature": None,
            "seed": None,
        }
        self.document = {
            "schema_version": 1,
            "default_model": "first",
            "ollama_base_url": "http://127.0.0.1:11434",
            "models": [
                profile,
                dict(profile, id="second", model_name="example:second"),
            ],
        }
        self.path.write_text(json.dumps(self.document), encoding="utf-8")
        self.registry = load_model_registry(self.path)
        self.backend = Mock()
        self.selection = ModelSelection(
            self.registry,
            self.registry.select(),
            self.backend,
            registry_path=self.path,
        )
        self.selection.runtime = Mock()

    def test_flags_survive_restart_without_runtime_operations(self) -> None:
        self.selection.set_enabled("second", enabled=False)
        restarted = load_model_registry(self.path)
        with self.assertRaisesRegex(ValueError, "disabled"):
            restarted.select("second")

        selection = ModelSelection(
            restarted, restarted.select(), self.backend, registry_path=self.path
        )
        selection.runtime = Mock()
        selection.set_enabled("second", enabled=True)

        self.assertEqual(load_model_registry(self.path), self.registry)
        self.assertEqual(self.selection.runtime.mock_calls, [])
        self.assertEqual(selection.runtime.mock_calls, [])

    def test_protected_and_unknown_profiles_leave_file_unchanged(self) -> None:
        original = self.path.read_bytes()
        scenarios = (
            ("first", "first", "active"),
            ("second", "first", "default"),
            ("second", "second", "active"),
            ("first", "missing", "Unknown"),
        )
        for active, target, message in scenarios:
            with self.subTest(active=active, target=target):
                self.selection.profile = self.registry.select(active)
                with self.assertRaisesRegex(ValueError, message):
                    self.selection.set_enabled(target, enabled=False)
                self.assertEqual(self.path.read_bytes(), original)
                self.assertEqual(self.selection.registry, self.registry)

    def test_failed_replace_preserves_file_memory_and_removes_temp(self) -> None:
        original = self.path.read_bytes()
        session = self.selection.session
        with patch(
            "home_llm.registry_store.os.replace", side_effect=OSError("blocked")
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not save"):
                self.selection.set_enabled("second", enabled=False)

        self.assertEqual(self.path.read_bytes(), original)
        self.assertIs(self.selection.registry, self.registry)
        self.assertIs(self.selection.session, session)
        self.assertEqual(set(self.path.parent.iterdir()), {self.path})

    def test_failed_write_preserves_original(self) -> None:
        original = self.path.read_bytes()
        with patch(
            "home_llm.registry_store.os.fsync", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not save"):
                self.selection.set_enabled("second", enabled=False)

        self.assertEqual(self.path.read_bytes(), original)
        self.assertIs(self.selection.registry, self.registry)
        self.assertEqual(set(self.path.parent.iterdir()), {self.path})

    def test_invalid_staged_file_is_not_committed(self) -> None:
        original = self.path.read_bytes()
        with (
            patch("home_llm.registry_store.load_model_registry") as validate,
            patch("home_llm.registry_store.os.replace") as commit,
        ):
            validate.side_effect = (
                self.registry,
                ValueError("Invalid staged registry"),
            )
            with self.assertRaisesRegex(ValueError, "Invalid staged"):
                self.selection.set_enabled("second", enabled=False)
            commit.assert_not_called()

        self.assertEqual(self.path.read_bytes(), original)
        self.assertIs(self.selection.registry, self.registry)
        self.assertEqual(set(self.path.parent.iterdir()), {self.path})

    def test_external_changes_are_not_overwritten(self) -> None:
        self.document["models"][1]["max_output_tokens"] = 256
        changed = json.dumps(self.document)
        self.path.write_text(changed, encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "changed on disk"):
            self.selection.set_enabled("second", enabled=False)

        self.assertEqual(self.path.read_text(encoding="utf-8"), changed)
        self.assertIs(self.selection.registry, self.registry)

    def test_session_overrides_are_not_written_to_registry(self) -> None:
        self.selection.profile = replace(self.selection.profile, max_output_tokens=64)
        session = self.selection.session
        self.selection.set_enabled("second", enabled=False)

        self.assertEqual(self.selection.profile.max_output_tokens, 64)
        self.assertEqual(load_model_registry(self.path).select().max_output_tokens, 512)
        self.assertIs(self.selection.session, session)

    def test_cli_validates_arguments_and_blocks_disabled_selection(self) -> None:
        inputs = iter(
            (
                "/enable",
                "/disable second extra",
                "/disable second",
                "/model second",
                "/enable second",
                "/disable missing",
                "/exit",
            )
        )
        output = StringIO()
        result = run_cli(
            self.backend,
            ConsoleRenderer(stream=output, use_color=False),
            "First",
            selection=self.selection,
            input_function=lambda prompt: next(inputs),
        )

        self.assertEqual(result, 0)
        self.assertEqual(load_model_registry(self.path), self.registry)
        self.assertEqual(self.selection.runtime.mock_calls, [])
        rendered = output.getvalue()
        self.assertIn("Usage: /enable <profile-id>", rendered)
        self.assertIn("Usage: /disable <profile-id>", rendered)
        self.assertIn("is disabled", rendered)
        self.assertIn("is enabled", rendered)
        self.assertIn("Unknown model ID", rendered)

    def test_startup_passes_the_selected_registry_path(self) -> None:
        with (
            patch("home_llm.cli.ConsoleRenderer"),
            patch("home_llm.cli.run_cli", return_value=0) as run,
        ):
            self.assertEqual(main(["--registry", str(self.path)]), 0)

        selection = run.call_args.kwargs["selection"]
        selection.set_enabled("second", enabled=False)
        with self.assertRaisesRegex(ValueError, "disabled"):
            load_model_registry(self.path).select("second")


if __name__ == "__main__":
    unittest.main()
