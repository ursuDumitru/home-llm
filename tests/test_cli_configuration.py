"""Verify that CLI configuration reaches the runtime adapter."""

import json
import unittest
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from home_llm.cli import main
from home_llm.model_backend import ChatMessage, TextChunk


class CliConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "models.json"
        self.document = {
            "schema_version": 1,
            "default_model": "custom",
            "ollama_base_url": "http://127.0.0.1:11434",
            "models": [
                {
                    "id": "custom",
                    "display_name": "Custom model",
                    "backend": "ollama",
                    "model_name": "new-family:small",
                    "enabled": True,
                    "context_length": 2048,
                    "max_output_tokens": 128,
                    "temperature": 0.25,
                    "seed": 7,
                }
            ],
        }
        self.write_registry()

    def write_registry(self) -> None:
        """Write only to this test's temporary directory."""

        self.path.write_text(json.dumps(self.document), encoding="utf-8")

    def generate_from_startup(self, *arguments: str) -> dict:
        """Exercise the real adapter using a fake HTTP response."""

        with (
            patch("home_llm.cli.run_cli", return_value=0) as run_mock,
            patch("home_llm.cli.ConsoleRenderer"),
        ):
            result = main(["--registry", str(self.path), *arguments])
        self.assertEqual(result, 0)
        backend = run_mock.call_args.kwargs["backend"]
        response = {"message": {"content": "Hello"}, "done": True}

        with patch("home_llm.ollama_backend.urlopen") as http_mock:
            http_mock.return_value = BytesIO(json.dumps(response).encode("utf-8"))
            events = list(backend.stream_chat([ChatMessage(role="user", content="Hi")]))

        self.assertEqual(events[0], TextChunk(text="Hello"))
        request = http_mock.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        return json.loads(request.data)

    def test_default_profile_settings_reach_http_request(self) -> None:
        payload = self.generate_from_startup()

        self.assertEqual(payload["model"], "new-family:small")
        self.assertEqual(
            payload["options"],
            {"num_ctx": 2048, "num_predict": 128, "temperature": 0.25, "seed": 7},
        )

    def test_accepts_profile_id_and_unique_runtime_name(self) -> None:
        second = dict(self.document["models"][0])
        second.update(id="second", model_name="different-family:large")
        self.document["models"].append(second)
        self.write_registry()

        for selector in ("second", "different-family:large"):
            with self.subTest(selector=selector):
                payload = self.generate_from_startup("--model", selector)
                self.assertEqual(payload["model"], "different-family:large")

    def test_overrides_do_not_rewrite_registry(self) -> None:
        original = self.path.read_text(encoding="utf-8")
        payload = self.generate_from_startup(
            "--context-length", "4096", "--max-output-tokens", "64"
        )

        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_predict"], 64)
        self.assertEqual(payload["options"]["temperature"], 0.25)
        self.assertEqual(self.path.read_text(encoding="utf-8"), original)

    def test_invalid_choices_fail_before_starting_chat(self) -> None:
        disabled = dict(self.document["models"][0])
        disabled.update(id="disabled", model_name="disabled:model", enabled=False)
        duplicate = dict(self.document["models"][0])
        duplicate.update(id="another-profile")
        self.document["models"].extend([disabled, duplicate])
        self.write_registry()

        scenarios = (
            (["--model", "missing"], "not configured"),
            (["--model", "disabled"], "is disabled"),
            (["--model", "new-family:small"], "ambiguous"),
            (["--context-length", "64"], "must be smaller than context_length"),
        )
        for arguments, expected_error in scenarios:
            with (
                self.subTest(arguments=arguments),
                patch("sys.stderr", new_callable=StringIO) as stderr,
                patch("home_llm.cli.run_cli") as run_mock,
            ):
                with self.assertRaises(SystemExit) as raised:
                    main(["--registry", str(self.path), *arguments])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(expected_error, stderr.getvalue())
                run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
