"""Configuration tests that need no model server or GPU."""

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from home_llm.model_registry import load_model_registry


class ModelRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "models.json"
        self.document = {
            "schema_version": 1,
            "default_model": "small",
            "ollama_base_url": "http://127.0.0.1:11434",
            "models": [
                {
                    "id": "small",
                    "display_name": "Small model",
                    "backend": "ollama",
                    "model_name": "example:small",
                    "enabled": True,
                    "context_length": 4096,
                    "max_output_tokens": 512,
                    "temperature": None,
                    "seed": None,
                }
            ],
        }

    def write_document(self) -> None:
        """Write a test fixture to its isolated temporary directory."""

        self.path.write_text(json.dumps(self.document), encoding="utf-8")

    def test_new_profile_needs_only_a_configuration_entry(self) -> None:
        extra = dict(self.document["models"][0])
        extra.update(id="another", model_name="another-family:latest")
        self.document["models"].append(extra)
        self.write_document()

        registry = load_model_registry(self.path)

        self.assertEqual(registry.select().id, "small")
        self.assertEqual(registry.select("another").model_name, "another-family:latest")
        self.assertIsNone(registry.select().temperature)
        with self.assertRaisesRegex(ValueError, "Unknown model ID"):
            registry.select("missing")

    def test_disabled_profile_is_listed_but_cannot_be_selected(self) -> None:
        disabled = dict(self.document["models"][0])
        disabled.update(id="disabled", enabled=False)
        self.document["models"].append(disabled)
        self.write_document()

        registry = load_model_registry(self.path)

        self.assertEqual(len(registry.models), 2)
        with self.assertRaisesRegex(ValueError, "is disabled"):
            registry.select("disabled")

    def test_rejects_invalid_profile_settings(self) -> None:
        baseline = deepcopy(self.document)
        invalid_values = (
            ("backend", "unknown"),
            ("model_name", "example:cloud-cloud"),
            ("enabled", "false"),
            ("context_length", True),
            ("context_length", 0),
            ("max_output_tokens", 4096),
            ("temperature", -1),
            ("temperature", float("nan")),
            ("seed", True),
            ("seed", -1),
            ("unexpected_setting", 1),
        )
        for field, value in invalid_values:
            with self.subTest(field=field, value=value):
                self.document = deepcopy(baseline)
                self.document["models"][0][field] = value
                self.write_document()
                with self.assertRaisesRegex(ValueError, "Cannot load model registry"):
                    load_model_registry(self.path)

    def test_rejects_invalid_registry_documents(self) -> None:
        baseline = deepcopy(self.document)
        variants = []
        for field, value in (
            ("schema_version", 2),
            ("default_model", "missing"),
            ("ollama_base_url", "https://example.com"),
            ("models", []),
        ):
            document = deepcopy(baseline)
            document[field] = value
            variants.append(document)
        duplicate = deepcopy(baseline)
        duplicate["models"].append(dict(duplicate["models"][0]))
        variants.append(duplicate)
        disabled_default = deepcopy(baseline)
        disabled_default["models"][0]["enabled"] = False
        variants.append(disabled_default)

        for index, document in enumerate(variants):
            with self.subTest(variant=index):
                self.document = document
                self.write_document()
                with self.assertRaises(ValueError):
                    load_model_registry(self.path)

        for text in ("{broken", '{"schema_version": 1, "schema_version": 2}'):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_model_registry(self.path)


if __name__ == "__main__":
    unittest.main()
