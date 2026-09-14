"""Optional integration tests requiring a local Ollama model."""

import os
import unittest

from home_llm.chat_session import ChatSession
from home_llm.model_backend import (
    GenerationComplete,
    TextChunk,
)
from home_llm.ollama_backend import OllamaBackend

RUN_INTEGRATION_TESTS = os.environ.get("HOME_LLM_RUN_INTEGRATION") == "1"


@unittest.skipUnless(
    RUN_INTEGRATION_TESTS,
    "Set HOME_LLM_RUN_INTEGRATION=1 to run local Ollama tests.",
)
class OllamaIntegrationTests(unittest.TestCase):
    def test_installed_model_returns_text_and_metrics(self) -> None:
        model_name = os.environ.get(
            "HOME_LLM_TEST_MODEL",
            "qwen3.5:4b",
        )

        self.assertFalse(
            model_name.casefold().endswith("-cloud"),
            "Integration tests must use a local model.",
        )

        backend = OllamaBackend(
            model=model_name,
            context_length=4096,
            max_output_tokens=32,
            timeout_seconds=180,
        )
        session = ChatSession(backend)

        events = list(
            session.send("Reply with one short sentence confirming you are running.")
        )

        answer = "".join(event.text for event in events if isinstance(event, TextChunk))
        completions = [
            event for event in events if isinstance(event, GenerationComplete)
        ]

        self.assertTrue(answer.strip())
        self.assertEqual(len(completions), 1)
        self.assertGreater(completions[0].generated_tokens, 0)
        self.assertGreater(completions[0].total_duration_ns, 0)

        self.assertEqual(
            session.messages[-1].content,
            answer,
        )


if __name__ == "__main__":
    unittest.main()
