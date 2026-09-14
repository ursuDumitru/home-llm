"""Deterministic tests for HomeLLM quality evaluation."""

import json
import unittest
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from home_llm.model_backend import (
    ChatMessage,
    GenerationComplete,
    StreamEvent,
    TextChunk,
)
from home_llm.quality_evaluation import (
    EvaluationCase,
    build_evaluation_artifact,
    evaluate_model,
    list_running_ollama_models,
    save_evaluation_artifact,
    score_answer,
    wait_for_ollama_model_unloaded,
)


class FakeEvaluationBackend:
    """Return predefined answers without contacting Ollama."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = iter(answers)
        self.received_messages: list[tuple[ChatMessage, ...]] = []

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
    ) -> Iterator[StreamEvent]:
        self.received_messages.append(tuple(messages))
        answer = next(self._answers)

        yield TextChunk(text=answer)
        yield GenerationComplete(
            prompt_tokens=20,
            generated_tokens=10,
            prompt_duration_ns=100_000_000,
            generation_duration_ns=500_000_000,
            total_duration_ns=600_000_000,
            load_duration_ns=1_000_000,
        )


class AnswerScoringTests(unittest.TestCase):
    def test_exact_scoring_rejects_additional_text(self) -> None:
        evaluation_case = EvaluationCase(
            identifier="exact",
            category="Exact",
            prompt="Return OK.",
            scoring_method="exact",
            expected_answer="OK",
        )

        self.assertTrue(score_answer(evaluation_case, "OK\n"))
        self.assertFalse(score_answer(evaluation_case, "Answer: OK"))

    def test_json_scoring_ignores_key_order(self) -> None:
        evaluation_case = EvaluationCase(
            identifier="json",
            category="JSON",
            prompt="Return JSON.",
            scoring_method="json",
            expected_answer={
                "name": "Mira",
                "count": 3,
            },
        )

        self.assertTrue(
            score_answer(
                evaluation_case,
                '{"count": 3, "name": "Mira"}',
            )
        )
        self.assertFalse(
            score_answer(
                evaluation_case,
                '```json\n{"count": 3, "name": "Mira"}\n```',
            )
        )

    def test_manual_case_has_no_automatic_score(self) -> None:
        evaluation_case = EvaluationCase(
            identifier="manual",
            category="Manual",
            prompt="Explain something.",
            scoring_method="manual",
        )

        self.assertIsNone(score_answer(evaluation_case, "An explanation."))


class ModelEvaluationTests(unittest.TestCase):
    def test_runs_each_case_as_an_independent_request(self) -> None:
        evaluation_cases = (
            EvaluationCase(
                identifier="first",
                category="Exact",
                prompt="First prompt.",
                scoring_method="exact",
                expected_answer="FIRST",
            ),
            EvaluationCase(
                identifier="second",
                category="Manual",
                prompt="Second prompt.",
                scoring_method="manual",
            ),
        )
        backend = FakeEvaluationBackend(
            answers=["FIRST", "Second answer."],
        )

        evaluation = evaluate_model(
            backend,
            model_name="test-model",
            evaluation_cases=evaluation_cases,
        )

        self.assertEqual(len(evaluation.case_results), 2)
        self.assertEqual(evaluation.automatic_pass_count, 1)
        self.assertEqual(evaluation.automatic_case_count, 1)

        self.assertEqual(
            backend.received_messages,
            [
                (
                    ChatMessage(
                        role="user",
                        content="First prompt.",
                    ),
                ),
                (
                    ChatMessage(
                        role="user",
                        content="Second prompt.",
                    ),
                ),
            ],
        )


class OllamaUnloadTests(unittest.TestCase):
    @patch("home_llm.quality_evaluation.urlopen")
    def test_lists_loaded_models_from_ollama_api(
        self,
        urlopen_mock,
    ) -> None:
        response_document = {
            "models": [
                {
                    "name": "qwen3.5:9b",
                    "model": "qwen3.5:9b",
                }
            ]
        }
        urlopen_mock.return_value = BytesIO(
            json.dumps(response_document).encode("utf-8")
        )

        running_models = list_running_ollama_models()

        self.assertEqual(
            running_models,
            {"qwen3.5:9b"},
        )

    @patch("home_llm.quality_evaluation.time.sleep")
    @patch("home_llm.quality_evaluation.list_running_ollama_models")
    def test_waits_until_model_is_absent(
        self,
        list_models_mock,
        sleep_mock,
    ) -> None:
        list_models_mock.side_effect = [
            {"qwen3.5:9b"},
            set(),
        ]

        wait_for_ollama_model_unloaded(
            "qwen3.5:9b",
            timeout_seconds=1.0,
            polling_interval_seconds=0.01,
        )

        self.assertEqual(list_models_mock.call_count, 2)
        sleep_mock.assert_called_once()


class EvaluationArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluation_case = EvaluationCase(
            identifier="manual",
            category="Manual",
            prompt="Explain something.",
            scoring_method="manual",
        )

        self.evaluations = {
            "A": evaluate_model(
                FakeEvaluationBackend(["Answer A"]),
                model_name="model-nine",
                evaluation_cases=(self.evaluation_case,),
            ),
            "B": evaluate_model(
                FakeEvaluationBackend(["Answer B"]),
                model_name="model-four",
                evaluation_cases=(self.evaluation_case,),
            ),
        }

    def test_builds_versioned_evaluation_document(
        self,
    ) -> None:
        document = build_evaluation_artifact(
            self.evaluations,
            {"manual": "b"},
            context_length=4096,
            max_output_tokens=512,
            gpu_reporting_enabled=False,
            created_at=datetime(
                2026,
                9,
                12,
                7,
                0,
                tzinfo=UTC,
            ),
        )

        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(
            document["models"]["A"]["model_name"],
            "model-nine",
        )
        self.assertEqual(
            document["models"]["B"]["manual_wins"],
            1,
        )
        self.assertEqual(
            document["manual_preferences"],
            {"manual": "b"},
        )
        self.assertEqual(
            document["settings"]["thinking_enabled"],
            False,
        )

    def test_saves_evaluation_json(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "evaluation.json"

            saved_path = save_evaluation_artifact(
                self.evaluations,
                {"manual": "tie"},
                context_length=4096,
                max_output_tokens=512,
                gpu_reporting_enabled=False,
                output_path=output_path,
            )

            self.assertEqual(saved_path, output_path)
            self.assertTrue(saved_path.is_file())

            document = json.loads(saved_path.read_text(encoding="utf-8"))

            self.assertEqual(
                document["models"]["A"]["case_results"][0]["answer_text"],
                "Answer A",
            )


if __name__ == "__main__":
    unittest.main()
