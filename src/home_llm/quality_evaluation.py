"""Blind quality comparison for two local language models."""

import argparse
import json
import secrets
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from home_llm.cli import local_model_name, positive_integer
from home_llm.gpu_monitor import (
    GpuMonitor,
    GpuReport,
    GpuSample,
    NvidiaSmiGpuMonitor,
)
from home_llm.model_backend import (
    ChatMessage,
    GenerationComplete,
    ModelBackend,
    TextChunk,
)
from home_llm.ollama_backend import OllamaBackend

DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_GPU_SAMPLE_INTERVAL = 0.1
EVALUATION_TEMPERATURE = 0.0
EVALUATION_SEED = 42

type ScoringMethod = Literal["exact", "json", "manual"]
type ExpectedAnswer = str | dict[str, object] | None
type GpuMonitorFactory = Callable[[], GpuMonitor]


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One model-independent quality evaluation case."""

    identifier: str
    category: str
    prompt: str
    scoring_method: ScoringMethod
    expected_answer: ExpectedAnswer = None


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    """One model's answer and score for one evaluation case."""

    case: EvaluationCase
    answer_text: str
    completion: GenerationComplete
    automatic_passed: bool | None
    gpu_report: GpuReport | None


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """All evaluation results for one model."""

    model_name: str
    case_results: tuple[EvaluationCaseResult, ...]

    @property
    def automatic_pass_count(self) -> int:
        return sum(result.automatic_passed is True for result in self.case_results)

    @property
    def automatic_case_count(self) -> int:
        return sum(result.automatic_passed is not None for result in self.case_results)

    @property
    def gpu_samples(self) -> tuple[GpuSample, ...]:
        return tuple(
            sample
            for result in self.case_results
            if result.gpu_report is not None
            for sample in result.gpu_report.samples
        )

    @property
    def peak_gpu_memory_used_mib(self) -> int | None:
        if not self.gpu_samples:
            return None

        return max(sample.memory_used_mib for sample in self.gpu_samples)

    @property
    def peak_gpu_utilization_percent(self) -> int | None:
        if not self.gpu_samples:
            return None

        return max(sample.utilization_percent for sample in self.gpu_samples)


EVALUATION_CASES = (
    EvaluationCase(
        identifier="exact_instruction",
        category="Exact instruction following",
        scoring_method="exact",
        expected_answer="HOME_LLM_EVAL_17",
        prompt=(
            "Reply with exactly the following ASCII text and nothing else: "
            "HOME_LLM_EVAL_17"
        ),
    ),
    EvaluationCase(
        identifier="json_extraction",
        category="Structured JSON extraction",
        scoring_method="json",
        expected_answer={
            "customer": "Mira Chen",
            "items": 3,
            "order_id": "AX-204",
            "paid": True,
        },
        prompt=(
            "Extract the information below into one JSON object. "
            "Return valid JSON only, without Markdown fences or explanation.\n\n"
            "Customer: Mira Chen\n"
            "Order ID: AX-204\n"
            "Items: 3\n"
            "Paid: yes\n\n"
            "Use exactly these keys: customer, items, order_id, paid. "
            "The items value must be an integer and paid must be a boolean."
        ),
    ),
    EvaluationCase(
        identifier="arithmetic_reasoning",
        category="Self-contained arithmetic reasoning",
        scoring_method="exact",
        expected_answer="174",
        prompt=(
            "A warehouse starts with 240 devices. It ships 35 percent "
            "of them and then receives 18 new devices. Return only the "
            "final number of devices, without explanation."
        ),
    ),
    EvaluationCase(
        identifier="summarization",
        category="Summarization",
        scoring_method="manual",
        prompt=(
            "Summarize the following text in at most two sentences. "
            "Preserve the important privacy and hardware facts.\n\n"
            "HomeLLM runs Ollama inside Ubuntu on WSL2. The Ollama HTTP "
            "API is bound to 127.0.0.1 and cloud functionality is disabled. "
            "Inference uses an NVIDIA RTX 5070 with 12 GB of VRAM. The "
            "Qwen 3.5 9B model runs completely on the GPU at a context "
            "length of 4096 tokens."
        ),
    ),
    EvaluationCase(
        identifier="python_debugging",
        category="Python debugging",
        scoring_method="manual",
        prompt=(
            "Explain the bug in this Python function and provide a corrected "
            "version. Keep the complete answer below 120 words.\n\n"
            "def add_item(item, items=[]):\n"
            "    items.append(item)\n"
            "    return items"
        ),
    ),
    EvaluationCase(
        identifier="technical_planning",
        category="Technical planning",
        scoring_method="manual",
        prompt=(
            "A developer has an RTX 5070 with 12 GB VRAM and wants to build "
            "a private assistant over local Markdown documents. Provide "
            "exactly four numbered implementation steps followed by one "
            "line beginning with 'Main risk:'. Do not recommend training a "
            "foundation model from scratch."
        ),
    ),
)


def positive_float(value: str) -> float:
    """Parse a floating-point command argument greater than zero."""

    try:
        parsed_value = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected a number, received {value!r}."
        ) from error

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            f"Expected a value greater than zero, received {parsed_value}."
        )

    return parsed_value


def score_answer(
    evaluation_case: EvaluationCase,
    answer_text: str,
) -> bool | None:
    """Automatically score an answer when the case permits it."""

    if evaluation_case.scoring_method == "manual":
        return None

    if evaluation_case.scoring_method == "exact":
        if not isinstance(evaluation_case.expected_answer, str):
            raise ValueError(
                f"Exact case {evaluation_case.identifier!r} "
                "requires a string expected answer."
            )

        return answer_text.strip() == evaluation_case.expected_answer

    if evaluation_case.scoring_method == "json":
        if not isinstance(evaluation_case.expected_answer, dict):
            raise ValueError(
                f"JSON case {evaluation_case.identifier!r} "
                "requires an object expected answer."
            )

        try:
            parsed_answer = json.loads(answer_text.strip())
        except json.JSONDecodeError:
            return False

        return parsed_answer == evaluation_case.expected_answer

    raise ValueError(f"Unknown scoring method: {evaluation_case.scoring_method!r}.")


def evaluate_case(
    backend: ModelBackend,
    evaluation_case: EvaluationCase,
    *,
    gpu_monitor_factory: GpuMonitorFactory | None = None,
) -> EvaluationCaseResult:
    """Generate and score one independent evaluation answer."""

    gpu_monitor = gpu_monitor_factory() if gpu_monitor_factory is not None else None

    if gpu_monitor is not None:
        gpu_monitor.start()

    gpu_report: GpuReport | None = None
    answer_fragments: list[str] = []
    completion: GenerationComplete | None = None

    try:
        for event in backend.stream_chat(
            (
                ChatMessage(
                    role="user",
                    content=evaluation_case.prompt,
                ),
            )
        ):
            if isinstance(event, TextChunk):
                answer_fragments.append(event.text)
                continue

            if isinstance(event, GenerationComplete):
                if completion is not None:
                    raise RuntimeError(
                        "The backend returned multiple completion events."
                    )

                completion = event
                continue

            raise TypeError(
                "The backend returned an unsupported evaluation event: "
                f"{type(event).__name__}."
            )
    finally:
        if gpu_monitor is not None:
            gpu_report = gpu_monitor.stop()

    answer_text = "".join(answer_fragments)

    if not answer_text.strip():
        raise RuntimeError(
            f"Evaluation case {evaluation_case.identifier!r} "
            "returned no assistant text."
        )

    if completion is None:
        raise RuntimeError(
            f"Evaluation case {evaluation_case.identifier!r} "
            "returned no completion metrics."
        )

    return EvaluationCaseResult(
        case=evaluation_case,
        answer_text=answer_text,
        completion=completion,
        automatic_passed=score_answer(
            evaluation_case,
            answer_text,
        ),
        gpu_report=gpu_report,
    )


def evaluate_model(
    backend: ModelBackend,
    *,
    model_name: str,
    evaluation_cases: Sequence[EvaluationCase],
    gpu_monitor_factory: GpuMonitorFactory | None = None,
) -> ModelEvaluation:
    """Run every evaluation case independently against one model."""

    case_results = tuple(
        evaluate_case(
            backend,
            evaluation_case,
            gpu_monitor_factory=gpu_monitor_factory,
        )
        for evaluation_case in evaluation_cases
    )

    return ModelEvaluation(
        model_name=model_name,
        case_results=case_results,
    )


def list_running_ollama_models(
    *,
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 5.0,
) -> set[str]:
    """Return model names currently loaded by local Ollama."""

    endpoint = f"{base_url.rstrip('/')}/api/ps"
    request = Request(endpoint, method="GET")

    try:
        with urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            document = json.load(response)
    except HTTPError as error:
        response_body = error.read().decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}: {response_body}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not query loaded Ollama models at {endpoint}."
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Ollama returned invalid JSON from /api/ps.") from error

    if not isinstance(document, dict):
        raise RuntimeError("Ollama /api/ps returned JSON that was not an object.")

    model_documents = document.get("models")

    if not isinstance(model_documents, list):
        raise RuntimeError("Ollama /api/ps response did not contain a model list.")

    model_names: set[str] = set()

    for model_document in model_documents:
        if not isinstance(model_document, dict):
            raise RuntimeError("Ollama /api/ps contained an invalid model entry.")

        for field_name in ("name", "model"):
            field_value = model_document.get(field_name)

            if isinstance(field_value, str):
                model_names.add(field_value)

    return model_names


def wait_for_ollama_model_unloaded(
    model_name: str,
    *,
    timeout_seconds: float = 15.0,
    polling_interval_seconds: float = 0.1,
) -> None:
    """Wait until Ollama no longer reports a model as loaded."""

    if timeout_seconds <= 0:
        raise ValueError("Unload timeout must be greater than zero.")

    if polling_interval_seconds <= 0:
        raise ValueError("Unload polling interval must be greater than zero.")

    deadline = time.monotonic() + timeout_seconds

    while True:
        running_models = list_running_ollama_models()

        if model_name not in running_models:
            return

        remaining_seconds = deadline - time.monotonic()

        if remaining_seconds <= 0:
            raise RuntimeError(
                f"Ollama did not unload {model_name!r} within "
                f"{timeout_seconds:.1f} seconds."
            )

        time.sleep(
            min(
                polling_interval_seconds,
                remaining_seconds,
            )
        )


def stop_ollama_model(model_name: str) -> None:
    """Stop a model and wait until Ollama confirms it is unloaded."""

    try:
        completed_process = subprocess.run(
            ["ollama", "stop", model_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
    except FileNotFoundError as error:
        raise RuntimeError("The ollama command was not found.") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"Timed out while stopping Ollama model {model_name!r}."
        ) from error

    if completed_process.returncode != 0:
        error_text = (
            completed_process.stderr.strip() or completed_process.stdout.strip()
        )
        raise RuntimeError(f"Could not unload the evaluated Ollama model: {error_text}")

    wait_for_ollama_model_unloaded(model_name)


def serialize_gpu_report(
    gpu_report: GpuReport | None,
) -> dict[str, object] | None:
    """Convert one GPU report into JSON-compatible values."""

    if gpu_report is None:
        return None

    return {
        "duration_seconds": gpu_report.duration_seconds,
        "error": gpu_report.error,
        "samples": [
            {
                "elapsed_seconds": sample.elapsed_seconds,
                "memory_used_mib": sample.memory_used_mib,
                "memory_total_mib": sample.memory_total_mib,
                "utilization_percent": (sample.utilization_percent),
            }
            for sample in gpu_report.samples
        ],
    }


def serialize_case_result(
    case_result: EvaluationCaseResult,
) -> dict[str, object]:
    """Convert one evaluation result into JSON-compatible values."""

    evaluation_case = case_result.case
    completion = case_result.completion

    return {
        "case": {
            "identifier": evaluation_case.identifier,
            "category": evaluation_case.category,
            "prompt": evaluation_case.prompt,
            "scoring_method": (evaluation_case.scoring_method),
            "expected_answer": (evaluation_case.expected_answer),
        },
        "answer_text": case_result.answer_text,
        "automatic_passed": case_result.automatic_passed,
        "completion": {
            "prompt_tokens": completion.prompt_tokens,
            "generated_tokens": completion.generated_tokens,
            "prompt_duration_ns": (completion.prompt_duration_ns),
            "generation_duration_ns": (completion.generation_duration_ns),
            "total_duration_ns": (completion.total_duration_ns),
            "load_duration_ns": (completion.load_duration_ns),
            "prompt_tokens_per_second": (completion.prompt_tokens_per_second),
            "generation_tokens_per_second": (completion.generation_tokens_per_second),
        },
        "gpu_report": serialize_gpu_report(case_result.gpu_report),
    }


def build_evaluation_artifact(
    evaluations: dict[str, ModelEvaluation],
    manual_preferences: dict[str, str],
    *,
    context_length: int,
    max_output_tokens: int,
    gpu_reporting_enabled: bool,
    created_at: datetime,
) -> dict[str, object]:
    """Build the complete versioned evaluation document."""

    missing_labels = {
        "A",
        "B",
    }.difference(evaluations)

    if missing_labels:
        raise ValueError(
            "Evaluation artifact is missing labels: "
            + ", ".join(sorted(missing_labels))
        )

    manual_wins = {
        "A": sum(preference == "a" for preference in manual_preferences.values()),
        "B": sum(preference == "b" for preference in manual_preferences.values()),
    }

    model_documents: dict[str, object] = {}

    for label in ("A", "B"):
        evaluation = evaluations[label]

        model_documents[label] = {
            "model_name": evaluation.model_name,
            "automatic_pass_count": (evaluation.automatic_pass_count),
            "automatic_case_count": (evaluation.automatic_case_count),
            "manual_wins": manual_wins[label],
            "peak_gpu_memory_used_mib": (evaluation.peak_gpu_memory_used_mib),
            "peak_gpu_utilization_percent": (evaluation.peak_gpu_utilization_percent),
            "case_results": [
                serialize_case_result(case_result)
                for case_result in evaluation.case_results
            ],
        }

    return {
        "schema_version": 1,
        "created_at": created_at.isoformat(),
        "activity": "existing-model inference evaluation",
        "settings": {
            "context_length": context_length,
            "max_output_tokens": max_output_tokens,
            "temperature": EVALUATION_TEMPERATURE,
            "seed": EVALUATION_SEED,
            "thinking_enabled": False,
            "gpu_reporting_enabled": (gpu_reporting_enabled),
        },
        "models": model_documents,
        "manual_preferences": manual_preferences,
        "manual_summary": {
            "wins": manual_wins,
            "ties": sum(
                preference == "tie" for preference in manual_preferences.values()
            ),
            "skipped": sum(
                preference == "skip" for preference in manual_preferences.values()
            ),
        },
    }


def save_evaluation_artifact(
    evaluations: dict[str, ModelEvaluation],
    manual_preferences: dict[str, str],
    *,
    context_length: int,
    max_output_tokens: int,
    gpu_reporting_enabled: bool,
    output_path: Path | None = None,
) -> Path:
    """Write a complete evaluation artifact and return its path."""

    created_at = datetime.now(UTC)

    if output_path is None:
        timestamp = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
        output_path = Path(
            "artifacts",
            "evaluations",
            f"evaluation-{timestamp}.json",
        )

    document = build_evaluation_artifact(
        evaluations,
        manual_preferences,
        context_length=context_length,
        max_output_tokens=max_output_tokens,
        gpu_reporting_enabled=gpu_reporting_enabled,
        created_at=created_at,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            document,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path


def render_case_comparison(
    console: Console,
    case_result_a: EvaluationCaseResult,
    case_result_b: EvaluationCaseResult,
) -> str | None:
    """Show two blinded answers and optionally request a preference."""

    evaluation_case = case_result_a.case

    console.rule(Text(f"{evaluation_case.category} ({evaluation_case.identifier})"))
    console.print(
        Panel(
            Text(evaluation_case.prompt),
            title="Prompt",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            Text(case_result_a.answer_text),
            title="Model A",
            border_style="green",
        )
    )
    console.print(
        Panel(
            Text(case_result_b.answer_text),
            title="Model B",
            border_style="magenta",
        )
    )

    if evaluation_case.scoring_method != "manual":
        status_a = "PASS" if case_result_a.automatic_passed else "FAIL"
        status_b = "PASS" if case_result_b.automatic_passed else "FAIL"

        console.print(
            f"Automatic score: A={status_a}, B={status_b}",
            markup=False,
        )
        return None

    preference = Prompt.ask(
        "Which answer is better?",
        choices=["a", "b", "tie", "skip"],
        default="tie",
        console=console,
    )

    return preference.casefold()


def render_final_summary(
    console: Console,
    evaluations: dict[str, ModelEvaluation],
    manual_preferences: dict[str, str],
) -> None:
    """Reveal model identities and render final scores."""

    manual_wins_a = sum(preference == "a" for preference in manual_preferences.values())
    manual_wins_b = sum(preference == "b" for preference in manual_preferences.values())
    ties = sum(preference == "tie" for preference in manual_preferences.values())
    skipped = sum(preference == "skip" for preference in manual_preferences.values())

    table = Table(title="HomeLLM Quality Evaluation")
    table.add_column("Label")
    table.add_column("Model")
    table.add_column("Automatic")
    table.add_column("Manual wins")
    table.add_column("Peak VRAM")
    table.add_column("Peak GPU")

    for label in ("A", "B"):
        evaluation = evaluations[label]
        peak_memory = evaluation.peak_gpu_memory_used_mib
        peak_utilization = evaluation.peak_gpu_utilization_percent

        table.add_row(
            label,
            evaluation.model_name,
            (f"{evaluation.automatic_pass_count}/{evaluation.automatic_case_count}"),
            str(manual_wins_a if label == "A" else manual_wins_b),
            (f"{peak_memory} MiB" if peak_memory is not None else "not measured"),
            (
                f"{peak_utilization}%"
                if peak_utilization is not None
                else "not measured"
            ),
        )

    console.print()
    console.print(table)
    console.print(
        f"Manual ties: {ties} | skipped: {skipped}",
        markup=False,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create command-line arguments for blind evaluation."""

    parser = argparse.ArgumentParser(
        prog="home-llm-evaluate",
        description=("Run a blind quality comparison between two local models."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        action="append",
        type=local_model_name,
        required=True,
        help=("Installed local model. Provide this argument exactly twice."),
    )
    parser.add_argument(
        "--context-length",
        type=positive_integer,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Context length used for each independent case.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=positive_integer,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum generated tokens for each case.",
    )
    parser.add_argument(
        "--gpu-report",
        action="store_true",
        help="Monitor GPU use separately for every evaluation case.",
    )
    parser.add_argument(
        "--gpu-sample-interval",
        type=positive_float,
        default=DEFAULT_GPU_SAMPLE_INTERVAL,
        help="Seconds between GPU samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "JSON output path. By default, create a timestamped "
            "file under artifacts/evaluations."
        ),
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the blind quality evaluation."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    if len(arguments.model) != 2:
        parser.error("--model must be provided exactly twice.")

    if arguments.model[0] == arguments.model[1]:
        parser.error("The two evaluation models must be different.")

    shuffled_models = list(arguments.model)
    secrets.SystemRandom().shuffle(shuffled_models)

    label_to_model = {
        "A": shuffled_models[0],
        "B": shuffled_models[1],
    }

    gpu_monitor_factory: GpuMonitorFactory | None = None

    if arguments.gpu_report:
        gpu_monitor_factory = partial(
            NvidiaSmiGpuMonitor,
            interval_seconds=arguments.gpu_sample_interval,
        )

    console = Console()
    evaluations: dict[str, ModelEvaluation] = {}

    console.print(
        (
            "Running a blind comparison. Model identities will be "
            "revealed after manual ratings."
        ),
        markup=False,
    )

    for label in ("A", "B"):
        model_name = label_to_model[label]
        console.print(
            f"Generating answers for Model {label}...",
            style="bold cyan",
            markup=False,
        )

        backend = OllamaBackend(
            model=model_name,
            context_length=arguments.context_length,
            max_output_tokens=arguments.max_output_tokens,
            temperature=EVALUATION_TEMPERATURE,
            seed=EVALUATION_SEED,
        )

        try:
            evaluations[label] = evaluate_model(
                backend,
                model_name=model_name,
                evaluation_cases=EVALUATION_CASES,
                gpu_monitor_factory=gpu_monitor_factory,
            )
        finally:
            stop_ollama_model(model_name)

    manual_preferences: dict[str, str] = {}

    for case_index, evaluation_case in enumerate(EVALUATION_CASES):
        preference = render_case_comparison(
            console,
            evaluations["A"].case_results[case_index],
            evaluations["B"].case_results[case_index],
        )

        if preference is not None:
            manual_preferences[evaluation_case.identifier] = preference

    render_final_summary(
        console,
        evaluations,
        manual_preferences,
    )

    try:
        artifact_path = save_evaluation_artifact(
            evaluations,
            manual_preferences,
            context_length=arguments.context_length,
            max_output_tokens=(arguments.max_output_tokens),
            gpu_reporting_enabled=arguments.gpu_report,
            output_path=arguments.output,
        )
    except (OSError, ValueError) as error:
        console.print(
            f"Could not save evaluation artifact: {error}",
            style="bold red",
            markup=False,
        )
        return 1

    console.print(
        f"Evaluation artifact saved to: {artifact_path}",
        style="bold green",
        markup=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
