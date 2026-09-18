"""Load and validate local model profiles from JSON."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """One selectable model and its generation defaults."""

    id: str
    display_name: str
    backend: str
    model_name: str
    enabled: bool
    context_length: int
    max_output_tokens: int
    temperature: float | None
    seed: int | None

    def __post_init__(self) -> None:
        for field in ("id", "display_name", "backend", "model_name"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonempty string.")
            if value != value.strip():
                raise ValueError(f"Remove surrounding whitespace from {field}.")

        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.id) is None:
            raise ValueError("Model ID must use lowercase letters, digits, - or _.")
        if self.backend != "ollama":
            raise ValueError(f"Unsupported backend {self.backend!r}; use 'ollama'.")
        if self.model_name.casefold().endswith("-cloud"):
            raise ValueError("Use a local model name; cloud profiles are disabled.")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a JSON boolean: true or false.")

        for field in ("context_length", "max_output_tokens"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer.")
        if self.max_output_tokens >= self.context_length:
            raise ValueError("max_output_tokens must be smaller than context_length.")

        if self.temperature is not None:
            if type(self.temperature) not in (int, float):
                raise ValueError("temperature must be a number or null.")
            if not 0 <= self.temperature <= 2:
                raise ValueError("Use a temperature between 0 and 2 for this app.")
            if not math.isfinite(self.temperature):
                raise ValueError("temperature must be finite.")
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer or null.")


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    """Validated registry shared by application components."""

    default_model: str
    ollama_base_url: str
    models: tuple[ModelProfile, ...]

    def select(self, model_id: str | None = None) -> ModelProfile:
        """Select an enabled profile, using the default when no ID is supplied."""

        selected_id = self.default_model if model_id is None else model_id
        for profile in self.models:
            if profile.id == selected_id:
                if not profile.enabled:
                    raise ValueError(
                        f"Model {selected_id!r} is disabled; enable it first."
                    )
                return profile
        raise ValueError(f"Unknown model ID {selected_id!r}; choose a configured ID.")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys instead of silently overwriting values."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}; keep only one value.")
        result[key] = value
    return result


def _local_url(value: object) -> str:
    """Accept a loopback HTTP server address without extra URL components."""

    if not isinstance(value, str) or value != value.strip():
        raise ValueError("ollama_base_url must be a string without surrounding spaces.")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.port == 0
    ):
        raise ValueError("Use a local server URL such as http://127.0.0.1:11434.")
    return value.rstrip("/")


def load_model_registry(path: Path) -> ModelRegistry:
    """Read configuration without contacting Ollama or loading model weights."""

    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
        expected_keys = {"schema_version", "default_model", "ollama_base_url", "models"}
        if not isinstance(document, dict) or set(document) != expected_keys:
            raise ValueError(f"Root fields must be exactly {sorted(expected_keys)}.")
        version = document["schema_version"]
        if type(version) is not int or version != 1:
            raise ValueError(f"Unsupported schema_version {version!r}; expected 1.")
        if not isinstance(document["default_model"], str):
            raise ValueError("default_model must be a model ID string.")
        entries = document["models"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("models must be a nonempty array of profiles.")

        profiles: list[ModelProfile] = []
        seen_ids: set[str] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"models[{index}] must be a JSON object.")
            try:
                profile = ModelProfile(**entry)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid models[{index}]: {error}") from error
            if profile.id in seen_ids:
                raise ValueError(
                    f"Duplicate model ID {profile.id!r}; choose unique IDs."
                )
            seen_ids.add(profile.id)
            profiles.append(profile)

        registry = ModelRegistry(
            default_model=document["default_model"],
            ollama_base_url=_local_url(document["ollama_base_url"]),
            models=tuple(profiles),
        )
        registry.select()
        return registry
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot load model registry {path}: {error}") from error
