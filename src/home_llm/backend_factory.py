"""Create runtime adapters from validated model profiles."""

from home_llm.model_backend import ModelBackend
from home_llm.model_registry import ModelProfile
from home_llm.ollama_backend import OllamaBackend


def create_backend(profile: ModelProfile, *, ollama_base_url: str) -> ModelBackend:
    """Create an adapter without downloading or loading model weights."""

    if not profile.enabled:
        raise ValueError(f"Model {profile.id!r} is disabled; enable it first.")
    if profile.backend != "ollama":
        raise ValueError(f"Unsupported backend {profile.backend!r}.")

    return OllamaBackend(
        model=profile.model_name,
        base_url=ollama_base_url,
        context_length=profile.context_length,
        max_output_tokens=profile.max_output_tokens,
        temperature=profile.temperature,
        seed=profile.seed,
    )
