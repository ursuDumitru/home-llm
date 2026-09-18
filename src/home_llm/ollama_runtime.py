"""Local Ollama inventory and model-loading operations."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from home_llm.model_registry import ModelProfile, _local_url


def canonical_model_name(name: str) -> str:
    """Match an omitted Ollama tag to its default :latest tag."""
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


class OllamaRuntime:
    """Inspect local runtime state and preload installed models."""

    def __init__(self, base_url: str) -> None:
        self._base_url = _local_url(base_url)

    def model_names(self, *, loaded: bool = False) -> frozenset[str]:
        """Return installed or loaded names; failures are not empty inventories."""
        path = "/api/ps" if loaded else "/api/tags"
        document = self._request(path)
        entries = document.get("models")
        if not isinstance(entries, list):
            raise RuntimeError(f"Ollama {path} returned an invalid model list.")

        names: set[str] = set()
        for entry in entries:
            name = entry.get("name") if isinstance(entry, dict) else None
            if not isinstance(name, str) or not name.strip():
                raise RuntimeError(f"Ollama {path} returned an invalid model name.")
            names.add(canonical_model_name(name))
        return frozenset(names)

    def prepare(self, profile: ModelProfile) -> None:
        """Check local installation, then load without adding a chat turn."""
        if canonical_model_name(profile.model_name) not in self.model_names():
            raise ValueError(
                f"Model {profile.model_name!r} is not installed in this Ollama server. "
                "Install it explicitly or choose an installed profile with /models. "
                "No download was started."
            )

        document = self._request(
            "/api/chat",
            payload={
                "model": profile.model_name,
                "messages": [],
                "stream": False,
                "think": False,
                "keep_alive": "10m",
                "options": {"num_ctx": profile.context_length},
            },
            timeout=180.0,
        )
        if document.get("done") is not True:
            raise RuntimeError("Ollama did not confirm model loading; try again.")

    def _request(
        self,
        path: str,
        *,
        payload: dict | None = None,
        timeout: float = 5.0,
    ) -> dict:
        request = Request(
            f"{self._base_url}{path}",
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                document = json.load(response)
        except HTTPError as error:
            raise RuntimeError(
                f"Ollama {path} returned HTTP {error.code}. "
                "Check the model installation and Ollama service logs."
            ) from error
        except (URLError, OSError) as error:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._base_url}, or the request timed out. "
                "Check that the ollama service is active, then retry."
            ) from error
        except ValueError as error:
            raise RuntimeError(f"Ollama {path} returned invalid JSON.") from error

        if not isinstance(document, dict):
            raise RuntimeError(f"Ollama {path} must return a JSON object.")
        if document.get("error"):
            raise RuntimeError(f"Ollama {path}: {document['error']}")
        return document
