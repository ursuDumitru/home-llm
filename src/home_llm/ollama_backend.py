"""Local Ollama implementation of the model backend interface."""

import json
from collections.abc import Iterator, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from home_llm.model_backend import (
    ChatMessage,
    GenerationComplete,
    StreamEvent,
    TextChunk,
)


class OllamaBackend:
    """Stream chat responses from a locally running Ollama server."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        context_length: int = 4096,
        max_output_tokens: int = 512,
        timeout_seconds: float = 300.0,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> None:
        parsed_url = urlparse(base_url)

        if parsed_url.scheme != "http":
            raise ValueError("The Ollama URL must use the local HTTP protocol.")

        if parsed_url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                "The Ollama URL must use a loopback hostname such as 127.0.0.1."
            )

        if not model:
            raise ValueError("The Ollama model name must not be empty.")

        if context_length <= 0:
            raise ValueError("Context length must be greater than zero.")

        if max_output_tokens <= 0:
            raise ValueError("Maximum output tokens must be greater than zero.")

        if temperature is not None and temperature < 0:
            raise ValueError("Temperature must not be negative.")

        if seed is not None and seed < 0:
            raise ValueError("Random seed must not be negative.")

        self._model = model
        self._endpoint = f"{base_url.rstrip('/')}/api/chat"
        self._context_length = context_length
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._seed = seed

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
    ) -> Iterator[StreamEvent]:
        """Send messages to Ollama and yield generic streaming events."""

        if not messages:
            raise ValueError("At least one chat message is required.")

        options: dict[str, int | float] = {
            "num_ctx": self._context_length,
            "num_predict": self._max_output_tokens,
        }

        if self._temperature is not None:
            options["temperature"] = self._temperature

        if self._seed is not None:
            options["seed"] = self._seed

        payload = {
            "model": self._model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "stream": True,
            "think": False,
            "keep_alive": "10m",
            "options": options,
        }

        request = Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue

                    document = self._decode_response_line(raw_line)
                    message = document.get("message", {})
                    text = message.get("content", "")

                    if text:
                        yield TextChunk(text=text)

                    if document.get("done") is True:
                        yield GenerationComplete(
                            prompt_tokens=int(document.get("prompt_eval_count", 0)),
                            generated_tokens=int(document.get("eval_count", 0)),
                            prompt_duration_ns=int(
                                document.get("prompt_eval_duration", 0)
                            ),
                            generation_duration_ns=int(
                                document.get("eval_duration", 0)
                            ),
                            total_duration_ns=int(document.get("total_duration", 0)),
                            load_duration_ns=int(document.get("load_duration", 0)),
                        )
                        return

        except HTTPError as error:
            response_body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama returned HTTP {error.code}: {response_body}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self._endpoint}. Check that the ollama service is active."
            ) from error

        raise RuntimeError("Ollama ended the response without a completion event.")

    @staticmethod
    def _decode_response_line(raw_line: bytes) -> dict[str, Any]:
        try:
            document = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Ollama returned an invalid streaming JSON response."
            ) from error

        if not isinstance(document, dict):
            raise RuntimeError("Ollama returned JSON that was not an object.")

        return document
