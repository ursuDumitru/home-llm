"""Persist model availability for a single-writer local CLI."""

import json
import os
import stat
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile

from home_llm.model_registry import ModelRegistry, load_model_registry


def update_model_enabled(
    path: Path,
    expected: ModelRegistry,
    model_id: str,
    *,
    enabled: bool,
    active_model_id: str,
) -> ModelRegistry:
    """Validate one flag change, atomically save it, and return the saved registry."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean.")

    profile = next((item for item in expected.models if item.id == model_id), None)
    if profile is None:
        raise ValueError(f"Unknown model ID {model_id!r}; use /models to list IDs.")

    if not enabled:
        if model_id == active_model_id:
            raise ValueError(
                "Cannot disable the active profile; select another model first."
            )
        if model_id == expected.default_model:
            raise ValueError(
                "Cannot disable the default profile. Change default_model in the "
                "registry to another enabled ID, then restart the CLI."
            )

    temporary_path: Path | None = None
    try:
        path = path.resolve(strict=True)
        if load_model_registry(path) != expected:
            raise ValueError(
                "The registry changed on disk. Restart the CLI before updating it; "
                "no changes were saved."
            )
        if profile.enabled == enabled:
            return expected

        updated = replace(
            expected,
            models=tuple(
                replace(item, enabled=enabled) if item.id == model_id else item
                for item in expected.models
            ),
        )
        document = {
            "schema_version": 1,
            "default_model": updated.default_model,
            "ollama_base_url": updated.ollama_base_url,
            "models": [asdict(item) for item in updated.models],
        }

        mode = stat.S_IMODE(path.stat().st_mode)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                document, temporary, indent=2, ensure_ascii=False, allow_nan=False
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())

        # Reuse the complete schema validator before replacing the original.
        validated = load_model_registry(temporary_path)
        temporary_path.chmod(mode)
        os.replace(temporary_path, path)
        temporary_path = None
        return validated
    except OSError as error:
        raise RuntimeError(
            f"Could not save model registry {path}: {error}. "
            "Check directory permissions and available disk space, then retry."
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as error:
                raise RuntimeError(
                    f"Could not remove temporary registry file {temporary_path}: {error}"
                ) from error
