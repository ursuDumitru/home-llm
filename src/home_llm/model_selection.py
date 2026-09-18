"""Model selection and conversation state, independent of terminal rendering."""

from dataclasses import dataclass, replace
from pathlib import Path

from home_llm.backend_factory import create_backend
from home_llm.chat_session import ChatSession
from home_llm.context_budget import ContextBudget
from home_llm.model_backend import ModelBackend
from home_llm.model_registry import ModelProfile, ModelRegistry, load_model_registry
from home_llm.ollama_runtime import OllamaRuntime, canonical_model_name
from home_llm.registry_store import update_model_enabled
from home_llm.session_store import SavedSession, SessionStore, create_session


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """A configured profile with fresh, possibly unavailable runtime status."""

    profile: ModelProfile
    installed: bool | None
    loaded: bool | None


class ModelSelection:
    """Keep the active profile and conversation together during model switches."""

    def __init__(
        self,
        registry: ModelRegistry,
        profile: ModelProfile,
        backend: ModelBackend,
        *,
        registry_path: Path | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self.registry = registry
        self.profile = profile
        self.session = ChatSession(
            backend,
            context_budget=ContextBudget(
                profile.context_length, profile.max_output_tokens
            ),
        )
        self.runtime = OllamaRuntime(registry.ollama_base_url)
        self._registry_path = (
            registry_path.resolve() if registry_path is not None else None
        )
        self.store = store if store is not None else SessionStore()
        self._record = create_session(profile, ())
        self._persisted = False

    @property
    def has_unsaved_changes(self) -> bool:
        """Compare current content with the last successfully saved snapshot."""
        if not self._persisted:
            return bool(self.session.messages)
        return (
            self.session.messages != self._record.messages
            or self.profile != self._record.profile
            or self.session.context_start != self._record.context_start
        )

    def require_saved(self, *, discard: bool = False) -> None:
        """Refuse replacement unless changes are saved or explicitly discarded."""
        if self.has_unsaved_changes and not discard:
            raise ValueError(
                "Unsaved changes. Use /save first, or repeat the command with --discard."
            )

    def save(self, title: str | None = None) -> SavedSession:
        """Save complete turns, effective settings, and the request-context boundary."""
        candidate = replace(
            self._record,
            profile=self.profile,
            messages=self.session.messages,
            context_start=self.session.context_start,
            title=self._record.title if title is None else title,
        )
        saved = self.store.save(candidate)
        self._record = saved
        self._persisted = True
        return saved

    def new(self, *, discard: bool = False) -> None:
        """Start a separate empty chat on the current profile, without a system prompt."""
        self.require_saved(discard=discard)
        backend = create_backend(
            self.profile, ollama_base_url=self.registry.ollama_base_url
        )
        session = ChatSession(
            backend,
            context_budget=ContextBudget(
                self.profile.context_length, self.profile.max_output_tokens
            ),
        )
        record = create_session(self.profile, ())
        self.session = session
        self._record = record
        self._persisted = False

    def clear(self) -> None:
        """Reset request context while retaining the full, possibly unsaved transcript."""
        self.session.clear()

    def load(self, session_id: str, *, discard: bool = False) -> tuple[str, ...]:
        """Validate and prepare a saved conversation before committing replacement."""
        self.require_saved(discard=discard)
        saved = self.store.load(session_id)
        registry = (
            load_model_registry(self._registry_path)
            if self._registry_path is not None
            else self.registry
        )
        if registry.ollama_base_url != self.registry.ollama_base_url:
            raise ValueError(
                "The Ollama endpoint changed; restart the CLI before loading."
            )

        current = registry.select(saved.profile.id)
        if (
            current.backend != saved.profile.backend
            or current.model_name != saved.profile.model_name
        ):
            raise ValueError(
                "The saved profile now points to a different runtime model. "
                "Restore its original registry mapping before loading this chat."
            )

        differences = tuple(
            f"{field}: registry={getattr(current, field)!r}, saved={getattr(saved.profile, field)!r}"
            for field in ("context_length", "max_output_tokens", "temperature", "seed")
            if getattr(current, field) != getattr(saved.profile, field)
        )

        backend = create_backend(
            saved.profile, ollama_base_url=registry.ollama_base_url
        )
        session = ChatSession.from_messages(
            backend,
            saved.messages,
            context_start=saved.context_start,
            context_budget=ContextBudget(
                saved.profile.context_length, saved.profile.max_output_tokens
            ),
        )
        self.runtime.prepare(saved.profile)

        # Nothing above replaces the active conversation if validation/loading fails.
        self.registry = registry
        self.profile = saved.profile
        self.session = session
        self._record = saved
        self._persisted = True
        return differences

    def set_enabled(self, model_id: str, *, enabled: bool) -> None:
        """Update in-memory configuration only after the file save succeeds."""
        if self._registry_path is None:
            raise ValueError("Registry updates require a registry file path.")

        self.registry = update_model_enabled(
            self._registry_path,
            self.registry,
            model_id,
            enabled=enabled,
            active_model_id=self.profile.id,
        )

    def list_models(self) -> tuple[tuple[ModelStatus, ...], tuple[str, ...]]:
        """Query each inventory independently; retain known status on partial failure."""
        errors: list[str] = []
        installed = loaded = None
        try:
            installed = self.runtime.model_names()
        except RuntimeError as error:
            errors.append(f"Installed status unavailable: {error}")
        try:
            loaded = self.runtime.model_names(loaded=True)
        except RuntimeError as error:
            errors.append(f"Loaded status unavailable: {error}")

        rows = tuple(
            ModelStatus(
                profile=profile,
                installed=(
                    None
                    if installed is None
                    else canonical_model_name(profile.model_name) in installed
                ),
                loaded=(
                    None
                    if loaded is None
                    else canonical_model_name(profile.model_name) in loaded
                ),
            )
            for profile in self.registry.models
        )
        return rows, tuple(errors)

    def select(self, model_id: str) -> None:
        """Commit a new empty session only after validation and loading succeed."""
        profile = self.registry.select(model_id)
        if profile.id == self.profile.id:
            return

        if self.session.messages:
            raise ValueError(
                "Model switching requires an empty conversation, including no system "
                "prompt. Use /save if needed, then /new before switching."
            )

        backend = create_backend(profile, ollama_base_url=self.registry.ollama_base_url)
        self.runtime.prepare(profile)
        session = ChatSession(
            backend,
            context_budget=ContextBudget(
                profile.context_length, profile.max_output_tokens
            ),
        )
        record = create_session(profile, ())

        self.profile = profile
        self.session = session
        self._record = record
        self._persisted = False
