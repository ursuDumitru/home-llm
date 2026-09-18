"""Versioned conversation files for a single-writer local application."""

import json
import os
import re
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from home_llm.model_backend import ChatMessage
from home_llm.model_registry import ModelProfile, _unique_object


@dataclass(frozen=True, slots=True)
class SavedSession:
    """Text history and effective settings; the profile is not registry permission."""

    session_id: str
    title: str | None
    created_at: str
    updated_at: str
    profile: ModelProfile
    messages: tuple[ChatMessage, ...]
    context_start: int = 0


def _validate_id(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ValueError("Session ID must be 32 lowercase hexadecimal characters.")
    return value


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Session timestamps must be UTC ISO-8601 strings.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Session timestamps must be UTC ISO-8601 strings.") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(None):
        raise ValueError("Session timestamps must include the UTC timezone.")
    return parsed


def _document(session: SavedSession) -> dict:
    profile = asdict(session.profile)
    # Availability belongs to the live registry, not the saved settings.
    del profile["enabled"]
    return {
        "schema_version": 2,
        "session_id": session.session_id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "profile": profile,
        "messages": [asdict(message) for message in session.messages],
        "context_start": session.context_start,
    }


def _decode(document: object) -> SavedSession:
    if not isinstance(document, dict):
        raise ValueError("Session must be a JSON object.")
    version = document.get("schema_version")
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("Unsupported session schema_version; expected 1 or 2.")

    fields = {
        "schema_version",
        "session_id",
        "title",
        "created_at",
        "updated_at",
        "profile",
        "messages",
    }
    if version == 2:
        fields.add("context_start")
    if set(document) != fields:
        raise ValueError(
            f"Session fields are missing or unknown for schema version {version}."
        )

    _validate_id(document["session_id"])
    title = document["title"]
    if title is not None and (not isinstance(title, str) or not title.strip()):
        raise ValueError("Session title must be nonempty text or null.")
    if _timestamp(document["updated_at"]) < _timestamp(document["created_at"]):
        raise ValueError("Session updated_at must not precede created_at.")

    raw_profile = document["profile"]
    if not isinstance(raw_profile, dict):
        raise ValueError("Session profile must be an object.")
    try:
        profile = ModelProfile(enabled=True, **raw_profile)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid saved profile: {error}") from error

    entries = document["messages"]
    if not isinstance(entries, list):
        raise ValueError("Session messages must be an array.")

    messages: list[ChatMessage] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"role", "content"}:
            raise ValueError(f"Message {index} must contain only role and content.")
        role, content = entry["role"], entry["content"]
        if (
            not isinstance(role, str)
            or not isinstance(content, str)
            or not content.strip()
        ):
            raise ValueError(f"Message {index} needs a string role and nonempty text.")
        messages.append(ChatMessage(role=role, content=content))

    # An optional leading system prompt is followed by complete user/assistant pairs.
    start = 1 if messages and messages[0].role == "system" else 0
    turns = messages[start:]
    if len(turns) % 2:
        raise ValueError("Save complete turns only; a user message has no answer.")
    for index, message in enumerate(turns):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message.role != expected_role:
            raise ValueError(
                "Messages must alternate user/assistant after the system prompt."
            )

    context_start = document.get("context_start", 0)
    if type(context_start) is not int or not 0 <= context_start <= len(turns) // 2:
        raise ValueError(
            "context_start must be a completed-turn count within the transcript."
        )

    return SavedSession(
        session_id=document["session_id"],
        title=title,
        created_at=document["created_at"],
        updated_at=document["updated_at"],
        profile=profile,
        messages=tuple(messages),
        context_start=context_start,
    )


def create_session(
    profile: ModelProfile,
    messages: tuple[ChatMessage, ...],
    *,
    title: str | None = None,
    context_start: int = 0,
) -> SavedSession:
    """Create a validated snapshot without writing files or contacting a model."""
    now = datetime.now(UTC).isoformat()
    session = SavedSession(
        uuid4().hex, title, now, now, profile, tuple(messages), context_start
    )
    return _decode(_document(session))


class SessionStore:
    """Save and load JSON text histories, with one writer per session file."""

    def __init__(self, directory: Path = Path("conversations")) -> None:
        self.directory = directory.resolve()

    def _path(self, session_id: str) -> Path:
        path = self.directory / f"{_validate_id(session_id)}.json"
        if path.is_symlink():
            raise ValueError(f"Session file must not be a symbolic link: {path}")
        return path

    def load(self, session_id: str) -> SavedSession:
        """Read a validated session; this does not select or load its model."""
        path = self._path(session_id)
        try:
            document = json.loads(
                path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
            )
            session = _decode(document)
            if session.session_id != session_id:
                raise ValueError("Session ID in the file does not match its filename.")
            return session
        except OSError as error:
            raise RuntimeError(f"Cannot read session {session_id}: {error}") from error
        except ValueError as error:
            raise ValueError(f"Invalid session {session_id}: {error}") from error

    def save(self, session: SavedSession) -> SavedSession:
        """Atomically save complete turns; return the snapshot with its new timestamp."""
        # Validate the input before creating directories or touching an existing file.
        session = _decode(_document(session))
        path = self._path(session.session_id)
        temporary_path: Path | None = None

        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            if path.exists():
                previous = self.load(session.session_id)
                if (
                    previous.created_at != session.created_at
                    or previous.updated_at != session.updated_at
                ):
                    raise ValueError(
                        "Session changed on disk; reload it before saving."
                    )

            saved = replace(session, updated_at=datetime.now(UTC).isoformat())
            document = _document(saved)
            _decode(document)

            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=f".{session.session_id}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(
                    document, temporary, ensure_ascii=False, indent=2, allow_nan=False
                )
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())

            os.replace(temporary_path, path)
            temporary_path = None
            return saved
        except OSError as error:
            raise RuntimeError(
                f"Cannot save session {session.session_id}: {error}. "
                "Check directory permissions and disk space."
            ) from error
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as error:
                    raise RuntimeError(
                        f"Cannot remove temporary session file {temporary_path}: {error}"
                    ) from error

    def list_sessions(self) -> tuple[tuple[SavedSession, ...], tuple[str, ...]]:
        """List newest updates first, reporting corrupt files without hiding valid ones."""
        sessions: list[SavedSession] = []
        errors: list[str] = []
        try:
            paths = sorted(
                path for path in self.directory.iterdir() if path.suffix == ".json"
            )
        except FileNotFoundError:
            return (), ()
        except OSError as error:
            raise RuntimeError(
                f"Cannot list sessions in {self.directory}: {error}"
            ) from error

        for path in paths:
            try:
                sessions.append(self.load(path.stem))
            except (ValueError, RuntimeError) as error:
                errors.append(f"{path.name}: {error}")

        sessions.sort(
            key=lambda item: (_timestamp(item.updated_at), item.session_id),
            reverse=True,
        )
        return tuple(sessions), tuple(errors)
