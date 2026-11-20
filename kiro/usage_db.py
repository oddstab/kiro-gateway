"""SQLite-based usage tracking for kiro-gateway.

Records every API request's token usage into a local SQLite database.
Designed to be consumed by Grafana via the frser-sqlite-datasource plugin.

Architecture notes:
- One module-level SQLite connection guarded by threading.Lock.
- All public functions are async; blocking work is offloaded via asyncio.to_thread
  so the FastAPI event loop is never blocked.
- Recording never raises: all errors are swallowed and logged so that a DB
  failure cannot affect the in-flight request.
"""

import asyncio
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

# Module-level shared state guarded by _lock.
_conn: sqlite3.Connection | None = None
_lock: threading.Lock = threading.Lock()


# --------------------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------------------


def _db_path() -> Path:
    """Return the configured DB path as a Path object.

    Reads kiro.config at call time so that tests can monkeypatch the value
    without having to reload the module.

    Returns:
        Configured USAGE_DB_PATH as an absolute or relative Path.
    """
    from kiro.config import USAGE_DB_PATH  # local import: must be deferrable for tests

    return Path(USAGE_DB_PATH)


def _is_enabled() -> bool:
    """Return True when usage tracking is enabled in config.

    Returns:
        Value of kiro.config.USAGE_DB_ENABLED.
    """
    from kiro.config import USAGE_DB_ENABLED

    return USAGE_DB_ENABLED


def _get_conn() -> sqlite3.Connection:
    """Return the module-level connection, creating it on first call.

    Must be called while holding _lock.

    Returns:
        An open sqlite3.Connection in WAL mode.

    Raises:
        OSError: If the parent directory cannot be created.
        sqlite3.Error: If the connection or schema init fails.
    """
    global _conn
    if _conn is not None:
        return _conn

    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), check_same_thread=False)
    # WAL allows concurrent reads while a write is in progress, which is
    # essential when Grafana queries the DB at the same time the gateway writes.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    _init_schema(conn)
    _conn = conn
    logger.info("[UsageDB] Initialized at {}", path)
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they do not yet exist.

    Args:
        conn: An open SQLite connection.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            ts_epoch REAL NOT NULL,
            api_format TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            is_stream INTEGER NOT NULL DEFAULT 0,
            model TEXT NOT NULL,
            resolved_model TEXT,
            account_id TEXT,
            account_label TEXT,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            cache_read_input_tokens INTEGER DEFAULT 0,
            cache_creation_input_tokens INTEGER DEFAULT 0,
            context_usage_percentage REAL,
            credits_used REAL,
            token_source TEXT,
            finish_reason TEXT,
            duration_ms REAL DEFAULT 0,
            status_code INTEGER DEFAULT 200,
            error_message TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_usage_ts_epoch ON usage_log(ts_epoch);
        CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_log(model);
        CREATE INDEX IF NOT EXISTS idx_usage_account ON usage_log(account_label);
    """)
    conn.commit()


def _derive_account_label(account_id: str | None) -> str | None:
    """Derive a short dashboard label from an account_id path.

    Rules:
    - None in → None out.
    - If account_id contains a path separator (/ or \\), return the file
      basename without extension.
    - Otherwise return account_id unchanged (e.g. "refresh_token_abc123").

    Args:
        account_id: Raw account identifier, possibly a filesystem path.

    Returns:
        Short label string, or None if account_id is None.
    """
    if account_id is None:
        return None
    if "/" in account_id or "\\" in account_id:
        return Path(account_id).stem
    return account_id


def _record_usage_sync(fields: dict[str, Any]) -> None:
    """Insert one usage row synchronously (blocking).

    This private function exists so tests can drive it without an event loop.
    It swallows all errors and logs them rather than propagating, ensuring that
    a DB failure never affects the calling request.

    Applies normalization before writing:
    - error_message is truncated to 500 chars.
    - total_tokens is set to prompt+completion when passed as 0 with non-zero
      prompt or completion.

    Args:
        fields: Column values for the row; missing optional fields default to
                None / 0 as appropriate.
    """
    # Work on a shallow copy so we never mutate the caller's dict.
    fields = dict(fields)

    # Normalize error_message length.
    msg = fields.get("error_message")
    if msg is not None and len(msg) > 500:
        fields["error_message"] = msg[:500]

    # Auto-sum total_tokens when not explicitly provided.
    if not fields.get("total_tokens"):
        prompt = fields.get("prompt_tokens") or 0
        completion = fields.get("completion_tokens") or 0
        if prompt or completion:
            fields["total_tokens"] = prompt + completion

    try:
        with _lock:
            conn = _get_conn()
            conn.execute(
                """
                INSERT INTO usage_log (
                    ts, ts_epoch, api_format, endpoint, is_stream,
                    model, resolved_model, account_id, account_label,
                    prompt_tokens, completion_tokens, total_tokens,
                    cache_read_input_tokens, cache_creation_input_tokens,
                    context_usage_percentage, credits_used, token_source,
                    finish_reason, duration_ms, status_code, error_message
                ) VALUES (
                    :ts, :ts_epoch, :api_format, :endpoint, :is_stream,
                    :model, :resolved_model, :account_id, :account_label,
                    :prompt_tokens, :completion_tokens, :total_tokens,
                    :cache_read_input_tokens, :cache_creation_input_tokens,
                    :context_usage_percentage, :credits_used, :token_source,
                    :finish_reason, :duration_ms, :status_code, :error_message
                )
                """,
                fields,
            )
            conn.commit()
    except sqlite3.Error as e:
        logger.error(
            "[UsageDB] Failed to record usage for model={}: {}",
            fields.get("model"),
            e,
        )
    except OSError as e:
        logger.error("[UsageDB] Filesystem error writing {}: {}", _db_path(), e)


def _sync_readonly_copy_sync() -> None:
    """Copy the main DB to a DELETE-mode readonly file (blocking).

    Grafana runs inside a Docker container that bind-mounts a Windows NTFS
    path via v9fs (e.g. /mnt/d/...).  WAL mode requires mmap'd shared memory
    through the -shm sidecar file.  v9fs over a Docker bind mount does not
    support the mmap semantics needed for -shm, so any container process
    that opens a WAL database on that mount gets ``disk I/O error``.

    DELETE journal mode uses no -shm file, so the same container can open the
    copy without error.  We therefore keep the live DB in WAL mode (good for
    the host-side writer) and produce a fresh DELETE-mode snapshot for
    the container-side reader on every sync interval.

    sqlite3's backup() reads the destination header before copying, so a
    snapshot left truncated by an earlier crash or partial write makes every
    later sync fail with "database disk image is malformed" indefinitely.  On
    that error the destination is emptied and the backup retried once, so the
    sync recovers on its own instead of needing the file removed by hand.

    Recovery truncates the file rather than deleting or renaming it, because
    Grafana keeps the snapshot open and Windows refuses to unlink or rename a
    file that another process holds open.  A zero-length file is a valid empty
    SQLite database, so backup() accepts it.
    """
    db = _db_path()
    dst_path = db.parent / "usage-readonly.db"
    try:
        with _lock:
            if _conn is None:
                return
            try:
                _backup_to(dst_path)
            except sqlite3.OperationalError:
                # Transient (locked, busy, cannot open): the file itself may be
                # fine, so keep the last good snapshot and let the next sync try.
                raise
            except sqlite3.DatabaseError as e:
                # SQLITE_CORRUPT / SQLITE_NOTADB surface as plain DatabaseError.
                logger.warning(
                    "[UsageDB] Readonly copy at {} is unusable ({}), rebuilding it",
                    dst_path,
                    e,
                )
                with open(dst_path, "wb"):
                    pass  # opening for write truncates to zero length
                _backup_to(dst_path)
        logger.debug("[UsageDB] Readonly copy synced to {}", dst_path)
    except sqlite3.Error as e:
        logger.error("[UsageDB] Failed to sync readonly copy: {}", e)
    except OSError as e:
        logger.error("[UsageDB] Filesystem error syncing readonly copy to {}: {}", dst_path, e)


def _backup_to(dst_path: Path) -> None:
    """Back up the live connection into dst_path using DELETE journal mode.

    Must be called while holding _lock with _conn already open.

    Args:
        dst_path: Destination database file; created if it does not exist.

    Raises:
        sqlite3.Error: If the destination cannot be opened or written.
    """
    dst = sqlite3.connect(str(dst_path))
    try:
        _conn.backup(dst)  # type: ignore[union-attr]  # caller guarantees _conn is open
        dst.execute("PRAGMA journal_mode=DELETE")
        dst.commit()
    finally:
        dst.close()


# --------------------------------------------------------------------------------------------------
# Public async API
# --------------------------------------------------------------------------------------------------


async def record_usage(
    *,
    api_format: str,
    endpoint: str,
    is_stream: bool,
    model: str,
    status_code: int,
    duration_ms: float,
    resolved_model: str | None = None,
    account_id: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    context_usage_percentage: float | None = None,
    credits_used: float | None = None,
    token_source: str | None = None,
    finish_reason: str | None = None,
    error_message: str | None = None,
) -> None:
    """Record one API request into the usage database.

    Returns immediately when USAGE_DB_ENABLED is false.  All DB work is
    offloaded to a thread pool so the FastAPI event loop is not blocked.

    Args:
        api_format: Wire protocol used by the client ('openai' or 'anthropic').
        endpoint: Request path (e.g. '/v1/chat/completions').
        is_stream: True if the response was streamed.
        model: Model name as sent by the client.
        status_code: HTTP status code returned to the client.
        duration_ms: Total round-trip duration in milliseconds.
        resolved_model: Normalized model name from model_resolver, if available.
        account_id: Full credentials path identifying the upstream account.
        prompt_tokens: Input token count (0 if unknown).
        completion_tokens: Output token count (0 if unknown).
        total_tokens: Total token count; auto-computed from prompt+completion
            when 0 is passed but at least one of prompt/completion is non-zero.
        cache_read_input_tokens: Tokens served from prompt cache.
        cache_creation_input_tokens: Tokens written into prompt cache.
        context_usage_percentage: Kiro-specific context window saturation (nullable).
        credits_used: Kiro metering credit amount (nullable).
        token_source: Origin of token counts ('context_usage', 'tiktoken', or 'unknown').
        finish_reason: Model stop reason (e.g. 'stop', 'length').
        error_message: Error text (truncated to 500 chars if longer).
    """
    if not _is_enabled():
        return

    now = datetime.now(timezone.utc)

    fields: dict[str, Any] = {
        "ts": now.isoformat(),
        "ts_epoch": now.timestamp(),
        "api_format": api_format,
        "endpoint": endpoint,
        "is_stream": 1 if is_stream else 0,
        "model": model,
        "resolved_model": resolved_model,
        "account_id": account_id,
        "account_label": _derive_account_label(account_id),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "context_usage_percentage": context_usage_percentage,
        "credits_used": credits_used,
        "token_source": token_source,
        "finish_reason": finish_reason,
        "duration_ms": duration_ms,
        "status_code": status_code,
        "error_message": error_message,
    }

    await asyncio.to_thread(_record_usage_sync, fields)


async def record_request_usage(
    *,
    api_format: str,
    endpoint: str,
    is_stream: bool,
    model: str,
    status_code: int,
    duration_ms: float,
    resolved_model: str | None = None,
    account_id: str | None = None,
    usage: dict[str, Any] | None = None,
    finish_reason_fallback: str | None = None,
    error_message: str | None = None,
) -> None:
    """Record one request, reading token counts from a usage mapping.

    Convenience wrapper over record_usage for the route handlers. Both the
    non-streaming response bodies and the streaming usage_sink expose token
    counts as a plain mapping, but under different key names per API, so this
    normalizes both shapes into the DB columns in one place instead of
    repeating the field plumbing at every call site.

    Accepted key spellings:
    - OpenAI style: prompt_tokens, completion_tokens, total_tokens
    - Anthropic style: input_tokens, output_tokens
    - Shared extras: cache_read_input_tokens, cache_creation_input_tokens,
      context_usage_percentage, credits_used, token_source, finish_reason

    An empty or missing usage mapping records zeros, which is the intended
    behaviour for error rows and for streams that failed before producing
    any usage data.

    Args:
        api_format: Wire protocol used by the client ('openai' or 'anthropic').
        endpoint: Request path (e.g. '/v1/chat/completions').
        is_stream: True if the response was streamed.
        model: Model name as sent by the client.
        status_code: HTTP status code returned to the client.
        duration_ms: Round-trip duration in milliseconds.
        resolved_model: Normalized model name from model_resolver, if available.
        account_id: Full credentials path identifying the upstream account.
        usage: Mapping of token counts in either API's spelling.
        finish_reason_fallback: Stop reason to use when the usage mapping has
            none. Anthropic non-streaming responses carry stop_reason at the
            top level rather than inside usage, so the caller supplies it here.
        error_message: Error text (truncated to 500 chars if longer).
    """
    usage = usage or {}

    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = usage.get("input_tokens") or 0

    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = usage.get("output_tokens") or 0

    await record_usage(
        api_format=api_format,
        endpoint=endpoint,
        is_stream=is_stream,
        model=model,
        status_code=status_code,
        duration_ms=duration_ms,
        resolved_model=resolved_model,
        account_id=account_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=usage.get("total_tokens") or 0,
        cache_read_input_tokens=usage.get("cache_read_input_tokens") or 0,
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens") or 0,
        context_usage_percentage=usage.get("context_usage_percentage"),
        credits_used=usage.get("credits_used"),
        token_source=usage.get("token_source"),
        finish_reason=usage.get("finish_reason") or finish_reason_fallback,
        error_message=error_message,
    )


async def sync_readonly_copy() -> None:
    """Produce a DELETE-mode snapshot of the main DB for Grafana.

    See _sync_readonly_copy_sync for the full explanation of why this is
    necessary when Grafana runs inside a Docker container on a v9fs bind mount.

    No-op when USAGE_DB_ENABLED is false or when the main DB has not yet been
    created (e.g. no requests have been processed since startup).
    """
    if not _is_enabled():
        return
    if _conn is None:
        return
    await asyncio.to_thread(_sync_readonly_copy_sync)


def close() -> None:
    """Close the module-level SQLite connection and reset module state.

    Safe to call multiple times or when the connection was never opened.
    Intended for use in tests and graceful shutdown handlers.
    """
    global _conn
    with _lock:
        if _conn is not None:
            try:
                _conn.close()
            except sqlite3.Error as e:
                logger.warning("[UsageDB] Error closing connection: {}", e)
            _conn = None
