# -*- coding: utf-8 -*-

"""
Unit tests for kiro/usage_db.py.

All tests are isolated: they monkeypatch USAGE_DB_PATH and USAGE_DB_ENABLED
so nothing is written inside the repo directory.  An autouse fixture resets
module-level connection state between every test to prevent cross-test
interference.
"""

import asyncio
import sqlite3
from pathlib import Path

import pytest


# --------------------------------------------------------------------------------------------------
# Autouse fixture: reset module state between tests
# --------------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_usage_db():
    """Close the module-level connection and reset _conn to None before each test.

    Without this, a connection opened by one test can leak into the next,
    causing confusing failures when USAGE_DB_PATH is remapped by monkeypatch.
    """
    import kiro.usage_db as udb

    udb.close()
    yield
    udb.close()


# --------------------------------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------------------------------


def _enable_db(monkeypatch, tmp_path: Path) -> Path:
    """Configure usage_db to write to a temp directory and enable it.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        tmp_path: Temporary directory provided by pytest.

    Returns:
        The Path to the configured database file.
    """
    db_path = tmp_path / "usage.db"
    monkeypatch.setattr("kiro.usage_db._is_enabled", lambda: True)
    monkeypatch.setattr("kiro.usage_db._db_path", lambda: db_path)
    return db_path


def _make_fields(**overrides: object) -> dict:
    """Return a minimal valid fields dict for _record_usage_sync.

    Args:
        **overrides: Column values to override in the defaults.

    Returns:
        A dict ready to pass to _record_usage_sync.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    defaults = {
        "ts": now.isoformat(),
        "ts_epoch": now.timestamp(),
        "api_format": "openai",
        "endpoint": "/v1/chat/completions",
        "is_stream": 0,
        "model": "test-model",
        "resolved_model": None,
        "account_id": None,
        "account_label": None,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "context_usage_percentage": None,
        "credits_used": None,
        "token_source": None,
        "finish_reason": None,
        "duration_ms": 100.0,
        "status_code": 200,
        "error_message": None,
    }
    defaults.update(overrides)
    return defaults


# --------------------------------------------------------------------------------------------------
# TestDisabledBehavior
# --------------------------------------------------------------------------------------------------


class TestDisabledBehavior:
    """Tests verifying that no files are created when tracking is disabled."""

    @pytest.mark.asyncio
    async def test_disabled_creates_no_file(self, tmp_path: Path, monkeypatch) -> None:
        """
        What it does: Calls record_usage when USAGE_DB_ENABLED=false.
        Purpose: Verify that no DB file or directory is created.
        """
        print("Setup: USAGE_DB_ENABLED=false, pointing at tmp_path...")
        db_path = tmp_path / "sub" / "usage.db"
        monkeypatch.setattr("kiro.usage_db._is_enabled", lambda: False)
        monkeypatch.setattr("kiro.usage_db._db_path", lambda: db_path)

        print("Action: calling record_usage...")
        import kiro.usage_db as udb
        await udb.record_usage(
            api_format="openai",
            endpoint="/v1/chat/completions",
            is_stream=False,
            model="x",
            status_code=200,
            duration_ms=1.0,
        )

        print("Assert: no file and no sub-directory created...")
        assert not db_path.exists(), "DB file must not be created when disabled"
        assert not (tmp_path / "sub").exists(), "DB directory must not be created when disabled"

    @pytest.mark.asyncio
    async def test_disabled_sync_readonly_copy_no_raise(self, tmp_path: Path, monkeypatch) -> None:
        """
        What it does: Calls sync_readonly_copy when disabled.
        Purpose: Verify no exception is raised and no file is created.
        """
        print("Setup: disabled...")
        monkeypatch.setattr("kiro.usage_db._is_enabled", lambda: False)
        monkeypatch.setattr("kiro.usage_db._db_path", lambda: tmp_path / "usage.db")

        import kiro.usage_db as udb
        await udb.sync_readonly_copy()  # must not raise
        assert not (tmp_path / "usage-readonly.db").exists()


# --------------------------------------------------------------------------------------------------
# TestSchema
# --------------------------------------------------------------------------------------------------


class TestSchema:
    """Tests verifying that the database schema is created correctly."""

    def test_all_columns_exist(self, tmp_path: Path, monkeypatch) -> None:
        """
        What it does: Initialises the DB and inspects PRAGMA table_info.
        Purpose: Every column defined in the spec must be present.
        """
        print("Setup: enable DB at tmp_path...")
        db_path = _enable_db(monkeypatch, tmp_path)

        import kiro.usage_db as udb

        with udb._lock:
            conn = udb._get_conn()

        print("Action: query column names via PRAGMA...")
        rows = conn.execute("PRAGMA table_info(usage_log)").fetchall()
        columns = {row[1] for row in rows}

        expected = {
            "id", "ts", "ts_epoch", "api_format", "endpoint", "is_stream",
            "model", "resolved_model", "account_id", "account_label",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "cache_read_input_tokens", "cache_creation_input_tokens",
            "context_usage_percentage", "credits_used", "token_source",
            "finish_reason", "duration_ms", "status_code", "error_message",
        }
        print(f"Assert: expected columns present, got {columns}...")
        assert expected.issubset(columns)

    def test_all_indexes_exist(self, tmp_path: Path, monkeypatch) -> None:
        """
        What it does: Checks sqlite_master for the three required indexes.
        Purpose: Grafana time-range queries depend on these indexes.
        """
        print("Setup: enable DB...")
        db_path = _enable_db(monkeypatch, tmp_path)

        import kiro.usage_db as udb

        with udb._lock:
            conn = udb._get_conn()

        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='usage_log'"
        ).fetchall()
        index_names = {row[0] for row in rows}

        print(f"Assert: three named indexes present, got {index_names}...")
        assert "idx_usage_ts_epoch" in index_names
        assert "idx_usage_model" in index_names
        assert "idx_usage_account" in index_names

    def test_repeated_init_is_idempotent(self, tmp_path: Path, monkeypatch) -> None:
        """
        What it does: Calls _get_conn twice and calls _init_schema twice.
        Purpose: CREATE TABLE IF NOT EXISTS must not raise on repeated calls.
        """
        print("Setup: enable DB...")
        db_path = _enable_db(monkeypatch, tmp_path)

        import kiro.usage_db as udb

        with udb._lock:
            conn = udb._get_conn()

        print("Action: call _init_schema again on same connection...")
        udb._init_schema(conn)  # must not raise

        rows = conn.execute("SELECT COUNT(*) FROM usage_log").fetchone()
        print(f"Assert: table still exists and empty, count={rows[0]}...")
        assert rows[0] == 0


# --------------------------------------------------------------------------------------------------
# TestRoundTrip
# --------------------------------------------------------------------------------------------------


class TestRoundTrip:
    """Tests verifying that written rows can be read back with correct values."""

    def test_fully_populated_row(self, tmp_path: Path, monkeypatch) -> None:
        """
        What it does: Writes a fully-populated row and reads every field back.
        Purpose: Confirm serialization fidelity for all column types.
        """
        print("Setup: enable DB...")
        db_path = _enable_db(monkeypatch, tmp_path)

        import kiro.usage_db as udb

        fields = _make_fields(
            api_format="anthropic",
            endpoint="/v1/messages",
            is_stream=1,
            model="claude-3",
            resolved_model="claude-3-5-sonnet",
            account_id="/creds/prod.json",
            account_label="prod",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cache_read_input_tokens=20,
            cache_creation_input_tokens=5,
            context_usage_percentage=42.5,
            credits_used=0.005,
            token_source="tiktoken",
            finish_reason="stop",
            duration_ms=250.5,
            status_code=200,
            error_message=None,
        )

        print("Action: write row via _record_usage_sync...")
        udb._record_usage_sync(fields)

        print("Assert: read back and compare all fields...")
        with udb._lock:
            conn = udb._get_conn()
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM usage_log LIMIT 1").fetchone()
        conn.row_factory = None

        assert row["api_format"] == "anthropic"
        assert row["endpoint"] == "/v1/messages"
        assert row["is_stream"] == 1
        assert row["model"] == "claude-3"
        assert row["resolved_model"] == "claude-3-5-sonnet"
        assert row["account_id"] == "/creds/prod.json"
        assert row["account_label"] == "prod"
        assert row["prompt_tokens"] == 100
        assert row["completion_tokens"] == 50
        assert row["total_tokens"] == 150
        assert row["cache_read_input_tokens"] == 20
        assert row["cache_creation_input_tokens"] == 5
        assert abs(row["context_usage_percentage"] - 42.5) < 1e-9
        assert abs(row["credits_used"] - 0.005) < 1e-9
        assert row["token_source"] == "tiktoken"
        assert row["finish_reason"] == "stop"
        assert abs(row["duration_ms"] - 250.5) < 1e-9
        assert row["status_code"] == 200
        assert row["error_message"] is None


# --------------------------------------------------------------------------------------------------
# TestNullHandling
# --------------------------------------------------------------------------------------------------


class TestNullHandling:
    """Tests verifying that optional columns store SQL NULL (not 0) when not provided."""

    def test_nullable_columns_store_sql_null(self, tmp_path: Path, monkeypatch) -> None:
        """
        What it does: Writes a row with None for all nullable columns.
        Purpose: Confirm that IS NULL is true, not == 0, for those columns.
        """
        print("Setup: enable DB...")
        _enable_db(monkeypatch, tmp_path)

        import kiro.usage_db as udb

        fields = _make_fields(
            context_usage_percentage=None,
            credits_used=None,
            account_id=None,
            finish_reason=None,
            resolved_model=None,
            token_source=None,
        )
        udb._record_usage_sync(fields)

        with udb._lock:
            conn = udb._get_conn()
        row = conn.execute(
            """
            SELECT
                context_usage_percentage IS NULL AS cpu_null,
                credits_used IS NULL AS cu_null,
                account_id IS NULL AS aid_null,
                finish_reason IS NULL AS fr_null,
                resolved_model IS NULL AS rm_null,
                token_source IS NULL AS ts_null
            FROM usage_log LIMIT 1
            """
        ).fetchone()

        print(f"Assert: all nullable fields are SQL NULL, got {row}...")
        assert row[0] == 1, "context_usage_percentage should be SQL NULL"
        assert row[1] == 1, "credits_used should be SQL NULL"
        assert row[2] == 1, "account_id should be SQL NULL"
        assert row[3] == 1, "finish_reason should be SQL NULL"
        assert row[4] == 1, "resolved_model should be SQL NULL"
        assert row[5] == 1, "token_source should be SQL NULL"


# --------------------------------------------------------------------------------------------------
# TestEdgeCases
# --------------------------------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for boundary conditions and derived-value logic."""

    def test_error_message_truncated_to_500(self, tmp_path: Path, monkeypatch) -> None:
        """
        What it does: Writes a row with a 600-char error_message.
        Purpose: Verify that it is stored truncated to exactly 500 chars.
        """
        print("Setup: enable DB, craft 600-char message...")
        _enable_db(monkeypatch, tmp_path)

        import kiro.usage_db as udb

        long_msg = "x" * 600
        fields = _make_fields(error_message=long_msg)
        udb._record_usage_sync(fields)

        with udb._lock:
            conn = udb._get_conn()
        row = conn.execute("SELECT error_message FROM usage_log LIMIT 1").fetchone()
        print(f"Assert: stored length == 500, got {len(row[0])}...")
        assert len(row[0]) == 500

    def test_error_message_exactly_500_unchanged(self, tmp_path: Path, monkeypatch) -> None:
        """
        What it does: Writes a row with exactly 500-char error_message.
        Purpose: Boundary — exactly 500 must not be truncated.
        """
        _enable_db(monkeypatch, tmp_path)
        import kiro.usage_db as udb

        msg = "y" * 500
        udb._record_usage_sync(_make_fields(error_message=msg))

        with udb._lock:
            conn = udb._get_conn()
        row = conn.execute("SELECT error_message FROM usage_log LIMIT 1").fetchone()
        assert len(row[0]) == 500

    @pytest.mark.parametrize(
        "account_id,expected_label",
        [
            (r"D:\creds\prod.json", "prod"),           # absolute Windows path
            ("/home/user/.config/creds.json", "creds"),  # absolute POSIX path
            ("refresh_token_abc123def456", "refresh_token_abc123def456"),  # no separator
            (None, None),                               # None in → None out
        ],
    )
    def test_account_label_derivation(
        self,
        account_id: str | None,
        expected_label: str | None,
    ) -> None:
        """
        What it does: Tests _derive_account_label for four distinct input forms.
        Purpose: Ensure label derivation is correct for Windows paths, POSIX paths,
                 bare token IDs, and None.
        """
        from kiro.usage_db import _derive_account_label

        print(f"Input: {account_id!r} -> expected {expected_label!r}...")
        result = _derive_account_label(account_id)
        assert result == expected_label

    def test_total_tokens_auto_sum_when_zero(self, tmp_path: Path, monkeypatch) -> None:
        """
        What it does: Passes total_tokens=0 with non-zero prompt/completion.
        Purpose: Verify that total_tokens is stored as the sum.
        """
        _enable_db(monkeypatch, tmp_path)
        import kiro.usage_db as udb

        fields = _make_fields(prompt_tokens=30, completion_tokens=20, total_tokens=0)
        udb._record_usage_sync(fields)

        with udb._lock:
            conn = udb._get_conn()
        row = conn.execute("SELECT total_tokens FROM usage_log LIMIT 1").fetchone()
        print(f"Assert: total_tokens == 50, got {row[0]}...")
        assert row[0] == 50

    @pytest.mark.asyncio
    async def test_total_tokens_auto_sum_via_async(self, tmp_path: Path, monkeypatch) -> None:
        """
        What it does: Same auto-sum test but through the public async API.
        Purpose: Confirm that the async wrapper applies the same logic.
        """
        _enable_db(monkeypatch, tmp_path)
        import kiro.usage_db as udb

        await udb.record_usage(
            api_format="openai",
            endpoint="/v1/chat/completions",
            is_stream=False,
            model="test",
            status_code=200,
            duration_ms=1.0,
            prompt_tokens=40,
            completion_tokens=10,
            total_tokens=0,
        )

        with udb._lock:
            conn = udb._get_conn()
        row = conn.execute("SELECT total_tokens FROM usage_log LIMIT 1").fetchone()
        assert row[0] == 50

    def test_total_tokens_explicit_nonzero_unchanged(self, tmp_path: Path, monkeypatch) -> None:
        """
        What it does: Passes total_tokens=99 with non-zero prompt/completion.
        Purpose: Explicitly provided non-zero value must be stored as-is.
        """
        _enable_db(monkeypatch, tmp_path)
        import kiro.usage_db as udb

        fields = _make_fields(prompt_tokens=30, completion_tokens=20, total_tokens=99)
        udb._record_usage_sync(fields)

        with udb._lock:
            conn = udb._get_conn()
        row = conn.execute("SELECT total_tokens FROM usage_log LIMIT 1").fetchone()
        assert row[0] == 99


# --------------------------------------------------------------------------------------------------
# TestErrors
# --------------------------------------------------------------------------------------------------


class TestErrors:
    """Tests verifying that failures are silently swallowed and logged."""

    def test_db_dir_is_a_file_logs_and_does_not_raise(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """
        What it does: Points USAGE_DB_PATH at a path whose parent is an existing file.
        Purpose: mkdir will fail with OSError; must be swallowed and logged.
        """
        print("Setup: create a file where the DB directory should be...")
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("i am a file")
        db_path = blocker / "usage.db"  # parent is a file, not a dir

        monkeypatch.setattr("kiro.usage_db._is_enabled", lambda: True)
        monkeypatch.setattr("kiro.usage_db._db_path", lambda: db_path)

        import kiro.usage_db as udb

        print("Action: call _record_usage_sync — expect swallowed error...")
        # Must not raise; errors are only logged.
        udb._record_usage_sync(_make_fields())

    def test_corrupt_db_file_logs_and_does_not_raise(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """
        What it does: Writes garbage bytes to the DB path, then calls record_usage.
        Purpose: sqlite3.DatabaseError must be caught, not propagated.
        """
        print("Setup: write garbage to DB path...")
        db_path = tmp_path / "corrupt.db"
        db_path.write_bytes(b"\x00\x01\x02\x03garbage bytes that are not SQLite")

        monkeypatch.setattr("kiro.usage_db._is_enabled", lambda: True)
        monkeypatch.setattr("kiro.usage_db._db_path", lambda: db_path)

        import kiro.usage_db as udb

        print("Action: call _record_usage_sync on corrupt file...")
        udb._record_usage_sync(_make_fields())  # must not raise

    @pytest.mark.asyncio
    async def test_record_usage_swallows_db_error(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """
        What it does: Uses the async API with a corrupt DB file.
        Purpose: Errors must never propagate out of record_usage.
        """
        db_path = tmp_path / "bad.db"
        db_path.write_bytes(b"not sqlite")

        monkeypatch.setattr("kiro.usage_db._is_enabled", lambda: True)
        monkeypatch.setattr("kiro.usage_db._db_path", lambda: db_path)

        import kiro.usage_db as udb

        await udb.record_usage(
            api_format="openai",
            endpoint="/v1/chat/completions",
            is_stream=False,
            model="test",
            status_code=200,
            duration_ms=1.0,
        )


# --------------------------------------------------------------------------------------------------
# TestConcurrency
# --------------------------------------------------------------------------------------------------


class TestConcurrency:
    """Tests verifying that concurrent writes are safe and complete without data loss."""

    @pytest.mark.asyncio
    async def test_50_concurrent_writes_produce_50_rows(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """
        What it does: Fires 50 record_usage coroutines concurrently via asyncio.gather.
        Purpose: Verify that the threading.Lock protects the connection and all
                 rows land safely with no duplicates lost.
        """
        print("Setup: enable DB at tmp_path...")
        db_path = _enable_db(monkeypatch, tmp_path)

        import kiro.usage_db as udb

        print("Action: gather 50 record_usage calls...")
        await asyncio.gather(
            *[
                udb.record_usage(
                    api_format="openai",
                    endpoint="/v1/chat/completions",
                    is_stream=False,
                    model=f"model-{i}",
                    status_code=200,
                    duration_ms=float(i),
                )
                for i in range(50)
            ]
        )

        with udb._lock:
            conn = udb._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM usage_log").fetchone()[0]
        print(f"Assert: 50 rows present, got {count}...")
        assert count == 50


# --------------------------------------------------------------------------------------------------
# TestReadonlyCopy
# --------------------------------------------------------------------------------------------------


class TestReadonlyCopy:
    """Tests for sync_readonly_copy behaviour."""

    @pytest.mark.asyncio
    async def test_readonly_copy_is_readable_and_delete_mode(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """
        What it does: Writes a row, syncs the readonly copy, opens it fresh.
        Purpose: Verify the copy is readable and uses DELETE journal mode.
        """
        print("Setup: enable DB, write one row...")
        db_path = _enable_db(monkeypatch, tmp_path)

        import kiro.usage_db as udb

        udb._record_usage_sync(_make_fields(model="copy-test"))

        print("Action: sync_readonly_copy...")
        await udb.sync_readonly_copy()

        dst = tmp_path / "usage-readonly.db"
        print(f"Assert: readonly copy exists at {dst}...")
        assert dst.exists()

        print("Assert: copy is readable with a fresh connection...")
        with sqlite3.connect(str(dst)) as copy_conn:
            row = copy_conn.execute(
                "SELECT journal_mode FROM pragma_journal_mode"
            ).fetchone()
            print(f"Assert: journal_mode=delete, got {row}...")
            assert row[0].lower() == "delete"

            count = copy_conn.execute("SELECT COUNT(*) FROM usage_log").fetchone()[0]
            assert count == 1

    @pytest.mark.asyncio
    async def test_sync_readonly_copy_noop_when_no_main_db(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """
        What it does: Calls sync_readonly_copy without ever opening the main DB.
        Purpose: Must not raise even when _conn is None.
        """
        print("Setup: enabled but _conn is None (never opened)...")
        monkeypatch.setattr("kiro.usage_db._is_enabled", lambda: True)
        monkeypatch.setattr("kiro.usage_db._db_path", lambda: tmp_path / "usage.db")

        import kiro.usage_db as udb

        assert udb._conn is None
        await udb.sync_readonly_copy()  # must not raise
        assert not (tmp_path / "usage-readonly.db").exists()


# --------------------------------------------------------------------------------------------------
# TestClose
# --------------------------------------------------------------------------------------------------


class TestClose:
    """Tests verifying that close() is safe under all conditions."""

    def test_close_safe_when_never_opened(self) -> None:
        """
        What it does: Calls close() on a module that has never opened a connection.
        Purpose: Must not raise.
        """
        import kiro.usage_db as udb

        assert udb._conn is None
        udb.close()  # must not raise

    def test_close_safe_to_call_twice(self, tmp_path: Path, monkeypatch) -> None:
        """
        What it does: Opens the connection, then calls close() twice.
        Purpose: Second close() must be a no-op, not raise.
        """
        print("Setup: open connection...")
        _enable_db(monkeypatch, tmp_path)

        import kiro.usage_db as udb

        with udb._lock:
            udb._get_conn()

        assert udb._conn is not None
        udb.close()
        assert udb._conn is None
        udb.close()  # second call must not raise
        assert udb._conn is None
