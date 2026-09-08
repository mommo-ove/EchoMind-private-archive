from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

import campus.store as store_module
from campus.store import CampusStore


def make_ticket(
    store,
    *,
    key="conv-1:billing",
    user_id="demo_user_01",
    category="billing",
    title="Synthetic campus-card duplicate charge",
    description="Synthetic demo report of two matching charges.",
):
    return store.create_ticket(
        idempotency_key=key,
        user_id=user_id,
        category=category,
        title=title,
        description=description,
    )


def create_legacy_ticket_database(database_path):
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE tickets (
                id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO tickets (
                id, idempotency_key, user_id, category, title, description,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ticket_legacy",
                "shared-key",
                "demo_user_01",
                "billing",
                "Synthetic legacy title",
                "Synthetic legacy description",
                "OPEN",
                "2026-01-01T00:00:00.000000+00:00",
                "2026-01-01T00:00:00.000000+00:00",
            ),
        )
        connection.commit()


def test_create_ticket_is_idempotent(tmp_path):
    store = CampusStore(tmp_path / "campus.db")

    first = make_ticket(store)
    second = make_ticket(store)

    assert first["id"] == second["id"]
    assert second == first


def test_concurrent_ticket_creation_is_idempotent(tmp_path):
    database_path = tmp_path / "campus.db"
    CampusStore(database_path)

    def create_once(_):
        return make_ticket(CampusStore(database_path))

    with ThreadPoolExecutor(max_workers=8) as executor:
        tickets = list(executor.map(create_once, range(16)))

    assert len({ticket["id"] for ticket in tickets}) == 1
    with closing(sqlite3.connect(database_path)) as connection:
        count = connection.execute(
            """
            SELECT COUNT(*) FROM tickets
            WHERE user_id = ? AND idempotency_key = ?
            """,
            ("demo_user_01", "conv-1:billing"),
        ).fetchone()[0]
    assert count == 1


def test_idempotency_key_is_scoped_by_user(tmp_path):
    store = CampusStore(tmp_path / "campus.db")

    first = make_ticket(store, key="shared-key", user_id="demo_user_01")
    second = make_ticket(store, key="shared-key", user_id="demo_user_02")

    assert first["id"] != second["id"]
    assert first["user_id"] == "demo_user_01"
    assert second["user_id"] == "demo_user_02"
    assert store.get_ticket(first["id"], user_id="demo_user_02") is None


def test_same_user_key_reuse_with_different_payload_is_rejected(tmp_path):
    store = CampusStore(tmp_path / "campus.db")
    original = make_ticket(store)

    with pytest.raises(ValueError, match="different ticket payload"):
        make_ticket(store, title="Different synthetic request")

    assert store.get_ticket(original["id"]) == original


def test_existing_global_idempotency_schema_migrates_to_user_scope(tmp_path):
    database_path = tmp_path / "campus.db"
    create_legacy_ticket_database(database_path)

    store = CampusStore(database_path)
    second_user = make_ticket(
        store,
        key="shared-key",
        user_id="demo_user_02",
    )

    assert second_user["id"] != "ticket_legacy"
    assert second_user["user_id"] == "demo_user_02"


def test_failed_legacy_migration_rolls_back_and_next_startup_recovers(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "campus.db"
    create_legacy_ticket_database(database_path)
    original_create = CampusStore._create_ticket_table

    def create_then_fail(connection):
        original_create(connection)
        raise RuntimeError("injected migration failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            CampusStore,
            "_create_ticket_table",
            staticmethod(create_then_fail),
        )
        with pytest.raises(RuntimeError, match="injected migration failure"):
            CampusStore(database_path)

    with closing(sqlite3.connect(database_path)) as connection:
        ticket = connection.execute(
            "SELECT id, user_id FROM tickets WHERE id = ?",
            ("ticket_legacy",),
        ).fetchone()
        legacy_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = ? AND name = ?",
            ("table", "tickets_legacy"),
        ).fetchone()
    assert ticket == ("ticket_legacy", "demo_user_01")
    assert legacy_table is None

    recovered = CampusStore(database_path)
    second_user = make_ticket(
        recovered,
        key="shared-key",
        user_id="demo_user_02",
    )
    assert recovered.get_ticket(
        "ticket_legacy",
        user_id="demo_user_01",
    ) is not None
    assert second_user["user_id"] == "demo_user_02"


def test_ticket_timestamps_are_timezone_aware_utc(tmp_path):
    ticket = make_ticket(CampusStore(tmp_path / "campus.db"))

    created_at = datetime.fromisoformat(ticket["created_at"])
    updated_at = datetime.fromisoformat(ticket["updated_at"])

    assert created_at.tzinfo is not None
    assert created_at.utcoffset() == timezone.utc.utcoffset(created_at)
    assert updated_at == created_at


def test_ticket_allows_open_processing_resolved_transitions(tmp_path):
    store = CampusStore(tmp_path / "campus.db")
    ticket = make_ticket(store)

    processing = store.update_ticket_status(ticket["id"], "PROCESSING")
    resolved = store.update_ticket_status(ticket["id"], "RESOLVED")

    assert processing["status"] == "PROCESSING"
    assert resolved["status"] == "RESOLVED"
    assert resolved["updated_at"] >= processing["updated_at"]


def test_ticket_rejects_resolved_to_open_transition(tmp_path):
    store = CampusStore(tmp_path / "campus.db")
    ticket = make_ticket(store)
    store.update_ticket_status(ticket["id"], "PROCESSING")
    store.update_ticket_status(ticket["id"], "RESOLVED")

    with pytest.raises(ValueError, match="RESOLVED.*OPEN"):
        store.update_ticket_status(ticket["id"], "OPEN")

    assert store.get_ticket(ticket["id"])["status"] == "RESOLVED"


def test_unknown_ticket_returns_none(tmp_path):
    store = CampusStore(tmp_path / "campus.db")

    assert store.get_ticket("ticket_missing") is None
    assert store.update_ticket_status("ticket_missing", "PROCESSING") is None


def test_ticket_lookup_is_scoped_to_requesting_user(tmp_path):
    store = CampusStore(tmp_path / "campus.db")
    ticket = make_ticket(store, user_id="demo_user_01")

    assert store.get_ticket(ticket["id"], user_id="demo_user_01") == ticket
    assert store.get_ticket(ticket["id"], user_id="demo_user_02") is None


def test_demo_transactions_are_queryable_by_user_and_date(tmp_path):
    store = CampusStore(tmp_path / "campus.db")
    demo_transactions = store.query_transactions("demo_user_01", days=30)
    transaction_date = demo_transactions[0]["occurred_at"][:10]

    matching = store.query_transactions(
        "demo_user_01",
        start_date=transaction_date,
        end_date=transaction_date,
    )

    assert matching
    assert all(item["user_id"] == "demo_user_01" for item in matching)
    assert all(item["occurred_at"].startswith(transaction_date) for item in matching)
    assert store.query_transactions("demo_user_missing", days=30) == []


def test_persistent_demo_transactions_refresh_into_default_window(tmp_path, monkeypatch):
    current_time = [datetime(2026, 1, 1, 12, tzinfo=timezone.utc)]
    monkeypatch.setattr(
        CampusStore,
        "_utc_now",
        staticmethod(lambda: current_time[0]),
    )
    database_path = tmp_path / "campus.db"
    CampusStore(database_path)

    current_time[0] += timedelta(days=14)
    reopened = CampusStore(database_path)
    transactions = reopened.query_transactions("demo_user_01")
    cutoff = current_time[0] - timedelta(days=7)

    assert transactions
    assert all(
        datetime.fromisoformat(item["occurred_at"]) >= cutoff
        for item in transactions
    )


def test_demo_refresh_does_not_overwrite_non_demo_id_collision(tmp_path):
    database_path = tmp_path / "campus.db"
    CampusStore(database_path)
    custom_row = (
        "demo_txn_001",
        "synthetic_fixture_user_01",
        -999,
        "Synthetic Fixture Merchant",
        "2020-01-01T00:00:00.000000+00:00",
    )
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute(
            "DELETE FROM campus_card_transactions WHERE id = ?",
            ("demo_txn_001",),
        )
        connection.execute(
            """
            INSERT INTO campus_card_transactions (
                id, user_id, amount_cents, merchant, occurred_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            custom_row,
        )
        connection.commit()

    CampusStore(database_path)

    with closing(sqlite3.connect(database_path)) as connection:
        persisted = connection.execute(
            """
            SELECT id, user_id, amount_cents, merchant, occurred_at
            FROM campus_card_transactions WHERE id = ?
            """,
            ("demo_txn_001",),
        ).fetchone()
    assert persisted == custom_row


def test_network_status_returns_structured_record(tmp_path):
    store = CampusStore(tmp_path / "campus.db")

    status = store.get_network_status("campus")

    assert status == {
        "site": "campus",
        "status": "OPERATIONAL",
        "message": "Synthetic demo status: campus network services are operational.",
        "updated_at": status["updated_at"],
    }
    updated_at = datetime.fromisoformat(status["updated_at"])
    assert updated_at.tzinfo is not None
    assert store.get_network_status("unknown-site") is None


def test_store_closes_every_sqlite_connection(tmp_path, monkeypatch):
    opened = []
    closed = []
    real_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def close(self):
            closed.append(id(self))
            super().close()

    def tracking_connect(*args, **kwargs):
        kwargs["factory"] = TrackingConnection
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(store_module.sqlite3, "connect", tracking_connect)
    store = CampusStore(tmp_path / "campus.db")
    ticket = make_ticket(store)
    store.get_ticket(ticket["id"])
    store.query_transactions("demo_user_01")
    store.get_network_status("campus")
    store.update_ticket_status(ticket["id"], "PROCESSING")
    store.update_ticket_status(ticket["id"], "RESOLVED")
    with pytest.raises(ValueError):
        store.update_ticket_status(ticket["id"], "OPEN")

    assert len(opened) > 1
    assert sorted(id(connection) for connection in opened) == sorted(closed)


def test_connect_closes_connection_when_pragma_setup_fails(tmp_path, monkeypatch):
    opened = []
    closed = []
    real_connect = sqlite3.connect

    class FailingPragmaConnection(sqlite3.Connection):
        def execute(self, sql, parameters=(), /):
            if sql.startswith("PRAGMA busy_timeout"):
                raise sqlite3.OperationalError("injected PRAGMA failure")
            return super().execute(sql, parameters)

        def close(self):
            closed.append(id(self))
            super().close()

    def failing_connect(*args, **kwargs):
        kwargs["factory"] = FailingPragmaConnection
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(store_module.sqlite3, "connect", failing_connect)

    with pytest.raises(sqlite3.OperationalError, match="injected PRAGMA failure"):
        CampusStore(tmp_path / "campus.db")

    assert len(opened) == 1
    assert closed == [id(opened[0])]
