"""SQLite persistence for synthetic campus-service data."""

from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import sqlite3
from typing import Any, Iterator
from uuid import uuid4


_DEMO_USER_PREFIX = "demo_user_"
_TICKET_STATUSES = {"OPEN", "PROCESSING", "RESOLVED"}
_TICKET_TRANSITIONS = {
    "OPEN": {"PROCESSING"},
    "PROCESSING": {"RESOLVED"},
    "RESOLVED": set(),
}


class CampusStore:
    """Store tickets and clearly synthetic campus demo records in SQLite."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 30000")
        except BaseException:
            connection.close()
            raise
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _serialize_time(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds")

    @staticmethod
    def _record(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    @staticmethod
    def _create_ticket_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (user_id, idempotency_key)
            )
            """
        )

    def _migrate_ticket_idempotency_scope(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        existing = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
            ("table", "tickets"),
        ).fetchone()
        if existing is None:
            self._create_ticket_table(connection)
            return

        normalized_sql = "".join(existing["sql"].upper().split())
        if "UNIQUE(USER_ID,IDEMPOTENCY_KEY)" in normalized_sql:
            return

        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("ALTER TABLE tickets RENAME TO tickets_legacy")
            self._create_ticket_table(connection)
            connection.execute(
                """
                INSERT INTO tickets (
                    id, idempotency_key, user_id, category, title, description,
                    status, created_at, updated_at
                )
                SELECT
                    id, idempotency_key, user_id, category, title, description,
                    status, created_at, updated_at
                FROM tickets_legacy
                """
            )
            connection.execute("DROP TABLE tickets_legacy")
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()

    def _initialize(self) -> None:
        with self._connection() as connection:
            self._migrate_ticket_idempotency_scope(connection)
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS campus_card_transactions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    amount_cents INTEGER NOT NULL,
                    merchant TEXT NOT NULL,
                    occurred_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS network_status (
                    site TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            now = self._utc_now()
            demo_transactions = (
                (
                    "demo_txn_001",
                    "demo_user_01",
                    -1280,
                    "Synthetic Demo Campus Cafe",
                    self._serialize_time(now - timedelta(minutes=30)),
                ),
                (
                    "demo_txn_002",
                    "demo_user_01",
                    -1280,
                    "Synthetic Demo Campus Cafe",
                    self._serialize_time(now - timedelta(days=1, minutes=30)),
                ),
                (
                    "demo_txn_003",
                    "demo_user_02",
                    -500,
                    "Synthetic Demo Campus Store",
                    self._serialize_time(now - timedelta(hours=2)),
                ),
            )
            refreshable_demo_transactions = tuple(
                transaction + (len(_DEMO_USER_PREFIX), _DEMO_USER_PREFIX)
                for transaction in demo_transactions
            )
            connection.executemany(
                """
                INSERT INTO campus_card_transactions (
                    id, user_id, amount_cents, merchant, occurred_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    occurred_at = excluded.occurred_at
                WHERE substr(
                    campus_card_transactions.user_id, 1, ?
                ) = ?
                """,
                refreshable_demo_transactions,
            )
            connection.execute(
                """
                INSERT INTO network_status (site, status, message, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(site) DO NOTHING
                """,
                (
                    "campus",
                    "OPERATIONAL",
                    "Synthetic demo status: campus network services are operational.",
                    self._serialize_time(now),
                ),
            )

    def create_ticket(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        category: str,
        title: str,
        description: str,
    ) -> dict[str, Any]:
        """Create one ticket, or return the existing row for the same key."""
        ticket_id = f"ticket_{uuid4().hex}"
        now = self._serialize_time(self._utc_now())
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO tickets (
                    id, idempotency_key, user_id, category, title, description,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, idempotency_key) DO NOTHING
                """,
                (
                    ticket_id,
                    idempotency_key,
                    user_id,
                    category,
                    title,
                    description,
                    "OPEN",
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM tickets
                WHERE user_id = ? AND idempotency_key = ?
                """,
                (user_id, idempotency_key),
            ).fetchone()
            ticket = dict(row)
            request_payload = (category, title, description)
            stored_payload = (
                ticket["category"],
                ticket["title"],
                ticket["description"],
            )
            if request_payload != stored_payload:
                raise ValueError(
                    "Idempotency key was already used with a different ticket payload"
                )
        return ticket

    def get_ticket(
        self,
        ticket_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return a ticket, optionally restricted to its owning user."""
        with self._connection() as connection:
            if user_id is None:
                row = connection.execute(
                    "SELECT * FROM tickets WHERE id = ?",
                    (ticket_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM tickets WHERE id = ? AND user_id = ?",
                    (ticket_id, user_id),
                ).fetchone()
        return self._record(row)

    def update_ticket_status(
        self,
        ticket_id: str,
        new_status: str,
    ) -> dict[str, Any] | None:
        """Apply an allowed ticket state transition."""
        normalized_status = new_status.upper()
        if normalized_status not in _TICKET_STATUSES:
            raise ValueError(f"Unknown ticket status: {new_status}")

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM tickets WHERE id = ?",
                (ticket_id,),
            ).fetchone()
            if current is None:
                return None

            current_status = current["status"]
            if normalized_status == current_status:
                return dict(current)
            if normalized_status not in _TICKET_TRANSITIONS[current_status]:
                raise ValueError(
                    f"Invalid ticket status transition: "
                    f"{current_status} -> {normalized_status}"
                )

            connection.execute(
                """
                UPDATE tickets
                SET status = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    normalized_status,
                    self._serialize_time(self._utc_now()),
                    ticket_id,
                    current_status,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM tickets WHERE id = ?",
                (ticket_id,),
            ).fetchone()
        return self._record(updated)

    @staticmethod
    def _date_boundary(
        value: str | date | datetime,
        *,
        end: bool,
    ) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, date):
            parsed = datetime.combine(value, time.min, tzinfo=timezone.utc)
            if end:
                parsed += timedelta(days=1)
        else:
            if len(value) == 10:
                parsed = datetime.combine(
                    date.fromisoformat(value),
                    time.min,
                    tzinfo=timezone.utc,
                )
                if end:
                    parsed += timedelta(days=1)
            else:
                parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def query_transactions(
        self,
        user_id: str,
        days: int | None = 7,
        *,
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Query synthetic transactions for one user and optional date window."""
        clauses = ["user_id = ?"]
        parameters: list[Any] = [user_id]

        if days is not None:
            if days <= 0:
                raise ValueError("days must be greater than zero")
            clauses.append("occurred_at >= ?")
            parameters.append(
                self._serialize_time(self._utc_now() - timedelta(days=days))
            )
        if start_date is not None:
            clauses.append("occurred_at >= ?")
            parameters.append(
                self._serialize_time(self._date_boundary(start_date, end=False))
            )
        if end_date is not None:
            clauses.append("occurred_at < ?")
            parameters.append(
                self._serialize_time(self._date_boundary(end_date, end=True))
            )

        query = (
            "SELECT * FROM campus_card_transactions WHERE "
            + " AND ".join(clauses)
            + " ORDER BY occurred_at DESC, id ASC"
        )
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def get_network_status(self, site: str) -> dict[str, Any] | None:
        """Return the structured synthetic network status for a site."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM network_status WHERE site = ?",
                (site,),
            ).fetchone()
        return self._record(row)
