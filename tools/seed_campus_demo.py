"""Seed repeatable synthetic campus scenarios into the SQLite demo store."""

import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from campus.store import CampusStore


def _advance_to(store: CampusStore, ticket: dict[str, Any], target: str) -> dict[str, Any]:
    current = ticket
    if target in {"PROCESSING", "RESOLVED"} and current["status"] == "OPEN":
        current = store.update_ticket_status(current["id"], "PROCESSING")
    if target == "RESOLVED" and current["status"] == "PROCESSING":
        current = store.update_ticket_status(current["id"], "RESOLVED")
    return current


def seed_demo_scenarios(store: CampusStore) -> dict[str, dict[str, Any]]:
    """Create stable tickets that complement the store's card/network fixtures."""
    open_ticket = store.create_ticket(
        idempotency_key="demo-open-network-ticket",
        user_id="demo_user_01",
        category="technical",
        title="Synthetic dorm network authentication failure",
        description="Synthetic scenario: building 3 repeatedly receives HTTP 401.",
    )
    processing_ticket = store.create_ticket(
        idempotency_key="demo-processing-billing-ticket",
        user_id="demo_user_01",
        category="billing",
        title="Synthetic duplicate campus-card charge review",
        description="Synthetic scenario: two demo charges have the same amount.",
    )
    resolved_ticket = store.create_ticket(
        idempotency_key="demo-resolved-network-ticket",
        user_id="demo_user_02",
        category="technical",
        title="Synthetic resolved campus network ticket",
        description="Synthetic scenario used to demonstrate ticket ownership isolation.",
    )

    return {
        "open_ticket": open_ticket,
        "processing_ticket": _advance_to(store, processing_ticket, "PROCESSING"),
        "resolved_ticket": _advance_to(store, resolved_ticket, "RESOLVED"),
    }


def main() -> None:
    database_path = Path(
        os.getenv("CAMPUS_DB_PATH", "data/campus/campus.db")
    )
    results = seed_demo_scenarios(CampusStore(database_path))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
