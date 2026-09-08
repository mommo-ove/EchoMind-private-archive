import asyncio
import threading

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

import api.main as main
from campus.store import CampusStore


def make_request(*, principal_id=None, state_user_id=None, query_string=b""):
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/tickets/ticket-1",
            "query_string": query_string,
            "headers": [],
        }
    )
    if principal_id is not None:
        request.state.principal_id = principal_id
    if state_user_id is not None:
        request.state.user_id = state_user_id
    return request


def create_ticket(store, *, user_id="owner", description="private details"):
    return store.create_ticket(
        idempotency_key=f"request-{user_id}",
        user_id=user_id,
        category="network",
        title="Cannot connect",
        description=description,
    )


def run(coro):
    return asyncio.run(coro)


def assert_http_error(status_code, coro):
    with pytest.raises(HTTPException) as error:
        run(coro)
    assert error.value.status_code == status_code
    return error.value


def dump(model):
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def test_ticket_routes_are_exposed():
    routes = {
        (route.path, method)
        for route in main.app.routes
        for method in getattr(route, "methods", set())
    }

    assert ("/tickets/{ticket_id}", "GET") in routes
    assert ("/tickets/{ticket_id}/status", "PATCH") in routes


@pytest.mark.parametrize("state_attribute", ["principal_id", "user_id"])
def test_get_ticket_returns_safe_owner_view_for_trusted_principal(
    state_attribute,
    tmp_path,
    monkeypatch,
):
    store = CampusStore(tmp_path / "campus.db")
    ticket = create_ticket(store)
    monkeypatch.setattr(main, "_campus_store", store)
    request_kwargs = {
        "principal_id" if state_attribute == "principal_id" else "state_user_id":
        "owner"
    }

    result = run(main.get_ticket(ticket["id"], make_request(**request_kwargs)))
    payload = dump(result)

    assert payload == {
        "id": ticket["id"],
        "user_id": "owner",
        "category": "network",
        "title": "Cannot connect",
        "description": "private details",
        "status": "OPEN",
        "created_at": ticket["created_at"],
        "updated_at": ticket["updated_at"],
    }
    assert "idempotency_key" not in payload


def test_get_ticket_accepts_matching_supplied_user_id(tmp_path, monkeypatch):
    store = CampusStore(tmp_path / "campus.db")
    ticket = create_ticket(store)
    monkeypatch.setattr(main, "_campus_store", store)

    result = run(
        main.get_ticket(
            ticket["id"],
            make_request(principal_id="owner"),
            user_id="owner",
        )
    )

    assert dump(result)["id"] == ticket["id"]


def test_get_ticket_rejects_supplied_user_id_that_does_not_match_principal(
    tmp_path,
    monkeypatch,
):
    store = CampusStore(tmp_path / "campus.db")
    ticket = create_ticket(store)
    monkeypatch.setattr(main, "_campus_store", store)

    assert_http_error(
        403,
        main.get_ticket(
            ticket["id"],
            make_request(principal_id="owner"),
            user_id="other-user",
        ),
    )


def test_get_ticket_returns_404_for_another_users_ticket(tmp_path, monkeypatch):
    store = CampusStore(tmp_path / "campus.db")
    ticket = create_ticket(store)
    monkeypatch.setattr(main, "_campus_store", store)

    assert_http_error(
        404,
        main.get_ticket(ticket["id"], make_request(principal_id="other-user")),
    )


def test_get_ticket_rejects_self_reported_identity_without_trusted_state(
    tmp_path,
    monkeypatch,
):
    store = CampusStore(tmp_path / "campus.db")
    ticket = create_ticket(store)
    monkeypatch.setattr(main, "_campus_store", store)

    assert_http_error(
        401,
        main.get_ticket(
            ticket["id"],
            make_request(query_string=b"user_id=owner"),
            user_id="owner",
        ),
    )


def test_get_ticket_returns_404_for_unknown_ticket(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_campus_store", CampusStore(tmp_path / "campus.db"))

    assert_http_error(
        404,
        main.get_ticket("ticket-missing", make_request(principal_id="owner")),
    )


def test_get_ticket_runs_blocking_store_call_off_event_loop_thread(monkeypatch):
    event_loop_thread = threading.get_ident()

    class RecordingStore:
        def get_ticket(self, ticket_id, *, user_id):
            assert ticket_id == "ticket-1"
            assert user_id == "owner"
            assert threading.get_ident() != event_loop_thread
            return None

    monkeypatch.setattr(main, "_campus_store", RecordingStore())

    assert_http_error(
        404,
        main.get_ticket("ticket-1", make_request(principal_id="owner")),
    )


@pytest.mark.parametrize("token", [None, "wrong-token"])
def test_update_ticket_status_rejects_missing_or_wrong_admin_token(
    token,
    tmp_path,
    monkeypatch,
):
    store = CampusStore(tmp_path / "campus.db")
    ticket = create_ticket(store)
    monkeypatch.setattr(main, "_campus_store", store)
    monkeypatch.setenv("INTENT_FEEDBACK_ADMIN_TOKEN", "correct-token")

    assert_http_error(
        403,
        main.update_ticket_status(
            ticket["id"],
            main.TicketStatusInput(status="PROCESSING"),
            x_intent_admin_token=token,
        ),
    )
    assert store.get_ticket(ticket["id"])["status"] == "OPEN"


@pytest.mark.parametrize(
    ("starting_status", "new_status"),
    [
        ("OPEN", "OPEN"),
        ("OPEN", "PROCESSING"),
        ("PROCESSING", "PROCESSING"),
        ("PROCESSING", "RESOLVED"),
        ("RESOLVED", "RESOLVED"),
    ],
)
def test_update_ticket_status_applies_every_valid_transition(
    starting_status,
    new_status,
    tmp_path,
    monkeypatch,
):
    store = CampusStore(tmp_path / "campus.db")
    ticket = create_ticket(store)
    if starting_status in {"PROCESSING", "RESOLVED"}:
        store.update_ticket_status(ticket["id"], "PROCESSING")
    if starting_status == "RESOLVED":
        store.update_ticket_status(ticket["id"], "RESOLVED")
    monkeypatch.setattr(main, "_campus_store", store)
    monkeypatch.setenv("INTENT_FEEDBACK_ADMIN_TOKEN", "correct-token")

    result = run(
        main.update_ticket_status(
            ticket["id"],
            main.TicketStatusInput(status=new_status),
            x_intent_admin_token="correct-token",
        )
    )
    payload = dump(result)

    assert payload["status"] == new_status
    assert "description" not in payload
    assert "idempotency_key" not in payload


@pytest.mark.parametrize(
    ("starting_status", "new_status"),
    [
        ("OPEN", "RESOLVED"),
        ("PROCESSING", "OPEN"),
        ("RESOLVED", "OPEN"),
        ("RESOLVED", "PROCESSING"),
    ],
)
def test_update_ticket_status_returns_safe_409_for_invalid_transition(
    starting_status,
    new_status,
    tmp_path,
    monkeypatch,
):
    store = CampusStore(tmp_path / "campus.db")
    ticket = create_ticket(store, description="secret transition details")
    if starting_status in {"PROCESSING", "RESOLVED"}:
        store.update_ticket_status(ticket["id"], "PROCESSING")
    if starting_status == "RESOLVED":
        store.update_ticket_status(ticket["id"], "RESOLVED")
    monkeypatch.setattr(main, "_campus_store", store)
    monkeypatch.setenv("INTENT_FEEDBACK_ADMIN_TOKEN", "correct-token")

    error = assert_http_error(
        409,
        main.update_ticket_status(
            ticket["id"],
            main.TicketStatusInput(status=new_status),
            x_intent_admin_token="correct-token",
        ),
    )

    assert "secret" not in error.detail
    assert starting_status not in error.detail


def test_update_ticket_status_returns_404_for_unknown_ticket(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "_campus_store", CampusStore(tmp_path / "campus.db"))
    monkeypatch.setenv("INTENT_FEEDBACK_ADMIN_TOKEN", "correct-token")

    assert_http_error(
        404,
        main.update_ticket_status(
            "ticket-missing",
            main.TicketStatusInput(status="PROCESSING"),
            x_intent_admin_token="correct-token",
        ),
    )


@pytest.mark.parametrize(
    "body",
    [
        {"status": "NOT_A_STATUS"},
        {"status": "processing"},
        {"status": 1},
        {"status": "PROCESSING", "description": "must not be accepted"},
        {},
    ],
)
def test_update_ticket_status_rejects_invalid_shape_with_422(
    body,
    tmp_path,
    monkeypatch,
):
    store = CampusStore(tmp_path / "campus.db")
    ticket = create_ticket(store)
    monkeypatch.setattr(main, "_campus_store", store)
    monkeypatch.setenv("INTENT_FEEDBACK_ADMIN_TOKEN", "correct-token")

    client = TestClient(main.app)
    response = client.patch(
        f"/tickets/{ticket['id']}/status",
        json=body,
        headers={"X-Intent-Admin-Token": "correct-token"},
    )
    client.close()

    assert response.status_code == 422
    assert "NOT_A_STATUS" not in response.text
    assert "must not be accepted" not in response.text
    assert store.get_ticket(ticket["id"])["status"] == "OPEN"


@pytest.mark.parametrize("token", [None, "wrong-token"])
def test_admin_dependency_rejects_before_ticket_body_validation(
    token,
    tmp_path,
    monkeypatch,
):
    store = CampusStore(tmp_path / "campus.db")
    ticket = create_ticket(store)
    monkeypatch.setattr(main, "_campus_store", store)
    monkeypatch.setenv("INTENT_FEEDBACK_ADMIN_TOKEN", "correct-token")
    headers = {} if token is None else {"X-Intent-Admin-Token": token}

    client = TestClient(main.app)
    response = client.patch(
        f"/tickets/{ticket['id']}/status",
        json={"status": "NOT_A_STATUS", "private": "raw-input"},
        headers=headers,
    )
    client.close()

    assert response.status_code == 403
    assert "NOT_A_STATUS" not in response.text
    assert "raw-input" not in response.text
    assert store.get_ticket(ticket["id"])["status"] == "OPEN"


def test_ticket_and_feedback_routes_share_admin_dependency():
    dependency_calls = {}
    for route in main.app.routes:
        if route.path in {"/tickets/{ticket_id}/status", "/intent/feedback"}:
            dependency_calls[route.path] = {
                dependency.call for dependency in route.dependant.dependencies
            }

    assert dependency_calls == {
        "/tickets/{ticket_id}/status": {main.require_admin_token},
        "/intent/feedback": {main.require_admin_token},
    }
