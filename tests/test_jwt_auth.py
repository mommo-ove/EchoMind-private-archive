import asyncio
import base64
import hashlib
import hmac
import json
import time

import httpx
import pytest
from fastapi import FastAPI, Request

from api.auth import JWTAuthMiddleware, JWTAuthSettings
from tools.create_demo_token import create_demo_token


SECRET = "demo-secret-that-is-at-least-32-bytes-long"


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _token(
    *,
    subject: str = "student_001",
    secret: str = SECRET,
    expires_in: int = 300,
    issuer: str = "echomind-demo",
    audience: str = "echomind-api",
) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + expires_in,
        "iss": issuer,
        "aud": audience,
    }
    encoded_header = _b64url(
        json.dumps(header, separators=(",", ":")).encode("utf-8")
    )
    encoded_payload = _b64url(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )
    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    signature = hmac.new(
        secret.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    return f"{encoded_header}.{encoded_payload}.{_b64url(signature)}"


def _app(settings: JWTAuthSettings) -> FastAPI:
    app = FastAPI()
    app.add_middleware(JWTAuthMiddleware, settings=settings)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/private")
    async def private(request: Request):
        return {
            "principal_id": getattr(
                request.state,
                "principal_id",
                None,
            )
        }

    return app


def _request(app: FastAPI, path: str, headers=None) -> httpx.Response:
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.get(path, headers=headers)

    return asyncio.run(send())


def _settings(**overrides) -> JWTAuthSettings:
    values = {
        "mode": "jwt",
        "secret_key": SECRET,
        "issuer": "echomind-demo",
        "audience": "echomind-api",
    }
    values.update(overrides)
    return JWTAuthSettings(**values)


def test_verified_bearer_subject_becomes_server_request_principal():
    response = _request(
        _app(_settings()),
        "/private",
        headers={
            "Authorization": f"Bearer {_token()}",
            "X-EchoMind-User": "attacker-controlled",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"principal_id": "student_001"}


def test_jwt_mode_allows_public_health_but_rejects_missing_private_token():
    app = _app(_settings())

    assert _request(app, "/health").status_code == 200
    private = _request(app, "/private")
    assert private.status_code == 401
    assert private.json() == {"detail": "Bearer token required"}
    assert private.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "token",
    [
        _token(secret="different-secret-that-is-also-long-enough"),
        _token(expires_in=-1),
        _token(subject=" "),
    ],
)
def test_invalid_expired_or_blank_subject_token_is_rejected(token):
    response = _request(
        _app(_settings()),
        "/private",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid bearer token"}


def test_jwt_settings_reject_weak_secret():
    with pytest.raises(ValueError, match="at least 32"):
        _settings(secret_key="short")


@pytest.mark.parametrize("field", ["issuer", "audience"])
def test_jwt_settings_require_issuer_and_audience(field):
    with pytest.raises(ValueError, match=field.upper()):
        _settings(**{field: None})


def test_disabled_mode_preserves_anonymous_development_access():
    response = _request(
        _app(JWTAuthSettings(mode="disabled")),
        "/private",
    )

    assert response.status_code == 200
    assert response.json() == {"principal_id": None}


def test_demo_token_is_accepted_by_the_same_authentication_boundary():
    token = create_demo_token(
        subject="demo_user_01",
        secret_key=SECRET,
        issuer="echomind-demo",
        audience="echomind-api",
        ttl_seconds=300,
    )

    response = _request(
        _app(_settings()),
        "/private",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json() == {"principal_id": "demo_user_01"}


@pytest.mark.parametrize(
    ("subject", "ttl_seconds"),
    [(" ", 300), ("demo_user_01", 0)],
)
def test_demo_token_rejects_invalid_subject_or_ttl(subject, ttl_seconds):
    with pytest.raises(ValueError):
        create_demo_token(
            subject=subject,
            secret_key=SECRET,
            issuer="echomind-demo",
            audience="echomind-api",
            ttl_seconds=ttl_seconds,
        )


def test_fastapi_application_installs_jwt_authentication_boundary():
    from api.main import app

    assert any(
        middleware.cls is JWTAuthMiddleware
        for middleware in app.user_middleware
    )
