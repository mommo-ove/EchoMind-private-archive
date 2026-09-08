"""Request authentication that turns a verified JWT subject into trusted state."""

from dataclasses import dataclass, field
import os
from typing import FrozenSet, Optional

import jwt
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


_DEFAULT_PUBLIC_PATHS = frozenset({
    "/health",
    "/metrics",
    "/docs",
    "/docs/oauth2-redirect",
    "/openapi.json",
    "/redoc",
})


@dataclass(frozen=True)
class JWTAuthSettings:
    """Validated configuration for optional local HS256 bearer authentication."""

    mode: str = "disabled"
    secret_key: str = ""
    issuer: Optional[str] = None
    audience: Optional[str] = None
    public_paths: FrozenSet[str] = field(
        default_factory=lambda: _DEFAULT_PUBLIC_PATHS
    )

    def __post_init__(self) -> None:
        normalized_mode = self.mode.strip().lower()
        object.__setattr__(self, "mode", normalized_mode)
        if normalized_mode not in {"disabled", "jwt"}:
            raise ValueError("AUTH_MODE must be disabled or jwt")
        if normalized_mode == "jwt":
            if len(self.secret_key.encode("utf-8")) < 32:
                raise ValueError("JWT_SECRET_KEY must be at least 32 bytes")
            for field_name, value in (
                ("JWT_ISSUER", self.issuer),
                ("JWT_AUDIENCE", self.audience),
            ):
                if (
                    not isinstance(value, str)
                    or not value
                    or value != value.strip()
                    or not value.isprintable()
                ):
                    raise ValueError(f"{field_name} is required in jwt mode")

    @classmethod
    def from_env(cls) -> "JWTAuthSettings":
        return cls(
            mode=os.getenv("AUTH_MODE", "disabled"),
            secret_key=os.getenv("JWT_SECRET_KEY", ""),
            issuer=os.getenv("JWT_ISSUER") or None,
            audience=os.getenv("JWT_AUDIENCE") or None,
        )


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Verify bearer JWTs before placing their subject in request.state."""

    def __init__(self, app, *, settings: JWTAuthSettings) -> None:
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next):
        if (
            self.settings.mode == "disabled"
            or request.method == "OPTIONS"
            or request.url.path in self.settings.public_paths
        ):
            return await call_next(request)

        authorization = request.headers.get("authorization", "")
        scheme, separator, token = authorization.partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not token
            or token != token.strip()
        ):
            return self._unauthorized("Bearer token required")

        try:
            claims = jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=["HS256"],
                issuer=self.settings.issuer,
                audience=self.settings.audience,
                options={
                    "require": ["sub", "iat", "exp"],
                    "verify_iss": self.settings.issuer is not None,
                    "verify_aud": self.settings.audience is not None,
                },
            )
            principal_id = claims.get("sub")
            if (
                not isinstance(principal_id, str)
                or not principal_id
                or principal_id != principal_id.strip()
                or len(principal_id) > 128
                or not principal_id.isprintable()
            ):
                raise jwt.InvalidTokenError("invalid subject")
        except jwt.InvalidTokenError:
            return self._unauthorized("Invalid bearer token")

        request.state.principal_id = principal_id
        return await call_next(request)

    @staticmethod
    def _unauthorized(detail: str) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"detail": detail},
            headers={"WWW-Authenticate": "Bearer"},
        )
