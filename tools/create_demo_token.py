"""Create a short-lived HS256 JWT for local EchoMind demonstrations only."""

import argparse
from datetime import datetime, timedelta, timezone
import os

import jwt


def create_demo_token(
    *,
    subject: str,
    secret_key: str,
    issuer: str,
    audience: str,
    ttl_seconds: int = 900,
) -> str:
    if (
        not isinstance(subject, str)
        or not subject
        or subject != subject.strip()
        or len(subject) > 128
        or not subject.isprintable()
    ):
        raise ValueError("subject must be a non-empty printable identifier")
    if len(secret_key.encode("utf-8")) < 32:
        raise ValueError("JWT_SECRET_KEY must be at least 32 bytes")
    if not isinstance(ttl_seconds, int) or not 1 <= ttl_seconds <= 86400:
        raise ValueError("ttl_seconds must be between 1 and 86400")
    if not issuer or not audience:
        raise ValueError("issuer and audience are required")

    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": subject,
            "iat": now,
            "exp": now + timedelta(seconds=ttl_seconds),
            "iss": issuer,
            "aud": audience,
        },
        secret_key,
        algorithm="HS256",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one short-lived local EchoMind demo token.",
    )
    parser.add_argument("subject", help="Authenticated demo user id")
    parser.add_argument("--ttl-seconds", type=int, default=900)
    args = parser.parse_args()

    token = create_demo_token(
        subject=args.subject,
        secret_key=os.getenv("JWT_SECRET_KEY", ""),
        issuer=os.getenv("JWT_ISSUER", "echomind-demo"),
        audience=os.getenv("JWT_AUDIENCE", "echomind-api"),
        ttl_seconds=args.ttl_seconds,
    )
    print(token)


if __name__ == "__main__":
    main()
