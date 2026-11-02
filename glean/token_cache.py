"""Local token cache for the Glean OAuth2 session.

Tokens are stored as plain JSON in a single file (bind-mounted out of the
container so they survive restarts). This is a local single-user dev cache —
never ship it or upload it to a shared deployment.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TokenCache:
    """Read/write the Glean OAuth token cache file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> Optional[dict]:
        try:
            with self.path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return None
            return data
        except (OSError, ValueError):
            return None

    def save(
        self,
        *,
        access_token: str,
        refresh_token: str,
        expires_at: float,
        client_id: str,
        token_endpoint: str = "",
        client_secret: str = "",
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
            "client_id": client_id,
            "token_endpoint": token_endpoint,
            "client_secret": client_secret,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        logger.info("Glean token cache updated (%s)", self.path)

    def clear(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass
        logger.info("Glean token cache cleared")