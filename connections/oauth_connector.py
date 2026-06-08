"""OAuth 2.0 client-credentials connector.

The connector is intentionally small and lazy: constructing ``OAuthClient``
does not perform network I/O. A token request is made only when ``get_token``
is called and the cached token is missing or near expiry.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

REFRESH_BUFFER_SECONDS = 300


@dataclass(frozen=True)
class OAuthRetryConfig:
    """OAuth retry and timeout settings."""

    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class OAuthConfig:
    """OAuth client-credentials settings."""

    token_endpoint: str
    client_id: str
    client_secret: str
    grant_type: str = "client_credentials"
    scope: str = ""
    retry: OAuthRetryConfig = field(default_factory=OAuthRetryConfig)

    @property
    def max_retries(self) -> int:
        """Return the configured token request retry limit."""
        return self.retry.max_retries

    @property
    def retry_delay_seconds(self) -> float:
        """Return the delay between failed token request attempts."""
        return self.retry.retry_delay_seconds

    @property
    def timeout_seconds(self) -> float:
        """Return the token request timeout."""
        return self.retry.timeout_seconds


def _should_retry_with_body_credentials(response: requests.Response) -> bool:
    """Return whether a token endpoint may need body credentials."""
    if response.status_code != 400:
        return False
    try:
        error_data = response.json()
    except ValueError:
        return False

    error_value = str(error_data.get("error", "")).lower()
    description = str(error_data.get("error_description", "")).lower()
    return (
        error_value in {"invalid_client", "unauthorized_client"}
        or "basic" in description
        or "client credential" in description
        or "client_secret" in description
        or "client_id" in description
    )


class OAuthClient:
    """Manage OAuth token lifecycle with on-demand refresh."""

    def __init__(self, config: OAuthConfig, verify: bool | str = True):
        """Store OAuth settings without fetching a token."""
        self.config = config
        self.verify = verify
        self._access_token = ""
        self._expires_at = 0.0
        self._refresh_lock = Lock()

    def get_token(self) -> str:
        """Return a valid access token, refreshing it when needed."""
        if self.is_expired():
            with self._refresh_lock:
                if self.is_expired():
                    self._fetch_token()
        return self._access_token

    def is_expired(self) -> bool:
        """Return True when the token is missing or close to expiry."""
        if not self._access_token:
            return True
        return self._expires_at - time.time() <= REFRESH_BUFFER_SECONDS

    def _fetch_token(self) -> None:
        """Fetch and cache a new access token."""
        if not self.config.token_endpoint:
            raise ValueError("OAUTH_ENDPOINT or OAUTH_TOKEN_ENDPOINT is required")
        if not self.config.client_id:
            raise ValueError("OAUTH_CLIENT_ID is required")
        if not self.config.client_secret:
            raise ValueError("OAUTH_CLIENT_SECRET is required")

        logger.info("Fetching OAuth token from %s", self.config.token_endpoint)

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                token_data = self._request_token_payload()
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    raise
                logger.warning(
                    "OAuth token request failed; retrying in %.1f seconds "
                    "(attempt %s/%s)",
                    self.config.retry_delay_seconds,
                    attempt,
                    self.config.max_retries,
                )
                time.sleep(self.config.retry_delay_seconds)
        else:
            raise RuntimeError("OAuth token request failed") from last_error

        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError("OAuth response missing access_token")

        expires_in = int(token_data.get("expires_in", 3600))
        self._access_token = str(access_token)
        self._expires_at = time.time() + expires_in
        logger.info("OAuth token obtained; expires in %s seconds", expires_in)

    def _request_token_payload(self) -> dict[str, Any]:
        """Request and validate one OAuth token response payload."""
        response = self._post_token_request(include_body_credentials=False)
        if _should_retry_with_body_credentials(response):
            logger.info("OAuth basic auth failed; retrying with body credentials")
            response = self._post_token_request(include_body_credentials=True)

        response.raise_for_status()
        token_data = response.json()
        if not isinstance(token_data, dict):
            raise ValueError("OAuth response must be a JSON object")
        return token_data

    def _post_token_request(
        self,
        include_body_credentials: bool,
    ) -> requests.Response:
        """Send one token endpoint request."""
        data: dict[str, Any] = {"grant_type": self.config.grant_type}
        if self.config.scope:
            data["scope"] = self.config.scope

        auth = None
        if include_body_credentials:
            data["client_id"] = self.config.client_id
            data["client_secret"] = self.config.client_secret
        else:
            auth = HTTPBasicAuth(self.config.client_id, self.config.client_secret)

        return requests.post(
            self.config.token_endpoint,
            data=data,
            auth=auth,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.config.timeout_seconds,
            verify=self.verify,
        )
