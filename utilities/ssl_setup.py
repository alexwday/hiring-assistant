"""SSL setup helpers for local and RBC-managed environments."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

from utilities.config import AppConfig, load_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SSLSetupResult:
    """Result of applying SSL setup."""

    success: bool
    verify: bool
    rbc_security_enabled: bool
    error: str | None

    @property
    def verify_value(self) -> bool | str:
        """Return the verification value accepted by requests/httpx."""
        if not self.verify:
            return False
        return True


def setup_ssl(config: AppConfig | None = None) -> SSLSetupResult:
    """Apply SSL configuration and enable RBC certificates when available."""
    config = config or load_config()
    if not config.ssl.verify:
        logger.info("SSL verification disabled")
        return SSLSetupResult(
            success=True,
            verify=False,
            rbc_security_enabled=False,
            error=None,
        )

    try:
        rbc_security = importlib.import_module("rbc_security")
    except ImportError:
        logger.warning("SSL_VERIFY=true but rbc_security is not available")
    else:
        enable_certs = getattr(rbc_security, "enable_certs", None)
        if callable(enable_certs):
            enable_certs()
            logger.info("RBC SSL certificates enabled")
            return SSLSetupResult(
                success=True,
                verify=True,
                rbc_security_enabled=True,
                error=None,
            )
        logger.warning("SSL_VERIFY=true but rbc_security.enable_certs is not callable")

    logger.info("SSL verification enabled with system certificates")
    return SSLSetupResult(
        success=True,
        verify=True,
        rbc_security_enabled=False,
        error=None,
    )
