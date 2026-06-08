"""PostgreSQL connection helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg2
from psycopg2.extensions import connection as PsycopgConnection
from psycopg2.extras import RealDictCursor

from utilities.config import AppConfig, load_config

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 5


def get_connection(config: AppConfig | None = None) -> PsycopgConnection:
    """Open a live PostgreSQL connection using runtime config."""
    config = config or load_config()
    db_config = config.database
    _validate_database_config(config)
    return psycopg2.connect(
        host=db_config.host,
        port=db_config.port,
        dbname=db_config.database,
        user=db_config.user,
        password=db_config.password,
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        application_name="base-rbc-llm-framework",
    )


@contextmanager
def connection_scope(config: AppConfig | None = None) -> Iterator[PsycopgConnection]:
    """Yield a PostgreSQL connection and close it when the block exits."""
    conn = get_connection(config)
    try:
        yield conn
    finally:
        conn.close()


def fetch_one(
    query: str,
    params: tuple[Any, ...] | dict[str, Any] | None = None,
    config: AppConfig | None = None,
) -> dict[str, Any] | None:
    """Run a SELECT query and return the first row as a dict."""
    with connection_scope(config) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None


def execute(
    query: str,
    params: tuple[Any, ...] | dict[str, Any] | None = None,
    config: AppConfig | None = None,
) -> int:
    """Run a write query and commit it."""
    with connection_scope(config) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rowcount = cursor.rowcount
        conn.commit()
    return rowcount


def check_connection(config: AppConfig | None = None) -> dict[str, Any]:
    """Run a live PostgreSQL health check and return database metadata."""
    config = config or load_config()
    with connection_scope(config) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT
                    current_user,
                    current_database(),
                    host(inet_server_addr()),
                    inet_server_port(),
                    current_setting('server_version')
            """)
            user, database, host, port, server_version = cursor.fetchone()

            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.schemata
                    WHERE schema_name = %s
                )
                """,
                (config.database.schema,),
            )
            schema_exists = bool(cursor.fetchone()[0])

    return {
        "user": user,
        "database": database,
        "host": host,
        "port": port,
        "server_version": server_version,
        "schema": config.database.schema,
        "schema_exists": schema_exists,
    }


def _validate_database_config(config: AppConfig) -> None:
    """Raise a clear error when required DB env values are missing."""
    missing = []
    if not config.database.host:
        missing.append("DB_HOST or POSTGRES_HOST")
    if not config.database.database:
        missing.append("DB_NAME or POSTGRES_DATABASE")
    if not config.database.user:
        missing.append("DB_USER or POSTGRES_USER")
    if missing:
        raise ValueError(f"Missing PostgreSQL config: {', '.join(missing)}")
