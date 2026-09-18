from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pgvector.psycopg import register_vector_async  # type: ignore[import-untyped]
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


async def _configure_connection(connection: AsyncConnection[dict[str, object]]) -> None:
    # PostgreSQL needs the pgvector type adapter before Python vectors can be
    # bound safely as SQL parameters (instead of being interpolated into SQL).
    await register_vector_async(connection)


class Database:
    def __init__(self, url: str) -> None:
        self.pool = AsyncConnectionPool(
            conninfo=url,
            min_size=1,
            max_size=5,
            open=False,
            kwargs={"row_factory": dict_row},
            configure=_configure_connection,
        )

    async def open(self) -> None:
        await self.pool.open(wait=True)

    async def close(self) -> None:
        await self.pool.close()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncConnection[dict[str, object]]]:
        async with self.pool.connection() as connection:
            yield connection

    async def is_healthy(self) -> bool:
        try:
            async with self.connection() as connection:
                await connection.execute("SELECT 1")
        except Exception:  # A health check reports failure without leaking credentials.
            return False
        return True
