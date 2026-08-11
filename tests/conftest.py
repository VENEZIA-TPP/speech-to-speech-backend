import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient

from app.main import app
from app.db.base import Base
from app.db.session import get_session

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# StaticPool
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestSessionLocal = async_sessionmaker(
    test_engine, expire_on_commit=False, class_=AsyncSession
)


async def override_get_session():
    async with TestSessionLocal() as session:
        yield session


@pytest.fixture(scope="session", autouse=True)
def cleanup_engine():
    yield
    import asyncio

    asyncio.run(test_engine.dispose())


@pytest_asyncio.fixture(scope="function")
async def db_session():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def client(db_session):
    app.dependency_overrides[get_session] = override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


async def _run_ddl(fn):
    async with test_engine.begin() as conn:
        await conn.run_sync(fn)


@pytest.fixture(scope="function")
def ws_client():
    """Sync TestClient for WebSocket tests (httpx AsyncClient can't speak WS)."""
    app.dependency_overrides[get_session] = override_get_session

    try:
        with TestClient(app) as tc:
            # El DDL corre en el portal del TestClient, no en un asyncio.run()
            # aparte: un asyncio.run() abre y cierra su propio loop, y create_all/
            # drop_all terminaban pisando la conexion que el handler de WS seguia
            # usando en el loop del TestClient - de ahi el "no active connection"
            # intermitente. Con tc.portal.call() todo el DDL y todo el trafico de
            # la sesion viven en el mismo loop.
            tc.portal.call(_run_ddl, Base.metadata.create_all)
            try:
                yield tc
            finally:
                tc.portal.call(_run_ddl, Base.metadata.drop_all)
                # Belt and braces: dejar el engine sin conexion cacheada para que el
                # proximo test arranque un checkout limpio. No es lo que arregla el
                # "no active connection" intermitente - eso lo arreglan el DDL sobre
                # el portal y el cierre explicito del WS del lado del cliente - pero
                # tampoco cuesta nada y acota el estado que un test le deja al
                # siguiente.
                tc.portal.call(test_engine.dispose)
    finally:
        app.dependency_overrides.clear()
