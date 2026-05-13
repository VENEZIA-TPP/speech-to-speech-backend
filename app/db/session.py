from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

engine = create_async_engine(settings.database_url, echo=False, future=True)
async_session = sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_session():
    async with async_session() as session:
        yield session
