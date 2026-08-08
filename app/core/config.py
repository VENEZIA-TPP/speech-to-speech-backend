from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    DATABASE_USER: str = "postgres"
    DATABASE_PASSWORD: str = "postgres"
    DATABASE_DB: str = "s2st_db"
    DATABASE_HOST: str = "db"
    DATABASE_PORT: int = 5432
    DATABASE_URL: str | None = None

    # Server
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    PUBLIC_URL: str = "http://localhost:8000"

    # ASR 
    ASR_MODEL: str = "stub"
    ASR_DEVICE: str = "cpu"

    # MT
    MT_MODEL: str = "stub"
    MT_DEVICE: str = "cpu"

    # TTS
    TTS_MODEL: str = "stub"
    TTS_DEVICE: str = "cpu"

    # Audio processing
    AUDIO_SAMPLE_RATE: int = 16000           
    AUDIO_CHUNK_DURATION_MS: int = 3000

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+asyncpg://{self.DATABASE_USER}:"
            f"{self.DATABASE_PASSWORD}@{self.DATABASE_HOST}:"
            f"{self.DATABASE_PORT}/{self.DATABASE_DB}"
        )

    @property
    def database_url_sync(self) -> str:
        """Synchronous URL for Alembic (psycopg2)."""
        if self.DATABASE_URL:
            return self.DATABASE_URL.replace("+asyncpg", "")
        return (
            f"postgresql://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}"
            f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_DB}"
        )


settings = Settings()
