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
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    # ASR
    ASR_MODEL: str = "stub"
    ASR_DEVICE: str = "cpu"

    # MT
    MT_MODEL: str = "stub"
    MT_DEVICE: str = "cpu"

    # TTS
    TTS_MODEL: str = "stub"
    TTS_DEVICE: str = "cpu"

    # Languages
    # Pairs the pipeline will accept, as "src-tgt" CSV. Delivered scope is
    # es<->en; the European expansion adds entries here, never in the code.
    SUPPORTED_LANGUAGE_PAIRS: str = "es-en,en-es"

    # Audio processing
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHUNK_DURATION_MS: int = 3000

    # Voice activity detection
    # A segment is what the pipeline runs on, and the VAD decides where one
    # ends. All of these except the silence threshold are a starting point
    # rather than calibrated truth: picked by reasoning about the domain and
    # verified against synthesized speech, never against a human speaker.
    VAD_THRESHOLD: float = 0.5
    # The latency<->quality dial, and the one most worth re-measuring. Below
    # ~200 ms an intra-phrase breath is indistinguishable from the end of a
    # sentence, and MT then translates a syntactically broken fragment. A
    # conversational agent can afford to cut at 64 ms because it can speculatively
    # reopen a turn it cut wrongly; there is no such mechanism here, so there is
    # nothing to make an aggressive cut tolerable.
    #
    # 250 comes from measuring a real recording rather than from reasoning:
    # sweeping 200/250/300/400/500 over one speaker showed 300 merging phrases
    # that the speaker said separately. It is the one VAD parameter calibrated
    # against a person - scripts/demo_vad.py --barrido is that experiment.
    VAD_MIN_SILENCE_MS: int = 250
    # Filters noise bursts. Measured on frames that actually cleared the
    # hysteresis floor, not on segment length, so trailing silence cannot pad a
    # burst over the bar.
    VAD_MIN_SPEECH_MS: int = 384
    # Pre-roll kept from before the trigger, so the attack of the first syllable
    # is not clipped. Costs no latency: it is audio already captured.
    VAD_SPEECH_PAD_MS: int = 300
    # Finite on purpose. Workers are shared across sessions, so an unbounded
    # segment is an unbounded blocking call on a shared resource: one speaker
    # who never pauses would stall every other session. The ceiling makes the
    # worst case bad but bounded.
    VAD_MAX_SPEECH_MS: int = 8000
    # How far back to look for the quietest frame when the ceiling fires, so the
    # forced cut lands in a gap between words instead of mid-syllable.
    VAD_CEILING_LOOKBACK_MS: int = 400

    # WebSocket
    # Cap on a single binary frame. 2 MB ~= 60 s of 16 kHz mono pcm_s16le, well
    # above AUDIO_CHUNK_DURATION_MS. Honest scope: by the time this is checked,
    # Starlette already buffered the frame. The real pre-buffer cap is uvicorn's
    # --ws-max-size; this check is what produces a clean 1009 close and a test.
    MAX_AUDIO_FRAME_BYTES: int = 2_000_000

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

    @property
    def supported_language_pairs(self) -> set[tuple[str, str]]:
        pairs = set()
        for entry in self.SUPPORTED_LANGUAGE_PAIRS.split(","):
            entry = entry.strip()
            if not entry:
                continue
            source, _, target = entry.partition("-")
            pairs.add((source, target))
        return pairs

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()
        ]


settings = Settings()
