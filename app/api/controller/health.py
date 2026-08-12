from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def health_check():
    return {
        "status": "ok",
        "services": {
            "asr": "stub",  # TODO: return model_name once Whisper is loaded
            "mt": "stub",  # TODO: return model_name once NLLB/MarianMT is loaded
            "tts": "stub",  # TODO: return model_name once the real TTS is loaded
        },
    }
