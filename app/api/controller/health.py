from fastapi import APIRouter, Depends

from app.dependencies import get_asr_service, get_mt_service, get_tts_service
from app.services.asr_service import ASRService
from app.services.mt_service import MTService
from app.services.tts_service import TTSService

router = APIRouter()


@router.get("/")
async def health_check(
    asr: ASRService = Depends(get_asr_service),
    mt: MTService = Depends(get_mt_service),
    tts: TTSService = Depends(get_tts_service),
):
    # The names come off the engines the lifespan actually built, not off
    # settings and never off a constant: this endpoint answers "what is this
    # process running", and a hardcoded answer here kept claiming three stubs
    # after a real backend was already serving. No fallback for an unbuilt
    # engine either - if the lifespan did not run, this process should not be
    # answering health checks at all.
    #
    # This endpoint is unauthenticated, so it reports backend names only -
    # never weight paths, revisions or dependency versions.
    return {
        "status": "ok",
        "services": {
            "asr": asr.model_name,
            "mt": mt.model_name,
            "tts": tts.model_name,
        },
    }
