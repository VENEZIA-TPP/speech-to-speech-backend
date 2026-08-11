"""Dos propiedades de "la inferencia corre fuera del event loop", cada una con
su propio test porque una no implica la otra.

1. El event loop no puede congelarse mientras corre una inferencia. Las N
   conexiones WebSocket de un proceso comparten un unico event loop. Una
   inferencia que corre inline sobre ese loop no congela a la sesion que la
   disparo: congela a todas. Marcar el metodo como `async def` no cambia nada
   - sin un `await` real adentro, el control nunca vuelve al loop. El canario
   de responsividad inyecta un backend que bloquea a proposito: contra un
   pipeline que ejecuta la inferencia inline el GET tarda lo mismo que la
   inferencia.

2. Sacar la inferencia del loop no alcanza si dos sesiones pueden correrla en
   paralelo contra la misma GPU: el limiter de un token por etapa es lo que
   serializa ese acceso. El canario de responsividad no ve esta diferencia -
   sin limiter el loop queda libre igual - asi que la serializacion tiene su
   propio test, parametrizado por etapa para que borrar el limiter de una
   sola etapa siga rompiendo la suite.
"""

import asyncio
import threading
import time

import pytest

from app.pipeline.contracts import ASRState, MTState, TTSState
from app.services.asr_service import ASRService
from app.services.mt_service import MTService
from app.services.tts_service import TTSService

# Largo comparado con el presupuesto de los 100 ms: si el loop quedara
# congelado, la diferencia es de un orden de magnitud y no una carrera.
BLOCKING_SECONDS = 2.0
RESPONSIVE_BUDGET_SECONDS = 0.1


def _blocking_engines(entered: threading.Event):
    """Un backend bloqueante por etapa, con la firma que tiene hoy el hook."""

    class BlockingASRService(ASRService):
        def _transcribe(self, state, audio_bytes, language):
            entered.set()
            time.sleep(BLOCKING_SECONDS)
            return ("blocked", language or "en", 1.0)

    class BlockingMTService(MTService):
        def _translate(self, state, text, source_language, target_language):
            entered.set()
            time.sleep(BLOCKING_SECONDS)
            return "blocked"

    class BlockingTTSService(TTSService):
        def _synthesize(self, state, text, language):
            entered.set()
            time.sleep(BLOCKING_SECONDS)
            return TTSService._synthesize(self, state, text, language)

    return {
        "asr": BlockingASRService,
        "mt": BlockingMTService,
        "tts": BlockingTTSService,
    }


@pytest.mark.parametrize("stage", ["asr", "mt", "tts"])
def test_event_loop_stays_responsive_during_inference(ws_client, stage):
    """Con una inferencia de 2 s en vuelo, GET /health/ responde en <100 ms.

    Esperar el threading.Event antes de cronometrar es lo que hace el test
    determinista: sin eso el GET podria salir antes de que el servidor haya
    empezado a procesar el chunk, y pasar por accidente.
    """
    from app.dependencies import get_asr_service, get_mt_service, get_tts_service
    from app.main import app

    getters = {"asr": get_asr_service, "mt": get_mt_service, "tts": get_tts_service}
    entered = threading.Event()
    engine_cls = _blocking_engines(entered)[stage]
    app.dependency_overrides[getters[stage]] = lambda: engine_cls("stub", "cpu")

    created = ws_client.post(
        "/sessions/", json={"source_language": "en", "target_language": "es"}
    ).json()

    with ws_client.websocket_connect(
        f"/pipeline/ws/{created['id']}", subprotocols=[created["ws_token"]]
    ) as ws:
        ws.send_bytes(b"fake_audio_chunk")
        assert entered.wait(timeout=5.0), "la inferencia nunca arranco"

        start = time.monotonic()
        response = ws_client.get("/health/")
        elapsed = time.monotonic() - start

        # Drenar la respuesta del pipeline (JSON + binario: las tres etapas
        # stub siempre producen audio sintetizado) para que
        # process_audio_chunk() termine su trabajo - incluida la escritura a
        # la DB - mientras la conexion y la app siguen vivas, antes de que el
        # cierre del `with` dispare el teardown.
        ws.receive_json()
        ws.receive_bytes()
        # Cerrar desde el cliente y esperar el close frame del propio
        # servidor, en vez de dejar que lo haga el teardown implicito del
        # `with`: ese teardown manda su propio close() seguido de un cancel()
        # casi inmediato, sin esperar a que el handler termine su `finally`
        # (complete_session() + cierre de la AsyncSession de la request). Sin
        # esto la cancelacion puede pisar esa sesion a mitad de cierre, mismo
        # patron que test_ws_abrupt_disconnect_does_not_leave_session_active.
        ws.close()
        assert ws.receive()["type"] == "websocket.close"

    assert response.status_code == 200
    assert elapsed < RESPONSIVE_BUDGET_SECONDS, (
        f"GET /health/ tardo {elapsed:.3f}s con una inferencia de "
        f"{BLOCKING_SECONDS}s en vuelo en la etapa {stage}: el event loop "
        f"quedo congelado para todas las conexiones del proceso"
    )


SERIALIZATION_SLEEP_SECONDS = 0.3


def _slow_engines(sleep_seconds: float):
    """Un backend lento por etapa, mismo patron que _blocking_engines pero sin
    threading.Event y con un sleep corto: 3 etapas x 2 corridas x
    BLOCKING_SECONDS sumaria 12s a la suite solo para medir serializacion.
    """

    class SlowASRService(ASRService):
        def _transcribe(self, state, audio_bytes, language):
            time.sleep(sleep_seconds)
            return ("slow", language or "en", 1.0)

    class SlowMTService(MTService):
        def _translate(self, state, text, source_language, target_language):
            time.sleep(sleep_seconds)
            return "slow"

    class SlowTTSService(TTSService):
        def _synthesize(self, state, text, language):
            time.sleep(sleep_seconds)
            return TTSService._synthesize(self, state, text, language)

    return {
        "asr": SlowASRService,
        "mt": SlowMTService,
        "tts": SlowTTSService,
    }


def _run_twice(stage, engine):
    """Dos llamadas concurrentes al metodo publico de `stage`, cada etapa con
    la firma que le corresponde hoy."""

    if stage == "asr":
        calls = (
            engine.transcribe(ASRState(), b"aaa"),
            engine.transcribe(ASRState(), b"bbb"),
        )
    elif stage == "mt":
        calls = (
            engine.translate(MTState(), "a", "en", "es"),
            engine.translate(MTState(), "b", "en", "es"),
        )
    else:
        calls = (
            engine.synthesize(TTSState(), "a", "es"),
            engine.synthesize(TTSState(), "b", "es"),
        )
    return asyncio.gather(*calls)


@pytest.mark.parametrize("stage", ["asr", "mt", "tts"])
async def test_inference_stage_is_serialized(stage):
    """Dos inferencias de la misma etapa no pueden solaparse.

    Parametrizado por etapa: el limiter vive uno por modulo, asi que borrarlo
    en mt_service.py o tts_service.py no toca el caso de asr y una version no
    parametrizada de este test seria ciega a esas dos etapas.
    """
    engine_cls = _slow_engines(SERIALIZATION_SLEEP_SECONDS)[stage]
    engine = engine_cls("stub", "cpu")

    start = time.monotonic()
    await _run_twice(stage, engine)
    elapsed = time.monotonic() - start

    assert elapsed >= 2 * SERIALIZATION_SLEEP_SECONDS, (
        f"dos inferencias de la etapa {stage} tardaron {elapsed:.3f}s en vez "
        f"de al menos {2 * SERIALIZATION_SLEEP_SECONDS}s: salieron en "
        f"paralelo, el limiter de un token no se esta aplicando en "
        f"{stage}_service.py"
    )
