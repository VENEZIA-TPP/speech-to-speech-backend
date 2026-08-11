"""El event loop no puede congelarse mientras corre una inferencia.

Las N conexiones WebSocket de un proceso comparten un unico event loop. Una
inferencia que corre inline sobre ese loop no congela a la sesion que la
disparo: congela a todas. Marcar el metodo como `async def` no cambia nada -
sin un `await` real adentro, el control nunca vuelve al loop.

Estos tests inyectan un backend que bloquea a proposito: contra un pipeline
que ejecuta la inferencia inline el GET tarda lo mismo que la inferencia.
"""

import asyncio
import threading
import time

import pytest

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


async def test_inference_stage_is_serialized():
    """Dos inferencias de la misma etapa no pueden solaparse.

    El limiter de un token es lo que serializa el acceso a la GPU. Sin el, el
    despacho usaria el pool de threads por defecto - 40 tokens, compartidos
    con las dependencias sincronas del framework - y las dos inferencias
    saldrian en paralelo. El canario de arriba no ve esa diferencia: sin
    limiter el event loop queda libre igual.
    """
    from app.pipeline.contracts import ASRState

    sleep_seconds = 0.3

    class SlowASRService(ASRService):
        def _transcribe(self, state, audio_bytes, language):
            time.sleep(sleep_seconds)
            return ("slow", language or "en", 1.0)

    engine = SlowASRService("stub", "cpu")

    start = time.monotonic()
    await asyncio.gather(
        engine.transcribe(ASRState(), b"aaa"),
        engine.transcribe(ASRState(), b"bbb"),
    )
    elapsed = time.monotonic() - start

    assert elapsed >= 2 * sleep_seconds, (
        f"dos inferencias de la misma etapa tardaron {elapsed:.3f}s en vez de "
        f"al menos {2 * sleep_seconds}s: salieron en paralelo, el limiter de "
        f"un token no se esta aplicando"
    )
