"""Engine immutability and per-session state isolation.

The four barriers against state leaking between sessions live in
docs/adr/0003-workers-persistentes-y-estado-por-sesion.md; these are barriers
#1 and #4.
"""

from dataclasses import FrozenInstanceError, dataclass

import pytest

from app.services.asr_service import ASRService
from app.services.mt_service import MTService
from app.services.tts_service import TTSService


@pytest.mark.parametrize(
    "engine",
    [ASRService("stub", "cpu"), MTService("stub", "cpu"), TTSService("stub", "cpu")],
    ids=lambda e: type(e).__name__,
)
@pytest.mark.parametrize("attribute", ["model_name", "buffer"])
def test_engine_is_frozen(engine, attribute):
    """A declared field AND a brand-new name must raise the SAME exception type.

    That stability is the whole reason __slots__ is written by hand instead of
    using slots=True, which raises an unreadable TypeError for new names
    (reproduced in docs/investigacion/fase-3-arquitectura.md section 0).
    `engine.buffer = ...` is the exact accident this PR exists to prevent.
    """
    with pytest.raises(FrozenInstanceError):
        setattr(engine, attribute, "leaked")


def test_engine_subclass_is_frozen():
    """A real backend must be a frozen dataclass itself.

    Measured 2026-08-10: a *plain* subclass gets a __dict__ back and
    `self.buffer = ...` silently succeeds, because the generated __setattr__
    only raises when type(self) is the declaring class or the name is a
    declared field. This test fixes the shape a real backend has to use.
    """

    @dataclass(frozen=True)
    class FakeParakeetASRService(ASRService):
        __slots__ = ()

    engine = FakeParakeetASRService("parakeet-tdt-0.6b-v3", "cuda")
    assert engine.model_name == "parakeet-tdt-0.6b-v3"
    with pytest.raises(FrozenInstanceError):
        engine.buffer = "leaked"
