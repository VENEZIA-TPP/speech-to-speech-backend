"""Engine immutability and per-session state isolation.

Two of the barriers that keep one session's state from reaching another: the
engine cannot hold state at all, and two sessions running against the same
engine must not see each other's.
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
    using slots=True, which raises an unreadable TypeError for new names.
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


async def test_two_sessions_do_not_share_state():
    """Two interleaved sessions must each count their own chunks: 3 and 3, not 6.

    Runs against the real stub engine rather than a double written for the
    test, because the stub genuinely writes its per-chunk memory into `state`.
    """
    from app.pipeline.contracts import ASRState

    asr = ASRService("stub", "cpu")
    session_a = ASRState()
    session_b = ASRState()

    for _ in range(3):
        await asr.transcribe(session_a, b"aaa")
        await asr.transcribe(session_b, b"bbb")

    assert session_a.chunks_seen == 3
    assert session_b.chunks_seen == 3


async def test_engine_holds_no_state_of_its_own():
    """After serving two sessions the engine itself must carry nothing.

    No __dict__ means there is not even a place to stash an attribute: this is
    what __slots__ buys on top of frozen (verified 2026-08-10).
    """
    from app.pipeline.contracts import ASRState

    asr = ASRService("stub", "cpu")
    await asr.transcribe(ASRState(), b"aaa")
    await asr.transcribe(ASRState(), b"bbb")

    assert not hasattr(asr, "__dict__")
    assert type(asr).__slots__ == ("model_name", "device")
