import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anyio import CapacityLimiter, to_thread

from app.core.config import settings
from app.pipeline.contracts import MTState

# One token per stage: inference must not land in the framework's default
# thread pool (40 tokens, shared with its sync dependencies), where 40
# inferences could run at once against a single GPU. Module level because the
# engine is a frozen dataclass with no __dict__, and there is one engine per
# stage per process anyway.
_LIMITER = CapacityLimiter(1)

# Every value MT_MODEL accepts. Anything else raises at construction: a typo
# like "opusmt" would otherwise fall through to the stub in silence, and the
# system would translate with placeholders in production without one line of
# output saying so.
_BACKENDS = ("stub", "opus-mt")

_PIP_INSTALL = "pip install ctranslate2 sentencepiece huggingface_hub"

# Third-party CTranslate2 conversions of Helsinki-NLP's OPUS-MT, apache-2.0
# like the upstream weights and carrying source.spm/target.spm, so tokenization
# needs sentencepiece alone - no transformers and no torch, not even offline.
#
# Pinned by commit because they are third-party repositories: an unpinned
# revision hands someone else the ability to change what this process
# translates with, between one deploy and the next, with no diff here.
#
# OPUS-MT is bilingual and directional, so there is one artifact per direction
# and no f-string can derive the repo name - the mapping has to be written out.
_OPUS_MT_REPOS = {
    ("es", "en"): (
        "michaelfeil/ct2fast-opus-mt-es-en",
        "437f5ffc6c8544943c685ea405650e0d17cf6098",
    ),
    ("en", "es"): (
        "michaelfeil/ct2fast-opus-mt-en-es",
        "76ec296588e2234f9b7dfad5254219a0f5ecb7af",
    ),
}


@dataclass
class MTResult:
    translated_text: str
    source_language: str
    target_language: str
    processing_time_ms: int


def _render(pairs) -> str:
    return ", ".join(f"{source}->{target}" for source, target in sorted(pairs))


@dataclass(frozen=True)
class _Translator:
    """One language pair's model plus the two tokenizers that frame it.

    Not shareable across pairs: es->en and en->es are different models with
    different vocabularies. Typed as Any because the concrete classes come from
    an import that deliberately does not happen at module level.
    """

    __slots__ = ("engine", "source_sp", "target_sp")
    engine: Any
    source_sp: Any
    target_sp: Any

    def translate(self, text: str) -> str:
        # The trailing </s> is mandatory and its absence does NOT raise.
        # Marian expects the source terminated; without it the decoder never
        # emits EOS and runs to max_decoding_length, returning text that starts
        # out correct and then degenerates into a loop of repeated fragments -
        # measured at ~2000 ms per segment against ~50 ms with the terminator.
        # A test asserting a content word appears in the output passes on the
        # broken version, which is why the check downstream bounds the length.
        tokens = self.source_sp.encode(text, out_type=str) + ["</s>"]
        results = self.engine.translate_batch([tokens])
        return self.target_sp.decode(results[0].hypotheses[0])


def _load_translator(pair: tuple[str, str], device: str) -> _Translator:
    """Fetch (or hit the local cache for) one pair's model and open it.

    The heavy imports live in here rather than at module level so MT_MODEL=stub
    pays nothing for a backend it does not use: the development machine has no
    GPU and should not import an inference runtime to run the test suite.
    """
    try:
        import ctranslate2
        import sentencepiece
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            f"MT_MODEL=opus-mt needs the translation runtime, which is not "
            f"installed: {_PIP_INSTALL}"
        ) from exc

    if pair not in _OPUS_MT_REPOS:
        raise ValueError(
            f"no OPUS-MT artifact is registered for {pair[0]}->{pair[1]}; "
            f"registered: {_render(_OPUS_MT_REPOS)}"
        )

    repo, revision = _OPUS_MT_REPOS[pair]
    path = Path(snapshot_download(repo, revision=revision))
    return _Translator(
        engine=ctranslate2.Translator(str(path), device=device),
        source_sp=sentencepiece.SentencePieceProcessor(str(path / "source.spm")),
        target_sp=sentencepiece.SentencePieceProcessor(str(path / "target.spm")),
    )


def _load_backend(model_name: str, device: str) -> dict | None:
    if model_name not in _BACKENDS:
        raise ValueError(
            f"unknown MT_MODEL {model_name!r}; valid values: "
            f"{', '.join(repr(name) for name in _BACKENDS)}"
        )
    if model_name == "stub":
        return None
    return {
        pair: _load_translator(pair, device)
        for pair in settings.supported_language_pairs
    }


@dataclass(frozen=True)
class MTService:
    """Immutable and shared process-wide. See ASRService for the full rationale.

    `_translators` is in __slots__ but deliberately NOT annotated, so it is a
    slot and not a dataclass field: it stays out of the generated __init__, and
    every existing MTService("stub", "cpu") call site keeps working unchanged.
    __post_init__ fills it with object.__setattr__, the one way past a frozen
    dataclass.
    """

    __slots__ = ("model_name", "device", "_translators")
    model_name: str
    device: str

    def __post_init__(self) -> None:
        # Every configured pair is loaded here, eagerly, rather than on first
        # use. Three reasons, in order of weight:
        #
        #   - it fails fast at startup when weights are missing, which is the
        #     contract the lifespan already relies on for the other engines;
        #   - it leaves the dict read-only at runtime, so there is no
        #     check-then-set to get wrong under concurrency;
        #   - it makes the invariant strong: every pair the API accepts has a
        #     model loaded, because session creation validates against this
        #     same configured list.
        #
        # A lazy version would be safe today only because the one-token limiter
        # serializes _translate(). Whoever raised that limiter to 2 would
        # reintroduce a double-load race, silently.
        object.__setattr__(
            self, "_translators", _load_backend(self.model_name, self.device)
        )

    async def translate(
        self,
        state: MTState,
        text: str,
        source_language: str,
        target_language: str,
    ) -> MTResult:
        start = time.monotonic()
        translated = await to_thread.run_sync(
            self._translate,
            state,
            text,
            source_language,
            target_language,
            limiter=_LIMITER,
        )
        processing_time_ms = int((time.monotonic() - start) * 1000)

        return MTResult(
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            processing_time_ms=processing_time_ms,
        )

    def _translate(
        self,
        state: MTState,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        # Synchronous on purpose: this is where a real model runs, and real
        # inference blocks. translate() dispatches it to a thread. Writing it
        # as `async def` is the trap: run_sync would hand back a coroutine it
        # never runs, so the inference would silently never happen.
        state.segments_seen += 1
        if self._translators is None:
            return f"[MT stub] {source_language}->{target_language}: {text}"

        translator = self._translators.get((source_language, target_language))
        if translator is None:
            # Reachable once ASR reports a detected language of its own: it
            # need not match the language the session declared, and the
            # combination can name a pair with no model loaded. Failing loudly
            # here is deliberate. The alternative - quietly translating from
            # the declared language instead of the detected one - is a choice
            # about what the transcript means, and it belongs with the ASR
            # replacement rather than with this one.
            raise ValueError(
                f"no MT model loaded for {source_language}->{target_language}; "
                f"MT_MODEL={self.model_name!r} was built for "
                f"{_render(self._translators)}"
            )
        return translator.translate(text)
