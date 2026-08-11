# Speech fixtures

## `es_sistema.wav`

1.82 s of Spanish — *"El sistema funciona correctamente"* — 16 kHz mono
pcm_s16le, 58 KB. Leading and trailing non-speech trimmed with the VAD itself,
so the file starts and ends on speech to the frame.

Generated on macOS with:

```
say -v Mónica -o out.aiff "El sistema funciona correctamente"
afconvert -f WAVE -d LEI16@16000 -c 1 out.aiff out.wav
```

then trimmed. Committed rather than generated at test time so the suite is
deterministic and runs anywhere, not only on macOS.

**Why a real recording is required at all:** Silero does not fire on synthetic
PCM. Measured against the vendored weights, silence, a 440 Hz sine, white noise
and a formant-shaped buzz with syllable-rate amplitude modulation all score
below 0.03, where the speech threshold is 0.5. A test that "sends audio" made of
`b"fake_audio_chunk"` or a tone produces zero segments, so any assertion built on
one would pass for the wrong reason. This file scores 1.000 peak, 0.896 mean.

**Why only one file:** every other case composes from it in `tests/test_vad.py`
by concatenating it with exact silence — pauses of any length, continuous speech
past the ceiling, silence-only. Composing beats recording variants: the gap is
exact to the sample, so a test for a 300 ms pause tests exactly 300 ms.

**What this file is not.** It is text-to-speech, not a human speaker, so it is
evidence for segmentation *behaviour* and for VAD timing, and it is not evidence
for transcription or translation quality. WER and BLEU need the recorded human
corpus; this fixture cannot stand in for it.
