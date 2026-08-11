# Vendored model weights

## `silero_vad_16k_op15.onnx`

Silero VAD, 16 kHz-only build, ONNX opset 15. 1.29 MB, MIT (see `LICENSE`).

Taken from the `silero-vad` 6.2.1 wheel on PyPI
(`silero_vad/data/silero_vad_16k_op15.onnx`), upstream
<https://github.com/snakers4/silero-vad>.

**Vendored rather than installed** because the PyPI package hard-requires
`torch>=1.12` and `torchaudio` as *base* dependencies — ONNX is only an extra —
which would add roughly a gigabyte of wheels to serve a 1.3 MB model. Vendoring
also means the tests and the server start offline, with no download step and no
network in CI.

The 16 kHz-only build was chosen over the general 2.33 MB `silero_vad.onnx`
after verifying both produce **bit-identical** speech probabilities on the same
input; the pipeline is 16 kHz mono throughout (`AUDIO_SAMPLE_RATE`), so the
8 kHz half of the larger file is dead weight.

To update: pull a newer wheel, extract the same path, and re-run
`benchmarks/vad_endpoint.py` — the committed result JSON is what makes a
regression in per-frame cost visible.
