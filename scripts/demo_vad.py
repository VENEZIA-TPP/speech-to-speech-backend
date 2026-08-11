"""Probar la segmentación por voz con tu propia grabación.

Es la única etapa del pipeline que hoy es real, así que es la única que tiene
sentido probar con voz propia: ASR, MT y TTS son stubs y devuelven texto fijo.

    # 1. Grabá con QuickTime Player (Archivo > Nueva grabación de audio) o con
    #    Notas de Voz, y exportá el archivo. Cualquier formato sirve.
    # 2. Pasáselo:

    .venv/bin/python scripts/demo_vad.py mi_voz.m4a

Convierte solo a 16 kHz mono, segmenta, dibuja dónde encontró voz, y guarda
cada frase recortada para escucharla.

    --min-silencio 200     probar otro umbral sin tocar la configuración
    --barrido              comparar 200 / 250 / 300 / 400 / 500 ms de una
    --reproducir           reproducir cada recorte al terminar
    --servidor             además, correr la sesión completa contra el servidor

El barrido es el experimento que importa: el umbral está elegido por
razonamiento sobre el dominio y verificado contra voz sintetizada, no calibrado
contra una persona. Quien hace pausas largas al pensar necesita un número
distinto de quien encadena las frases.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.pipeline.contracts import VADState  # noqa: E402
from app.pipeline.vad import FRAME_SAMPLES, build_segmenter, pcm_from_wav  # noqa: E402

SR = settings.AUDIO_SAMPLE_RATE
SALIDA = Path("demo_salida")


def a_wav_16k(origen: Path) -> Path:
    """Convertir cualquier formato que lea CoreAudio a 16 kHz mono pcm_s16le.

    Se usa afconvert, que viene con macOS y remuestrea bien. Hacerlo a mano en
    numpy sería tirar el filtro anti-aliasing y meterle al VAD un audio con
    artefactos que no estaban en la grabación.
    """
    if origen.suffix.lower() == ".wav":
        with wave.open(str(origen), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2):
                return origen

    if not shutil.which("afconvert"):
        raise SystemExit(
            f"{origen.name} no es 16 kHz mono pcm_s16le y afconvert no está "
            "disponible para convertirlo."
        )
    destino = Path(tempfile.gettempdir()) / f"{origen.stem}_16k.wav"
    subprocess.run(
        [
            "afconvert",
            "-f",
            "WAVE",
            "-d",
            f"LEI16@{SR}",
            "-c",
            "1",
            str(origen),
            str(destino),
        ],
        check=True,
        capture_output=True,
    )
    print(f"  convertido a 16 kHz mono → {destino.name}")
    return destino


def segmentar(datos: bytes, min_silencio_ms: int | None = None):
    """Devuelve (segmentos cerrados, duración total en segundos)."""
    seg = build_segmenter()
    if min_silencio_ms is not None:
        frames = round(min_silencio_ms * SR / 1000 / FRAME_SAMPLES)
        object.__setattr__(seg, "min_silence_frames", frames)

    muestras = pcm_from_wav(datos)
    estado = VADState()
    segmentos = []
    for k in range(0, len(muestras) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        frame = muestras[k : k + FRAME_SAMPLES]
        cerrado = seg._feed_frame(estado, frame)
        if cerrado:
            segmentos.append(((k + FRAME_SAMPLES) / SR, cerrado))
    ultimo = seg.flush(estado)
    if ultimo:
        segmentos.append((len(muestras) / SR, ultimo))
    return segmentos, len(muestras) / SR


def mapa(segmentos, duracion, ancho=60):
    """Una línea con la extensión de cada segmento dentro de la grabación.

    Ojo: es la extensión del *segmento*, no la de la voz. Todo segmento se lleva
    el pre-roll de antes del ataque y la cola de silencio que el VAD escuchó
    antes de decidir que la frase había terminado, así que dos frases seguidas
    se ven casi pegadas aunque hayas hecho una pausa clara entre ellas. Lo que
    importa mirar es cuántos números distintos hay, no dónde empieza cada uno.
    """
    celdas = ["·"] * ancho
    for i, (fin, s) in enumerate(segmentos):
        ini = max(0.0, fin - s.duration_ms / 1000)
        a = max(0, int(ini / duracion * ancho))
        b = min(ancho, max(a + 1, int(fin / duracion * ancho)))
        for k in range(a, b):
            celdas[k] = str(i % 10)
    return "".join(celdas)


def main() -> None:
    ap = argparse.ArgumentParser(description="Probar el VAD con tu propia voz.")
    ap.add_argument("audio", type=Path)
    ap.add_argument("--min-silencio", type=int, default=None, metavar="MS")
    ap.add_argument("--barrido", action="store_true")
    ap.add_argument("--reproducir", action="store_true")
    ap.add_argument("--servidor", action="store_true")
    args = ap.parse_args()

    if not args.audio.exists():
        raise SystemExit(f"no existe: {args.audio}")

    datos = a_wav_16k(args.audio).read_bytes()

    if args.barrido:
        print("\n  cuánto cambia el corte según el umbral de silencio:\n")
        print(f"    {'umbral':>8}  {'frases':>6}   dónde quedaron")
        for ms in (200, 250, 300, 400, 500):
            segs, dur = segmentar(datos, ms)
            marca = "  ← configurado" if ms == settings.VAD_MIN_SILENCE_MS else ""
            print(f"    {ms:>5} ms  {len(segs):>6}   {mapa(segs, dur)}{marca}")
        print("\n  Si al subir el umbral se juntan frases que para vos son distintas,")
        print("  está alto. Si al bajarlo una frase se parte al medio, está bajo.")
        return

    segmentos, duracion = segmentar(datos, args.min_silencio)
    umbral = args.min_silencio or settings.VAD_MIN_SILENCE_MS

    SALIDA.mkdir(exist_ok=True)
    for viejo in SALIDA.glob("voz_*.wav"):
        viejo.unlink()

    print(f"\n  {args.audio.name}  ·  {duracion:.2f} s  ·  umbral {umbral} ms\n")
    print(f"    {mapa(segmentos, duracion)}")
    print(f"    0s{' ' * 54}{duracion:.1f}s")
    print("    (cada dígito es una frase; incluye pre-roll y cola de silencio)\n")

    if not segmentos:
        print("  No se detectó ninguna frase. Revisá que se te escuche en la")
        print("  grabación: el VAD ignora tonos, ruido y silencio, y sólo")
        print("  responde a voz.\n")
        return

    for i, (fin, s) in enumerate(segmentos):
        destino = SALIDA / f"voz_{i}.wav"
        destino.write_bytes(s.pcm)
        ini = max(0.0, fin - s.duration_ms / 1000)
        print(
            f"    frase {i}  {ini:5.2f}s → {fin:5.2f}s   {s.duration_ms:>5} ms   "
            f"cerrada por {s.reason}   {destino}"
        )

    print(f"\n  {len(segmentos)} frases · guardadas en {SALIDA}/")
    print("  Escuchalas: si alguna se corta a mitad de palabra, el umbral está")
    print("  bajo; si dos frases tuyas quedaron pegadas, está alto.")
    print("  Probá otros valores:  --barrido\n")

    if args.reproducir and shutil.which("afplay"):
        for i in range(len(segmentos)):
            print(f"  ♪ frase {i}")
            subprocess.run(["afplay", str(SALIDA / f"voz_{i}.wav")])

    if args.servidor:
        print("  Para la sesión completa contra el servidor hace falta que estén")
        print("  levantados Postgres y uvicorn:\n")
        print("    docker compose up -d db")
        print(
            '    export DATABASE_URL="postgresql+asyncpg://postgres:postgres'
            '@localhost:5433/s2st_db"'
        )
        print("    .venv/bin/alembic upgrade head")
        print("    .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000\n")
        print("  Y después, en otra terminal:\n")
        print("    .venv/bin/python scripts/smoke_test.py\n")


if __name__ == "__main__":
    main()
