FROM python:3.11-slim
WORKDIR /usr/src/app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--reload", "--host", "${HOST:-0.0.0.0}", "--port", "${PORT:-8000}", "--ws-max-size", "${MAX_AUDIO_FRAME_BYTES:-2000000}"]
