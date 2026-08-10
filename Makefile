.PHONY: help install dev up down migrate push rollback history shell test lint

help:
	@echo "Comandos disponibles:"
	@echo "  make install    - Instalar dependencias"
	@echo "  make dev        - Iniciar servidor en modo desarrollo"
	@echo "  make up         - Levantar servicios con Docker Compose"
	@echo "  make down       - Detener servicios Docker"
	@echo "  make migrate    - Crear migración automática (usar: make migrate MSG='mensaje')"
	@echo "  make push       - Aplicar migraciones pendientes"
	@echo "  make rollback   - Revertir última migración"
	@echo "  make history    - Ver historial de migraciones"
	@echo "  make shell      - Abrir shell en contenedor web"
	@echo "  make test       - Ejecutar tests"
	@echo "  make lint       - Ejecutar linters"

install:
	pip install -r requirements.txt

dev:
	uvicorn app.main:app --reload --host $${HOST:-0.0.0.0} --port $${PORT:-8000} --ws-max-size $${MAX_AUDIO_FRAME_BYTES:-2000000}

up:
	docker-compose up -d

down:
	docker-compose down

migrate:
	@if [ -z "$(MSG)" ]; then echo "Error: Debes proporcionar un mensaje. Uso: make migrate MSG='tu mensaje'"; exit 1; fi
	alembic revision --autogenerate -m "$(MSG)"

push:
	alembic upgrade head

rollback:
	alembic downgrade -1

history:
	alembic history --verbose

shell:
	docker-compose exec web bash

test:
	PYTHONPATH=. pytest tests/ -v

lint:
	flake8 app/
	black app/ --check
