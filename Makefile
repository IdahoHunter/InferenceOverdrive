.PHONY: gpu-check build up down logs health

gpu-check:
	./scripts/check-gpu.sh

build:
	docker compose build llm-server

up:
	docker compose up -d llm-server open-webui

down:
	docker compose down

logs:
	docker compose logs -f llm-server

health:
	./scripts/health.sh
