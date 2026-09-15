.PHONY: up down dev eval

up: ## Поднять контейнеры (Redis + Qdrant)
	docker compose up -d

down: ## Остановить контейнеры
	docker compose down

dev: ## Запустить API в дев-режиме (автоперезагрузка)
	uv run uvicorn main:app --reload

eval: ## Замер удержания характера (make eval PERSONA=<name>)
	uv run python -m eval.drift $(if $(PERSONA),--persona $(PERSONA))
