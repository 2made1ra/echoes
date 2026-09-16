.PHONY: help up down dev test \
	eval eval-quick eval-full eval-sweep eval-all \
	eval-mock eval-mock-drift eval-mock-errors eval-mock-all

# Параметры замера; любой можно задать: make eval PERSONA=<name> RUNS=5 N=4
# Пустой параметр — флаг не передаётся, значение берётся из .env и persona.toml.
PERSONA ?=
RUNS ?=
N ?=
MODEL ?=
# Пресеты: диалогов для быстрой и точной проверки, N для перебора.
QUICK_RUNS ?= 1
FULL_RUNS ?= 5
SWEEP ?= 3 6 12
# Мок-пресеты: сильный дрейф (базовый шанс слома и прирост за реплику) и доля
# сбоев провайдера на запрос (1% — примерно каждый пятый диалог из 24 реплик).
MOCK_DRIFT_BASE ?= 0.05
MOCK_DRIFT_STEP ?= 0.1
MOCK_ERROR_RATE ?= 0.01
# Персоны со сценарием замера — все, у кого есть eval.toml.
EVAL_PERSONAS := $(notdir $(patsubst %/eval.toml,%,$(wildcard persona/*/eval.toml)))

DRIFT := uv run python -m eval.drift
opt = $(if $(2),$(1) $(2))
# Флаги без персоны, N и числа диалогов — их цели подставляют сами.
base = $(call opt,--model,$(MODEL))
flags = $(base) $(call opt,--persona,$(PERSONA)) $(call opt,--reinject-every,$(N))

help: ## Список команд
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

up: ## Поднять контейнеры (Redis + Qdrant)
	docker compose up -d

down: ## Остановить контейнеры
	docker compose down

dev: ## Запустить API в дев-режиме (автоперезагрузка)
	uv run uvicorn main:app --reload

test: ## Тесты замера, без сети
	uv run python -m unittest discover eval

# --- замер на LLM: 2 × диалогов × длина сценария запросов к модели ----------

eval: ## Замер дрейфа (PERSONA= RUNS= N= MODEL=)
	$(DRIFT) $(flags) $(call opt,--runs,$(RUNS))

eval-quick: ## Дешёвая проверка: QUICK_RUNS диалог на условие
	$(DRIFT) $(flags) --runs $(QUICK_RUNS)

eval-full: ## Точнее: FULL_RUNS диалогов на условие
	$(DRIFT) $(flags) --runs $(FULL_RUNS)

eval-sweep: ## Подбор N: отдельный прогон на каждое N из SWEEP (SWEEP="4 8")
	@for n in $(SWEEP); do \
		$(DRIFT) $(base) $(call opt,--persona,$(PERSONA)) $(call opt,--runs,$(RUNS)) --reinject-every $$n || exit 1; \
	done

eval-all: ## Замер всех персон со сценарием (RUNS= N= MODEL=)
	@for p in $(EVAL_PERSONAS); do \
		$(DRIFT) $(base) --persona $$p $(call opt,--reinject-every,$(N)) $(call opt,--runs,$(RUNS)) || exit 1; \
	done

# --- замер на мок-модели: без сети, проверяет сам замер, а не персону ------

eval-mock: ## Мок с параметрами из .env
	$(DRIFT) --mock $(flags) $(call opt,--runs,$(RUNS))

eval-mock-drift: ## Мок с сильным дрейфом: без переинжекта роль ломается быстро
	EVAL_MOCK_BREAK_BASE=$(MOCK_DRIFT_BASE) EVAL_MOCK_BREAK_PER_TURN=$(MOCK_DRIFT_STEP) \
		$(DRIFT) --mock $(flags) $(call opt,--runs,$(RUNS))

eval-mock-errors: ## Мок со сбоями провайдера: упавшие диалоги видны под таблицей
	EVAL_MOCK_ERROR_RATE=$(MOCK_ERROR_RATE) $(DRIFT) --mock $(flags) $(call opt,--runs,$(RUNS))

eval-mock-all: ## Мок для всех персон со сценарием
	@for p in $(EVAL_PERSONAS); do \
		$(DRIFT) --mock --persona $$p $(call opt,--runs,$(RUNS)) || exit 1; \
	done
