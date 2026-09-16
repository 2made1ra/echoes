# Запуск

## Что нужно

- Python 3.14 и [uv](https://docs.astral.sh/uv/)
- Docker — для Redis и Qdrant
- Ключ OpenAI-совместимого провайдера, у которого есть chat-модель и модель эмбеддингов (`text-embedding-3-small`)

## 1. Персона

Персоны в репозиторий не входят — их создаёт пользователь. Для первого запуска подойдёт шаблон:

```bash
cp -r public_docs/persona-example persona/keeper
```

Как написать свою — [creating-a-persona.md](creating-a-persona.md).

## 2. Конфиг

```bash
cp .env.example .env
```

В `.env` вписать `LLM_BASE_URL`, `LLM_API_KEY` и `CHAT_MODEL`. Остальное работает со значениями по умолчанию.

## 3. Хранилища

```bash
docker compose up -d        # или make up
```

Redis слушает на порту `6379`, Qdrant — на `6333`. Без Qdrant сервис тоже запустится, но без долговременной памяти: `/health` покажет `"longterm": false`.

## 4. Сервис

```bash
uv sync
uv run uvicorn main:app --reload   # или make dev
```

- Чат: http://localhost:8000 — персона выбирается в шапке
- Состояние: http://localhost:8000/health
- API: http://localhost:8000/docs

## 5. Замер дрейфа

```bash
uv run python -m eval.drift --persona keeper --mock   # без сети: проверить замер
uv run python -m eval.drift --persona keeper          # настоящий прогон
uv run python -m eval.drift --persona keeper --runs 5 --reinject-every 4
uv run python -m unittest discover eval               # тесты замера
```

Нужен `eval.toml` в директории персоны. Отчёты складываются в `eval/results/`.

Прогон стоит 2 × `EVAL_RUNS` × длина сценария запросов к модели. По умолчанию это 2 × 3 × 24 = 144 запроса.

То же через `make` (`make help` — полный список):

| Команда | Что делает | Запросов при сценарии из 24 реплик |
|---|---|---|
| `make eval PERSONA=keeper` | обычный замер; `RUNS=`, `N=`, `MODEL=` переопределяют настройки | 144 |
| `make eval-quick PERSONA=keeper` | один диалог на условие: быстро проверить, что всё живо | 48 |
| `make eval-full PERSONA=keeper` | 5 диалогов на условие: результат устойчивее | 240 |
| `make eval-sweep PERSONA=keeper SWEEP="3 6 12"` | отдельный замер на каждое N — для подбора N | 144 на каждое N |
| `make eval-all` | замер всех персон, у которых есть `eval.toml` | 144 на персону |
| `make eval-mock PERSONA=keeper` | мок-модель вместо LLM | 0 |
| `make eval-mock-drift PERSONA=keeper` | мок с сильным дрейфом | 0 |
| `make eval-mock-errors PERSONA=keeper` | мок со сбоями провайдера | 0 |
| `make eval-mock-all` | мок для всех персон | 0 |
| `make test` | тесты замера | 0 |

## Остановка

```bash
docker compose down          # или make down; данные остаются в volumes
```

## Если что-то не так

| Симптом | Причина |
|---|---|
| `не задан LLM_API_KEY` | Нет `.env` или в нём пустой ключ |
| `в persona нет ни одной персоны` | Не создана ни одна персона — см. шаг 1 |
| Ошибка про `persona/<name>` при старте | У персоны не хватает файла или секции — см. [creating-a-persona.md](creating-a-persona.md) |
| `DEFAULT_PERSONA=... не найдена` | В `.env` указано имя, которого нет среди директорий `persona/` |
| `503 Redis недоступен` | Не подняты контейнеры: `docker compose up -d` |
| `502 LLM-провайдер` | Ошибка провайдера — текст в ответе и в логе сервиса |
