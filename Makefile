COMPOSE = docker compose

.PHONY: build up logs stop test

build:
	$(COMPOSE) build

up: build
	$(COMPOSE) up -d
	$(COMPOSE) logs -f bot

logs:
	$(COMPOSE) logs -f

stop:
	$(COMPOSE) down

test: build
	$(COMPOSE) run --rm --no-deps bot python -m pytest -q
