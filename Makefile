COMPOSE = docker compose
GPU = $(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml

.PHONY: build run gpu logs stop shell clean

build:
	$(COMPOSE) build

# Запуск детачем: закрытый терминал не обрывает распознавание.
run: build
	$(COMPOSE) up -d --force-recreate
	$(COMPOSE) logs -f

gpu:
	$(GPU) build
	$(GPU) up -d --force-recreate
	$(GPU) logs -f

logs:
	$(COMPOSE) logs -f

stop:
	$(COMPOSE) down

shell:
	$(COMPOSE) run --rm --entrypoint bash timecodes

clean:
	rm -rf work/*.wav
