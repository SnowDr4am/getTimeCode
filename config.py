"""Настройки из окружения."""

import os
import re
from pathlib import Path


def parse_ids(raw: str) -> frozenset[int]:
    return frozenset(int(part) for part in re.split(r"[,;\s]+", raw) if part)


BOT_TOKEN = os.getenv("BOT_TOKEN", "")
# Остальным пользователям бот не отвечает вовсе.
ACCESS_IDS = parse_ids(os.getenv("ACCESS_IDS", ""))

# Локальный telegram-bot-api: облачный отдаёт боту файлы только до 20 МБ.
BOT_API_URL = os.getenv("BOT_API_URL", "http://bot-api:8081")
# Том telegram-bot-api смонтирован в оба контейнера по одному пути: getFile в local-режиме
# отдаёт абсолютный путь, а переименование в WORK_DIR работает только в пределах тома.
DATA_DIR = Path(os.getenv("DATA_DIR", "/var/lib/telegram-bot-api"))
WORK_DIR = DATA_DIR / "timecodes-work"
