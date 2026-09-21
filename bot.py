"""Telegram-бот: принимает эфир, ставит в очередь, возвращает тайм-коды."""

import asyncio
import logging
import math
import re
import shutil
import uuid
from functools import partial
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, Message, ReplyParameters

import asr
import config
import pipeline
from jobs import Job, JobQueue

MAX_FILE_SIZE = 2000 * 2**20  # потолок local-режима telegram-bot-api
# getFile в local-режиме возвращается только когда сервер скачал файл целиком.
DOWNLOAD_TIMEOUT_SEC = 3600

log = logging.getLogger(__name__)
router = Router()
# Чужим бот не отвечает: сообщения без подходящего хендлера aiogram молча пропускает.
router.message.filter(F.from_user.id.in_(config.ACCESS_IDS))


def humanize(seconds: float) -> str:
    minutes = max(1, math.ceil(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes} мин" if hours else f"{minutes} мин"


def safe_title(file_name: str | None) -> str:
    stem = Path(file_name or "").stem
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", stem).strip(" ._")[:100] or "эфир"


class TelegramSender:
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    @staticmethod
    def _reply(message_id: int) -> ReplyParameters:
        return ReplyParameters(message_id=message_id, allow_sending_without_reply=True)

    async def text(self, chat_id: int, text: str, reply_to: int) -> None:
        await self._bot.send_message(chat_id, text, reply_parameters=self._reply(reply_to))

    async def file(self, chat_id: int, name: str, data: bytes, caption: str,
                   reply_to: int) -> None:
        await self._bot.send_document(chat_id, BufferedInputFile(data, name), caption=caption,
                                      reply_parameters=self._reply(reply_to))


@router.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer("Пришли запись эфира видео- или аудиофайлом (до 2 ГБ) — "
                         "верну тайм-коды для YouTube и расшифровку.")


@router.message(F.video | F.audio | F.document)
async def media(message: Message, bot: Bot, queue: JobQueue) -> None:
    file = message.video or message.audio or message.document
    mime = file.mime_type or ""
    if message.document and not mime.startswith(("video/", "audio/")):
        await message.reply("Нужен видео- или аудиофайл.")
        return
    if (file.file_size or 0) > MAX_FILE_SIZE:
        await message.reply("Файл больше 2 ГБ — Telegram не отдаст его боту.")
        return

    status = await message.reply("Принял, скачиваю…")
    path = None
    try:
        remote = await bot.get_file(file.file_id, request_timeout=DOWNLOAD_TIMEOUT_SEC)
        # Файл уходит из кэша telegram-bot-api в свою папку: повторная отправка того же
        # видео не должна достаться задаче, которая его уже удалила.
        path = config.WORK_DIR / f"{uuid.uuid4().hex}{Path(remote.file_path).suffix}"
        Path(remote.file_path).rename(path)
        duration = await asr.probe_duration(path)
    except Exception as exc:
        log.exception("скачивание")
        if path:
            path.unlink(missing_ok=True)
        await status.edit_text(f"Не удалось скачать файл: {str(exc)[:pipeline.ERROR_LIMIT]}")
        return
    if duration is None:
        path.unlink(missing_ok=True)
        await status.edit_text("В файле нет звуковой дорожки.")
        return

    job = Job(message.chat.id, message.message_id, path, duration,
              safe_title(getattr(file, "file_name", None)))
    ahead, eta = queue.put(job)
    if ahead:
        text = (f"Эфир в очереди, перед ним {ahead}. "
                f"Результат примерно через {humanize(eta)}.")
    else:
        text = f"Эфир взят в обработку, результат будет примерно через {humanize(eta)}."
    await status.edit_text(text)


@router.message()
async def hint(message: Message) -> None:
    await message.answer("Жду видео- или аудиофайл с эфиром.")


async def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not config.BOT_TOKEN or not config.KIE_API_KEY or not config.ACCESS_IDS:
        raise SystemExit("В .env нужны BOT_TOKEN, KIE_API_KEY и ACCESS_IDS")

    # Очередь живёт в памяти: после рестарта недоделанные файлы никому не нужны.
    shutil.rmtree(config.WORK_DIR, ignore_errors=True)
    config.WORK_DIR.mkdir(parents=True)

    session = AiohttpSession(api=TelegramAPIServer.from_base(config.BOT_API_URL, is_local=True))
    bot = Bot(config.BOT_TOKEN, session=session)
    queue = JobQueue(partial(pipeline.process, sender=TelegramSender(bot)), asr.SPEED)
    worker = asyncio.create_task(queue.run())

    dp = Dispatcher(queue=queue)
    dp.include_router(router)
    try:
        await dp.start_polling(bot)
    finally:
        worker.cancel()


if __name__ == "__main__":
    asyncio.run(main())
