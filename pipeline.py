"""Обработка одного эфира: звук -> расшифровка -> .txt с промптом в чат.

Ни видео, ни результаты не хранятся: всё удаляется сразу после ответа."""

import logging
from typing import Protocol

import asr
import timecodes
from jobs import Job

ERROR_LIMIT = 300  # текст исключения бывает длиннее разумного сообщения

log = logging.getLogger(__name__)


class Sender(Protocol):
    async def text(self, chat_id: int, text: str, reply_to: int) -> None: ...

    async def file(self, chat_id: int, name: str, data: bytes, caption: str,
                   reply_to: int) -> None: ...


async def transcribe(job: Job) -> list[dict]:
    pcm = job.source.with_suffix(".pcm")
    try:
        await asr.extract_audio(job.source, pcm)
        # Видео больше не нужно: на время распознавания держим на диске только звук.
        job.source.unlink(missing_ok=True)
        return await asr.run_transcribe(pcm)
    finally:
        job.source.unlink(missing_ok=True)
        pcm.unlink(missing_ok=True)


async def process(job: Job, sender: Sender) -> None:
    try:
        segments = await transcribe(job)
    except Exception as exc:
        log.exception("распознавание %s", job.title)
        await sender.text(job.chat_id, f"Не удалось распознать эфир: {str(exc)[:ERROR_LIMIT]}",
                          job.reply_to)
        return
    if not segments:
        await sender.text(job.chat_id, "Речь в эфире не распознана.", job.reply_to)
        return

    blocks = asr.split_blocks(segments)
    prompt = timecodes.request(timecodes.render_transcript(blocks), job.duration)
    # Промпт в начале файла: файл целиком отправляется в любую нейросеть.
    document = f"{timecodes.SYSTEM_PROMPT}\n\n{prompt}\n".encode()
    await sender.file(job.chat_id, f"{job.title}.txt", document,
                      "Промпт и расшифровка эфира — отправь файл в нейросеть.", job.reply_to)
