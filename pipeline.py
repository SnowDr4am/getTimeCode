"""Обработка одного эфира: звук -> расшифровка -> тайм-коды от LLM -> ответ в чат.

Ни видео, ни результаты не хранятся: всё удаляется сразу после ответа."""

import logging
from typing import Protocol

import asr
import llm
import timecodes
from jobs import Job

MESSAGE_LIMIT = 4096  # предел длины сообщения Telegram
ERROR_LIMIT = 300     # подпись ограничена 1024 символами, тело ошибки шлюза бывает длиннее

log = logging.getLogger(__name__)


class Sender(Protocol):
    async def text(self, chat_id: int, text: str, reply_to: int) -> None: ...

    async def file(self, chat_id: int, name: str, data: bytes, caption: str,
                   reply_to: int) -> None: ...


def split_message(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """Режем по строкам, чтобы тайм-код не разорвался между сообщениями."""
    parts, current = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            parts.append(current)
            current = ""
        current += line
    if current.strip():
        parts.append(current)
    return [p.strip() for p in parts if p.strip()]


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
    # Промпт в начале файла: при сбое LLM файл целиком скармливается любой нейросети вручную.
    document = f"{timecodes.SYSTEM_PROMPT}\n\n{prompt}\n".encode()
    name = f"{job.title}.txt"

    try:
        answer = await llm.complete(timecodes.SYSTEM_PROMPT, prompt)
    except llm.LLMError as exc:
        log.error("LLM для %s: %s", job.title, exc)
        await sender.file(
            job.chat_id, name, document,
            f"Нейросеть не ответила ({str(exc)[:ERROR_LIMIT]}). В файле промпт и расшифровка — "
            "их можно отправить в любую нейросеть вручную.",
            job.reply_to,
        )
        return

    marks = [b["start"] for b in blocks]
    for part in split_message(timecodes.normalize(answer, marks)):
        await sender.text(job.chat_id, part, job.reply_to)
    await sender.file(job.chat_id, name, document, "Расшифровка эфира", job.reply_to)
