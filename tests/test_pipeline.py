"""Эфир целиком: ответ пользователю и уборка файлов. Распознавание и LLM подменены."""

import pytest

import asr
import llm
import pipeline
from jobs import Job


class _Sender:
    def __init__(self):
        self.texts, self.files = [], []

    async def text(self, chat_id, text, reply_to):
        self.texts.append(text)

    async def file(self, chat_id, name, data, caption, reply_to):
        self.files.append((name, data.decode(), caption))


SEGMENTS = [{"start": 0.0, "end": 5.0, "text": "Всем привет."},
            {"start": 65.0, "end": 70.0, "text": "Главная мысль."}]


@pytest.fixture
def job(tmp_path, monkeypatch):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")

    async def extract(src, dst):
        dst.write_bytes(b"\0\0")

    async def run_transcribe(pcm):
        assert pcm.exists()
        return SEGMENTS

    monkeypatch.setattr(asr, "extract_audio", extract)
    monkeypatch.setattr(asr, "run_transcribe", run_transcribe)
    return Job(chat_id=1, reply_to=2, source=source, duration=70, title="эфир")


def _leftovers(job):
    return list(job.source.parent.iterdir())


async def test_успех_тайм_коды_и_расшифровка(job, monkeypatch):
    prompts = []

    async def complete(system, user):
        prompts.append((system, user))
        return "00:00 Привет\n01:05 Главное"

    monkeypatch.setattr(llm, "complete", complete)
    sender = _Sender()
    await pipeline.process(job, sender)

    assert sender.texts == ["00:00 Привет\n01:05 Главное"]
    name, data, caption = sender.files[0]
    assert name == "эфир.txt"
    assert data.startswith(pipeline.SYSTEM_PROMPT)
    assert "[00:00:00] Всем привет. Главная мысль." in data
    assert prompts[0][0] == pipeline.SYSTEM_PROMPT
    assert _leftovers(job) == []


async def test_ошибка_llm_отдаёт_файл_с_промптом(job, monkeypatch):
    async def complete(system, user):
        raise llm.LLMError("HTTP 500: boom")

    monkeypatch.setattr(llm, "complete", complete)
    sender = _Sender()
    await pipeline.process(job, sender)

    assert sender.texts == []
    name, data, caption = sender.files[0]
    assert data.startswith(pipeline.SYSTEM_PROMPT + "\n\n[00:00:00]")
    assert "HTTP 500" in caption
    assert _leftovers(job) == []


async def test_сбой_распознавания_убирает_файлы(job, monkeypatch):
    async def broken(pcm):
        raise RuntimeError("ffmpeg упал")

    monkeypatch.setattr(asr, "run_transcribe", broken)
    sender = _Sender()
    await pipeline.process(job, sender)

    assert "ffmpeg упал" in sender.texts[0]
    assert sender.files == []
    assert _leftovers(job) == []


async def test_тишина(job, monkeypatch):
    async def silent(pcm):
        return []

    monkeypatch.setattr(asr, "run_transcribe", silent)
    sender = _Sender()
    await pipeline.process(job, sender)
    assert sender.texts == ["Речь в эфире не распознана."]


def test_длинный_ответ_режется_по_строкам():
    text = "\n".join(f"{i:02d}:00 строка {'x' * 50}" for i in range(200))
    parts = pipeline.split_message(text, limit=1000)
    assert all(len(p) <= 1000 for p in parts)
    assert "\n".join(parts) == text


def test_сверхдлинная_строка_тоже_режется():
    parts = pipeline.split_message("x" * 2500, limit=1000)
    assert [len(p) for p in parts] == [1000, 1000, 500]
