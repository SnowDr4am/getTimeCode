"""Эфир целиком: ответ пользователю и уборка файлов. Распознавание подменено."""

import pytest

import asr
import pipeline
import timecodes
from jobs import Job


class _Sender:
    def __init__(self):
        self.texts, self.files = [], []

    async def text(self, chat_id, text, reply_to):
        self.texts.append(text)

    async def file(self, chat_id, name, data, caption, reply_to):
        self.files.append((name, data.decode(), caption))


# Длинная пауза после 30 с делит расшифровку на два блока: [00:00] и [01:05].
SEGMENTS = [{"start": 0.0, "end": 30.0, "text": "Всем привет."},
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


async def test_успех_только_файл_с_промптом_и_расшифровкой(job):
    sender = _Sender()
    await pipeline.process(job, sender)

    assert sender.texts == []
    [(name, data, caption)] = sender.files
    assert name == "эфир.txt"
    assert data.startswith(timecodes.SYSTEM_PROMPT + "\n\nДлительность эфира: 01:10. Нужно")
    assert data.rstrip().endswith("[00:00] Всем привет.\n[01:05] Главная мысль.")
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
