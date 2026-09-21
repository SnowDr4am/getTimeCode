"""Нарезка дорожки, распознавание кусками и сборка расшифровки."""

import numpy as np
import pytest

import asr


@pytest.fixture
def chunk_60(monkeypatch):
    monkeypatch.setattr(asr, "ASR_CHUNK_SEC", 60.0)


def test_короткая_дорожка_одним_куском(chunk_60):
    assert asr.chunk_bounds([10.0, 20.0], 45.0) == [(0.0, 45.0)]


def test_ровно_по_лимиту_не_режется(chunk_60):
    assert asr.chunk_bounds([10.0], 60.0) == [(0.0, 60.0)]


def test_рез_попадает_на_ближайшую_паузу(chunk_60):
    gaps = [float(t) for t in range(10, 200, 10)]
    assert asr.chunk_bounds(gaps, 200.0) == [(0.0, 70.0), (70.0, 140.0), (140.0, 200.0)]


def test_без_пауз_режем_жёстко(chunk_60):
    bounds = asr.chunk_bounds([], 200.0)
    assert bounds == [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0), (180.0, 200.0)]


def test_далёкая_пауза_не_растягивает_кусок(chunk_60):
    """Пауза за пределом допуска хуже, чем шов посреди слова: иначе снова упрёмся в память."""
    bounds = asr.chunk_bounds([500.0], 600.0)
    assert bounds[0] == (0.0, 60.0)
    assert all(end - start <= 2 * asr.ASR_CHUNK_SEC for start, end in bounds)


@pytest.mark.parametrize("total", [61.0, 200.0, 9723.0])
def test_куски_покрывают_дорожку_без_дыр(chunk_60, total):
    gaps = [float(t) for t in range(7, int(total), 7)]
    bounds = asr.chunk_bounds(gaps, total)
    assert bounds[0][0] == 0.0
    assert bounds[-1][1] == total
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False))
    assert all(end > start for start, end in bounds)


class _Segment:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _FakeModel:
    """По два сегмента на кусок с локальными таймкодами — проверяем сдвиг к общему таймлайну."""

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio, **kwargs):
        assert audio.dtype == np.float32
        length = len(audio) / asr.SAMPLE_RATE
        return [_Segment(0.0, 1.0, " раз "), _Segment(length - 1.0, length, "два")], None


def test_таймкоды_сдвигаются_к_общему_таймлайну(chunk_60, monkeypatch, tmp_path):
    pcm = tmp_path / "audio.pcm"
    np.zeros(180 * asr.SAMPLE_RATE, dtype=np.int16).tofile(pcm)
    monkeypatch.setattr("faster_whisper.WhisperModel", _FakeModel)

    segments = asr.transcribe(str(pcm))

    assert [(s["start"], s["end"]) for s in segments] == [
        (0.0, 1.0), (59.0, 60.0),
        (60.0, 61.0), (119.0, 120.0),
        (120.0, 121.0), (179.0, 180.0),
    ]
    assert [s["text"] for s in segments] == ["раз", "два"] * 3


def test_пустая_дорожка_без_сегментов(monkeypatch, tmp_path):
    pcm = tmp_path / "audio.pcm"
    pcm.write_bytes(b"")
    monkeypatch.setattr("faster_whisper.WhisperModel", _FakeModel)
    assert asr.transcribe(str(pcm)) == []


def _seg(start, end, text="слово"):
    return {"start": start, "end": end, "text": text}


def test_блок_режется_по_паузе_после_минимальной_длины():
    segments = [_seg(0, 10), _seg(10, 30), _seg(33, 40), _seg(40, 50)]
    blocks = asr.split_blocks(segments)
    assert [(b["start"], b["end"]) for b in blocks] == [(0, 30), (33, 50)]


def test_короткий_блок_не_режется_по_паузе():
    segments = [_seg(0, 10), _seg(15, 20)]
    assert len(asr.split_blocks(segments)) == 1


def test_длинная_речь_без_пауз_режется_по_длине():
    segments = [_seg(t, t + 10) for t in range(0, 200, 10)]
    blocks = asr.split_blocks(segments)
    assert all(b["end"] - b["start"] <= asr.BLOCK_SEC for b in blocks)
    assert blocks[-1]["end"] == 200


def test_расшифровка_с_таймкодами_блоков():
    blocks = [{"start": 0, "end": 30, "text": "Привет."},
              {"start": 3725.4, "end": 3800, "text": "Итог."}]
    assert asr.render_transcript(blocks) == "[00:00:00] Привет.\n[01:02:05] Итог."


async def test_длительность_и_звук_через_ffmpeg(tmp_path):
    """Реальный ffmpeg: 3 секунды тона -> PCM ровно на 3 секунды."""
    src = tmp_path / "tone.mp4"
    await asr._run("ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi",
                   "-i", "sine=duration=3", "-c:a", "aac", str(src))
    assert round(await asr.probe_duration(src)) == 3

    pcm = tmp_path / "tone.pcm"
    await asr.extract_audio(src, pcm)
    assert len(asr.load_pcm(pcm)) == pytest.approx(3 * asr.SAMPLE_RATE, rel=0.02)


async def test_файл_без_звука_не_медиа(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_text("это не видео")
    assert await asr.probe_duration(junk) is None
