"""Нарезка дорожки на куски распознавания."""

import numpy as np
import pytest

import main


@pytest.fixture
def chunk_60(monkeypatch):
    monkeypatch.setattr(main, "ASR_CHUNK_SEC", 60.0)


def test_короткая_дорожка_одним_куском(chunk_60):
    assert main.chunk_bounds([10.0, 20.0], 45.0) == [(0.0, 45.0)]


def test_ровно_по_лимиту_не_режется(chunk_60):
    assert main.chunk_bounds([10.0], 60.0) == [(0.0, 60.0)]


def test_рез_попадает_на_ближайшую_паузу(chunk_60):
    gaps = [float(t) for t in range(10, 200, 10)]
    assert main.chunk_bounds(gaps, 200.0) == [(0.0, 70.0), (70.0, 140.0), (140.0, 200.0)]


def test_без_пауз_режем_жёстко(chunk_60):
    bounds = main.chunk_bounds([], 200.0)
    assert bounds == [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0), (180.0, 200.0)]


def test_далёкая_пауза_не_растягивает_кусок(chunk_60):
    """Пауза за пределом допуска хуже, чем шов посреди слова: иначе снова упрёмся в память."""
    bounds = main.chunk_bounds([500.0], 600.0)
    assert bounds[0] == (0.0, 60.0)
    assert all(end - start <= 2 * main.ASR_CHUNK_SEC for start, end in bounds)


@pytest.mark.parametrize("total", [61.0, 200.0, 9723.0])
def test_куски_покрывают_дорожку_без_дыр(chunk_60, total):
    gaps = [float(t) for t in range(7, int(total), 7)]
    bounds = main.chunk_bounds(gaps, total)
    assert bounds[0][0] == 0.0
    assert bounds[-1][1] == total
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
    assert all(end > start for start, end in bounds)


class _Segment:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _Info:
    language = "ru"


class _FakeModel:
    """По два сегмента на кусок с локальными таймкодами — проверяем сдвиг к общему таймлайну."""

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio, **kwargs):
        length = len(audio) / main.SAMPLE_RATE
        return [_Segment(0.0, 1.0, " раз "), _Segment(length - 1.0, length, "два")], _Info()


def test_таймкоды_сдвигаются_к_общему_таймлайну(chunk_60, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "decode_audio",
                        lambda *a, **k: np.zeros(180 * main.SAMPLE_RATE, dtype=np.float32))
    monkeypatch.setattr(main, "WhisperModel", _FakeModel)

    segments = main.run_asr(tmp_path / "audio.wav", "cpu", "int8")

    assert [(s["start"], s["end"]) for s in segments] == [
        (0.0, 1.0), (59.0, 60.0),
        (60.0, 61.0), (119.0, 120.0),
        (120.0, 121.0), (179.0, 180.0),
    ]
    assert [s["text"] for s in segments] == ["раз", "два"] * 3
