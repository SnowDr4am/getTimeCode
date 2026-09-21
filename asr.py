"""Звук из видео -> сегменты речи с таймкодами -> текст по смысловым блокам."""

import asyncio
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000  # whisper всё равно ресемплит к этой частоте
WHISPER_MODEL = "small"
LANGUAGE = "ru"
# На сервере 2 vCPU: больше потоков только толкаются между собой.
CPU_THREADS = 2
# Распознавание и ffmpeg уступают CPU соседним проектам на сервере.
NICE = 10
# Секунд эфира за секунду работы: замер на сервере — 3.16x (small, beam 1, лимит 1.5 CPU),
# пик памяти 1.3 ГБ на 10-минутном куске.
SPEED = 3.0

# Мел-спектрограмма считается сразу на весь вход, пик памяти линеен по длине куска.
ASR_CHUNK_SEC = 600.0
# Паузы ищутся окнами: вся дорожка во float32 — это 0,6 ГБ на 2,5 часа.
VAD_WINDOW_SEC = 600.0
VAD_SILENCE_MS = 500
PROGRESS_SEC = 300.0  # шаг лога прогресса по таймлайну эфира

BLOCK_SEC = 90.0      # целевая длина смыслового блока
MIN_BLOCK_SEC = 25.0  # раньше неё по паузе не режем
PAUSE_SEC = 1.5       # пауза как граница мысли


def hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


async def _run(*cmd: str) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        "nice", "-n", str(NICE), *cmd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode:
        raise RuntimeError(f"{cmd[0]}: {err.decode(errors='replace').strip()[-500:]}")
    return out


async def probe_duration(path: Path) -> float | None:
    """Длительность в секундах; None — звуковой дорожки нет или файл не медиа."""
    try:
        out = await _run("ffprobe", "-v", "error", "-show_entries",
                         "format=duration:stream=codec_type", "-of", "json", str(path))
    except RuntimeError:
        return None
    info = json.loads(out)
    if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        return None
    duration = float(info.get("format", {}).get("duration") or 0)
    return duration or None


async def extract_audio(src: Path, dst: Path) -> None:
    """Сырой PCM без заголовка: его читает memmap, не поднимая дорожку в память целиком."""
    await _run("ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src),
               "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", str(dst))


def load_pcm(path: Path) -> np.ndarray:
    if path.stat().st_size == 0:
        return np.zeros(0, dtype=np.int16)
    return np.memmap(path, dtype=np.int16, mode="r")


def to_float(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0


def speech_gaps(pcm: np.ndarray) -> list[float]:
    """Середины пауз между речью, в секундах. Пауза на стыке окон теряется — это лишь
    один кандидат на рез из сотен."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(min_silence_duration_ms=VAD_SILENCE_MS)
    window = int(VAD_WINDOW_SEC * SAMPLE_RATE)
    gaps = []
    for offset in range(0, len(pcm), window):
        speech = get_speech_timestamps(to_float(pcm[offset:offset + window]), options)
        gaps += [(offset + (prev["end"] + nxt["start"]) / 2) / SAMPLE_RATE
                 for prev, nxt in zip(speech, speech[1:], strict=False)]
    return gaps


def chunk_bounds(gaps: list[float], total: float) -> list[tuple[float, float]]:
    """Границы кусков распознавания по паузам между речью: стык не должен попадать
    на середину слова. Пауз нет — режем жёстко, память важнее шва."""
    if total <= ASR_CHUNK_SEC:
        return [(0.0, total)]

    bounds, start = [], 0.0
    while total - start > ASR_CHUNK_SEC:
        limit = start + ASR_CHUNK_SEC
        cut = next((g for g in gaps if g > limit), None)
        if cut is None or cut > limit + ASR_CHUNK_SEC:
            cut = limit
        bounds.append((start, cut))
        start = cut
    bounds.append((start, total))
    return bounds


def transcribe(pcm_path: str) -> list[dict]:
    """CPU-bound: выполняется в отдельном процессе, см. run_transcribe."""
    from faster_whisper import WhisperModel

    pcm = load_pcm(Path(pcm_path))
    if not len(pcm):
        return []
    duration = len(pcm) / SAMPLE_RATE
    chunks = chunk_bounds(speech_gaps(pcm) if duration > ASR_CHUNK_SEC else [], duration)
    print(f"[whisper] длительность={hms(duration)}, кусков={len(chunks)}", flush=True)

    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8",
                         cpu_threads=CPU_THREADS)
    result = []
    started, reported = time.monotonic(), 0.0
    for offset, end in chunks:
        segments, _ = model.transcribe(
            to_float(pcm[int(offset * SAMPLE_RATE):int(end * SAMPLE_RATE)]),
            language=LANGUAGE,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": VAD_SILENCE_MS},
        )
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            at = offset + seg.end
            result.append({"start": offset + seg.start, "end": at, "text": text})
            if at - reported >= PROGRESS_SEC:
                reported = at
                speed = at / (time.monotonic() - started)
                print(f"[whisper] {hms(at)} / {hms(duration)} ({speed:.2f}x)", flush=True)
    return result


async def run_transcribe(pcm_path: Path) -> list[dict]:
    """Процесс живёт одну задачу: вместе с ним уходит из памяти и модель."""
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                             initializer=os.nice, initargs=(NICE,)) as pool:
        return await asyncio.get_running_loop().run_in_executor(pool, transcribe, str(pcm_path))


def split_blocks(segments: list[dict]) -> list[dict]:
    """Режем поток сегментов на смысловые блоки: по длинной паузе, но не короче MIN_BLOCK_SEC."""
    blocks, current = [], []

    def flush():
        if current:
            blocks.append({
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join(s["text"] for s in current),
            })
            current.clear()

    for i, seg in enumerate(segments):
        current.append(seg)
        length = seg["end"] - current[0]["start"]
        pause = segments[i + 1]["start"] - seg["end"] if i + 1 < len(segments) else 0.0
        if length >= BLOCK_SEC or (length >= MIN_BLOCK_SEC and pause >= PAUSE_SEC):
            flush()
    flush()
    return blocks


def render_transcript(blocks: list[dict]) -> str:
    return "\n".join(f"[{hms(b['start'])}] {b['text']}" for b in blocks)
