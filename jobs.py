"""Очередь эфиров: в работе всегда один, остальные ждут и знают, сколько."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

# Извлечение звука, запуск модели и ответ LLM поверх самого распознавания.
OVERHEAD_SEC = 300.0
# Текущая задача может затянуться дольше оценки — «0 минут» в ответе вводил бы в заблуждение.
MIN_REMAINING_SEC = 60.0

log = logging.getLogger(__name__)


@dataclass(eq=False)
class Job:
    chat_id: int
    reply_to: int
    source: Path
    duration: float
    title: str


class JobQueue:
    def __init__(self, handler: Callable[[Job], Awaitable[None]], speed: float) -> None:
        self._handler = handler
        self._speed = speed
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._waiting: list[Job] = []
        self._current: Job | None = None
        self._started = 0.0

    def estimate(self, job: Job) -> float:
        return job.duration / self._speed + OVERHEAD_SEC

    def put(self, job: Job) -> tuple[int, float]:
        """Сколько задач впереди и через сколько секунд ждать результат."""
        wait = sum(self.estimate(j) for j in self._waiting)
        if self._current is not None:
            elapsed = time.monotonic() - self._started
            wait += max(self.estimate(self._current) - elapsed, MIN_REMAINING_SEC)
        ahead = len(self._waiting) + (self._current is not None)
        self._waiting.append(job)
        self._queue.put_nowait(job)
        return ahead, wait + self.estimate(job)

    async def run(self) -> None:
        while True:
            job = await self._queue.get()
            self._waiting.remove(job)
            self._current, self._started = job, time.monotonic()
            try:
                await self._handler(job)
            except Exception:
                log.exception("задача %s упала", job.title)
            finally:
                log.info("задача %s: %.0f с", job.title, time.monotonic() - self._started)
                self._current = None
