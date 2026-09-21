"""Очередь: одна задача в работе, честная оценка ожидания."""

import asyncio
from pathlib import Path

import pytest

import jobs


def _job(duration: float, title: str = "эфир") -> jobs.Job:
    return jobs.Job(chat_id=1, reply_to=1, source=Path(title), duration=duration, title=title)


async def _idle(job):
    pass


def test_первый_эфир_сразу_в_работу():
    queue = jobs.JobQueue(_idle, speed=2.0)
    ahead, eta = queue.put(_job(600))
    assert ahead == 0
    assert eta == 300 + jobs.OVERHEAD_SEC


def test_ожидание_суммирует_очередь():
    queue = jobs.JobQueue(_idle, speed=1.0)
    queue.put(_job(600))
    ahead, eta = queue.put(_job(1200))
    assert ahead == 1
    assert eta == 600 + 1200 + 2 * jobs.OVERHEAD_SEC


async def test_обрабатывается_строго_по_одной():
    running, peak, done = 0, 0, []

    async def handler(job):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        done.append(job.title)

    queue = jobs.JobQueue(handler, speed=1.0)
    worker = asyncio.create_task(queue.run())
    for title in "abc":
        queue.put(_job(10, title))
    while len(done) < 3:
        await asyncio.sleep(0.01)
    worker.cancel()

    assert done == ["a", "b", "c"]
    assert peak == 1


async def test_текущая_задача_учитывается_в_ожидании():
    started, release = asyncio.Event(), asyncio.Event()

    async def handler(job):
        started.set()
        await release.wait()

    queue = jobs.JobQueue(handler, speed=1.0)
    worker = asyncio.create_task(queue.run())
    queue.put(_job(600))
    await started.wait()

    ahead, eta = queue.put(_job(600))
    release.set()
    worker.cancel()

    assert ahead == 1
    assert eta == pytest.approx(2 * (600 + jobs.OVERHEAD_SEC), abs=1)


async def test_упавшая_задача_не_останавливает_очередь():
    done = []

    async def handler(job):
        if job.title == "bad":
            raise RuntimeError("boom")
        done.append(job.title)

    queue = jobs.JobQueue(handler, speed=1.0)
    worker = asyncio.create_task(queue.run())
    queue.put(_job(10, "bad"))
    queue.put(_job(10, "good"))
    while not done:
        await asyncio.sleep(0.01)
    worker.cancel()
    assert done == ["good"]
