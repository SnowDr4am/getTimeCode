"""Доступ к боту, время ожидания, имя файла расшифровки."""

from datetime import datetime

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update

import bot
import config
from jobs import JobQueue

ADMIN, STRANGER = 111, 222


class _RecordingSession(BaseSession):
    """Вместо Telegram: запоминает каждый вызов API, чтобы проверить, что бот промолчал."""

    def __init__(self):
        super().__init__()
        self.calls = []

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        return Message(message_id=1, date=datetime.now(), chat=Chat(id=ADMIN, type="private"))

    async def stream_content(self, *args, **kwargs):
        yield b""

    async def close(self):
        pass


async def _idle(job):
    pass


# Роутер модульный и цепляется к диспетчеру один раз.
_queue = JobQueue(_idle, speed=1.0)
_dp = Dispatcher(queue=_queue)
_dp.include_router(bot.router)


@pytest.fixture
def telegram(monkeypatch):
    monkeypatch.setattr(config, "ACCESS_IDS", frozenset({ADMIN}))
    session = _RecordingSession()

    async def feed(**message):
        payload = {"message_id": 1, "date": 0, **message}
        update = Update.model_validate({"update_id": 1, "message": payload})
        await _dp.feed_update(Bot("42:TEST", session=session), update)
        return session.calls

    return feed


def _private(user_id):
    return {"chat": {"id": user_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "first_name": "x"}}


def _group(user_id):
    return {"chat": {"id": -100, "type": "supergroup", "title": "g"},
            "from": {"id": user_id, "is_bot": False, "first_name": "x"}}


VIDEO = {"video": {"file_id": "f", "file_unique_id": "u", "width": 1, "height": 1,
                   "duration": 10, "file_name": "эфир.mp4", "file_size": 100}}


@pytest.mark.parametrize("content", [{"text": "/start"}, {"text": "привет"}, VIDEO,
                                     {"document": {"file_id": "f", "file_unique_id": "u",
                                                   "mime_type": "video/mp4"}}])
async def test_чужому_бот_не_отвечает_и_ничего_не_качает(telegram, content):
    assert await telegram(**_private(STRANGER), **content) == []
    assert _queue._waiting == []


async def test_в_группе_молчит_даже_для_админа(telegram):
    """Иначе результат увидели бы все участники группы."""
    assert await telegram(**_group(ADMIN), **VIDEO) == []
    assert await telegram(**_group(STRANGER), text="/start") == []


async def test_сообщение_от_имени_канала_игнорируется(telegram):
    calls = await telegram(chat={"id": -100, "type": "supergroup", "title": "g"},
                           sender_chat={"id": -200, "type": "channel", "title": "c"},
                           text="/start")
    assert calls == []


async def test_админу_в_личке_отвечает(telegram):
    calls = await telegram(**_private(ADMIN), text="/start")
    assert [type(c) for c in calls] == [SendMessage]
    assert calls[0].chat_id == ADMIN


def test_айди_через_запятую_и_пробелы():
    assert config.parse_ids("1, 2;3\n4") == frozenset({1, 2, 3, 4})
    assert config.parse_ids("") == frozenset()


@pytest.mark.parametrize("seconds, text", [
    (10, "1 мин"), (60, "1 мин"), (61, "2 мин"), (3600, "1 ч 0 мин"), (5000, "1 ч 24 мин"),
])
def test_время_ожидания(seconds, text):
    assert bot.humanize(seconds) == text


@pytest.mark.parametrize("name, title", [
    ("Эфир 12.09.mp4", "Эфир 12.09"),
    ('a:b*?"<>|c.mov', "a_b_c"),
    (None, "эфир"),
    ("...mp4", "эфир"),
])
def test_имя_расшифровки(name, title):
    assert bot.safe_title(name) == title
