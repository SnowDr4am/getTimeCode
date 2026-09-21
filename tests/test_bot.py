"""Мелочи бота: доступ, время ожидания, имя файла расшифровки."""

import pytest

import bot
import config


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
