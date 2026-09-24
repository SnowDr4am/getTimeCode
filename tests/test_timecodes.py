"""Промпт и метки расшифровки."""

import timecodes


def test_запрос_знает_длину_и_число_тайм_кодов():
    prompt = timecodes.request("[00:00:00] Привет.", 7206)
    assert prompt.startswith("Длительность эфира: 02:00:06. Нужно 30–48 тайм-кодов")
    assert prompt.endswith("Расшифровка:\n[00:00:00] Привет.")


def test_короткому_эфиру_минимум_три_главы():
    assert "Нужно 3 тайм-кодов" in timecodes.request("", 60)


def test_метки_расшифровки_в_формате_ответа():
    blocks = [{"start": 0, "end": 30, "text": "Привет."},
              {"start": 203.4, "end": 260, "text": "Тема."},
              {"start": 3725.4, "end": 3800, "text": "Итог."}]
    assert timecodes.render_transcript(blocks) == (
        "[00:00] Привет.\n[03:23] Тема.\n[01:02:05] Итог.")
