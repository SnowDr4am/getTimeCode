"""Разбор ответов шлюза Kie и повторы. Шлюз подменяется локальным aiohttp-сервером."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import llm

OK = {
    "status": "completed",
    "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": "00:00 Начало"}]},
    ],
}


def test_текст_берётся_из_сообщения_а_не_из_размышлений():
    assert llm.parse_response(200, OK) == "00:00 Начало"


def test_ошибка_в_конверте_при_http_200():
    body = {"code": 401, "msg": "Unauthorized"}
    with pytest.raises(llm.LLMError, match="401") as exc:
        llm.parse_response(200, body)
    assert not exc.value.retryable


@pytest.mark.parametrize("status, retryable",
                         [(400, False), (401, False), (429, True), (500, True)])
def test_http_ошибки(status, retryable):
    with pytest.raises(llm.LLMError) as exc:
        llm.parse_response(status, {"error": {"message": "boom"}})
    assert exc.value.retryable is retryable


def test_пустое_тело_500():
    with pytest.raises(llm.LLMError) as exc:
        llm.parse_response(500, {})
    assert exc.value.retryable


def test_ответ_без_текста():
    with pytest.raises(llm.LLMError):
        llm.parse_response(200, {"status": "incomplete", "output": [{"type": "reasoning"}]})


def test_не_json():
    with pytest.raises(llm.LLMError):
        llm.parse_response(502, None)


async def _serve(responses: list[tuple[int, object]]):
    calls = []

    async def handler(request: web.Request) -> web.Response:
        calls.append(await request.json())
        status, body = responses[min(len(calls), len(responses)) - 1]
        if isinstance(body, str):
            return web.Response(status=status, text=body)
        return web.json_response(body, status=status)

    app = web.Application()
    app.router.add_post("/responses", handler)
    server = TestServer(app)
    await server.start_server()
    return server, calls


async def test_запрос_с_промптом_и_расшифровкой(monkeypatch):
    monkeypatch.setattr(llm.config, "KIE_API_KEY", "key")
    server, calls = await _serve([(200, OK)])
    try:
        text = await llm.complete("промпт", "расшифровка", url=str(server.make_url("/responses")))
    finally:
        await server.close()

    assert text == "00:00 Начало"
    assert calls[0]["model"] == llm.MODEL
    assert calls[0]["instructions"] == "промпт"
    assert calls[0]["input"][0]["content"][0]["text"] == "расшифровка"


async def test_сбой_шлюза_повторяется():
    server, calls = await _serve([(502, "<html>bad gateway</html>"), (200, OK)])
    try:
        text = await llm.complete("p", "u", url=str(server.make_url("/responses")))
    finally:
        await server.close()
    assert text == "00:00 Начало"
    assert len(calls) == 2


async def test_ошибки_кончаются_исключением():
    server, calls = await _serve([(500, {})])
    try:
        with pytest.raises(llm.LLMError):
            await llm.complete("p", "u", url=str(server.make_url("/responses")))
    finally:
        await server.close()
    assert len(calls) == llm.ATTEMPTS


async def test_отказ_по_ключу_не_повторяется():
    server, calls = await _serve([(200, {"code": 401, "msg": "Unauthorized"})])
    try:
        with pytest.raises(llm.LLMError, match="401"):
            await llm.complete("p", "u", url=str(server.make_url("/responses")))
    finally:
        await server.close()
    assert len(calls) == 1
