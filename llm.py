"""Grok через шлюз Kie (OpenAI Responses API).

Шлюз отдаёт ошибки и честным HTTP-статусом, и в HTTP 200 с конвертом
`{"code": 401, "msg": "..."}` — поэтому смотрим на оба."""

import asyncio
import json
import logging

import aiohttp

import config

KIE_URL = "https://api.kie.ai/grok/v1/responses"
MODEL = "grok-4-3"
# Вход — расшифровка многочасового эфира, плюс размышления модели: ответ идёт минутами.
TIMEOUT_SEC = 900
READ_TIMEOUT_SEC = 180  # тишина в стриме дольше этого — соединение мертво
REASONING = "medium"
ATTEMPTS = 2

log = logging.getLogger(__name__)


class LLMError(Exception):
    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def _retryable(code: int) -> bool:
    return code == 429 or code >= 500


def parse_response(status: int, data: object) -> str:
    if not isinstance(data, dict):
        raise LLMError(f"HTTP {status}: тело ответа не JSON-объект", _retryable(status))

    error = data.get("error")
    message = (error.get("message") if isinstance(error, dict) else error) or data.get("msg")
    if status >= 400:
        raise LLMError(f"HTTP {status}: {message or 'без описания'}", _retryable(status))
    code = data.get("code")
    if isinstance(code, int) and code != 200:
        raise LLMError(f"код {code}: {message or 'без описания'}", _retryable(code))
    if error:
        raise LLMError(f"ошибка шлюза: {message}")

    # В output до сообщения идут блоки размышлений (type=reasoning) — берём только текст ответа.
    text = "".join(
        part.get("text") or ""
        for item in data.get("output") or []
        if isinstance(item, dict) and item.get("type") == "message"
        for part in item.get("content") or []
        if isinstance(part, dict) and part.get("type") == "output_text"
    ).strip()
    if not text:
        raise LLMError(f"пустой ответ модели (status={data.get('status')})", retryable=True)
    return text


async def _events(resp: aiohttp.ClientResponse):
    """События SSE как словари. Строки режем сами: `async for` по resp.content падает на
    строке длиннее 64 КБ, а финальное событие несёт весь ответ целиком."""
    buffer, name = b"", ""
    async for chunk in resp.content.iter_any():
        buffer += chunk
        *lines, buffer = buffer.split(b"\n")
        for raw in lines:
            line = raw.decode(errors="replace").strip()
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                try:
                    event = json.loads(line[5:])
                except ValueError:
                    continue
                if isinstance(event, dict):
                    event.setdefault("type", name)
                    yield event


async def _read(resp: aiohttp.ClientResponse) -> str:
    # Отказ (неверный ключ, лимиты) шлюз отдаёт обычным JSON даже на стриминговый запрос.
    if resp.content_type != "text/event-stream":
        try:
            data = await resp.json(content_type=None)
        except ValueError:
            data = None
        return parse_response(resp.status, data)

    async for event in _events(resp):
        kind = event["type"]
        if kind == "response.completed":
            return parse_response(200, event.get("response"))
        if kind in ("response.failed", "response.incomplete", "error"):
            error = (event.get("response") or {}).get("error") or event.get("message")
            raise LLMError(f"{kind}: {error or 'без описания'}", retryable=True)
    raise LLMError("стрим оборвался до ответа", retryable=True)


async def complete(system: str, user: str, *, url: str = KIE_URL) -> str:
    payload = {
        # Без стрима Cloudflare перед шлюзом рвёт ответ через 100 с (HTTP 524), а на
        # многочасовом эфире модель думает дольше; в стриме шлюз шлёт keepalive.
        "model": MODEL,
        "stream": True,
        "instructions": system,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": user}]}],
        "reasoning": {"effort": REASONING},
    }
    headers = {"Authorization": f"Bearer {config.KIE_API_KEY}"}
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_SEC, sock_read=READ_TIMEOUT_SEC)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(1, ATTEMPTS + 1):
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    return await _read(resp)
            except LLMError as exc:
                error = exc
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                error = LLMError(f"сеть: {exc!r}", retryable=True)
            log.warning("LLM, попытка %d/%d: %s", attempt, ATTEMPTS, error)
            if not error.retryable:
                break
    raise error
