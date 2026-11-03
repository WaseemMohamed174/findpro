import json
import urllib.request

from . import settings
from .runtime import supervisor

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LLMError(Exception):
    pass


class _Remote:
    name = ""

    def _port(self):
        process = supervisor.get_worker(self.name)
        return process.port if process and process.is_ready() else None

    def ready(self):
        return self._port() is not None

    def _open(self, path, payload, timeout, stream=False):
        port = self._port()
        if port is None:
            raise LLMError(f"{self.name} is not ready")
        headers = {"Content-Type": "application/json", "X-Worker-Token": settings.WORKER_TOKEN}
        if stream:
            headers["Accept"] = "text/event-stream"
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        return _opener.open(req, timeout=timeout)


class Embedder(_Remote):
    name = "embedder"

    def embed(self, texts, kind="passage"):
        try:
            with self._open("/embed", {"texts": texts, "kind": kind}, 120.0) as resp:
                return json.loads(resp.read().decode("utf-8"))["vectors"]
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"embedder call failed: {e}")


class LLM(_Remote):
    name = "llm"

    def _payload(self, messages, max_tokens, temperature, stream, json_mode=False, repeat_penalty=None):
        payload = {
            "model": settings.LLM_NAME,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if repeat_penalty:
            payload["repeat_penalty"] = repeat_penalty
        return payload

    def chat(self, messages, json_mode=False, max_tokens=None, temperature=None):
        payload = self._payload(
            messages,
            max_tokens or settings.LLM_MAX_TOKENS,
            settings.LLM_TEMP_REPLY if temperature is None else temperature,
            False,
            json_mode,
        )
        try:
            with self._open("/v1/chat/completions", payload, settings.WORKER_CALL_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"llm chat failed: {e}")

    def stream(self, messages, max_tokens=None, temperature=None):
        payload = self._payload(
            messages,
            max_tokens or settings.LLM_MAX_TOKENS,
            settings.LLM_TEMP_REPLY if temperature is None else temperature,
            True,
            repeat_penalty=settings.LLM_REPEAT_PENALTY,
        )
        try:
            resp = self._open(
                "/v1/chat/completions", payload, settings.WORKER_CALL_TIMEOUT, stream=True
            )
            buf = b""
            for chunk in iter(lambda: resp.read(1024), b""):
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data: ") or line == b"data: [DONE]":
                        continue
                    try:
                        delta = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                    except (ValueError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"llm stream failed: {e}")


llm = LLM()
embedder = Embedder()
