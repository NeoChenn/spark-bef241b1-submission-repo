"""Gemini reasoning wrapper: slow path only, never in the real-time control loop.

Behaviour:
- Reads GEMINI_API_KEY from the environment. The key is never stored or logged.
- Retries 429 and 503 with exponential backoff.
- Caps calls per minute (the organisers disqualify abuse).
- Logs every call (model, prompt, config, output) to the decision log.
- If anything fails, raises GeminiUnavailable. Callers must fall back locally,
  for example by flagging a human. Core decisions must never depend on this call.

NOTE: the live call path is untested until a key is available. It uses
`client.models.generate_content` from the google-genai SDK. Check the Gemini skills
for the current recommended API before relying on it.
"""

import os
import time
from collections import deque
from typing import Any, Callable

from . import config
from .decision_log import DecisionLog


class GeminiUnavailable(RuntimeError):
    """Raised when a Gemini call cannot be made or has failed. Callers must fall back."""


class RateGuard:
    """Sliding-window limit on calls per minute."""

    def __init__(self, max_per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.max = max_per_minute
        self.clock = clock
        self.calls: deque[float] = deque()

    def allow(self) -> bool:
        now = self.clock()
        while self.calls and now - self.calls[0] >= 60:
            self.calls.popleft()
        if len(self.calls) >= self.max:
            return False
        self.calls.append(now)
        return True


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, 503):
        return True
    text = str(exc)
    return "429" in text or "503" in text


class GeminiReasoner:
    def __init__(
        self,
        model: str | None = None,
        log: DecisionLog | None = None,
        max_calls_per_minute: int | None = None,
        call_fn: Callable[[str, str, Any], str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
    ):
        self.model = model or config.REASONING_MODEL
        self.log = log
        self.guard = RateGuard(max_calls_per_minute or config.GEMINI_MAX_CALLS_PER_MINUTE)
        self.call_fn = call_fn or self._live_call
        self.sleep = sleep
        self.max_retries = max_retries

    @staticmethod
    def _live_call(model: str, prompt: str, generation_config: Any) -> str:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise GeminiUnavailable("GEMINI_API_KEY is not set")
        from google import genai  # imported lazily so tests and offline runs work without it

        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=model, contents=prompt, config=generation_config)
        return response.text

    def ask(self, prompt: str, generation_config: Any = None) -> str:
        """Ask the model. Raises GeminiUnavailable on any failure."""
        if not self.model:
            raise GeminiUnavailable("no model set: define GEMINI_REASONING_MODEL")
        if not self.guard.allow():
            raise GeminiUnavailable("local rate guard: too many calls this minute")

        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                output = self.call_fn(self.model, prompt, generation_config)
                self._log(prompt, generation_config, output=output)
                return output
            except GeminiUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - SDK raises varied types
                last = exc
                if not _is_retryable(exc) or attempt == self.max_retries:
                    break
                self.sleep(2**attempt)
        self._log(prompt, generation_config, error=str(last))
        raise GeminiUnavailable(f"Gemini call failed: {last}") from last

    def _log(self, prompt: str, generation_config: Any, **result: Any) -> None:
        if self.log:
            self.log.write(
                "genai_call",
                model=self.model,
                prompt=prompt[:4000],
                config=str(generation_config),
                **{k: (v[:4000] if isinstance(v, str) else v) for k, v in result.items()},
            )
