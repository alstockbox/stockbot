from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    pass


class HttpTransport:
    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = float(timeout)

    def get_json(self, url: str, headers: dict[str, str] | None = None):
        request = Request(url, headers=headers or {}, method="GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise ProviderError(f"market data request failed: {type(exc).__name__}") from exc
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProviderError("market data provider returned invalid JSON") from exc
