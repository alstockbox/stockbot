from __future__ import annotations

import httpx


class ProviderError(RuntimeError):
    pass


class HttpTransport:
    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = float(timeout)

    def get_json(self, url: str, headers: dict[str, str] | None = None):
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, headers=headers or {})
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"market data request failed: HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderError(
                f"market data request failed: {type(exc).__name__}"
            ) from exc
        except ValueError as exc:
            raise ProviderError("market data provider returned invalid JSON") from exc
