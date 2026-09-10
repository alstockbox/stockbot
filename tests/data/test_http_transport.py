from types import SimpleNamespace

import pytest

import stockbot.data.providers.http as http_module
from stockbot.data.providers.http import HttpTransport, ProviderError


class _FakeRequestError(Exception):
    pass


class _FakeHTTPStatusError(Exception):
    def __init__(self, response):
        super().__init__("upstream HTTP failure")
        self.response = response


class _FakeResponse:
    def __init__(self, status_code: int, payload, *, body: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise _FakeHTTPStatusError(self)

    def json(self):
        return self._payload


def test_http_transport_uses_modern_client_and_preserves_safe_error_contract(monkeypatch):
    responses = [
        _FakeResponse(200, {"ok": True}),
        _FakeResponse(404, {}, body="sensitive upstream body"),
    ]
    calls = []

    class _FakeClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, *, headers):
            calls.append((url, headers, self.timeout))
            return responses.pop(0)

    fake_httpx = SimpleNamespace(
        Client=_FakeClient,
        HTTPStatusError=_FakeHTTPStatusError,
        RequestError=_FakeRequestError,
    )
    monkeypatch.setattr(http_module, "httpx", fake_httpx, raising=False)

    def _forbid_urllib(*args, **kwargs):
        raise AssertionError("urllib transport must not be used")

    monkeypatch.setattr(http_module, "urlopen", _forbid_urllib, raising=False)

    transport = HttpTransport(timeout=12.5)
    headers = {"Accept": "application/json", "User-Agent": "StockBot/2"}
    url = "https://example.test/private?token=secret"

    assert transport.get_json(url, headers) == {"ok": True}
    assert calls[0] == (url, headers, 12.5)

    with pytest.raises(ProviderError) as exc_info:
        transport.get_json(url, headers)

    message = str(exc_info.value)
    assert "HTTP 404" in message
    assert "sensitive upstream body" not in message
    assert "token=secret" not in message
