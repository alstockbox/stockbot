from io import BytesIO
from urllib.error import HTTPError

import pytest

import stockbot.data.providers.http as http_module
from stockbot.data.providers.http import HttpTransport, ProviderError


def test_http_transport_surfaces_http_status_without_response_body(monkeypatch):
    def _raise_http_error(request, timeout):
        raise HTTPError(
            request.full_url,
            404,
            "Not Found",
            hdrs=None,
            fp=BytesIO(b"sensitive upstream body"),
        )

    monkeypatch.setattr(http_module, "urlopen", _raise_http_error)

    with pytest.raises(ProviderError) as exc_info:
        HttpTransport().get_json("https://example.test/private?token=secret")

    message = str(exc_info.value)
    assert "HTTP 404" in message
    assert "sensitive upstream body" not in message
    assert "token=secret" not in message
