from pathlib import Path


def test_provider_smoke_workflow_is_separate_safe_and_reported():
    workflow = Path(".github/workflows/provider-smoke.yml")
    assert workflow.exists(), "external provider smoke workflow is missing"

    text = workflow.read_text(encoding="utf-8")
    assert "name: StockBot Provider Smoke" in text
    assert "workflow_dispatch:" in text
    assert "schedule:" in text
    assert "pull_request:" in text
    assert "python -m stockbot.cli.provider_smoke" in text
    assert "--report provider-smoke.json" in text
    assert "TIINGO_API_TOKEN: ${{ secrets.TIINGO_API_TOKEN }}" in text
    assert "Probe raw Riksbank HTTP endpoint" in text
    assert "curl -sS" in text
    assert "https://api.riksbank.se/monetary_policy_data/v1/forecasts?series=SEQRATENAYNA" in text
    assert "riksbank-http-status.txt" in text
    assert "Probe Riksbank Accept header" in text
    assert "Accept: application/json" in text
    assert "riksbank-accept-status.txt" in text
    assert "Probe Riksbank StockBot user agent" in text
    assert "StockBot/2" in text
    assert "riksbank-user-agent-status.txt" in text
    assert "Probe Riksbank provider headers" in text
    assert "riksbank-provider-headers-status.txt" in text
    assert "python -m stockbot.cli.riksbank_macro" in text
    assert "--series policy_rate" in text
    assert "Probe external Riksbank default forecast" in text
    assert "--output riksbank-default-smoke.json" in text
    assert "--policy-round 2026:2" in text
    assert "--output riksbank-explicit-round-smoke.json" in text
    assert "--policy-round latest" in text
    assert "--output riksbank-smoke.json" in text
    assert "actions/upload-artifact@v4" in text
    assert "riksbank-http-status.txt" in text
    assert "riksbank-accept-status.txt" in text
    assert "riksbank-user-agent-status.txt" in text
    assert "riksbank-provider-headers-status.txt" in text
    assert "riksbank-default-smoke.json" in text
    assert "riksbank-explicit-round-smoke.json" in text
    assert "riksbank-smoke.json" in text
    assert "BROKER" not in text.upper()
