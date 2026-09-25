from stockbot.runtime.readiness import paper_readiness


def test_paper_readiness_is_fail_closed_for_live_execution():
    report = paper_readiness(466.0)
    assert report.paper_shadow_ready
    assert report.live_ready is False
    assert report.checks["live_disabled_by_default"]
