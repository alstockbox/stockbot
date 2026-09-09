from stockbot.research import (
    DeepResearchFinding,
    JsonlDeepResearchMemory,
    ResearchCycleManifest,
    build_research_cycle_id,
    make_deep_findings,
    make_research_cycle_manifest,
    rank_adaptive_parent_records,
)


def test_deep_feedback_public_api_exports_are_stable():
    assert DeepResearchFinding.__name__ == "DeepResearchFinding"
    assert JsonlDeepResearchMemory.__name__ == "JsonlDeepResearchMemory"
    assert ResearchCycleManifest.__name__ == "ResearchCycleManifest"
    assert callable(build_research_cycle_id)
    assert callable(make_deep_findings)
    assert callable(make_research_cycle_manifest)
    assert callable(rank_adaptive_parent_records)
