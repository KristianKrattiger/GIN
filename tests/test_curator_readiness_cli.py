"""format_report renders the readiness gauge for the CLI."""
from gin.curator.readiness import ReadinessReport, ReadinessTarget
from scripts.curator_readiness import format_report


def test_format_report_shows_counts_and_verdict():
    rep = ReadinessReport(new_issue_frame=3, new_agree=12, new_unrelated=15,
                          target=ReadinessTarget(20, 20, 20), ready=False)
    out = format_report(rep)
    assert "issue_frame 3/20" in out
    assert "agree 12/20" in out
    assert "unrelated 15/20" in out
    assert "READY: False" in out
