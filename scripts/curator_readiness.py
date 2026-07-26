"""Print sub-project B's labeling readiness without launching the server.

    venv/Scripts/python.exe scripts/curator_readiness.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gin.curator.readiness import ReadinessReport, ReadinessTarget, readiness
from gin.curator.store import Store

DEFAULT_LOG = Path("data/curator/labels.jsonl")


def format_report(rep: ReadinessReport) -> str:
    t = rep.target
    return (
        f"issue_frame {rep.new_issue_frame}/{t.issue_frame}\n"
        f"agree {rep.new_agree}/{t.agree}\n"
        f"unrelated {rep.new_unrelated}/{t.unrelated}\n"
        f"story {rep.new_story}/{t.story}\n"
        f"READY: {rep.ready}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="curator labeling readiness for sub-project B")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--issue-frame-target", type=int, default=20)
    ap.add_argument("--agree-target", type=int, default=20)
    ap.add_argument("--unrelated-target", type=int, default=20)
    ap.add_argument("--story-target", type=int, default=20)
    args = ap.parse_args()
    target = ReadinessTarget(
        args.issue_frame_target,
        args.agree_target,
        args.unrelated_target,
        args.story_target,
    )
    print(format_report(readiness(Store(args.log), target)))


if __name__ == "__main__":
    main()
