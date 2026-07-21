"""--curator lets a second labeler stamp their own name on every LabelRecord."""
from scripts.curator_serve import parse_args


def test_default_curator_is_kristian():
    args = parse_args([])
    assert args.curator == "kristian"


def test_curator_flag_overrides_default():
    args = parse_args(["--curator", "alex"])
    assert args.curator == "alex"
