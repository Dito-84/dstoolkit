"""Repository settings kept in files: the branch ruleset must name the checks CI actually runs."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# These check files that live only in the repository, not in the source distribution on PyPI
pytestmark = pytest.mark.skipif(not (ROOT / ".github").is_dir(), reason="needs the repository's .github folder")


def _ci_check_names() -> set[str]:
    """Check names GitHub shows for .github/workflows/ci.yml (job name, or id, plus matrix values)."""
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    jobs = re.split(r"\n  (?=[\w-]+:\n)", text.split("\njobs:\n", 1)[1])
    names = set()
    for job in jobs:
        job_id = job.split(":", 1)[0].strip()
        explicit = re.search(r"^    name: (.+)$", job, re.M)
        versions = re.search(r"python-version: \[([^\]]+)\]", job)
        if explicit:
            names.add(explicit.group(1).strip())
        elif versions:
            for version in versions.group(1).split(","):
                names.add(f"{job_id} ({version.strip().strip(chr(34))})")
        else:
            names.add(job_id)
    return names


def test_ruleset_requires_exactly_the_checks_ci_runs():
    # A required check that CI never reports (after renaming a job, say) would block every merge
    ruleset = json.loads((ROOT / ".github" / "rulesets" / "protect-main.json").read_text(encoding="utf-8"))
    rules = {r["type"]: r.get("parameters", {}) for r in ruleset["rules"]}
    required = {c["context"] for c in rules["required_status_checks"]["required_status_checks"]}
    assert required == _ci_check_names() == {
        "test (3.10)", "test (3.13)", "test (3.14)", "test (minimum versions)", "browser-app"}
    assert rules["pull_request"]["allowed_merge_methods"] == ["squash"]


def test_release_tags_are_protected():
    # Publishing a release uploads to PyPI: only admins (who bypass the ruleset) can create, move or delete v* tags
    ruleset = json.loads((ROOT / ".github" / "rulesets" / "protect-release-tags.json").read_text(encoding="utf-8"))
    assert ruleset["target"] == "tag" and ruleset["enforcement"] == "active"
    assert ruleset["conditions"]["ref_name"]["include"] == ["refs/tags/v*"]
    assert {r["type"] for r in ruleset["rules"]} == {"creation", "update", "deletion", "non_fast_forward"}
    assert ruleset["bypass_actors"] == [{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}]


def test_publish_checks_the_release_before_uploading():
    text = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    build = text.split("\n  build:\n", 1)[1].split("\n  testpypi:\n", 1)[0]
    steps = ["merge-base --is-ancestor", "doesn't match __version__", "pytest -q", "python -m build"]
    positions = [build.find(step) for step in steps]
    assert -1 not in positions and positions == sorted(positions), positions   # every check before the build
    assert "fetch-depth: 0" in build
