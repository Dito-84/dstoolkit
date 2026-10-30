"""Repository settings kept in files: the branch ruleset must name the checks CI actually runs."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    assert required == _ci_check_names() == {"test (3.10)", "test (3.13)", "test (minimum versions)", "browser-app"}
    assert rules["pull_request"]["allowed_merge_methods"] == ["squash"]
