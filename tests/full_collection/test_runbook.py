from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parent.parent.parent


def test_readme_hpc_section_present():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "## Full Collection on HPC" in readme


def test_readme_hpc_section_mentions_make_targets_in_order():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    section_start = readme.index("## Full Collection on HPC")
    section = readme[section_start:]
    i_serve = section.index("make serve")
    i_health = section.index("make health")
    i_collect = section.index("make collect")
    i_report = section.index("make collect-report")
    i_stop = section.index("make stop")
    assert i_serve < i_health < i_collect < i_report < i_stop


def test_readme_mentions_manifest_location():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("## Full Collection on HPC"):]
    assert "artifacts/dialogues/manifest.json" in section


def test_readme_mentions_resume_behavior():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("## Full Collection on HPC"):]
    assert "resumable" in section.lower() or "resume" in section.lower()
