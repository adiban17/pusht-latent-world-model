"""Phase 8: README.md is rendered from the committed results; no number is typed by hand."""
import pytest

from wm.eval import readme as rd
from wm.repro import REPO_ROOT
from wm.registry import load_registry


@pytest.fixture(scope="module")
def built():
    return rd.build(REPO_ROOT)


def test_render_clean(built):
    assert built["missing"] == []
    assert built["untraced"] == []
    assert built["claims_missing"] == []
    assert built["figures_missing"] == []


def test_committed_readme_in_sync(built):
    """README.md and docs/readme_sources.md on disk are exactly what the template + results give."""
    assert (REPO_ROOT / rd.README).read_text() == built["readme"]
    assert (REPO_ROOT / rd.SOURCES).read_text() == built["sources"]
    assert rd.build(REPO_ROOT)["readme"] == built["readme"]  # deterministic


def test_untraced_numbers_detects_stray_digits():
    assert rd.untraced_numbers("success was 34.7% here") == ["34.7%"]
    assert rd.untraced_numbers("ok {{f:x}} at k = 10 on a T4 for cem_wm_1step, `code 42`") == []
    assert rd.untraced_numbers("text\n## References\n- Holm 1979") == []


def test_tables_match_summary_md(built):
    summary = (REPO_ROOT / "results/final/summary.md").read_text()
    for name in ("main", "primary", "tags", "bins_move", "bins_turn", "sweep"):
        assert built["tables"][name].markdown in summary, name


def test_facts_against_json_keys(built):
    F = built["facts"]
    assert F["cem_wm_success_mean"].value == "34.7"
    assert F["bc_clean_success_mean"].value == "67.5"
    assert F["pt_cem_wm_1step_succ_holm"].value == "0.0327"
    assert F["transitions"].value == "20,000" and F["test_transitions"].value == "3,000"
    assert F["cem_wm_model_error_pct"].value == "92" and F["cem_wm_failed"].value == "392"
    assert F["ang_wm_k10"].value == "60.2" and F["ang_verdict"].value == "confirmed"
    for name, fact in F.items():
        assert fact.source, name


def test_resume_claims_listed(built):
    assert "## Resume claims" in built["sources"]
    for claim, names in rd.RESUME_CLAIMS:
        assert claim in built["sources"]
        for n in names:
            assert built["facts"][n].value in built["readme"], n


def test_make_report_registered():
    entry = load_registry()["jobs"]["make_report"]
    assert entry["module"] == "wm.jobs.make_report" and entry["needs"] == [] and entry["makes"] == []
