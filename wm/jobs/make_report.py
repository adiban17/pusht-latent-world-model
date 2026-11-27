"""Job make_report (Phase 8): render README.md from docs/README.template.md and the committed
results JSONs (wm.eval.readme), and write docs/readme_sources.md (every README number -> its source).

CHECKS (FAIL): every placeholder resolves; no untraced digits in the template's static text; every
resume claim is rendered; every figure exists; rendering is idempotent. WARN: README length.
"""
from __future__ import annotations

from pathlib import Path

from wm.eval import readme as rd
from wm.jobs import JobContext
from wm.report import Report

ROOT = Path(__file__).resolve().parents[2]
MIN_LINES, MAX_LINES = 300, 450


def run(ctx: JobContext, report: Report) -> None:
    out = rd.build(ROOT)
    again = rd.build(ROOT)
    report.check_true("placeholders_resolved", not out["missing"],
                      "all resolved" if not out["missing"] else ", ".join(out["missing"]), "every {{f:}} / {{t:}} known")
    report.check_true("no_untraced_numbers", not out["untraced"],
                      "none" if not out["untraced"] else ", ".join(out["untraced"][:20]),
                      "digits in static template text only from the allowlist")
    report.check_true("resume_claims_rendered", not out["claims_missing"],
                      f"{len(rd.RESUME_CLAIMS)} claims" if not out["claims_missing"] else ", ".join(out["claims_missing"]),
                      "every resume-claim fact appears in README")
    report.check_true("figures_exist", not out["figures_missing"],
                      f"{len(rd.FIGURES)} figures" if not out["figures_missing"] else ", ".join(out["figures_missing"]),
                      "every figure path exists")
    report.check_true("idempotent", out["readme"] == again["readme"] and out["sources"] == again["sources"],
                      "identical", "two renders give the same README")
    n_lines = out["readme"].count("\n")
    report.check("readme_length", "PASS" if MIN_LINES <= n_lines <= MAX_LINES else "WARN", f"{n_lines} lines",
                 f"{MIN_LINES}-{MAX_LINES}; WARN only")

    if not report.failed:
        (ROOT / rd.README).write_text(out["readme"])
        (ROOT / rd.SOURCES).write_text(out["sources"])
        report.save(rd.README)
        report.save(rd.SOURCES)
    report.metric("facts_used", sum(k.startswith("f:") for k in out["used"]))
    report.metric("tables_used", sum(k.startswith("t:") for k in out["used"]))
    report.metric("facts_available", len(out["facts"]))
    for claim, names in rd.RESUME_CLAIMS:
        report.metric("claim", f"{claim} -> " + " / ".join(out["facts"][n].value for n in names))
