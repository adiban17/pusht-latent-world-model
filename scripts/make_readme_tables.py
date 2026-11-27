"""Render README.md + docs/readme_sources.md from docs/README.template.md and the committed results.

Usage: .venv/bin/python scripts/make_readme_tables.py   (same as `run.py --job make_report`, without the REPORT)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wm.eval import readme as rd  # noqa: E402


def main() -> int:
    out = rd.build(ROOT)
    problems = out["missing"] + out["untraced"] + out["claims_missing"] + out["figures_missing"]
    if problems:
        print("not written; problems:", ", ".join(problems))
        return 1
    (ROOT / rd.README).write_text(out["readme"])
    (ROOT / rd.SOURCES).write_text(out["sources"])
    print(f"wrote {rd.README} ({out['readme'].count(chr(10))} lines) and {rd.SOURCES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
