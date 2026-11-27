"""Regenerate results/final/summary.md from results/final/final_eval.json (+ imagined_angle.json).

Usage: .venv/bin/python scripts/make_final_summary.py [results/final]
Every number comes from the JSONs (wm.eval.final_summary); nothing is typed by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wm.eval.final_summary import write_summary  # noqa: E402


def main() -> None:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/final")
    final = json.loads((d / "final_eval.json").read_text())
    angle_path = d / "imagined_angle.json"
    angle = json.loads(angle_path.read_text()) if angle_path.exists() else None
    (d / "summary.md").write_text(write_summary(final, angle))
    print(f"wrote {d / 'summary.md'}")


if __name__ == "__main__":
    main()
