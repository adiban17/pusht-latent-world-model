import json

from wm.repro import REPO_ROOT


def test_runner_has_exactly_three_cells():
    nb = json.loads((REPO_ROOT / "notebooks" / "runner.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 3 and len(nb["cells"]) == 3
    assert code[0].startswith("# CELL 1: setup.")
    assert 'GITHUB_USER = "adiban17"' in code[0]
    assert "!pip install -q -e ." in code[0]
    assert code[1].startswith("# CELL 2: the only cell you edit")
    assert code[2] == "# CELL 3: run. Never edit.\n!python run.py --job {JOB} --mode {MODE} {EXTRA}"
