"""Phase 7 fix: files written by jobs into results/ or outputs/ do not make git_info '-dirty'."""
import subprocess

from wm.repro import git_info


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_results_and_outputs_ignored(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")
    assert not git_info(tmp_path).endswith("-dirty")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "final_eval.json").write_text("{}")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "log.txt").write_text("x")
    assert not git_info(tmp_path).endswith("-dirty")
    (tmp_path / "code.py").write_text("x = 1")
    assert git_info(tmp_path).endswith("-dirty")
