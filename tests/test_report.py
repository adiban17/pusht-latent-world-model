import pytest

from wm.report import FOOTER, HEADER, Report, format_duration


def _report():
    r = Report(job="env_check", mode="smoke", platform="local", device="cpu", git="abc1234")
    r.check("render64", "PASS", "(64, 64, 3) uint8", "shape and dtype")
    r.metric("matmul_2048_ms", 12.5)
    r.save("outputs/env_check/frame.png")
    return r


def test_layout():
    lines = _report().render(elapsed=3.21).splitlines()
    assert lines[0] == "==================== RUN REPORT ====================" == HEADER
    assert lines[-1] == FOOTER and len(FOOTER) == len(HEADER) and set(FOOTER) == {"="}
    assert lines[1] == "job: env_check   mode: smoke   platform: local   device: cpu"
    assert lines[2] == "git: abc1234   time: 3.2s   params: n/a"
    assert lines[3:10] == [
        "CHECKS",
        " [PASS] render64: (64, 64, 3) uint8 (shape and dtype)",
        "METRICS",
        " matmul_2048_ms = 12.5",
        "SAVED",
        " outputs/env_check/frame.png",
        "W&B: off",
    ]


def test_exit_code_and_file(tmp_path, capsys):
    r = _report()
    assert r.finish(tmp_path / "job") == 0
    assert (tmp_path / "job" / "REPORT.md").read_text().startswith(HEADER)
    assert HEADER in capsys.readouterr().out
    r.check("bad", "FAIL", 1, "< 0")
    assert r.finish(tmp_path / "job") == 1


def test_warn_is_not_failure(tmp_path):
    r = _report()
    r.check("meh", "WARN", 0.3, ">= 0.35")
    assert r.finish(tmp_path) == 0


def test_bad_status():
    with pytest.raises(ValueError):
        _report().check("x", "OK", 1, "")


def test_format_duration():
    assert format_duration(4.24) == "4.2s"
    assert format_duration(185) == "3m05s"
    assert format_duration(3723) == "1h02m03s"
