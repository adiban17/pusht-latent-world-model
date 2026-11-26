import torch
from torch import nn

from wm.jobs._trainer import ROUNDTRIP_ATOL, ROUNDTRIP_RTOL, check_roundtrip, roundtrip
from wm.report import Report


def _pair():
    torch.manual_seed(0)
    a = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5), nn.Linear(8, 4))
    b = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.5), nn.Linear(8, 4))
    b.load_state_dict(a.state_dict())
    return a.train(), b.train()


def test_identical_reload_passes_and_sets_eval_mode():
    a, b = _pair()
    rt = roundtrip(a, b, a, b, torch.randn(5, 8))       # dropout would differ in train mode
    assert rt["ok"] and rt["weights_identical"] and rt["max_abs_diff"] == 0.0
    assert not a.training and not b.training


def test_float_noise_within_tolerance_passes_but_real_differences_fail():
    a, b = _pair()
    x = torch.randn(5, 8)
    noise = lambda f: (lambda x: f(x) * (1 + ROUNDTRIP_RTOL / 10))          # noqa: E731
    assert roundtrip(a, b, a, noise(b), x)["ok"]
    big = lambda f: (lambda x: f(x) + 10 * (ROUNDTRIP_ATOL + ROUNDTRIP_RTOL))  # noqa: E731
    assert not roundtrip(a, b, a, big(b), x)["ok"]
    with torch.no_grad():
        b[0].bias[0] += 1e-7                                   # weights must be bit-identical
    rt = roundtrip(a, b, a, b, x)
    assert not rt["weights_identical"] and not rt["ok"]


def test_report_line_shows_diff_and_tolerance():
    a, b = _pair()
    r = Report(job="t", mode="smoke", platform="local", device="cpu", git="x", wandb="off")
    check_roundtrip(r, roundtrip(a, b, a, b, torch.randn(2, 8)), "outputs")
    line = r.render()
    assert "max |diff| 0.00e+00" in line and "rtol 0.0001, atol 1e-05" in line
