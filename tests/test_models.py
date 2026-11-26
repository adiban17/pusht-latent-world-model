import pytest
import torch

from wm.models.encoder import latent_norm
from wm.models.world_model import EXPECTED_PARAMS, WorldModel, build_world_model, param_counts
from wm.repro import REPO_ROOT, load_yaml

CFG = load_yaml(REPO_ROOT / "configs/wm_v1.yaml")      # the §4 model as written (all Phase 3b flags off)
WM_CFG = load_yaml(REPO_ROOT / "configs/wm.yaml")    # the adopted A+B recipe


def test_param_counts_exact():
    counts = param_counts(build_world_model(CFG))
    assert counts == EXPECTED_PARAMS
    assert counts["total"] == 4_184_899 and counts["planner"] == 2_442_336


def test_shapes_and_latent_norm():
    torch.manual_seed(0)
    m = WorldModel()
    x = torch.rand(2, 5, 3, 64, 64) - 0.5
    z = m.encode(x)
    assert z.shape == (2, 5, 256)
    assert torch.allclose(z.mean(-1), torch.zeros(2, 5), atol=1e-5)
    # at init the pre-norm features are tiny (var ~1e-4), so LN's eps=1e-5 keeps var(z) a bit below 1
    assert (z.var(-1, unbiased=False) <= 1.0 + 1e-4).all()
    assert m.decode(z).shape == (2, 5, 3, 64, 64)
    y = latent_norm(torch.randn(3, 256) * 7 + 2)
    assert torch.allclose(y.mean(-1), torch.zeros(3), atol=1e-5)
    assert torch.allclose(y.var(-1, unbiased=False), torch.ones(3), atol=1e-3)


def test_planning_api_batched():
    torch.manual_seed(0)
    m = WorldModel()
    z_ctx, a_ctx = torch.randn(7, 2, 256), torch.rand(7, 2, 2) * 2 - 1
    h = m.warmup(z_ctx, a_ctx)
    assert h.shape == (7, 320)
    z_hat = m.rollout(torch.randn(7, 256), h, torch.rand(7, 10, 2) * 2 - 1)
    assert z_hat.shape == (7, 10, 256)
    # batched rollout == one-by-one rollout
    one = m.rollout(torch.randn(1, 256), h[:1], torch.zeros(1, 10, 2))
    assert one.shape == (1, 10, 256)


def test_window_loss_and_teacher_forcing():
    torch.manual_seed(0)
    frames, actions = torch.rand(2, 13, 3, 64, 64) - 0.5, torch.rand(2, 12, 2) * 2 - 1
    w = {"rec": 1.0, "lat": 0.1, "img": 1.0}
    m = WorldModel()
    out = m.window_loss(frames, actions, w)
    assert out["z"].shape == (2, 13, 256) and out["z_hat"].shape == (2, 10, 256)
    expected = out["rec"] + 0.1 * out["lat"] + out["img"]
    assert torch.allclose(out["loss"], expected)
    m.teacher_forcing = True
    out_tf = m.window_loss(frames, actions, w)
    assert torch.allclose(out_tf["z_hat"][:, 0], out["z_hat"][:, 0])   # step 1 is identical
    assert not torch.allclose(out_tf["z_hat"][:, 1:], out["z_hat"][:, 1:])
    out["loss"].backward()
    assert all(p.grad is not None for p in m.parameters())


def test_collapse_fix_variants_keep_param_counts_and_default_off():
    base = WorldModel()
    assert base.input_mean is None and not base.detach_dynamics_inputs
    cfg = {**CFG, "input_mean_frame": True, "detach_dynamics_inputs": True}
    m = build_world_model(cfg)
    assert param_counts(m) == EXPECTED_PARAMS          # the mean frame is a buffer
    assert "input_mean" in m.state_dict()
    m.set_input_mean(torch.full((3, 64, 64), 0.25))
    x = torch.full((1, 3, 64, 64), 0.25)
    m2 = build_world_model(CFG)
    m2.load_state_dict({k: v for k, v in m.state_dict().items() if k != "input_mean"})
    assert torch.allclose(m.encode(x), m2.encode(torch.zeros_like(x)), atol=1e-6)


def test_detach_variant_stops_latent_loss_reaching_encoder():
    torch.manual_seed(0)
    frames, actions = torch.rand(2, 13, 3, 64, 64) - 0.5, torch.rand(2, 12, 2) * 2 - 1
    m = WorldModel(detach_dynamics_inputs=True)
    out = m.window_loss(frames, actions, {"rec": 0.0, "lat": 1.0, "img": 1.0})
    out["loss"].backward()
    assert all(p.grad is None or not p.grad.any() for p in m.encoder.parameters())
    assert any(p.grad is not None and p.grad.any() for p in m.dynamics.parameters())


def test_variance_term_only_in_loss_when_weighted():
    torch.manual_seed(0)
    frames, actions = torch.rand(2, 13, 3, 64, 64) - 0.5, torch.rand(2, 12, 2) * 2 - 1
    m = WorldModel()
    w = {"rec": 1.0, "lat": 0.1, "img": 1.0}
    off = m.window_loss(frames, actions, w)
    on = m.window_loss(frames, actions, {**w, "var": 1.0})
    assert off["var"] > 0.5                                  # near-collapsed at init
    assert torch.allclose(on["loss"], off["loss"] + on["var"])


# ---- Phase 3b options ------------------------------------------------------------------------
from wm.models.world_model import EXPECTED_PARAMS_PROPRIO  # noqa: E402


def test_phase3b_flags_default_off_and_absent_from_wm_v1_yaml():
    m = build_world_model(CFG)
    assert m.fg_mean is None and not m.proprio and not hasattr(m.dynamics, "p_out")
    for k in ("fg_weight", "fg_alpha", "fg_threshold", "proprio", "keep_best"):
        assert k not in CFG
    assert "proprio" not in CFG["loss"]


def test_wm_yaml_builds_the_a_plus_b_model():
    m = build_world_model(WM_CFG)
    assert m.proprio and m.fg_mean is not None and m.fg_alpha == 20.0
    assert param_counts(m) == EXPECTED_PARAMS_PROPRIO


def test_proprio_param_counts():
    m = build_world_model({**CFG, "proprio": True})
    counts = param_counts(m)
    assert counts == EXPECTED_PARAMS_PROPRIO
    assert counts["total"] == 4_186_053 and counts["planner"] == 2_443_490
    assert counts["dynamics"] - EXPECTED_PARAMS["dynamics"] == 2 * 256 + (320 * 2 + 2)
    assert not m.dynamics.p_out.weight.any() and not m.dynamics.p_out.bias.any()   # zero init


def test_fg_weight_alpha_zero_equals_unweighted_and_matches_hand_computation():
    torch.manual_seed(0)
    frames = torch.full((1, 13, 3, 64, 64), 0.5)          # white background
    frames[..., 10:20, 10:20] = -0.5                       # a 10x10 black square
    actions = torch.zeros(1, 12, 2)
    w = {"rec": 1.0, "lat": 0.1, "img": 1.0}
    plain = WorldModel()
    ref = plain.window_loss(frames, actions, w)
    m0 = WorldModel(fg_weight=True, fg_alpha=0.0)
    m0.load_state_dict({**plain.state_dict(), "fg_mean": torch.full((3, 64, 64), 0.5)})
    out0 = m0.window_loss(frames, actions, w)
    assert torch.allclose(out0["rec"], ref["rec"]) and torch.allclose(out0["img"], ref["img"])

    m = WorldModel(fg_weight=True, fg_alpha=20.0)
    m.load_state_dict(m0.state_dict())
    out = m.window_loss(frames, actions, w)
    weight = torch.ones(64, 64)
    weight[10:20, 10:20] = 21.0
    dec = plain.decode(ref["z"]).float()
    hand = (weight * (dec - frames) ** 2).mean()
    assert torch.allclose(out["rec"], hand, rtol=1e-5)
    assert param_counts(m) == EXPECTED_PARAMS and "fg_mean" in m.state_dict()


def test_proprio_rollout_api_and_loss():
    torch.manual_seed(0)
    m = WorldModel(proprio=True)
    with torch.no_grad():                                  # non-zero head so p actually moves
        m.dynamics.p_out.weight.normal_(0, 0.01)
    z_ctx, a_ctx, p_ctx = torch.randn(5, 2, 256), torch.rand(5, 2, 2) * 2 - 1, torch.rand(5, 2, 2) * 2 - 1
    h = m.warmup(z_ctx, a_ctx, p_ctx)
    acts, z0, p0 = torch.rand(5, 10, 2) * 2 - 1, torch.randn(5, 256), torch.rand(5, 2) * 2 - 1
    z_hat, p_hat = m.rollout(z0, h, acts, p0=p0)
    assert z_hat.shape == (5, 10, 256) and p_hat.shape == (5, 10, 2)
    z1, p1 = m.rollout(z0[:1], h[:1], acts[:1], p0=p0[:1])
    assert torch.allclose(z1, z_hat[:1], atol=1e-5) and torch.allclose(p1, p_hat[:1], atol=1e-6)
    with pytest.raises(ValueError):
        m.rollout(z0, h, acts)
    with pytest.raises(ValueError):
        m.warmup(z_ctx, a_ctx)

    frames, actions = torch.rand(2, 13, 3, 64, 64) - 0.5, torch.rand(2, 12, 2) * 2 - 1
    pusher = torch.rand(2, 13, 2) * 2 - 1
    w = {"rec": 1.0, "lat": 0.1, "img": 1.0, "proprio": 1.0}
    with pytest.raises(ValueError):
        m.window_loss(frames, actions, w)
    out = m.window_loss(frames, actions, w, pusher=pusher)
    assert out["p_hat"].shape == (2, 10, 2)
    assert torch.allclose(out["proprio"], ((out["p_hat"] - pusher[:, 3:]) ** 2).mean())
    assert torch.allclose(out["loss"], out["rec"] + 0.1 * out["lat"] + out["img"] + out["proprio"])
    out["loss"].backward()
    assert m.dynamics.p_out.weight.grad.abs().sum() > 0


def test_proprio_teacher_forcing_feeds_real_pusher():
    torch.manual_seed(0)
    m = WorldModel(proprio=True, teacher_forcing=True)
    with torch.no_grad():
        m.dynamics.p_out.weight.normal_(0, 0.01)
    frames, actions = torch.rand(1, 13, 3, 64, 64) - 0.5, torch.rand(1, 12, 2) * 2 - 1
    pusher = torch.rand(1, 13, 2) * 2 - 1
    out = m.window_loss(frames, actions, {"rec": 1.0, "lat": 0.1, "img": 1.0}, pusher=pusher)
    # one-step predictions from the real pusher: p_hat_{k+1} - p_k is the head output only
    h = m.warmup(out["z"][:, :2], actions[:, :2], pusher[:, :2])
    z_k, p_k, _ = m.dynamics.step_proprio(out["z"][:, 2], pusher[:, 2], actions[:, 2], h)
    assert torch.allclose(out["p_hat"][:, 0], p_k)
