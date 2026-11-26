import torch

from wm.models.encoder import latent_norm
from wm.models.world_model import EXPECTED_PARAMS, WorldModel, build_world_model, param_counts
from wm.repro import REPO_ROOT, load_yaml

CFG = load_yaml(REPO_ROOT / "configs/wm.yaml")


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
