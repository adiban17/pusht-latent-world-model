"""Phase 6b cost_space diagnostic: pixel cost, goal mask without the pusher, tau_pixel, the
pre-registered decision rule, the reproduction comparison, and that the default is unchanged."""
import numpy as np
import pytest
import torch

from wm.envs.pusht import get_state, make_env, render64, reset_to_state
from wm.eval.planning import calibrate_tau_pixel, compare_runs, select_cost_arm, summarize
from wm.models.pixel_unet import PixelUNet
from wm.models.world_model import WorldModel
from wm.planning.cem import CEMConfig, latent_cost, pixel_cost
from wm.planning.goal_mask import goal_weight, pusher_mask
from wm.planning.mpc import CEMPolicy, PixelCEMPolicy, run_episode
from wm.registry import input_hashes, load_registry
from wm.repro import REPO_ROOT, load_yaml

SMALL = CEMConfig(candidates=8, horizon=3, rounds=2, elites=2, cost_space="pixel")


def test_default_cost_space_is_latent_and_plan_yaml_unchanged():
    assert CEMConfig().cost_space == "latent"
    plan = load_yaml(REPO_ROOT / "configs/plan.yaml")
    assert "cost_space" not in plan["cem"] and plan["frozen"] is True
    assert CEMConfig.from_dict(plan["cem"]).cost_space == "latent"


def test_pixel_cost_structure_and_weight():
    goal = torch.zeros(3, 64, 64)
    w = torch.ones(1, 64, 64)
    same = torch.zeros(2, 3, 3, 64, 64)
    assert torch.allclose(pixel_cost(same, goal, w, 0.1), torch.zeros(2))
    x = torch.full((1, 3, 3, 64, 64), 0.2)                 # every step: distance 0.04
    assert pixel_cost(x, goal, w, 0.1).item() == pytest.approx(0.04 + 0.1 * 2 * 0.04, rel=1e-5)
    w2 = w.clone()
    w2[0, :32] = 21.0                                       # top half weighted x21
    top = torch.zeros(1, 1, 3, 64, 64)
    top[..., :32, :] = 0.2
    bottom = torch.zeros(1, 1, 3, 64, 64)
    bottom[..., 32:, :] = 0.2
    assert pixel_cost(top, goal, w2, 0.1).item() == pytest.approx(21 * pixel_cost(bottom, goal, w2, 0.1).item(), rel=1e-5)
    assert pixel_cost(torch.full((1, 1, 3, 64, 64), 3.0), goal, w, 0.0).item() == pytest.approx(0.25)   # clipped to 0.5
    z = torch.ones(1, 3, 4)
    assert latent_cost(z, torch.zeros(4), 0.1).item() == pytest.approx(4 + 0.1 * 8)


@pytest.fixture(scope="module")
def rendered():
    """(frame, state) pairs from env resets and from a state with the pusher far from the block."""
    env = make_env()
    out = []
    for k in range(6):
        obs, _ = env.reset(seed=k)
        out.append((render64(obs["pixels"]), get_state(env)))
    far = np.array([60.0, 60.0, 330.0, 300.0, 0.7], np.float32)
    obs = reset_to_state(env, far)
    out.append((render64(obs["pixels"]), get_state(env)))
    return out


def test_pusher_mask_only_near_the_pusher(rendered):
    yy, xx = np.mgrid[:64, :64]
    for frame, s in rendered:
        m = pusher_mask(frame)
        px = s[:2] * 64 / 512
        d = np.hypot(xx + 0.5 - px[0], yy + 0.5 - px[1])
        assert m.any() and d[m].max() <= 3.0
        assert m[d < 1.0].all()                             # the pusher's centre pixels are removed


def test_goal_weight_removes_pusher_from_fg(rendered):
    wm = WorldModel(fg_weight=True)
    wm.set_fg_mean(torch.full((3, 64, 64), 0.5))           # white mean: everything non-white is fg
    frame, _ = rendered[0]
    w, info = goal_weight(wm, frame)
    fg, pm = info["fg"], info["pusher"]
    expect = 1.0 + 20.0 * (fg & ~pm)
    assert torch.allclose(w[0], torch.as_tensor(expect, dtype=torch.float32))
    assert info["removed"] == int((fg & pm).sum()) > 0 and 0 < info["removed_share"] < 1
    with pytest.raises(RuntimeError):
        goal_weight(WorldModel(), frame)


def test_cem_pixel_cost_episode_runs_and_guards(rendered):
    env = make_env()
    obs, _ = env.reset(seed=0)
    s, f = get_state(env), render64(obs["pixels"])
    torch.manual_seed(0)
    wm = WorldModel(proprio=True, fg_weight=True).eval()
    wm.set_fg_mean(torch.full((3, 64, 64), 0.5))
    goal = s.copy()
    goal[2] += 60.0
    ep = run_episode(env, s, f, goal, CEMPolicy(wm, SMALL, "cpu"), budget=3, keep_plans=True)
    assert ep["steps"] == 3 and len(ep["terminal_costs"]) == 3 and np.isfinite(ep["costs"]).all()
    with pytest.raises(RuntimeError):
        CEMPolicy(WorldModel(proprio=True), SMALL, "cpu")
    with pytest.raises(ValueError):
        PixelCEMPolicy(PixelUNet(), wm, SMALL, "cpu")


def test_calibrate_tau_pixel_synthetic():
    states = np.array([[0, 0, 100, 100, 0.0], [0, 0, 105, 100, 0.0], [0, 0, 300, 300, 0.0]], np.float32)
    dist = np.array([[0, 1.0, 9.0], [2.0, 0, 9.0], [9.0, 9.0, 0]])
    t = calibrate_tau_pixel(lambda i, j: dist[i, j], states, np.zeros(3, int), percentile=100)
    assert t["n_success_pairs"] == 2 and t["tau"] == pytest.approx(2.0) and t["median_non_success"] == 9.0


def s_(k):
    return {"success": {"k": k}}


def test_select_cost_arm_margins():
    assert select_cost_arm(s_(13), s_(15))[0] is False     # +2
    assert select_cost_arm(s_(13), s_(16))[0] is True      # +3
    assert select_cost_arm(s_(13), s_(17))[0] is True      # +4
    assert "post-hoc diagnostic, chosen on val" in select_cost_arm(s_(13), s_(16))[1]


def ep(success, pos, ang):
    return {"success": success, "pos_success": pos, "start_pos_px": 50.0, "start_ang_deg": 0.0,
            "final_pos_px": 10.0, "final_ang_deg": ang, "min_pos_px": 10.0, "steps": 5, "seconds": 1.0,
            "policy_s": 0.5, "env_s": 0.5, "costs": [], "terminal_costs": []}


def test_summarize_angle_when_position_reached():
    pairs = [{"turn_deg": 30.0}] * 4
    s = summarize([ep(True, True, 5.0), ep(False, True, -40.0), ep(False, True, 60.0), ep(False, False, 90.0)], pairs)
    assert s["pos_reached_n"] == 3 and s["pos_reached_median_final_ang_deg"] == 40.0
    assert s["pos_reached_failed_n"] == 2 and s["pos_reached_failed_median_final_ang_deg"] == 50.0


def test_compare_runs():
    a = [{"pair": i, "success": f} for i, f in enumerate([True, False, True])]
    b = [{"pair": i, "success": f} for i, f in enumerate([True, True, False])]
    r = compare_runs(a, b, {"budget": 40, "seed": 0}, {"budget": 40, "seed": 0})
    assert r["comparable"] and r["agree"] == 1 and r["only_ours"] == 1 and r["only_ref"] == 1
    assert compare_runs(a, b, {"budget": 10, "seed": 0}, {"budget": 40, "seed": 0})["comparable"] is False
    assert compare_runs(a, b[:2], {"budget": 40, "seed": 0}, {"budget": 40, "seed": 0})["comparable"] is False


def test_registry_inputs():
    reg = load_registry()
    assert set(input_hashes(reg["jobs"]["plan_cost_ablation"], reg)) == {"eval-pairs", "ckpt-wm", "results-plan-val"}
