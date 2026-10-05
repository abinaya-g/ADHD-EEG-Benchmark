"""SHE Stage 1 unit tests (experiments/SHE_SPEC.md §19)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from lakaie.core import CountedObjective, Recorder, make_checkpoints
from lakaie.she.algorithm import run_she
from lakaie.she.config import ROOT, SHEConfig, VARIANTS, Variant, config_for
from lakaie.she.evidence import Evidence, LogisticModel, features, fit_linkage, group_pairs, posterior
from lakaie.she.hypotheses import Frame, T, T_inv, candidate, draw_mask, pick_tournament, unit_mask
from n1.mapping import draw_mask as n1_draw_mask
from n1.mapping import pick_receiver as n1_pick
from n1.synthetic import SyntheticProblem


def _frame(D=8, seed=0):
    rng = np.random.default_rng(seed)
    fr = Frame(D, -np.ones(D) * 5, np.ones(D) * 5)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    fr.B, fr.mu = Q, rng.standard_normal(D)
    fr.set_linkage(np.array([[0, 1], [1, 2], [4, 5]]), group_pairs(D, np.array([[0, 1], [1, 2], [4, 5]]),
                                                                  np.array([3.0, 2.0, 1.0]), 5))
    return fr


def test_T_inv():
    fr = _frame()
    x = np.random.default_rng(1).standard_normal(8)
    for h in (0, 1, 2, 3):
        assert np.allclose(T_inv(h, T(h, x, fr), fr), x, atol=1e-9)


def test_masks_and_groups():
    fr = _frame()
    rng = np.random.default_rng(2)
    gid = np.zeros(8, int)
    for k, g in enumerate(fr.groups):
        gid[g] = k
    assert sorted(np.concatenate(fr.groups).tolist()) == list(range(8))
    assert max(len(g) for g in fr.groups) <= 5
    for _ in range(200):
        assert unit_mask(0, rng, fr, 0.2).all()
        for h in (1, 2, 3):
            m = unit_mask(h, rng, fr, 0.2)
            assert m.any()
            if h == 2:                      # union of whole groups
                for g in fr.groups:
                    assert m[g].all() or not m[g].any()
    # full mask -> donor, as required for comparability
    xr, xd = np.zeros(8), np.ones(8)
    for h in (1, 3):
        x, _ = candidate(h, xr, xd, fr, type("R", (), {"random": lambda s, n: np.zeros(n),
                                                       "integers": lambda s, n: 0})(), 1.0)
        assert np.allclose(x, xd)
    g = group_pairs(10, np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]), np.arange(6, 0, -1.0), 3)
    assert max(len(v) for v in g) <= 3


def test_n1_semantics_equal():
    F = np.random.default_rng(3).random(30)
    for s in range(50):
        a, b = np.random.default_rng(s), np.random.default_rng(s)
        assert pick_tournament(a, F) == n1_pick(b, F)
        a, b = np.random.default_rng(s), np.random.default_rng(s)
        assert np.array_equal(draw_mask(a, 12, 0.2), n1_draw_mask(b, 12, 0.2))


def test_irls_step_matches_newton():
    rng = np.random.default_rng(4)
    Phi = np.hstack([np.ones((50, 1)), rng.standard_normal((50, 3))])
    y = (rng.random(50) < 0.4).astype(float)
    sw = rng.random(50) + 0.5
    m = LogisticModel(4, kappa2=1.0, y0=0.1, eps_p=1e-4)
    th = m.theta.copy()
    p = 1 / (1 + np.exp(-Phi @ th))
    pen = np.array([0, 1, 1, 1.0])
    g = Phi.T @ (sw * (y - p)) - pen * th
    H = Phi.T @ np.diag(sw * p * (1 - p)) @ Phi + np.diag(pen)
    ref = th + np.linalg.solve(H, g)
    m.irls_step(Phi, y, sw)
    assert np.allclose(m.theta, ref, atol=1e-8)


def test_prequential_loss_recursion():
    cfg = SHEConfig(rho=0.9)
    D = 4
    ev = Evidence(cfg, D)
    rng = np.random.default_rng(5)
    ells, ws = [], []
    for i in range(30):
        z1, z3 = rng.standard_normal(D), rng.standard_normal(D)
        y = int(rng.random() < 0.3)
        w = float(rng.random() + 0.5)
        pre = ev.losses(z1, z3, y)
        ell = ev.add(z1, z3, y, w)
        assert np.allclose(pre, ell)                 # scored with pre-update parameters
        ells.append(ell); ws.append(w)
        if i % 10 == 9:
            ev.refit()
    ells, ws = np.array(ells), np.array(ws)
    n = len(ws)
    brute = np.sum((0.9 ** (n - 1 - np.arange(n)))[:, None] * ws[:, None] * ells, axis=0)
    assert np.allclose(ev.L, brute)
    p = posterior(ev.L, (0, 1, 2, 3), 1.0, 0.05)
    assert abs(p.sum() - 1) < 1e-12 and p.min() >= 0.05 - 1e-12


def test_features_shapes():
    z = np.zeros((3, 5))
    assert features(0, z, z, np.zeros((0, 2), int)).shape == (3, 2)
    assert features(2, z, z, np.array([[0, 1], [2, 3]])).shape == (3, 13)


def test_linkage_recovers_planted_pair():
    rng = np.random.default_rng(6)
    n, D = 1500, 6
    z = rng.standard_normal((n, D))
    u = -1.0 + 1.5 * z[:, 0] * z[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-u))).astype(float)
    pairs, groups, info = fit_linkage(z, y, np.ones(n), SHEConfig())
    assert [0, 1] in pairs.tolist()


def _run(seed, shadows=True, fid=2, D=10, fe=6000, variant="SHE-NoE3"):
    pr = SyntheticProblem(fid, D, run_seed=seed)
    obj = CountedObjective(pr, fe, make_checkpoints(fe))
    v = VARIANTS[variant]
    if not shadows:
        v = Variant(v.name, v.selection, v.fixed_h, v.hyps, v.e3, v.h0_eval, False)
    out = run_she(pr, v, config_for("SYN"), seed, obj, fe, Recorder(pr.lb, pr.ub))
    return out, obj


@pytest.mark.parametrize("variant", ["SHE-NoE3", "SHE-Uniform"])
def test_fe_budget_exact_and_E(variant):
    out, obj = _run(74000001, variant=variant)
    s = out.summary
    assert obj.fe == 6000 and s["fe_audit_ok"] and s["exact_E_ok"]
    assert s["exchange_fe"] == int(out.arrays["r_cost"].sum())
    assert s["h0_attempts"] == int((out.arrays["r_gen"] == 0).sum())   # h0 free (A-3)
    assert np.all(out.arrays["r_cost"][out.arrays["r_gen"] == 0] == 0)


def test_determinism():
    a, _ = _run(74000002)
    b, _ = _run(74000002)
    c, _ = _run(74000003)
    for k in ("r_gen", "r_y", "r_pi", "r_loss"):
        assert np.array_equal(a.arrays[k], b.arrays[k])
    assert not np.array_equal(a.arrays["r_y"], c.arrays["r_y"])


def test_shadow_is_passive():
    a, oa = _run(74000004, shadows=True)
    b, ob = _run(74000004, shadows=False)
    for k in ("r_gen", "r_y", "r_pi", "r_loss"):
        assert np.array_equal(a.arrays[k], b.arrays[k])
    assert np.array_equal(oa.curve, ob.curve, equal_nan=True)


def test_seed_masters_disjoint():
    she = yaml.safe_load((ROOT / "configs" / "she.yaml").read_text())["masters"]
    rep = yaml.safe_load((ROOT / "configs" / "reproducibility.yaml").read_text())["master_seeds"]
    n1 = json.loads((ROOT / "configs" / "seeds.json").read_text())["masters"]
    existing = set(rep.values()) | set(n1.values())
    for k, v in she.items():
        if k == "main":
            assert v == rep["main"]           # deliberate pairing with MPHBS main runs
        else:
            assert v not in existing
    vals = [v for k, v in she.items() if k != "main"]
    assert len(set(vals)) == len(vals)
