import numpy as np
import pytest

from lakaie.algorithms import hba, mpa, mphb, mphbs
from lakaie.core import CountedObjective, Recorder
from lakaie.registry import iterations_for

BC1 = dict(K=7, Ni=3, rho_sub=0.25, w_best=0.05, gamma=0.3, fad_before_p2=True, fad_after_p2=False)
AC1 = dict(K=3, Ni=9, rho_sub=0.25, w_best=0.05, gamma=0.3, fad_before_p2=False, fad_after_p2=True)


@pytest.mark.parametrize("name,cfg,max_fe,dim,expected", [
    ("HBA", {}, 200000, 20, (6665, 199980)), ("MPA", {}, 200000, 20, (3333, 199980)),
    ("MPHB", {}, 200000, 20, (2221, 199950)), ("MPHBS", BC1, 200000, 20, (1738, 199930)),
    ("HBA", {}, 150000, 31, (4999, 150000)), ("MPA", {}, 150000, 31, (2500, 150000)),
    ("MPHB", {}, 150000, 31, (1666, 150000)), ("MPHBS", AC1, 150000, 31, (919, 149857)),
])
def test_fe_budgets_match_authors_audit(name, cfg, max_fe, dim, expected):
    """FE totals reported in the authors' FE_BUDGET_AUDIT.txt / paper Table 2."""
    assert iterations_for(name, cfg, max_fe, dim, 30) == expected


def _sphere(X):
    return np.sum((X - 0.3) ** 2, axis=1)


@pytest.mark.parametrize("mod,cfg,per_iter,init", [
    (hba, {}, 30, 30), (mpa, {}, 60, 0), (mphb, {}, 90, 60), (mphbs, BC1, 115, 60), (mphbs, AC1, 163, 60)])
def test_fe_per_iteration_and_bounds(mod, cfg, per_iter, init):
    lb, ub = -np.ones(20), np.ones(20)
    seen = []

    def f(X):
        seen.append(X.copy())
        return _sphere(X)
    obj = CountedObjective(f, None)
    rec = Recorder(lb, ub)
    out = mod.run(obj, 5, lb, ub, 20, 30, np.random.default_rng(3), cfg, rec)
    assert obj.fe == init + 5 * per_iter
    allX = np.vstack(seen)
    assert np.all(allX >= lb) and np.all(allX <= ub)
    assert out["best"] == pytest.approx(obj.best)       # algorithm best == best evaluated
    assert np.all(np.diff([r["best"] for r in rec.rows]) <= 0)


@pytest.mark.parametrize("mod,cfg", [(hba, {}), (mpa, {}), (mphb, {}), (mphbs, BC1)])
def test_determinism(mod, cfg):
    lb, ub = -np.ones(10), np.ones(10)
    r = []
    for _ in range(2):
        obj = CountedObjective(_sphere, None)
        r.append(mod.run(obj, 4, lb, ub, 10, 30, np.random.default_rng(11), cfg, None)["best"])
    assert r[0] == r[1]


def test_mphbs_small_dimension_single_block():
    lb, ub = -np.ones(5), np.ones(5)
    obj = CountedObjective(_sphere, None)
    mphbs.run(obj, 3, lb, ub, 5, 30, np.random.default_rng(0), BC1, None)
    assert obj.fe == 60 + 3 * (90 + 1 + 24)


def test_mphbs_actions_helpers():
    Q = np.zeros((2, 4, 2)); Q[0, 2, 1] = 1.0; Q[0, 1, 0] = 0.5
    assert mphbs.greedy(Q, 0) == (1, 2)
    assert mphbs.second_best(Q, 0) == (0, 1)
    res = np.zeros((4, 2, 2)); res[:, 0, 0] = [0, 1, 2, 3]; res[:, 0, 1] = [10, 11, 12, 13]
    assert mphbs.closest_action(10.4, res, 0) == (1, 0)
    assert mphbs.closest_action(-5, res, 0) == (0, 0)       # clipped to min
