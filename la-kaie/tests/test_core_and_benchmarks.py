import numpy as np
import pytest

from lakaie.benchmarks.cec2022 import OPTIMA, CEC2022Problem, DATA_DIR, cec2022_raw
from lakaie.benchmarks.fir import FIRProblem
from lakaie.core import (BudgetExceeded, CountedObjective, calibrate_iterations,
                         make_checkpoints, matlab_round)


def test_cec_optimum_at_shift():
    for fid in range(1, 13):
        o = np.array((DATA_DIR / f"shift_data_{fid}.txt").read_text().split("\n")[0].split(), float)[:20]
        assert abs(cec2022_raw(o, fid)[0] - OPTIMA[fid]) < 1e-8 * OPTIMA[fid]
        assert abs(CEC2022Problem(fid, 20)(o)[0]) < 1e-6


def test_fir_analytic():
    for cid in range(1, 9):
        p = FIRProblem(cid)
        assert p.dim == 31
        assert p(np.zeros(31))[0] == pytest.approx(1.0)
        d = np.zeros(31); d[0] = 1
        assert p(d)[0] == pytest.approx(p.Ws)


def test_budget_guard_and_counting():
    obj = CountedObjective(lambda X: np.sum(X ** 2, 1), 10, make_checkpoints(10, 5))
    obj(np.ones((6, 3)))
    assert obj.fe == 6
    with pytest.raises(BudgetExceeded):
        obj(np.ones((5, 3)))
    assert obj.fe == 6                      # nothing evaluated on refusal
    obj(np.zeros((4, 3)))
    assert obj.fe == 10 and obj.best == 0.0
    assert np.all(np.diff(obj.curve) <= 0)


def test_checkpoint_curve_exact_resolution():
    vals = iter([5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
    obj = CountedObjective(lambda X: np.array([next(vals) for _ in range(len(X))]), 6,
                           np.array([1, 2, 3, 4, 5, 6]))
    obj(np.zeros((4, 1))); obj(np.zeros((2, 1)))
    assert obj.curve.tolist() == [5.0, 4.0, 3.0, 2.0, 1.0, 0.5]


def test_nan_is_rejected():
    obj = CountedObjective(lambda X: np.full(len(X), np.nan), 5)
    with pytest.raises(FloatingPointError):
        obj(np.zeros((1, 2)))


def test_matlab_round():
    assert matlab_round(7.5) == 8 and matlab_round(2.5) == 3 and matlab_round(-2.5) == -3


def test_calibration_linear():
    def run_with_T(obj, T, rng):
        obj(np.zeros((60, 2)))
        for _ in range(T):
            obj(np.zeros((115, 2)))
    assert calibrate_iterations(run_with_T, 200000, 2, 30) == (1738, 199930)
