"""CEC2022 single-objective bound-constrained benchmark (official C implementation).

The objective values come from the official competition code
(``third_party/cec2022_official/cec22_test_func.cpp``, P. N. Suganthan's
2022-SO-BO repository), compiled unchanged except for the POSIX build patch
recorded in ``third_party/cec2022_official/build_patch.diff``. Shift, rotation
and shuffle data are the official ``input_data`` files (byte-identical to the
files shipped in the MPHBS reproducibility package).
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = ROOT / "third_party" / "cec2022_official"
DATA_DIR = OFFICIAL_DIR / "input_data"
LIB_PATH = OFFICIAL_DIR / "libcec22.so"
SRC_PATH = OFFICIAL_DIR / "cec22_test_func_posix.cpp"

# Official optimum values F_i* (CEC2022 technical report, Table I).
OPTIMA = {1: 300.0, 2: 400.0, 3: 600.0, 4: 800.0, 5: 900.0, 6: 1800.0,
          7: 2000.0, 8: 2200.0, 9: 2300.0, 10: 2400.0, 11: 2600.0, 12: 2700.0}
NAMES = {
    1: "Shifted and full Rotated Zakharov",
    2: "Shifted and full Rotated Rosenbrock",
    3: "Shifted and full Rotated Expanded Schaffer f6",
    4: "Shifted and full Rotated Non-Continuous Rastrigin",
    5: "Shifted and full Rotated Levy",
    6: "Hybrid Function 1 (N=3)",
    7: "Hybrid Function 2 (N=6)",
    8: "Hybrid Function 3 (N=5)",
    9: "Composition Function 1 (N=5)",
    10: "Composition Function 2 (N=4)",
    11: "Composition Function 3 (N=5)",
    12: "Composition Function 4 (N=6)",
}
BOUNDS = (-100.0, 100.0)
VALID_DIMS = (2, 10, 20)

_lib = None
_lock = threading.Lock()


def build_library(force: bool = False) -> Path:
    if LIB_PATH.exists() and not force:
        return LIB_PATH
    cmd = ["g++", "-O2", "-fPIC", "-shared", "-o", str(LIB_PATH), str(SRC_PATH)]
    subprocess.run(cmd, check=True, capture_output=True)
    return LIB_PATH


def _get_lib():
    global _lib
    if _lib is None:
        with _lock:
            if _lib is None:
                build_library()
                lib = ctypes.CDLL(str(LIB_PATH))
                lib.cec22_test_func.argtypes = [
                    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                    ctypes.c_int, ctypes.c_int, ctypes.c_int]
                lib.cec22_test_func.restype = None
                lib.cec22_set_data_dir.argtypes = [ctypes.c_char_p]
                lib.cec22_set_data_dir(str(DATA_DIR).encode())
                _lib = lib
    return _lib


def cec2022_raw(X: np.ndarray, func_id: int) -> np.ndarray:
    """Raw official CEC2022 value F_i(x) for each row of X (shape (m, D) or (D,))."""
    if not 1 <= func_id <= 12:
        raise ValueError(f"CEC2022 function id must be 1..12, got {func_id}")
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    m, d = X.shape
    if d not in VALID_DIMS:
        raise ValueError(f"CEC2022 is defined for D in {VALID_DIMS}, got {d}")
    if d == 2 and func_id in (6, 7, 8):
        raise ValueError("Hybrid functions F6-F8 are not defined for D=2")
    x = np.ascontiguousarray(X.reshape(-1))
    f = np.zeros(m, dtype=np.float64)
    _get_lib().cec22_test_func(x, f, d, m, func_id)
    return f


class CEC2022Problem:
    """Callable returning the residual error F_i(x) - F_i* (the quantity the
    MPHBS protocol optimizes and reports, paper Eq. (93)).

    The optimum value is a constant offset; every compared algorithm receives
    the identical callable. LA-KAIE is shift-invariant by construction (see
    tests/test_lakaie.py::test_shift_invariance), so it cannot exploit F_i*.
    """

    def __init__(self, func_id: int, dim: int = 20):
        self.func_id = int(func_id)
        self.dim = int(dim)
        self.lb = np.full(self.dim, BOUNDS[0])
        self.ub = np.full(self.dim, BOUNDS[1])
        self.optimum = OPTIMA[self.func_id]
        self.name = f"F{self.func_id}"
        self.benchmark = "CEC2022"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return cec2022_raw(X, self.func_id) - self.optimum
