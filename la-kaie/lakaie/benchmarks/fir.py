"""FIR filter-design benchmark (MPHBS paper Sec. 5.1.2, Eqs. (98)-(108)).

Case definitions are read from ``benchmarks/fir/fir_cases.yaml``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CASES_FILE = ROOT / "benchmarks" / "fir" / "fir_cases.yaml"


def load_cases(path: Path = CASES_FILE) -> dict:
    return yaml.safe_load(Path(path).read_text())


def band_indices(case: dict, n_fft: int):
    """Passband / stopband index sets over the half spectrum (0-based)."""
    w = np.arange(n_fft // 2 + 1) / (n_fft / 2)
    t = case["type"].upper()
    if t == "LPF":
        P = np.where(w <= case["Fp"])[0]
        S = np.where(w >= case["Fs"])[0]
    elif t == "HPF":
        P = np.where(w >= case["Fp"])[0]
        S = np.where(w <= case["Fs"])[0]
    elif t == "BPF":
        P = np.where((w >= case["Fp1"]) & (w <= case["Fp2"]))[0]
        S = np.where((w <= case["Fs1"]) | (w >= case["Fs2"]))[0]
    elif t == "BSF":
        P = np.where((w <= case["Fp1"]) | (w >= case["Fp2"]))[0]
        S = np.where((w >= case["Fs1"]) & (w <= case["Fs2"]))[0]
    else:
        raise ValueError(f"unsupported FIR type {t}")
    return w, P, S


class FIRProblem:
    """F(h) = Wp*MSEp + Ws*MSEs evaluated on the NFFT-point half spectrum."""

    def __init__(self, case_id: int, spec: dict | None = None):
        spec = spec or load_cases()
        common = spec["common"]
        case = next(c for c in spec["cases"] if c["id"] == case_id)
        self.case = case
        self.case_id = int(case_id)
        self.name = f"FIR{case_id}"
        self.benchmark = "FIR"
        self.dim = int(common["dimension"])
        self.n_fft = int(common["n_fft"])
        self.lb = np.full(self.dim, float(common["lower_bound"]))
        self.ub = np.full(self.dim, float(common["upper_bound"]))
        self.Wp = float(common["Wp"])
        self.Ws = float(case["Ws"])
        self.w, self.P, self.S = band_indices(case, self.n_fft)
        self.optimum = None  # unknown for FIR; values are reported as raw fitness

    def components(self, X: np.ndarray):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        H = np.fft.rfft(X, n=self.n_fft, axis=1)       # bins 0..NFFT/2
        mse_s = np.mean(H[:, self.S].real ** 2 + H[:, self.S].imag ** 2, axis=1) if self.S.size else 0.0
        mse_p = np.mean((np.abs(H[:, self.P]) - 1.0) ** 2, axis=1) if self.P.size else 0.0
        return mse_p, mse_s

    def __call__(self, X: np.ndarray) -> np.ndarray:
        mse_p, mse_s = self.components(X)
        return self.Wp * mse_p + self.Ws * mse_s

    # ---- gray-box structural information (used only by the optional
    # LA-KAIE-FIRPrior variant; derived from the objective definition itself,
    # never from evaluations) ----
    def stopband_quadratic_form(self) -> np.ndarray:
        """Q such that MSEs(h) = h^T Q h (exact: stopband energy is quadratic)."""
        n = np.arange(self.dim)
        omega = np.pi * self.w[self.S]
        C = np.cos(np.outer(omega, n))
        S = np.sin(np.outer(omega, n))
        return (C.T @ C + S.T @ S) / self.S.size
