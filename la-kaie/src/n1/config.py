"""N1 hyperparameters (spec §10) and arm definitions (spec §11).

All values are fixed a priori; none is tuned on N1 results.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

HYPOTHESES = ("HI", "HB", "HR")
SELECTORS = ("HBA", "MPA", "JOINT", "POOLED", "DECOUPLED")


@dataclass(frozen=True)
class N1Config:
    N: int = 30                      # population size per optimizer
    E_x: int = 24                    # exchange evaluations per ACTIVE iteration (= MPHBS B-C1 E)
    r_probe: int = 2                 # probe evaluations per SUPPRESSED iteration
    p_x: float = 0.2                 # per-unit inclusion probability in the exchange mask
    W_min: int = 60                  # W = max(W_min, W_per_dim * D)
    W_per_dim: int = 5
    U: int = 5                       # selection epoch interval (iterations)
    kappa: float = 3.0               # hysteresis in log-evidence units (Delta BIC = 2 kappa)
    alpha_link: float = 0.01         # partition edge test level (Bonferroni over pairs)
    alpha: float = 0.05              # nominal SPRT-style boundary parameters
    beta: float = 0.05
    delta: float = math.log(2.0)     # effect margin on the log-odds scale
    n_min: int = 20
    n_max: int = 200
    W_nat_iters: int = 10            # W_nat = W_nat_iters * 3N native attempts
    lambda_nat: float = 1e-2         # ridge on theta_1, theta_2 of the native-control model
    include_fad: bool = True
    eps_rel: float = 1e-8            # eigenvalue floor: eps_rel * tr(S) / D
    s_floor: float = 1e-12           # scale floor relative to box width
    p_clip: float = 1e-4
    eps_l: float = 1e-12
    irls_max_iter: int = 25
    irls_tol: float = 1e-8
    fisher_clip: float = 0.999999
    support_q: tuple = (0.01, 0.99)  # step-length support clipping quantiles
    a10_floor: float = 0.1           # A10 success-rate control
    a10_lr: float = 0.1

    def W(self, D: int) -> int:
        return max(self.W_min, self.W_per_dim * D)

    def W_nat(self) -> int:
        return self.W_nat_iters * 3 * self.N

    def as_dict(self) -> dict:
        d = asdict(self)
        d["support_q"] = list(self.support_q)
        return d


@dataclass(frozen=True)
class Arm:
    """One experimental arm (spec §11)."""
    name: str
    exchange: str                    # "none" | "fixed" | "selector" | "success_rate"
    driving: str | None = None       # driving selector for exchange="selector"
    fixed_k: str | None = None       # for exchange="fixed"
    use_h0: bool = False             # SPRT-style rule acts on the exchange count
    shadows: bool = True             # compute all selectors passively
    log_windows: bool = False        # store displacement streams (offline JOINT-HALF)
    harness: str | None = None       # T-SPRT calibration only: "null" | "positive"

    def always_active(self) -> bool:
        return self.exchange != "none" and not self.use_h0


ARMS: dict[str, Arm] = {
    "A0": Arm("A0", "none", log_windows=True),
    "A0_noshadow": Arm("A0_noshadow", "none", shadows=False),
    "A1": Arm("A1", "fixed", fixed_k="HI"),
    "A2": Arm("A2", "fixed", fixed_k="HB"),
    "A3": Arm("A3", "fixed", fixed_k="HR"),
    "A5": Arm("A5", "selector", driving="HBA", use_h0=True),
    "A6": Arm("A6", "selector", driving="MPA", use_h0=True),
    "A7": Arm("A7", "selector", driving="JOINT", use_h0=True),
    "A8": Arm("A8", "selector", driving="JOINT", use_h0=False),
    "A10": Arm("A10", "success_rate"),
    "A11": Arm("A11", "selector", driving="DECOUPLED", use_h0=True),
    "P": Arm("P", "selector", driving="POOLED", use_h0=True),
    # T-SPRT calibration Level 2 harness fixtures (spec §7.2); NOT N1 variants.
    "H_null": Arm("H_null", "selector", driving="JOINT", use_h0=True, harness="null"),
    "H_pos": Arm("H_pos", "selector", driving="JOINT", use_h0=True, harness="positive"),
}
ARMS["A9"] = ARMS["A7"]              # A9 == A7 (same configuration, not run twice)


def config_hash(obj) -> str:
    """sha256 of a canonical JSON serialisation."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


__all__ = ["N1Config", "Arm", "ARMS", "HYPOTHESES", "SELECTORS", "config_hash", "field"]
