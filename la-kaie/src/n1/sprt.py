"""SPRT-style prequential sequential likelihood-ratio decision rule (spec §4.8, §7).

Uses Wald's LLR accumulation and Wald's boundary *formulas* as nominal
thresholds. No classical type-I/II guarantee is claimed: p0 is estimated
online, observations are dependent, and there is truncation and reset
(spec §7.1). Realised behaviour is measured by T-SPRT calibration (§7.2).
"""
from __future__ import annotations

import math

from .config import N1Config

ACTIVE, SUPPRESSED = "ACTIVE", "SUPPRESSED"
INFORMATIVE, UNINFORMATIVE = "INFORMATIVE", "UNINFORMATIVE"


def p1_from_p0(p0: float, delta: float) -> float:
    z = math.log(p0 / (1.0 - p0)) + delta
    return 1.0 / (1.0 + math.exp(-z))


class SPRTStyleRule:
    def __init__(self, cfg: N1Config):
        self.cfg = cfg
        self.A = math.log((1.0 - cfg.beta) / cfg.alpha)
        self.B = math.log(cfg.beta / (1.0 - cfg.alpha))
        self.state = ACTIVE              # initial state (spec §7)
        self.llr = 0.0
        self.n = 0
        self.decisions: list[dict] = []

    def increment(self, p0: float, y: int) -> float:
        p1 = p1_from_p0(p0, self.cfg.delta)
        if y:
            return math.log(p1 / p0)
        return math.log((1.0 - p1) / (1.0 - p0))

    def update(self, p0: float, y: int, t: int = -1, fe: int = -1) -> str | None:
        """Accumulate one outcome; returns a decision or None. The state change
        is read by the caller at the next iteration's plan step."""
        self.llr += self.increment(p0, y)
        self.n += 1
        if self.n < self.cfg.n_min:
            return None
        decision, truncated = None, False
        if self.llr >= self.A:
            decision = INFORMATIVE
        elif self.llr <= self.B:
            decision = UNINFORMATIVE
        elif self.n >= self.cfg.n_max:
            decision = INFORMATIVE if self.llr >= (self.A + self.B) / 2.0 else UNINFORMATIVE
            truncated = True
        if decision is None:
            return None
        prev = self.state
        self.state = ACTIVE if decision == INFORMATIVE else SUPPRESSED
        self.decisions.append({"t": t, "fe": fe, "decision": decision, "n": self.n,
                               "llr": self.llr, "truncated": truncated,
                               "state_before": prev, "state_after": self.state})
        self.llr, self.n = 0.0, 0          # reset_after_decision
        return decision
