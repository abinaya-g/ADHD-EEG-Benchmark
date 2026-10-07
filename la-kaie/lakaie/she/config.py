"""SHE configuration and variants (experiments/SHE_SPEC.md §14, §18, Amendment 1)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "she.yaml"
HYPS = (0, 1, 2, 3)          # h0 whole-vector, h1 separable, h2 linkage, h3 rotated


@dataclass(frozen=True)
class SHEConfig:
    N: int = 30
    E: int = 24
    fad: str = "before"
    p_x: float = 0.2
    eta: float = 1.0
    rho: float = 0.995
    pi_min: float = 0.05
    U: int = 20
    lam_m: float = 0.995
    kappa2: float = 1.0
    W_x: int = 1000
    W_L: int = 1000
    W_n: int = 3000
    q_e: float = 0.5
    lam_LW: float = 0.2
    g_max: int = 5
    alpha: float = 0.05
    beta_min: float = 0.6931471805599453
    lam_cap: float = 0.5
    r_probe: int = 2
    eps_p: float = 1e-4
    y0: float = 0.1

    def W_A(self, D: int) -> int:
        """Artefact window per population (N1 rule): 2 * max(60, 5D) successful steps."""
        return 2 * max(60, 5 * D)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Variant:
    """Only the fields that a variant changes; everything else is shared."""
    name: str
    selection: str = "evidence"          # evidence | uniform | fixed
    fixed_h: int | None = None
    hyps: tuple = HYPS
    e3: bool = False                     # Stage 3; Stage 1 variants run without E3
    h0_eval: bool = False                # Amendment 1 A-3
    shadows: bool = True                 # passive donor-split evidence (spec §13)
    nested: bool = False                 # Amendment 2 B-1 (SHE-v2): h0 features nested in h1-h3
    zv_log: bool = False                 # Amendment 2 B-2: passive zero-variance window logging
    extra: dict = field(default_factory=dict)


VARIANTS = {
    "SHE-NoE3": Variant("SHE-NoE3"),
    "SHE-Uniform": Variant("SHE-Uniform", selection="uniform"),
    # Amendment 2 (spec §25): identical to the Stage 1 variants except the nested models
    "SHE-v2-NoE3": Variant("SHE-v2-NoE3", nested=True, zv_log=True),
    "SHE-v2-Uniform": Variant("SHE-v2-Uniform", selection="uniform", nested=True, zv_log=True),
}


def load_yaml() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def config_for(benchmark: str, overrides: dict | None = None) -> SHEConfig:
    y = load_yaml()
    d = dict(y["defaults"])
    d["E"] = int(y["exchange_budget"][benchmark])
    d["fad"] = str(y["fad_placement"][benchmark])
    d.update(overrides or {})
    return SHEConfig(**d)


def with_overrides(cfg: SHEConfig, **kw) -> SHEConfig:
    return replace(cfg, **kw)


def config_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
