"""Research recording structures (spec §3).

* RecordingObjective - transparent proxy around the single CountedObjective;
  records every evaluated batch in call order and counts FE per phase tag.
* NativeAttemptBuffer - FIFO of the last W_nat native attempts (NA).
* StructuralWindow   - FIFO of the last 2W standardized successful *native*
  displacements of one population (Z_H or Z_M), split into OLD / CUR.
* ColumnLog          - append-only columnar log (exchange / selection logs).
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

NATIVE_OPS = ("HBA", "MPA", "FAD")
OP_CODE = {"HBA": 0, "MPA": 1, "FAD": 2}
POP_CODE = {"H": 0, "M": 1}


class RecordingObjective:
    """Forwards every call unchanged to ``obj`` (the run's CountedObjective).

    It never evaluates anything itself; it only records (X, f) of each
    batch while ``capture`` is on and counts FE per ``tag``. Values returned
    are the counter's values, bit for bit.
    """

    def __init__(self, obj):
        self._obj = obj
        self.tag = "init"
        self.fe_by_tag: dict[str, int] = defaultdict(int)
        self._batches: list[tuple[np.ndarray, np.ndarray]] = []

    @property
    def fe(self) -> int:
        return self._obj.fe

    def remaining(self) -> int:
        return self._obj.remaining()

    def __call__(self, X):
        X = np.atleast_2d(X)
        f = self._obj(X)
        self.fe_by_tag[self.tag] += X.shape[0]
        self._batches.append((X.copy(), np.array(f, copy=True)))
        return f

    def eval1(self, x) -> float:
        return float(self(np.asarray(x)[None, :])[0])

    def begin(self, tag: str):
        self.tag = tag
        self._batches = []

    def take(self) -> list[tuple[np.ndarray, np.ndarray]]:
        b, self._batches = self._batches, []
        return b


class NativeAttemptBuffer:
    """Ring of (t, pop, op, l, y) for the last ``capacity`` native attempts."""

    def __init__(self, capacity: int):
        self.cap = int(capacity)
        self.t = np.zeros(self.cap, np.int64)
        self.pop = np.zeros(self.cap, np.int8)
        self.op = np.zeros(self.cap, np.int8)
        self.l = np.zeros(self.cap)
        self.y = np.zeros(self.cap, np.int8)
        self.n_total = 0

    def extend(self, t: int, pop: str, op: str, l: np.ndarray, y: np.ndarray):
        m = len(l)
        idx = (self.n_total + np.arange(m)) % self.cap
        self.t[idx] = t
        self.pop[idx] = POP_CODE[pop]
        self.op[idx] = OP_CODE[op]
        self.l[idx] = l
        self.y[idx] = y
        self.n_total += m

    def __len__(self):
        return min(self.n_total, self.cap)

    def arrays(self):
        n = len(self)
        if self.n_total <= self.cap:
            sl = slice(0, n)
            return self.l[sl], self.pop[sl], self.y[sl]
        order = (self.n_total + np.arange(self.cap)) % self.cap   # oldest first
        return self.l[order], self.pop[order], self.y[order]


class StructuralWindow:
    """FIFO of the last 2W standardized successful native displacements.

    Only ``append_native`` exists: exchange-accepted moves have no entry
    point (spec §3.2, prevents self-confirmation). Rows are optionally
    streamed (``keep_stream``) so that windows can be reconstructed offline.
    """

    def __init__(self, D: int, W: int, keep_stream: bool = False):
        self.D, self.W = D, W
        self.cap = 2 * W
        self.rows = np.zeros((self.cap, D))
        self.ops = np.zeros(self.cap, np.int8)
        self.n_total = 0
        self.keep_stream = keep_stream
        self.stream_rows: list[np.ndarray] = []
        self.stream_ops: list[np.ndarray] = []
        self.stream_t: list[np.ndarray] = []

    def append_native(self, Z: np.ndarray, op: str, t: int):
        if op not in NATIVE_OPS:
            raise ValueError(f"only native operators may enter structural windows, got {op!r}")
        Z = np.atleast_2d(Z)
        if Z.shape[0] == 0:
            return
        m = Z.shape[0]
        idx = (self.n_total + np.arange(m)) % self.cap
        self.rows[idx] = Z
        self.ops[idx] = OP_CODE[op]
        self.n_total += m
        if self.keep_stream:
            self.stream_rows.append(Z.astype(np.float32))
            self.stream_ops.append(np.full(m, OP_CODE[op], np.int8))
            self.stream_t.append(np.full(m, t, np.int32))

    def __len__(self):
        return min(self.n_total, self.cap)

    def _ordered(self) -> np.ndarray:
        n = len(self)
        if self.n_total <= self.cap:
            return self.rows[:n]
        order = (self.n_total + np.arange(self.cap)) % self.cap
        return self.rows[order]

    def ordered_ops(self) -> np.ndarray:
        n = len(self)
        if self.n_total <= self.cap:
            return self.ops[:n]
        order = (self.n_total + np.arange(self.cap)) % self.cap
        return self.ops[order]

    def cur(self) -> np.ndarray | None:
        """Newest W rows (None if fewer than W)."""
        R = self._ordered()
        return R[-self.W:].copy() if R.shape[0] >= self.W else None

    def old(self) -> np.ndarray | None:
        """The W rows preceding CUR (None if fewer than 2W): disjoint, older."""
        R = self._ordered()
        return R[-2 * self.W:-self.W].copy() if R.shape[0] >= 2 * self.W else None

    def stream(self):
        if not self.stream_rows:
            return (np.zeros((0, self.D), np.float32), np.zeros(0, np.int8), np.zeros(0, np.int32))
        return (np.concatenate(self.stream_rows), np.concatenate(self.stream_ops),
                np.concatenate(self.stream_t))


def rows_digest(*arrays: np.ndarray) -> str:
    """Order-independent digest of a multiset of rows (T-E4-OBS)."""
    R = np.vstack([a for a in arrays if a is not None and a.size])
    R = R[np.lexsort(R.T[::-1])]
    return hashlib.sha1(np.ascontiguousarray(R).tobytes()).hexdigest()


class ColumnLog:
    """Append-only list of dict rows -> dict of numpy arrays."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, **kw):
        self.rows.append(kw)

    def __len__(self):
        return len(self.rows)

    def as_arrays(self, prefix: str) -> dict:
        if not self.rows:
            return {}
        keys = []
        for r in self.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        out = {}
        for k in keys:
            vals = [r.get(k, None) for r in self.rows]
            if all(isinstance(v, (bool, np.bool_)) for v in vals):
                out[f"{prefix}{k}"] = np.asarray(vals, dtype=bool)
            elif all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in vals):
                out[f"{prefix}{k}"] = np.asarray(vals, dtype=np.int64)
            elif all(v is None or isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
                out[f"{prefix}{k}"] = np.asarray([np.nan if v is None else v for v in vals], dtype=np.float64)
            else:
                out[f"{prefix}{k}"] = np.asarray(["" if v is None else str(v) for v in vals])
        return out
