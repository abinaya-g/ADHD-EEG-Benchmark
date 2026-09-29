"""T-MPHBS: the validated backbone and MPHBS port are frozen.

The hashes are those of the files at commit f0cd2de (N1 Stage 1). These
files are unchanged in behaviour since the baseline validation (the only
edit after it, commit 7c619a1, removed an unused import from
hybrid_common.py). Any change to them makes this test fail, which is
the R-BASE-1 stop condition. The existing suite (tests/test_baselines.py)
separately checks the FE totals 199,930 and 149,857 and determinism.
"""
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FROZEN = {
    "lakaie/algorithms/hybrid_common.py": "07296f6055271333e5828214688a7192dc1c9356c39af4c82f0d3951fadaa8b3",
    "lakaie/algorithms/hba.py": "e53256efd1b7b3204a7343781360b9999a2f2a0611dca94f1c0e84065188e379",
    "lakaie/algorithms/mpa.py": "3da63c94bd17c0282aa6a9110eeb7b28d8c3674fd3802f4b26ae1f86c1b9c771",
    "lakaie/algorithms/mphbs.py": "26f915107fc6cee6e460c2328ebe1cdee54b46de16ec42fa8da8ebf77172128a",
    "lakaie/core.py": "66bf6d04800da57d647c7fef23921c5123310b3be17461654e5149fe78b090f8",
}


@pytest.mark.parametrize("rel,digest", sorted(FROZEN.items()))
def test_backbone_file_unchanged(rel, digest):
    assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest
