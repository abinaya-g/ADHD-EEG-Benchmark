"""N1: statistical structural evidence for information exchange between
heterogeneous optimizers (HBA + MPA).

Implements IMPLEMENTATION_SPEC.md rev. 2.1. The native backbone is the
validated ``lakaie.algorithms.hybrid_common.HybridState`` (unchanged).
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # spec §9: one BLAS thread for determinism
