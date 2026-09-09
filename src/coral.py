"""
CORAL (Sun & Saenko, 2016) feature alignment.

coral_transform() and covariance_distance() are reused, essentially
verbatim, from adhd-coral.ipynb (cell-1 / cell-2 / cell-5). The mechanics
are unchanged; what changes here (Phase 13) is how the function is *used*:
the evaluation code that calls this module (see evaluation.py:
run_coral_experiment) makes explicit, in its saved output, that computing
cov_target from the outer-fold test subjects' UNLABELLED features is a
transductive unsupervised domain-adaptation setting, not a standard
inductive subject-independent evaluation. Both quantities are saved
(covariance distance before/after) so this can be reported quantitatively
rather than asserted.

Note on the CORAL formulation: I'm implementing this from the general
domain-adaptation mechanics (whitening the source covariance, then
re-coloring with the target covariance) as reused from the original
notebook's own implementation, not from a fresh read of Sun & Saenko
(2016). Verify the exact formulation against the original paper before
citing it precisely in a Methods section.
"""
import numpy as np


def coral_transform(source_features, target_features, lambda_reg=1e-3):
    source_centered = source_features - source_features.mean(axis=0, keepdims=True)
    target_centered = target_features - target_features.mean(axis=0, keepdims=True)

    n_s, n_t, d = source_features.shape[0], target_features.shape[0], source_features.shape[1]
    cov_source = (source_centered.T @ source_centered) / (n_s - 1) + lambda_reg * np.eye(d)
    cov_target = (target_centered.T @ target_centered) / (n_t - 1) + lambda_reg * np.eye(d)

    def matrix_power(mat, power):
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
        return eigvecs @ np.diag(eigvals ** power) @ eigvecs.T

    cov_source_inv_sqrt = matrix_power(cov_source, -0.5)
    cov_target_sqrt = matrix_power(cov_target, 0.5)

    source_aligned = source_centered @ cov_source_inv_sqrt @ cov_target_sqrt
    source_aligned = source_aligned + target_features.mean(axis=0, keepdims=True)
    return source_aligned


def covariance_distance(feat_a, feat_b):
    """Frobenius norm of the difference between two covariance matrices."""
    cov_a = np.cov(feat_a, rowvar=False)
    cov_b = np.cov(feat_b, rowvar=False)
    return float(np.linalg.norm(cov_a - cov_b, ord="fro"))
