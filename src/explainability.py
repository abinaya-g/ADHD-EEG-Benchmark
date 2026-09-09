"""
Phase 15: explainability via Integrated Gradients (Sundararajan et al.,
2017), implemented directly with tf.GradientTape rather than an external
package (no SHAP/captum dependency needed for a single-model, single-input
Keras graph). No causal claims are made anywhere in this module or its
output -- it reports attribution magnitudes only.

I'm implementing the standard IG formulation (straight-line path from a
zero baseline, Riemann-sum approximation of the path integral) as I recall
its mechanics, not from a fresh re-read of Sundararajan et al. (2017);
verify the exact formulation against the paper before describing it
precisely in a Methods section.
"""
import numpy as np
import tensorflow as tf


def integrated_gradients(model, x, baseline=None, steps=50):
    """
    x : single input, shape (n_channels, n_timesamples, 1)
    Returns an attribution map of the same shape as x.
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    baseline = tf.zeros_like(x) if baseline is None else tf.convert_to_tensor(baseline, dtype=tf.float32)

    alphas = tf.linspace(0.0, 1.0, steps + 1)
    interpolated = tf.stack([baseline + a * (x - baseline) for a in alphas], axis=0)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated, training=False)
    grads = tape.gradient(preds, interpolated)

    avg_grads = (grads[:-1] + grads[1:]) / 2.0
    integrated = tf.reduce_mean(avg_grads, axis=0) * (x - baseline)
    return integrated.numpy()


def batch_channel_temporal_importance(model, X, max_samples=30, steps=50, seed=0):
    """
    Runs IG on up to `max_samples` epochs (randomly subsampled for
    tractability) and reduces the attribution map to:
      - channel_importance: mean |attribution| per channel, per sample
      - temporal_importance: mean |attribution| per time bin (downsampled
        to 100 bins so the profile is readable regardless of fs)
    Returns per-sample arrays so mean +/- variability across folds/samples
    can be reported (Phase 15: "if attribution is unstable across folds,
    report this rather than hiding it").
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(max_samples, n), replace=False)

    n_channels, n_time = X.shape[1], X.shape[2]
    n_bins = min(100, n_time)
    bin_edges = np.linspace(0, n_time, n_bins + 1).astype(int)

    channel_imp = np.zeros((len(idx), n_channels))
    temporal_imp = np.zeros((len(idx), n_bins))

    for row, i in enumerate(idx):
        attr = np.abs(integrated_gradients(model, X[i], steps=steps))[..., 0]  # (channels, time)
        channel_imp[row] = attr.mean(axis=1)
        for b in range(n_bins):
            temporal_imp[row, b] = attr[:, bin_edges[b]:bin_edges[b + 1]].mean()

    return {
        "sample_indices": idx,
        "channel_importance": channel_imp,          # (n_samples, n_channels)
        "temporal_importance": temporal_imp,          # (n_samples, n_bins)
        "channel_importance_mean": channel_imp.mean(axis=0),
        "channel_importance_sd": channel_imp.std(axis=0),
        "temporal_importance_mean": temporal_imp.mean(axis=0),
        "temporal_importance_sd": temporal_imp.std(axis=0),
    }
