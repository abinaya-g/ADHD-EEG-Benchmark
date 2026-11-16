"""
Revision experiments that need TensorFlow (+ the raw EEG, ideally a GPU).
Run on Kaggle exactly like run_all_experiments.py (same data root, same
protocol, same seeds). Writes CSVs to results_revision/.

  --task leakage     Leakage-permitting vs subject-grouped evaluation of the SAME
                     pipeline (CNN+classical heads and EEGNet): identical
                     architecture, optimiser, early-stopping rule, epoch budget
                     selection and aggregation; the ONLY difference is that the
                     outer (and inner) folds are drawn at the epoch level, so one
                     child's epochs can appear in train and test. Implemented by
                     giving every epoch its own pseudo-"subject" id for splitting
                     (run_nested_repeat splits by this id), then mapping back to
                     the true child for subject-level aggregation.
  --task coral       CORAL covariance-regularisation sweep (lambda in LAMBDAS) and
                     per-fold covariance distances before/after alignment.
  --task resolution  128 -> 512 Hz interpolation for the remaining repetitions
                     (the paper ran repetition 0 only), CNN and EEGNet.

Examples (Kaggle):
  !python run_revision_deep.py --task leakage --repeats 1
  !python run_revision_deep.py --task coral --repeats 1
  !python run_revision_deep.py --task resolution --repeat-indices 1,2,3,4
  !python run_revision_deep.py --task leakage --smoke-test   # synthetic, minutes on CPU
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd

from src import config, data, evaluation, models

LAMBDAS = (1e-3, 1e-4, 1e-2, 1e-1)  # first entry = the value used in the paper


def versions():
    import scipy
    import sklearn
    import tensorflow as tf
    return dict(python=platform.python_version(), tensorflow=tf.__version__, numpy=np.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__, pandas=pd.__version__,
                gpus=[g.name for g in tf.config.list_physical_devices("GPU")])


def common_kwargs(args, fs, n_time):
    return dict(outer_folds=args.outer_folds, inner_folds=args.inner_folds, fs=fs, n_timesamples=n_time,
                max_epochs=args.max_epochs, batch_size=config.BATCH_SIZE,
                patience=config.EARLY_STOPPING_PATIENCE, learning_rate=config.LEARNING_RATE)


def task_leakage(args, X, y, groups, fs, n_time, out):
    # epoch-level pseudo ids: every epoch is its own "group" -> random, class-balanced
    # epoch-level folds; the real subject id is restored afterwards.
    ep_ids = np.array([f"{g}__ep{i}" for i, g in enumerate(groups)])
    to_subject = dict(zip(ep_ids, groups))
    rows = []
    for i in args.repeat_list:
        seed = config.RANDOM_SEEDS[i]
        for name, build, clf in (("CNN", models.build_cnn, True), ("EEGNet", models.build_eegnet, False)):
            preds, _, _ = evaluation.run_nested_repeat(
                name, build, X, y, ep_ids, seed=seed, repeat_id=i, preprocessing_label=f"{fs}Hz",
                run_classifiers=clf, run_coral=False, **common_kwargs(args, fs, n_time))
            df = pd.DataFrame(preds)
            df["split"] = "epoch_level_leaky"
            df["subject_id"] = df["subject_id"].map(to_subject)
            rows.append(df)
            pd.concat(rows).to_csv(f"{out}/leakage_epoch_level_predictions.csv", index=False)  # checkpoint
    return pd.concat(rows)


def task_coral(args, X, y, groups, fs, n_time, out):
    all_p, all_c = [], []
    for i in args.repeat_list:
        p, _, c = evaluation.run_nested_repeat(
            "CNN", models.build_cnn, X, y, groups, seed=config.RANDOM_SEEDS[i], repeat_id=i,
            preprocessing_label=f"{fs}Hz", run_classifiers=False, run_coral=True,
            coral_lambdas=LAMBDAS, **common_kwargs(args, fs, n_time))
        all_p += p
        all_c += c
        pd.DataFrame(all_c).to_csv(f"{out}/coral_lambda_sweep_folds.csv", index=False)
    pd.DataFrame(all_p).to_csv(f"{out}/coral_lambda_sweep_predictions.csv", index=False)
    return pd.DataFrame(all_c)


def task_resolution(args, X, y, groups, fs, n_time, out):
    target_fs = fs * 2 if args.smoke_test else config.FS_INTERP_HZ  # 2x only in the synthetic smoke test
    # same order as run_all_experiments.py: interpolate the RAW epochs, then per-epoch normalise
    Xi = data.normalize_all(data.resample_pipeline_b(args.X_raw, orig_fs=fs, target_fs=target_fs))
    rows = []
    for i in args.repeat_list:
        for name, build in (("CNN", models.build_cnn), ("EEGNet", models.build_eegnet)):
            p, _, _ = evaluation.run_nested_repeat(
                name, build, Xi, y, groups, seed=config.RANDOM_SEEDS[i], repeat_id=i,
                preprocessing_label="128to512_interp", run_classifiers=False, run_coral=False,
                **{**common_kwargs(args, fs, n_time), "fs": target_fs,
                   "n_timesamples": Xi.shape[2]})
            rows += p
            pd.DataFrame(rows).to_csv(f"{out}/resolution_interp_predictions.csv", index=False)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["leakage", "coral", "resolution"])
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--repeat-indices", type=str, default=None)
    ap.add_argument("--outer-folds", type=int, default=config.OUTER_FOLDS)
    ap.add_argument("--inner-folds", type=int, default=config.INNER_FOLDS)
    ap.add_argument("--max-epochs", type=int, default=config.MAX_EPOCHS)
    ap.add_argument("--out", default="results_revision")
    args = ap.parse_args()
    args.repeat_list = [int(v) for v in args.repeat_indices.split(",")] if args.repeat_indices else list(range(args.repeats))
    out = args.out + ("_SYNTHETIC" if args.smoke_test else "")
    os.makedirs(out, exist_ok=True)

    if args.smoke_test:
        X, y, groups, _ = data.make_synthetic_dataset(n_adhd_subjects=15, n_control_subjects=15, min_epochs=3,
                                                      max_epochs=5, n_timesamples=1152, class_signal=0.9, seed=0)
        fs, n_time = 128, 1152
        args.outer_folds, args.inner_folds, args.max_epochs = 3, 2, 3
    else:
        X, y, groups, _ = data.load_dataset()
        fs, n_time = config.FS_ASSUMED_HZ, config.EPOCH_LEN_A
    args.X_raw = X
    X = data.normalize_all(X)
    json.dump(versions(), open(f"{out}/versions_{args.task}.json", "w"), indent=1)

    t0 = time.time()
    res = {"leakage": task_leakage, "coral": task_coral, "resolution": task_resolution}[args.task](
        args, X, y, groups, fs, n_time, out)
    print(res.head())
    print(f"done in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
