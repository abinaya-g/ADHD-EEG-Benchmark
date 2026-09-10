"""
Model architectures.

build_cnn() is reused VERBATIM from the audited original notebooks
(adhd-baseline.ipynb cell 24 / adhd-coral.ipynb cell 0) -- same layers,
same names, same default fs=128 kernel-size scaling. It is not modified.

build_cnn_ablation() generalizes it with use_spatial / use_temporal /
use_bn / use_pool flags for the Phase 12 ablation study, collapsing to the
original architecture when all flags are True.

EEGNet / ShallowConvNet / DeepConvNet are transparent Keras
re-implementations of the standard architectures (Lawhern et al. 2018;
Schirrmeister et al. 2017), written here because the `braindecode` /
EEGModels package availability could not be verified in advance. I am
implementing these from the architecture descriptions as I recall them,
not from a fresh read of the papers -- verify layer widths/kernel sizes
against the original sources before stating them as exact replications in
a Methods section.

Every builder returns (full_model, feature_model): full_model outputs the
sigmoid class probability, feature_model outputs the flattened feature
vector immediately before the final classification layer, so classical
classifiers (LR/SVM/RF/GNB/KNN) can be trained on any architecture's
penultimate features using the same downstream code.
"""
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.constraints import max_norm


def set_all_seeds(seed):
    """Seeds Python, NumPy, and TensorFlow. Documented residual
    nondeterminism: some TensorFlow/cuDNN GPU ops (e.g. certain
    convolution backward passes) remain nondeterministic even with all
    three seeds fixed unless TF_DETERMINISTIC_OPS=1 is also set (enabled
    below) and a GPU-specific deterministic kernel exists; CPU execution
    (the only mode available in this environment) is deterministic given
    these three seeds for the layers used here."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Original architecture (reused verbatim)
# ---------------------------------------------------------------------------
def build_cnn(n_channels=19, n_timesamples=3840, fs=128):
    temporal_kernel_1 = int(fs / 4)
    temporal_kernel_2 = int(fs / 8)
    inp = layers.Input(shape=(n_channels, n_timesamples, 1), name="input_eeg")
    x = layers.Conv2D(16, (10, 1), padding="valid", activation="relu", name="spatial_conv1")(inp)
    x = layers.BatchNormalization(name="bn_spatial1")(x)
    x = layers.AveragePooling2D((2, 1), name="pool_spatial1")(x)
    x = layers.Conv2D(16, (4, 1), padding="valid", activation="relu", name="spatial_conv2")(x)
    x = layers.BatchNormalization(name="bn_spatial2")(x)
    x = layers.AveragePooling2D((2, 1), name="pool_spatial2")(x)
    x = layers.Reshape((x.shape[2], x.shape[3]), name="reshape_to_temporal")(x)
    x = layers.Conv1D(32, temporal_kernel_1, padding="valid", activation="relu", name="temporal_conv1")(x)
    x = layers.BatchNormalization(name="bn_temporal1")(x)
    x = layers.AveragePooling1D(temporal_kernel_1 // 2, name="pool_temporal1")(x)
    x = layers.Conv1D(32, temporal_kernel_2, padding="valid", activation="relu", name="temporal_conv2")(x)
    x = layers.BatchNormalization(name="bn_temporal2")(x)
    x = layers.AveragePooling1D(temporal_kernel_2 // 2, name="pool_temporal2")(x)
    flat = layers.Flatten(name="flatten")(x)
    dense1 = layers.Dense(64, activation="relu", name="dense1")(flat)
    dense2 = layers.Dense(32, activation="relu", name="dense2")(dense1)
    out = layers.Dense(1, activation="sigmoid", name="classification")(dense2)
    model = models.Model(inputs=inp, outputs=out, name="adhd_cnn")
    feature_model = models.Model(inputs=inp, outputs=dense1, name="adhd_cnn_features")
    return model, feature_model


# ---------------------------------------------------------------------------
# Ablation variants of the same architecture (Phase 12)
# ---------------------------------------------------------------------------
def build_cnn_ablation(n_channels=19, n_timesamples=3840, fs=128,
                        use_spatial=True, use_temporal=True,
                        use_bn=True, use_pool=True,
                        temporal_kernel_1=None, temporal_kernel_2=None,
                        name="cnn_ablation"):
    if not use_spatial and not use_temporal:
        raise ValueError("At least one of use_spatial/use_temporal must be True")

    tk1 = temporal_kernel_1 or int(fs / 4)
    tk2 = temporal_kernel_2 or int(fs / 8)

    inp = layers.Input(shape=(n_channels, n_timesamples, 1), name="input_eeg")
    x = inp

    if use_spatial:
        x = layers.Conv2D(16, (10, 1), padding="valid", activation="relu", name="spatial_conv1")(x)
        if use_bn:
            x = layers.BatchNormalization(name="bn_spatial1")(x)
        if use_pool:
            x = layers.AveragePooling2D((2, 1), name="pool_spatial1")(x)
        x = layers.Conv2D(16, (4, 1), padding="valid", activation="relu", name="spatial_conv2")(x)
        if use_bn:
            x = layers.BatchNormalization(name="bn_spatial2")(x)
        if use_pool:
            x = layers.AveragePooling2D((2, 1), name="pool_spatial2")(x)
        # x is (batch, height, time, filters). The original build_cnn()
        # hardcodes Reshape((time, filters)) because its fixed pooling
        # schedule always collapses height to 1 for n_channels=19 -- valid
        # only under that specific schedule. Here height is not guaranteed
        # to reach 1 (e.g. use_pool=False leaves height=7, not 1), so
        # instead permute time to the front and fold whatever height is
        # left into the feature/filter axis. This reduces to exactly the
        # same (time, filters) shape build_cnn() produces whenever height
        # == 1, and degrades gracefully (extra feature channels, not a
        # shape error) otherwise.
        x = layers.Permute((2, 1, 3), name="permute_time_first_spatial")(x)
        x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]), name="reshape_to_temporal")(x)
    else:
        # No spatial mixing: treat raw channels as the Conv1D "feature" axis.
        x = layers.Permute((2, 1, 3), name="permute_time_first")(x)
        x = layers.Reshape((n_timesamples, n_channels), name="reshape_raw_channels")(x)

    if use_temporal:
        x = layers.Conv1D(32, tk1, padding="valid", activation="relu", name="temporal_conv1")(x)
        if use_bn:
            x = layers.BatchNormalization(name="bn_temporal1")(x)
        if use_pool:
            x = layers.AveragePooling1D(max(tk1 // 2, 1), name="pool_temporal1")(x)
        x = layers.Conv1D(32, tk2, padding="valid", activation="relu", name="temporal_conv2")(x)
        if use_bn:
            x = layers.BatchNormalization(name="bn_temporal2")(x)
        if use_pool:
            x = layers.AveragePooling1D(max(tk2 // 2, 1), name="pool_temporal2")(x)

    flat = layers.Flatten(name="flatten")(x)
    dense1 = layers.Dense(64, activation="relu", name="dense1")(flat)
    dense2 = layers.Dense(32, activation="relu", name="dense2")(dense1)
    out = layers.Dense(1, activation="sigmoid", name="classification")(dense2)
    model = models.Model(inputs=inp, outputs=out, name=name)
    feature_model = models.Model(inputs=inp, outputs=dense1, name=name + "_features")
    return model, feature_model


ABLATION_CONFIGS = {
    "A_full": dict(use_spatial=True, use_temporal=True, use_bn=True, use_pool=True),
    "B_no_spatial": dict(use_spatial=False, use_temporal=True, use_bn=True, use_pool=True),
    "C_no_temporal": dict(use_spatial=True, use_temporal=False, use_bn=True, use_pool=True),
    "G_no_batchnorm": dict(use_spatial=True, use_temporal=True, use_bn=False, use_pool=True),
    "H_no_pooling": dict(use_spatial=True, use_temporal=True, use_bn=True, use_pool=False),
}
# Ablations D/E/F (direct-CNN vs CNN+LR vs CNN+RBF-SVM) are NOT separate
# architectures -- they are evaluation-protocol choices applied to the
# A_full model's own outputs / dense1 features, handled in evaluation.py.


# ---------------------------------------------------------------------------
# EEGNet (Lawhern et al., 2018) -- transparent re-implementation
# ---------------------------------------------------------------------------
def build_eegnet(n_channels=19, n_timesamples=3840, fs=128,
                  F1=8, D=2, F2=None, kern_length=None, dropout_rate=0.5,
                  name="eegnet"):
    F2 = F2 or F1 * D
    kern_length = kern_length or max(int(fs / 2), 4)

    inp = layers.Input(shape=(n_channels, n_timesamples, 1), name="input_eeg")
    # EEGNet expects (Samples, Chans, 1); our data is (Chans, Samples, 1),
    # so permute once at the input.
    x = layers.Permute((2, 1, 3), name="permute_samples_first")(inp)

    x = layers.Conv2D(F1, (kern_length, 1), padding="same", use_bias=False, name="temporal_conv")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.DepthwiseConv2D((1, n_channels), use_bias=False, depth_multiplier=D,
                                depthwise_constraint=max_norm(1.), name="depthwise_spatial_conv")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("elu", name="elu1")(x)
    x = layers.AveragePooling2D((4, 1), name="pool1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    x = layers.SeparableConv2D(F2, (16, 1), padding="same", use_bias=False, name="separable_conv")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("elu", name="elu2")(x)
    x = layers.AveragePooling2D((8, 1), name="pool2")(x)
    x = layers.Dropout(dropout_rate, name="drop2")(x)

    flat = layers.Flatten(name="flatten")(x)
    dense1 = layers.Dense(64, activation="relu", name="dense1",
                           kernel_constraint=max_norm(0.25))(flat)
    out = layers.Dense(1, activation="sigmoid", name="classification",
                        kernel_constraint=max_norm(0.25))(dense1)
    model = models.Model(inputs=inp, outputs=out, name=name)
    feature_model = models.Model(inputs=inp, outputs=dense1, name=name + "_features")
    return model, feature_model


# ---------------------------------------------------------------------------
# ShallowConvNet (Schirrmeister et al., 2017) -- transparent re-implementation
# ---------------------------------------------------------------------------
def build_shallow_convnet(n_channels=19, n_timesamples=3840, fs=128,
                           n_filters=40, dropout_rate=0.5, name="shallow_convnet"):
    kern_length = max(int(fs / 5.12), 5)  # scales to ~25 at fs=128, as in the original paper
    pool_size = max(int(fs / 1.71), 4)    # scales to ~75 at fs=128
    pool_stride = max(int(fs / 8.53), 1)  # scales to ~15 at fs=128

    inp = layers.Input(shape=(n_channels, n_timesamples, 1), name="input_eeg")
    x = layers.Permute((2, 1, 3), name="permute_samples_first")(inp)

    x = layers.Conv2D(n_filters, (kern_length, 1), padding="valid", name="temporal_conv")(x)
    x = layers.Conv2D(n_filters, (1, n_channels), padding="valid", use_bias=False, name="spatial_conv")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation(tf.square, name="square")(x)
    x = layers.AveragePooling2D((pool_size, 1), strides=(pool_stride, 1), name="pool1")(x)
    x = layers.Activation(lambda v: tf.math.log(tf.clip_by_value(v, 1e-7, 1e7)), name="log")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    flat = layers.Flatten(name="flatten")(x)
    dense1 = layers.Dense(64, activation="relu", name="dense1")(flat)
    out = layers.Dense(1, activation="sigmoid", name="classification")(dense1)
    model = models.Model(inputs=inp, outputs=out, name=name)
    feature_model = models.Model(inputs=inp, outputs=dense1, name=name + "_features")
    return model, feature_model


# ---------------------------------------------------------------------------
# DeepConvNet (Schirrmeister et al., 2017) -- transparent re-implementation
# ---------------------------------------------------------------------------
def build_deep_convnet(n_channels=19, n_timesamples=3840, fs=128,
                        dropout_rate=0.5, name="deep_convnet"):
    kern_length = max(int(fs / 12.8), 3)  # scales to ~10 at fs=128

    inp = layers.Input(shape=(n_channels, n_timesamples, 1), name="input_eeg")
    x = layers.Permute((2, 1, 3), name="permute_samples_first")(inp)

    x = layers.Conv2D(25, (kern_length, 1), padding="valid", name="conv1a")(x)
    x = layers.Conv2D(25, (1, n_channels), padding="valid", use_bias=False, name="conv1b")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("elu", name="elu1")(x)
    x = layers.MaxPooling2D((3, 1), strides=(3, 1), name="pool1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    for i, n_filt in enumerate((50, 100, 200), start=2):
        x = layers.Conv2D(n_filt, (kern_length, 1), padding="valid", name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("elu", name=f"elu{i}")(x)
        x = layers.MaxPooling2D((3, 1), strides=(3, 1), name=f"pool{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"drop{i}")(x)

    flat = layers.Flatten(name="flatten")(x)
    dense1 = layers.Dense(64, activation="relu", name="dense1")(flat)
    out = layers.Dense(1, activation="sigmoid", name="classification")(dense1)
    model = models.Model(inputs=inp, outputs=out, name=name)
    feature_model = models.Model(inputs=inp, outputs=dense1, name=name + "_features")
    return model, feature_model


ARCHITECTURE_BUILDERS = {
    "CNN": build_cnn,
    "EEGNet": build_eegnet,
    "ShallowConvNet": build_shallow_convnet,
    "DeepConvNet": build_deep_convnet,
}


def make_classifiers():
    """Fresh classifier instances per fold (fit only on that fold's
    training features -- see evaluation.py). class_weight documented per
    Phase 7: LR/SVM/RF support class_weight='balanced'; GNB and KNN do
    not expose a class-weighting mechanism in scikit-learn."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    return {
        "LR": LogisticRegression(penalty="l2", max_iter=2000, class_weight="balanced"),
        "LinearSVM": SVC(kernel="linear", probability=True, class_weight="balanced"),
        "NLSVM": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "RF": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=0),
        "GNB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


CLASSIFIER_SUPPORTS_CLASS_WEIGHT = {
    "LR": True, "LinearSVM": True, "NLSVM": True, "RF": True, "GNB": False, "KNN": False,
}
